import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from os import cpu_count
from functools import partial

from maestro import DEBUG, MULTICORE
from maestro.frame import Frame


CPU_COUNT = min(cpu_count(), 8) # Limit to 8 cores for now


# Many of the preprocessing steps are identical between calibration frames.
class CombinationFrame(Frame):
    """A Base class for a CombinationFrame object

    This class is intended to be a base class for the different types of calibration
    frames (i.e. bias, dark, flat). It inherits from the Frame class and adds the
    functionality to combine multiple frames into a single master frame.

    Attributes:
        frame_paths (list): A list of paths to the frame files
        n_frames (int): The number of frames to be combined
        frame_format (str): The format of the frame files (e.g. 'CR3', 'NEF', 'JPG', 'PNG')
        loadkwargs (dict): The keyword arguments to be passed to the frame loading function
        master_method (str): The method to use to create the master frame (e.g. 'median', 'mean')
        master_dtype (numpy.dtype): The data type of the master frame (use np.float16)
        master_frame (numpy.ndarray): The master frame data in RGB format
        rgb (numpy.ndarray): The master frame data in RGB format
        dimensions (numpy.ndarray): The dimensions of the master frame data [x, y, color]
        npix (int): The number of pixels in the master frame
        dtype (str): The data type of the master frame data
        frame_type (str): The type of frame (e.g. 'Bias', 'Dark', 'Flat')
    """

    def __init__(self, frame_paths, frame_format=None, loadkwargs=None, method='median',
                 keep_individual=False):
        """A CombinationFrame object constructor

        This constructor initializes a CombinationFrame object by loading the frame data
        from the specified file paths. The frame format and load arguments can be specified,
        but can be inferred from the file paths if not. The method for creating the master
        frame can also be specified. The master frame is created using the specified method
        and stored as an attribute. The master frame is also converted to an RGB image for
        display. The few pieces of metadata we can get from the frame are also stored as
        attributes. The individual frames can be stored if desired, though this will take
        up more memory. 

        Args:
            frame_paths (list): A list of paths to the frame files
            frame_format (str, optional): The format of the frame files. Defaults to None.
            loadkwargs (dict, optional): The keyword arguments to be passed to the frame 
                loading function. Defaults to None.
            method (str, optional): The method to use to create the master frame. Defaults to 'median'.
            keep_individual (bool, optional): Whether to keep the individual frames in memory. Defaults to False.
        """
        if DEBUG:
            print('Initializing preprocess base...')

        if not hasattr(frame_paths, '__iter__'):
            frame_paths = [frame_paths]

        # Set the frame paths, format, and load arguments
        self.frame_paths = frame_paths
        self.n_frames = len(frame_paths)
        
        if frame_format is None:
            self.frame_format = self._get_frame_format(frame_paths[0])
        else:
            self.frame_format = frame_format.lower()

        if loadkwargs is None:
            self.loadkwargs = self._get_frame_loadkwargs()
        else:
            self.loadkwargs = loadkwargs

        # Set the method for creating the master frame
        self.master_method = method

        # Set the data type for the master frame (use np.float16)
        self.master_dtype = np.float16

        # Create the master frame
        self.master_frame = self.create_master_frame(self.master_method, keep_individual)

        # Convert the master frame to an RGB image for display
        self.rgb = self.master_frame.astype(np.uint8)
    
        # Set the few pieces of metadata we can get from the frame
        # TODO: Get more metadata from raw formats (CR3, NEF, etc)
        self.dimensions = self.master_frame.shape
        self.npix = self.dimensions[0] * self.dimensions[1]
        self.dtype = self.master_frame.dtype
        self.frame_type = 'CombinationFrame'

    
    def create_master_frame(self, method='median', keep_individual=False):
        """Create a master frame from the individual frames

        This method creates a master frame from the individual frames using the specified
        method. The method can be 'median', 'mean', or 'clipped_mean'. The master frame
        is stored as an attribute and returned. The individual frames can be stored if
        desired, though this will take up more memory.

        Args:
            method (str): The method to use to create the master frame. Defaults to 'median'. 
            keep_individual (bool): Whether to keep the individual frames in memory. 
                Defaults to False.

        Raises:
            NotImplementedError: If the specified method is not implemented

        Returns:
            numpy.ndarray: The master frame data in RGB format (dtype=master_dtype)
        """
        if DEBUG:
            print(f'Creating master frame using method: {method}')
        
        all_frames = self._load_all_frames()
        all_rgb = [frame.rgb for frame in all_frames]

        # Method can be 'median', 'mean', 'clipped_mean'
        
        # Now we need to use the different methods
        if method.lower() == 'median':
            master_frame = np.median(all_rgb, axis=0)

        elif method.lower() == 'mean':
            master_frame = np.mean(all_rgb, axis=0)

        elif method.lower() == 'clipped_mean':
            # Sigma clipping of outliers. We will use a 3-sigma clipping
            # to do so, we will calculate the current mean and standard deviation
            mean, std = np.mean(all_rgb, axis=0), np.std(all_rgb, axis=0)
            # Now we will remove the outliers
            to_keep = np.abs(all_rgb - mean) < 3*std
            # Now recalculate the mean of the clipped data
            master_frame = np.mean(all_rgb[to_keep], axis=0)

        else:
            raise NotImplementedError(f"Master frame creation method not implemented: {method}")
        
        if keep_individual:
            self.individual_frames = all_frames
        else:
            del all_frames # Free up memory (should be done automatically, but just in case)
            self.individual_frames = None

        return master_frame.astype(self.master_dtype)
    

    def _load_all_frames(self):
        """Load all of the individual frames

        This method loads all of the individual frames using the frame loading function
        and the specified frame format and load arguments. The individual frames are
        then returned.

        If the MULTICORE flag in maestro.__init__.py is set to True, the frames will be
        loaded in parallel using multiprocessing to get around the GIL. Otherwise, the
        frames will be loaded sequentially.

        Returns:
            list: A list of individual Frame objects
        """

        frame_gen = partial(Frame, frame_format=self.frame_format, loadkwargs=self.loadkwargs)
        if MULTICORE:
            # For this method, we will load all of the the different frames
            # using multiprocessing to get around the GIL
            with Pool(CPU_COUNT) as p:
                all_frames = list(p.map(frame_gen, self.frame_paths))
        
        else:
            all_frames = [frame_gen(path) for path in tqdm(self.frame_paths)]

        return all_frames


# Inherited classes for the different calibration frames -----------------------

class Bias(CombinationFrame):
    """The Bias class used to make master bias frames

    This class is used to create master bias frames from a set of individual bias frames.
    It inherits from the CombinationFrame class and adds the functionality to create a
    master bias frame using the specified method.

    Attributes:
        frame_paths (list): A list of paths to the frame files
        n_frames (int): The number of frames to be combined
        frame_format (str): The format of the frame files (e.g. 'CR3', 'NEF', 'JPG', 'PNG')
        loadkwargs (dict): The keyword arguments to be passed to the frame loading function
        master_method (str): The method to use to create the master frame (e.g. 'median', 'mean')
        master_dtype (numpy.dtype): The data type of the master frame (use np.float16)
        master_frame (numpy.ndarray): The master frame data in RGB format
        rgb (numpy.ndarray): The master frame data in RGB format
        dimensions (numpy.ndarray): The dimensions of the master frame data [x, y, color]
        npix (int): The number of pixels in the master frame
        dtype (str): The data type of the master frame data
        frame_type (str): The type of frame 'Bias'
    """
    def __init__(self, bias_paths, frame_format=None, loadkwargs=None, 
                 method='median', keep_individual=False):
        """The Bias class constructor

        This constructor initializes a Bias object by loading the bias frame data from the
        specified file paths. The frame format and load arguments can be specified, but can
        be inferred from the file paths if not. The method for creating the master frame can
        also be specified. The master frame is created using the specified method and stored
        as an attribute. The master frame is also converted to an RGB image for display. The
        few pieces of metadata we can get from the frame are also stored as attributes. The
        individual frames can be stored if desired, though this will take up more memory.

        Args:
            bias_paths (list): A list of paths to the bias frame files
            frame_format (str, optional): The format of the frame files. Defaults to None.
            loadkwargs (dict, optional): The keyword arguments to be passed to the frame
                loading function. Defaults to None.
            method (str): The method to use to create the master frame. Defaults to 'median'.
            keep_individual (bool): Whether to keep the individual frames in memory. Defaults to False. 
        """
        if DEBUG:
            print('Initializing bias frame...')

        super().__init__(bias_paths, frame_format, loadkwargs, method, keep_individual)
        self.frame_type = 'Bias'


class Dark(CombinationFrame):
    """The Dark class used to make master dark frames

    This class is used to create master dark frames from a set of individual dark frames.
    It inherits from the CombinationFrame class and adds the functionality to create a
    master dark frame using the specified method. 

    Attributes:
        frame_paths (list): A list of paths to the frame files
        n_frames (int): The number of frames to be combined
        frame_format (str): The format of the frame files (e.g. 'CR3', 'NEF', 'JPG', 'PNG')
        loadkwargs (dict): The keyword arguments to be passed to the frame loading function
        master_method (str): The method to use to create the master frame (e.g. 'median', 'mean')
        master_dtype (numpy.dtype): The data type of the master frame (use np.float16)
        master_frame (numpy.ndarray): The master frame data in RGB format
        rgb (numpy.ndarray): The master frame data in RGB format
        dimensions (numpy.ndarray): The dimensions of the master frame data [x, y, color]
        npix (int): The number of pixels in the master frame
        dtype (str): The data type of the master frame data
        frame_type (str): The type of frame 'Dark'
    """
    def __init__(self, dark_paths, frame_format=None, loadkwargs=None, 
                 method='median', keep_individual=False):
        """The Dark class constructor

        This constructor initializes a Dark object by loading the dark frame data from the
        specified file paths. The frame format and load arguments can be specified, but can
        be inferred from the file paths if not. The method for creating the master frame can
        also be specified. The master frame is created using the specified method and stored
        as an attribute. The master frame is also converted to an RGB image for display. The
        few pieces of metadata we can get from the frame are also stored as attributes. The
        individual frames can be stored if desired, though this will take up more memory.

        Args:
            dark_paths (list): A list of paths to the dark frame files
            frame_format (str, optional): The format of the frame files. Defaults to None.
            loadkwargs (dict, optional): The keyword arguments to be passed to the frame
                loading function. Defaults to None.
            method (str): The method to use to create the master frame. Defaults to 'median'.
            keep_individual (bool): Whether to keep the individual frames in memory. Defaults to False. 
        """
        if DEBUG:
            print('Initializing dark frame...')

        super().__init__(dark_paths, frame_format, loadkwargs, method, keep_individual)
        self.frame_type = 'Dark'


class Flat(CombinationFrame):
    """The Flat class used to make master flat frames

    This class is used to create master flat frames from a set of individual flat frames.
    It inherits from the CombinationFrame class and adds the functionality to create a
    master flat frame using the specified method.

    Attributes:
        frame_paths (list): A list of paths to the frame files
        n_frames (int): The number of frames to be combined
        frame_format (str): The format of the frame files (e.g. 'CR3', 'NEF', 'JPG', 'PNG')
        loadkwargs (dict): The keyword arguments to be passed to the frame loading function
        master_method (str): The method to use to create the master frame (e.g. 'median', 'mean')
        master_dtype (numpy.dtype): The data type of the master frame (use np.float16)
        master_frame (numpy.ndarray): The master frame data in RGB format
        rgb (numpy.ndarray): The master frame data in RGB format
        dimensions (numpy.ndarray): The dimensions of the master frame data [x, y, color]
        npix (int): The number of pixels in the master frame
        dtype (str): The data type of the master frame data
        frame_type (str): The type of frame 'Flat'
    """
    def __init__(self, flat_paths, frame_format=None, loadkwargs=None, 
                 method='median', keep_individual=False):
        """The Flat class constructor

        This constructor initializes a Flat object by loading the flat frame data from the
        specified file paths. The frame format and load arguments can be specified, but can
        be inferred from the file paths if not. The method for creating the master frame can
        also be specified. The master frame is created using the specified method and stored
        as an attribute. The master frame is also converted to an RGB image for display. The
        few pieces of metadata we can get from the frame are also stored as attributes. The
        individual frames can be stored if desired, though this will take up more memory.

        Args:
            flat_paths (list): A list of paths to the flat frame files
            frame_format (str, optional): The format of the frame files. Defaults to None.
            loadkwargs (dict, optional): The keyword arguments to be passed to the frame
                loading function. Defaults to None.
            method (str): The method to use to create the master frame. Defaults to 'median'.
            keep_individual (bool): Whether to keep the individual frames in memory. Defaults to False.
        """
        if DEBUG:
            print('Initializing flat frame...')

        super().__init__(flat_paths, frame_format, loadkwargs, method, keep_individual)
        self.frame_type = 'Flat'
