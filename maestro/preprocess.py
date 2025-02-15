import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from os import cpu_count
from functools import partial

from maestro import DEBUG
from maestro.frame import Frame


CPU_COUNT = cpu_count()


# Many of the preprocessing steps are identical between calibration frames.
class CombinationFrame(Frame):
    """

    _extended_summary_

    Args:
        Frame (_type_): _description_
    """

    def __init__(self, frame_paths, frame_format=None, loadkwargs=None, method='median',
                 keep_individual=False):
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
        self.master_dtype = np.uint8

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
        # For this method, we will load all of the the different frames
        # using multiprocessing to get around the GIL

        frame_gen = partial(Frame, frame_format=self.frame_format, loadkwargs=self.loadkwargs)

        with Pool(CPU_COUNT) as p:
            all_frames = list(p.map(frame_gen, self.frame_paths))
        
        return all_frames


# Inherited classes for the different calibration frames -----------------------

class Bias(CombinationFrame):
    def __init__(self, bias_paths, frame_format=None, loadkwargs=None, 
                 method='median', keep_individual=False):
        if DEBUG:
            print('Initializing bias frame...')

        super().__init__(bias_paths, frame_format, loadkwargs, method, keep_individual)
        self.frame_type = 'Bias'


class Dark(CombinationFrame):
    def __init__(self, dark_paths, frame_format=None, loadkwargs=None, 
                 method='median', keep_individual=False):
        if DEBUG:
            print('Initializing dark frame...')

        super().__init__(dark_paths, frame_format, loadkwargs, method, keep_individual)
        self.frame_type = 'Dark'


class Flat(CombinationFrame):
    def __init__(self, flat_paths, frame_format=None, loadkwargs=None, 
                 method='median', keep_individual=False):
        if DEBUG:
            print('Initializing flat frame...')

        super().__init__(flat_paths, frame_format, loadkwargs, method, keep_individual)
        self.frame_type = 'Flat'
