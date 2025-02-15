
import numpy as np
import rawpy

from maestro import DEBUG


class Frame:
    """A Base class for a Frame object

    This class is a base class for a Frame object. It only has basic functionality
    for loading the frames in their various formats. It is intended to be subclassed
    for the specific frame types (i.e. bias, dark, flat, light, etc).

    Attributes:
        frame_path (str): The path to the frame file
        frame_format (str): The format of the frame file (e.g. 'CR3', 'NEF', 'JPG', 'PNG')
        loadkwargs (dict): The keyword arguments to be passed to the frame loading function
        rgb (numpy.ndarray): The frame data in RGB format
        dimensions (numpy.ndarray): The dimensions of the frame data [x, y, color]
        npix (int): The number of pixels in the frame
        dtype (str): The data type of the frame data
    """

    def __init__(self, frame_path, frame_format=None, loadkwargs=None):
        """A Frame object constructor

        This constructor initializes a Frame object by loading the frame data from the
        specified file path. The frame format and load arguments can be specified, but
        can be inferred from the file path if not.

        If frame_format is a raw format (e.g. CR3, NEF), the loadkwargs will be
        given to the rawpy.postprocess() function. Generally, the defaults are fine,
        but can be overridden if you want the customization.

        Args:
            frame_path (str): The path to the frame file
            frame_format (str, optional): The format of the frame file. Defaults to None.
            loadkwargs (dict, optional): The keyword arguments to be passed to the frame 
                loading function. Defaults to None.
        """
        if DEBUG:
            print('Initializing frame...')

        # Set the frame path, format, and load arguments
        self.frame_path = frame_path

        if frame_format is None:
            self.frame_format = self._get_frame_format(frame_path) 
        else: 
            self.frame_format = frame_format.lower()

        if loadkwargs is None:
            self.loadkwargs = self._get_frame_loadkwargs()
        else:
            self.loadkwargs = loadkwargs

        
        # Now load the frame
        self.rgb = self._load_frame()


        # Now set the few pieces of metadata we can get from the frame
        # TODO: Get more metadata from raw formats (CR3, NEF, etc)
        self.dimensions = self.rgb.shape
        self.npix = self.dimensions[0] * self.dimensions[1]
        self.dtype = self.rgb.dtype


    def get_grayscale(self, method='NTSC'):
        """Get the frame data in grayscale

        This method converts the frame data from RGB to grayscale using the specified
        method. The default method is the NTSC standard (0.299R + 0.587G + 0.114B).
        Other methods can be specified, but are not currently implemented.

        Args:
            method (str, optional): The method to use for grayscale conversion. 
                Defaults to 'NTSC'.

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            numpy.ndarray: The frame data in grayscale format [x,y,1]
        """
        if DEBUG:
            print('Getting frame grayscale...')

        if (method is None) or (method.lower() == 'ntsc'):
            factor = (0.299, 0.587, 0.114)
        else:
            raise NotImplementedError(f"Grayscale conversion method not implemented: {method}")
        
        gray = factor[0]*self.rgb[:,:,0] + \
               factor[1]*self.rgb[:,:,1] + \
               factor[2]*self.rgb[:,:,2]
        return gray


    def _get_frame_format(self, frame_path):
        """A hidden method to get the frame format if not specified

        NOTE: that this method breaks if the frame_path has multiple '.' characters
        in the extension. This is a known issue and will be fixed in a future version.

        Args:
            frame_path (str): The path to the frame file

        Returns:
            str: The format of the frame file
        """
        if DEBUG:
            print('Getting frame format...')

        fmt = frame_path.split('.')[-1]
        return fmt.lower()
    

    def _get_frame_loadkwargs(self):
        """A hidden method to get the frame load keyword arguments if not specified

        This method returns the default keyword arguments for loading a frame in the
        specified format. This is useful for raw formats (e.g. CR3, NEF) where the
        rawpy.postprocess() function is used.

        Returns:
            dict: The keyword arguments for loading the frame
        """
        if DEBUG:
            print('Getting frame load keyword arguments...')

        if self.frame_format == 'cr3':
            lkwa = {'use_camera_wb':True, 'no_auto_scale':False,
                    'no_auto_bright':True, 'chromatic_aberration':(1,1)}
        else:
            lkwa = {}

        return lkwa
    

    def _load_frame(self):
        """A hidden method to load the frame data

        This method loads the frame data from the specified file path using the
        appropriate loading function. The loading function is determined by the
        frame format.

        Raises:
            NotImplementedError: If the frame format is not implemented

        Returns:
            numpy.ndarray: The frame data in RGB format [x,y,3]
        """
        if DEBUG:
            print('Loading frame...')

        func = None

        if self.frame_format == 'png':
            func = _load_png
        elif self.frame_format == 'jpg':
            func = _load_jpg
        elif self.frame_format == 'cr3':
            func = _load_cr3
        else:
            raise NotImplementedError(f"frame format not implemented: {self.frame_format}")
        
        return func(self.frame_path, self.loadkwargs)


# Custom loading functions -----------------------------------------------------

def _load_png(frame_path, loadkwargs):
    pass

def _load_jpg(frame_path, loadkwargs):
    pass

def _load_cr3(frame_path, loadkwargs):
    """A custom loading function for CR3 files

    This function loads a CR3 file using the rawpy package. The keyword arguments
    are passed to the rawpy.postprocess() function.

    Args:
        frame_path (str): The path to the CR3 file
        loadkwargs (dict): The keyword arguments for loading the CR3 file

    Returns:
        numpy.ndarray: The frame data in RGB format [x,y,3]
    """
    # loading a CR3 file can be done through the rawpy python package
    # https://pypi.org/project/rawpy/
    with rawpy.imread(frame_path) as raw:
        rgb = raw.postprocess(**loadkwargs)

    return rgb

