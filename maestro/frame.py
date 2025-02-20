
import numpy as np
import rawpy
from matplotlib import pyplot as plt

from maestro import DEBUG

global FORMAT_SUPPORT
FORMAT_SUPPORT = ['png', 'jpg', 'cr3', 'master', 'null']



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
        frame_type (str): The type of frame (e.g. 'bias', 'dark', 'flat', 'light', 'master', 'null')
    """
    def __init__(self, frame_path, frame_format=None, loadkwargs=None, frame_type=None,
                 rgb_set=None):
        """A Frame object constructor

        This constructor initializes a Frame object by loading the frame data from the
        specified file path. The frame format and load arguments can be specified, but
        can be inferred from the file path if not.

        If frame_format is a raw format (e.g. CR3, NEF), the loadkwargs will be
        given to the rawpy.postprocess() function. Generally, the defaults are fine,
        but can be overridden if you want the customization.

        NOTE: If setting the frame data directly (i.e. rgb_set), the numpy array
        will enforce positivity. This is because the frame data should not have
        negative values.

        Args:
            frame_path (str): The path to the frame file
            frame_format (str, optional): The format of the frame file. Defaults to None.
            loadkwargs (dict, optional): The keyword arguments to be passed to the frame 
                loading function. Defaults to None.
            rgb_set (numpy.ndarray, optional): The frame data in RGB format. Only works
                if frame_format is 'master'. Defaults to None.
        """
        if DEBUG:
            print('Initializing frame...')

        # Set the frame path, format, and load arguments
        self.frame_path = frame_path

        if frame_format is None:
            self.frame_format = _get_frame_format(frame_path) 
        else: 
            self.frame_format = frame_format.lower()

        if loadkwargs is None:
            self.loadkwargs = _get_frame_load_kwargs(self.frame_format)
        else:
            self.loadkwargs = loadkwargs
        
        # Now load the frame
        if rgb_set is not None:
            # Enforce positivity
            rgb_set[rgb_set < 0] = 0
            self.rgb = rgb_set
        else:
            self.rgb = self._load_frame()

        # Now set the few pieces of metadata we can get from the frame
        # TODO: Get more metadata from raw formats (CR3, NEF, etc)
        self.dimensions = self.rgb.shape
        self.npix = self.dimensions[0] * self.dimensions[1]
        self.dtype = self.rgb.dtype
        self.frame_type = 'Unknown' if frame_type is None else frame_type


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


    def show(self, grayscale=False, scaling=1.0, to_uint8=True):
        """Show the frame data through matplotlib imshow

        This method shows the frame data using the matplotlib imshow function. The
        frame data can be shown in grayscale if specified. Note that this method
        will show the frame data as a uint8 array, so the data may be clipped if
        the data is not in the range [0, 255]. 
        
        Alternatively, the 'scaling' parameter can be used to scale the data before
        showing. This parameter is a float that will be multiplied by the data before
        showing.

        Args:
            grayscale (bool): A flag to show the frame data in grayscale.
                Defaults to False.
            auto_scale (bool): A flag to rescale the frame data to [0, 255].
                Defaults to False.
            to_uint8 (bool): A flag to convert the frame data to uint8 before
                showing. Defaults to False.
        """
        to_show = self.rgb if not grayscale else self.get_grayscale()

        to_show = scaling*to_show

        if to_uint8:
            to_show = to_show.astype(np.uint8)

        plt.imshow(to_show, cmap='gray')


    def __add__(self, other):
        """Method overload for '+' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for +: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to add them')
            
            new_rgb = self.rgb + other.rgb
        
        else:
            new_rgb = self.rgb + other
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame
    

    def __radd__(self, other):
        """Method overload for '+' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for +: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to add them')
            
            new_rgb = other.rgb + self.rgb
        
        else:
            new_rgb = other + self.rgb
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame


    def __sub__(self, other):
        """Method overload for '-' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for -: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to subtract them')
            
            new_rgb = self.rgb - other.rgb
        
        else:
            new_rgb = self.rgb - other
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame
    

    def __rsub__(self, other):
        """Method overload for '-' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for -: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to subtract them')
            
            new_rgb = other.rgb - self.rgb
        
        else:
            new_rgb = other - self.rgb
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame


    def __mul__(self, other):
        """Method overload for '*' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for *: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to multiply them')
            
            new_rgb = self.rgb * other.rgb
        
        else:
            new_rgb =  self.rgb * other
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame
    
    
    def __rmul__(self, other):
        """Method overload for '*' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for *: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to multiply them')
            
            new_rgb = other.rgb * self.rgb
        
        else:
            new_rgb = other * self.rgb
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame


    def __truediv__(self, other):
        """Method overload for '/' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for /: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to divide them')
            
            new_rgb = self.rgb / other.rgb
        
        else:
            new_rgb = self.rgb / other
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame
    
    
    def __rtruediv__(self, other):
        """Method overload for '/' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for /: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to divide them')
            
            new_rgb = other.rgb / self.rgb
        
        else:
            new_rgb = other / self.rgb
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame
    
    
    def __pow__(self, other):
        """Method overload for '**' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for **: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to exponentiate them')
            
            new_rgb = self.rgb ** other.rgb
        
        else:
            new_rgb = self.rgb ** other
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame
    
    
    def __rpow__(self, other):
        """Method overload for '**' operator"""
        # Use guard clauses to check if this is allowed
        if type(other) not in MATH_SUPPORTED_TYPES:
            raise TypeError(f"unsupported operand type(s) for **: 'Frame' and '{type(other)}'")
        
        # Check if the frames are the same size
        if type(other) is Frame:
            if self.dimensions != other.dimensions:
                raise ValueError('Frames must be the same size to exponentiate them')
            
            new_rgb = other.rgb ** self.rgb
        
        else:
            new_rgb = other ** self.rgb
        
        # Now we need to return a new Frame object
        new_frame = Frame(frame_path=None, frame_format='null', rgb_set=new_rgb)
        return new_frame


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
        elif self.frame_format == 'master':
            raise NotImplementedError('Master frame loading does not make sense')
        elif self.frame_format == 'null':
            raise NotImplementedError('Null frame loading does not make sense')
        else:
            raise NotImplementedError(f"frame format not implemented: {self.frame_format}")
        
        return func(self.frame_path, self.loadkwargs)


# Supported types for math operations -------------------------------------------

MATH_SUPPORTED_TYPES = [int, float, np.ndarray, Frame, 
                        np.float64, np.float32, np.float16,
                        np.int64, np.int32, np.int16, np.int8,
                        np.uint64, np.uint32, np.uint16, np.uint8]


# Default loading arguments -----------------------------------------------------

def _get_frame_format(frame_path):
    """A hidden function to get the frame format if not specified

    NOTE: that this method breaks if the frame_path has multiple '.' characters
    in the extension. This is a known issue and will be fixed in a future version.

    Args:
        frame_path (str): The path to the frame file

    Returns:
        str: The format of the frame file
    """
    if DEBUG:
        print('Getting frame format...')

    fmt = (frame_path.split('.')[-1]).lower()

    if fmt not in FORMAT_SUPPORT:
        raise ValueError(f'Unsupported frame format: {fmt}')
    return fmt


def _get_frame_load_kwargs(frame_format):
    """A hidden function to get the frame load keyword arguments if not specified

    This method returns the default keyword arguments for loading a frame in the
    specified format. This is useful for raw formats (e.g. CR3, NEF) where the
    rawpy.postprocess() function is used.

    Returns:
        dict: The keyword arguments for loading the frame
    """
    if DEBUG:
        print('Getting frame load keyword arguments...')

    if frame_format in ['cr3']:
        lkwa = {'use_camera_wb':True, 'no_auto_scale':False,
                'no_auto_bright':True, 'chromatic_aberration':(1,1)}
    else:
        lkwa = {}

    return lkwa


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

