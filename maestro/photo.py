
import numpy as np

import rawpy

class Photo_base:

    def __init__(self, photo_path, photo_format=None, loadkwargs=None, verbose=False):
        self.verbose = verbose
        print('Initializing photo...') if self.verbose else None;

        self.photo_path = photo_path

        # Start with loading the photo format and load arguments
        self.photo_format, self.loadkwargs = self._parse_photo_args(photo_path, 
                                                    photo_format, loadkwargs)

        self.rgb = self._load_photo(photo_path)
        
        self.xdim, self.ydim = self.rgb.shape[:2]
        self.npix = self.xdim * self.ydim

        self.dtype = self.rgb.dtype
        return

    # Exposed methods ----------------------------------------------------------
    
    def reload(self):
        print('Reloading photo...') if self.verbose else None;

        self.rgb = self._load_photo(self.photo_path, self.photo_format, self.loadkwargs,
                                    return_attributes=False)
        return

    def get_grayscale(self, method='NTSC'):
        print('Converting to grayscale...') if self.verbose else None;

        if (method is None) or (method.lower() == 'ntsc'):
            factor = (0.299, 0.587, 0.114)
        else:
            raise ValueError(f"Grayscale conversion method not implemented: {method}")
        
        gray = factor[0]*self.rgb[:,:,0] + factor[1]*self.rgb[:,:,1] + factor[2]*self.rgb[:,:,2]
        return gray

    # Internal methods ---------------------------------------------------------

    def _parse_photo_args(self, photo_path, photo_format, loadkwargs):
        print('Parsing photo arguments...') if self.verbose else None;
        
        # For photo format
        if photo_format is None:
            # Use the file extension to determine the format
            fmt = photo_path.split('.')[-1]
        
        # For loading arguments
        if loadkwargs is None:
            # Raw formats require special default arguments
            if fmt.lower() == 'cr3':
                lkwa = {'use_camera_wb':True, 'no_auto_scale':False,
                              'no_auto_bright':True, 'chromatic_aberration':(1,1)}
            else:
                lkwa = {}

        return fmt, lkwa
        

    def _load_photo(self, photo_path):
        print('Loading photo...') if self.verbose else None;

        if self.photo_format.lower() == 'png':
            return _load_png(photo_path, self.loadkwargs)
        
        if self.photo_format.lower() == 'jpg':
            return _load_jpg(photo_path, self.loadkwargs)
        
        if self.photo_format.lower() == 'cr3':
            return _load_cr3(photo_path, self.loadkwargs)

        return None
    
    
    def _get_photo_attributes(self, photo_path):
        pass



# Custom loading functions -----------------------------------------------------

def _load_png(photo_path, loadkwargs):
    pass

def _load_jpg(photo_path, loadkwargs):
    pass

def _load_cr3(photo_path, loadkwargs):
    # loading a CR3 file can be done through the rawpy python package
    # https://pypi.org/project/rawpy/
    with rawpy.imread(photo_path) as raw:
        rgb = raw.postprocess(**loadkwargs)

    return rgb

