import numpy as np

from .photo import Photo_base

from tqdm.auto import tqdm


# Preprocess_base class --------------------------------------------------------
# Many of the preprocessing steps are identical between calibration frames. Use
# this class to have subsiquent versions inherit much of the functionality

class Preprocess_base(Photo_base):
    def __init__(self, photo_paths, photo_format=None, loadkwargs=None, method='median', verbose=False):
        self.verbose = verbose
        print('Initializing calibration frame...') if self.verbose else None;

        if not hasattr(photo_paths, '__iter__'):
            raise ValueError("Photo paths must be an iterable (list)")

        self.photo_path = photo_paths
        self.photo_count = len(photo_paths)

        # Start by loading the photo format and load arguments
        self.photo_format, self.loadkwargs = self._parse_photo_args(photo_paths[0], 
                                                    photo_format, loadkwargs)
        
        # Need to find the dimensions of the calibration frames (assume they are all the same)
        temp_frame = self._load_photo(photo_paths[0])
        self.xdim, self.ydim = temp_frame.shape[:2]
        self.npix = self.xdim * self.ydim

        # Get the original data type (We will use more accurate data types for the master)
        self.original_dtype = temp_frame.dtype
        
        # Many frames (biases) tend to be dominated by single values,
        # we should use floats for the master frames
        self.master_dtype = np.float16

        # Create the master frame
        self.master_frame = np.zeros((self.xdim, self.ydim), dtype=self.master_dtype)

        # Now create the master frame
        self.master_method = method
        self.master_frame = self.create_master_frame(method)
        self.rgb = self.master_frame.astype(self.original_dtype)

        if self.frame_type is None:
            self.frame_type = 'Preprocess_base'
        

    
    def create_master_frame(self, method='median'):
        print(f'Creating master frame using method: {method}') if self.verbose else None;
        
        # Method can be 'median', 'mean', 'clipped_mean'

        # Load all of the calibration frames
        all_frames = [self._load_photo(b) for b in tqdm(self.photo_path, 
                                                    desc=f'Loading {self.frame_type} frames')]
        
        # Now we need to use the different methods
        if method.lower() == 'median':
            master_frame = np.median(all_frames, axis=0)

        elif method.lower() == 'mean':
            master_frame = np.mean(all_frames, axis=0)

        elif method.lower() == 'clipped_mean':
            # Sigma clipping of outliers. We will use a 3-sigma clipping
            # to do so, we will calculate the current mean and standard deviation
            mean, std = np.mean(all_frames, axis=0), np.std(all_frames, axis=0)

            # Now we will remove the outliers
            to_keep = np.abs(all_frames - mean) < 3*std
            
            # Now recalculate the mean of the clipped data
            master_frame = np.mean(all_frames[to_keep], axis=0)

        else:
            raise ValueError(f"Master frame creation method not implemented: {method}")
        
        del all_frames # Free up memory (should be done automatically, but just in case)
        return master_frame.astype(self.master_dtype)
    
    # Overloaded methods --------------------------------------------------------
    
    def reload(self):
        print('Reloading calibration frames...') if self.verbose else None;

        self.master_frame = self.create_master_frame(self.master_method)
        self.rgb = self.master_frame.astype(self.original_dtype)
        return


# Bias class -------------------------------------------------------------------

class Bias_frame(Preprocess_base):
    def __init__(self, photo_paths, photo_format=None, loadkwargs=None, method='median', verbose=False):
        print('Initializing bias frame...') if verbose else None;
        self.frame_type = 'Bias'
        super().__init__(photo_paths, photo_format, loadkwargs, method, verbose)


# Dark class -------------------------------------------------------------------

class Dark_frame(Preprocess_base):
    def __init__(self, photo_paths, photo_format=None, loadkwargs=None, method='median', verbose=False):
        print('Initializing dark frame...') if verbose else None;
        self.frame_type = 'Dark'
        super().__init__(photo_paths, photo_format, loadkwargs, method)



# Flat class -------------------------------------------------------------------

class Flat_frame(Preprocess_base):
    def __init__(self, photo_paths, photo_format=None, loadkwargs=None, method='median', verbose=False):
        print('Initializing flat frame...') if verbose else None;
        self.frame_type = 'Flat'
        super().__init__(photo_paths, photo_format, loadkwargs, method)


