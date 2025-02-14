import numpy as np

from .photo import Photo_base


class Bias_photo(Photo_base):
    def __init__(self, file_paths, photo_format=None, loadkwargs=None, method='median'):

        self.bias_paths = file_paths
        self.bias_count = len(file_paths)

        # Start by loading the photo format and load arguments
        self.photo_format, self.loadkwargs = self._parse_photo_args(file_paths[0], 
                                                    photo_format, loadkwargs)
        
        # Need to find the dimensions of the bias frames (assume they are all the same)
        temp_bias = self._load_photo(file_paths[0])
        self.xdim, self.ydim = temp_bias.shape[:2]
        self.npix = self.xdim * self.ydim

        # Get the original data type (We will use more accurate data types for the master bias)
        self.original_dtype = temp_bias.dtype
        
        # Biases tend to be dominated by single values, we should use floats for the master bias
        self.master_dtype = np.float16

        # Create the master bias frame
        self.master_bias = np.zeros((self.xdim, self.ydim), dtype=self.master_dtype)

        # Now create the master bias
        self.rgb = self.create_master_bias(method)

    
    def create_master_bias(self, method='median'):
        
        # Method can be 'median', 'mean', 'clipped_mean'

        # Load all of the bias frames
        all_biases = [self._load_photo(b) for b in self.bias_paths]
        
        # Now we need to use the different methods
        if method.lower() == 'median':
            master_bias = np.median(all_biases, axis=0)

        elif method.lower() == 'mean':
            master_bias = np.mean(all_biases, axis=0)

        elif method.lower() == 'clipped_mean':
            # Sigma clipping of outliers. We will use a 3-sigma clipping
            # to do so, we will calculate the current mean and standard deviation
            mean, std = np.mean(all_biases, axis=0), np.std(all_biases, axis=0)

            # Now we will remove the outliers
            to_keep = np.abs(all_biases - mean) < 3*std
            
            # Now recalculate the mean of the clipped data
            master_bias = np.mean(all_biases[to_keep], axis=0)

        else:
            raise ValueError(f"Master bias creation method not implemented: {method}")
        
        return master_bias.astype(self.master_dtype)
    
    # Overloaded methods --------------------------------------------------------
    
    def reload(self):
        self.master_bias = self.create_master_bias()
        return




