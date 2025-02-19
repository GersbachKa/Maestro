
import numpy as np
import warnings
import os
from glob import glob
from tqdm.auto import tqdm

from maestro import DEBUG
from maestro.frame import Frame, FORMAT_SUPPORT, _get_frame_load_kwargs

MASTERING_METHODS = ['median', 'mean', 'average', 'clipped_mean', 'clipped_average']


class Stacker:

    def __init__(self, stacker_file=None, folder_path=None, bias_paths=None,
                 dark_paths=None, flat_paths=None, light_paths=None, 
                 frame_format=None, frame_load_kwargs=None,
                 mastering_method='median', 
                 camera_settings=None, keep_intermediate=False):
        if DEBUG:
            print('Initializing stacker...')
        
        # Check if stacker file is provided, if so, load that and ignore other inputs
        if stacker_file is not None:
            if DEBUG:
                print(f'Loading stacker file: {stacker_file}')
            self.load_stacker(stacker_file)
        
        # Paths to the frames
        self.frame_paths = {'lights': [], 'biases': [], 'darks': [], 'flats': []}

        # Frame objects
        self.biases, self.master_bias = [], []
        self.darks, self.master_dark = [], []
        self.flats, self.master_flat = [], []
        self.lights, self.reduced_lights = [], []

        # Set frame format and load kwargs
        self.set_frame_format(frame_format)
        self.set_frame_load_kwargs(frame_load_kwargs)

        # Load frames from paths
        if folder_path is not None:
            self.import_folder(folder_path)
        if bias_paths is not None:
            self.import_bias(bias_paths)
        if dark_paths is not None:
            self.import_dark(dark_paths)
        if flat_paths is not None:
            self.import_flat(flat_paths)
        if light_paths is not None:
            self.import_light(light_paths)
        
        

        

        # Set mastering method
        self.mastering_method = mastering_method

        # Camera settings
        self.camera_settings = {'iso':None, 'exposure_time':None, 'f_stop':None, 
                                'focal_length':None, 'camera_model':None, 
                                'camera_manufacturer':None, 'camera_name':None}
        
        if camera_settings is not None:
            self.camera_settings.update(camera_settings)
        
        # Keep intermediate frames if desired
        self.keep_intermediate = keep_intermediate
        

    def import_folder(self, folder_path, overwrite=False):
        if DEBUG:
            print(f'Importing folder from {folder_path}...')
        
        # Get the list of subdirectories
        subdirs = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        # Check subdirectories names for frame types
        for sub in subdirs:
            files = glob(sub+'/*')

            if 'bias' in sub.split('/')[-1].lower():
                self.import_bias(files, overwrite)
            elif 'dark' in sub.split('/')[-1].lower():
                self.import_dark(files, overwrite)
            elif 'flat' in sub.split('/')[-1].lower():
                self.import_flat(files, overwrite)
            elif 'light' in sub.split('/')[-1].lower():
                self.import_light(files, overwrite)


    def import_bias(self, frame_paths=None, overwrite=False):
        if overwrite:
            self.frame_paths['biases'] = []
            self.biases, self.master_bias = [], []
        
        added = self._import_frames(frame_paths, 'biases')

        if added > 0:
            # New biases added, so we need to update the master bias
            self.biases, self.master_bias = [], []

        return added 
    

    def import_dark(self, frame_paths=None, overwrite=False):
        if overwrite:
            self.frame_paths['darks'] = []
            self.darks, self.master_dark  = [], []

        added = self._import_frames(frame_paths, 'darks')

        if added > 0:
            # New darks added, so we need to update the master dark
            self.darks, self.master_dark = [], []

        return added


    def import_flat(self, frame_paths=None, overwrite=False):
        if overwrite:
            self.frame_paths['flats'] = []
            self.flats, self.master_flat = [], []

        added = self._import_frames(frame_paths, 'flats')

        if added > 0:
            # New flats added, so we need to update the master flat
            self.flats, self.master_flat = [], []

        return added


    def import_light(self, frame_paths=None, overwrite=False):
        if overwrite:
            self.frame_paths['lights'] = []
            self.lights, self.reduced_lights = [], []

        added = self._import_frames(frame_paths, 'lights')

        # Reload all lights
        self.lights = self._load_frames(self.frame_paths['lights'], 'lights')
        
        # Reset reduced lights
        self.reduced_lights = []

        return added


    def _import_frames(self, frame_paths, frame_type):
        if DEBUG:
            print(f'Importing {frame_type} frames...')

        pre = len(self.frame_paths[frame_type])

        if frame_paths is None:
            return 0 
        if hasattr(frame_paths, '__iter__'):
            self.frame_paths[frame_type] += frame_paths
        elif isinstance(frame_paths, str):
            self.frame_paths[frame_type].append(frame_paths)
        else:
            raise ValueError('frame_paths must be a list of strings or a string')
        
        # Remove duplicates
        self.frame_paths[frame_type] = list(set(self.frame_paths[frame_type]))

        post = len(self.frame_paths[frame_type])
        return post - pre
    

    def _load_frames(self, frame_paths, ftype=None):
        if DEBUG:
            print(f'Loading many frames...')
        
        # Check if format and load kwargs are set
        if self.frame_format is None:
            self.set_frame_format(None)

        frames = [Frame(path, self.frame_format, self.frame_load_kwargs, ftype) \
                  for path in tqdm(frame_paths, desc=f'Loading {ftype}')]
        
        return frames


    def set_frame_format(self, frame_format=None):
        if DEBUG:
            print(f'Setting frame format to {frame_format}')
        
        if frame_format is None:
            # Get frame format from all frames
            paths = self.frame_paths['lights'] + self.frame_paths['biases'] + \
                    self.frame_paths['darks'] + self.frame_paths['flats']
            # Get file extensions: Will not work for double extensions like .tar.gz
            splits = [path.split('.')[-1] for path in paths]
            # Get unique extensions
            extension = np.unique(splits)
            if len(extension) == 0:
                self.frame_format = None
                return

            if len(extension) > 1:
                raise ValueError(f'Multiple frame formats detected and are '+
                                  'not supported yet. Unique formats: {extension}') 
            if extension[0].lower() not in FORMAT_SUPPORT:
                raise ValueError(f'Unsupported frame format: {extension[0]}')
            
            self.frame_format = extension[0]
            return 

        fmt = frame_format.lower().split('.')[-1]

        if fmt not in FORMAT_SUPPORT:
            raise ValueError(f'Unsupported frame format: {fmt}')
        
        self.frame_format = frame_format
        return
    

    def set_frame_load_kwargs(self, frame_load_kwargs=None):
        if DEBUG:
            print(f'Setting frame load kwargs to {frame_load_kwargs}')
        
        if frame_load_kwargs is None:
            # Default load kwargs depending on frame format
            self.frame_load_kwargs = None
        elif isinstance(frame_load_kwargs, dict):
            self.frame_load_kwargs = frame_load_kwargs
        else:
            self.frame_load_kwargs = dict(frame_load_kwargs)

        return


    def set_mastering_method(self, method):
        if DEBUG:
            print(f'Setting mastering method to {method}')
        
        if method.lower() not in MASTERING_METHODS:
            raise ValueError(f'Unsupported mastering method: {method}')
        else:
            self.mastering_method = method.lower()
        return


    def create_master_frame(self, frame_type, method=None, keep_individual=None):
        if DEBUG:
            print(f'Creating master {frame_type} frame using {self.mastering_method} method')
        
        # Initial checks--------------------------------------------------------
        ftype = None
        ftype = 'biases' if frame_type.lower() in ['bias','biases'] else ftype
        ftype = 'darks' if frame_type.lower() in ['dark','darks'] else ftype
        ftype = 'flats' if frame_type.lower() in ['flat','flats'] else ftype

        if frame_type.lower() in ['light','lights']:
            # Master frames are intended to be used by calibration frames, not lights
            raise ValueError('Mastering light frames are done by stacking.' + \
                             'Check stack_light_frame() method')
        if ftype is None:
            raise ValueError(f'Unknown frame type: {frame_type}')
        
        if method is None and self.mastering_method is None:
            raise ValueError('Mastering method not set. Please set it using set_mastering_method()')
        
        mtd = method.lower() if method is not None else self.mastering_method

        keep = self.keep_intermediate if keep_individual is None else keep_individual

        # Load all frames-------------------------------------------------------        
        all_frames = self._load_frames(self.frame_paths[ftype], ftype)
        all_rgb = [frame.rgb for frame in all_frames]

        # Apply mastering methods-----------------------------------------------
        if mtd in ['median']:
            master_frame = median_mastering(all_rgb)
        elif mtd in ['mean', 'average']:
            master_frame = mean_mastering(all_rgb)
        elif mtd in ['clipped_mean', 'clipped_average']:
            master_frame = clipped_mean_mastering(all_rgb)
        else:
            raise Exception(f'Unsupported mastering method {method}')

        # Save master frames----------------------------------------------------
        if ftype == 'biases':
            self.biases = all_frames if keep else None
            master_frame.frame_type = 'master_bias'
            self.master_bias = master_frame
        elif ftype == 'darks':
            self.darks = all_frames if keep else None
            master_frame.frame_type = 'master_dark'
            self.master_dark = master_frame
        elif ftype == 'flats':
            self.flats = all_frames if keep else None
            master_frame.frame_type = 'master_flat'
            self.master_flat = master_frame
        else:
            raise Exception('Unknown frame type: How did you get here?!?')

        return master_frame
    

    def calibrate_light_frame(self):
        if DEBUG:
            print('Calibrating light frames...')
        pass


    def stack_light_frame(self):
        if DEBUG:
            print('Stacking light frames...')
        pass


    def compute_light_offsets(self):
        if DEBUG:
            print('Computing light offsets...')
        pass


    def save_stacker(self, stacker_file):
        if DEBUG:
            print(f'Saving stacker to {stacker_file}')
        pass


    def load_stacker(self, stacker_file):
        if DEBUG:
            print(f'Loading stacker from {stacker_file}')
        pass

    
def median_mastering(all_rgb):
    master_frame = np.median(all_rgb, axis=0)
    frame_obj = Frame(None, 'master', rgb_set=master_frame)
    return frame_obj


def mean_mastering(all_rgb):
    master_frame = np.mean(all_rgb, axis=0)
    frame_obj = Frame(None, 'master', rgb_set=master_frame)
    return frame_obj


def clipped_mean_mastering(all_rgb):
    # Sigma clipping of outliers. We will use a 3-sigma clipping
    # to do so, we will calculate the current mean and standard deviation
    mean, std = np.mean(all_rgb, axis=0), np.std(all_rgb, axis=0)
    # Now we will remove the outliers
    to_keep = np.abs(all_rgb - mean) < 3*std
    # Now recalculate the mean of the clipped data
    master_frame = np.mean(all_rgb[to_keep], axis=0)
    frame_obj = Frame(None, 'master', rgb_set=master_frame)
    return frame_obj