
import numpy as np
import warnings
import os
from glob import glob
from tqdm.auto import tqdm

from maestro import DEBUG
from maestro.frame import Frame, FORMAT_SUPPORT, _get_frame_load_kwargs
from maestro.exceptions import *

MASTERING_METHODS = ['median', 'mean', 'average', 'clipped_mean', 'clipped_average']


class Stacker:

    def __init__(self, stacker_file=None, folder_path=None, bias_paths=None,
                 dark_paths=None, flat_paths=None, light_paths=None, load_all=False,
                 frame_format=None, frame_load_kwargs=None,
                 mastering_method='median', 
                 camera_settings=None):
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
        self.biases, self.master_bias = [], None
        self.darks, self.master_dark = [], None
        self.flats, self.master_flat = [], None
        self.lights, self.reduced_lights = [], []

        # Set frame format and load kwargs
        self.set_frame_format(frame_format)
        self.set_frame_load_kwargs(frame_load_kwargs)

        # Load frames from paths
        if folder_path is not None:
            self.import_folder(folder_path, load_all=load_all)
        if bias_paths is not None:
            self.import_bias(bias_paths, load_frames=load_all)
        if dark_paths is not None:
            self.import_dark(dark_paths, load_frames=load_all)
        if flat_paths is not None:
            self.import_flat(flat_paths, load_frames=load_all)
        if light_paths is not None:
            self.import_light(light_paths, load_frames=True)

        # Set mastering method
        self.mastering_method = mastering_method

        # Camera settings
        self.camera_settings = {'iso':None, 'exposure_time':None, 'f_stop':None, 
                                'focal_length':None, 'camera_model':None, 
                                'camera_manufacturer':None, 'camera_name':None}
        
        if camera_settings is not None:
            self.camera_settings.update(camera_settings)
        

    def import_folder(self, folder_path, overwrite=False, load_all=False):
        if DEBUG:
            print(f'Importing folder from {folder_path}...')
        
        # Get the list of subdirectories
        subdirs = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        # Check subdirectories names for frame types
        for sub in subdirs:
            files = glob(sub+'/*')

            if 'bias' in sub.split('/')[-1].lower():
                print(f'Importing bias frames from [{sub}]')
                added = self.import_bias(files, overwrite, load_all)
                if load_all:
                    print(f'Imported and loaded {added} bias frames')
                else:
                    print(f'Imported {added} bias frames')

            elif 'dark' in sub.split('/')[-1].lower():
                print(f'Importing dark frames from [{sub}]')
                added = self.import_dark(files, overwrite, load_all)
                if load_all:
                    print(f'Imported and loaded {added} dark frames')
                else:
                    print(f'Imported {added} dark frames')

            elif 'flat' in sub.split('/')[-1].lower():
                print(f'Importing flat frames from [{sub}]')
                added = self.import_flat(files, overwrite, load_all)
                if load_all:
                    print(f'Imported and loaded {added} flat frames')
                else:
                    print(f'Imported {added} flat frames')

            elif 'light' in sub.split('/')[-1].lower():
                print(f'Importing light frames from [{sub}]')
                added = self.import_light(files, overwrite, True)
                print(f'Imported and loaded {added} light frames')

            else:
                print(f'Unrecognized subdirectory. [{sub}], skipping...')


    def import_bias(self, frame_paths=None, overwrite=False, load_frames=False):
        if overwrite:
            self.frame_paths['biases'] = []
            self.biases, self.master_bias = [], None
        
        if load_frames:
            added, frames = self._import_frames(frame_paths, 'biases', True)
            self.biases = frames
            self.master_bias = None
        else:
            added = self._import_frames(frame_paths, 'biases', False)
            if added > 0:
                self.biases, self.master_bias = [], None
            
        return added 
    

    def import_dark(self, frame_paths=None, overwrite=False, load_frames=False):
        if overwrite:
            self.frame_paths['darks'] = []
            self.darks, self.master_dark  = [], None

        if load_frames:
            added, frames = self._import_frames(frame_paths, 'darks', True)
            self.darks = frames
            self.master_dark = None
        else:
            added = self._import_frames(frame_paths, 'darks', False)
            if added > 0:
                self.darks, self.master_dark = [], None
                
        return added


    def import_flat(self, frame_paths=None, overwrite=False, load_frames=False):
        if overwrite:
            self.frame_paths['flats'] = []
            self.flats, self.master_flat = [], None

        if load_frames:
            added, frames = self._import_frames(frame_paths, 'flats', True)
            self.flats = frames
            self.master_flat = None
        else:
            added = self._import_frames(frame_paths, 'flats', False)
            if added > 0:
                self.flats, self.master_flat = [], None
        
        return added


    def import_light(self, frame_paths=None, overwrite=False, load_frames=True):
        if overwrite:
            self.frame_paths['lights'] = []
            self.lights, self.reduced_lights = [], []

        if load_frames:
            added, frames = self._import_frames(frame_paths, 'lights', True)
            self.lights = frames
            self.reduced_lights = []
        else:
            added = self._import_frames(frame_paths, 'lights', False)
            if added > 0:
                self.lights, self.reduced_lights = [], []
            
        return added


    def _import_frames(self, frame_paths, frame_type, load_frames=False):
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
            raise UnknownFramePathException('frame_paths must be a list of strings or a string')
        
        # Remove duplicates
        self.frame_paths[frame_type] = list(set(self.frame_paths[frame_type]))

        post = len(self.frame_paths[frame_type])

        if load_frames:
            frames = self._load_frames(self.frame_paths[frame_type], frame_type)
            return post-pre, frames
        else:
            return post-pre
    

    def _load_frames(self, frame_paths, ftype=None):
        if DEBUG:
            print(f'Loading frames...')
        
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
            extension = list(set(splits))
            if len(extension) == 0:
                self.frame_format = None
                return

            if len(extension) > 1:
                msg = f'Multiple frame formats detected and are not supported yet. '+\
                        f'Unique formats: {extension}'
                raise MultipleFrameFormatException(msg) 
            
            if extension[0].lower() not in FORMAT_SUPPORT:
                msg = f'Unsupported frame format: {extension[0]}'
                raise UnsupportedFrameFormatException(msg)
            
            self.frame_format = extension[0]

        else:
            fmt = frame_format.lower().split('.')[-1]

            if fmt not in FORMAT_SUPPORT:
                raise UnsupportedFrameFormatException(f'Unsupported frame format: {fmt}')
        
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
            msg = f'Unsupported mastering method: {method}'
            raise UnsupportedMasteringMethodException(msg)
        
        else:
            self.mastering_method = method.lower()

        return


    def create_master_frame(self, frame_type, method=None):
        if DEBUG:
            print(f'Creating master {frame_type} frame using {self.mastering_method} method')
        
        # Initial checks--------------------------------------------------------
        ftype = None
        ftype = 'biases' if frame_type.lower() in ['bias','biases'] else ftype
        ftype = 'darks' if frame_type.lower() in ['dark','darks'] else ftype
        ftype = 'flats' if frame_type.lower() in ['flat','flats'] else ftype

        if frame_type.lower() in ['light','lights']:
            # Master frames are intended to be used by calibration frames, not lights
            msg = 'Mastering light frames is done by stacking. Check stack_light_frame() method'
            raise MasteringLightsException(msg)
        
        if ftype is None:
            raise UnknownFrameTypeException(f'Unknown frame type: {frame_type}')
        
        if method is None and self.mastering_method is None:
            msg = 'Mastering method not set. Please set it using set_mastering_method()'
            raise UnsetMasteringMethodException(msg)
        
        mtd = method.lower() if method is not None else self.mastering_method

        # Load all frames-------------------------------------------------------    
        # check if there are frames to master
        if len(self.frame_paths[ftype]) == 0:
            msg = f'No {ftype} frames to master. Add them using the import methods.'
            raise MissingFramesException(msg)
        
        # Check if the frames are loaded
        all_frames = None
        if ftype == 'biases' and len(self.biases)>0:
            all_frames = self.biases
        elif ftype == 'darks' and len(self.darks)>0:
            all_frames = self.darks
        elif ftype == 'flats' and len(self.flats)>0:
            all_frames = self.flats
        
        if all_frames is None:
            # Need to load the frames then
            all_frames = self._load_frames(self.frame_paths[ftype], ftype)
        
        # Get the RGB data from the frames
        all_rgb = [frame.rgb for frame in all_frames]

        # Apply mastering methods-----------------------------------------------
        if mtd in ['median']:
            master_frame = median_mastering(all_rgb)
        elif mtd in ['mean', 'average']:
            master_frame = mean_mastering(all_rgb)
        elif mtd in ['clipped_mean', 'clipped_average']:
            master_frame = clipped_mean_mastering(all_rgb)
        else:
            msg = f'Unsupported mastering method: {mtd}'
            raise UnsupportedMasteringMethodException(msg)

        # Save master frames----------------------------------------------------
        if ftype == 'biases':
            master_frame.frame_type = 'master_bias'
            self.master_bias = master_frame
        elif ftype == 'darks':
            master_frame.frame_type = 'master_dark'
            self.master_dark = master_frame
        elif ftype == 'flats':
            master_frame.frame_type = 'master_flat'
            self.master_flat = master_frame
        else:
            msg = 'Unknown frame type: How did you get here?!?'
            raise UnknownFrameTypeException(msg)

        return master_frame
    

    def calibrate_light_frames(self, ignore_missing_masters=False):
        if DEBUG:
            print('Calibrating light frames...')

        if len(self.lights) == 0:
            msg = 'No light frames to calibrate. Add them using the import methods.'
            raise MissingFramesException(msg)
        
        # Calibrating the light frames is easy if we have the master calibration frames
        if self.master_bias is None:
            # Create master bias
            try:
                B = self.create_master_frame('bias')
            except MissingFramesException as e:
                if ignore_missing_masters:
                    warnings.warn('Missing bias frames, ignoring...')
                    B = 0
                else:
                    raise e
        else:
            B = self.master_bias
        
        if self.master_dark is None:
            # Create master dark
            try:
                D = self.create_master_frame('dark')
            except MissingFramesException as e:
                if ignore_missing_masters:
                    warnings.warn('Missing dark frames, ignoring...')
                    D = 0
                else:
                    raise e
        else:
            D = self.master_dark
        
        if self.master_flat is None:
            # Create master flat
            try:
                F = self.create_master_frame('flat')
            except MissingFramesException as e:
                if ignore_missing_masters:
                    warnings.warn('Missing flat frames, ignoring...')
                    F = 0
                else:
                    raise e
        else:
            F = self.master_flat
        
        # Load light frames if they are not loaded
        if len(self.lights) == 0:
            self.lights = self._load_frames(self.frame_paths['lights'], 'lights')

        # Now calibrate the light frames using the calibration equation
        reduced_lights = []

        dark_wo_bias = D - B
        flat_wo_bias = F - B

        for light in tqdm(self.lights, desc='Creating reduced lights'):
            # reduced = ( L-(D-B) - B ) / ( F-B )
            reduced = ((light - B) - dark_wo_bias) / (flat_wo_bias) 

            # We can keep brightness roughly the same by calculating the means
            # and multiplying by the ratio of the means
            mean = np.mean(light.rgb)
            reduced_mean = np.mean(reduced.rgb)
            reduced *= mean/reduced_mean

            reduced_lights.append(reduced)

        self.reduced_lights = reduced_lights
        return reduced_lights


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