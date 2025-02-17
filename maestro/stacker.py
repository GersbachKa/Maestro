
import numpy as np
from scipy.optimize import minimize

from maestro.frame import Frame


class Light(Frame):
    def __init__(self, frame_path, frame_format=None, loadkwargs=None,
                 exclude=False):
        super().__init__(frame_path, frame_format, loadkwargs)

        self.frame_type = 'Light'
        self.offset = None # The affine offset of this frame relative to the first
        self.exclude = exclude # Whether to exclude this frame from stacking

    def get_stars(self, threshold=0.3, maximum=0.95):
        # Detect stars/bright objects in the frame
        pass


    def compute_affine(self, other, threshold=0.3, maximum=0.95):
        # Compute the affine transformation to align this frame with another
        # get stars in both frames
        selfstar = self.get_stars(threshold, maximum)
        otherstar = other.get_stars(threshold, maximum)

        # Now we need to define a cost function to minimize
        def cost_function(params):
            # The params will be the affine transformation matrix
            pass
        pass
        