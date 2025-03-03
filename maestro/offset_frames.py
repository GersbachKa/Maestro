
import numpy as np
from scipy.optimize import minimize


def compute_affine(frame1, frame2, allow_skew=False, allow_dithering=False):
    pass
    

def find_bright_sources(frames, threshold=0.5):
    bright_sources = []

    for frame in frames:
        gray = frame.get_grayscale()
        max_val = np.max(gray)
        mask = gray < (threshold*max_val)
        gray[mask] = 0
        bright_sources.append(gray)
    
    return bright_sources

        