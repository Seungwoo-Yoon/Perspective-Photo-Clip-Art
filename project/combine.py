import numpy as np
from calibration import *

def mapping(x: np.ndarray, P_origin: CameraParameter, P_target: CameraParameter) -> np.ndarray:
    # TODO
    # map the 2D coordinate in origin picture to the target picture
    raise NotImplementedError()

def mask(image: np.ndarray):
    # TODO
    # get the valid region from the image
    raise NotImplementedError()

def overwrite(bg: np.ndarray, obj: np.ndarray, 
              P_background: CameraParameter, P_target: CameraParameter) -> np.ndarray:
    # TODO
    # overwrite object on the background
    raise NotImplementedError()