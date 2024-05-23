import numpy as np
from calibration import *
import cv2

def mapping(x: np.ndarray, P_origin: CameraParameter, P_target: CameraParameter) -> np.ndarray:
    # map the 2D coordinate in origin picture to the target picture
    mapped_obj = x.copy()
    # Coordinate Mapping
    # TODO
    
    # Rotate Vanishing point
    theta = 0.0 #FIXME from vanishing point
    R = cv2.getRotationMatrix2D(tuple(mapped_obj.shape[1]/2, mapped_obj.shape[0]/2), theta, 1.0)
    mapped_obj = cv2.warpAffine(mapped_obj, R, tuple(mapped_obj.shape[1],mapped_obj.shape[0]))
    return mapped_obj

def mask(image: np.ndarray):
    # TODO
    # get the valid region from the image
    raise NotImplementedError()

def overwrite(bg: np.ndarray, obj: np.ndarray, 
              P_background: CameraParameter, P_target: CameraParameter) -> np.ndarray:
    # overwrite object on the background
    mapped_obj = mapping(obj, P_background, P_target)
    # TODO
    raise NotImplementedError()