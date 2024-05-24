import numpy as np
from calibration import *
import cv2

def mapping(x: np.ndarray, P_origin: CameraParameter, P_target: CameraParameter) -> np.ndarray:
    # map the 2D coordinate in origin picture to the target picture
    # Coordinate Mapping
    mapped_obj = x.copy()
    height, width = mapped_obj.shape
    map_matrix = np.dot(P_origin, np.linalg.pinv(P_target))
    img_new = cv2.warpPerspective(x, map_matrix, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Rotate Vanishing point
    theta = 0.0 #FIXME from vanishing point
    R = cv2.getRotationMatrix2D(tuple(width/2, height/2), theta, 1.0)
    mapped_obj = cv2.warpAffine(mapped_obj, R, tuple(width,height))
    
    return mapped_obj

def mask(image: np.ndarray):
    # get the valid region from the image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    mask[image != 0] = 1
    return mask

def overwrite(bg: np.ndarray, obj: np.ndarray, 
              P_background: CameraParameter, P_target: CameraParameter, obj_offset: np.ndarray) -> np.ndarray:
    # |obj_offset : np.array([x_offset, y_offset])
    # overwrite object on the background
    if not obj_offset.shape != (2,):
        raise ValueError("obj_offset must be a numpy with 2 elements")
    
    # Map
    mapped_obj = mapping(obj, P_background, P_target)
    
    # Get mask
    mask = mask(mapped_obj).astype(bool)
    
    # Overlay
    bg_img = bg.copy()
    x_offset, y_offset = obj_offset
    # -set range of interest
    mapped_obj_height, mapped_obj_width = mapped_obj.shape
    bg_img_height, bg_img_width = bg_img.shape
    x_end = x_offset + mapped_obj_width
    y_end = y_offset + mapped_obj_height
    x_end = min(x_end, bg_img_width)
    y_end = min(y_end, bg_img_height)
    x_start = max(x_offset, 0)
    y_start = max(y_offset, 0)
    obj_x_start = x_start - x_offset
    obj_y_start = y_start - y_offset
    # -image clipping
    obj_valid = mapped_obj[obj_y_start:y_end - y_offset,
                           obj_x_start:x_end - x_offset]
    mask_valid = mask[obj_y_start:y_end - y_offset,
                      obj_x_start:x_end - x_offset]
    roi_bg_img = bg_img[y_start:y_end, x_start:x_end]
    roi_bg_img[mask_valid] = obj_valid[mask_valid]
    return bg_img