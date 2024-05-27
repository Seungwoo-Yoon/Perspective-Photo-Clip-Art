import numpy as np
from calibration import *
import cv2

def mapping(x: np.ndarray, P_origin: CameraParameter, P_target: CameraParameter) -> np.ndarray:
    # map the 2D coordinate in origin picture to the target picture
    # Coordinate Mapping
    x = homogeneous(x)
    mapped_x = P_target.K @ (
        P_target.R @ np.linalg.inv(P_origin.R) @ (np.linalg.inv(P_origin.K) @ x - P_origin.t)
        + P_target.t
    )
    
    return euclidian(mapped_x)

def mask(image: np.ndarray):
    # get the valid region from the image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    mask[image[:,:,-1] != 0] = 1
    return mask

def overwrite(bg: np.ndarray, obj: np.ndarray, 
              P_background: CameraParameter, P_object: CameraParameter) -> np.ndarray:
    
    new_image = bg.copy()
    alpha_mask = mask(obj)
    W, H = bg.shape[1], bg.shape[0]
    objH, objW = obj.shape[0], obj.shape[1]

    for x in range(W):
        print(x)
        for y in range(H):
            mapped_coordinate = mapping(np.array([x, y]), P_background, P_object)
            print(mapped_coordinate)
            if 0 <= int(mapped_coordinate[1]) < objH and 0 <= int(mapped_coordinate[0]) < objW:
                print(1)
                if alpha_mask[int(mapped_coordinate[1]), int(mapped_coordinate[0])] == 1:
                    new_image[y, x] = obj[int(mapped_coordinate[1]), int(mapped_coordinate[0])][0:3]

    return new_image