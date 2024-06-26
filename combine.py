import numpy as np
from calibration import *
import cv2
from tqdm import tqdm

def mapping(x: np.ndarray, P_origin: CameraParameter, P_target: CameraParameter) -> np.ndarray:
    # map the 2D coordinate in origin picture to the target picture
    # Coordinate Mapping
    x = multiple_homogeneous(x)

    # scale = C_target_norm / C_origin_norm
    scale = 1

    mapped_x = (P_target.P @ np.diag([scale, scale, scale, 1]) @ np.linalg.pinv(P_origin.P) @ x.T).T
    
    return multiple_euclidian(mapped_x)

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
    xs, ys = np.meshgrid(range(W), range(H))
    mapped_coordinates = np.array((xs, ys)).transpose((1, 2, 0))
    mapped_coordinates = mapping(mapped_coordinates.reshape(-1, 2), P_background, P_object).reshape(H, W, 2)

    for x in range(W):
        for y in range(H):
            mapped_coordinate = mapped_coordinates[y, x]
            if 0 <= int(mapped_coordinate[1]) < objH and 0 <= int(mapped_coordinate[0]) < objW:
                if alpha_mask[int(mapped_coordinate[1]), int(mapped_coordinate[0])] == 1:
                    new_image[y, x, :3] = obj[int(mapped_coordinate[1]), int(mapped_coordinate[0]), :-1]

    return new_image