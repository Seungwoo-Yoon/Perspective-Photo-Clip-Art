import numpy as np
from calibration import *
import cv2
from tqdm import tqdm

def mapping(x: np.ndarray, P_origin: CameraParameter, P_target: CameraParameter, option=None) -> np.ndarray:
    # map the 2D coordinate in origin picture to the target picture
    # Coordinate Mapping
    x = multiple_homogeneous(x)
    
    if option is None:
        mapped_x = (P_target.P @ np.linalg.pinv(P_origin.P) @ x.T).T
    else:
        mapped_x = (P_target.P @ np.linalg.pinv(P_origin.P) @ option @ x.T).T # for bg to obj
        # mapped_x = (option @ P_target.P @ np.diag([scale, scale, scale, 1]) @ np.linalg.pinv(P_origin.P) @ x.T).T # for obj to bg
        # mapped_x = (np.linalg.inv(option) @ P_target.P @ np.diag([scale, scale, scale, 1]) @ np.linalg.pinv(P_origin.P) @ x.T).T
    
    return multiple_euclidian(mapped_x)

def mask(image: np.ndarray):
    # get the valid region from the image
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    mask[image[:,:,-1] != 0] = 1
    return mask

def interpolate_pixel(image, y, x):
    x0, x1 = int(x), min(int(x) + 1, image.shape[1] - 1)
    y0, y1 = int(y), min(int(y) + 1, image.shape[0] - 1)

    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def mycross(a,b):
    return a[1]*b[0] - a[0]*b[1]

def myPerspective(ori_p,bg_p):
    # ori_p: 4x2, bg_p: 4x2
    A = np.zeros((8,9))
    for i in range(4):
        A[2*i,:3] = np.array([*ori_p[i],1])
        A[2*i+1,3:6] = np.array([*ori_p[i],1])
        A[2*i,6:] = -bg_p[i][0] * np.array([*ori_p[i],1])
        A[2*i+1,6:] = -bg_p[i][1] * np.array([*ori_p[i],1])
    _,_,Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3,3)

def getSupportPoints(vanishing_lines,start,end,lastPoints=None, scale=1.0, scale2=1.0):
    #= Hyperparameter =#
    line_length = 90 # 90 #FIXME important for natualness
    sec_line_offset = 1e-6
    #= Hyperparameter =#
    ori_p_1 = vanishing_lines[0,0]
    if lastPoints is None:
        temp_ori_p = vanishing_lines[start,1]
        ori_p_2 = ori_p_1 + line_length * (temp_ori_p - ori_p_1) / np.linalg.norm(temp_ori_p - ori_p_1) * scale#FIXME
    else:
        ori_p_2 = lastPoints[0]
        line_length = np.linalg.norm(ori_p_2 - ori_p_1, ord=2)
    temp_ori_p = vanishing_lines[end,1]
    ori_p_3 = ori_p_1 + sec_line_offset * (temp_ori_p - ori_p_1) / np.linalg.norm(temp_ori_p - ori_p_1)
    ori_p_4 = ori_p_1 + (sec_line_offset + line_length*scale2) * (temp_ori_p - ori_p_1) / np.linalg.norm(temp_ori_p - ori_p_1)
    return np.array([ori_p_1,ori_p_2,ori_p_3,ori_p_4])



def overwrite(bg: np.ndarray, obj: np.ndarray, bg_vp, obj_vp, bg_h, obj_h, background_origin, object_origin) -> np.ndarray:
    P_bg = calibration(background_origin, bg_vp, bg_h)
    P_obj = calibration(object_origin, obj_vp, obj_h)

    new_image = bg.copy()
    alpha_mask = mask(obj)
    W, H = new_image.shape[1], new_image.shape[0]
    objH, objW = obj.shape[0], obj.shape[1]
    xs, ys = np.meshgrid(range(W), range(H))
                
    # Divide the vanishing lines into 6 groups
    vanishing_lines = np.zeros((6,2,2),dtype=int)
    vanishing_lines[:,0] = np.array([int(object_origin[1]), int(object_origin[0])])[None]
    vanishing_lines[:3,1] = np.array([
        [int(obj_vp.x[1]), int(obj_vp.x[0])],
        [int(obj_vp.y[1]), int(obj_vp.y[0])],
        [int(obj_vp.z[1]), int(obj_vp.z[0])],
    ])
    vanishing_lines[3:,1] = 2*vanishing_lines[:3,0] - vanishing_lines[:3,1]

    bg_vanishing_lines = np.zeros((6,2,2),dtype=int)
    bg_vanishing_lines[:,0] = np.array([int(background_origin[1]), int(background_origin[0])])[None]
    bg_vanishing_lines[:3,1] = np.array([
        [int(bg_vp.x[1]), int(bg_vp.x[0])],
        [int(bg_vp.y[1]), int(bg_vp.y[0])],
        [int(bg_vp.z[1]), int(bg_vp.z[0])],
    ])
    bg_vanishing_lines[3:,1] = 2*bg_vanishing_lines[:3,0] - bg_vanishing_lines[:3,1]

    # Make masks for each region
    vectors = vanishing_lines[:, 1] - vanishing_lines[:, 0]
    absolute_angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    absolute_angles -= absolute_angles[0]
    absolute_angles = (absolute_angles + 720) % 360
    line_order = np.argsort(absolute_angles)

    y, x = np.mgrid[:obj.shape[0],:obj.shape[1]]
    masks = []
    for i in range(6):
        a = vectors[line_order[i]]
        b = vectors[line_order[(i+1)%6]]
        target = np.array([y-int(object_origin[1]),x-int(object_origin[0])]) 
        mas = ((np.sign(mycross(a,target)) * np.sign(mycross(a,b))) >=0) * ((np.sign(mycross(b,target)) * np.sign(mycross(b,a))) >= 0)
        masks.append(mas)
            
    # Draw masks on the object
    # for i, color in enumerate([[0, 0, 255, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 0, 255, 255],[0, 255, 0, 255],[255, 0, 0, 255]]):
    #     color_mask = np.zeros((*masks[i].shape, 4), dtype=np.float32)
    #     color_mask[masks[i]] = color#[0, 0, 255, 255]
    #     new_image_float = obj.astype(np.float32)
    #     new_image_float = cv2.addWeighted(new_image_float, 1.0, color_mask, 0.9, 0)
    #     obj = new_image_float.astype(np.uint8)
        
    # Calculate mapped vanishing lines
    new_image = new_image.copy()
    mapped_vanishing_lines = np.zeros((6,2,2),dtype=int)
    mapped_to_bg = P_obj.P @ np.linalg.pinv(P_bg.P)
    mapped_vanishing_lines[:,0] = multiple_euclidian((np.linalg.pinv(mapped_to_bg) @ multiple_homogeneous(np.flip(vanishing_lines[:,0],axis=-1)).T).T)
    mapped_vanishing_lines[:,1] = multiple_euclidian((np.linalg.pinv(mapped_to_bg) @ multiple_homogeneous(np.flip(vanishing_lines[:,1],axis=-1)).T).T)
    mapped_vanishing_lines = np.flip(mapped_vanishing_lines,axis=-1)

    # Get 4 points for perspective transform
    ori_p = getSupportPoints(mapped_vanishing_lines,0,5,scale=1.0)
    bg_p = getSupportPoints(bg_vanishing_lines,0,5)

    # Calculate perspective matrix
    perspective_matrix = myPerspective(np.flip(ori_p,axis=-1),np.flip(bg_p,axis=-1))

    # Generate mapped coordinates
    xs, ys = np.meshgrid(range(W), range(H))
    mapped_coordinates = np.array((xs, ys)).transpose((1, 2, 0))
    mapped_coordinates = mapping(mapped_coordinates.reshape(-1, 2), P_bg, P_obj, np.linalg.inv(perspective_matrix)).reshape(H, W, 2)

    # Overwrite the image
    y = mapped_coordinates[..., 1].astype(int)
    x = mapped_coordinates[..., 0].astype(int)
    valid_mask = (y >= 0) & (y < objH) & (x >= 0) & (x < objW)
    y = y[valid_mask]
    x = x[valid_mask]
    combined_mask = valid_mask.copy()
    combined_mask[valid_mask] *= (alpha_mask[y, x] == 1) * (masks[0][y, x] == 1) 
    new_image[combined_mask] = obj[y[combined_mask[valid_mask]], x[combined_mask[valid_mask]], :-1]
    
    # Calculate the perspective matrix for the next step
    mapped_ori_p_first = np.flip(multiple_euclidian((perspective_matrix @ multiple_homogeneous(np.flip(ori_p[3:4],axis=-1)).T).T),axis=-1)
    mapped_vanishing_lines[:,0] = multiple_euclidian((perspective_matrix @ multiple_homogeneous(np.flip(mapped_vanishing_lines[:,0],axis=-1)).T).T)
    mapped_vanishing_lines[:,1] = multiple_euclidian((perspective_matrix @ multiple_homogeneous(np.flip(mapped_vanishing_lines[:,1],axis=-1)).T).T)
    mapped_vanishing_lines = np.flip(mapped_vanishing_lines,axis=-1)

    ori_p = getSupportPoints(mapped_vanishing_lines,5,1,scale=1.06)
    bg_p = getSupportPoints(bg_vanishing_lines,5,1)

    perspective_matrix = myPerspective(np.flip(ori_p,axis=-1),np.flip(bg_p,axis=-1)) @ perspective_matrix
    mapped_coordinates = np.array((xs, ys)).transpose((1, 2, 0))
    mapped_coordinates = mapping(mapped_coordinates.reshape(-1, 2), P_bg, P_obj, np.linalg.inv(perspective_matrix)).reshape(H, W, 2)

    y = mapped_coordinates[..., 1].astype(int)
    x = mapped_coordinates[..., 0].astype(int)
    valid_mask = (y >= 0) & (y < objH) & (x >= 0) & (x < objW)
    y = y[valid_mask]
    x = x[valid_mask]
    combined_mask = valid_mask.copy()
    combined_mask[valid_mask] &= (alpha_mask[y, x] == 1) * (masks[1][y, x] == 1)
    new_image[combined_mask] = obj[y[combined_mask[valid_mask]], x[combined_mask[valid_mask]], :-1]
    
    cv2.line(new_image, [int(background_origin[0]), int(background_origin[1])], [int(bg_vp.z[0]), int(bg_vp.z[1])], (0, 0, 255), 3)
    cv2.line(new_image, [int(background_origin[0]), int(background_origin[1])], [int(bg_vp.x[0]), int(bg_vp.x[1])], (255, 0, 0), 3)
    cv2.line(new_image, [int(background_origin[0]), int(background_origin[1])], [int(bg_vp.y[0]), int(bg_vp.y[1])], (0, 256, 0), 3)
    return new_image