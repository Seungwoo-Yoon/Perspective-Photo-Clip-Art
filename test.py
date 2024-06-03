import cv2
import numpy as np
from vanishing_point import *
from height import *
from combine import *
from coordinate import *

from util.camera_pose_visualizer import CameraPoseVisualizer
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates
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

object_img = np.array(cv2.imread('object.png', cv2.IMREAD_UNCHANGED))
background_img = np.array(cv2.imread('background.JPG'))

print(background_img.shape)
print(object_img.shape)

background_x = np.array([
    [[158, 342], [389, 292]],
    [[183, 482], [340, 413]],
    [[131, 271], [314, 250]],
    [[156, 133], [180, 134]],
    [[149, 73], [175, 80]],
    [[94, 75], [128, 78]],
    [[101, 117], [135, 121]],
    [[10, 451], [155, 406]],
    [[174, 401], [364, 344]],
    [[181, 442], [352, 379]]
]) / 500 * 3024
background_y = np.array([
    [[131, 272], [160, 343]],
    [[316, 251], [387, 289]],
    [[345, 421], [458, 548]],
    [[157, 409], [179, 480]]
]) / 500 * 3024
background_z = np.array([
    [[161, 346], [184, 481]],
    [[387, 295], [339, 412]],
    [[357, 41], [314, 239]],
    [[24, 75], [126, 404]],
    [[470, 565], [498, 546]],
    [[132, 277], [156, 408]]
]) / 500 * 3024
background_height = np.array(
    [[161, 346], [183, 484]]
) / 500 * 3024

# background_height = np.array(
#     [[183, 484], [161, 346]]
# ) / 500 * 3024
background_height_value = -30

# background_origin = np.array([159, 340]) / 500 * 3024
background_origin = np.array([183, 318]) / 500 * 3024

    
object_x = np.array([
    [[207, 421], [409, 339]],
    [[197, 170], [488, 117]],
    [[139, 106], [406, 73]]
]) / 500 * 433
object_y = np.array([
    [[140, 109], [198, 170]],
    [[405, 74], [490, 116]],
    [[166, 350], [208, 423]]
]) / 500 * 433
object_z = np.array([
    [[199, 171], [207, 422]],
    [[411, 339], [489, 119]],
    [[139, 108], [167, 353]]
]) / 500 * 433
object_height = np.array([
    [208, 423], [199, 173]
]) / 500 * 433
object_height_value = 15
object_origin = np.array([207, 422]) / 500 * 433

# cv2.imshow('img', background_img)
# cv2.waitKey()
# cv2.imshow('img', object_img)
# cv2.waitKey()

# exit()

bg_vp = VanishingPoint(background_x, background_y, background_z)
# bg_vp.rotate(-np.pi / 3)
obj_vp = VanishingPoint(object_x, object_y, object_z)

bg_h = HeightInformation(background_height[0], background_height[1], background_height_value)
obj_h = HeightInformation(object_height[0], object_height[1], object_height_value)

P_bg = calibration(background_origin, bg_vp, bg_h)
P_obj = calibration(object_origin, obj_vp, obj_h)

pz, L = height_projection(object_origin, obj_vp, obj_h)
background_img = cv2.line(background_img, [int(background_origin[0]), int(background_origin[1])], [int(bg_vp.z[0]), int(bg_vp.z[1])], (256, 0, 0), 3)
background_img = cv2.line(background_img, [int(background_origin[0]), int(background_origin[1])], [int(bg_vp.x[0]), int(bg_vp.x[1])], (0, 0, 256), 3)
background_img = cv2.line(background_img, [int(background_origin[0]), int(background_origin[1])], [int(bg_vp.y[0]), int(bg_vp.y[1])], (0, 256, 0), 3)

object_img = cv2.line(object_img, [int(object_origin[0]), int(object_origin[1])], [int(obj_vp.z[0]), int(obj_vp.z[1])], (256, 0, 0), 3)
object_img = cv2.line(object_img, [int(object_origin[0]), int(object_origin[1])], [int(obj_vp.x[0]), int(obj_vp.x[1])], (0, 0, 256), 3)
object_img = cv2.line(object_img, [int(object_origin[0]), int(object_origin[1])], [int(obj_vp.y[0]), int(obj_vp.y[1])], (0, 256, 0), 3)

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

vectors = vanishing_lines[:, 1] - vanishing_lines[:, 0]
absolute_angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
absolute_angles -= absolute_angles[0]
absolute_angles = (absolute_angles + 720) % 360
line_order = np.argsort(absolute_angles)

y, x = np.mgrid[:object_img.shape[0],:object_img.shape[1]]
masks = []
for i in range(6):
    a = vectors[line_order[i]]
    b = vectors[line_order[(i+1)%6]]
    target = y-int(object_origin[1]),x-int(object_origin[0])
    mas = (mycross(a,target) * mycross(a,b) >=0) *\
        (mycross(b,target) * mycross(b,a) >= 0)
    masks.append(mas)
    
# # For test combine
# new_image = background_img.copy()
# alpha_mask = mask(object_img)
# W, H = background_img.shape[1], background_img.shape[0]
# objH, objW = object_img.shape[0], object_img.shape[1]
# xs, ys = np.meshgrid(range(W), range(H))
# mapped_coordinates = np.array((xs, ys)).transpose((1, 2, 0))
# mapped_coordinates = mapping(mapped_coordinates.reshape(-1, 2), P_bg, P_obj).reshape(H, W, 2)
# for x in tqdm(range(W)):
#     for y in range(H):
#         mapped_coordinate = mapped_coordinates[y, x]
#         if 0 <= int(mapped_coordinate[1]) < objH and 0 <= int(mapped_coordinate[0]) < objW:
#             if True: # alpha_mask[int(mapped_coordinate[1]), int(mapped_coordinate[0])] == 1:
#                 new_image[y, x] = object_img[int(mapped_coordinate[1]), int(mapped_coordinate[0]), :-1]
                
# Calculate mapped vanishing lines
new_image = background_img.copy()

mapped_vanishing_lines = np.zeros((6,2,2),dtype=int)
mapped_to_bg = P_obj.P @ np.linalg.pinv(P_bg.P)
mapped_vanishing_lines[:,0] = multiple_euclidian((np.linalg.pinv(mapped_to_bg) @ multiple_homogeneous(np.flip(vanishing_lines[:,0],axis=-1)).T).T)
mapped_vanishing_lines[:,1] = multiple_euclidian((np.linalg.pinv(mapped_to_bg) @ multiple_homogeneous(np.flip(vanishing_lines[:,1],axis=-1)).T).T)
mapped_vanishing_lines = np.flip(mapped_vanishing_lines,axis=-1)
if False: #test mapped_vanishing_lines
    cv2.circle(new_image, (int(mapped_vanishing_lines[0,0][1]), int(mapped_vanishing_lines[0,0][0])), 10, (0, 0, 255), -1)
    cv2.circle(new_image, (int(mapped_vanishing_lines[0,1][1]), int(mapped_vanishing_lines[0,1][0])), 10, (0, 0, 255), -1)
    cv2.circle(new_image, (int(mapped_vanishing_lines[1,1][1]), int(mapped_vanishing_lines[1,1][0])), 10, (0, 255, 0), -1)
    cv2.circle(new_image, (int(mapped_vanishing_lines[2,1][1]), int(mapped_vanishing_lines[2,1][0])), 10, (255, 0, 0), -1)
    cv2.circle(new_image, (int(mapped_vanishing_lines[3,1][1]), int(mapped_vanishing_lines[3,1][0])), 20, (0, 0, 255), -1)
    cv2.circle(new_image, (int(mapped_vanishing_lines[4,1][1]), int(mapped_vanishing_lines[4,1][0])), 20, (0, 255, 0), -1)
    cv2.circle(new_image, (int(mapped_vanishing_lines[5,1][1]), int(mapped_vanishing_lines[5,1][0])), 20, (255, 0, 0), -1)

# Get points for perspective transform
# for x to y (ex)
ori_p = getSupportPoints(mapped_vanishing_lines,0,5)
bg_p = getSupportPoints(bg_vanishing_lines,0,5)
print(np.linalg.norm(ori_p[0]-ori_p[1],axis=-1))
print(np.linalg.norm(ori_p[2]-ori_p[3],axis=-1))
print("--")
print(np.linalg.norm(bg_p[0]-bg_p[1],axis=-1))
print(np.linalg.norm(bg_p[2]-bg_p[3],axis=-1))

# Calculate perspective matrix
perspective_matrix = myPerspective(np.flip(ori_p,axis=-1),np.flip(bg_p,axis=-1))
alpha_mask = mask(object_img)
W, H = background_img.shape[1], background_img.shape[0]
objH, objW = object_img.shape[0], object_img.shape[1]

xs, ys = np.meshgrid(range(W), range(H))
mapped_coordinates = np.array((xs, ys)).transpose((1, 2, 0))
mapped_coordinates = mapping(mapped_coordinates.reshape(-1, 2), P_bg, P_obj, np.linalg.inv(perspective_matrix)).reshape(H, W, 2)
for x in tqdm(range(W)):
    for y in range(H):
        mapped_coordinate = mapped_coordinates[y, x]
        if 0 <= int(mapped_coordinate[1]) < objH and 0 <= int(mapped_coordinate[0]) < objW:
            if True: # alpha_mask[int(mapped_coordinate[1]), int(mapped_coordinate[0])] == 1:
                new_image[y, x] = object_img[int(mapped_coordinate[1]), int(mapped_coordinate[0]), :-1]


# xs, ys = np.meshgrid(range(objW), range(objH))
# mapped_coordinates = np.array((xs, ys)).transpose((1, 2, 0))
# mapped_coordinates = mapping(mapped_coordinates.reshape(-1, 2), P_obj, P_bg, perspective_matrix).reshape(objH, objW, 2)
# # mapped_coordinates = mapped_coordinates.transpose((2, 0, 1))
# count_image = np.zeros_like(new_image)
# for x in tqdm(range(objW)):
#     for y in range(objH):
#         mapped_coordinate = mapped_coordinates[y, x]
#         if 0 <= int(mapped_coordinate[1]) < H and 0 <= int(mapped_coordinate[0]) < W:
#             if True:#masks[0][y, x] == 1 and alpha_mask[y,x] == 1:
#                 # new_image[int(mapped_coordinate[1]), int(mapped_coordinate[0])] = object_img[y, x, :-1]# * (1 - masks[0][y, x])
#                 new_image[int(mapped_coordinate[1]), int(mapped_coordinate[0])] += interpolate_pixel(object_img, y, x)[:3]
#                 # new_image = map_coordinates(object_img, mapped_coordinates, order=1, mode='nearest', cval=0.0)[0]

#==============================================================================
# mapped_ori_p_first = np.flip(multiple_euclidian((perspective_matrix @ multiple_homogeneous(np.flip(ori_p[3:4],axis=-1)).T).T),axis=-1)
# mapped_vanishing_lines[:,0] = multiple_euclidian((perspective_matrix @ multiple_homogeneous(np.flip(mapped_vanishing_lines[:,0],axis=-1)).T).T)
# mapped_vanishing_lines[:,1] = multiple_euclidian((perspective_matrix @ multiple_homogeneous(np.flip(mapped_vanishing_lines[:,1],axis=-1)).T).T)
# mapped_vanishing_lines = np.flip(mapped_vanishing_lines,axis=-1)

# if True: #test support points for perspective transform
#     # cv2.circle(new_image, (int(ori_p[0,1]), int(ori_p[0,0])), 10, (255, 255, 0), -1)
#     # cv2.circle(new_image, (int(ori_p[1,1]), int(ori_p[1,0])), 10, (255, 255, 0), -1) #cyan
#     cv2.circle(new_image, (int(ori_p[2,1]), int(ori_p[2,0])), 30, (255, 0, 255), -1) #magenta
#     cv2.circle(new_image, (int(ori_p[3,1]), int(ori_p[3,0])), 30, (255, 0, 255), -1)
    
#     # cv2.circle(new_image, (int(bg_p[0,1]), int(bg_p[0,0])), 10, (255, 0, 255), -1)
#     # cv2.circle(new_image, (int(bg_p[1,1]), int(bg_p[1,0])), 10, (255, 0, 255), -1)
#     cv2.circle(new_image, (int(bg_p[2,1]), int(bg_p[2,0])), 30, (255, 255, 255), -1)
#     cv2.circle(new_image, (int(bg_p[3,1]), int(bg_p[3,0])), 30, (255, 255, 255), -1)

# ori_p = getSupportPoints(mapped_vanishing_lines,5,1,scale=0.93)
# # bg_p = getSupportPoints(bg_vanishing_lines,5,1)#,mapped_ori_p_first,scale=0.93) #FIXME
# bg_p = getSupportPoints(bg_vanishing_lines,5,1)
# # bg_p = getSupportPoints(bg_vanishing_lines,5,1,scale=0.93,scale2=1.5)
# print(np.linalg.norm(ori_p[0]-ori_p[1],axis=-1))
# print(np.linalg.norm(ori_p[2]-ori_p[3],axis=-1))
# print("--")
# print(np.linalg.norm(bg_p[0]-bg_p[1],axis=-1))
# print(np.linalg.norm(bg_p[2]-bg_p[3],axis=-1))
# # bg_p = getSupportPoints(bg_vanishing_lines,5,1,scale=0.93)
    
# perspective_matrix = myPerspective(np.flip(ori_p,axis=-1),np.flip(bg_p,axis=-1)) @ perspective_matrix
# mapped_coordinates = np.array((xs, ys)).transpose((1, 2, 0))
# mapped_coordinates = mapping(mapped_coordinates.reshape(-1, 2), P_obj, P_bg, perspective_matrix).reshape(objH, objW, 2)
    
# for x in tqdm(range(objW)):
#     for y in range(objH):
#         mapped_coordinate = mapped_coordinates[y, x]
#         if 0 <= int(mapped_coordinate[1]) < H and 0 <= int(mapped_coordinate[0]) < W:
#             if True: # masks[1][y, x] == 1 and alpha_mask[y,x] == 1:
#                 new_image[int(mapped_coordinate[1]), int(mapped_coordinate[0])] = object_img[y, x, :-1]# * (1 - masks[0][y, x])
#                 #interpolate with neighbor pixels of the background image
        
# if True: #test support points for perspective transform
#     cv2.circle(new_image, (int(ori_p[0,1]), int(ori_p[0,0])), 20, (255, 0, 0), -1)
#     cv2.circle(new_image, (int(ori_p[1,1]), int(ori_p[1,0])), 20, (255, 0, 0), -1)
#     cv2.circle(new_image, (int(ori_p[2,1]), int(ori_p[2,0])), 20, (0, 255, 255), -1)
#     cv2.circle(new_image, (int(ori_p[3,1]), int(ori_p[3,0])), 20, (0, 255, 255), -1)
    
#     cv2.circle(new_image, (int(bg_p[0,1]), int(bg_p[0,0])), 10, (0, 0, 255), -1)
#     cv2.circle(new_image, (int(bg_p[1,1]), int(bg_p[1,0])), 10, (0, 0, 255), -1)
#     cv2.circle(new_image, (int(bg_p[2,1]), int(bg_p[2,0])), 10, (0, 255, 0), -1)
#     cv2.circle(new_image, (int(bg_p[3,1]), int(bg_p[3,0])), 10, (0, 255, 0), -1)
#==============================================================================

# Calculate mapped masks

# Get perspective transform matrix
# perspective = np.array()

scaled_image = cv2.resize(new_image, (new_image.shape[1]//4, new_image.shape[0]//4))
cv2.imshow('object_img', scaled_image)
cv2.waitKey()
cv2.destroyAllWindows()
breakpoint()

# # check masks
# for i, color in enumerate([[0, 0, 255, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 0, 255, 255],[0, 255, 0, 255],[255, 0, 0, 255]]):
#     color_mask = np.zeros((*masks[i].shape, 4), dtype=np.float32)
#     color_mask[masks[i]] = color#[0, 0, 255, 255]
#     new_image_float = new_image.astype(np.float32)
#     new_image_float = cv2.addWeighted(new_image_float, 1.0, color_mask[0 <= int(mapped_coordinate[1]) < objH, 0 <= int(mapped_coordinate[0]) < objW], 0.4, 0)
#     new_image = new_image_float.astype(np.uint8)
# # Display the result
# scaled_image = cv2.resize(new_image, (new_image.shape[1]//4, new_image.shape[0]//4))
# cv2.imshow('object_img', scaled_image)
# cv2.waitKey()
# cv2.destroyAllWindows()
# breakpoint()

# # check masks
# for i, color in enumerate([[0, 0, 255, 255],[0, 255, 0, 255],[255, 0, 0, 255],[0, 0, 255, 255],[0, 255, 0, 255],[255, 0, 0, 255]]):
#     color_mask = np.zeros((*masks[i].shape, 4), dtype=np.float32)
#     color_mask[masks[i]] = color#[0, 0, 255, 255]
#     object_img_float = object_img.astype(np.float32)
#     object_img_float = cv2.addWeighted(object_img_float, 1.0, color_mask, 0.4, 0)
#     object_img = object_img_float.astype(np.uint8)


# # Rotate the object image to vanishing line from background image
# bg_vanishing_lines = np.zeros((6,2,2),dtype=int)
# # target_angle = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
# target_angle = absolute_angles[5]
# print(target_angle)
# target_angle = 90
# rotation_matrix = cv2.getRotationMatrix2D((object_img.shape[1]//2, object_img.shape[0]//2), target_angle, 1)
# object_img = cv2.warpAffine(object_img, rotation_matrix, (object_img.shape[1], object_img.shape[0]))

# Display the result
# cv2.imshow('object_img', object_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# breakpoint()

visualizer = CameraPoseVisualizer([-25, 25], [-25, 25], [0, 50])
visualizer.extrinsic2pyramid(np.linalg.inv(P_bg.K) @ P_bg.P, 'r', 10)
visualizer.extrinsic2pyramid(np.linalg.inv(P_obj.K) @ P_obj.P, 'b', 10)
# visualizer.show()

new_img = overwrite(background_img, object_img, P_bg, P_obj)
print("done")
# cv2.imshow('img', new_img)
cv2.imwrite('result.jpg', new_img)
cv2.imwrite('object_img.jpg', object_img)
cv2.waitKey()

