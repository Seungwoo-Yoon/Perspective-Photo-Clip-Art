import cv2
import numpy as np
from vanishing_point import *
from height import *
from combine import *
from coordinate import *

from util.camera_pose_visualizer import CameraPoseVisualizer

object_img = np.array(cv2.imread('object.png', cv2.IMREAD_UNCHANGED))
background_img = np.array(cv2.imread('background.jpg'))

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

cv2.imshow('img', background_img)
cv2.waitKey()

visualizer = CameraPoseVisualizer([-25, 25], [-25, 25], [0, 50])
visualizer.extrinsic2pyramid(np.linalg.inv(P_bg.K) @ P_bg.P, 'r', 10)
visualizer.extrinsic2pyramid(np.linalg.inv(P_obj.K) @ P_obj.P, 'b', 10)
# visualizer.show()

new_img = overwrite(background_img, object_img, P_bg, P_obj)

cv2.imshow('img', new_img)
cv2.imwrite('result.jpg', new_img)
cv2.waitKey()