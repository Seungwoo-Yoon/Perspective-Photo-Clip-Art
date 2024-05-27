import numpy as np
import cv2
from urllib import request
from PIL import Image
import io

import js
from js import document
from pyodide.ffi import create_proxy


from height import *
from vanishing_point import *
from calibration import *
from combine import *


python_event_object = document.getElementById('python-event')

def event_handler(event):
    background_img = js.backgroundImageBase64
    with request.urlopen(background_img) as response:
        background_img = response.read()
    background_img = np.array(Image.open(io.BytesIO(background_img)))

    background_x = np.array(js.backgroundX)
    background_y = np.array(js.backgroundY)
    background_z = np.array(js.backgroundZ)
    background_height = np.array(js.backgroundHeight)
    background_height_value = js.backgroundHeightValue
    background_origin = np.array(js.backgroundOrigin)

    
    object_img = js.objectImageBase64
    with request.urlopen(object_img) as response:
        object_img = response.read()
    object_img = np.array(Image.open(io.BytesIO(object_img)))
    
    object_x = np.array(js.objectX)
    object_y = np.array(js.objectY)
    object_z = np.array(js.objectZ)
    object_height = np.array(js.objectHeight)
    object_height_value = js.objectHeightValue
    object_origin = np.array(js.objectOrigin)

    rotation = js.rotation

    # bg_vp = VanishingPoint(background_x, background_y, background_z)
    # obj_vp = VanishingPoint(object_x, object_y, object_z)
    
    # bg_h = HeightInformation(background_height[0], background_height[1], background_height_value)
    # obj_h = HeightInformation(object_height[0], object_height[1], object_height_value)

    # P_bg = calibration(background_origin, bg_vp, bg_h)
    # P_obj = calibration(object_origin, obj_vp, obj_h)

    # new_img = overwrite(background_img, object_img, P_bg, P_obj, object_origin)
    cv2.imshow('img', background_img)



proxy = create_proxy(event_handler)
python_event_object.addEventListener("python", proxy)