import numpy as np
import cv2
from urllib import request
from PIL import Image
import io
import base64

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
    # background_img = np.flip(background_img, axis=1)

    W = background_img.shape[1]

    background_x = np.array(js.backgroundX) / 500 * W
    background_y = np.array(js.backgroundY) / 500 * W
    background_z = np.array(js.backgroundZ) / 500 * W
    background_height = np.array(js.backgroundHeight) / 500 * W
    background_height_value = js.backgroundHeightValue
    background_origin = np.array(js.backgroundOrigin) / 500 * W

    
    object_img = js.objectImageBase64
    with request.urlopen(object_img) as response:
        object_img = response.read()
    object_img = np.array(Image.open(io.BytesIO(object_img)))
    
    W = object_img.shape[1]

    object_x = np.array(js.objectX) / 500 * W
    object_y = np.array(js.objectY) / 500 * W
    object_z = np.array(js.objectZ) / 500 * W
    object_height = np.array(js.objectHeight) / 500 * W
    object_height_value = js.objectHeightValue
    object_origin = np.array(js.objectOrigin) / 500 * W

    rotation = js.rotation * np.pi / 180

    bg_vp = VanishingPoint(background_x, background_y, background_z)
    obj_vp = VanishingPoint(object_x, object_y, object_z)
    
    bg_h = HeightInformation(background_height[0], background_height[1], background_height_value)
    obj_h = HeightInformation(object_height[0], object_height[1], object_height_value)

    P_bg = calibration(background_origin, bg_vp, bg_h)
    P_obj = calibration(object_origin, obj_vp, obj_h)

    new_img = Image.fromarray(overwrite(background_img, object_img, P_bg, P_obj))
    new_img.resize((500, int(500 * background_img.shape[0] / background_img.shape[1])))

    buffer = io.BytesIO()
    new_img.save(buffer, format='png')
    js.newImageBase64 = "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()))[2:-1]
    js.drawResult(js.newImageBase64)



proxy = create_proxy(event_handler)
python_event_object.addEventListener("python", proxy)