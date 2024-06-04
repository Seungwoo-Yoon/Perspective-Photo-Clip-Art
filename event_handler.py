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

    background_x = np.array(js.backgroundX) / js.backgroundCanvas.width * W
    background_y = np.array(js.backgroundY) / js.backgroundCanvas.width * W
    background_z = np.array(js.backgroundZ) / js.backgroundCanvas.width * W
    background_height = np.array(js.backgroundHeight) / js.backgroundCanvas.width * W
    background_height_value = js.backgroundHeightValue
    background_origin = np.array(js.backgroundOrigin) / js.backgroundCanvas.width * W

    
    object_img = js.objectImageBase64
    with request.urlopen(object_img) as response:
        object_img = response.read()
    object_img = np.array(Image.open(io.BytesIO(object_img)))
    
    W = object_img.shape[1]

    object_x = np.array(js.objectX) / js.objectCanvas.width  * W
    object_y = np.array(js.objectY) / js.objectCanvas.width  * W
    object_z = np.array(js.objectZ) / js.objectCanvas.width  * W
    object_height = np.array(js.objectHeight) / js.objectCanvas.width  * W
    object_height_value = js.objectHeightValue
    object_origin = np.array(js.objectOrigin) / js.objectCanvas.width * W

    rotation = float(js.rotation) * np.pi / 180

    bg_vp = VanishingPoint(background_x, background_y, background_z)
    bg_vp.rotate(rotation)
    obj_vp = VanishingPoint(object_x, object_y, object_z)
    
    bg_h = HeightInformation(background_height[0], background_height[1], background_height_value)
    obj_h = HeightInformation(object_height[0], object_height[1], object_height_value)

    new_img = Image.fromarray(overwrite(background_img, object_img, bg_vp, obj_vp, bg_h, obj_h, background_origin, object_origin))
    new_img.resize((js.backgroundCanvas.width, int(js.backgroundCanvas.width * background_img.shape[0] / background_img.shape[1])))

    buffer = io.BytesIO()
    new_img.save(buffer, format='png')
    js.newImageBase64 = "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()))[2:-1]
    js.drawResult(js.newImageBase64)



proxy = create_proxy(event_handler)
python_event_object.addEventListener("python", proxy)