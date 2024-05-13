import numpy as np

import js
from js import document
from pyodide.ffi import create_proxy

python_event_object = document.getElementById('python-event')

def event_handler(event):
    background_x = np.array(js.backgroundX)
    background_y = np.array(js.backgroundY)
    background_z = np.array(js.backgroundZ)
    background_height = np.array(js.backgroundHeight)
    background_height_value = js.backgroundHeightValue

proxy = create_proxy(event_handler)
python_event_object.addEventListener("python", proxy)