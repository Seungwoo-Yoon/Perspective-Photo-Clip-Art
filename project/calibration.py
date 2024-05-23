import numpy as np
from vanishing_point import *
from height import *

class CameraParameter:
    def __init__(self, K, R, t) -> None:
        self.P = K @ np.concatenate((R, t), axis=1)
        self.K = K
        self.R = R
        self.t = t


def calibration(origin: np.ndarray, vanshing: VanishingPoint, height_info: HeightInformation) \
    -> CameraParameter:
    # TODO
    # return the camera parameter calibrated from 5 informations
    
    raise NotImplementedError()
