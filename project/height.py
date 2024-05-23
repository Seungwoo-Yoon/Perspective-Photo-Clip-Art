import numpy as np
from vanishing_point import *
from coordinate import *

class HeightInformation:
    def __init__(self, ground_point: np.ndarray, offset_point: np.ndarray, length: float) -> None:
        self.ground_point = ground_point
        self.offset_point = offset_point
        self.length = length


def height_projection(origin: np.ndarray, vp: VanishingPoint, height_info: HeightInformation) \
    -> tuple (np.ndarray, float):
    # get the height of a point in z-axis
    # return point(2D) and height

    pz = (origin + vp.z) / 2     # arbitrary point in z-axis to get the height

    # calculate necessary values (notation in lecture note 9 page 42, 43)
    b1 = homogeneous(height_info.ground_point)
    t1 = homogeneous(height_info.offset_point)
    b2 = homogeneous(origin)

    u = np.cross(np.cross(b1, b2), np.cross(homogeneous(vp.x), homogeneous(vp.y)))
    t1_tilde = np.cross(np.cross(t1, u), np.cross(homogeneous(vp.z), b2))

    # calculate the height (propsal page 13)
    L = (np.linalg.norm(vp.z - origin) / (np.linalg.norm(euclidian(t1_tilde) - origin) + 1e-8) - 1) * height_info.length

    return pz, L
