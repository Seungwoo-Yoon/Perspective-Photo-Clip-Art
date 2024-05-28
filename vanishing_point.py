import numpy as np
from coordinate import *

class VanishingPoint:
    def __init__(self, xlist: np.ndarray, ylist: np.ndarray, zlist: np.ndarray) -> None:
        # TODO
        # get x, y, z vanishing point from user input
        # need to specify the input

        # list consists of line segments with endpoints 
        # [[[startx1, starty1], [endx1, endy1]], 
        #  [[startx2, starty2], [endx2, endy2]], ...]

        self.x = self.find_vanishing(xlist)
        self.y = self.find_vanishing(ylist)
        self.z = self.find_vanishing(zlist)

    def find_vanishing(self, arr):
        lines = []
        for line in arr:
            fr = homogeneous(line[0])
            to = homogeneous(line[1])
            lines.append(np.cross(fr, to))

        if len(lines) == 2:
            # If there are only 2 segements, crossing point is deterministically computed
            vanishing = np.cross(lines[0], lines[1])  
        else:
            # Use DLT to find a crossing point which minimizes dot product with all the lines
            lines = np.array(lines)
            _, _, V_T = np.linalg.svd(lines)
            vanishing = V_T[-1]
        
        vanishing = euclidian(vanishing)

        return vanishing

    def rotate(self, theta: float, axis='z') -> None:
        if axis not in ['z']:
            raise ValueError('unexpected axis')
        
        # rotate the vanishing point (proposal page 18)
        x = np.cos(theta) * homogeneous(self.x) - np.sin(theta) * homogeneous(self.y)
        y = np.sin(theta) * homogeneous(self.x) + np.cos(theta) * homogeneous(self.y)
        
        self.x, self.y = euclidian(x), euclidian(y)