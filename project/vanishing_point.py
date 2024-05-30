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

    def find_vanishing(arr):
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
            print(lines)
            _, _, V_T = np.linalg.svd(lines)
            vanishing = V_T[-1]

        for line in lines:
            print(line, vanishing)
            print(np.dot(line, vanishing))
        print()
        
        vanishing = euclidian(vanishing)

        return vanishing

    def rotate(self, theta: float, axis='z') -> None:
        if axis not in ['z']:
            raise ValueError('unexpected axis')
        
        # rotate the vanishing point (proposal page 18)
        x = np.cos(theta) * self.x + np.sin(theta) * self.y
        y = -np.sin(theta) * self.x + np.cos(theta) * self.y
        self.x, self.y = x, y


if __name__ == "__main__":
    xlist = np.array([[[751, 157], [915, 244]],
                      [[750, 423], [877, 448]],
                      [[879, 625], [1036, 596]]])
    
    ylist = np.array([[[749, 420], [752, 29]],
                      [[856, 589], [855, 259]]])
    
    zlist = np.array([[[750, 419], [563, 449]],
                      [[752, 154], [506, 282]]])
    V = VanishingPoint(xlist, ylist, zlist)
    print(V.x)
    print(V.y)
    print(V.z)