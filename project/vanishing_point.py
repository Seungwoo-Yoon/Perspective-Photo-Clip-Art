import numpy as np

class VanishingPoint:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        self.x = x 
        self.y = y
        self.z = z

    @staticmethod
    def generate(xlist, ylist, zlist) -> VanishingPoint:
        # TODO
        # get x, y, z vanishing point from user input
        # need to specify the input

        raise NotImplementedError()

    def rotate(self, theta: float, axis='z') -> None:
        if axis not in ['z']:
            raise ValueError('unexpected axis')
        
        # rotate the vanishing point (proposal page 18)
        x = np.cos(theta) * self.x + np.sin(theta) * self.y
        y = -np.sin(theta) * self.x + np.cos(theta) * self.y
        self.x, self.y = x, y