import numpy as np

def euclidian(a: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return a[:-1] / (a[-1] + eps)

def homogeneous(a: np.ndarray) -> np.ndarray:
    return np.concatenate((a, np.array([1,])))
