import numpy as np

def euclidian(a: np.ndarray):
    eps = 1e-8
    return a[:-1] / (a[-1] + eps)

def multiple_euclidian(a):
    eps = 1e-8
    return a[:, :2] / (a[:, 2] + eps)[:, None]

def homogeneous(a: np.ndarray):
    return np.concatenate((a, np.array([1,])))

def multiple_homogeneous(a):
    N = a.shape[0]
    return np.concatenate((a, np.ones((N, 1))), axis=-1)