import numpy as np
from vanishing_point import *
from height import *
from coordinate import *

class CameraParameter:
    def __init__(self, K, R, t) -> None:
        self.P = K @ np.concatenate((R, t.reshape(3, 1)), axis=1)
        self.K = K
        self.R = R
        self.t = t

def rotate(P: CameraParameter, theta):
    K = P.K
    R = P.R
    t = P.t
    # C = -R.T @ t
    t = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) @ t
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) @ R
    # t = -R @ C
    return CameraParameter(K, R, t)

def calibration(origin: np.ndarray, vanishing: VanishingPoint, height_info: HeightInformation) \
    -> CameraParameter:
    Vx = vanishing.x
    Vy = vanishing.y
    Vz = vanishing.z
    pz, L = height_projection(origin, vanishing, height_info)
    
    # compute K

    A = np.array([[Vx[0]*Vy[0]+Vx[1]*Vy[1], Vx[0]+Vy[0], Vx[1]+Vy[1], 1],
                  [Vy[0]*Vz[0]+Vy[1]*Vz[1], Vy[0]+Vz[0], Vy[1]+Vz[1] ,1],
                  [Vz[0]*Vx[0]+Vz[1]*Vx[1], Vz[0]+Vx[0], Vz[1]+Vx[1], 1]])
    _, _, Vt = np.linalg.svd(A)
    w = Vt[-1]
    W = np.array([[w[0],0,w[1]],
                  [0,w[0],w[2]],
                  [w[1],w[2],w[3]]])
    
    W = W / np.linalg.norm(W)

    # D, V = np.linalg.eig(np.linalg.inv(W))
    # # D = np.max((np.zeros_like(D), D), axis=0)
    # D = np.diag(D)
    # W = V @ D @ np.linalg.inv(V)

    K = np.linalg.inv(np.linalg.cholesky(W)).T
    K = K / K[2, 2]
    
    # compute R
    Kinv = np.linalg.inv(K)
    K_vx = np.dot(Kinv, homogeneous(Vx)).T
    K_vy = np.dot(Kinv, homogeneous(Vy)).T
    K_vz = np.dot(Kinv, homogeneous(Vz)).T
    QR = np.stack([K_vx, K_vy, K_vz], axis=1)
    R, _ = np.linalg.qr(QR)

    R_ret = None
    t = None

    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            R_new = np.empty_like(R)
            
            R_new[:, 0] = R[:, 0] * i
            R_new[:, 1] = R[:, 1] * j
            R_new[:, 2] = np.cross(R_new[:, 0], R_new[:, 1])

            # compute t
            K_p0 = Kinv @ homogeneous(origin)
            K_pz = Kinv @ homogeneous(pz)

            M = L * R_new[:, 2]
            A = np.array([
                [K_p0[2], 0, -K_p0[0]],
                [0, K_p0[2], -K_p0[1]],
                [K_pz[2], 0, -K_pz[0]],
                [0, K_pz[2], -K_pz[1]]
            ])
            b = np.array([0, 0, - M[0] * K_pz[2] + M[2] * K_pz[0], - M[1] * K_pz[2] + M[2] * K_pz[1]])
            t_new = np.linalg.pinv(A) @ b

            projected_origin = (K @ np.concatenate((R_new, t_new.reshape(3, 1)), axis=1) @ np.array([0, 0, 0, 1]))

            if projected_origin[-1] > 0:
                R_ret = R_new
                t = t_new

    
    # return the camera parameter calibrated from 5 informations
    return CameraParameter(K,R_ret,t)