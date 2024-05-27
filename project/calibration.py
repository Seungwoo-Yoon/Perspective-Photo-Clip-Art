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
    Vx = vanshing.x # vanshing
    Vy = vanshing.y # vanshing
    Vz = vanshing.z # vanshing
    
    L = height_info.length
    
    # compute K
    A = np.array([[Vx[0]*Vy[0]+Vx[1]*Vy[1], Vx[0]+Vy[0], Vx[1]+Vy[1],1],
                  [Vy[0]*Vz[0]+Vy[1]*Vz[1], Vy[0]+Vz[0], Vy[1]+Vz[1],1],
                  [Vz[0]*Vx[0]+Vz[1]*Vx[1], Vz[0]+Vx[0], Vz[1]+Vx[1],1]])
    _, _, Vt = np.linalg.svd(A)
    w = Vt.T[:,-1]
    W = np.array([[w[0],0,w[1]],
                  [0,w[0],w[2]],
                  [w[1],w[2],w[3]]])
    K = np.linalg.inv(np.linalg.cholesky(W)).T
    
    # compute R
    Kinv = np.linalg.inv(K)
    K_vx = np.dot(Kinv,Vx).T
    K_vy = np.dot(Kinv,Vy).T
    K_vz = np.dot(Kinv,Vz).T
    QR = np.stack([K_vx,K_vy,K_vz],axis=1)
    R,_ = np.linalg.qr(QR)
    
    # compute t
    Kinv_pz = np.dot(Kinv,np.array([[0,0,L,1]]))
    omega2 = 1.0 #FIXME
    t = omega2 * Kinv_pz - L * R[:,-1]
    
    # return the camera parameter calibrated from 5 informations
    return CameraParameter(K,R,t)
    # raise NotImplementedError()