"""Cok sahneli PolSAR yukleyici (T3 6-bilesen kompleks gosterim).

Ober T6 olarak dagitilmis; resiprok monostatik hedef icin gecerli olan
sol-ust 3x3 blok (T11,T22,T33,T12,T13,T23) kullanilir -- ayni Pauli tabani,
dolayisiyla U(theta) donme grubu aynen gecerli.
"""
import numpy as np, scipy.io as sio, os
from scipy.ndimage import distance_transform_edt

ROOT = "c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/PolSAR Data"
SCENES = {
  "flevoland": dict(path=f"{ROOT}/Flevoland/T3",  gt=f"{ROOT}/Flevoland/Flevoland_gt.mat",
                    R=750,  C=1024, ncl=15),
  "sanfran":   dict(path=f"{ROOT}/san_francisco/T3", gt=f"{ROOT}/san_francisco/SanFrancisco_gt.mat",
                    R=900,  C=1024, ncl=5),
  "ober":      dict(path=f"{ROOT}/Oberpfaffenhofen/ESAR_Oberpfaffenhofen_T6",
                    gt=f"{ROOT}/Oberpfaffenhofen/Oberpfaffenhofen_gt.mat",
                    R=1300, C=1200, ncl=3),
}

AIR = {"air_gz": "gz", "air_sh": "sh", "air_bj": "bj"}

def load_scene(name):
    if name in AIR:
        # AIR-PolSAR-Seg-2.0: built from GF-3 SLC by air2.py, cached by air_cache.py.
        # The archive is L1A and ships no calibration constants, so this scene is
        # used as a large unsaturated benchmark and as a stress test of the group
        # action we apply ourselves -- not for physical orientation claims.
        import air_cache
        d = np.load(air_cache.path(AIR[name]))
        X, gt = d["X"], d["gt"].astype(np.int64)
        return X, gt, int(gt.max())
    cfg = SCENES[name]; R,C,p = cfg["R"], cfg["C"], cfg["path"]
    rd = lambda n: np.fromfile(os.path.join(p,n), dtype="<f4").reshape(R,C)
    X = np.empty((R,C,6), np.complex64)
    X[...,0]=rd("T11.bin"); X[...,1]=rd("T22.bin"); X[...,2]=rd("T33.bin")
    X[...,3]=rd("T12_real.bin")+1j*rd("T12_imag.bin")
    X[...,4]=rd("T13_real.bin")+1j*rd("T13_imag.bin")
    X[...,5]=rd("T23_real.bin")+1j*rd("T23_imag.bin")
    gt = sio.loadmat(cfg["gt"])["gt"].astype(np.int64)
    assert gt.shape==(R,C), f"{name}: gt {gt.shape} != {(R,C)}"
    return X, gt, cfg["ncl"]

def bdist(gt):
    b=np.zeros(gt.shape,bool)
    b[:-1,:]|=gt[:-1,:]!=gt[1:,:]; b[1:,:]|=gt[:-1,:]!=gt[1:,:]
    b[:,:-1]|=gt[:,:-1]!=gt[:,1:]; b[:,1:]|=gt[:,:-1]!=gt[:,1:]
    return distance_transform_edt(~b)

def to33(X6):
    T=np.empty(X6.shape[:-1]+(3,3),np.complex128)
    T[...,0,0]=X6[...,0]; T[...,1,1]=X6[...,1]; T[...,2,2]=X6[...,2]
    T[...,0,1]=X6[...,3]; T[...,1,0]=np.conj(X6[...,3])
    T[...,0,2]=X6[...,4]; T[...,2,0]=np.conj(X6[...,4])
    T[...,1,2]=X6[...,5]; T[...,2,1]=np.conj(X6[...,5])
    return T
