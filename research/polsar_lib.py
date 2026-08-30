"""Ortak kutuphane: veri, polarimetrik donme U(theta), kompleks CNN."""
import numpy as np, scipy.io as sio, torch, torch.nn as nn
from scipy.ndimage import distance_transform_edt

R, C = 750, 1024
DP = "c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/Flevoland"
TP = DP + "/T3"
rd = lambda n: np.fromfile(TP+"/"+n, dtype="<f4").reshape(R,C)

def load6():
    X = np.empty((R,C,6), np.complex64)
    X[...,0]=rd("T11.bin"); X[...,1]=rd("T22.bin"); X[...,2]=rd("T33.bin")
    X[...,3]=rd("T12_real.bin")+1j*rd("T12_imag.bin")
    X[...,4]=rd("T13_real.bin")+1j*rd("T13_imag.bin")
    X[...,5]=rd("T23_real.bin")+1j*rd("T23_imag.bin")
    return X

def load_gt():
    return sio.loadmat(DP+"/Flevoland_gt.mat")["gt"].astype(np.int64)

def bdist(gt):
    b=np.zeros(gt.shape,bool)
    b[:-1,:]|=gt[:-1,:]!=gt[1:,:]; b[1:,:]|=gt[:-1,:]!=gt[1:,:]
    b[:,:-1]|=gt[:,:-1]!=gt[:,1:]; b[:,1:]|=gt[:,:-1]!=gt[:,1:]
    return distance_transform_edt(~b)

# ---------------------------------------------------------------------------
# Polarimetrik donme: bakis ekseni etrafinda theta -> T' = U T U^T
#   U = [[1,0,0],[0,c,s],[0,-s,c]],  c=cos(2t), s=sin(2t)
# 6-bilesen gosteriminde kapali form (turetildi):
#   t11' = t11                                   (degismez)
#   t22' = c^2 t22 + s^2 t33 + 2cs Re(t23)
#   t33' = s^2 t22 + c^2 t33 - 2cs Re(t23)
#   t12' =  c t12 + s t13
#   t13' = -s t12 + c t13
#   t23' = cs(t33-t22) + (c^2-s^2) Re(t23) + i Im(t23)     -> Im(t23) DEGISMEZ
# ---------------------------------------------------------------------------
def rot6_torch(xr, xi, th):
    """xr,xi: (...,6,H,W) gercek/imajiner. th: skaler veya (B,) radyan."""
    if torch.is_tensor(th) and th.ndim == 1:
        th = th.view(-1, *([1]*(xr.ndim-2)))   # (B,) -> (B,1,1) : dilim (B,H,W)
    c, s = torch.cos(2*th), torch.sin(2*th)
    t11r,t22r,t33r = xr[...,0,:,:], xr[...,1,:,:], xr[...,2,:,:]
    t11i,t22i,t33i = xi[...,0,:,:], xi[...,1,:,:], xi[...,2,:,:]
    t12r,t13r,t23r = xr[...,3,:,:], xr[...,4,:,:], xr[...,5,:,:]
    t12i,t13i,t23i = xi[...,3,:,:], xi[...,4,:,:], xi[...,5,:,:]
    c2, s2, cs = c*c, s*s, c*s
    o = torch.empty_like(xr); q = torch.empty_like(xi)
    o[...,0,:,:] = t11r;                       q[...,0,:,:] = t11i
    o[...,1,:,:] = c2*t22r + s2*t33r + 2*cs*t23r
    q[...,1,:,:] = c2*t22i + s2*t33i
    o[...,2,:,:] = s2*t22r + c2*t33r - 2*cs*t23r
    q[...,2,:,:] = s2*t22i + c2*t33i
    o[...,3,:,:] =  c*t12r + s*t13r;           q[...,3,:,:] =  c*t12i + s*t13i
    o[...,4,:,:] = -s*t12r + c*t13r;           q[...,4,:,:] = -s*t12i + c*t13i
    o[...,5,:,:] = cs*(t33r-t22r) + (c2-s2)*t23r
    q[...,5,:,:] = cs*(t33i-t22i) + t23i       # Hermityen veride Im(t23): roll-invariant
    if xr.shape[-3] > 6:                       # ek DEGISMEZ kanallar (or. log-guc) aynen gecer
        o[...,6:,:,:] = xr[...,6:,:,:]
        q[...,6:,:,:] = xi[...,6:,:,:]
    return o, q

def rot6_np(X, th):
    c, s = np.cos(2*th), np.sin(2*th); c2,s2,cs = c*c, s*s, c*s
    Y = np.empty_like(X)
    t11,t22,t33,t12,t13,t23 = [X[...,k] for k in range(6)]
    Y[...,0]=t11
    Y[...,1]=c2*t22 + s2*t33 + 2*cs*t23.real
    Y[...,2]=s2*t22 + c2*t33 - 2*cs*t23.real
    Y[...,3]= c*t12 + s*t13
    Y[...,4]=-s*t12 + c*t13
    Y[...,5]=cs*(t33-t22) + (c2-s2)*t23.real + 1j*t23.imag
    return Y

# ---------------------------- kompleks CNN ----------------------------
class CConv(nn.Module):
    def __init__(s,i,o,k=3,p=1):
        super().__init__(); s.r=nn.Conv2d(i,o,k,padding=p); s.i=nn.Conv2d(i,o,k,padding=p)
    def forward(s,xr,xi): return s.r(xr)-s.i(xi), s.r(xi)+s.i(xr)
class CLin(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.r=nn.Linear(i,o); s.i=nn.Linear(i,o)
    def forward(s,xr,xi): return s.r(xr)-s.i(xi), s.r(xi)+s.i(xr)
def crelu(a,b): return torch.relu(a), torch.relu(b)
def cpool(a,b): return nn.functional.avg_pool2d(a,2), nn.functional.avg_pool2d(b,2)

class CVCNN(nn.Module):
    def __init__(s,ncl,cin=6):
        super().__init__()
        s.c1=CConv(cin,32); s.c2=CConv(32,64); s.c3=CConv(64,128)
        s.f1=CLin(128*3*3,128); s.f2=CLin(128,ncl); s.do=nn.Dropout(0.3)
    def forward(s,xr,xi):
        xr,xi=crelu(*s.c1(xr,xi)); xr,xi=crelu(*s.c2(xr,xi)); xr,xi=cpool(xr,xi)
        xr,xi=crelu(*s.c3(xr,xi)); xr,xi=cpool(xr,xi)
        xr=s.do(xr.flatten(1)); xi=s.do(xi.flatten(1))
        xr,xi=crelu(*s.f1(xr,xi)); xr,xi=s.f2(xr,xi)
        return torch.sqrt(xr**2+xi**2+1e-9)
