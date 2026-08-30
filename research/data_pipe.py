"""Ortak veri hatti: ham T3 GPU'da, patch cikarma + donme + standardizasyon."""
import numpy as np, torch
from polsar_lib import load6, load_gt, bdist, rot6_torch
from eqnorm import EqNorm
class Pipe:
    def __init__(s, dev="cuda", W=15, norm="channel"):
        s.dev=dev; s.W=W; s.M=W//2
        s.gt=load_gt(); s.ncl=int(s.gt.max()); Xraw=load6(); s.dist=bdist(s.gt)
        s.norm=norm; s.eqn=EqNorm(Xraw).to(dev) if norm=="equivariant" else None
        mu=np.array([Xraw[...,k].mean() for k in range(6)],np.complex64)
        sd=np.array([Xraw[...,k].std()  for k in range(6)],np.float32)
        Xp=np.pad(Xraw,((s.M,s.M),(s.M,s.M),(0,0)),mode="constant")
        s.Pr=torch.from_numpy(np.ascontiguousarray(Xp.real)).to(dev)
        s.Pi=torch.from_numpy(np.ascontiguousarray(Xp.imag)).to(dev)
        s.MUr=torch.tensor(mu.real,device=dev).view(1,6,1,1)
        s.MUi=torch.tensor(mu.imag,device=dev).view(1,6,1,1)
        s.SD =torch.tensor(sd,device=dev).view(1,6,1,1)
        s.off=torch.arange(W,device=dev)
        s.lr,s.lc=np.nonzero(s.gt>0); s.y=s.gt[s.lr,s.lc]-1
    def grab(s,r,c,th=None):
        rr=r[:,None,None]+s.off[None,:,None]; cc=c[:,None,None]+s.off[None,None,:]
        xr=s.Pr[rr,cc].permute(0,3,1,2).contiguous(); xi=s.Pi[rr,cc].permute(0,3,1,2).contiguous()
        if th is not None: xr,xi=rot6_torch(xr,xi,th)
        if s.eqn is not None: return s.eqn(xr,xi)
        return (xr-s.MUr)/s.SD, (xi-s.MUi)/s.SD
