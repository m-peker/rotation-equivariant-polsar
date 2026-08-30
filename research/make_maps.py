"""Donme altinda TAM SAHNE siniflandirma haritalari (en carpici gorsel)."""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool, rot6_torch
from steerable import SteerNet, field_stats
from polsar_data import load_scene

SC="flevoland"; BUD=133
class CVCNNb(nn.Module):
    def __init__(s,ncl,cin=7):
        super().__init__(); s.c1=CConv(cin,32); s.c2=CConv(32,64); s.c3=CConv(64,128)
        s.f1=CLin(128*3*3,128); s.f2=CLin(128,ncl); s.do=nn.Dropout(0.3)
    def forward(s,xr,xi):
        xr,xi=crelu(*s.c1(xr,xi)); xr,xi=crelu(*s.c2(xr,xi)); xr,xi=cpool(xr,xi)
        xr,xi=crelu(*s.c3(xr,xi)); xr,xi=cpool(xr,xi)
        xr=s.do(xr.flatten(1)); xi=s.do(xi.flatten(1))
        xr,xi=crelu(*s.f1(xr,xi)); xr,xi=s.f2(xr,xi)
        return torch.sqrt(xr**2+xi**2+1e-9)

P=MSPipe(SC,norm="equivariant")
import pipe_ms
class RawPipe(pipe_ms.MSPipe):
    def grab(s,r,c,th=None):
        rr=r[:,None,None]+s.off[None,:,None]; cc=c[:,None,None]+s.off[None,None,:]
        xr=s.Pr[rr,cc].permute(0,3,1,2).contiguous(); xi=s.Pi[rr,cc].permute(0,3,1,2).contiguous()
        if th is not None: xr,xi=rot6_torch(xr,xi,th)
        return xr,xi
Praw=RawPipe(SC,norm="equivariant")
Xs,_,_=load_scene(SC); ST=field_stats(Xs); del Xs

rng=np.random.default_rng(1000)
tr=np.concatenate([rng.choice(np.nonzero(P.y==k)[0],min(BUD,(P.y==k).sum()),replace=False) for k in range(P.ncl)])
Rt=torch.from_numpy(P.lr[tr]).cuda();Ct=torch.from_numpy(P.lc[tr]).cuda();Yt=torch.from_numpy(P.y[tr]).cuda()

def train(kind,aug,epochs=120):
    pipe = Praw if kind=="steer" else P
    torch.manual_seed(0)
    net=(SteerNet(P.ncl,stats=ST) if kind=="steer" else CVCNNb(P.ncl)).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm=torch.randperm(len(tr),device="cuda")
        for s in range(0,len(tr),128):
            b=pm[s:s+128]
            th=(torch.rand(len(b),device="cuda")*np.pi) if aug else None
            xr,xi=pipe.grab(Rt[b],Ct[b],th)
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
        sch.step()
    net.eval(); return net,pipe

def full_map(net,pipe,thdeg):
    R,C=P.gt.shape
    rr,cc=np.meshgrid(np.arange(R),np.arange(C),indexing="ij")
    rr=rr.ravel(); cc=cc.ravel()
    th=torch.tensor(np.deg2rad(thdeg),device="cuda",dtype=torch.float32) if thdeg else None
    out=np.empty(len(rr),np.uint8)
    with torch.no_grad():
        for s in range(0,len(rr),8192):
            r=torch.from_numpy(rr[s:s+8192]).cuda(); c=torch.from_numpy(cc[s:s+8192]).cuda()
            xr,xi=pipe.grab(r,c,th)
            out[s:s+8192]=(net(xr,xi).argmax(1)+1).cpu().numpy().astype(np.uint8)
    return out.reshape(R,C)

import os
ALLM=[("baseline","base",False),("rotAUG","base",True),("steerable","steer",False)]
# bitmis olanlari atla (diskte .npy varsa yeniden kosma)
MODELS=[m for m in ALLM if not all(os.path.exists(f"figs/map_{m[0]}_{t}.npy") for t in (0,22.5,45))]
print("kosulacak modeller:", [m[0] for m in MODELS], flush=True)
res={}
for nm,k,aug in MODELS:
    t0=time.time(); net,pipe=train(k,aug)
    for th in (0,22.5,45):
        m=full_map(net,pipe,th)
        np.save(f"figs/map_{nm}_{th}.npy", m)
        ok=(m[P.lr,P.lc]==P.gt[P.lr,P.lc])
        res[(nm,th)]=100*ok.mean()
        print(f"{nm:<10} th={th:<5} OA={100*ok.mean():6.2f}",flush=True)
    print(f"   ({time.time()-t0:.0f}s)",flush=True)
    del net; torch.cuda.empty_cache()

print("haritalar kaydedildi")
