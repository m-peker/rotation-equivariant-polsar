"""
Deney 16 - STEERABLE vs AYRIK G-CNN vs temel modeller.

Ayrik G-CNN (N=8) yalnizca theta = m*22.5 deg noktalarinda TAM; aradaki acilarda
kayip veriyor. Steerable ag grubun harmonik tabaninda calisir -> HER acida tam
(dogrulandi: 8.7e-15, keyfi acilarda).

Bu deney SUREKLI aci taramasi yapar: izgara ustu VE izgara disi acilar.
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool
from equivariant import EqCVCNN, EqCVCNN_F
from steerable import SteerNet

THETAS=[0,5,10,11.25,15,22.5,30,33.75,45,60,67.5,90]
GRID8={0,22.5,45,67.5,90}
BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]
BUDGET={"flevoland":133,"sanfran":400,"ober":666}

class CVCNNb(nn.Module):
    def __init__(s,ncl,cin=7,w=1):
        super().__init__(); a,b,c=32*w,64*w,128*w
        s.c1=CConv(cin,a); s.c2=CConv(a,b); s.c3=CConv(b,c)
        s.f1=CLin(c*3*3,128*w); s.f2=CLin(128*w,ncl); s.do=nn.Dropout(0.3)
    def forward(s,xr,xi):
        xr,xi=crelu(*s.c1(xr,xi)); xr,xi=crelu(*s.c2(xr,xi)); xr,xi=cpool(xr,xi)
        xr,xi=crelu(*s.c3(xr,xi)); xr,xi=cpool(xr,xi)
        xr=s.do(xr.flatten(1)); xi=s.do(xi.flatten(1))
        xr,xi=crelu(*s.f1(xr,xi)); xr,xi=s.f2(xr,xi)
        return torch.sqrt(xr**2+xi**2+1e-9)

class SteerWrap(nn.Module):
    """SteerNet ham T3 bekler (kendi ayristirmasini yapar) -> boru hatti ham vermeli."""
    def __init__(s,ncl): super().__init__(); s.n=SteerNet(ncl)
    def forward(s,xr,xi): return s.n(xr,xi)

def build(k,ncl):
    return {"base":lambda: CVCNNb(ncl,cin=7),
            "eqmax":lambda: EqCVCNN(ncl,cin=7,N=8),
            "eqf":  lambda: EqCVCNN_F(ncl,cin=7,N=8),
            "steer":lambda: SteerWrap(ncl)}[k]()

def sample(P,n,seed):
    rng=np.random.default_rng(1000+seed); tr=[]
    for k in range(P.ncl):
        idx=np.nonzero(P.y==k)[0]; tr.append(rng.choice(idx,min(n,len(idx)),replace=False))
    tr=np.concatenate(tr); te=np.setdiff1d(np.arange(len(P.y)),tr)
    return tr, P.cap_eval(te)

def run(P,Praw,k,aug,seed,epochs=120):
    pipe = Praw if k=="steer" else P            # steerable HAM girdi alir
    itr,ite=sample(P,BUDGET[P.scene],seed)
    Rt=torch.from_numpy(P.lr[itr]).cuda();Ct=torch.from_numpy(P.lc[itr]).cuda()
    Yt=torch.from_numpy(P.y[itr]).cuda()
    torch.manual_seed(seed); net=build(k,P.ncl).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm=torch.randperm(len(itr),device="cuda")
        for s in range(0,len(itr),128):
            b=pm[s:s+128]
            th=(torch.rand(len(b),device="cuda")*np.pi) if aug else None
            xr,xi=pipe.grab(Rt[b],Ct[b],th)
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
        sch.step()
    net.eval(); Re=torch.from_numpy(P.lr[ite]).cuda();Ce=torch.from_numpy(P.lc[ite]).cuda()
    accs=[];bnd=None
    for tdeg in THETAS:
        th=torch.tensor(np.deg2rad(tdeg),device="cuda",dtype=torch.float32) if tdeg else None
        pr=np.empty(len(ite),np.int64)
        with torch.no_grad():
            for s in range(0,len(ite),4096):
                xr,xi=pipe.grab(Re[s:s+4096],Ce[s:s+4096],th)
                pr[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
        ok=pr==P.y[ite]; accs.append(100*ok.mean())
        if tdeg==0:
            d=P.dist[P.lr[ite],P.lc[ite]]
            bnd=[100*ok[(d>=lo)&(d<=hi)].mean() for lo,hi in BINS]
    npar=sum(p.numel() for p in net.parameters()); del net; torch.cuda.empty_cache()
    return accs,bnd,npar

SC="flevoland"
P=MSPipe(SC,norm="equivariant",eval_cap=150_000)
Praw=MSPipe(SC,norm="raw6",eval_cap=150_000) if False else None
# steerable icin ham girdi gerek: norm kapali bir Pipe
import pipe_ms
class RawPipe(pipe_ms.MSPipe):
    def grab(s,r,c,th=None):
        rr=r[:,None,None]+s.off[None,:,None]; cc=c[:,None,None]+s.off[None,None,:]
        xr=s.Pr[rr,cc].permute(0,3,1,2).contiguous(); xi=s.Pi[rr,cc].permute(0,3,1,2).contiguous()
        if th is not None:
            from polsar_lib import rot6_torch; xr,xi=rot6_torch(xr,xi,th)
        return xr,xi
Praw=RawPipe(SC,norm="equivariant",eval_cap=150_000)

ARMS=[("baseline no-aug","base",False),("baseline rot-AUG","base",True),
      ("EQ-max   N=8","eqmax",False),("EQ-fourier N=8","eqf",False),
      ("STEERABLE","steer",False)]
print(f"### {P.info()} ###",flush=True)
print("theta:"+"".join(f"{t:>7}" for t in THETAS),flush=True)
print("      "+"".join(f"{'*' if t in GRID8 else '':>7}" for t in THETAS)+"   (* = N=8 izgarasi)",flush=True)
R={}
for nm,k,aug in ARMS:
    A=[];B=[];t0=time.time()
    for sd in (0,1):
        a,b,npar=run(P,Praw,k,aug,sd); A.append(a);B.append(b)
    A=np.array(A);B=np.array(B); R[nm]=(A.mean(0),B.mean(0),npar)
    print(f"{nm:<17}"+"".join(f"{m:7.2f}" for m in A.mean(0))+f"  sinir={B.mean(0)[0]:6.2f} p={npar:>7} ({time.time()-t0:.0f}s)",flush=True)

print("\n=== IZGARA USTU vs IZGARA DISI acilar (steerable'in iddiasi) ===")
print(f"{'model':<18}{'izgara-ustu ort':>17}{'izgara-disi ort':>17}{'FARK':>8}{'en kotu':>9}")
for nm,_,_ in ARMS:
    m=R[nm][0]
    on =np.mean([m[i] for i,t in enumerate(THETAS) if t in GRID8])
    off=np.mean([m[i] for i,t in enumerate(THETAS) if t not in GRID8])
    print(f"{nm:<18}{on:17.2f}{off:17.2f}{on-off:8.2f}{m.min():9.2f}")
