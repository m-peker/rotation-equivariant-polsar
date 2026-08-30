"""
Deney 13 - DUZELTILMIS karsilastirma (irrep/esdegisir normalizasyon ile).

exp11 gecersizdi: kanal-basina z-score U(theta) ile yer degistirmiyordu, bu yuzden
esdegisken ag boru hattinda esdegisirligini kaybediyordu (uctan uca sapma 5.4e-2).
Simdi eqnorm ile sapma 1.3e-5 (float32 tabani). Temel modeller de AYNI
normalizasyonla kosuluyor ki karsilastirma adil olsun.
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
from data_pipe import Pipe
from polsar_lib import CVCNN
from equivariant import EqCVCNN, EqCVCNN_F
from sklearn.model_selection import train_test_split

P=Pipe(norm="equivariant"); ncl=P.ncl; y=P.y; lr_,lc_=P.lr,P.lc; dist=P.dist
THETAS=[0,10,22.5,30,45]; ONGRID={0,22.5,45}
BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]

def build(kind,N):
    return {"std":lambda: CVCNN(ncl), "eqmax":lambda: EqCVCNN(ncl,N=N),
            "eqf":lambda: EqCVCNN_F(ncl,N=N)}[kind]()

def run(seed,kind,N,aug,epochs):
    itr,ite=train_test_split(np.arange(len(y)),test_size=0.99,random_state=345+seed,stratify=y)
    Rt=torch.from_numpy(lr_[itr]).cuda();Ct=torch.from_numpy(lc_[itr]).cuda();Yt=torch.from_numpy(y[itr]).cuda()
    torch.manual_seed(seed); net=build(kind,N).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm=torch.randperm(len(itr),device="cuda")
        for s in range(0,len(itr),128):
            b=pm[s:s+128]
            th=(torch.rand(len(b),device="cuda")*np.pi) if aug else None
            xr,xi=P.grab(Rt[b],Ct[b],th)
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
    net.eval(); Re_=torch.from_numpy(lr_[ite]).cuda();Ce=torch.from_numpy(lc_[ite]).cuda()
    accs=[];bnd=None
    for tdeg in THETAS:
        th=torch.tensor(np.deg2rad(tdeg),device="cuda",dtype=torch.float32) if tdeg else None
        pr=np.empty(len(ite),np.int64)
        with torch.no_grad():
            for s in range(0,len(ite),4096):
                xr,xi=P.grab(Re_[s:s+4096],Ce[s:s+4096],th)
                pr[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
        ok=pr==y[ite]; accs.append(100*ok.mean())
        if tdeg==0:
            d=dist[lr_[ite],lc_[ite]]; bnd=[100*ok[(d>=lo)&(d<=hi)].mean() for lo,hi in BINS]
    npar=sum(p.numel() for p in net.parameters()); del net; torch.cuda.empty_cache()
    return accs,bnd,npar

CFG=[("baseline  no-aug",  "std",  8,False,120),
     ("baseline  rot-AUG", "std",  8,True ,120),
     ("EQ-max    N=8",     "eqmax",8,False,120),
     ("EQ-fourier N=8",    "eqf",  8,False,120),
     ("EQ-fourier N=16",   "eqf", 16,False,120)]
print("theta:" + "".join(f"{t:>9}" for t in THETAS) + "     (0/22.5/45 = GRUP ICI)", flush=True)
R={}
for nm,k,N,aug,ep in CFG:
    A=[];B=[];t0=time.time()
    for sd in (0,1,2):
        a,b,npar=run(sd,k,N,aug,ep); A.append(a); B.append(b)
    A=np.array(A);B=np.array(B); R[nm]=(A,B,npar)
    print(f"{nm:<18}"+"".join(f"{m:9.2f}" for m in A.mean(0))+f"  p={npar:>7} ({time.time()-t0:.0f}s)",flush=True)
    print(f"{'   sinir/ic':<18}{B.mean(0)[0]:9.2f}{B.mean(0)[-1]:9.2f}",flush=True)

print("\n=== OZET ===")
print(f"{'model':<18}{'th=0':>8}{'grup-ici min':>14}{'tum min':>10}{'cokus':>8}{'params':>10}")
for nm,_,_,_,_ in CFG:
    A,_,npar=R[nm]; m=A.mean(0)
    ong=[m[i] for i,t in enumerate(THETAS) if t in ONGRID]
    print(f"{nm:<18}{m[0]:8.2f}{min(ong):14.2f}{m.min():10.2f}{m[0]-m.min():8.2f}{npar:10d}")
