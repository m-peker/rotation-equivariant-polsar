"""
Deney 11 - Esdegisken ag vs artirim: tam karsilastirma + ablasyonlar.

Bilinen temeller (exp10):
  baseline no-aug   97.37 -> 29.31 (cokuyor)
  baseline rot-AUG  94.69 duz      (dayanikli ama temiz dogruluktan -2.7..-3.7)

Sinanacak: esdegisken ag rot-AUG'u geciyor mu, ve grup disi acilarda ne oluyor?
Ablasyon: max vs Fourier okuma, N=4/8/16, artirimla birlesim.
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
from data_pipe import Pipe
from equivariant import EqCVCNN, EqCVCNN_F
from sklearn.model_selection import train_test_split

P=Pipe(); ncl=P.ncl; y=P.y; lr_,lc_=P.lr,P.lc; dist=P.dist
# 0/22.5/45/67.5/90 = N=8 icin GRUP ICI ; 10/20/30/60 = GRUP DISI
THETAS=[0,10,20,22.5,30,45,60,90]
ONGRID={0,22.5,45,67.5,90}
BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]

def run(seed, cls, N, aug, epochs):
    itr,ite=train_test_split(np.arange(len(y)),test_size=0.99,random_state=345+seed,stratify=y)
    Rt=torch.from_numpy(lr_[itr]).cuda();Ct=torch.from_numpy(lc_[itr]).cuda();Yt=torch.from_numpy(y[itr]).cuda()
    torch.manual_seed(seed); net=cls(ncl,N=N).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm=torch.randperm(len(itr),device="cuda")
        for s in range(0,len(itr),128):
            b=pm[s:s+128]
            th=(torch.rand(len(b),device="cuda")*np.pi) if aug else None
            xr,xi=P.grab(Rt[b],Ct[b],th)
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
    net.eval()
    Re_=torch.from_numpy(lr_[ite]).cuda();Ce=torch.from_numpy(lc_[ite]).cuda()
    accs=[]; bnd=None
    for tdeg in THETAS:
        th=torch.tensor(np.deg2rad(tdeg),device="cuda",dtype=torch.float32) if tdeg else None
        pr=np.empty(len(ite),np.int64)
        with torch.no_grad():
            for s in range(0,len(ite),2048):
                xr,xi=P.grab(Re_[s:s+2048],Ce[s:s+2048],th)
                pr[s:s+2048]=net(xr,xi).argmax(1).cpu().numpy()
        ok=pr==y[ite]; accs.append(100*ok.mean())
        if tdeg==0:
            d=dist[lr_[ite],lc_[ite]]
            bnd=[100*ok[(d>=lo)&(d<=hi)].mean() for lo,hi in BINS]
    npar=sum(p.numel() for p in net.parameters()); del net; torch.cuda.empty_cache()
    return accs,bnd,npar

CFG=[("EQ-max   N=8  no-aug", EqCVCNN,  8, False,120),
     ("EQ-fourierN=8  no-aug", EqCVCNN_F,8, False,120),
     ("EQ-fourierN=8  rot-AUG",EqCVCNN_F,8, True ,120),
     ("EQ-fourierN=4  no-aug", EqCVCNN_F,4, False,120),
     ("EQ-fourierN=16 no-aug", EqCVCNN_F,16,False,120),
     ("EQ-fourierN=8  360ep",  EqCVCNN_F,8, False,360)]
print("theta:      " + "".join(f"{t:>8}" for t in THETAS) + "   (* = grup ici)", flush=True)
print("            " + "".join(f"{'*' if t in ONGRID else '':>8}" for t in THETAS), flush=True)
R={}
for nm,cls,N,aug,ep in CFG:
    A=[];B=[];t0=time.time()
    for sd in (0,1,2):
        a,b,npar=run(sd,cls,N,aug,ep); A.append(a); B.append(b)
    A=np.array(A);B=np.array(B); R[nm]=(A,B,npar)
    print(f"{nm:<22}" + "".join(f"{m:8.2f}" for m in A.mean(0)) + f"  p={npar}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"{'  sinir d=1-2 / ic d=12+':<22}{B.mean(0)[0]:8.2f}{B.mean(0)[-1]:8.2f}", flush=True)

print("\n=== OZET (theta=0 / en kotu aci / cokus / ortalama) ===")
print(f"{'model':<24}{'th=0':>8}{'enkotu':>9}{'cokus':>8}{'ort':>8}{'params':>10}")
print(f"{'baseline no-aug  (exp10)':<24}{97.37:8.2f}{29.31:9.2f}{68.06:8.2f}{62.51:8.2f}{487262:10d}")
print(f"{'baseline rot-AUG (exp10)':<24}{94.69:8.2f}{94.27:9.2f}{0.42:8.2f}{94.56:8.2f}{487262:10d}")
for nm,_,_,_,_ in CFG:
    A,B,npar=R[nm]; m=A.mean(0)
    print(f"{nm:<24}{m[0]:8.2f}{m.min():9.2f}{m[0]-m.min():8.2f}{m.mean():8.2f}{npar:10d}")
