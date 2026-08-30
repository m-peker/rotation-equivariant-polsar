"""
Deney 10 - ESAS KARSILASTIRMA.

Iddia: insa gereci esdegisirlik, veri artiriminin odettigi temiz-dogruluk
maliyetini ODEMEDEN donme dayanikliligi verir.

  baseline no-aug  : theta=0'da iyi, donmede COKUYOR        (exp08: 97.71 / 29.79)
  baseline rot-AUG : donmede saglam ama theta=0'da -3.19    (exp08: 94.52 / 93.59)
  EQUIVARIANT      : ??? -- ikisini birden verebiliyor mu?
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
from data_pipe import Pipe
from polsar_lib import CVCNN
from equivariant import EqCVCNN
from sklearn.model_selection import train_test_split

P=Pipe(); ncl=P.ncl; y=P.y; lr_,lc_=P.lr,P.lc
THETAS=[0,10,20,30,45,60,90]

def run(seed, kind, aug, epochs=120, N=8):
    itr,ite=train_test_split(np.arange(len(y)),test_size=0.99,random_state=345+seed,stratify=y)
    Rt=torch.from_numpy(lr_[itr]).cuda();Ct=torch.from_numpy(lc_[itr]).cuda()
    Yt=torch.from_numpy(y[itr]).cuda()
    torch.manual_seed(seed)
    net=(EqCVCNN(ncl,N=N) if kind=="eq" else CVCNN(ncl)).cuda()
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
    out=[]
    for tdeg in THETAS:
        th=torch.tensor(np.deg2rad(tdeg),device="cuda",dtype=torch.float32) if tdeg else None
        pr=np.empty(len(ite),np.int64)
        with torch.no_grad():
            for s in range(0,len(ite),2048):
                xr,xi=P.grab(Re_[s:s+2048],Ce[s:s+2048],th)
                pr[s:s+2048]=net(xr,xi).argmax(1).cpu().numpy()
        out.append(100*(pr==y[ite]).mean())
    npar=sum(p.numel() for p in net.parameters()); del net; torch.cuda.empty_cache()
    return out,npar

CFG=[("baseline no-aug","std",False),("baseline rot-AUG","std",True),
     ("EQUIVAR  no-aug","eq", False),("EQUIVAR  rot-AUG","eq", True)]
R={}
for nm,kind,aug in CFG:
    acc=[];t0=time.time()
    for sd in (0,1,2):
        o,npar=run(sd,kind,aug); acc.append(o)
    acc=np.array(acc); R[nm]=acc
    print(f"{nm:<18} params={npar:>7}  " + " ".join(f"{m:6.2f}" for m in acc.mean(0)) + f"   ({time.time()-t0:.0f}s)", flush=True)

print("\n=== OA (%) vs polarimetrik donme acisi ===")
print(f"{'model':<18}" + "".join(f"{t:>8}deg" for t in THETAS))
for nm,_,_ in CFG:
    a=R[nm]; print(f"{nm:<18}" + "".join(f"{m:8.2f}  " for m in a.mean(0)))
    print(f"{'  (std)':<18}" + "".join(f"{s:8.2f}  " for s in a.std(0)))
print("\n=== ozet ===")
print(f"{'model':<18}{'th=0':>8}{'en kotu':>10}{'cokus':>8}{'ort':>8}")
for nm,_,_ in CFG:
    a=R[nm].mean(0); print(f"{nm:<18}{a[0]:8.2f}{a.min():10.2f}{a[0]-a.min():8.2f}{a.mean():8.2f}")
