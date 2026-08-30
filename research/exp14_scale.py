"""
Deney 14 - RISK TESTI: kapasite arttikca esdegisirlik avantaji eriyor mu?

Bu, calismanin en buyuk tehdidi. Eger ag SOTA-rekabetci olcege buyudugunde
rot-AUG, EQ'yu yakaliyorsa hikaye zayiflar. Simdi ogrenmek, 2 ay sonra
ogrenmekten iyi.

Ayrica egitim tarifi iyilestirildi (cosine LR) -- yayinlanmis sayilara
(98.35-99.75) yaklasmak icin gerekli.
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
from data_pipe import Pipe
from polsar_lib import CConv, CLin, crelu, cpool
from equivariant import EqCVCNN
from sklearn.model_selection import train_test_split

P=Pipe(norm="equivariant"); ncl=P.ncl; y=P.y; lr_,lc_=P.lr,P.lc; dist=P.dist
THETAS=[0,10,22.5]; BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]

class CVCNNw(nn.Module):
    def __init__(s,ncl,cin=6,w=1):
        super().__init__(); a,b,c=32*w,64*w,128*w
        s.c1=CConv(cin,a); s.c2=CConv(a,b); s.c3=CConv(b,c)
        s.f1=CLin(c*3*3,128*w); s.f2=CLin(128*w,ncl); s.do=nn.Dropout(0.3)
    def forward(s,xr,xi):
        xr,xi=crelu(*s.c1(xr,xi)); xr,xi=crelu(*s.c2(xr,xi)); xr,xi=cpool(xr,xi)
        xr,xi=crelu(*s.c3(xr,xi)); xr,xi=cpool(xr,xi)
        xr=s.do(xr.flatten(1)); xi=s.do(xi.flatten(1))
        xr,xi=crelu(*s.f1(xr,xi)); xr,xi=s.f2(xr,xi)
        return torch.sqrt(xr**2+xi**2+1e-9)

def run(seed,kind,w,aug,epochs=120):
    itr,ite=train_test_split(np.arange(len(y)),test_size=0.99,random_state=345+seed,stratify=y)
    Rt=torch.from_numpy(lr_[itr]).cuda();Ct=torch.from_numpy(lc_[itr]).cuda();Yt=torch.from_numpy(y[itr]).cuda()
    torch.manual_seed(seed)
    net=(EqCVCNN(ncl,N=8,w=w) if kind=="eq" else CVCNNw(ncl,w=w)).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)   # iyilestirilmis tarife
    lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm=torch.randperm(len(itr),device="cuda")
        for s in range(0,len(itr),128):
            b=pm[s:s+128]
            th=(torch.rand(len(b),device="cuda")*np.pi) if aug else None
            xr,xi=P.grab(Rt[b],Ct[b],th)
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
        sch.step()
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

CFG=[("base no-aug  w1","std",1,False),("base rot-AUG w1","std",1,True ),("EQ-max       w1","eq",1,False),
     ("base no-aug  w2","std",2,False),("base rot-AUG w2","std",2,True ),("EQ-max       w2","eq",2,False)]
print("cosine LR, 120ep, eqnorm.  theta:" + "".join(f"{t:>9}" for t in THETAS), flush=True)
R={}
for nm,k,w,aug in CFG:
    A=[];B=[];t0=time.time()
    for sd in (0,1):
        a,b,npar=run(sd,k,w,aug); A.append(a); B.append(b)
    A=np.array(A);B=np.array(B); R[nm]=(A.mean(0),B.mean(0),npar)
    print(f"{nm:<17}"+"".join(f"{m:9.2f}" for m in A.mean(0))+f"  sinir={B.mean(0)[0]:6.2f}  p={npar:>8} ({time.time()-t0:.0f}s)",flush=True)

print("\n=== RISK CEVABI: EQ'nun rot-AUG'a ustunlugu kapasiteyle nasil degisiyor? ===")
for w in (1,2):
    e=R[f"EQ-max       w{w}"][0]; a=R[f"base rot-AUG w{w}"][0]; n=R[f"base no-aug  w{w}"][0]
    eb=R[f"EQ-max       w{w}"][1][0]; ab=R[f"base rot-AUG w{w}"][1][0]
    print(f"w={w}:  EQ-rotAUG (th=0) = {e[0]-a[0]:+.2f}   sinirda = {eb-ab:+.2f}   "
          f"EQ'nun no-aug'a gore acigi = {e[0]-n[0]:+.2f}")
