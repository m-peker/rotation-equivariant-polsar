"""
Deney 09 - Artirimin 3.19 puanlik maliyeti YAPISAL mi, yoksa az egitim/kapasite mi?

Bu, B yonunun (insa gereci esdegisirlik) kader sorusu:
  - Daha uzun egitim + daha genis ag ile rot-AUG 97.7'ye ciksa -> B'nin degeri yok.
  - Bosluk kalirsa -> B'nin somut, olculmus hedefi var: "esdegisirligi bedava al".

Ayrica: az-etiket rejiminde artirimin maliyeti buyuyor mu? (esdegisirlik orada
daha cok deger tasir)
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
from polsar_lib import *
from sklearn.model_selection import train_test_split
dev="cuda"

gt=load_gt(); ncl=int(gt.max()); Xraw=load6()
mu=np.array([Xraw[...,k].mean() for k in range(6)],np.complex64)
sd=np.array([Xraw[...,k].std() for k in range(6)],np.float32)
W=15;M=W//2
Xp=np.pad(Xraw,((M,M),(M,M),(0,0)),mode="constant")
Pr=torch.from_numpy(np.ascontiguousarray(Xp.real)).to(dev)
Pi=torch.from_numpy(np.ascontiguousarray(Xp.imag)).to(dev)
MUr=torch.tensor(mu.real,device=dev).view(1,6,1,1);MUi=torch.tensor(mu.imag,device=dev).view(1,6,1,1)
SD=torch.tensor(sd,device=dev).view(1,6,1,1); off=torch.arange(W,device=dev)
def grab(r,c,th=None):
    rr=r[:,None,None]+off[None,:,None]; cc=c[:,None,None]+off[None,None,:]
    xr=Pr[rr,cc].permute(0,3,1,2).contiguous(); xi=Pi[rr,cc].permute(0,3,1,2).contiguous()
    if th is not None: xr,xi=rot6_torch(xr,xi,th)
    return (xr-MUr)/SD,(xi-MUi)/SD

class CVCNNw(nn.Module):
    """genislik carpanli surum."""
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

lr_,lc_=np.nonzero(gt>0); y=gt[lr_,lc_]-1
def split(seed,per_class=None):
    if per_class is None:
        return train_test_split(np.arange(len(y)),test_size=0.99,random_state=345+seed,stratify=y)
    rg=np.random.default_rng(1000+seed); tr=[]
    for k in range(ncl):
        idx=np.nonzero(y==k)[0]; tr.append(rg.choice(idx,min(per_class,len(idx)),replace=False))
    tr=np.concatenate(tr); te=np.setdiff1d(np.arange(len(y)),tr)
    return tr,te

def run(seed,aug,epochs,width,per_class=None,evth=(0,45)):
    itr,ite=split(seed,per_class)
    Rt=torch.from_numpy(lr_[itr]).to(dev);Ct=torch.from_numpy(lc_[itr]).to(dev);Yt=torch.from_numpy(y[itr]).to(dev)
    torch.manual_seed(seed); net=CVCNNw(ncl,w=width).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm=torch.randperm(len(itr),device=dev)
        for s in range(0,len(itr),128):
            b=pm[s:s+128]
            th=(torch.rand(len(b),device=dev)*np.pi) if aug else None
            xr,xi=grab(Rt[b],Ct[b],th); opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
    net.eval(); Re_=torch.from_numpy(lr_[ite]).to(dev);Ce=torch.from_numpy(lc_[ite]).to(dev)
    out=[]
    for tdeg in evth:
        th=torch.tensor(np.deg2rad(tdeg),device=dev,dtype=torch.float32) if tdeg else None
        pr=np.empty(len(ite),np.int64)
        with torch.no_grad():
            for s in range(0,len(ite),4096):
                xr,xi=grab(Re_[s:s+4096],Ce[s:s+4096],th); pr[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
        out.append(100*(pr==y[ite]).mean())
    npar=sum(p.numel() for p in net.parameters())
    del net; torch.cuda.empty_cache()
    return out, npar

print("### BOLUM 1: artirimin maliyeti yapisal mi? (standart %1 protokol) ###", flush=True)
print(f"{'kosul':<28}{'params':>9}  {'th=0':>7} {'th=45':>7}", flush=True)
cfgs=[("no-aug  120ep 1x",False,120,1),("no-aug  360ep 1x",False,360,1),
      ("rot-AUG 120ep 1x",True,120,1),("rot-AUG 360ep 1x",True,360,1),
      ("rot-AUG 360ep 2x",True,360,2),("rot-AUG 720ep 2x",True,720,2)]
P1={}
for nm,aug,ep,wd in cfgs:
    r=[];t0=time.time()
    for sd_ in (0,1):
        o,npar=run(sd_,aug,ep,wd); r.append(o)
    r=np.array(r); P1[nm]=r.mean(0)
    print(f"{nm:<28}{npar:>9}  {r[:,0].mean():7.2f} {r[:,1].mean():7.2f}   ({time.time()-t0:.0f}s)", flush=True)

base=P1["no-aug  120ep 1x"][0]
bestaug=max(P1[k][0] for k in P1 if k.startswith("rot-AUG"))
print(f"\n>>> no-aug theta=0 : {base:.2f}")
print(f">>> EN IYI rot-AUG theta=0 : {bestaug:.2f}   -> KALAN BOSLUK {base-bestaug:+.2f} puan", flush=True)

print("\n### BOLUM 2: az-etiket rejiminde artirimin maliyeti buyuyor mu? ###", flush=True)
print(f"{'etiket/sinif':<14}{'no-aug th0':>12}{'AUG th0':>10}{'maliyet':>9}{'no-aug th45':>13}{'AUG th45':>10}", flush=True)
for pc in (10,25,50,None):
    ra=[];rb=[]
    for sd_ in (0,1,2):
        o,_=run(sd_,False,300,1,per_class=pc); ra.append(o)
        o,_=run(sd_,True ,300,1,per_class=pc); rb.append(o)
    ra=np.array(ra);rb=np.array(rb)
    lbl=str(pc) if pc else "tumu(%1)"
    print(f"{lbl:<14}{ra[:,0].mean():12.2f}{rb[:,0].mean():10.2f}{rb[:,0].mean()-ra[:,0].mean():+9.2f}"
          f"{ra[:,1].mean():13.2f}{rb[:,1].mean():10.2f}", flush=True)
