"""
Deney 08 - B yonunun on kabulu + ASIL KONTROL.

S1: Mevcut CV-CNN polarimetrik donme U(theta) altinda cokuyor mu?
S2: (KRITIK) Basit donme ARTIRIMI bunu zaten cozuyor mu?
    Coziyorsa "insa gereci esdegisirlik" fikrinin degeri dusuk olur -- hakem
    "sadece augment et" der. Cozmuyorsa B'ye yer var.
S3: Artirim temiz (theta=0) dogrulugu bozuyor mu? Yonelimi BILGI olan siniflarda
    (Buildings) zarar veriyor mu?
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
from polsar_lib import *
from sklearn.model_selection import train_test_split
dev="cuda"

gt=load_gt(); ncl=int(gt.max()); Xraw=load6(); dist=bdist(gt)
mu=np.array([Xraw[...,k].mean() for k in range(6)],np.complex64)
sd=np.array([Xraw[...,k].std()  for k in range(6)],np.float32)
W=15;M=W//2
Xp=np.pad(Xraw,((M,M),(M,M),(0,0)),mode="constant")
Pr=torch.from_numpy(np.ascontiguousarray(Xp.real)).to(dev)
Pi=torch.from_numpy(np.ascontiguousarray(Xp.imag)).to(dev)
MUr=torch.tensor(mu.real,device=dev).view(1,6,1,1); MUi=torch.tensor(mu.imag,device=dev).view(1,6,1,1)
SD =torch.tensor(sd,device=dev).view(1,6,1,1)
off=torch.arange(W,device=dev)

def grab(r,c,th=None):
    rr=r[:,None,None]+off[None,:,None]; cc=c[:,None,None]+off[None,None,:]
    xr=Pr[rr,cc].permute(0,3,1,2).contiguous(); xi=Pi[rr,cc].permute(0,3,1,2).contiguous()
    if th is not None: xr,xi = rot6_torch(xr,xi,th)
    return (xr-MUr)/SD, (xi-MUi)/SD

lr_,lc_=np.nonzero(gt>0); y=gt[lr_,lc_]-1
THETAS=[0,10,20,30,45,60,90]
def train(seed, aug):
    itr,ite=train_test_split(np.arange(len(y)),test_size=0.99,random_state=345+seed,stratify=y)
    Rt=torch.from_numpy(lr_[itr]).to(dev);Ct=torch.from_numpy(lc_[itr]).to(dev);Yt=torch.from_numpy(y[itr]).to(dev)
    torch.manual_seed(seed); net=CVCNN(ncl).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(120):
        pm=torch.randperm(len(itr),device=dev)
        for s in range(0,len(itr),128):
            b=pm[s:s+128]
            th=(torch.rand(len(b),device=dev)*np.pi) if aug else None
            xr,xi=grab(Rt[b],Ct[b],th)
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
    net.eval(); return net, ite

def evaluate(net, ite, thdeg):
    Re_=torch.from_numpy(lr_[ite]).to(dev);Ce=torch.from_numpy(lc_[ite]).to(dev)
    th=torch.tensor(np.deg2rad(thdeg),device=dev,dtype=torch.float32) if thdeg else None
    pr=np.empty(len(ite),np.int64)
    with torch.no_grad():
        for s in range(0,len(ite),4096):
            xr,xi=grab(Re_[s:s+4096],Ce[s:s+4096],th)
            pr[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
    t=y[ite]; ok=pr==t
    aa=np.mean([ok[t==k].mean() for k in range(ncl) if (t==k).sum()>0])
    return 100*ok.mean(), 100*aa, 100*ok[t==ncl-1].mean()   # OA, AA, Buildings

res={}
for aug in (False,True):
    tag="rot-AUG" if aug else "no-aug "
    for seed in (0,1,2):
        t0=time.time(); net,ite=train(seed,aug)
        for th in THETAS:
            oa,aa,bd=evaluate(net,ite,th)
            res.setdefault((tag,th),[]).append((oa,aa,bd))
        print(f"{tag} seed{seed} bitti ({time.time()-t0:.0f}s)", flush=True)
        del net; torch.cuda.empty_cache()

print("\n=== OA (%), theta = polarimetrik donme acisi ===")
print(f"{'egitim':<10}" + "".join(f"{t:>8}deg" for t in THETAS))
for tag in ("no-aug ","rot-AUG"):
    a=np.array([[x[0] for x in res[(tag,t)]] for t in THETAS])
    print(f"{tag:<10}" + "".join(f"{m:8.2f}  " for m in a.mean(1)))
    print(f"{'  (std)':<10}" + "".join(f"{s:8.2f}  " for s in a.std(1)))
print("\n=== AA (%) ===")
for tag in ("no-aug ","rot-AUG"):
    a=np.array([[x[1] for x in res[(tag,t)]] for t in THETAS])
    print(f"{tag:<10}" + "".join(f"{m:8.2f}  " for m in a.mean(1)))
print("\n=== Buildings sinifi (yonelim BILGI tasir) (%) ===")
for tag in ("no-aug ","rot-AUG"):
    a=np.array([[x[2] for x in res[(tag,t)]] for t in THETAS])
    print(f"{tag:<10}" + "".join(f"{m:8.2f}  " for m in a.mean(1)))

n0=np.mean([x[0] for x in res[("no-aug ",0)]]); a0=np.mean([x[0] for x in res[("rot-AUG",0)]])
nw=np.mean([np.mean([x[0] for x in res[("no-aug ",t)]]) for t in THETAS])
aw=np.mean([np.mean([x[0] for x in res[("rot-AUG",t)]]) for t in THETAS])
nmin=min(np.mean([x[0] for x in res[("no-aug ",t)]]) for t in THETAS)
amin=min(np.mean([x[0] for x in res[("rot-AUG",t)]]) for t in THETAS)
print(f"\n--- ozet ---")
print(f"theta=0 dogrulugu   : no-aug {n0:.2f}  |  rot-AUG {a0:.2f}   (artirimin temiz maliyeti: {a0-n0:+.2f})")
print(f"tum theta ortalamasi: no-aug {nw:.2f}  |  rot-AUG {aw:.2f}   (fark {aw-nw:+.2f})")
print(f"EN KOTU theta       : no-aug {nmin:.2f}  |  rot-AUG {amin:.2f}   (fark {amin-nmin:+.2f})")
print(f"no-aug cokusu (0 -> en kotu): {n0-nmin:.2f} puan")
print(f"rot-AUG kalan bosluk (0 -> en kotu): {a0-amin:.2f} puan")
