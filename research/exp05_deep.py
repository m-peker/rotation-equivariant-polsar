"""
Deney 05 - Guclu derin temel modelin kalan hatasi NEREDE?

Tek soru: CV-CNN sinifi bir ag standart protokolde ~%99'a ciktiginda,
kalan hatanin yuzde kaci sinir piksellerinde birikiyor?

Protokol: makalenin aynisi -- %1 egitim / %99 test, rastgele stratified split.
Model: kompleks-degerli CNN (cart_relu, softmax_real_with_abs) -- CV-MsAtViT sinifi.
"""
import numpy as np, scipy.io as sio, torch, torch.nn as nn, time, sys
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split

R, C = 750, 1024
DP = "c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/Flevoland"
TP = DP + "/T3"
dev = "cuda"
rd = lambda n: np.fromfile(TP+"/"+n, dtype="<f4").reshape(R,C)

def load6():
    """T3'un 6 bagimsiz bileseni -> (R,C,6) complex64."""
    X = np.empty((R,C,6), np.complex64)
    X[...,0]=rd("T11.bin"); X[...,1]=rd("T22.bin"); X[...,2]=rd("T33.bin")
    X[...,3]=rd("T12_real.bin")+1j*rd("T12_imag.bin")
    X[...,4]=rd("T13_real.bin")+1j*rd("T13_imag.bin")
    X[...,5]=rd("T23_real.bin")+1j*rd("T23_imag.bin")
    return X

def bdist(gt):
    b=np.zeros(gt.shape,bool)
    b[:-1,:]|=gt[:-1,:]!=gt[1:,:]; b[1:,:]|=gt[:-1,:]!=gt[1:,:]
    b[:,:-1]|=gt[:,:-1]!=gt[:,1:]; b[:,1:]|=gt[:,:-1]!=gt[:,1:]
    return distance_transform_edt(~b)

# ---------------- kompleks katmanlar ----------------
class CConv(nn.Module):
    def __init__(s, i, o, k=3, p=1):
        super().__init__(); s.r=nn.Conv2d(i,o,k,padding=p); s.i=nn.Conv2d(i,o,k,padding=p)
    def forward(s, xr, xi): return s.r(xr)-s.i(xi), s.r(xi)+s.i(xr)

class CLin(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.r=nn.Linear(i,o); s.i=nn.Linear(i,o)
    def forward(s, xr, xi): return s.r(xr)-s.i(xi), s.r(xi)+s.i(xr)

def crelu(xr, xi): return torch.relu(xr), torch.relu(xi)
def cpool(xr, xi): return nn.functional.avg_pool2d(xr,2), nn.functional.avg_pool2d(xi,2)

class CVCNN(nn.Module):
    def __init__(s, ncl, cin=6):
        super().__init__()
        s.c1=CConv(cin,32); s.c2=CConv(32,64); s.c3=CConv(64,128)
        s.f1=CLin(128*3*3,128); s.f2=CLin(128,ncl); s.do=nn.Dropout(0.3)
    def forward(s, xr, xi):
        xr,xi = crelu(*s.c1(xr,xi)); xr,xi = crelu(*s.c2(xr,xi)); xr,xi = cpool(xr,xi)   # 15->7
        xr,xi = crelu(*s.c3(xr,xi)); xr,xi = cpool(xr,xi)                                # 7->3
        xr=s.do(xr.flatten(1)); xi=s.do(xi.flatten(1))
        xr,xi = crelu(*s.f1(xr,xi)); xr,xi = s.f2(xr,xi)
        return torch.sqrt(xr**2+xi**2+1e-9)        # softmax_real_with_abs -> logit=|z|

# ---------------- veri ----------------
gt = sio.loadmat(DP+"/Flevoland_gt.mat")["gt"].astype(np.int64); ncl=int(gt.max())
X = load6()
for k in range(6):                                   # kanal bazli kompleks z-score
    X[...,k] = (X[...,k]-X[...,k].mean())/X[...,k].std()
dist = bdist(gt)
W=15; M=W//2
Xp = np.pad(X, ((M,M),(M,M),(0,0)), mode="constant")
Pr = torch.from_numpy(Xp.real.copy()).to(dev)        # (R+2M, C+2M, 6)
Pi = torch.from_numpy(Xp.imag.copy()).to(dev)
off = torch.arange(W, device=dev)

def grab(r, c):
    rr = r[:,None,None]+off[None,:,None]; cc = c[:,None,None]+off[None,None,:]
    return (Pr[rr,cc].permute(0,3,1,2).contiguous(),
            Pi[rr,cc].permute(0,3,1,2).contiguous())

lr_,lc_ = np.nonzero(gt>0); y = gt[lr_,lc_]-1
print(f"etiketli {len(y)} piksel, {ncl} sinif, GPU {torch.cuda.get_device_name(0)}", flush=True)

BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]
allres=[]
for seed in (0,1,2):
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.99, random_state=345+seed, stratify=y)
    Rtr=torch.from_numpy(lr_[itr]).to(dev); Ctr=torch.from_numpy(lc_[itr]).to(dev)
    Ytr=torch.from_numpy(y[itr]).to(dev)
    torch.manual_seed(seed)
    net=CVCNN(ncl).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
    lossf=nn.CrossEntropyLoss()
    t0=time.time(); net.train()
    for ep in range(120):
        perm=torch.randperm(len(itr),device=dev)
        for s in range(0,len(itr),128):
            b=perm[s:s+128]
            xr,xi=grab(Rtr[b],Ctr[b])
            opt.zero_grad(); l=lossf(net(xr,xi),Ytr[b]); l.backward(); opt.step()
    net.eval()
    Rte=torch.from_numpy(lr_[ite]).to(dev); Cte=torch.from_numpy(lc_[ite]).to(dev)
    pred=np.empty(len(ite),np.int64)
    with torch.no_grad():
        for s in range(0,len(ite),4096):
            xr,xi=grab(Rte[s:s+4096],Cte[s:s+4096])
            pred[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
    ok = pred==y[ite]; d=dist[lr_[ite],lc_[ite]]
    oa=100*ok.mean()
    print(f"\nseed {seed}: OA={oa:.2f}  ({time.time()-t0:.0f}s, {len(itr)} egitim)", flush=True)
    per=[]
    for lo,hi in BINS:
        m=(d>=lo)&(d<=hi); per.append(100*ok[m].mean())
    print("  d-katmanli dogruluk: " + "  ".join(f"d={lo}-{hi if hi<99 else '+'}:{p:.2f}" for (lo,hi),p in zip(BINS,per)), flush=True)
    err=~ok
    print(f"  HATA DAGILIMI: toplam {err.sum()} hata", flush=True)
    for thr in (2,4,7):
        fe=100*(d[err]<=thr).mean(); fb=100*(d<=thr).mean()
        print(f"    d<={thr}: hatalarin %{fe:.1f}'i  |  test kumesinin %{fb:.1f}'i  |  zenginlesme x{fe/fb:.2f}", flush=True)
    allres.append((oa,per,[100*(d[err]<=t).mean() for t in (2,4,7)],[100*(d<=t).mean() for t in (2,4,7)]))

o=np.array([a[0] for a in allres]); P=np.array([a[1] for a in allres])
E=np.array([a[2] for a in allres]); B=np.array([a[3] for a in allres])
print(f"\n=== 3 tohum ozeti ===\nOA = {o.mean():.2f} +- {o.std():.2f}")
print("d-katmanli: " + "  ".join(f"d={lo}-{hi if hi<99 else '+'}:{m:.2f}+-{s:.2f}" for (lo,hi),m,s in zip(BINS,P.mean(0),P.std(0))))
for i,thr in enumerate((2,4,7)):
    print(f"hatalarin d<={thr} orani: %{E[:,i].mean():.1f}  (taban %{B[:,i].mean():.1f})  zenginlesme x{E[:,i].mean()/B[:,i].mean():.2f}")
