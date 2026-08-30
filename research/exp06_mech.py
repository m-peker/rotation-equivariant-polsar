"""
Deney 06 - Sinir hatalarinin MEKANIZMASI nedir?

Exp05: hatalarin %72.5'i sinirdan <=2 px (x3.14 zenginlesme).
Soru: bu hatalar KOMSU SINIFA mi kaciyor (patch kirlenmesi), yoksa rastgele mi?
Cevap yontemin ne olacagini belirler.
"""
import numpy as np, scipy.io as sio, torch, time
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
exec(open("exp05_deep.py").read().split("# ---------------- veri")[0])

gt = sio.loadmat(DP+"/Flevoland_gt.mat")["gt"].astype(np.int64); ncl=int(gt.max())
X = load6()
for k in range(6): X[...,k]=(X[...,k]-X[...,k].mean())/X[...,k].std()
dist=bdist(gt); W=15; M=W//2
Xp=np.pad(X,((M,M),(M,M),(0,0)),mode="constant")
Pr=torch.from_numpy(Xp.real.copy()).to(dev); Pi=torch.from_numpy(Xp.imag.copy()).to(dev)
off=torch.arange(W,device=dev)
def grab(r,c):
    rr=r[:,None,None]+off[None,:,None]; cc=c[:,None,None]+off[None,None,:]
    return Pr[rr,cc].permute(0,3,1,2).contiguous(), Pi[rr,cc].permute(0,3,1,2).contiguous()
gtp = np.pad(gt,((M,M),(M,M)),mode="constant")
lr_,lc_=np.nonzero(gt>0); y=gt[lr_,lc_]-1
import torch.nn as nn

agg={}
for seed in (0,1):
    itr,ite = train_test_split(np.arange(len(y)),test_size=0.99,random_state=345+seed,stratify=y)
    Rtr=torch.from_numpy(lr_[itr]).to(dev);Ctr=torch.from_numpy(lc_[itr]).to(dev);Ytr=torch.from_numpy(y[itr]).to(dev)
    torch.manual_seed(seed); net=CVCNN(ncl).to(dev)
    opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(120):
        pm=torch.randperm(len(itr),device=dev)
        for s in range(0,len(itr),128):
            b=pm[s:s+128]; xr,xi=grab(Rtr[b],Ctr[b])
            opt.zero_grad(); lf(net(xr,xi),Ytr[b]).backward(); opt.step()
    net.eval(); Rte=torch.from_numpy(lr_[ite]).to(dev);Cte=torch.from_numpy(lc_[ite]).to(dev)
    pred=np.empty(len(ite),np.int64)
    with torch.no_grad():
        for s in range(0,len(ite),4096):
            xr,xi=grab(Rte[s:s+4096],Cte[s:s+4096]); pred[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
    tr_lab=y[ite]; ok=pred==tr_lab; rr_=lr_[ite]; cc_=lc_[ite]; d=dist[rr_,cc_]
    print(f"\n=== seed {seed}: OA={100*ok.mean():.2f} ===", flush=True)

    # patch icindeki sinif kompozisyonu
    er=np.nonzero(~ok)[0]
    def patch_labels(i):
        r,c=rr_[i],cc_[i]; return gtp[r:r+W, c:c+W].ravel()
    n_in_patch=0; n_major=0; n_true_minor=0; truefrac_err=[]; truefrac_ok=[]
    rng=np.random.default_rng(0)
    smp_ok = rng.choice(np.nonzero(ok)[0], min(20000,ok.sum()), replace=False)
    for i in er:
        pl=patch_labels(i); nz=pl[pl>0]
        if len(nz)==0: continue
        p=pred[i]+1; t=tr_lab[i]+1
        if (nz==p).any(): n_in_patch+=1
        cnt=np.bincount(nz,minlength=ncl+2)
        if cnt.argmax()==p: n_major+=1
        tf=(nz==t).mean(); truefrac_err.append(tf)
        if cnt[p]>cnt[t]: n_true_minor+=1
    for i in smp_ok:
        pl=patch_labels(i); nz=pl[pl>0]
        if len(nz)==0: continue
        truefrac_ok.append((nz==tr_lab[i]+1).mean())
    ne=len(er)
    # taban: rastgele bir yanlis sinifin patch'te bulunma olasiligi
    base=0
    for i in er[:5000]:
        pl=patch_labels(i); nz=pl[pl>0]
        if len(nz)==0: continue
        others=[k for k in range(1,ncl+1) if k!=tr_lab[i]+1]
        base += np.mean([ (nz==k).any() for k in others ])
    base=100*base/min(5000,ne)
    print(f"  hata sayisi: {ne}")
    print(f"  tahmin edilen sinif patch ICINDE       : %{100*n_in_patch/ne:.1f}   (rastgele taban %{base:.1f})")
    print(f"  tahmin edilen sinif patch'te COGUNLUK  : %{100*n_major/ne:.1f}")
    print(f"  tahmin edilen sinif gercek siniftan COK: %{100*n_true_minor/ne:.1f}")
    print(f"  patch'te gercek sinifin orani  -- HATA : {np.mean(truefrac_err):.3f}")
    print(f"  patch'te gercek sinifin orani  -- DOGRU: {np.mean(truefrac_ok):.3f}")
    agg[seed]=(100*n_in_patch/ne,100*n_major/ne,100*n_true_minor/ne,
               np.mean(truefrac_err),np.mean(truefrac_ok),base)

A=np.array([agg[s] for s in agg])
print("\n=== 2 tohum ortalamasi ===")
print(f"tahmin edilen sinif patch icinde        : %{A[:,0].mean():.1f}  (taban %{A[:,5].mean():.1f} -> zenginlesme x{A[:,0].mean()/A[:,5].mean():.2f})")
print(f"tahmin edilen sinif patch'te cogunluk   : %{A[:,1].mean():.1f}")
print(f"tahmin edilen sinif gercekten daha cok  : %{A[:,2].mean():.1f}")
print(f"patch'te gercek sinif orani  HATA/DOGRU : {A[:,3].mean():.3f} / {A[:,4].mean():.3f}")
