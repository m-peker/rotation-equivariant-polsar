"""
Deney 07 - Sinir hatasi bir KOVARYANS KARISIMI etkisi mi?

Hipotez: saglayicinin cok-bakislamasi sinir pikselinde T ~ a*S_t + (1-a)*S_komsu
uretiyor. Iki sinifin karisimi HPD uzayinda UCUNCU bir sinifin yanina dusuyor
-> ag, patch'te hic bulunmayan bir sinifi tahmin ediyor (exp06 bulgusu).

Kesin test: karisimi SENTEZLEYIP agin yaptigi hatayi ONGOREBILIYOR muyuz?
"""
import numpy as np, scipy.io as sio, torch, torch.nn as nn
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
exec(open("exp05_deep.py").read().split("# ---------------- veri")[0])

gt=sio.loadmat(DP+"/Flevoland_gt.mat")["gt"].astype(np.int64); ncl=int(gt.max())
Xr=load6()                                   # ham (standardize edilmemis) -> fizik icin
X=Xr.copy()
for k in range(6): X[...,k]=(X[...,k]-X[...,k].mean())/X[...,k].std()
dist=bdist(gt); W=15; M=W//2
Xp=np.pad(X,((M,M),(M,M),(0,0)),mode="constant")
Pr=torch.from_numpy(Xp.real.copy()).to(dev);Pi=torch.from_numpy(Xp.imag.copy()).to(dev)
off=torch.arange(W,device=dev)
def grab(r,c):
    rr=r[:,None,None]+off[None,:,None]; cc=c[:,None,None]+off[None,None,:]
    return Pr[rr,cc].permute(0,3,1,2).contiguous(),Pi[rr,cc].permute(0,3,1,2).contiguous()
gtp=np.pad(gt,((M,M),(M,M)),mode="constant")
lr_,lc_=np.nonzero(gt>0); y=gt[lr_,lc_]-1

itr,ite=train_test_split(np.arange(len(y)),test_size=0.99,random_state=345,stratify=y)
Rtr=torch.from_numpy(lr_[itr]).to(dev);Ctr=torch.from_numpy(lc_[itr]).to(dev);Ytr=torch.from_numpy(y[itr]).to(dev)
torch.manual_seed(0); net=CVCNN(ncl).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss()
net.train()
for ep in range(120):
    pm=torch.randperm(len(itr),device=dev)
    for s in range(0,len(itr),128):
        b=pm[s:s+128]; xr,xi=grab(Rtr[b],Ctr[b]); opt.zero_grad(); lf(net(xr,xi),Ytr[b]).backward(); opt.step()
net.eval()
Rte=torch.from_numpy(lr_[ite]).to(dev);Cte=torch.from_numpy(lc_[ite]).to(dev)
pred=np.empty(len(ite),np.int64)
with torch.no_grad():
    for s in range(0,len(ite),4096):
        xr,xi=grab(Rte[s:s+4096],Cte[s:s+4096]); pred[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
tl=y[ite]; ok=pred==tl; rr_=lr_[ite]; cc_=lc_[ite]; d=dist[rr_,cc_]
print(f"OA={100*ok.mean():.2f}", flush=True)

# ---- HAM T'den 3x3 matrisler ve saf sinif uc-uyeleri (sadece IC pikseller) ----
def T33(idx6):
    T=np.empty(idx6.shape[:-1]+(3,3),np.complex128)
    T[...,0,0]=idx6[...,0];T[...,1,1]=idx6[...,1];T[...,2,2]=idx6[...,2]
    T[...,0,1]=idx6[...,3];T[...,1,0]=np.conj(idx6[...,3])
    T[...,0,2]=idx6[...,4];T[...,2,0]=np.conj(idx6[...,4])
    T[...,1,2]=idx6[...,5];T[...,2,1]=np.conj(idx6[...,5])
    return T
SIG=np.empty((ncl,3,3),np.complex128)
for k in range(ncl):
    m=(gt==k+1)&(dist>7)                       # SAF ic pikseller
    if m.sum()<50: m=(gt==k+1)&(dist>4)
    SIG[k]=T33(Xr[m]).mean(0); SIG[k]+=1e-8*np.trace(SIG[k]).real*np.eye(3)
inv=np.linalg.inv(SIG); ld=np.array([np.linalg.slogdet(SIG[k])[1] for k in range(ncl)])
def wishart_cls(T):
    return np.argmin(ld[None,:]+np.einsum("kij,nji->nk",inv,T).real,1)

# ---- sinir hatalari icin karisim uydur ----
rng=np.random.default_rng(0)
err=np.nonzero((~ok)&(d<=2))[0]; cor=np.nonzero(ok&(d<=2))[0]
err=rng.choice(err,min(3000,len(err)),replace=False)
cor=rng.choice(cor,min(3000,len(cor)),replace=False)
AL=np.linspace(0,1,21)

def analyze(sel, tag):
    T_obs=T33(Xr[rr_[sel],cc_[sel]])
    alphas=np.empty(len(sel)); mixcls=np.empty(len(sel),np.int64); adj=np.empty(len(sel),np.int64)
    for n,i in enumerate(sel):
        t=tl[i]; pl=gtp[rr_[i]:rr_[i]+W, cc_[i]:cc_[i]+W].ravel()
        nz=pl[(pl>0)&(pl!=t+1)]
        a = (np.bincount(nz).argmax()-1) if len(nz) else t
        adj[n]=a
        cand=AL[:,None,None]*SIG[t]+(1-AL)[:,None,None]*SIG[a]        # (21,3,3)
        lo=np.linalg.slogdet(cand)[1]
        ic=np.linalg.inv(cand)
        dd=np.linalg.norm(cand-T_obs[n][None],axis=(1,2))             # Oklid uyum
        j=dd.argmin(); alphas[n]=AL[j]
        mixcls[n]=wishart_cls(cand[j][None])[0]
    obs_cls=wishart_cls(T_obs)
    p=pred[sel]; t=tl[sel]
    print(f"\n--- {tag} (n={len(sel)}) ---")
    print(f"  uydurulan alpha (gercek sinifin agirligi): ort {alphas.mean():.3f}  medyan {np.median(alphas):.3f}")
    print(f"  alpha < 0.9 olan pay                     : %{100*(alphas<0.9).mean():.1f}")
    print(f"  komsu sinif != gercek sinif olan pay     : %{100*(adj!=t).mean():.1f}")
    if tag.startswith("HATA"):
        print(f"  KARISIM, agin tahminiyle AYNI sinifa dusuyor : %{100*(mixcls==p).mean():.1f}")
        print(f"  gozlenen T, agin tahminiyle ayni sinifa      : %{100*(obs_cls==p).mean():.1f}")
        print(f"  karisim ne gercek ne komsu (UCUNCU sinif)    : %{100*((mixcls!=t)&(mixcls!=adj)).mean():.1f}")
    print(f"  karisim gercek sinifa dusuyor                : %{100*(mixcls==t).mean():.1f}")
    return alphas

ae=analyze(err,"HATA (sinir, d<=2)")
ac=analyze(cor,"DOGRU (sinir, d<=2)")
print(f"\n>>> alpha farki (dogru - hata): {ac.mean()-ae.mean():+.3f}")
