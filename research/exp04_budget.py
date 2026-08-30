"""
Deney 04 - Kazanc etiket butcesi kuculdukce BUYUYOR mu?

Exp03 gosterdi: uzamsal baglami olan siniflandiricida adaptive-boxcar farki
sadece +1.15 puan (50 etiket/sinif). Deep net ~%99'dan basladigi icin bu daha da
kuculur -> standart protokolde dogruluk hikayesi zayif.

Tek kalan umut: az-etiket rejimi. Geometrik on-bilgi, veri azaldikca deger kazanir.
Eger 5 etiket/sinifta fark +5 puan ve uzeriyse makale var; +1'de kalirsa yok.

3 tohum x {5,10,20,50} etiket/sinif x {native, boxcar-15, adaptive-15}
"""
import numpy as np, scipy.io as sio, time
from scipy.ndimage import uniform_filter, distance_transform_edt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

exec(open("exp02_reachable.py").read().split("# ---------------- kurulum")[0])

gt = sio.loadmat(DP+"/Flevoland_gt.mat")["gt"].astype(np.int32); ncl = int(gt.max())
T0 = load_T(); dist = bdist(gt); logT0 = herm_log(T0)
lr_,lc_ = np.nonzero(gt>0)

def feats(T):
    L = herm_log(T)
    return np.stack([L[...,0,0].real,L[...,1,1].real,L[...,2,2].real,
                     L[...,0,1].real,L[...,0,1].imag,L[...,0,2].real,L[...,0,2].imag,
                     L[...,1,2].real,L[...,1,2].imag],-1).astype(np.float32)

def patches(F, idx, w=15):
    m=w//2; Fp=np.pad(F,((m,m),(m,m),(0,0)),mode="edge"); r,c=idx
    out=np.empty((len(r),w*w*F.shape[-1]),np.float32)
    for i,(rr,cc) in enumerate(zip(r,c)): out[i]=Fp[rr:rr+w,cc:cc+w].ravel()
    return out

rng0 = np.random.default_rng(123)
sub = rng0.choice(len(lr_), 30000, replace=False)
EV = (lr_[sub], lc_[sub]); ed = dist[EV[0],EV[1]]; et = gt[EV[0],EV[1]]
BUDGETS=(5,10,20,50); SEEDS=(0,1,2)

print("adaptive-15 hesaplaniyor...", flush=True); t=time.time()
A15 = adaptive(logT0,15,8.0,manifold=True); print(f"  {time.time()-t:.0f}s", flush=True)
CONDS = [("native",T0), ("boxcar-15",boxcar(T0,15)), ("adaptive-15",A15)]

res = {}   # (cond,budget) -> list of (oa, boundary_oa)
for nm, T in CONDS:
    F = feats(T); Xev = patches(F, EV)
    sc_all = None
    for B in BUDGETS:
        for sd in SEEDS:
            rg = np.random.default_rng(1000+sd)
            tr_r,tr_c = [],[]
            for k in range(1,ncl+1):
                rr,cc = np.nonzero(gt==k)
                p = rg.choice(len(rr), min(B,len(rr)), replace=False)
                tr_r.append(rr[p]); tr_c.append(cc[p])
            tr=(np.concatenate(tr_r),np.concatenate(tr_c))
            Xtr = patches(F,tr); ytr = gt[tr[0],tr[1]]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=0.02, max_iter=3000, n_jobs=-1)
            clf.fit(sc.transform(Xtr), ytr)
            ok = clf.predict(sc.transform(Xev)) == et
            mb = ed<=2
            res.setdefault((nm,B),[]).append((100*ok.mean(), 100*ok[mb].mean()))
        oa = np.array([x[0] for x in res[(nm,B)]]); bo = np.array([x[1] for x in res[(nm,B)]])
        print(f"{nm:<12} B={B:<3} OA={oa.mean():5.2f}+-{oa.std():.2f}   sinir(d<=2)={bo.mean():5.2f}+-{bo.std():.2f}", flush=True)
    del F, Xev

print("\n=== adaptive-15 eksi boxcar-15 (puan, ortalama) ===")
print(f"{'butce':<8} {'OA fark':>9} {'sinir fark':>12}")
for B in BUDGETS:
    a=np.array(res[("adaptive-15",B)]); b=np.array(res[("boxcar-15",B)])
    print(f"{B:<8} {a[:,0].mean()-b[:,0].mean():+9.2f} {a[:,1].mean()-b[:,1].mean():+12.2f}")
print("\n=== boxcar-15 eksi native (on-toplamanin degeri) ===")
for B in BUDGETS:
    b=np.array(res[("boxcar-15",B)]); n=np.array(res[("native",B)])
    print(f"{B:<8} {b[:,0].mean()-n[:,0].mean():+9.2f} {b[:,1].mean()-n[:,1].mean():+12.2f}")
