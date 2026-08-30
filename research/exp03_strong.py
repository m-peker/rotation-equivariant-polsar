"""
Deney 03 - Kazanc, UZAMSAL BAGLAMI olan bir siniflandiricinin ustune de biniyor mu?

Bu, kullanicinin asil endisesi: CNN 15x15 patch gordugu icin boxcar'in sinir
hasarini kendisi telafi ediyor olabilir. O zaman katkimiz anlamsizlasir.

Vekil siniflandirici: 15x15 patch uzerinde Log-Euclidean oznitelikleri (9-dim/piksel
-> 2025 oznitelik) + L2 lojistik regresyon. Uzamsal baglami VAR, agirliklari
icerikten bagimsiz (konvolusyon gibi) -> CNN'in birinci katmani icin dogru vekil.

Girdi rejimleri: native / boxcar-15 / adaptive-15 (gt yok) / oracle-15 (ust sinir)
Eger adaptive, native'i patch-tabanli siniflandiricida da yeniyorsa, katki
uzamsal baglamdan BAGIMSIZ olarak vardir.
"""
import numpy as np, scipy.io as sio, time
from scipy.ndimage import uniform_filter, distance_transform_edt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

exec(open("exp02_reachable.py").read().split("# ---------------- kurulum")[0])  # yardimci fonksiyonlar

gt = sio.loadmat(DP+"/Flevoland_gt.mat")["gt"].astype(np.int32); ncl = int(gt.max())
T0 = load_T(); dist = bdist(gt); logT0 = herm_log(T0)
lr_,lc_ = np.nonzero(gt>0)

rng = np.random.default_rng(0)
tr_r,tr_c = [],[]
for k in range(1,ncl+1):
    rr,cc = np.nonzero(gt==k); p = rng.choice(len(rr),min(50,len(rr)),replace=False)
    tr_r.append(rr[p]); tr_c.append(cc[p])
tr = (np.concatenate(tr_r), np.concatenate(tr_c))
ts = set(zip(tr[0].tolist(),tr[1].tolist()))
keep = np.array([(a,b) not in ts for a,b in zip(lr_.tolist(),lc_.tolist())])
ALL = (lr_[keep], lc_[keep])
sub = rng.choice(len(ALL[0]), 40000, replace=False)          # patch testi icin alt orneklem
ev  = (ALL[0][sub], ALL[1][sub]); ed = dist[ev[0],ev[1]]; et = gt[ev[0],ev[1]]
BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]

# ---- sigma tepesini kesinlestir (w=15) ----
print("--- sigma taramasi devami (w=15, Wishart ML, tum degerlendirme kumesi) ---", flush=True)
evA = ALL; edA = dist[evA[0],evA[1]]; etA = gt[evA[0],evA[1]]
def wml(T, e, t):
    S = wfit(T,gt,tr,ncl); return 100*(wpred(T,S,e)==t).mean()
for sg in (6.0, 8.0, 12.0):
    t0=time.time(); A = adaptive(logT0,15,sg,manifold=True)
    print(f"  sigma={sg:<5} OA={wml(A,evA,etA):5.2f}   ({time.time()-t0:.0f}s)", flush=True)
    del A

# ---- patch tabanli guclu siniflandirici ----
def feats(T):
    """her piksel icin log-Euclidean 9-dim gercek oznitelik."""
    L = herm_log(T)
    return np.stack([L[...,0,0].real, L[...,1,1].real, L[...,2,2].real,
                     L[...,0,1].real, L[...,0,1].imag,
                     L[...,0,2].real, L[...,0,2].imag,
                     L[...,1,2].real, L[...,1,2].imag], -1).astype(np.float32)

def patches(F, idx, w=15):
    m = w//2
    Fp = np.pad(F, ((m,m),(m,m),(0,0)), mode="edge")
    r,c = idx
    out = np.empty((len(r), w*w*F.shape[-1]), np.float32)
    for i,(rr,cc) in enumerate(zip(r,c)):
        out[i] = Fp[rr:rr+w, cc:cc+w].ravel()
    return out

print("\n--- patch tabanli (15x15) L2 lojistik regresyon ---", flush=True)
print(f"{'girdi rejimi':<18}   OA   " + "".join(f" | d={lo}-{hi if hi<99 else '+':<2}" for lo,hi in BINS))
print("-"*70)
A_best = adaptive(logT0, 15, 8.0, manifold=True)
conds = [("native", T0), ("boxcar-15", boxcar(T0,15)),
         ("adaptive-15", A_best), ("oracle-15", oracle(T0,gt,15))]
out = {}
for nm, T in conds:
    F = feats(T)
    Xtr, Xev = patches(F,tr), patches(F,ev)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=2000, n_jobs=-1)
    clf.fit(sc.transform(Xtr), gt[tr[0],tr[1]])
    ok = clf.predict(sc.transform(Xev)) == et
    per = [100*ok[(ed>=lo)&(ed<=hi)].mean() for lo,hi in BINS]
    out[nm] = (100*ok.mean(), per)
    print(f"{nm:<18} {100*ok.mean():5.2f}" + "".join(f" | {p:6.2f}" for p in per), flush=True)
    del F, Xtr, Xev

print("\n=== patch siniflandiricida farklar (puan) ===")
n = out["native"]
for nm in ("boxcar-15","adaptive-15","oracle-15"):
    v = out[nm]
    print(f"{nm+' - native':<22} {v[0]-n[0]:+5.2f}" + "".join(f" | {a-b:+6.2f}" for a,b in zip(v[1],n[1])))
