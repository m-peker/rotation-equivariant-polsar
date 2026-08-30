"""
Deney 02 - Tavan buyuk pencerede ne kadar, ve gt OLMADAN ne kadari ulasilabilir?

Eklenen rejim:
  adaptive-w : Log-Euclidean benzerlik agirlikli toplama (gt KULLANMAZ)
               w_ij ~ exp(-d_LE(T_i,T_j)^2 / 2s^2),  d_LE = ||log T_i - log T_j||_F
               ve toplama MANIFOLD uzerinde: exp( sum_j w_j log T_j / sum_j w_j )
Bu, onerdigim katmanin ELLE TASARLANMIS (ogrenilmemis) hali.
Ayrica ayni agirliklarla Oklid ortalamasi -> manifold ortalamasinin katkisini izole eder.
"""
import numpy as np, scipy.io as sio, time
from scipy.ndimage import uniform_filter, distance_transform_edt

R, C = 750, 1024
DP = "c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/Flevoland"
TP = DP + "/T3"
IDX = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
rd = lambda n: np.fromfile(TP+"/"+n, dtype="<f4").reshape(R,C).astype(np.float64)

def load_T():
    T = np.empty((R,C,3,3), np.complex128)
    T[...,0,0],T[...,1,1],T[...,2,2] = rd("T11.bin"),rd("T22.bin"),rd("T33.bin")
    for (i,j),nm in zip([(0,1),(0,2),(1,2)], ["T12","T13","T23"]):
        v = rd(nm+"_real.bin") + 1j*rd(nm+"_imag.bin")
        T[...,i,j] = v; T[...,j,i] = np.conj(v)
    return T

to_comp = lambda T: np.stack([T[...,i,j] for (i,j) in IDX], -1)
def rebuild(cp):
    T = np.empty((R,C,3,3), np.complex128)
    for k,(i,j) in enumerate(IDX):
        T[...,i,j] = cp[...,k]
        if i!=j: T[...,j,i] = np.conj(cp[...,k])
    return T

def boxcar(T,w):
    cp = to_comp(T); out = np.empty_like(cp)
    for k in range(6):
        out[...,k] = (uniform_filter(cp[...,k].real,w,mode="nearest")
                      +1j*uniform_filter(cp[...,k].imag,w,mode="nearest"))
    return rebuild(out)

def oracle(T,gt,w):
    cp = to_comp(T); out = np.zeros_like(cp)
    for c in np.unique(gt):
        m = (gt==c).astype(np.float64)
        cnt = uniform_filter(m,w,mode="constant",cval=0.); cnt[cnt<=0]=np.nan
        sel = gt==c
        for k in range(6):
            sr = uniform_filter(cp[...,k].real*m,w,mode="constant",cval=0.)
            si = uniform_filter(cp[...,k].imag*m,w,mode="constant",cval=0.)
            out[...,k][sel] = ((sr/cnt)+1j*(si/cnt))[sel]
    return rebuild(out)

# ---- matris log / exp (3x3 Hermityen, ozdeger tabani ile duzenlilestirilmis) ----
def herm_log(T, floor=1e-6):
    w,V = np.linalg.eigh(T)                     # (R,C,3), (R,C,3,3)
    tr = np.clip(w.sum(-1, keepdims=True), 1e-30, None)
    w = np.clip(w, floor*tr, None)              # negatif/sifir ozdegerleri tabanla
    return (V*np.log(w)[...,None,:]) @ np.conj(np.swapaxes(V,-1,-2))

def herm_exp(L):
    w,V = np.linalg.eigh(L)
    return (V*np.exp(w)[...,None,:]) @ np.conj(np.swapaxes(V,-1,-2))

def shift(A, di, dj):
    """kenarlarda tekrarlamali (nearest) kaydirma."""
    B = np.roll(A, (di,dj), axis=(0,1))
    if di>0:   B[:di]  = B[di:di+1]
    elif di<0: B[di:]  = B[di-1:di]
    if dj>0:   B[:,:dj]  = B[:,dj:dj+1]
    elif dj<0: B[:,dj:]  = B[:,dj-1:dj]
    return B

def adaptive(logT, w, sigma, manifold=True):
    """gt kullanmayan, Log-Euclidean benzerlik agirlikli toplama."""
    m = w//2
    acc  = np.zeros_like(logT)
    accw = np.zeros((R,C,1,1))
    Tlin = herm_exp(logT) if not manifold else None
    accE = np.zeros((R,C,3,3), np.complex128) if not manifold else None
    for di in range(-m, m+1):
        for dj in range(-m, m+1):
            Ls = shift(logT, di, dj)
            d2 = np.sum(np.abs(Ls-logT)**2, axis=(-1,-2))          # ||.||_F^2 (Log-Euclidean)
            wt = np.exp(-d2/(2*sigma**2))[..., None, None]
            accw += wt
            if manifold: acc += wt*Ls
            else:        accE += wt*shift(Tlin, di, dj)
    if manifold: return herm_exp(acc/accw)
    return accE/accw

# ---------------- siniflandirici ----------------
def wfit(T,gt,tr,ncl,ridge=1e-8):
    S = np.empty((ncl,3,3), np.complex128); r,c = tr; lab = gt[r,c]
    for k in range(ncl):
        s = lab==k+1; S[k] = T[r[s],c[s]].mean(0)
        S[k] += ridge*np.trace(S[k]).real*np.eye(3)
    return S

def wpred(T,S,ev,chunk=200_000):
    r,c = ev; ncl = S.shape[0]
    inv = np.linalg.inv(S); ld = np.array([np.linalg.slogdet(S[k])[1] for k in range(ncl)])
    pr = np.empty(len(r), np.int32)
    for s in range(0,len(r),chunk):
        e = min(s+chunk,len(r))
        tr_ = np.einsum("kij,nji->nk", inv, T[r[s:e],c[s:e]]).real
        pr[s:e] = np.argmin(ld[None,:]+tr_,1)+1
    return pr

def bdist(gt):
    b = np.zeros(gt.shape,bool)
    b[:-1,:] |= gt[:-1,:]!=gt[1:,:]; b[1:,:] |= gt[:-1,:]!=gt[1:,:]
    b[:,:-1] |= gt[:,:-1]!=gt[:,1:]; b[:,1:] |= gt[:,:-1]!=gt[:,1:]
    return distance_transform_edt(~b)

# ---------------- kurulum ----------------
gt = sio.loadmat(DP+"/Flevoland_gt.mat")["gt"].astype(np.int32); ncl = int(gt.max())
T0 = load_T(); dist = bdist(gt)
lr,lc = np.nonzero(gt>0)
rng = np.random.default_rng(0); tr_r,tr_c = [],[]
for k in range(1,ncl+1):
    rr,cc = np.nonzero(gt==k); p = rng.choice(len(rr),min(50,len(rr)),replace=False)
    tr_r.append(rr[p]); tr_c.append(cc[p])
tr = (np.concatenate(tr_r), np.concatenate(tr_c))
ts = set(zip(tr[0].tolist(),tr[1].tolist()))
keep = np.array([(a,b) not in ts for a,b in zip(lr.tolist(),lc.tolist())])
ev = (lr[keep], lc[keep]); ed = dist[ev[0],ev[1]]; et = gt[ev[0],ev[1]]

BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]
def ev_run(name,T,store):
    S = wfit(T,gt,tr,ncl); ok = wpred(T,S,ev)==et
    per = [100*ok[(ed>=lo)&(ed<=hi)].mean() for lo,hi in BINS]
    store[name] = (100*ok.mean(), per)
    print(f"{name:<16} {100*ok.mean():5.2f}" + "".join(f" | {p:6.2f}" for p in per), flush=True)

print("rejim              OA   " + "".join(f" | d={lo}-{hi if hi<99 else '+':<2}" for lo,hi in BINS))
print("-"*66)
res={}
for w in (9,11,15):
    ev_run(f"boxcar-{w}", boxcar(T0,w), res)
    ev_run(f"oracle-{w}", oracle(T0,gt,w), res)

print("\nmatris log hesaplaniyor...", flush=True); t=time.time()
logT = herm_log(T0); print(f"  {time.time()-t:.1f}s", flush=True)

print("\n--- adaptive (gt KULLANMAZ), w=15, sigma taramasi ---", flush=True)
for sg in (0.5, 1.0, 2.0, 4.0):
    t=time.time(); A = adaptive(logT,15,sg,manifold=True)
    ev_run(f"adapt-LE-s{sg}", A, res); print(f"    ({time.time()-t:.0f}s)", flush=True)

print("\n--- ayni agirliklar, Oklid ortalamasi (manifold katkisini izole eder) ---", flush=True)
best = max([f"adapt-LE-s{s}" for s in (0.5,1.0,2.0,4.0)], key=lambda k: res[k][0])
sg_best = float(best.split("s")[-1])
ev_run(f"adapt-EU-s{sg_best}", adaptive(logT,15,sg_best,manifold=False), res)

print("\n=== OZET: w=15 ===")
b = res["boxcar-15"]; o = res["oracle-15"]; a = res[best]; e = res[f"adapt-EU-s{sg_best}"]
print(f"{'':<22}   OA   " + "".join(f" | d={lo}-{hi if hi<99 else '+':<2}" for lo,hi in BINS))
for nm,v in [("boxcar",b),("adaptive-LE (gt yok)",a),("adaptive-EU (gt yok)",e),("ORACLE (gt ile)",o)]:
    print(f"{nm:<22} {v[0]:5.2f}" + "".join(f" | {p:6.2f}" for p in v[1]))
print(f"\n{'tavan (oracle-boxcar)':<22} {o[0]-b[0]:+5.2f}" + "".join(f" | {x-y:+6.2f}" for x,y in zip(o[1],b[1])))
print(f"{'ulasilan (adapt-boxcar)':<22} {a[0]-b[0]:+5.2f}" + "".join(f" | {x-y:+6.2f}" for x,y in zip(a[1],b[1])))
gap = o[0]-b[0]
print(f"\n>>> tavanin ulasilan orani (OA): %{100*(a[0]-b[0])/gap:.0f}" if abs(gap)>1e-9 else "")
print(f">>> sigma = {sg_best}")
