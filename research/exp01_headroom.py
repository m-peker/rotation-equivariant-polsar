"""
Deney 01 - Tavan olcumu (headroom).

Tez: sabit boxcar cok-bakislama sinir bolgelerinde kovaryans kestirimini bozar.
Ogrenilmis uyarlanir toplama bunu duzeltebilir. Bu deney, duzeltilebilecek
maksimum miktari (oracle ust sinir) olcer -- derin ag olmadan.

Siniflandirici: Lee'nin Wishart ML siniflandiricisi (kapali form).
  d_c(T) = ln|S_c| + tr(S_c^-1 T),  atama = argmin_c
Bu, T3'u dogrudan HPD matrisi olarak kullanan klasik referans.

Uc toplama rejimi karsilastiriliyor:
  1. native   : verili T3 (ek toplama yok)
  2. boxcar-w : w x w duzgun pencere (literaturun yaptigi)
  3. oracle-w : w x w pencere AMA sadece ayni gt sinifindaki komsular
                -> ogrenilmis uyarlanir toplamanin ust siniri
"""
import numpy as np
from scipy.ndimage import uniform_filter, distance_transform_edt

R, C = 750, 1024
DP = "c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/Flevoland"
TP = DP + "/T3"

def rd(n):
    return np.fromfile(TP + "/" + n, dtype="<f4").reshape(R, C).astype(np.float64)

def load_T():
    """T3 -> (R,C,3,3) kompleks Hermityen."""
    T11, T22, T33 = rd("T11.bin"), rd("T22.bin"), rd("T33.bin")
    T12 = rd("T12_real.bin") + 1j*rd("T12_imag.bin")
    T13 = rd("T13_real.bin") + 1j*rd("T13_imag.bin")
    T23 = rd("T23_real.bin") + 1j*rd("T23_imag.bin")
    T = np.empty((R, C, 3, 3), dtype=np.complex128)
    T[..., 0, 0] = T11; T[..., 1, 1] = T22; T[..., 2, 2] = T33
    T[..., 0, 1] = T12; T[..., 1, 0] = np.conj(T12)
    T[..., 0, 2] = T13; T[..., 2, 0] = np.conj(T13)
    T[..., 1, 2] = T23; T[..., 2, 1] = np.conj(T23)
    return T

# --- bagimsiz 6 bileseni (3 real kosegen + 3 kompleks) filtrelemek yeterli ---
IDX = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]

def rebuild(comp):
    """6 bilesenden (R,C,6) kompleks -> (R,C,3,3) Hermityen."""
    T = np.empty((R, C, 3, 3), dtype=np.complex128)
    for k, (i, j) in enumerate(IDX):
        T[..., i, j] = comp[..., k]
        if i != j:
            T[..., j, i] = np.conj(comp[..., k])
    return T

def to_comp(T):
    return np.stack([T[..., i, j] for (i, j) in IDX], axis=-1)

def boxcar(T, w):
    """w x w duzgun (boxcar) cok-bakislama."""
    comp = to_comp(T)
    out = np.empty_like(comp)
    for k in range(6):
        out[..., k] = (uniform_filter(comp[..., k].real, size=w, mode="nearest")
                       + 1j*uniform_filter(comp[..., k].imag, size=w, mode="nearest"))
    return rebuild(out)

def oracle(T, gt, w):
    """w x w pencere, ama sadece merkez pikselle ayni gt sinifindaki komsular.
    Ogrenilmis uyarlanir toplamanin ULASILABILIR UST SINIRI (gt kullaniyor)."""
    comp = to_comp(T)
    out = np.zeros_like(comp)
    for c in np.unique(gt):
        m = (gt == c).astype(np.float64)
        cnt = uniform_filter(m, size=w, mode="constant", cval=0.0)
        cnt[cnt <= 0] = np.nan
        sel = gt == c
        for k in range(6):
            sr = uniform_filter(comp[..., k].real*m, size=w, mode="constant", cval=0.0)
            si = uniform_filter(comp[..., k].imag*m, size=w, mode="constant", cval=0.0)
            out[..., k][sel] = ((sr/cnt) + 1j*(si/cnt))[sel]
    return rebuild(out)

# ---------------- Wishart ML siniflandirici ----------------
def wishart_fit(T, gt, idx_train, n_cls, ridge=1e-8):
    """Sinif basi ML kovaryans kestirimi = egitim piksellerinin T ortalamasi."""
    Sig = np.empty((n_cls, 3, 3), dtype=np.complex128)
    r, c = idx_train
    lab = gt[r, c]
    for k in range(n_cls):
        s = lab == (k + 1)
        Sig[k] = T[r[s], c[s]].mean(axis=0)
        Sig[k] += ridge*np.trace(Sig[k]).real*np.eye(3)   # PSD duzenlilestirme
    return Sig

def wishart_predict(T, Sig, idx_eval, chunk=200_000):
    r, c = idx_eval
    n_cls = Sig.shape[0]
    inv = np.empty_like(Sig); ld = np.empty(n_cls)
    for k in range(n_cls):
        inv[k] = np.linalg.inv(Sig[k])
        sgn, l = np.linalg.slogdet(Sig[k]); ld[k] = l
    pred = np.empty(len(r), dtype=np.int32)
    for s in range(0, len(r), chunk):
        e = min(s + chunk, len(r))
        Tb = T[r[s:e], c[s:e]]                       # (n,3,3)
        # tr(inv_k @ T) = sum_ij inv_k[i,j] * T[j,i]
        tr = np.einsum("kij,nji->nk", inv, Tb).real  # (n,n_cls)
        pred[s:e] = np.argmin(ld[None, :] + tr, axis=1) + 1
    return pred

# ---------------- sinira uzaklik ----------------
def boundary_distance(gt):
    """gt degerinin degistigi her yer sinir (etiketsiz bolge de tarla kenaridir)."""
    b = np.zeros(gt.shape, bool)
    b[:-1, :] |= gt[:-1, :] != gt[1:, :]
    b[1:, :]  |= gt[:-1, :] != gt[1:, :]
    b[:, :-1] |= gt[:, :-1] != gt[:, 1:]
    b[:, 1:]  |= gt[:, :-1] != gt[:, 1:]
    return distance_transform_edt(~b)

# ---------------- ana akis ----------------
import scipy.io as sio
gt = sio.loadmat(DP + "/Flevoland_gt.mat")["gt"].astype(np.int32)
n_cls = int(gt.max())
T0 = load_T()
dist = boundary_distance(gt)

lab_r, lab_c = np.nonzero(gt > 0)
print(f"etiketli piksel: {len(lab_r)}  sinif: {n_cls}")
for lo, hi in [(1,2),(3,4),(5,7),(8,11),(12,99)]:
    m = (dist[lab_r, lab_c] >= lo) & (dist[lab_r, lab_c] <= hi)
    print(f"  sinira uzaklik {lo}-{hi if hi<99 else '+'} px : {m.sum():7d}  (%{100*m.mean():.1f})")
m7 = dist[lab_r, lab_c] <= 7
print(f"  >>> 15x15 patch (margin 7) sinirdan etkilenen: %{100*m7.mean():.1f}")

# egitim: sinif basi N piksel (dengeli, few-shot benzeri)
rng = np.random.default_rng(0)
N_TRAIN = 50
tr_r, tr_c = [], []
for k in range(1, n_cls + 1):
    rr, cc = np.nonzero(gt == k)
    p = rng.choice(len(rr), size=min(N_TRAIN, len(rr)), replace=False)
    tr_r.append(rr[p]); tr_c.append(cc[p])
tr = (np.concatenate(tr_r), np.concatenate(tr_c))
tr_set = set(zip(tr[0].tolist(), tr[1].tolist()))
keep = np.array([(a, b) not in tr_set for a, b in zip(lab_r.tolist(), lab_c.tolist())])
ev = (lab_r[keep], lab_c[keep])
ev_dist = dist[ev[0], ev[1]]
ev_true = gt[ev[0], ev[1]]
print(f"egitim {len(tr[0])} / degerlendirme {len(ev[0])} piksel\n")

BINS = [(1,2),(3,4),(5,7),(8,11),(12,99)]
hdr = "rejim        " + "  OA  " + "".join(f" | d={lo}-{hi if hi<99 else '+':<2}" for lo, hi in BINS)
print(hdr); print("-"*len(hdr))
results = {}
for name, T in [("native", T0)] + [(f"{kind}-{w}", None) for w in (3, 5, 7) for kind in ("boxcar", "oracle")]:
    if T is None:
        kind, w = name.split("-"); w = int(w)
        T = boxcar(T0, w) if kind == "boxcar" else oracle(T0, gt, w)
    Sig = wishart_fit(T, gt, tr, n_cls)
    pr = wishart_predict(T, Sig, ev)
    ok = pr == ev_true
    oa = ok.mean()
    row = f"{name:<12} {100*oa:5.2f}"
    per = []
    for lo, hi in BINS:
        m = (ev_dist >= lo) & (ev_dist <= hi)
        per.append(100*ok[m].mean())
        row += f" | {100*ok[m].mean():6.2f}"
    print(row)
    results[name] = (100*oa, per)
    del T

print("\n=== ORACLE - BOXCAR farki (ogrenilmis toplamanin TAVANI, puan) ===")
print("pencere      " + "  OA  " + "".join(f" | d={lo}-{hi if hi<99 else '+':<2}" for lo, hi in BINS))
for w in (3, 5, 7):
    bo, bp = results[f"boxcar-{w}"]; oo, op = results[f"oracle-{w}"]
    print(f"w={w:<11} {oo-bo:+5.2f}" + "".join(f" | {o-b:+6.2f}" for o, b in zip(op, bp)))
