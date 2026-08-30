"""
Donme-ESDEGISIR ve dinamik-aralik-guvenli normalizasyon.

Temel gozlem: grup etkisi U(theta) DOGRUSALDIR, dolayisiyla
    x -> s(x) * x     ile   s = DEGISMEZ bir skaler
donusumu grupla YER DEGISTIRIR:  U(s*x) = s*U(x).
Iz  span = T11+T22+T33  degismezdir (U T U^T izini korur).

Bu yuzden koherans matrisinin SEKLINI genliginden ayiriyoruz:
  kanal 0..5 : T / span          -> PSD geregi sinirli ( |.| <= 1 ), esdegisir
  kanal 6    : (log span - med)/mad  -> DEGISMEZ, gucu tasir, agir kuyruk log ile ezilir

Boylece hem tum girdiler O(1) (float32 hassasiyeti korunur) hem esdegisirlik tam.
Onceki iki deneme basarisiz oldu ve neden onemli:
  - kanal-basina z-score : U(theta) ile yer degistirmiyor  -> esdegisirligi bozuyor
  - MAD tabanli olcek    : agir kuyrugu sisiriyor (std 51, |max| 1e4) -> float32 eriyor

Irrep yapisi (merkezleme yalnizca degismez bilesenlere uygulanabilir):
  agirlik 0: T11/span , (T22+T33)/2span , Im T23/span , log span
  agirlik 2: (T12, T13)/span
  agirlik 4: ((T22-T33)/2span , Re T23/span)
"""
import numpy as np, torch

def _rob(x):
    m=float(np.median(x)); s=float(np.median(np.abs(x-m)))*1.4826+1e-12
    return m,s

class EqNorm:
    n_ch = 7
    def __init__(s, Xraw):
        T11=Xraw[...,0].real.astype(np.float64); T22=Xraw[...,1].real.astype(np.float64)
        T33=Xraw[...,2].real.astype(np.float64)
        span=np.clip(T11+T22+T33, 1e-20, None)
        s.mls, s.sls = _rob(np.log(span).ravel())          # log-guc: degismez kanal
        a=(T11/span); u=(T22+T33)/(2*span); z=(Xraw[...,5].imag.astype(np.float64)/span)
        s.ma,_ = _rob(a.ravel()); s.mu,_ = _rob(u.ravel()); s.mz,_ = _rob(z.ravel())
        # iz-normalize edildikten sonra tum bilesenler O(1); ek olcek gereksiz
    def to(s, dev):
        for k in ("mls","sls","ma","mu","mz"):
            setattr(s, k+"_t", torch.tensor(getattr(s,k), device=dev, dtype=torch.float32))
        return s
    def __call__(s, xr, xi):
        """(B,6,H,W) ham -> (B,7,H,W) esdegisir normalize (kanal 6 DEGISMEZ)."""
        span = torch.clamp(xr[:,0]+xr[:,1]+xr[:,2], min=1e-20)
        inv  = 1.0/span
        a = xr[:,0]*inv - s.ma_t
        u = (xr[:,1]+xr[:,2])*0.5*inv - s.mu_t
        v = (xr[:,1]-xr[:,2])*0.5*inv
        w = xr[:,5]*inv
        z = xi[:,5]*inv - s.mz_t
        p0,p1 = xr[:,3]*inv, xr[:,4]*inv
        q0,q1 = xi[:,3]*inv, xi[:,4]*inv
        L = (torch.log(span) - s.mls_t)/s.sls_t          # degismez guc kanali
        Z = torch.zeros_like(a)
        return (torch.stack([a, u+v, u-v, p0, p1, w, L], 1),
                torch.stack([Z, Z,   Z,   q0, q1, z, Z], 1))

def rot7_torch(xr, xi, th):
    """rot6 ile ayni, ama 7. kanal (log-guc) DEGISMEZ oldugu icin dokunulmaz."""
    from polsar_lib import rot6_torch
    a,b = rot6_torch(xr[:,:6], xi[:,:6], th)
    return torch.cat([a, xr[:,6:]],1), torch.cat([b, xi[:,6:]],1)
