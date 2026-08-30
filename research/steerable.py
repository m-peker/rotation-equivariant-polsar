"""
STEERABLE polarimetrik-donme esdegisir ag  --  SUREKLI theta icin TAM esdegisir.

Ayrik G-CNN (N yonelim) yalnizca theta = m*pi/N noktalarinda tamdir; aradaki
acilarda 2-3 puan kaybettiriyor. Bunun yerine grubun KENDI harmonik tabanini
kullaniyoruz.

Hermityen 3x3 matrisler 9-boyutlu REEL uzay; U(theta) etkisi ayrisiyor:
  agirlik 0 (degismez, 3 reel) : T11 , u=(T22+T33)/2 , Im T23      [+ log span]
  agirlik 2 (2 duzlem, 4 reel) : (Re T12, Re T13) , (Im T12, Im T13)
  agirlik 4 (1 duzlem, 2 reel) : ( (T22-T33)/2 , Re T23 )
Her duzlem bir kompleks sayi olarak yazilirsa:  z_k -> e^{-i k theta} z_k
  z_a = ReT12 + i ReT13 ,  z_b = ImT12 + i ImT13   (k=2)
  z_c = (T22-T33)/2 + i ReT23                       (k=4)
Dogrulama: (p0,p1) -> (c p0 + s p1, -s p0 + c p1) , c=cos2t, s=sin2t
           z = p0 + i p1  =>  z' = (c - i s) z = e^{-2i theta} z   OK

Esdegisir islemler:
  - uzamsal konvolusyon: her alan icinde, sabit (kompleks) cekirdek -> faz disari cikar
  - agirliklar arasi bilgi akisi: Clebsch-Gordan carpimlari
        z2 * z2        -> k=4
        z4 * conj(z2)  -> k=2
        |z2|^2, |z4|^2 -> k=0
  - dogrusalsizlik: k=0 icin serbest; k!=0 icin DEGISMEZ kapi ile carpim (fazi korur)
  - degismez okuma: k=0 alanlari, |z2|, |z4| ve Re/Im( z4 * conj(z2^2) )  [4-4=0]
    Sonuncusu agirlik-2 ve agirlik-4 bilesenleri arasindaki BAGIL fazi tasir:
    fiziksel olarak yonelim-duyarli ama donmeden bagimsiz bilgi.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def cmul(ar,ai,br,bi):  return ar*br-ai*bi, ar*bi+ai*br
def cmulc(ar,ai,br,bi): return ar*br+ai*bi, ai*br-ar*bi      # a * conj(b)

class CConv(nn.Module):
    def __init__(s,i,o,k=3):
        super().__init__(); p=k//2
        s.r=nn.Conv2d(i,o,k,padding=p,bias=False); s.i=nn.Conv2d(i,o,k,padding=p,bias=False)
    def forward(s,a,b): return s.r(a)-s.i(b), s.r(b)+s.i(a)

def decompose(xr, xi):
    """(B,6,H,W) HAM T3 -> (w0[B,4,H,W] reel, w2[B,2,H,W] kompleks, w4[B,1,H,W] kompleks).
    Iz-normalize + log-guc: iz DEGISMEZ oldugu icin bolme esdegisirligi bozmaz."""
    T11,T22,T33 = xr[:,0],xr[:,1],xr[:,2]
    span = torch.clamp(T11+T22+T33, min=1e-20); inv = 1.0/span
    w0 = torch.stack([T11*inv, (T22+T33)*0.5*inv, xi[:,5]*inv, torch.log(span)], 1)
    z2r = torch.stack([xr[:,3], xi[:,3]],1)*inv[:,None]      # Re T12 , Im T12
    z2i = torch.stack([xr[:,4], xi[:,4]],1)*inv[:,None]      # Re T13 , Im T13
    z4r = ((T22-T33)*0.5*inv)[:,None]
    z4i = (xr[:,5]*inv)[:,None]
    return w0, (z2r,z2i), (z4r,z4i)

class SteerLift(nn.Module):
    """Ham alanlari (w0:4, z2:2, z4:1) ortak kanal sayisina yukseltir.
    1x1 (kompleks) konvolusyon her alanin KENDI icinde kaldigi icin esdegisir.
    w0 uzerindeki BatchNorm da esdegisirligi BOZMAZ: w0 zaten DEGISMEZ oldugundan
    batch istatistikleri donmeyle degismez. log(span) kanalinin olcegini duzeltir."""
    def __init__(s, c0, c, w0_m=None, w0_s=None, s2=1.0, s4=1.0):
        super().__init__()
        # SABIT, sahneden turetilen normalizasyon (BatchNorm degil: batch'e
        # bagimlilik olmasin, esdegisirlik argumani kosulsuz kalsin).
        s.register_buffer("m0", torch.zeros(1,4,1,1) if w0_m is None else torch.tensor(w0_m,dtype=torch.get_default_dtype()).view(1,4,1,1))
        s.register_buffer("v0", torch.ones (1,4,1,1) if w0_s is None else torch.tensor(w0_s,dtype=torch.get_default_dtype()).view(1,4,1,1))
        s.register_buffer("q2", torch.tensor(float(s2),dtype=torch.get_default_dtype()))
        s.register_buffer("q4", torch.tensor(float(s4),dtype=torch.get_default_dtype()))
        s.p0=nn.Conv2d(4,c0,1); s.p2=CConv(2,c,1); s.p4=CConv(1,c,1)
    def forward(s,w0,z2,z4):
        # DEGISMEZ kanallari kirp: sifir-dolgu bolgeleri span=0 -> log(span)~-46
        # aykiri degeri uretiyor. Degismez kanali kirpmak degismezligi BOZMAZ.
        w0=torch.clamp((w0-s.m0)/s.v0, -8.0, 8.0)
        z2=(z2[0]/s.q2, z2[1]/s.q2); z4=(z4[0]/s.q4, z4[1]/s.q4)
        return s.p0(w0), s.p2(*z2), s.p4(*z4)

class SteerLayer(nn.Module):
    """CG karisimi + alan-ici konvolusyon + DEGISMEZ kapili dogrusalsizlik.
    Tum agirlik siniflari ayni kanal sayisina (c) sahip -> CG eslesmeleri elemanwise."""
    def __init__(s, c0i, ci, c0o, co, k=3):
        super().__init__()
        s.k0 = nn.Conv2d(c0i + 2*ci, c0o, k, padding=k//2)   # + |z2|^2 , |z4|^2
        s.k2 = CConv(2*ci, co, k)                            # z2 , z4*conj(z2)
        s.k4 = CConv(2*ci, co, k)                            # z4 , z2*z2
        s.g2 = nn.Conv2d(c0o, co, 1); s.g4 = nn.Conv2d(c0o, co, 1)
    def forward(s, w0, z2, z4):
        z2r,z2i = z2; z4r,z4i = z4
        n2 = z2r**2 + z2i**2
        n4 = z4r**2 + z4i**2
        h0 = F.relu(s.k0(torch.cat([w0, n2, n4],1)))
        cg4r, cg4i = cmul (z2r,z2i,z2r,z2i)                  # k = 2+2 = 4
        cg2r, cg2i = cmulc(z4r,z4i,z2r,z2i)                  # k = 4-2 = 2
        h2r,h2i = s.k2(torch.cat([z2r,cg2r],1), torch.cat([z2i,cg2i],1))
        h4r,h4i = s.k4(torch.cat([z4r,cg4r],1), torch.cat([z4i,cg4i],1))
        g2 = torch.sigmoid(s.g2(h0)); g4 = torch.sigmoid(s.g4(h0))
        return h0, (h2r*g2, h2i*g2), (h4r*g4, h4i*g4)

def pool(w0, z2, z4):
    f=lambda t: F.avg_pool2d(t,2)
    return f(w0), (f(z2[0]),f(z2[1])), (f(z4[0]),f(z4[1]))

def invariants(w0, z2, z4):
    """Tam DEGISMEZ okuma (|z| ve BAGIL FAZ)."""
    z2r,z2i=z2; z4r,z4i=z4
    m2 = torch.sqrt(z2r**2+z2i**2+1e-12)
    m4 = torch.sqrt(z4r**2+z4i**2+1e-12)
    sqr,sqi = cmul(z2r,z2i,z2r,z2i)               # z2^2 : k=4
    xr,xi   = cmulc(z4r,z4i,sqr,sqi)              # z4 * conj(z2^2) : k=0
    d = (torch.sqrt(sqr**2+sqi**2+1e-12)*m4)+1e-12
    return torch.cat([w0, m2, m4, xr/d, xi/d], 1)

def field_stats(Xraw):
    """Sahneden SABIT normalizasyon sabitleri. Hepsi DEGISMEZ istatistikler."""
    T11=Xraw[...,0].real.astype(np.float64); T22=Xraw[...,1].real.astype(np.float64)
    T33=Xraw[...,2].real.astype(np.float64)
    span=np.clip(T11+T22+T33,1e-20,None); inv=1.0/span
    ch=[T11*inv,(T22+T33)*0.5*inv,Xraw[...,5].imag.astype(np.float64)*inv,np.log(span)]
    m=[float(np.median(x)) for x in ch]
    v=[float(np.median(np.abs(x-mm)))*1.4826+1e-9 for x,mm in zip(ch,m)]
    z2=np.sqrt((Xraw[...,3].real**2+Xraw[...,4].real**2)*inv**2)
    z4=np.sqrt((((T22-T33)*0.5)**2+Xraw[...,5].real**2)*inv**2)
    return m, v, float(np.median(z2))+1e-9, float(np.median(z4))+1e-9

class SteerNet(nn.Module):
    def __init__(s, ncl, c=24, c0=16, stats=None):
        super().__init__()
        s.lift=SteerLift(c0,c,*(stats if stats is not None else (None,None,1.0,1.0)))
        s.l1=SteerLayer(c0,c,c0,c); s.l2=SteerLayer(c0,c,c0,c); s.l3=SteerLayer(c0,c,c0,c)
        nin=(c0 + 4*c)*3*3
        s.head=nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                             nn.Linear(nin,128), nn.ReLU(), nn.Linear(128,ncl))
    def forward(s, xr, xi):
        w0,z2,z4 = decompose(xr,xi)
        w0,z2,z4 = s.lift(w0,z2,z4)
        w0,z2,z4 = s.l1(w0,z2,z4)
        w0,z2,z4 = s.l2(w0,z2,z4); w0,z2,z4 = pool(w0,z2,z4)   # 15->7
        w0,z2,z4 = s.l3(w0,z2,z4); w0,z2,z4 = pool(w0,z2,z4)   # 7->3
        return s.head(invariants(w0,z2,z4))
