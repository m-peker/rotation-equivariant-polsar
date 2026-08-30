"""
Polarimetrik donme grubuna ESDEGISIR kompleks-degerli ag.

Grup: G = {U(theta)}, U(a)U(b)=U(a+b), periyot pi  (dogrulandi: polsar_lib)
N yonelime ayriklastirinca G ~= Z_N (dongusel).

Mimari (Cohen&Welling G-CNN'in bu gruba tasinmasi):
  1) LIFT   : x -> [conv(U(th_n) x, W)]_{n=0..N-1}     ciktisi G uzerinde fonksiyon
              girdi U(th_m) ile donerse cikti n ekseninde DONGUSEL KAYAR
  2) GCONV  : y[n] = sum_m conv(x[m], W[(m-n) mod N])   kaymayla degisir (commute)
  3) POOL_G : n ekseninde max -> DEGISMEZ cikti

Parametreler yonelimler arasinda PAYLASILIR -> esdegisirlik bedava gelmez ama
veri artirimi gibi kapasite/ornek maliyeti odetmez. Test edilecek iddia bu.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from polsar_lib import rot6_torch

class CLift(nn.Module):
    """Girdiyi N yonelime yukselt + paylasimli kompleks konvolusyon."""
    def __init__(s, cin, cout, N=8, k=3):
        super().__init__(); s.N=N
        s.wr=nn.Parameter(torch.randn(cout,cin,k,k)*(2.0/(cin*k*k))**0.5)
        s.wi=nn.Parameter(torch.randn(cout,cin,k,k)*(2.0/(cin*k*k))**0.5)
        s.br=nn.Parameter(torch.zeros(cout)); s.bi=nn.Parameter(torch.zeros(cout))
        s.p=k//2
    def forward(s, xr, xi):                       # (B,6,H,W)
        outr=[]; outi=[]
        for n in range(s.N):
            th=torch.tensor(n*np.pi/s.N, device=xr.device, dtype=xr.dtype)
            ar,ai=rot6_torch(xr,xi,th)
            # kompleks konv: (Wr+iWi)*(ar+i ai)
            r=F.conv2d(ar,s.wr,s.br,padding=s.p)-F.conv2d(ai,s.wi,None,padding=s.p)
            i=F.conv2d(ai,s.wr,s.bi,padding=s.p)+F.conv2d(ar,s.wi,None,padding=s.p)
            outr.append(r); outi.append(i)
        return torch.stack(outr,1), torch.stack(outi,1)      # (B,N,C,H,W)

class CGConv(nn.Module):
    """Dongusel grup konvolusyonu: y[n] = sum_m conv(x[m], W[(m-n) mod N])."""
    def __init__(s, cin, cout, N=8, k=3):
        super().__init__(); s.N=N; s.ci=cin; s.co=cout; s.k=k; s.p=k//2
        sc=(2.0/(cin*N*k*k))**0.5
        s.wr=nn.Parameter(torch.randn(N,cout,cin,k,k)*sc)
        s.wi=nn.Parameter(torch.randn(N,cout,cin,k,k)*sc)
        s.br=nn.Parameter(torch.zeros(cout)); s.bi=nn.Parameter(torch.zeros(cout))
        idx=torch.zeros(N,N,dtype=torch.long)
        for n in range(N):
            for m in range(N): idx[n,m]=(m-n)%N
        s.register_buffer("idx",idx)
    def _big(s,w):
        # (N,co,ci,k,k) -> (N*co, N*ci, k,k) blok yapisi
        W=w[s.idx]                                   # (N,N,co,ci,k,k)
        W=W.permute(0,2,1,3,4,5).reshape(s.N*s.co, s.N*s.ci, s.k, s.k)
        return W
    def forward(s, xr, xi):                          # (B,N,C,H,W)
        B,N,C,H,W_=xr.shape
        ar=xr.reshape(B,N*C,H,W_); ai=xi.reshape(B,N*C,H,W_)
        Wr=s._big(s.wr); Wi=s._big(s.wi)
        br=s.br.repeat(s.N); bi=s.bi.repeat(s.N)
        r=F.conv2d(ar,Wr,br,padding=s.p)-F.conv2d(ai,Wi,None,padding=s.p)
        i=F.conv2d(ai,Wr,bi,padding=s.p)+F.conv2d(ar,Wi,None,padding=s.p)
        return r.reshape(B,s.N,s.co,H,W_), i.reshape(B,s.N,s.co,H,W_)

def gcrelu(a,b): return torch.relu(a), torch.relu(b)
def gpool_sp(a,b):
    B,N,C,H,W=a.shape
    f=lambda t: F.avg_pool2d(t.reshape(B*N,C,H,W),2).reshape(B,N,C,H//2,W//2)
    return f(a),f(b)
def pool_group(a,b):
    """G ekseninde genlik-max -> DEGISMEZ (kompleks degeri korur)."""
    mag=torch.sqrt(a**2+b**2+1e-12)
    j=mag.argmax(1,keepdim=True)
    return a.gather(1,j).squeeze(1), b.gather(1,j).squeeze(1)

class CLin(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.r=nn.Linear(i,o); s.i=nn.Linear(i,o)
    def forward(s,xr,xi): return s.r(xr)-s.i(xi), s.r(xi)+s.i(xr)

class EqCVCNN(nn.Module):
    def __init__(s, ncl, cin=6, N=8, w=1):
        super().__init__(); s.N=N
        a,b,c=16*w,24*w,32*w
        s.lift=CLift(cin,a,N); s.g1=CGConvFFT(a,b,N); s.g2=CGConvFFT(b,c,N)
        s.f1=CLin(c*3*3,128); s.f2=CLin(128,ncl); s.do=nn.Dropout(0.3)
    def features(s,xr,xi):
        xr,xi=gcrelu(*s.lift(xr,xi))
        xr,xi=gcrelu(*s.g1(xr,xi)); xr,xi=gpool_sp(xr,xi)      # 15->7
        xr,xi=gcrelu(*s.g2(xr,xi)); xr,xi=gpool_sp(xr,xi)      # 7->3
        return xr,xi                                            # (B,N,C,3,3)
    def forward(s,xr,xi):
        xr,xi=s.features(xr,xi)
        xr,xi=pool_group(xr,xi)                                 # DEGISMEZ
        xr=s.do(xr.flatten(1)); xi=s.do(xi.flatten(1))
        xr,xi=gcrelu(*s.f1(xr,xi)); xr,xi=s.f2(xr,xi)
        return torch.sqrt(xr**2+xi**2+1e-9)

# ---------------------------------------------------------------------------
# Zengin degismez okuma: grup ekseni uzerinde dongusel Fourier genlikleri.
# x[n] -> X[k] = sum_n x[n] e^{-2pi i kn/N};  girdi U_m ile donerse x dongusel
# kayar -> X[k] yalnizca e^{-2pi i km/N} faz carpani alir -> |X[k]| DEGISMEZ.
# k=0 ortalama (max'tan daha kararli), k>=1 orbit'in sekil bilgisini tasir:
# tam degismezlik korunur ama betimleyici cok daha zengin olur.
# ---------------------------------------------------------------------------
def pool_group_fourier(a, b, K=3):
    """(B,N,C,H,W) -> (B,K*C,H,W) gercek, degismez."""
    B,N,C,H,W = a.shape
    z = torch.complex(a, b)                       # (B,N,C,H,W)
    Z = torch.fft.fft(z, dim=1)                   # grup ekseni uzerinde DFT
    K = min(K, N)
    return Z[:, :K].abs().reshape(B, K*C, H, W)   # |X[k]| : degismez

class EqCVCNN_F(nn.Module):
    """Fourier-degismez okumali surum."""
    def __init__(s, ncl, cin=6, N=8, w=1, K=3):
        super().__init__(); s.N=N; s.K=K
        a,b,c = 16*w, 24*w, 32*w
        s.lift=CLift(cin,a,N); s.g1=CGConvFFT(a,b,N); s.g2=CGConvFFT(b,c,N)
        s.f1=nn.Linear(K*c*3*3, 128); s.f2=nn.Linear(128, ncl); s.do=nn.Dropout(0.3)
    def forward(s,xr,xi):
        xr,xi=gcrelu(*s.lift(xr,xi))
        xr,xi=gcrelu(*s.g1(xr,xi)); xr,xi=gpool_sp(xr,xi)
        xr,xi=gcrelu(*s.g2(xr,xi)); xr,xi=gpool_sp(xr,xi)
        v=pool_group_fourier(xr,xi,s.K).flatten(1)
        return s.f2(torch.relu(s.f1(s.do(v))))


# ---------------------------------------------------------------------------
# FFT-diagonalised group convolution.
#
# CGConv evaluates  y[n] = sum_k conv(x[(n+k) mod N], W[k])  by materialising an
# (N*Cout, N*Cin) block-circulant kernel, so it performs N^2 channel-pairs of
# spatial convolution. The sum is a cyclic correlation along the group axis, and
# a DFT along that axis diagonalises it:
#
#     yhat[f] = What[f] * xhat[f],     What[f] = sum_k W[k] e^{+2 pi i f k / N}
#
# which is N independent Cin->Cout convolutions -- an N-fold reduction. For N=8
# that is the difference between 293 and ~40 MMACs, and it is what makes the
# discrete variant practical to train.
# ---------------------------------------------------------------------------
class CGConvFFT(nn.Module):
    def __init__(s, cin, cout, N=8, k=3):
        super().__init__()
        s.N, s.ci, s.co, s.k, s.p = N, cin, cout, k, k // 2
        sc = (2.0 / (cin * N * k * k)) ** 0.5
        s.wr = nn.Parameter(torch.randn(N, cout, cin, k, k) * sc)
        s.wi = nn.Parameter(torch.randn(N, cout, cin, k, k) * sc)
        s.br = nn.Parameter(torch.zeros(cout)); s.bi = nn.Parameter(torch.zeros(cout))

    def forward(s, xr, xi):
        B, N, C, H, W_ = xr.shape
        X = torch.fft.fft(torch.complex(xr, xi), dim=1)              # (B,N,C,H,W)
        Wf = torch.fft.ifft(torch.complex(s.wr, s.wi), dim=0) * s.N  # e^{+i} sign
        # N independent complex convolutions, batched as a grouped convolution
        Xr = X.real.reshape(B, N * C, H, W_).contiguous()
        Xi = X.imag.reshape(B, N * C, H, W_).contiguous()
        Kr = Wf.real.reshape(N * s.co, C, s.k, s.k).contiguous()
        Ki = Wf.imag.reshape(N * s.co, C, s.k, s.k).contiguous()
        yr = F.conv2d(Xr, Kr, None, padding=s.p, groups=N) \
           - F.conv2d(Xi, Ki, None, padding=s.p, groups=N)
        yi = F.conv2d(Xi, Kr, None, padding=s.p, groups=N) \
           + F.conv2d(Xr, Ki, None, padding=s.p, groups=N)
        Y = torch.complex(yr, yi).reshape(B, N, s.co, H, W_)
        y = torch.fft.ifft(Y, dim=1)
        return (y.real + s.br.view(1, 1, -1, 1, 1),
                y.imag + s.bi.view(1, 1, -1, 1, 1))
