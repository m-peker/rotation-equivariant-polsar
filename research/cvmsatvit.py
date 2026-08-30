"""
CV-MsAtViT (Alkhatib 2025, IJAEOG) -- orijinal Keras/cvnn kodunun PyTorch portu.

Kaynak: ../main_CV_MsAtViT.py , ../CoordAttention.py
Amac: (a) adil kiyas temeli, (b) YAYINLANMIS mimarinin de donmede coktugunu gostermek.

Sadakat notu: orijinal ViT bloğu gercek ve sanal parcalara AYRI MHA uyguluyor
(makalenin "kompleks" iddiasinin zayif noktasi). Portta bunu AYNEN koruyorum;
duzeltilmis Hermityen-ic-carpimli surum ayri bir varyant (attn="hermitian").
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def cart_relu(a,b):    return torch.relu(a), torch.relu(b)
def cart_gelu(a,b):    return F.gelu(a), F.gelu(b)
def cart_sigmoid(a,b): return torch.sigmoid(a), torch.sigmoid(b)

class CConv3d(nn.Module):
    def __init__(s,i,o,k):
        super().__init__()
        p=tuple(x//2 for x in k)
        s.r=nn.Conv3d(i,o,k,padding=p); s.i=nn.Conv3d(i,o,k,padding=p)
    def forward(s,a,b): return s.r(a)-s.i(b), s.r(b)+s.i(a)

class CConv2d(nn.Module):
    def __init__(s,i,o,k=3,p=1):
        super().__init__(); s.r=nn.Conv2d(i,o,k,padding=p); s.i=nn.Conv2d(i,o,k,padding=p)
    def forward(s,a,b): return s.r(a)-s.i(b), s.r(b)+s.i(a)

class CDense(nn.Module):
    def __init__(s,i,o,bias=True):
        super().__init__(); s.r=nn.Linear(i,o,bias=bias); s.i=nn.Linear(i,o,bias=bias)
    def forward(s,a,b): return s.r(a)-s.i(b), s.r(b)+s.i(a)

class CoordAttC(nn.Module):
    """CoordAttention.py'nin kompleks portu. Not: orijinalde BN trainable=False."""
    def __init__(s,c,h,w,reduction=4):
        super().__init__(); s.h,s.w=h,w
        mip=max(8,c//reduction)
        s.conv1=CConv2d(c,mip,1,0)
        s.bn_r=nn.BatchNorm2d(mip,affine=True); s.bn_i=nn.BatchNorm2d(mip,affine=True)
        for p in list(s.bn_r.parameters())+list(s.bn_i.parameters()): p.requires_grad_(False)
        s.ch=CConv2d(mip,c,1,0); s.cw=CConv2d(mip,c,1,0)
    @staticmethod
    def _act(a,b):
        # h-swish: x * relu6(x+3)/6  (kompleks carpim)
        tr=torch.clamp(a+3,0,6)/6; ti=torch.clamp(b+3,0,6)/6
        return a*tr-b*ti, a*ti+b*tr
    def forward(s,a,b):
        xh_r=a.mean(3,keepdim=True); xh_i=b.mean(3,keepdim=True)       # (B,C,H,1)
        xw_r=a.mean(2,keepdim=True).permute(0,1,3,2)                   # (B,C,W,1)
        xw_i=b.mean(2,keepdim=True).permute(0,1,3,2)
        yr=torch.cat([xh_r,xw_r],2); yi=torch.cat([xh_i,xw_i],2)
        yr,yi=s.conv1(yr,yi); yr=s.bn_r(yr); yi=s.bn_i(yi); yr,yi=s._act(yr,yi)
        hr,wr=torch.split(yr,[s.h,s.w],2); hi,wi=torch.split(yi,[s.h,s.w],2)
        wr=wr.permute(0,1,3,2); wi=wi.permute(0,1,3,2)
        ahr,ahi=cart_sigmoid(*s.ch(hr,hi)); awr,awi=cart_sigmoid(*s.cw(wr,wi))
        # a_h * a_w (kompleks carpim), sonra x * (.)
        gr=ahr*awr-ahi*awi; gi=ahr*awi+ahi*awr
        return a*gr-b*gi, a*gi+b*gr

class Block(nn.Module):
    def __init__(s,d,heads,attn="split"):
        super().__init__(); s.attn=attn
        s.n1r=nn.LayerNorm(d); s.n1i=nn.LayerNorm(d)
        s.n2r=nn.LayerNorm(d); s.n2i=nn.LayerNorm(d)
        if attn=="split":
            s.ar=nn.MultiheadAttention(d,heads,dropout=0.1,batch_first=True)
            s.ai=nn.MultiheadAttention(d,heads,dropout=0.1,batch_first=True)
        else:
            s.q=CDense(d,d); s.k=CDense(d,d); s.v=CDense(d,d); s.o=CDense(d,d); s.h=heads
        s.m1=CDense(d,d*2); s.m2=CDense(d*2,d); s.do=nn.Dropout(0.1)
    def forward(s,xr,xi):
        ar,ai=s.n1r(xr),s.n1i(xi)
        if s.attn=="split":
            or_,_=s.ar(ar,ar,ar); oi,_=s.ai(ai,ai,ai)
        else:
            # Hermityen ic carpim: skor = Re(q^H k)/sqrt(d) -> reel ve sanali BAGLAR
            qr,qi=s.q(ar,ai); kr,ki=s.k(ar,ai); vr,vi=s.v(ar,ai)
            B,T,D=qr.shape; hd=D//s.h
            rs=lambda t: t.view(B,T,s.h,hd).transpose(1,2)
            qr,qi,kr,ki,vr,vi=map(rs,(qr,qi,kr,ki,vr,vi))
            sc=(qr@kr.transpose(-1,-2)+qi@ki.transpose(-1,-2))/np.sqrt(hd)
            w=sc.softmax(-1)
            or_=(w@vr).transpose(1,2).reshape(B,T,D); oi=(w@vi).transpose(1,2).reshape(B,T,D)
            or_,oi=s.o(or_,oi)
        xr,xi=xr+or_, xi+oi
        hr,hi=s.n2r(xr),s.n2i(xi)
        hr,hi=cart_gelu(*s.m1(hr,hi)); hr,hi=s.do(hr),s.do(hi)
        hr,hi=cart_gelu(*s.m2(hr,hi)); hr,hi=s.do(hr),s.do(hi)
        return xr+hr, xi+hi

class CVMsAtViT(nn.Module):
    def __init__(s,ncl,cin=6,ws=15,patch=3,dim=32,heads=4,layers=4,mlp=(1024,512),attn="split"):
        super().__init__()
        s.ws=ws; s.patch=patch; s.np=(ws//patch)**2
        s.p1=nn.ModuleList([CConv3d(1,8,(3,3,1)),CConv3d(8,8,(3,3,1))])
        s.p2=nn.ModuleList([CConv3d(1,8,(1,1,3)),CConv3d(8,8,(1,1,3))])
        s.p3=nn.ModuleList([CConv3d(1,8,(3,3,3)),CConv3d(8,8,(3,3,3))])
        s.red=CConv2d(24*cin,24)
        s.ca=CoordAttC(24,ws,ws,4)
        pd=patch*patch*24
        s.proj=CDense(pd,dim); s.pos=nn.Embedding(s.np,dim)
        s.blocks=nn.ModuleList([Block(dim,heads,attn) for _ in range(layers)])
        s.nfr=nn.LayerNorm(dim); s.nfi=nn.LayerNorm(dim)
        s.do=nn.Dropout(0.5)
        d=s.np*dim; s.h1=CDense(d,mlp[0]); s.h2=CDense(mlp[0],mlp[1]); s.do2=nn.Dropout(0.3)
        s.out=CDense(mlp[1],ncl)
    def forward(s,xr,xi):
        B=xr.shape[0]
        a=xr.permute(0,2,3,1).unsqueeze(1); b=xi.permute(0,2,3,1).unsqueeze(1)  # (B,1,H,W,6)
        outs=[]
        for path in (s.p1,s.p2,s.p3):
            u,v=a,b
            for cv in path: u,v=cart_relu(*cv(u,v))
            outs.append((u,v))
        u=torch.cat([o[0] for o in outs],1); v=torch.cat([o[1] for o in outs],1)  # (B,24,H,W,6)
        u=u.permute(0,2,3,4,1).reshape(B,s.ws,s.ws,-1).permute(0,3,1,2)
        v=v.permute(0,2,3,4,1).reshape(B,s.ws,s.ws,-1).permute(0,3,1,2)
        u,v=cart_relu(*s.red(u,v)); u,v=s.ca(u,v)
        p=s.patch
        f=lambda t: (t.unfold(2,p,p).unfold(3,p,p).permute(0,2,3,1,4,5).reshape(B,s.np,-1))
        u,v=f(u),f(v)
        u,v=s.proj(u,v)
        pe=s.pos(torch.arange(s.np,device=u.device))[None]
        u=u+pe                                        # orijinaldeki gibi: yalniz reel eksene
        for blk in s.blocks: u,v=blk(u,v)
        u,v=s.nfr(u),s.nfi(v)
        u=s.do(u.flatten(1)); v=s.do(v.flatten(1))
        u,v=cart_gelu(*s.h1(u,v)); u,v=s.do2(u),s.do2(v)
        u,v=cart_gelu(*s.h2(u,v)); u,v=s.do2(u),s.do2(v)
        u,v=s.out(u,v)
        return torch.sqrt(u**2+v**2+1e-9)
