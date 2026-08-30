"""
Deney 15 - ANA TABLO: 3 sahne x 4 model, iz-normalize+log-guc esdegisir girdi.

Protokol: sinif basina dengeli ~2000 toplam etiket (sahneler arasi ADIL ve
egitim maliyeti sabit):  flevoland 133x15, sanfran 400x5, ober 666x3
Degerlendirme: kalan tum etiketli pikseller (200K'da kirpilir).
TF32 KAPALI -> grup acilarindaki degismezlik birebir gorunsun.
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool
from equivariant import EqCVCNN_F
from cvmsatvit import CVMsAtViT

THETAS=[0,10,22.5,45]; ONG={0,22.5,45}
BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]
BUDGET={"flevoland":133,"sanfran":400,"ober":666}
SEEDS=(0,1)

class CVCNNb(nn.Module):
    def __init__(s,ncl,cin=7,w=1):
        super().__init__(); a,b,c=32*w,64*w,128*w
        s.c1=CConv(cin,a); s.c2=CConv(a,b); s.c3=CConv(b,c)
        s.f1=CLin(c*3*3,128*w); s.f2=CLin(128*w,ncl); s.do=nn.Dropout(0.3)
    def forward(s,xr,xi):
        xr,xi=crelu(*s.c1(xr,xi)); xr,xi=crelu(*s.c2(xr,xi)); xr,xi=cpool(xr,xi)
        xr,xi=crelu(*s.c3(xr,xi)); xr,xi=cpool(xr,xi)
        xr=s.do(xr.flatten(1)); xi=s.do(xi.flatten(1))
        xr,xi=crelu(*s.f1(xr,xi)); xr,xi=s.f2(xr,xi)
        return torch.sqrt(xr**2+xi**2+1e-9)

def sample(P,n,seed):
    rng=np.random.default_rng(1000+seed); tr=[]
    for k in range(P.ncl):
        idx=np.nonzero(P.y==k)[0]
        tr.append(rng.choice(idx,min(n,len(idx)),replace=False))
    tr=np.concatenate(tr); te=np.setdiff1d(np.arange(len(P.y)),tr)
    return tr, P.cap_eval(te)

def build(kind,ncl):
    return {"base":lambda: CVCNNb(ncl,cin=7),
            "eqf": lambda: EqCVCNN_F(ncl,cin=7,N=8),
            "msat":lambda: CVMsAtViT(ncl,cin=7)}[kind]()

def run(P,kind,aug,seed,epochs=120,bs=128):
    itr,ite=sample(P,BUDGET[P.scene],seed)
    Rt=torch.from_numpy(P.lr[itr]).cuda();Ct=torch.from_numpy(P.lc[itr]).cuda()
    Yt=torch.from_numpy(P.y[itr]).cuda()
    torch.manual_seed(seed); net=build(kind,P.ncl).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lf=nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm=torch.randperm(len(itr),device="cuda")
        for s in range(0,len(itr),bs):
            b=pm[s:s+bs]
            th=(torch.rand(len(b),device="cuda")*np.pi) if aug else None
            xr,xi=P.grab(Rt[b],Ct[b],th)
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    Re=torch.from_numpy(P.lr[ite]).cuda();Ce=torch.from_numpy(P.lc[ite]).cuda()
    accs=[];bnd=None;aa0=None
    for tdeg in THETAS:
        th=torch.tensor(np.deg2rad(tdeg),device="cuda",dtype=torch.float32) if tdeg else None
        pr=np.empty(len(ite),np.int64)
        with torch.no_grad():
            for s in range(0,len(ite),4096):
                xr,xi=P.grab(Re[s:s+4096],Ce[s:s+4096],th)
                pr[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
        t=P.y[ite]; ok=pr==t; accs.append(100*ok.mean())
        if tdeg==0:
            d=P.dist[P.lr[ite],P.lc[ite]]
            bnd=[100*ok[(d>=lo)&(d<=hi)].mean() if ((d>=lo)&(d<=hi)).sum()>0 else np.nan for lo,hi in BINS]
            aa0=100*np.mean([ok[t==k].mean() for k in range(P.ncl) if (t==k).sum()>0])
    npar=sum(p.numel() for p in net.parameters()); del net; torch.cuda.empty_cache()
    return accs,bnd,aa0,npar

ARMS=[("baseline no-aug","base",False),("baseline rot-AUG","base",True),
      ("EQ-fourier N=8","eqf",False),("CV-MsAtViT no-aug","msat",False)]
print("theta:"+"".join(f"{t:>9}" for t in THETAS)+"   (0/22.5/45 GRUP ICI)  TF32 kapali",flush=True)
ALL={}
for sc in ("flevoland","sanfran","ober"):
    P=MSPipe(sc,norm="equivariant",eval_cap=200_000)
    print(f"\n### {P.info()}   butce={BUDGET[sc]}/sinif ###",flush=True)
    for nm,k,aug in ARMS:
        if k=="msat" and sc!="flevoland": continue      # pahali: yalniz Flevoland'da
        A=[];B=[];AAv=[];t0=time.time()
        for sd in SEEDS:
            a,b,aa,npar=run(P,k,aug,sd); A.append(a);B.append(b);AAv.append(aa)
        A=np.array(A);B=np.array(B)
        ALL[(sc,nm)]=(A.mean(0),A.std(0),B.mean(0),float(np.mean(AAv)),npar)
        print(f"  {nm:<19}"+"".join(f"{m:9.2f}" for m in A.mean(0))+
              f"  AA={np.mean(AAv):6.2f} sinir={B.mean(0)[0]:6.2f} p={npar:>8} ({time.time()-t0:.0f}s)",flush=True)
    del P; torch.cuda.empty_cache()

print("\n\n================ OZET ================")
print(f"{'sahne':<11}{'model':<19}{'th=0':>8}{'AA':>7}{'grupici-min':>13}{'th=10':>8}{'cokus':>8}{'sinir':>8}{'params':>10}")
for (sc,nm),(m,sd,b,aa,npar) in ALL.items():
    ong=min(m[i] for i,t in enumerate(THETAS) if t in ONG)
    off=m[THETAS.index(10)]
    print(f"{sc:<11}{nm:<19}{m[0]:8.2f}{aa:7.2f}{ong:13.2f}{off:8.2f}{m[0]-m.min():8.2f}{b[0]:8.2f}{npar:10d}")
