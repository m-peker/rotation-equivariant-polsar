"""
Deney 17 - DURUST PROTOKOL: sizinti sifir oldugunda problem ne kadar cozulmus?

AIR-PolSAR-Seg v1 yalniz GENLIK cikti (faz yok) -> kompleks yontem orada
uygulanamaz. Ama doymamisliga ulasmak icin yeni veriye gerek yok: standart
rastgele split'te test piksellerinin %84.2'sinin patch'i egitim patch'iyle
ORTUSUYOR. Sizintiyi sifirlayinca gercek tavan ortaya cikiyor.

Tasarim (temiz izolasyon): TEST KUMESI ve ETIKET BUTCESI iki kolda AYNI,
degisen tek sey egitim piksellerinin teste uzamsal yakinligi.
  T = blok+tampon ayrik test maskesi (sabit)
  A) SIZINTILI : N/sinif, T'ye komsu olabilen her yerden
  B) TEMIZ     : N/sinif, yalniz ayrik train bolgesinden (>=16 px uzak)
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool
from equivariant import EqCVCNN_F
from disjoint import block_buffer_split, verify_no_overlap

BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]
BUDGET={"flevoland":133,"sanfran":400,"ober":666}
BLOCK={"flevoland":96,"sanfran":128,"ober":160}

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

def pick(gt,mask,ncl,n,seed):
    rng=np.random.default_rng(seed); rs=[];cs=[]
    for k in range(1,ncl+1):
        r,c=np.nonzero((gt==k)&mask)
        if len(r)==0: continue
        p=rng.choice(len(r),min(n,len(r)),replace=False); rs.append(r[p]); cs.append(c[p])
    return np.concatenate(rs), np.concatenate(cs)

def train_eval(P,kind,tr_rc,te_rc,seed,epochs=120):
    r,c=tr_rc
    Rt=torch.from_numpy(r).cuda();Ct=torch.from_numpy(c).cuda()
    Yt=torch.from_numpy(P.gt[r,c]-1).cuda()
    torch.manual_seed(seed)
    net=(EqCVCNN_F(P.ncl,cin=7,N=8) if kind=="eq" else CVCNNb(P.ncl,cin=7)).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lf=nn.CrossEntropyLoss(); net.train(); n=len(r)
    for ep in range(epochs):
        pm=torch.randperm(n,device="cuda")
        for s in range(0,n,128):
            b=pm[s:s+128]; xr,xi=P.grab(Rt[b],Ct[b])
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
        sch.step()
    net.eval(); er,ec=te_rc
    Re=torch.from_numpy(er).cuda();Ce=torch.from_numpy(ec).cuda()
    pr=np.empty(len(er),np.int64)
    with torch.no_grad():
        for s in range(0,len(er),4096):
            xr,xi=P.grab(Re[s:s+4096],Ce[s:s+4096])
            pr[s:s+4096]=net(xr,xi).argmax(1).cpu().numpy()
    ok=pr==(P.gt[er,ec]-1); d=P.dist[er,ec]
    bnd=[100*ok[(d>=lo)&(d<=hi)].mean() if ((d>=lo)&(d<=hi)).sum()>0 else np.nan for lo,hi in BINS]
    del net; torch.cuda.empty_cache()
    return 100*ok.mean(), bnd

res={}
for sc in ("flevoland","sanfran","ober"):
    P=MSPipe(sc,norm="equivariant",eval_cap=150_000)
    print(f"\n### {sc}  blok={BLOCK[sc]} tampon=8  butce={BUDGET[sc]}/sinif ###",flush=True)
    for seed in (0,1):
        trm,tem,B=block_buffer_split(P.gt,block=BLOCK[sc],test_frac=0.7,seed=seed)
        te=np.nonzero(tem); leak=(P.gt>0)&(~tem)
        rng=np.random.default_rng(seed)
        if len(te[0])>150_000:
            s=rng.choice(len(te[0]),150_000,replace=False); te=(te[0][s],te[1][s])
        for kind in ("base","eq"):
            for arm,msk in (("A-SIZINTILI",leak),("B-TEMIZ",trm)):
                tr=pick(P.gt,msk,P.ncl,BUDGET[sc],seed)
                m=np.zeros(P.gt.shape,bool); m[tr[0],tr[1]]=True
                ov=verify_no_overlap(m,tem)
                oa,bnd=train_eval(P,kind,tr,te,seed)
                res.setdefault((sc,kind,arm),[]).append((oa,bnd[0]))
                print(f"   seed{seed} {kind:<5}{arm:<12} OA={oa:6.2f} sinir={bnd[0]:6.2f}  (ortusen test px={ov})",flush=True)
    del P; torch.cuda.empty_cache()

print("\n\n=== SIZINTININ SISIRDIGI PUAN (ayni test, ayni butce) ===")
print(f"{'sahne':<11}{'model':<7}{'A-sizintili':>13}{'B-temiz':>10}{'SISME':>8}{'sinir sisme':>13}")
for sc in ("flevoland","sanfran","ober"):
    for kind in ("base","eq"):
        a=np.array(res[(sc,kind,"A-SIZINTILI")]); b=np.array(res[(sc,kind,"B-TEMIZ")])
        print(f"{sc:<11}{kind:<7}{a[:,0].mean():13.2f}{b[:,0].mean():10.2f}"
              f"{a[:,0].mean()-b[:,0].mean():+8.2f}{a[:,1].mean()-b[:,1].mean():+13.2f}")
