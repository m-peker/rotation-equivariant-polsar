"""
Deney 12 - Rastgele split'in sisirdigi puan tam olarak kac?

Tasarim (temiz izolasyon): TEST KUMESI ve ETIKET BUTCESI iki kolda AYNI.
Degisen tek sey egitim piksellerinin test'e uzamsal YAKINLIGI.

  T  = blok+tampon ayrik test maskesi (sabit)
  A) SIZINTILI : N etiket/sinif, T'ye komsu olabilen her yerden (tampon dahil)
  B) TEMIZ     : N etiket/sinif, yalnizca ayrik train bolgesinden (>=16 px uzak)
  ikisi de T uzerinde degerlendirilir -> fark = SAF SIZINTI etkisi

Ek olarak literaturun bildirdigi sayi (rastgele test kumesi) de raporlanir.
"""
import numpy as np, torch, torch.nn as nn, time, sys
sys.path.insert(0,'.')
from data_pipe import Pipe
from polsar_lib import CVCNN
from equivariant import EqCVCNN_F
from disjoint import block_buffer_split, verify_no_overlap

P=Pipe(); gt=P.gt; ncl=P.ncl; dist=P.dist
BINS=[(1,2),(3,4),(5,7),(8,11),(12,99)]
N_PER=50; BLOCK=96

def train_eval(tr_rc, te_rc, cls, seed, epochs=120, N=8):
    r,c = tr_rc
    Rt=torch.from_numpy(r).cuda(); Ct=torch.from_numpy(c).cuda()
    Yt=torch.from_numpy(gt[r,c]-1).cuda()
    torch.manual_seed(seed)
    net=(EqCVCNN_F(ncl,N=N) if cls=="eq" else CVCNN(ncl)).cuda()
    opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.CrossEntropyLoss(); net.train()
    n=len(r)
    for ep in range(epochs):
        pm=torch.randperm(n,device="cuda")
        for s in range(0,n,128):
            b=pm[s:s+128]; xr,xi=P.grab(Rt[b],Ct[b])
            opt.zero_grad(); lf(net(xr,xi),Yt[b]).backward(); opt.step()
    net.eval()
    er,ec = te_rc
    Re=torch.from_numpy(er).cuda(); Ce=torch.from_numpy(ec).cuda()
    pr=np.empty(len(er),np.int64)
    with torch.no_grad():
        for s in range(0,len(er),2048):
            xr,xi=P.grab(Re[s:s+2048],Ce[s:s+2048])
            pr[s:s+2048]=net(xr,xi).argmax(1).cpu().numpy()
    ok = pr == (gt[er,ec]-1); d=dist[er,ec]
    bnd=[100*ok[(d>=lo)&(d<=hi)].mean() for lo,hi in BINS]
    del net; torch.cuda.empty_cache()
    return 100*ok.mean(), bnd

def pick(mask, n, seed):
    rng=np.random.default_rng(seed); rs=[];cs=[]
    for k in range(1,ncl+1):
        r,c=np.nonzero((gt==k)&mask)
        if len(r)==0: continue
        p=rng.choice(len(r),min(n,len(r)),replace=False); rs.append(r[p]); cs.append(c[p])
    return np.concatenate(rs), np.concatenate(cs)

print(f"protokol: blok={BLOCK}, tampon=8, {N_PER} etiket/sinif\n", flush=True)
res={}
for seed in (0,1,2):
    trm, tem, B = block_buffer_split(gt, block=BLOCK, test_frac=0.7, seed=seed)
    te_rc = np.nonzero(tem)
    leak_mask = (gt>0) & (~tem)          # T disindaki HER yer (tampon dahil) -> sizintili
    for cls in ("std","eq"):
        for arm, msk in (("A-SIZINTILI",leak_mask), ("B-TEMIZ",trm)):
            tr_rc = pick(msk, N_PER, seed)
            m=np.zeros(gt.shape,bool); m[tr_rc[0],tr_rc[1]]=True
            ov = verify_no_overlap(m, tem)
            oa,bnd = train_eval(tr_rc, te_rc, cls, seed)
            res.setdefault((cls,arm),[]).append((oa,bnd[0],bnd[-1]))
            print(f"  seed{seed} {cls:<4} {arm:<12} OA={oa:6.2f}  sinir={bnd[0]:6.2f} ic={bnd[-1]:6.2f}  (ortusen test px: {ov})", flush=True)

print("\n=== SONUC: ayni test kumesi, ayni butce, tek fark uzamsal yakinlik ===")
print(f"{'model':<8}{'kol':<14}{'OA':>8}{'sinir':>8}{'ic':>8}")
for cls in ("std","eq"):
    for arm in ("A-SIZINTILI","B-TEMIZ"):
        a=np.array(res[(cls,arm)])
        print(f"{cls:<8}{arm:<14}{a[:,0].mean():8.2f}{a[:,1].mean():8.2f}{a[:,2].mean():8.2f}")
    a=np.array(res[(cls,"A-SIZINTILI")]); b=np.array(res[(cls,"B-TEMIZ")])
    print(f"{'':<8}{'>>> SISME':<14}{a[:,0].mean()-b[:,0].mean():+8.2f}{a[:,1].mean()-b[:,1].mean():+8.2f}{a[:,2].mean()-b[:,2].mean():+8.2f}\n")
