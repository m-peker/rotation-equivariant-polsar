"""
Uzamsal-ayrik (blok + tampon) bolutleme.

Rastgele piksel split'i patch-tabanli siniflandirmada SIZINTILIDIR: komsu iki
pikselin 15x15 patch'leri ~%87 ortak girdi paylasir. Hiperspektralde blok+tampon
ile rastgele split arasinda 30.56 puan fark olculmus (Ahmad 2024).
PolSAR'da bu yapilmamis.

Kural: goruntu SxS bloklara bolunur, bloklar train/test'e atanir, ve her blogun
kenarindan B piksel ATILIR. Patch yaricapi M=W//2 icin B >= (W+1)/2 secilirse
bir train merkezi ile bir test merkezi arasindaki mesafe >= 2B > W olur
-> patch'ler HIC ortusmez.
"""
import numpy as np
from scipy.ndimage import binary_dilation

def block_buffer_split(gt, W=15, block=96, test_frac=0.7, seed=0, ensure_cover=True):
    R,C = gt.shape
    B = (W+1)//2                       # 15 -> 8 ; 2B=16 > 15  => sifir ortusme
    nbr, nbc = int(np.ceil(R/block)), int(np.ceil(C/block))
    rng = np.random.default_rng(seed)
    assign = rng.random(nbr*nbc) < test_frac      # True = test blogu
    br = np.arange(R)//block; bc = np.arange(C)//block
    bid = br[:,None]*nbc + bc[None,:]
    is_test_block = assign[bid]
    # blok kenarindan B piksel tampon
    off_r = np.minimum(np.arange(R)%block, (block-1)-(np.arange(R)%block))
    off_r = np.minimum(off_r, np.minimum(np.arange(R), R-1-np.arange(R)) + block)  # goruntu kenari tampon degil
    off_c = np.minimum(np.arange(C)%block, (block-1)-(np.arange(C)%block))
    off_c = np.minimum(off_c, np.minimum(np.arange(C), C-1-np.arange(C)) + block)
    inner = (off_r[:,None] >= B) & (off_c[None,:] >= B)
    lab = gt > 0
    if ensure_cover:
        # SINIF-FARKINDA duzeltme: bir sinifin train blogu yoksa, o siniftan en cok
        # piksel iceren test blogunu train'e cevir. (Flevoland'da siniflar bitisik
        # tarlalar oldugu icin saf rastgele atama bazi siniflari train'de birakmiyor.)
        for k in range(1, int(gt.max())+1):
            cls = (gt==k) & inner
            if not cls.any(): continue
            if (cls & ~is_test_block).sum() > 0: continue
            cnt = np.bincount(bid[cls].ravel(), minlength=nbr*nbc)
            order = np.argsort(cnt)[::-1]
            for b in order:
                if cnt[b] == 0: break
                if assign[b]:
                    assign[b] = False; break
            is_test_block = assign[bid]
    tr_mask = lab & inner & (~is_test_block)
    te_mask = lab & inner & ( is_test_block)
    return tr_mask, te_mask, B


def sample_per_class(gt, tr_mask, n_per_class, seed=0):
    """Egitim BOLGESINDEN sinif basina n piksel sec (protokol etkisini butceden ayirir)."""
    rng = np.random.default_rng(seed); rs=[]; cs=[]
    for k in range(1, int(gt.max())+1):
        r,c = np.nonzero((gt==k) & tr_mask)
        if len(r)==0: continue
        p = rng.choice(len(r), min(n_per_class,len(r)), replace=False)
        rs.append(r[p]); cs.append(c[p])
    return np.concatenate(rs), np.concatenate(cs)

def verify_no_overlap(tr_mask, te_mask, W=15):
    """train merkezlerinin patch ayak izini genislet; test merkezi dusuyor mu?"""
    foot = np.ones((W,W), bool)
    grown = binary_dilation(tr_mask, structure=foot)   # bir train patch'inin dokundugu her piksel
    # bir test MERKEZI, train merkezinden < W uzaklikta ise patch'ler ortusur
    clash = grown & te_mask
    return int(clash.sum())

if __name__ == "__main__":
    import sys; sys.path.insert(0,'.')
    from polsar_lib import load_gt
    gt = load_gt(); ncl = int(gt.max()); lab = int((gt>0).sum())
    print(f"etiketli piksel toplam: {lab}\n")
    print(f"{'blok':>6}{'tampon':>8}{'train':>9}{'test':>9}{'atilan':>9}{'atilan%':>9}{'ORTUSME':>9}  sinif kapsami")
    for blk in (64, 96, 128, 160):
        tr, te, B = block_buffer_split(gt, block=blk, test_frac=0.7, seed=0)
        clash = verify_no_overlap(tr, te)
        drop = lab - int(tr.sum()) - int(te.sum())
        ctr = np.array([int(((gt==k)&tr).sum()) for k in range(1,ncl+1)])
        cte = np.array([int(((gt==k)&te).sum()) for k in range(1,ncl+1)])
        cov = f"{int((ctr>0).sum())}/{ncl} tr, {int((cte>0).sum())}/{ncl} te"
        print(f"{blk:>6}{B:>8}{int(tr.sum()):>9}{int(te.sum()):>9}{drop:>9}{100*drop/lab:>8.1f}%{clash:>9}  {cov}")
    print("\n--- referans: RASTGELE split'te ortusme ---")
    rng=np.random.default_rng(0); r,c = np.nonzero(gt>0)
    p = rng.permutation(len(r)); ntr = int(0.01*len(r))
    trm = np.zeros(gt.shape,bool); tem = np.zeros(gt.shape,bool)
    trm[r[p[:ntr]], c[p[:ntr]]] = True; tem[r[p[ntr:]], c[p[ntr:]]] = True
    cl = verify_no_overlap(trm, tem)
    print(f"rastgele %1 split: {int(trm.sum())} train, {int(tem.sum())} test")
    print(f"  patch'i bir train patch'iyle ORTUSEN test pikseli: {cl}  (%{100*cl/tem.sum():.1f})")
    # en kucuk sinif icin blok=96'da durum
    tr,te,_ = block_buffer_split(gt, block=96, test_frac=0.7, seed=0)
    print("\n--- blok=96, sinif basina train/test ---")
    for k in range(1,ncl+1):
        print(f"  sinif {k:2d}: train {int(((gt==k)&tr).sum()):6d}  test {int(((gt==k)&te).sum()):6d}")
