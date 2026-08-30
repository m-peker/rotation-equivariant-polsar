"""What are Beijing's off-palette colours? A sixth class, or background?"""
import glob, os, numpy as np, tifffile as tf

ROOT = ("c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/"
        "AIR-PolSAR-Seg-2.0-New")
d = [x for x in glob.glob(os.path.join(ROOT, "*", "*"))
     if os.path.isdir(x) and "__MACOSX" not in x
     and os.path.exists(os.path.join(x, "bj_hh.tiff"))][0]
f = sorted(glob.glob(os.path.join(d, "modified", "*_gt_modify.tiff")))[0]

a = tf.imread(f)[..., :3]
cols, cnt = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
order = np.argsort(-cnt)
tot = cnt.sum()
print("Beijing gt_modify: %d distinct colours" % len(cols))
for i in order[:12]:
    print("   RGB%-16s %8.4f %%" % (tuple(int(x) for x in cols[i]),
                                    100 * cnt[i] / tot))

# where do the off-palette pixels sit? contiguous region or scattered?
PAL = np.array([(255, 255, 0), (0, 255, 0), (0, 255, 255),
                (0, 0, 255), (255, 0, 0)], np.int16)
flat = a.reshape(-1, 3).astype(np.int16)
d1 = np.abs(flat[:, None, :] - PAL[None]).sum(2).min(1)
off = (d1 > 30).reshape(a.shape[:2])
print("\noff-palette fraction: %.3f %%" % (100 * off.mean()))
rows = off.mean(1); colsf = off.mean(0)
print("rows fully off-palette   : %d of %d" % ((rows > 0.99).sum(), len(rows)))
print("columns fully off-palette: %d of %d" % ((colsf > 0.99).sum(), len(colsf)))
nz = np.nonzero(rows > 0.5)[0]
if len(nz):
    print("row band with >50%% off-palette: %d..%d" % (nz.min(), nz.max()))
nzc = np.nonzero(colsf > 0.5)[0]
if len(nzc):
    print("col band with >50%% off-palette: %d..%d" % (nzc.min(), nzc.max()))
