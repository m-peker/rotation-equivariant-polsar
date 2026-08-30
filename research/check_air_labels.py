"""How much of the AIR-PolSAR-Seg-2.0 ground truth does exact colour matching lose?

The label maps are distributed as RGB images and two of the three regions carry
compression artefacts: Shanghai alone has 2337 distinct colours where the palette
has six. Exact matching therefore silently discards every disturbed pixel. This
measures the damage and checks whether a sixth class is present at all -- the
dataset paper names six categories and our reader found five.
"""
import glob, os, sys, numpy as np, tifffile as tf

ROOT = ("c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/"
        "AIR-PolSAR-Seg-2.0-New")
PALETTE = {(255, 255, 0): "yellow", (0, 255, 0): "green",
           (0, 255, 255): "cyan", (0, 0, 255): "blue",
           (255, 0, 0): "red", (255, 255, 255): "white",
           (0, 0, 0): "black"}
NAMES = {"gz": "Guangzhou", "sh": "Shanghai", "bj": "Beijing"}


def region_dir(tag):
    for d in glob.glob(os.path.join(ROOT, "*", "*")):
        if os.path.isdir(d) and "__MACOSX" not in d \
           and os.path.exists(os.path.join(d, "%s_hh.tiff" % tag)):
            return d
    raise FileNotFoundError(tag)


keys = np.array(list(PALETTE.keys()), np.int16)
labels = list(PALETTE.values())

for tag in ("gz", "sh", "bj"):
    d = region_dir(tag)
    f = sorted(glob.glob(os.path.join(d, "modified", "*_gt_modify.tiff"))) \
        or sorted(glob.glob(os.path.join(d, "modified", "*_gt.tiff")))
    a = tf.imread(f[0])[..., :3].astype(np.int16)
    flat = a.reshape(-1, 3)
    n = len(flat)

    exact = np.zeros(n, bool)
    for k in keys:
        exact |= (flat == k).all(1)

    # nearest palette colour and how far it is
    dist = np.abs(flat[:, None, :] - keys[None, :, :]).sum(2)
    near = dist.argmin(1)
    dmin = dist.min(1)

    print("=== %s (%s) ===" % (NAMES[tag], os.path.basename(f[0])))
    print("  pixels                    : %d" % n)
    print("  exact palette match       : %.3f %%" % (100 * exact.mean()))
    print("  within L1 distance 30     : %.3f %%" % (100 * (dmin <= 30).mean()))
    print("  further than 30 (unclear) : %.3f %%" % (100 * (dmin > 30).mean()))
    print("  lost by exact matching    : %.3f %%"
          % (100 * ((~exact) & (dmin <= 30)).mean()))
    print("  nearest-colour histogram:")
    for i, name in enumerate(labels):
        m = (near == i) & (dmin <= 30)
        if m.mean() > 1e-6:
            print("      %-7s %7.3f %%" % (name, 100 * m.mean()))
    del dist, near, dmin, flat, a
    print()
