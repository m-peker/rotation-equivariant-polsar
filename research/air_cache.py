"""Build and cache the AIR-PolSAR-Seg-2.0 coherency matrices.

Reading three GF-3 acquisitions from int16 SLC takes minutes and a lot of
memory, so it is done once and stored as npz. The cache is what the pipeline
loads.
"""
import numpy as np, os, sys
sys.path.insert(0, ".")
from air2 import load_T3, REGIONS

LOOK = 5
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "air_cache")
os.makedirs(CACHE, exist_ok=True)


def path(tag):
    return os.path.join(CACHE, "air_%s_look%d.npz" % (tag, LOOK))


def build(tag, force=False):
    p = path(tag)
    if os.path.exists(p) and not force:
        print("  cached: %s" % os.path.basename(p))
        return p
    print("### %s ###" % REGIONS[tag], flush=True)
    X, gt = load_T3(tag, look=LOOK)
    np.savez_compressed(p, X=X, gt=gt)
    print("  wrote %s  (%.0f MB)" % (os.path.basename(p),
                                     os.path.getsize(p) / 1e6), flush=True)
    return p


if __name__ == "__main__":
    for t in ("gz", "sh", "bj"):
        build(t)
