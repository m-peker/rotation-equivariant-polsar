"""Exp. 39 - measure the train/test patch overlap instead of asserting it.

The manuscript states that 84.2 per cent of test pixels in the standard split
have patches overlapping a training patch, and the patch-size table carries a
column of overlap fractions. Neither was traceable to a stored measurement: the
first appeared only in the prose and the second was a hard-coded dictionary in
the table generator. That is precisely the failure the audit exists to prevent,
so both are measured here and both now read from this file.

Two patches of side W centred at p and q overlap iff they differ by less than W
in each axis, so the set of test pixels whose patch meets some training patch is
the training mask dilated by W-1 under the Chebyshev metric. No training is
involved; this is a property of the split and the patch size alone.

Both sampling schemes used in the paper are measured, because they give
different answers and the difference is the whole point:

  balanced   the class-balanced budget this paper trains with
  ratio-1%   the stratified one-per-cent split common in this literature

Resumable into exp39_results.json.
"""
import numpy as np, json, os, sys
sys.path.insert(0, ".")
from scipy.ndimage import binary_dilation
from polsar_data import load_scene

SCENES = ("flevoland", "sanfran", "ober")
BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}
WS = (7, 11, 15, 19)
SEEDS = (0, 1, 2)
STORE = "exp39_results.json"


def overlap_fraction(gt, tr_rc, W):
    """Fraction of the remaining labelled pixels whose W x W patch meets a
    training patch."""
    m = np.zeros(gt.shape, bool)
    m[tr_rc[0], tr_rc[1]] = True
    reach = binary_dilation(m, np.ones((2 * W - 1, 2 * W - 1), bool))
    lab = gt > 0
    test = lab & ~m
    return float(reach[test].mean()), int(test.sum())


def sample_balanced(gt, n_per_class, rng):
    rs, cs = [], []
    for k in range(1, int(gt.max()) + 1):
        r, c = np.nonzero(gt == k)
        if len(r) == 0:
            continue
        p = rng.choice(len(r), min(n_per_class, len(r)), replace=False)
        rs.append(r[p]); cs.append(c[p])
    return np.concatenate(rs), np.concatenate(cs)


def sample_ratio(gt, frac, rng):
    rs, cs = [], []
    for k in range(1, int(gt.max()) + 1):
        r, c = np.nonzero(gt == k)
        if len(r) == 0:
            continue
        n = max(1, int(round(frac * len(r))))
        p = rng.choice(len(r), n, replace=False)
        rs.append(r[p]); cs.append(c[p])
    return np.concatenate(rs), np.concatenate(cs)


def main():
    R = json.load(open(STORE)) if os.path.exists(STORE) else {}
    print("%-11s %-10s %-3s %10s %10s" % ("scene", "sampling", "W", "overlap %", "train px"),
          flush=True)
    for sc in SCENES:
        _, gt, _ = load_scene(sc)
        for mode in ("balanced", "ratio1"):
            for W in WS:
                key = "%s|%s|W=%d" % (sc, mode, W)
                if key in R:
                    continue
                fr, ntr = [], 0
                for sd in SEEDS:
                    rng = np.random.default_rng(1000 + sd)
                    tr = (sample_balanced(gt, BUDGET[sc], rng) if mode == "balanced"
                          else sample_ratio(gt, 0.01, rng))
                    f, _ = overlap_fraction(gt, tr, W)
                    fr.append(100 * f); ntr = len(tr[0])
                R[key] = dict(overlap=float(np.mean(fr)),
                              sd=float(np.std(fr)), train=ntr)
                json.dump(R, open(STORE, "w"), indent=1)
                print("%-11s %-10s %-3d %10.2f %10d"
                      % (sc, mode, W, np.mean(fr), ntr), flush=True)
        del gt
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
