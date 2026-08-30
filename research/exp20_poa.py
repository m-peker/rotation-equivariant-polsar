"""Exp. 20 - is the rotation we test for actually present in the data?

The strongest objection to the paper is that our robustness test is synthetic:
we apply U(theta) ourselves and then show that a method built to be invariant to
it is invariant. This measures whether the shift occurs in the scenes without us
putting it there.

The polarisation orientation angle is estimable from the data, and the classical
estimator is exactly the phase of our weight-4 component:

    z_4 = (T22 - T33)/2 + i Re T23,      z_4 -> e^{-4 i theta} z_4
    =>  theta_POA = -arg(z_4) / 4                        (period pi/2)

So the decomposition of Section II gives the estimator for free. If the POA has
a wide spread within a scene -- and in particular if it differs systematically
between classes or between fields of the same class -- then a classifier that is
not invariant to it is exposed to a shift that is genuinely there.
"""
import numpy as np, sys, json
sys.path.insert(0, ".")
from polsar_data import load_scene, bdist
from scipy.ndimage import label as cc_label

OUT = {}
print("%-16s %8s %8s %8s %8s %9s" %
      ("scene", "std", "IQR", "p5", "p95", "range90"))
for sc, nm in [("flevoland", "Flevoland"), ("sanfran", "San Francisco"),
               ("ober", "Oberpfaffenhofen")]:
    X, gt, ncl = load_scene(sc)
    T22 = X[..., 1].real.astype(np.float64)
    T33 = X[..., 2].real.astype(np.float64)
    w4 = (T22 - T33) / 2 + 1j * X[..., 5].real.astype(np.float64)
    mag = np.abs(w4)
    poa = -np.angle(w4) / 4.0                      # radians, period pi/2
    deg = np.degrees(poa)

    m = gt > 0
    # weight by |z_4|: where the weight-4 component is negligible the angle is
    # not identifiable, so an unweighted histogram would be dominated by noise.
    thr = np.percentile(mag[m], 40)
    sel = m & (mag > thr)
    d = deg[sel]
    q1, q3 = np.percentile(d, [25, 75])
    p5, p95 = np.percentile(d, [5, 95])
    print("%-16s %8.2f %8.2f %8.2f %8.2f %9.2f"
          % (nm, d.std(), q3 - q1, p5, p95, p95 - p5))

    # per-class medians: does the orientation differ systematically by class?
    permed, perspread = [], []
    for k in range(1, ncl + 1):
        s = sel & (gt == k)
        if s.sum() < 200:
            continue
        permed.append(float(np.median(deg[s])))
        perspread.append(float(np.percentile(deg[s], 75) - np.percentile(deg[s], 25)))
    # between-field spread within a class (same class, different fields)
    fieldspread = []
    for k in range(1, ncl + 1):
        lab, n = cc_label(gt == k)
        med = []
        for j in range(1, n + 1):
            s = sel & (lab == j)
            if s.sum() >= 200:
                med.append(float(np.median(deg[s])))
        if len(med) >= 3:
            fieldspread.append(float(np.std(med)))
    OUT[sc] = dict(std=float(d.std()), iqr=float(q3 - q1),
                   range90=float(p95 - p5),
                   class_medians=permed,
                   class_median_spread=float(np.std(permed)) if permed else None,
                   within_class_field_spread=fieldspread,
                   hist=np.histogram(d, bins=60, range=(-22.5, 22.5))[0].tolist())
    print("     per-class median spread : %.2f deg  over %d classes"
          % (np.std(permed) if permed else float("nan"), len(permed)))
    if fieldspread:
        print("     same-class field-to-field spread : %.2f deg (median over classes)"
              % np.median(fieldspread))
    del X

json.dump(OUT, open("exp20_results.json", "w"), indent=1)
print("\nwritten to exp20_results.json")
print("""
Reading: the orientation angle is not a constant of the scene. A network that is
not invariant to it must learn the variation from labels instead of being told
it, which is what the augmentation experiments show it can do only at a cost.""")
