"""Exp. 25 - is the orientation angle a field, or is it speckle?

Exp. 20 showed the per-class medians of the orientation angle coincide, which
bounds the claim of the paper. It left one question open: whether the wide
per-pixel spread is a real spatial field that simply has no class-level mean, or
whether it is estimation noise carrying no information at all. The two readings
differ for the paper -- a structured field that varies within scenes is
something a classifier is exposed to; pure noise is not.

The test needs no extra data. Speckle in a multi-looked product decorrelates
within a couple of pixels; terrain- or structure-driven orientation varies over
much longer distances. So we compare the spatial autocorrelation of the
orientation field against two references measured on the same pixels:

  span   total power, which carries genuine scene structure  -> upper reference
  shuffled  the same angle values randomly permuted           -> noise floor

If the orientation field decorrelates like the shuffled control, it is noise.
If it tracks the span, it is a field.
"""
import numpy as np, sys, json
sys.path.insert(0, ".")
from polsar_data import load_scene


def autocorr_profile(F, mask, lags):
    """Correlation of F with itself shifted by each lag, along rows and cols."""
    out = []
    for L in lags:
        vals = []
        for ax in (0, 1):
            a = np.roll(F, L, axis=ax)
            m = mask & np.roll(mask, L, axis=ax)
            if ax == 0:
                m[:L] = False
            else:
                m[:, :L] = False
            if m.sum() < 5000:
                continue
            x, y = F[m], a[m]
            vals.append(np.corrcoef(x, y)[0, 1])
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


LAGS = [1, 2, 3, 5, 8, 12, 20, 32, 50, 80]
OUT = {}
rng = np.random.default_rng(0)

print("%-16s %s" % ("", "  ".join("%5d" % l for l in LAGS)))
for sc, nm in [("flevoland", "Flevoland"), ("sanfran", "San Francisco"),
               ("ober", "Oberpfaffenhofen")]:
    X, gt, ncl = load_scene(sc)
    T22 = X[..., 1].real.astype(np.float64)
    T33 = X[..., 2].real.astype(np.float64)
    z = (T22 - T33) / 2 + 1j * X[..., 5].real.astype(np.float64)
    ang = -np.angle(z) / 4.0
    span = (X[..., 0].real + T22 + T33).astype(np.float64)
    del X

    mask = (gt > 0) & (np.abs(z) > np.percentile(np.abs(z)[gt > 0], 40))
    # angle is circular with period pi/2; correlate its unit vector instead
    F = np.cos(4 * ang)
    S = np.log(np.clip(span, 1e-20, None))
    Sh = F.copy()
    idx = np.nonzero(mask)
    perm = rng.permutation(len(idx[0]))
    Sh[idx] = F[idx[0][perm], idx[1][perm]]

    a_ang = autocorr_profile(F, mask, LAGS)
    a_spn = autocorr_profile(S, mask, LAGS)
    a_shf = autocorr_profile(Sh, mask, LAGS)
    OUT[sc] = dict(lags=LAGS, angle=a_ang, span=a_spn, shuffled=a_shf)

    print("%-16s %s   orientation" % (nm, "  ".join("%5.2f" % v for v in a_ang)))
    print("%-16s %s   span (reference)" % ("", "  ".join("%5.2f" % v for v in a_spn)))
    print("%-16s %s   shuffled (noise floor)" % ("", "  ".join("%5.2f" % v for v in a_shf)))
    # decorrelation lag: first lag where the profile drops below 1/e
    def dl(p):
        for l, v in zip(LAGS, p):
            if v < 0.3679:
                return l
        return LAGS[-1]
    print("%-16s orientation decorrelates at ~%d px, span at ~%d px, noise at ~%d px"
          % ("", dl(a_ang), dl(a_spn), dl(a_shf)))
    print()

json.dump(OUT, open("exp25_results.json", "w"), indent=1)
print("stored in exp25_results.json")
