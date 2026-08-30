"""Exp. 28 - why orientation angle compensation is not exactly invariant.

Canonicalisation should give exact invariance: the estimator theta_hat =
-arg(z_c)/4 is exactly equivariant, so rotating the data and then canonicalising
ought to land on the same representative. It does not, and the reason is the
principal value. arg returns a value in (-pi, pi], so theta_hat is confined to
(-pi/4, pi/4]. Rotating by beta shifts arg by -4 beta; when that crosses the
branch cut the estimate wraps, and the total rotation applied differs by pi/2
from the unrotated case.

A rotation by pi/2 is not the identity on T. Setting theta = pi/2 in the group
action gives cos 2theta = -1 and sin 2theta = 0, so T11, T22, T33 and Re T23 are
untouched while T12 and T13 -- the entire weight-2 sector -- change sign. So the
wrapped pixels are handed to the network with one irreducible block negated.

This script measures how many pixels that affects, per weight class, which is
the quantity that decides whether the classical baseline is a real alternative
to an architectural guarantee.
"""
import numpy as np, torch, json, sys
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
from exp27_oac import OACPipe

ANGLES = [10.0, 22.5, 45.0]
CH = [(0, "T11", "weight 0"), (3, "T12", "weight 2"), (4, "T13", "weight 2"),
      (5, "T23", "weight 4")]
OUT = {}

for sc in ("flevoland", "sanfran", "ober"):
    OUT[sc] = {}
    P = {m: OACPipe(sc, norm="equivariant", eval_cap=1000, oac=m)
         for m in ("pixel", "patch", "smooth")}
    n = min(4096, len(P["pixel"].lr))
    g = np.random.default_rng(0)
    sel = np.sort(g.choice(len(P["pixel"].lr), n, replace=False))
    r = torch.from_numpy(P["pixel"].lr[sel]).cuda()
    c = torch.from_numpy(P["pixel"].lc[sel]).cuda()
    print("### %s  (%d patches) ###" % (sc, n), flush=True)
    for m in ("pixel", "patch", "smooth"):
        OUT[sc][m] = {}
        x0 = P[m].grab(r, c)
        for deg in ANGLES:
            th = torch.full((n,), np.deg2rad(deg), device="cuda")
            x1 = P[m].grab(r, c, th)
            row = {}
            for ch, nm, wc in CH:
                a, b = x0[0][:, ch], x1[0][:, ch]
                row[nm] = dict(
                    weight=wc,
                    flipped=float(((a * b) < 0).float().mean()),
                    reldiff=float((a - b).abs().mean()
                                  / a.abs().mean().clamp(min=1e-12)))
            OUT[sc][m][str(deg)] = row
            print("  %-6s %5.1f deg   T12 flipped %5.1f %%   T11 reldiff %.1e"
                  % (m, deg, 100 * row["T12"]["flipped"], row["T11"]["reldiff"]),
                  flush=True)
        del x0
    del P
    torch.cuda.empty_cache()

json.dump(OUT, open("exp28_results.json", "w"), indent=1)
print("\nstored in exp28_results.json")
