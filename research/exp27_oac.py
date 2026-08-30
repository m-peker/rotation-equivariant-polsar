"""Exp. 27 - orientation angle compensation as a baseline.

The classical route to rotation invariance is Lee and Ainsworth's orientation
angle compensation: estimate the angle per pixel, rotate the data to a canonical
orientation, feed any complex-valued CNN. It needs no architectural change and
no extra parameters, and our own decomposition supplies the estimator for free,
so a reader is entitled to ask what it buys. The manuscript cites the method but
never runs it. This runs it.

The estimator is exactly equivariant -- rotating the data by alpha shifts
arg(z_c) by exactly -4 alpha -- so canonicalisation is invariant in exact
arithmetic, and we expect the rotation curve to be nearly flat. What we are
actually measuring is the price of getting invariance this way rather than
architecturally. Three things can make it differ from the architectural route:

  1. the pi/2 ambiguity. arg returns a principal value, so theta_hat lives in
     (-pi/4, pi/4]. Under a global rotation the estimate wraps, and the two
     branches differ by a rotation of pi/2, which is NOT the identity on T: it
     exchanges T22 and T33. Wrapped pixels are canonicalised into a different
     representative and the network sees a different input.
  2. per-pixel canonicalisation destroys the relative orientation between
     neighbouring pixels. Exp. 25 measured 5-7 deg of genuine orientation
     variation inside a 15x15 patch, so this is real structure being removed.
     The architectural route keeps it.
  3. where |z_c| is small the angle is not identifiable and the rotation applied
     is essentially arbitrary.

So we run three variants. Per-pixel is the classical method and pays (2) in
full. Per-patch rotates the whole patch by its centre estimate, which keeps the
relative structure but is fully exposed to (3). Smoothed averages z_c over the
patch before taking the argument, which is what practitioners actually do.

Resumable into exp27_results.json.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool, rot6_torch

THETAS = [0, 10, 22.5, 45]
BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}
SEEDS = (0, 1, 2)
STORE = "exp27_results.json"


class OACPipe(MSPipe):
    """MSPipe with orientation angle compensation applied after any test-time
    rotation and before normalisation, which is where a preprocessing step
    would sit in practice."""

    def __init__(s, *a, oac="pixel", **k):
        super().__init__(*a, **k)
        s.oac = oac

    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)

        # z_c = (T22 - T33)/2 + i Re T23 is the weight-4 component; it rotates
        # as e^{-4 i alpha}, so rotating by arg(z_c)/4 drives its argument to
        # zero, which is the canonical orientation.
        zr = (xr[:, 1] - xr[:, 2]) / 2
        zi = xr[:, 5]
        if s.oac == "smooth":
            zr = F.avg_pool2d(zr.unsqueeze(1), s.W, stride=1).squeeze(1)
            zi = F.avg_pool2d(zi.unsqueeze(1), s.W, stride=1).squeeze(1)
        if s.oac in ("patch", "smooth"):
            # one angle for the whole patch: the centre pixel, or the centre of
            # the smoothed field, which is the patch average of z_c
            zr = zr[:, s.M, s.M] if s.oac == "patch" else zr[:, 0, 0]
            zi = zi[:, s.M, s.M] if s.oac == "patch" else zi[:, 0, 0]
        alpha = torch.atan2(zi, zr) / 4.0
        xr, xi = rot6_torch(xr, xi, alpha)

        if s.eqn is not None:
            return s.eqn(xr, xi)
        raise RuntimeError("exp27 expects the equivariance-preserving norm")


class CVCNNb(nn.Module):
    def __init__(s, ncl, cin=7):
        super().__init__()
        s.c1 = CConv(cin, 32); s.c2 = CConv(32, 64); s.c3 = CConv(64, 128)
        s.f1 = CLin(128 * 3 * 3, 128); s.f2 = CLin(128, ncl); s.do = nn.Dropout(0.3)

    def forward(s, xr, xi):
        xr, xi = crelu(*s.c1(xr, xi))
        xr, xi = crelu(*s.c2(xr, xi)); xr, xi = cpool(xr, xi)
        xr, xi = crelu(*s.c3(xr, xi)); xr, xi = cpool(xr, xi)
        xr = s.do(xr.flatten(1)); xi = s.do(xi.flatten(1))
        xr, xi = crelu(*s.f1(xr, xi)); xr, xi = s.f2(xr, xi)
        return torch.sqrt(xr ** 2 + xi ** 2 + 1e-9)


def run(P, seed, epochs=120):
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET[P.scene], int((P.y == k).sum())),
                                    replace=False) for k in range(P.ncl)])
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = CVCNNb(P.ncl, cin=7).cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), 128):
            b = pm[s:s + 128]
            xr, xi = P.grab(Rt[b], Ct[b])
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()

    net.eval()
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    t = P.y[ite]; out = []
    with torch.no_grad():
        for deg in THETAS:
            pr = np.empty(len(ite), np.int64)
            for s in range(0, len(ite), 4096):
                n = len(Re[s:s + 4096])
                th = (torch.full((n,), np.deg2rad(deg), device="cuda")
                      if deg else None)
                xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096], th)
                pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
            out.append(float(100 * (pr == t).mean()))
    del net; torch.cuda.empty_cache()
    return out


def main():
    R = json.load(open(STORE)) if os.path.exists(STORE) else {}
    print("theta:" + "".join("%9s" % t for t in THETAS), flush=True)
    for sc in ("flevoland", "sanfran", "ober"):
        for mode in ("pixel", "patch", "smooth"):
            key = "%s|OAC-%s" % (sc, mode)
            if key in R:
                continue
            P = OACPipe(sc, norm="equivariant", eval_cap=150_000, oac=mode)
            t0 = time.time()
            acc = [run(P, sd) for sd in SEEDS]
            A = np.array(acc)
            R[key] = dict(mean=A.mean(0).tolist(), std=A.std(0).tolist())
            json.dump(R, open(STORE, "w"), indent=1)
            print("  %-22s %s   drop %5.2f  (%.0fs)"
                  % (key, "".join("%9.2f" % v for v in A.mean(0)),
                     A.mean(0)[0] - A.mean(0).min(), time.time() - t0), flush=True)
            del P; torch.cuda.empty_cache()
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
