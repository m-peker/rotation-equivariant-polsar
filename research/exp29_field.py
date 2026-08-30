"""Exp. 29 - a spatially structured rotation, instead of a constant one.

The rotation experiments elsewhere in this paper apply one angle to the whole
scene. That is the right test for the equivariance property but it is not a
physically realistic perturbation: terrain tilts the polarisation basis by an
amount that varies from place to place, so the angle is a field, not a constant.
A constant angle is also the case a canonicalisation handles best, because one
angle estimated per pixel is enough to undo it.

This applies a smooth random rotation field instead. The field is calibrated
against the orientation structure actually measured in these scenes by Exp. 25:
its correlation length is chosen so that the ratio between the variation seen
inside one 15x15 patch and the variation over the whole scene matches the
measured field, and its amplitude is swept from the measured value upward.

The comparison it enables is the one a constant angle cannot make. Orientation
angle compensation removes a per-pixel angle, which also removes the relative
orientation between neighbouring pixels -- real structure, 5-7 deg of it inside
a patch by Exp. 25. An architecturally equivariant network is invariant to a
global rotation while keeping that relative structure. Under a constant angle
the two are indistinguishable; under a field they need not be.

Note what is NOT claimed: a network equivariant to the global group is not
invariant to a spatially varying rotation, and we do not expect a flat curve
here. The question is which method degrades least.

Resumable into exp29_results.json.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from scipy.ndimage import gaussian_filter
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool, rot6_torch
from equivariant import EqCVCNN_F
from steerable import SteerNet, field_stats
from polsar_data import load_scene
from exp27_oac import OACPipe, CVCNNb

BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}
SEEDS = (0, 1, 2)
AMPS = [0.0, 5.0, 10.0, 20.0]          # degrees, standard deviation of the field
STORE = "exp29_results.json"
W = 15


def measured_ratio(scene):
    """Variation inside a 15x15 window divided by variation over the scene,
    for the real orientation field. This is the shape statistic the synthetic
    field is matched to."""
    X, gt, _ = load_scene(scene)
    T22 = X[..., 1].real.astype(np.float64)
    T33 = X[..., 2].real.astype(np.float64)
    z = (T22 - T33) / 2 + 1j * X[..., 5].real.astype(np.float64)
    del X
    m = gt > 0
    ang = np.degrees(-np.angle(z) / 4.0)
    # the structured component: what survives a local average
    lo = gaussian_filter(ang, 3.0)
    loc = lo - gaussian_filter(lo, W / 2.0)      # variation within a patch
    return float(loc[m].std() / lo[m].std()), float(lo[m].std())


def make_field(shape, sd_deg, sigma, rng):
    """Smooth Gaussian random field, scaled to the requested standard deviation."""
    f = gaussian_filter(rng.standard_normal(shape), sigma)
    f = f / (f.std() + 1e-12) * np.deg2rad(sd_deg)
    return f.astype(np.float32)


def fit_sigma(target_ratio):
    """Pick the smoothing scale that reproduces the measured within-patch ratio."""
    rng = np.random.default_rng(0)
    best, bs = None, 1e9
    for sig in np.arange(1.0, 40.0, 0.5):
        f = gaussian_filter(rng.standard_normal((512, 512)), sig)
        loc = f - gaussian_filter(f, W / 2.0)
        r = loc.std() / (f.std() + 1e-12)
        if abs(r - target_ratio) < bs:
            bs, best = abs(r - target_ratio), sig
    return float(best)


class FieldPipe(MSPipe):
    """Applies a per-pixel rotation field stored on the GPU, then normalises."""

    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s.F = None
        s.raw = False          # SteerNet consumes raw T and normalises itself

    def set_field(s, f):
        s.F = (None if f is None else
               torch.from_numpy(np.pad(f, s.M, mode="edge")).to(s.dev))

    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if s.F is not None:
            xr, xi = rot6_torch(xr, xi, s.F[rr, cc])
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return (xr, xi) if s.raw else s.eqn(xr, xi)


class FieldOAC(OACPipe):
    """The same, with orientation angle compensation after the field."""

    def __init__(s, *a, **k):
        super().__init__(*a, **k)
        s.F = None

    def set_field(s, f):
        s.F = (None if f is None else
               torch.from_numpy(np.pad(f, s.M, mode="edge")).to(s.dev))

    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if s.F is not None:
            xr, xi = rot6_torch(xr, xi, s.F[rr, cc])
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return s._finish(xr, xi)

    def _finish(s, xr, xi):
        import torch.nn.functional as F
        zr = (xr[:, 1] - xr[:, 2]) / 2
        zi = xr[:, 5]
        if s.oac == "smooth":
            zr = F.avg_pool2d(zr.unsqueeze(1), s.W, stride=1).squeeze(1)[:, 0, 0]
            zi = F.avg_pool2d(zi.unsqueeze(1), s.W, stride=1).squeeze(1)[:, 0, 0]
        elif s.oac == "patch":
            zr, zi = zr[:, s.M, s.M], zi[:, s.M, s.M]
        alpha = torch.atan2(zi, zr) / 4.0
        xr, xi = rot6_torch(xr, xi, alpha)
        return s.eqn(xr, xi)


def train(pipe, kind, seed, stats, aug=False, epochs=120):
    P = pipe
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET[P.scene], int((P.y == k).sum())),
                                    replace=False) for k in range(P.ncl)])
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = {"base": lambda: CVCNNb(P.ncl, cin=7),
           "eqf": lambda: EqCVCNN_F(P.ncl, cin=7, N=8),
           "steer": lambda: SteerNet(P.ncl, stats=stats)}[kind]().cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), 128):
            b = pm[s:s + 128]
            th = (torch.rand(len(b), device="cuda") * np.pi) if aug else None
            xr, xi = P.grab(Rt[b], Ct[b], th)
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    return net, tr


def evaluate(pipe, net, tr):
    P = pipe
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    pr = np.empty(len(ite), np.int64)
    with torch.no_grad():
        for s in range(0, len(ite), 4096):
            xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096])
            pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
    return float(100 * (pr == P.y[ite]).mean())


ARMS = [("baseline", "plain", "base", False),
        ("baseline + rot. aug.", "plain", "base", True),
        ("CV-CNN + OAC", "oac", "base", False),
        ("Equivariant", "plain", "eqf", False),
        ("Steerable", "plain", "steer", False)]


def main():
    R = json.load(open(STORE)) if os.path.exists(STORE) else {}
    for sc in ("flevoland", "sanfran", "ober"):
        todo = [a for a in ARMS if "%s|%s" % (sc, a[0]) not in R]
        if not todo:
            print("### %s cached ###" % sc, flush=True)
            continue
        ratio, sd_meas = measured_ratio(sc)
        sigma = fit_sigma(ratio)
        print("\n### %s ###  measured within/total ratio %.3f -> sigma %.1f px, "
              "structured sd %.2f deg" % (sc, ratio, sigma, sd_meas), flush=True)
        Pp = FieldPipe(sc, norm="equivariant", eval_cap=150_000)
        Po = FieldOAC(sc, norm="equivariant", eval_cap=150_000, oac="smooth")
        Xs, _, _ = load_scene(sc); ST = field_stats(Xs); del Xs
        rngf = np.random.default_rng(4242)
        fields = {a: make_field(Pp.gt.shape, a, sigma, rngf) for a in AMPS if a > 0}
        for nm, pk, kind, aug in todo:
            P = Po if pk == "oac" else Pp
            Pp.raw = (kind == "steer")
            t0 = time.time(); acc = []
            for sd in SEEDS:
                P.set_field(None)
                net, tr = train(P, kind, sd, ST, aug=aug)
                row = []
                for a in AMPS:
                    P.set_field(None if a == 0 else fields[a])
                    row.append(evaluate(P, net, tr))
                acc.append(row)
                del net; torch.cuda.empty_cache()
            A = np.array(acc)
            R["%s|%s" % (sc, nm)] = dict(amps=AMPS, mean=A.mean(0).tolist(),
                                         std=A.std(0).tolist(), sigma=sigma,
                                         ratio=ratio, sd_measured=sd_meas)
            json.dump(R, open(STORE, "w"), indent=1)
            print("  %-22s %s   (%.0fs)"
                  % (nm, "".join("%9.2f" % v for v in A.mean(0)), time.time() - t0),
                  flush=True)
        del Pp, Po
        torch.cuda.empty_cache()
    print("\nstored in " + STORE)


if __name__ == "__main__":
    print("amp (deg):" + "".join("%9s" % a for a in AMPS), flush=True)
    main()
