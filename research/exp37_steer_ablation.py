"""Exp. 37 - why is the steerable network the weaker of the two we propose?

It is exact where the discrete network is only exact on a grid, yet it sits
\SI{2.35}{pp} below it on Flevoland and \SIrange{3}{5}{pp} below the plain
baseline on the AIR scenes, with a large seed spread. A reviewer is entitled to
ask whether that is the price of exactness or a fixable defect in our
construction, and the paper currently does not say.

Three parts of the construction can be varied without touching the equivariance
argument, because equivariance holds weight class by weight class regardless of
which of these is chosen:

  gate     how the harmonic fields are made non-linear. The default multiplies
           each field by a real gate built from the invariant stream. The
           alternative is a norm non-linearity acting on the field's own
           magnitude, and the null case is no non-linearity at all.
  readout  what reaches the classifier. The default is the invariant scalars,
           the magnitudes and the weight-2/weight-4 relative phase. Dropping the
           relative phase, or keeping only the invariant scalars, removes
           information the network is otherwise allowed to use.
  width    whether the gap is simply capacity.

The truncation was already tested in Exp. 30 and is not the cause.

Resumable into exp37_results.json.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
from steerable import (SteerLift, SteerLayer, decompose, pool, cmul, cmulc,
                       field_stats)
from polsar_data import load_scene

SCENE = "flevoland"
BUDGET = 133
SEEDS = (0, 1, 2, 3, 4)
THETAS = [0, 10, 45]
STORE = "exp37_results.json"


class RawPipe(MSPipe):
    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return xr, xi


class Layer(SteerLayer):
    """SteerLayer with the non-linearity on the harmonic fields swappable."""

    def __init__(s, c0i, ci, c0o, co, k=3, gate="invariant"):
        super().__init__(c0i, ci, c0o, co, k)
        s.mode = gate
        if gate == "norm":
            s.b2 = nn.Parameter(torch.zeros(1, co, 1, 1))
            s.b4 = nn.Parameter(torch.zeros(1, co, 1, 1))

    def forward(s, w0, z2, z4):
        z2r, z2i = z2; z4r, z4i = z4
        n2 = z2r ** 2 + z2i ** 2
        n4 = z4r ** 2 + z4i ** 2
        h0 = F.relu(s.k0(torch.cat([w0, n2, n4], 1)))
        cg4r, cg4i = cmul(z2r, z2i, z2r, z2i)
        cg2r, cg2i = cmulc(z4r, z4i, z2r, z2i)
        h2r, h2i = s.k2(torch.cat([z2r, cg2r], 1), torch.cat([z2i, cg2i], 1))
        h4r, h4i = s.k4(torch.cat([z4r, cg4r], 1), torch.cat([z4i, cg4i], 1))
        if s.mode == "invariant":
            g2 = torch.sigmoid(s.g2(h0)); g4 = torch.sigmoid(s.g4(h0))
        elif s.mode == "norm":
            # z -> z * relu(|z| + b)/|z| : acts on the field's own magnitude,
            # which is invariant, so the phase still factors out exactly
            m2 = torch.sqrt(h2r ** 2 + h2i ** 2 + 1e-12)
            m4 = torch.sqrt(h4r ** 2 + h4i ** 2 + 1e-12)
            g2 = F.relu(m2 + s.b2) / m2
            g4 = F.relu(m4 + s.b4) / m4
        else:                                   # "none": linear fields
            g2 = g4 = 1.0
        return h0, (h2r * g2, h2i * g2), (h4r * g4, h4i * g4)


def readout(w0, z2, z4, mode):
    z2r, z2i = z2; z4r, z4i = z4
    m2 = torch.sqrt(z2r ** 2 + z2i ** 2 + 1e-12)
    m4 = torch.sqrt(z4r ** 2 + z4i ** 2 + 1e-12)
    if mode == "w0":
        return w0
    if mode == "mag":
        return torch.cat([w0, m2, m4], 1)
    sqr, sqi = cmul(z2r, z2i, z2r, z2i)
    xr, xi = cmulc(z4r, z4i, sqr, sqi)
    d = torch.sqrt(sqr ** 2 + sqi ** 2 + 1e-12) * m4 + 1e-12
    return torch.cat([w0, m2, m4, xr / d, xi / d], 1)


NCH = {"full": lambda c0, c: c0 + 4 * c, "mag": lambda c0, c: c0 + 2 * c,
       "w0": lambda c0, c: c0}


class Net(nn.Module):
    def __init__(s, ncl, c=24, c0=16, stats=None, gate="invariant",
                 rout="full"):
        super().__init__()
        s.rout = rout
        s.lift = SteerLift(c0, c, *(stats if stats is not None
                                    else (None, None, 1.0, 1.0)))
        s.l1 = Layer(c0, c, c0, c, gate=gate)
        s.l2 = Layer(c0, c, c0, c, gate=gate)
        s.l3 = Layer(c0, c, c0, c, gate=gate)
        nin = NCH[rout](c0, c) * 3 * 3
        s.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                               nn.Linear(nin, 128), nn.ReLU(),
                               nn.Linear(128, ncl))

    def forward(s, xr, xi):
        w0, z2, z4 = decompose(xr, xi)
        w0, z2, z4 = s.lift(w0, z2, z4)
        w0, z2, z4 = s.l1(w0, z2, z4)
        w0, z2, z4 = s.l2(w0, z2, z4); w0, z2, z4 = pool(w0, z2, z4)
        w0, z2, z4 = s.l3(w0, z2, z4); w0, z2, z4 = pool(w0, z2, z4)
        return s.head(readout(w0, z2, z4, s.rout))


ARMS = [
    ("gate: invariant (ours)", dict(gate="invariant", rout="full")),
    ("gate: norm non-lin.",    dict(gate="norm", rout="full")),
    ("gate: none (linear)",    dict(gate="none", rout="full")),
    ("readout: no rel. phase", dict(gate="invariant", rout="mag")),
    ("readout: invariants only", dict(gate="invariant", rout="w0")),
    ("width: c=48",            dict(gate="invariant", rout="full", c=48)),
    ("width: c=12",            dict(gate="invariant", rout="full", c=12)),
]


def run(P, kw, seed, stats, epochs=120):
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET, int((P.y == k).sum())),
                                    replace=False) for k in range(P.ncl)])
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = Net(P.ncl, stats=stats, **kw).cuda()
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
    p = sum(q.numel() for q in net.parameters())
    del net; torch.cuda.empty_cache()
    return out, p


def main():
    R = json.load(open(STORE)) if os.path.exists(STORE) else {}
    P = RawPipe(SCENE, norm="equivariant", eval_cap=150_000)
    Xs, _, _ = load_scene(SCENE); ST = field_stats(Xs); del Xs
    print("%-26s %8s %8s %8s %7s %9s" % ("arm", "OA", "OA10", "OA45", "sd", "params"),
          flush=True)
    for nm, kw in ARMS:
        if nm in R:
            v = R[nm]
            print("%-26s %8.2f %8.2f %8.2f %7.2f %9d  (cached)"
                  % (nm, v["mean"][0], v["mean"][1], v["mean"][2], v["sd"],
                     v["params"]), flush=True)
            continue
        t0 = time.time(); acc = []
        for sd in SEEDS:
            a, p = run(P, kw, sd, ST)
            acc.append(a)
        A = np.array(acc)
        R[nm] = dict(mean=A.mean(0).tolist(), sd=float(A[:, 0].std()), params=p)
        json.dump(R, open(STORE, "w"), indent=1)
        print("%-26s %8.2f %8.2f %8.2f %7.2f %9d  (%.0fs)"
              % (nm, A.mean(0)[0], A.mean(0)[1], A.mean(0)[2], A[:, 0].std(),
                 p, time.time() - t0), flush=True)
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
