"""Exp. 33 - transfer between two real acquisitions of the same scene.

Every rotation in this paper so far has been applied by us. The reviewers of an
earlier draft, and our own Discussion, identify the same gap: the operational
case for rotation invariance is transfer across acquisitions, and a synthetic
rotation cannot stand in for it.

The Oberpfaffenhofen product is distributed as a T6 matrix, which is the
polarimetric interferometric coherency of a repeat-pass pair. Its upper-left
3x3 block is the coherency of the first acquisition and its lower-right block
that of the second; both are valid monostatic coherency matrices in the same
Pauli basis, coregistered, and they share one ground truth map. The rest of this
paper uses only the first block. This experiment uses both.

Two things are measured:

  1. the orientation angle difference between the acquisitions, which is a real
     physical quantity and not something we impose
  2. transfer accuracy -- train on one acquisition, test on the other -- for
     every method, so that whatever the orientation difference turns out to be,
     its consequence is measured rather than assumed

The normalisation constants are taken from the TRAINING acquisition only, so
nothing about the test acquisition is used before evaluation.

Resumable into exp33_results.json.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import json, os, sys, time, scipy.io as sio
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from eqnorm import EqNorm
from polsar_lib import CConv, CLin, crelu, cpool, rot6_torch
from equivariant import EqCVCNN_F
from steerable import SteerNet, field_stats
from polsar_data import bdist
from exp27_oac import CVCNNb

ROOT = ("c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/PolSAR Data/"
        "Oberpfaffenhofen")
P6 = ROOT + "/ESAR_Oberpfaffenhofen_T6"
R, C, NCL = 1300, 1200, 3
BUDGET = 666
SEEDS = (0, 1, 2)
W = 15
STORE = "exp33_results.json"


def _rd(n):
    return np.fromfile(os.path.join(P6, n), dtype="<f4").reshape(R, C)


def block(d1, d2, d3, ab, ac, bc):
    X = np.empty((R, C, 6), np.complex64)
    X[..., 0] = _rd(d1); X[..., 1] = _rd(d2); X[..., 2] = _rd(d3)
    X[..., 3] = _rd(ab + "_real.bin") + 1j * _rd(ab + "_imag.bin")
    X[..., 4] = _rd(ac + "_real.bin") + 1j * _rd(ac + "_imag.bin")
    X[..., 5] = _rd(bc + "_real.bin") + 1j * _rd(bc + "_imag.bin")
    return X


def poa(X):
    z = (X[..., 1].real.astype(np.float64) - X[..., 2].real.astype(np.float64)) / 2 \
        + 1j * X[..., 5].real.astype(np.float64)
    return -np.angle(z) / 4.0, np.abs(z)


class TwoAcq:
    """Holds both acquisitions; the normalisation is fixed to the training one."""

    def __init__(s, A, B, gt, dev="cuda"):
        s.dev = dev; s.W = W; s.M = W // 2
        s.gt = gt; s.ncl = NCL
        s.lr, s.lc = np.nonzero(gt > 0); s.y = gt[s.lr, s.lc] - 1
        s.dist = bdist(gt)
        s.eqn = EqNorm(A).to(dev)          # constants from acquisition A only
        s.stats = field_stats(A)
        s.pad = {}
        for k, X in (("A", A), ("B", B)):
            Xp = np.pad(X, ((s.M, s.M), (s.M, s.M), (0, 0)), mode="constant")
            s.pad[k] = (torch.from_numpy(np.ascontiguousarray(Xp.real)).to(dev),
                        torch.from_numpy(np.ascontiguousarray(Xp.imag)).to(dev))
        s.off = torch.arange(W, device=dev)
        s.which = "A"; s.raw = False; s.oac = None

    def grab(s, r, c, th=None):
        Pr, Pi = s.pad[s.which]
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        if s.oac:
            zr = Fn.avg_pool2d(((xr[:, 1] - xr[:, 2]) / 2).unsqueeze(1),
                               W, stride=1).squeeze(1)[:, 0, 0]
            zi = Fn.avg_pool2d(xr[:, 5].unsqueeze(1), W, stride=1).squeeze(1)[:, 0, 0]
            xr, xi = rot6_torch(xr, xi, torch.atan2(zi, zr) / 4.0)
        return (xr, xi) if s.raw else s.eqn(xr, xi)

    def cap_eval(s, idx, cap=150_000):
        if len(idx) <= cap:
            return idx
        g = np.random.default_rng(20260827)
        return np.sort(g.choice(idx, cap, replace=False))


def run(P, kind, seed, aug=False, oac=False, epochs=120):
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET, int((P.y == k).sum())),
                                    replace=False) for k in range(P.ncl)])
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = {"base": lambda: CVCNNb(P.ncl, cin=7),
           "eqf": lambda: EqCVCNN_F(P.ncl, cin=7, N=8),
           "steer": lambda: SteerNet(P.ncl, stats=P.stats)}[kind]().cuda()
    P.raw = (kind == "steer"); P.oac = oac
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train()
    P.which = "A"
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), 128):
            b = pm[s:s + 128]
            th = (torch.rand(len(b), device="cuda") * np.pi) if aug else None
            xr, xi = P.grab(Rt[b], Ct[b], th)
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    t = P.y[ite]; out = {}
    with torch.no_grad():
        for w in ("A", "B"):
            P.which = w
            pr = np.empty(len(ite), np.int64)
            for s in range(0, len(ite), 4096):
                xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096])
                pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
            out[w] = float(100 * (pr == t).mean())
    P.raw = False; P.oac = None
    del net; torch.cuda.empty_cache()
    return out


ARMS = [("baseline", "base", False, False),
        ("baseline + rot. aug.", "base", True, False),
        ("CV-CNN + OAC", "base", False, True),
        ("Equivariant", "eqf", False, False),
        ("Steerable", "steer", False, False)]


def main():
    A = block("T11.bin", "T22.bin", "T33.bin", "T12", "T13", "T23")
    B = block("T44.bin", "T55.bin", "T66.bin", "T45", "T46", "T56")
    gt = sio.loadmat(ROOT + "/Oberpfaffenhofen_gt.mat")["gt"].astype(np.int64)

    R_ = json.load(open(STORE)) if os.path.exists(STORE) else {}
    if "orientation" not in R_:
        pA, mA = poa(A); pB, mB = poa(B)
        m = (gt > 0) & (mA > np.percentile(mA[gt > 0], 50)) \
                     & (mB > np.percentile(mB[gt > 0], 50))
        d = np.degrees((pB - pA + np.pi / 4) % (np.pi / 2) - np.pi / 4)[m]
        R_["orientation"] = dict(n=int(m.sum()), median=float(np.median(d)),
                                 median_abs=float(np.median(np.abs(d))),
                                 sd=float(d.std()),
                                 frac_gt2=float((np.abs(d) > 2).mean()),
                                 frac_gt5=float((np.abs(d) > 5).mean()),
                                 frac_gt10=float((np.abs(d) > 10).mean()))
        json.dump(R_, open(STORE, "w"), indent=1)
    o = R_["orientation"]
    print("real orientation shift between the two acquisitions: "
          "median |d| %.2f deg, sd %.2f deg, %.1f %% above 5 deg"
          % (o["median_abs"], o["sd"], 100 * o["frac_gt5"]), flush=True)

    P = TwoAcq(A, B, gt)
    del A, B
    print("\n%-22s %8s %8s %8s" % ("", "acq. 1", "acq. 2", "drop"), flush=True)
    for nm, kind, aug, oac in ARMS:
        if nm in R_:
            continue
        t0 = time.time()
        res = [run(P, kind, sd, aug=aug, oac=oac) for sd in SEEDS]
        a = float(np.mean([r["A"] for r in res]))
        b = float(np.mean([r["B"] for r in res]))
        R_[nm] = dict(acq1=a, acq2=b,
                      sd1=float(np.std([r["A"] for r in res])),
                      sd2=float(np.std([r["B"] for r in res])))
        json.dump(R_, open(STORE, "w"), indent=1)
        print("%-22s %8.2f %8.2f %8.2f   (%.0fs)"
              % (nm, a, b, a - b, time.time() - t0), flush=True)
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
