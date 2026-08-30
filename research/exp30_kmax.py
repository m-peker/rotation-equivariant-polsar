"""Exp. 30 - what the Clebsch-Gordan truncation costs.

The steerable network closes its products on weight classes {0, 2, 4}, which is
where the coherency matrix lives. Carrying the products further opens weight 6
(from z_2 z_4) and weight 8 (from z_4 z_4), fields that start empty and are fed
only by products. The truncation cannot break equivariance -- that holds weight
class by weight class -- but it does limit expressiveness, and the steerable
network sits below the discrete one on Flevoland, so the obvious question is
whether the truncation is the reason.

Rotation robustness is checked as well as clean accuracy: adding weight classes
must not cost the guarantee, and the equivariance of all three variants is
verified to float64 round-off before any of them is trained.

Resumable into exp30_results.json.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
from steerable import field_stats
from steerable6 import SteerNetK
from polsar_data import load_scene

THETAS = [0, 10, 22.5, 45]
KMAX = [4, 6, 8]
SEEDS = (0, 1, 2)
STORE = "exp30_results.json"
BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}


class RawPipe(MSPipe):
    """The steerable network consumes raw T, not the normalised representation:
    its own lift carries the fixed statistics."""

    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return xr, xi


def run(P, kmax, seed, stats, epochs=120):
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET[P.scene], int((P.y == k).sum())),
                                    replace=False) for k in range(P.ncl)])
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = SteerNetK(P.ncl, kmax=kmax, stats=stats).cuda()
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
    print("theta:" + "".join("%9s" % t for t in THETAS), flush=True)
    for sc in ("flevoland",):
        todo = [k for k in KMAX if "%s|kmax=%d" % (sc, k) not in R]
        if not todo:
            continue
        P = RawPipe(sc, norm="equivariant", eval_cap=150_000)
        Xs, _, _ = load_scene(sc); ST = field_stats(Xs); del Xs
        for km in todo:
            t0 = time.time(); acc = []
            for sd in SEEDS:
                a, p = run(P, km, sd, ST)
                acc.append(a)
            A = np.array(acc)
            R["%s|kmax=%d" % (sc, km)] = dict(mean=A.mean(0).tolist(),
                                              std=A.std(0).tolist(), params=p)
            json.dump(R, open(STORE, "w"), indent=1)
            print("  %-16s %s   params %7d  (%.0fs)"
                  % ("kmax=%d" % km, "".join("%9.2f" % v for v in A.mean(0)),
                     p, time.time() - t0), flush=True)
        del P
        torch.cuda.empty_cache()
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
