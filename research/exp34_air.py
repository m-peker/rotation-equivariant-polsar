"""Exp. 34 - the whole comparison on AIR-PolSAR-Seg-2.0.

The three classic scenes are saturated: published accuracies on Flevoland sit
above 98 per cent and differences between methods are within seed noise. That is
the reviewers' objection to the accuracy argument of this paper, and it is a
fair one. This runs the same comparison on a benchmark that is not saturated
and not small: three GF-3 quad-pol acquisitions over Beijing, Shanghai and
Guangzhou, 3.7 million labelled pixels against Flevoland's 208 thousand, five
classes, heavily imbalanced, with irregular boundaries.

The scope of what this scene can support is limited and stated in air2.py: the
archive is L1A and ships no calibration constants, the channels are measurably
unbalanced, and so the rotation we apply here is a group action on the matrix we
hold rather than a physical target rotation. That is enough for what is asked of
it. Whether a network is equivariant to the action is a property of the network;
whether a benchmark is saturated is a property of the benchmark.

Resumable into exp34_results.json.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
from equivariant import EqCVCNN_F
from steerable import SteerNet, field_stats
from polsar_data import load_scene
from exp27_oac import OACPipe, CVCNNb
from cvmsatvit import CVMsAtViT

THETAS = [0, 10, 22.5, 45]
SCENES = ("air_gz", "air_sh", "air_bj")
BUDGET = 400          # labels per class; five classes -> 2000, as elsewhere
SEEDS = (0, 1, 2)
STORE = "exp34_results.json"


class RawPipe(MSPipe):
    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return xr, xi


def run(P, Praw, Poac, kind, aug, oac, seed, stats, epochs=120):
    pipe = Praw if kind == "steer" else (Poac if oac else P)
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
           "steer": lambda: SteerNet(P.ncl, stats=stats),
           "msat": lambda: CVMsAtViT(P.ncl, cin=7)}[kind]().cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), 128):
            b = pm[s:s + 128]
            th = (torch.rand(len(b), device="cuda") * np.pi) if aug else None
            xr, xi = pipe.grab(Rt[b], Ct[b], th)
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    t = P.y[ite]; out = []; aa = None
    with torch.no_grad():
        for deg in THETAS:
            pr = np.empty(len(ite), np.int64)
            for s in range(0, len(ite), 4096):
                n = len(Re[s:s + 4096])
                th = (torch.full((n,), np.deg2rad(deg), device="cuda")
                      if deg else None)
                xr, xi = pipe.grab(Re[s:s + 4096], Ce[s:s + 4096], th)
                pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
            ok = pr == t
            out.append(float(100 * ok.mean()))
            if deg == 0:
                aa = float(100 * np.nanmean([ok[t == k].mean()
                                             for k in range(P.ncl)]))
    p = sum(q.numel() for q in net.parameters())
    del net; torch.cuda.empty_cache()
    return out, aa, p


ARMS = [("baseline", "base", False, False),
        ("baseline + rot. aug.", "base", True, False),
        ("CV-CNN + OAC", "base", False, True),
        ("Equivariant", "eqf", False, False),
        ("Steerable", "steer", False, False),
        ("CV-MsAtViT", "msat", False, False)]


def main():
    R = json.load(open(STORE)) if os.path.exists(STORE) else {}
    print("theta:" + "".join("%9s" % t for t in THETAS), flush=True)
    for sc in SCENES:
        todo = [a for a in ARMS if "%s|%s" % (sc, a[0]) not in R]
        if not todo:
            print("### %s cached ###" % sc, flush=True)
            continue
        P = MSPipe(sc, norm="equivariant", eval_cap=150_000)
        Praw = RawPipe(sc, norm="equivariant", eval_cap=150_000)
        Poac = OACPipe(sc, norm="equivariant", eval_cap=150_000, oac="smooth")
        Xs, _, _ = load_scene(sc); ST = field_stats(Xs); del Xs
        print("\n### %s  (%d labelled, %d classes) ###"
              % (sc, len(P.y), P.ncl), flush=True)
        for nm, kind, aug, oac in todo:
            t0 = time.time(); acc = []; aas = []
            for sd in SEEDS:
                a, aa, p = run(P, Praw, Poac, kind, aug, oac, sd, ST)
                acc.append(a); aas.append(aa)
            A = np.array(acc)
            R["%s|%s" % (sc, nm)] = dict(mean=A.mean(0).tolist(),
                                         std=A.std(0).tolist(),
                                         aa=float(np.mean(aas)), params=p)
            json.dump(R, open(STORE, "w"), indent=1)
            print("  %-22s %s  AA %5.2f  drop %5.2f  (%.0fs)"
                  % (nm, "".join("%9.2f" % v for v in A.mean(0)),
                     np.mean(aas), A.mean(0)[0] - A.mean(0).min(),
                     time.time() - t0), flush=True)
        del P, Praw, Poac
        torch.cuda.empty_cache()
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
