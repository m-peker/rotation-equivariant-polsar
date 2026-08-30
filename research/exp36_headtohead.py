"""Exp. 36 - where a 2.9M-parameter transformer should stop winning.

On the three classic scenes CV-MsAtViT is ahead of both proposed networks at
theta = 0, by 0.78, 1.20 and 0.52 pp. We do not dispute that and we do not think
it is surprising: those benchmarks are saturated, the training set is 2000
labelled pixels, and 84.2 per cent of the test patches overlap a training patch
(Section on leakage). Under those conditions capacity is nearly free.

Two conditions remove that freedom, and neither has been measured:

  budget   with 10 to 50 labels per class, 2.87M parameters is a liability
           rather than an advantage, and the proposed networks have 10x fewer
  disjoint under the block-and-buffer partition the test patches no longer
           overlap the training patches, so memorising them stops paying

This runs CV-MsAtViT head to head with both proposed networks along each axis.
The outcome is reported whichever way it falls; the paper's argument does not
depend on winning here, but the question is fair and currently unanswered.

Resumable into exp36_results.json.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
from equivariant import EqCVCNN_F
from steerable import SteerNet, field_stats
from cvmsatvit import CVMsAtViT
from polsar_data import load_scene
from disjoint import block_buffer_split, sample_per_class
from exp27_oac import CVCNNb

SCENE = "flevoland"
BUDGETS = [10, 25, 50, 133]
SEEDS = (0, 1, 2, 3, 4)
STORE = "exp36_results.json"
W = 15

MODELS = [("CV-CNN", "base"), ("Equivariant", "eqf"),
          ("Steerable", "steer"), ("CV-MsAtViT", "msat")]


class RawPipe(MSPipe):
    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return xr, xi


def build(kind, ncl, stats):
    return {"base": lambda: CVCNNb(ncl, cin=7),
            "eqf": lambda: EqCVCNN_F(ncl, cin=7, N=8),
            "steer": lambda: SteerNet(ncl, stats=stats),
            "msat": lambda: CVMsAtViT(ncl, cin=7)}[kind]().cuda()


def fit_eval(P, Praw, kind, tr, ite, seed, stats, epochs=120):
    pipe = Praw if kind == "steer" else P
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = build(kind, P.ncl, stats)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train()
    bs = min(128, max(8, len(tr) // 4))
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), bs):
            b = pm[s:s + bs]
            xr, xi = pipe.grab(Rt[b], Ct[b])
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    pr = np.empty(len(ite), np.int64)
    with torch.no_grad():
        for s in range(0, len(ite), 4096):
            xr, xi = pipe.grab(Re[s:s + 4096], Ce[s:s + 4096])
            pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
    ok = pr == P.y[ite]
    del net; torch.cuda.empty_cache()
    return float(100 * ok.mean())


def main():
    R = json.load(open(STORE)) if os.path.exists(STORE) else {}
    P = MSPipe(SCENE, norm="equivariant", eval_cap=150_000)
    Praw = RawPipe(SCENE, norm="equivariant", eval_cap=150_000)
    Xs, _, _ = load_scene(SCENE); ST = field_stats(Xs); del Xs

    # ---------- axis 1: label budget, standard split --------------------
    print("=== label budget, standard split ===", flush=True)
    print("%-14s %s" % ("", "".join("%9d" % b for b in BUDGETS)), flush=True)
    for nm, kind in MODELS:
        key = "budget|%s" % nm
        if key not in R:
            row, t0 = [], time.time()
            for b in BUDGETS:
                acc = []
                for sd in SEEDS:
                    rng = np.random.default_rng(1000 + sd)
                    tr = np.concatenate([
                        rng.choice(np.nonzero(P.y == k)[0],
                                   min(b, int((P.y == k).sum())), replace=False)
                        for k in range(P.ncl)])
                    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
                    acc.append(fit_eval(P, Praw, kind, tr, ite, sd, ST))
                row.append([float(np.mean(acc)), float(np.std(acc))])
            R[key] = row
            json.dump(R, open(STORE, "w"), indent=1)
            print("%-14s %s  (%.0fs)"
                  % (nm, "".join("%9.2f" % v[0] for v in row), time.time() - t0),
                  flush=True)
        else:
            print("%-14s %s  (cached)"
                  % (nm, "".join("%9.2f" % v[0] for v in R[key])), flush=True)

    # ---------- axis 2: leakage-free partition --------------------------
    # sample_per_class returns pixel coordinates; the pipeline indexes its own
    # list of labelled pixels, so map one to the other once.
    LUT = np.full(P.gt.shape, -1, np.int64)
    LUT[P.lr, P.lc] = np.arange(len(P.y))
    print("\n=== block-and-buffer partition, budget 133 ===", flush=True)
    for nm, kind in MODELS:
        key = "disjoint|%s" % nm
        if key in R:
            print("%-14s %6.2f +- %.2f  (cached)"
                  % (nm, R[key][0], R[key][1]), flush=True)
            continue
        acc, t0 = [], time.time()
        for sd in SEEDS:
            trm, tem = block_buffer_split(P.gt, W=W, seed=sd)
            r, c = sample_per_class(P.gt, trm, 133, sd)
            tr = LUT[r, c]
            assert (tr >= 0).all(), "sampled a pixel that is not in the label list"
            ite = P.cap_eval(np.nonzero(tem[P.lr, P.lc])[0])
            acc.append(fit_eval(P, Praw, kind, tr, ite, sd, ST))
        R[key] = [float(np.mean(acc)), float(np.std(acc))]
        json.dump(R, open(STORE, "w"), indent=1)
        print("%-14s %6.2f +- %.2f  (%.0fs)"
              % (nm, np.mean(acc), np.std(acc), time.time() - t0), flush=True)

    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
