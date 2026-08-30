"""Exp. 38 - how far do the roll-invariant features alone take you?

There is a third classical route to rotation invariance that the paper cites but
does not run: extract roll-invariant features and hand them to an ordinary
classifier~\\cite{chen2017roll}. Our own weight-0 component is exactly that set --
$T_{11}$, $\\tfrac12(T_{22}+T_{33})$, $\\Im T_{23}$ and the span -- so the
experiment costs nothing to set up and isolates what the architecture adds over
simply discarding everything that rotates.

Such a network is invariant by construction, like ours, because its input is.
The question is what that costs in accuracy: the weight-2 and weight-4 sectors
carry real information, and a method that drops them keeps only part of the
polarimetry.

Resumable into exp38_results.json.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
from steerable import decompose
from polsar_data import load_scene

BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666,
          "air_gz": 400, "air_sh": 400, "air_bj": 400}
SEEDS = (0, 1, 2)
THETAS = [0, 10, 45]
STORE = "exp38_results.json"


class InvPipe(MSPipe):
    """Returns only the weight-0 (roll-invariant) channels of the patch."""

    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        w0, _, _ = decompose(xr, xi)          # (B,4,H,W): T11, u, Im T23, log span
        return torch.clamp(w0, -8.0, 8.0)


class RealCNN(nn.Module):
    """The baseline CNN with real convolutions -- the input has no phase left."""

    def __init__(s, ncl, cin=4):
        super().__init__()
        s.c1 = nn.Conv2d(cin, 32, 3, padding=1)
        s.c2 = nn.Conv2d(32, 64, 3, padding=1)
        s.c3 = nn.Conv2d(64, 128, 3, padding=1)
        s.f1 = nn.Linear(128 * 3 * 3, 128); s.f2 = nn.Linear(128, ncl)
        s.do = nn.Dropout(0.3)

    def forward(s, x):
        x = torch.relu(s.c1(x))
        x = torch.max_pool2d(torch.relu(s.c2(x)), 2)
        x = torch.max_pool2d(torch.relu(s.c3(x)), 2)
        x = s.do(x.flatten(1))
        return s.f2(torch.relu(s.f1(x)))


def run(P, seed, epochs=120):
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET[P.scene], int((P.y == k).sum())),
                                    replace=False) for k in range(P.ncl)])
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = RealCNN(P.ncl).cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), 128):
            b = pm[s:s + 128]
            opt.zero_grad()
            lf(net(P.grab(Rt[b], Ct[b])), Yt[b]).backward(); opt.step()
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
                pr[s:s + 4096] = net(P.grab(Re[s:s + 4096], Ce[s:s + 4096],
                                            th)).argmax(1).cpu().numpy()
            ok = pr == t
            out.append(float(100 * ok.mean()))
            if deg == 0:
                aa = float(100 * np.nanmean([ok[t == k].mean()
                                             for k in range(P.ncl)]))
    p = sum(q.numel() for q in net.parameters())
    del net; torch.cuda.empty_cache()
    return out, aa, p


def main():
    R = json.load(open(STORE)) if os.path.exists(STORE) else {}
    print("%-10s %8s %8s %8s %8s %9s" % ("scene", "OA", "AA", "OA10", "OA45", "params"),
          flush=True)
    for sc in ("flevoland", "sanfran", "ober", "air_gz", "air_sh", "air_bj"):
        if sc in R:
            v = R[sc]
            print("%-10s %8.2f %8.2f %8.2f %8.2f %9d  (cached)"
                  % (sc, v["mean"][0], v["aa"], v["mean"][1], v["mean"][2],
                     v["params"]), flush=True)
            continue
        P = InvPipe(sc, norm="equivariant", eval_cap=150_000)
        t0 = time.time(); acc = []; aas = []
        for sd in SEEDS:
            a, aa, p = run(P, sd)
            acc.append(a); aas.append(aa)
        A = np.array(acc)
        R[sc] = dict(mean=A.mean(0).tolist(), aa=float(np.mean(aas)),
                     sd=float(A[:, 0].std()), params=p)
        json.dump(R, open(STORE, "w"), indent=1)
        print("%-10s %8.2f %8.2f %8.2f %8.2f %9d  (%.0fs)"
              % (sc, A.mean(0)[0], np.mean(aas), A.mean(0)[1], A.mean(0)[2],
                 p, time.time() - t0), flush=True)
        del P; torch.cuda.empty_cache()
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
