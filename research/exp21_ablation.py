"""Exp. 21 - ablation of the proposed components.

Five axes, all on Flevoland, three seeds, resumable into exp21_results.json.

  A. orientations N in {4, 8, 16}          -- how finely must the group be sampled?
  B. readout: max vs Fourier magnitudes    -- does the richer invariant pay?
  C. normalisation: irrep vs per-channel   -- the claim of Section III-C, tested
     (both seven-channel, so only the commutation property differs; measured
      deviation under rotation is 2.7e-07 against 3.8e-02)
  D. label budget                          -- behaviour as supervision shrinks
  E. patch size                            -- spatial context

C is the one that matters most. We assert that per-channel standardisation
breaks equivariance; running the same network under it, and measuring both the
accuracy and the deviation under rotation, turns that assertion into a result.

Patch size varies the spatial dimensions, so the head uses adaptive pooling to a
fixed 3x3 grid; this keeps the parameter count constant across E and is applied
in every arm so the comparison stays internally consistent.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
import equivariant as EQ

SCENE = "flevoland"
SEEDS = (0, 1, 2)
STORE = "exp21_results.json"
THETAS = [0, 10, 45]


class EqAda(nn.Module):
    """EqCVCNN_F with an adaptive spatial pool, so patch size is a free knob."""
    def __init__(s, ncl, cin=7, N=8, c=24, c0=16, K=3, readout="fourier"):
        super().__init__()
        s.N, s.K, s.readout = N, K, readout
        a, b, cc = 16, 24, 32
        s.lift = EQ.CLift(cin, a, N)
        s.g1 = EQ.CGConvFFT(a, b, N)
        s.g2 = EQ.CGConvFFT(b, cc, N)
        nin = (K * cc if readout == "fourier" else cc) * 3 * 3
        s.f1 = nn.Linear(nin, 128); s.f2 = nn.Linear(128, ncl)
        s.do = nn.Dropout(0.3)

    def forward(s, xr, xi):
        xr, xi = EQ.gcrelu(*s.lift(xr, xi))
        xr, xi = EQ.gcrelu(*s.g1(xr, xi)); xr, xi = EQ.gpool_sp(xr, xi)
        xr, xi = EQ.gcrelu(*s.g2(xr, xi))
        B, N, C, H, W = xr.shape
        f = lambda t: F.adaptive_avg_pool2d(t.reshape(B * N, C, H, W), 3) \
                       .reshape(B, N, C, 3, 3)
        xr, xi = f(xr), f(xi)
        if s.readout == "fourier":
            v = EQ.pool_group_fourier(xr, xi, s.K)
        else:
            a, b = EQ.pool_group(xr, xi)
            v = torch.sqrt(a ** 2 + b ** 2 + 1e-12)
        return s.f2(torch.relu(s.f1(s.do(v.flatten(1)))))


def evaluate(net, P, ite, W):
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    out = {}
    for td in THETAS:
        th = (torch.tensor(np.deg2rad(td), device="cuda", dtype=torch.float32)
              if td else None)
        pr = np.empty(len(ite), np.int64)
        with torch.no_grad():
            for s in range(0, len(ite), 4096):
                xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096], th)
                pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
        out[td] = float(100 * (pr == P.y[ite]).mean())
    return out


def run(P, seed, N=8, readout="fourier", budget=133, epochs=120):
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(budget, int((P.y == k).sum())), replace=False)
                         for k in range(P.ncl)])
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = EqAda(P.ncl, cin=7, N=N, readout=readout).cuda()
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
    acc = evaluate(net, P, ite, P.W)
    npar = sum(p.numel() for p in net.parameters())
    del net; torch.cuda.empty_cache()
    return acc, npar


R = json.load(open(STORE)) if os.path.exists(STORE) else {}
PIPES = {}


def pipe(norm, W):
    key = (norm, W)
    if key not in PIPES:
        PIPES[key] = MSPipe(SCENE, norm=norm, W=W, eval_cap=100_000)
    return PIPES[key]


def do(tag, **kw):
    if tag in R:
        print("  %-26s cached" % tag, flush=True)
        return
    norm = kw.pop("norm", "equivariant")
    W = kw.pop("W", 15)
    P = pipe(norm, W)
    A, t0 = [], time.time()
    for sd in SEEDS:
        a, npar = run(P, sd, **kw)
        A.append(a)
    m = {str(t): float(np.mean([a[t] for a in A])) for t in THETAS}
    sd_ = float(np.std([a[0] for a in A]))
    R[tag] = dict(mean=m, sd0=sd_, params=int(npar), norm=norm, W=W, **{k: v for k, v in kw.items()})
    json.dump(R, open(STORE, "w"), indent=1)
    print("  %-26s th0=%6.2f+-%.2f  th10=%6.2f  th45=%6.2f  p=%d  (%.0fs)"
          % (tag, m["0"], sd_, m["10"], m["45"], npar, time.time() - t0), flush=True)


def main():
    print("=== A. orientations ===", flush=True)
    for N in (4, 8, 16):
        do("A|N=%d" % N, N=N)

    print("=== B. readout ===", flush=True)
    do("B|max", readout="max")
    do("B|fourier", readout="fourier")

    print("=== C. normalisation (the Section III-C claim) ===", flush=True)
    do("C|irrep", norm="equivariant")
    # Channel count matched: the equivariant normalisation carries a log-power
    # channel, so a six-channel per-channel baseline would confound the broken
    # commutation with the missing channel. "channel7" adds the same channel.
    do("C|per-channel", norm="channel7")

    print("=== D. label budget ===", flush=True)
    for bg in (10, 25, 50, 133, 300):
        do("D|budget=%d" % bg, budget=bg)

    print("=== E. patch size ===", flush=True)
    for W in (7, 11, 15, 19):
        do("E|W=%d" % W, W=W)

    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
