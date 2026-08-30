"""Exp. 15 - MAIN TABLE: 3 scenes x 4 models, trace-normalised + log-power input.

Resumable replacement for exp15_main.py. Every (scene, arm) result is written to
exp15_results.json the moment it finishes, and completed combinations are skipped
on restart, so stopping the job costs at most one arm instead of the whole run.

Protocol: class-balanced budget of ~2000 labels in total, so the comparison is
fair across scenes and the training cost stays constant:
    flevoland 133x15, sanfran 400x5, ober 666x3
Evaluation: all remaining labelled pixels, capped at 200k.
TF32 is disabled so invariance at the group angles shows up exactly.
"""
import numpy as np, torch, torch.nn as nn, time, sys, json, os
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool
from equivariant import EqCVCNN_F
from cvmsatvit import CVMsAtViT
from steerable import SteerNet, field_stats
from polsar_data import load_scene
from polsar_lib import rot6_torch
import pipe_ms

THETAS = [0, 10, 22.5, 45]
ONG = {0, 22.5, 45}
BINS = [(1, 2), (3, 4), (5, 7), (8, 11), (12, 99)]
BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}
SEEDS = (0, 1, 2)
STORE = "exp15_results.json"


class CVCNNb(nn.Module):
    def __init__(s, ncl, cin=7, w=1):
        super().__init__()
        a, b, c = 32 * w, 64 * w, 128 * w
        s.c1 = CConv(cin, a); s.c2 = CConv(a, b); s.c3 = CConv(b, c)
        s.f1 = CLin(c * 3 * 3, 128 * w); s.f2 = CLin(128 * w, ncl)
        s.do = nn.Dropout(0.3)

    def forward(s, xr, xi):
        xr, xi = crelu(*s.c1(xr, xi))
        xr, xi = crelu(*s.c2(xr, xi)); xr, xi = cpool(xr, xi)
        xr, xi = crelu(*s.c3(xr, xi)); xr, xi = cpool(xr, xi)
        xr = s.do(xr.flatten(1)); xi = s.do(xi.flatten(1))
        xr, xi = crelu(*s.f1(xr, xi)); xr, xi = s.f2(xr, xi)
        return torch.sqrt(xr ** 2 + xi ** 2 + 1e-9)


class RawPipe(pipe_ms.MSPipe):
    """Steerable consumes raw T3 and performs its own decomposition."""
    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return xr, xi


def build(kind, ncl):
    return {"base": lambda: CVCNNb(ncl, cin=7),
            "eqf": lambda: EqCVCNN_F(ncl, cin=7, N=8),
            "steer": lambda: SteerNet(ncl, stats=STATS),
            "msat": lambda: CVMsAtViT(ncl, cin=7)}[kind]()


def sample(P, n, seed):
    rng = np.random.default_rng(1000 + seed)
    tr = [rng.choice(np.nonzero(P.y == k)[0],
                     min(n, int((P.y == k).sum())), replace=False)
          for k in range(P.ncl)]
    tr = np.concatenate(tr)
    te = np.setdiff1d(np.arange(len(P.y)), tr)
    return tr, P.cap_eval(te)


def run(P, kind, aug, seed, epochs=120, bs=128, Praw=None):
    pipe = Praw if kind == "steer" else P
    itr, ite = sample(P, BUDGET[P.scene], seed)
    Rt = torch.from_numpy(P.lr[itr]).cuda()
    Ct = torch.from_numpy(P.lc[itr]).cuda()
    Yt = torch.from_numpy(P.y[itr]).cuda()
    torch.manual_seed(seed)
    net = build(kind, P.ncl).cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss()
    net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(itr), device="cuda")
        for s in range(0, len(itr), bs):
            b = pm[s:s + bs]
            th = (torch.rand(len(b), device="cuda") * np.pi) if aug else None
            xr, xi = pipe.grab(Rt[b], Ct[b], th)
            opt.zero_grad()
            lf(net(xr, xi), Yt[b]).backward()
            opt.step()
        sch.step()
    net.eval()
    Re = torch.from_numpy(P.lr[ite]).cuda()
    Ce = torch.from_numpy(P.lc[ite]).cuda()
    accs, bnd, aa0 = [], None, None
    for tdeg in THETAS:
        th = (torch.tensor(np.deg2rad(tdeg), device="cuda", dtype=torch.float32)
              if tdeg else None)
        pr = np.empty(len(ite), np.int64)
        with torch.no_grad():
            for s in range(0, len(ite), 4096):
                xr, xi = pipe.grab(Re[s:s + 4096], Ce[s:s + 4096], th)
                pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
        t = P.y[ite]
        ok = pr == t
        accs.append(float(100 * ok.mean()))
        if tdeg == 0:
            d = P.dist[P.lr[ite], P.lc[ite]]
            bnd = [float(100 * ok[(d >= lo) & (d <= hi)].mean())
                   if ((d >= lo) & (d <= hi)).sum() > 0 else float("nan")
                   for lo, hi in BINS]
            aa0 = float(100 * np.mean([ok[t == k].mean()
                                       for k in range(P.ncl) if (t == k).sum() > 0]))
    npar = sum(p.numel() for p in net.parameters())
    del net
    torch.cuda.empty_cache()
    return accs, bnd, aa0, npar


ARMS = [("baseline no-aug", "base", False),
        ("baseline rot-AUG", "base", True),
        ("Equivariant", "eqf", False),
        ("Steerable", "steer", False),
        ("CV-MsAtViT", "msat", False)]

R = json.load(open(STORE)) if os.path.exists(STORE) else {}
print("theta:" + "".join("%9s" % t for t in THETAS) +
      "   (0/22.5/45 on-grid)  TF32 off", flush=True)
if R:
    print("resuming: %d arm(s) already stored" % len(R), flush=True)

for sc in ("flevoland", "sanfran", "ober"):
    todo = [a for a in ARMS
            if not (a[1] == "msat" and sc != "flevoland")
            and "%s|%s" % (sc, a[0]) not in R]
    if not todo:
        print("\n### %s: already complete, skipped ###" % sc, flush=True)
        continue
    P = MSPipe(sc, norm="equivariant", eval_cap=150_000)
    Praw = RawPipe(sc, norm="equivariant", eval_cap=150_000)
    Xs, _, _ = load_scene(sc)
    globals()["STATS"] = field_stats(Xs)
    del Xs
    print("\n### %s   budget=%d/class ###" % (P.info(), BUDGET[sc]), flush=True)
    for nm, k, aug in todo:
        A, B, AAv = [], [], []
        t0 = time.time()
        for sd in SEEDS:
            a, b, aa, npar = run(P, k, aug, sd, Praw=Praw)
            A.append(a); B.append(b); AAv.append(aa)
        A = np.array(A); B = np.array(B, dtype=float)
        R["%s|%s" % (sc, nm)] = dict(mean=A.mean(0).tolist(), std=A.std(0).tolist(),
                                     bnd=B.mean(0).tolist(), aa=float(np.mean(AAv)),
                                     params=int(npar))
        json.dump(R, open(STORE, "w"), indent=1)          # persisted immediately
        print("  %-19s" % nm + "".join("%9.2f" % m for m in A.mean(0)) +
              "  AA=%6.2f bnd=%6.2f p=%8d (%.0fs)"
              % (np.mean(AAv), B.mean(0)[0], npar, time.time() - t0), flush=True)
    del P, Praw
    torch.cuda.empty_cache()

print("\n\n================ SUMMARY ================")
print("%-11s%-19s%8s%7s%13s%8s%8s%8s%10s"
      % ("scene", "model", "th=0", "AA", "on-grid min", "th=10", "drop", "bnd", "params"))
for key in sorted(R):
    sc, nm = key.split("|")
    v = R[key]
    m = np.array(v["mean"])
    ong = min(m[i] for i, t in enumerate(THETAS) if t in ONG)
    print("%-11s%-19s%8.2f%7.2f%13.2f%8.2f%8.2f%8.2f%10d"
          % (sc, nm, m[0], v["aa"], ong, m[THETAS.index(10)], m[0] - m.min(),
             v["bnd"][0], v["params"]))
