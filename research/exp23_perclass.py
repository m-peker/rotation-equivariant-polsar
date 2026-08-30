"""Exp. 23 - per-class accuracy for the headline models.

Standard content in this literature and absent from our tables: average accuracy
is reported but not the breakdown, so a reader cannot see which classes a method
fails on. Only theta = 0 is evaluated here, since the rotation behaviour is
already in Table I; that keeps the run to one angle instead of four.

Resumable into exp23_results.json.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool, rot6_torch
from equivariant import EqCVCNN_F
from steerable import SteerNet, field_stats
from cvmsatvit import CVMsAtViT
from polsar_data import load_scene, SCENES
import pipe_ms

BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}
SEEDS = (0, 1, 2)
STORE = "exp23_results.json"

CLASSES = {
    "flevoland": ["Water", "Forest", "Lucerne", "Grass", "Rapeseed", "Beet",
                  "Potatoes", "Peas", "Stem beans", "Bare soil", "Wheat",
                  "Wheat 2", "Wheat 3", "Barley", "Buildings"],
    "sanfran": ["Bare soil", "Mountain", "Water", "Urban", "Vegetation"],
    "ober": ["Built-up", "Wood land", "Open areas"],
}


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


class RawPipe(pipe_ms.MSPipe):
    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return xr, xi


def run(P, Praw, kind, aug, seed, stats, epochs=120):
    pipe = Praw if kind == "steer" else P
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET[P.scene], int((P.y == k).sum())),
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
            xr, xi = pipe.grab(Rt[b], Ct[b], (torch.rand(len(b), device="cuda") * np.pi)
                               if aug else None)
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    pr = np.empty(len(ite), np.int64)
    with torch.no_grad():
        for s in range(0, len(ite), 4096):
            xr, xi = pipe.grab(Re[s:s + 4096], Ce[s:s + 4096])
            pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
    t = P.y[ite]; ok = pr == t
    per = [float(100 * ok[t == k].mean()) if (t == k).sum() else float("nan")
           for k in range(P.ncl)]
    del net; torch.cuda.empty_cache()
    return per, float(100 * ok.mean())


ARMS = [("baseline", "base", False), ("baseline + rot. aug.", "base", True),
        ("Equivariant", "eqf", False), ("Steerable", "steer", False),
        ("CV-MsAtViT", "msat", False)]

R = json.load(open(STORE)) if os.path.exists(STORE) else {}
for sc in ("flevoland", "sanfran", "ober"):
    todo = [a for a in ARMS if "%s|%s" % (sc, a[0]) not in R]
    if not todo:
        print("### %s cached ###" % sc, flush=True)
        continue
    P = MSPipe(sc, norm="equivariant", eval_cap=150_000)
    Praw = RawPipe(sc, norm="equivariant", eval_cap=150_000)
    Xs, _, _ = load_scene(sc); ST = field_stats(Xs); del Xs
    print("\n### %s ###" % sc, flush=True)
    for nm, k, aug in todo:
        acc, oa = [], []
        t0 = time.time()
        for sd in SEEDS:
            p, o = run(P, Praw, k, aug, sd, ST)
            acc.append(p); oa.append(o)
        A = np.array(acc)
        R["%s|%s" % (sc, nm)] = dict(per=np.nanmean(A, 0).tolist(),
                                     per_sd=np.nanstd(A, 0).tolist(),
                                     oa=float(np.mean(oa)),
                                     aa=float(np.nanmean(np.nanmean(A, 0))))
        json.dump(R, open(STORE, "w"), indent=1)
        worst = int(np.nanargmin(np.nanmean(A, 0)))
        print("  %-21s OA=%6.2f AA=%6.2f  worst: %s %.2f  (%.0fs)"
              % (nm, np.mean(oa), np.nanmean(np.nanmean(A, 0)),
                 CLASSES[sc][worst], np.nanmean(A, 0)[worst], time.time() - t0),
              flush=True)
    del P, Praw
    torch.cuda.empty_cache()
print("\nstored in " + STORE)
