"""Exp. 26 - per-class accuracy under rotation, Flevoland.

Two statements in the manuscript are about single classes under rotation: that
Buildings collapses when the baseline is rotated, and that augmentation recovers
it. Both predate the current measurement files, so neither is checkable by the
audit. This measures them.

Same training as Exp. 23; the only change is that the test patches are rotated
by theta before the forward pass, so the arms are comparable cell by cell with
the theta = 0 table.

Resumable into exp26_results.json.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool
from equivariant import EqCVCNN_F

SCENE = "flevoland"
BUDGET = 133
SEEDS = (0, 1, 2)
THETAS = (0.0, 45.0)
STORE = "exp26_results.json"
CLASSES = ["Water", "Forest", "Lucerne", "Grass", "Rapeseed", "Beet",
           "Potatoes", "Peas", "Stem beans", "Bare soil", "Wheat",
           "Wheat 2", "Wheat 3", "Barley", "Buildings"]


class CVCNNb(nn.Module):
    """The same baseline as Exp. 23, repeated here so this script stands alone."""

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


def run(P, kind, aug, seed, epochs=120):
    """Train one arm, then score it at every angle in THETAS."""
    rng = np.random.default_rng(1000 + seed)
    tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                    min(BUDGET, int((P.y == k).sum())),
                                    replace=False) for k in range(P.ncl)])
    ite = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
    Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
    Yt = torch.from_numpy(P.y[tr]).cuda()
    torch.manual_seed(seed)
    net = (CVCNNb(P.ncl, cin=7) if kind == "base"
           else EqCVCNN_F(P.ncl, cin=7, N=8)).cuda()
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
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    t = P.y[ite]
    out = {}
    with torch.no_grad():
        for deg in THETAS:
            pr = np.empty(len(ite), np.int64)
            for s in range(0, len(ite), 4096):
                n = len(Re[s:s + 4096])
                th = (torch.full((n,), np.deg2rad(deg), device="cuda")
                      if deg else None)
                xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096], th)
                pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
            ok = pr == t
            out[deg] = ([float(100 * ok[t == k].mean()) if (t == k).sum()
                         else float("nan") for k in range(P.ncl)],
                        float(100 * ok.mean()))
    del net; torch.cuda.empty_cache()
    return out


ARMS = [("baseline", "base", False), ("baseline + rot. aug.", "base", True),
        ("Equivariant", "eqf", False)]

R = json.load(open(STORE)) if os.path.exists(STORE) else {}
todo = [a for a in ARMS if a[0] not in R]
if todo:
    P = MSPipe(SCENE, norm="equivariant", eval_cap=150_000)
    for nm, k, aug in todo:
        t0 = time.time()
        acc = {d: [] for d in THETAS}; oa = {d: [] for d in THETAS}
        for sd in SEEDS:
            o = run(P, k, aug, sd)
            for d in THETAS:
                acc[d].append(o[d][0]); oa[d].append(o[d][1])
        R[nm] = {str(d): dict(per=np.nanmean(acc[d], 0).tolist(),
                              per_sd=np.nanstd(acc[d], 0).tolist(),
                              oa=float(np.mean(oa[d]))) for d in THETAS}
        json.dump(R, open(STORE, "w"), indent=1)
        print("  %-21s OA0=%6.2f OA45=%6.2f  Buildings %6.2f -> %6.2f  (%.0fs)"
              % (nm, R[nm]["0.0"]["oa"], R[nm]["45.0"]["oa"],
                 R[nm]["0.0"]["per"][14], R[nm]["45.0"]["per"][14],
                 time.time() - t0), flush=True)
else:
    print("all arms cached")

print()
print("%-12s %s" % ("class", "  ".join("%-16s" % a[0] for a in ARMS)))
for i, c in enumerate(CLASSES):
    print("%-12s %s" % (c, "  ".join(
        "%6.2f -> %6.2f" % (R[a[0]]["0.0"]["per"][i], R[a[0]]["45.0"]["per"][i])
        for a in ARMS)))
print("\nstored in " + STORE)
