"""Exp. 18 - separating leakage removal from covariate shift.

exp17 measured a +18.9 pp gap on Flevoland between training adjacent to the test
region and training in a spatially disjoint region. Per-seed the clean arm gave
65.94 and 88.25, so the mean is not usable as stated: the gap conflates two
effects.

  (i)  leakage removal - test patches no longer share input with training patches
  (ii) covariate shift - on a scene of small, spatially concentrated fields, a
       block-level split can leave a class represented by one or two fields,
       so the model must generalise to unseen fields of the same class

This script (a) runs more seeds to get an honest interval, and (b) records, for
every seed and class, how many distinct connected field components the training
partition covers. If the clean-arm accuracy tracks field coverage, effect (ii)
dominates and must be reported as such.
"""
import numpy as np, torch, torch.nn as nn, json, time, sys
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from scipy.ndimage import label as cc_label
from pipe_ms import MSPipe
from polsar_lib import CConv, CLin, crelu, cpool
from disjoint import block_buffer_split, verify_no_overlap

BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}
BLOCK = {"flevoland": 96, "sanfran": 128, "ober": 160}
SEEDS = (0, 1, 2, 3, 4)
BINS = [(1, 2), (12, 99)]


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


def field_components(gt, ncl):
    """Connected components per class = distinct fields."""
    comp = {}
    for k in range(1, ncl + 1):
        lab, n = cc_label(gt == k)
        comp[k] = (lab, n)
    return comp


def coverage(gt, mask, comp, ncl):
    """Fraction of a class's fields that the mask touches, averaged over classes."""
    fr = []
    for k in range(1, ncl + 1):
        lab, n = comp[k]
        if n == 0:
            continue
        touched = len(np.unique(lab[mask & (lab > 0)]))
        fr.append(touched / n)
    return float(np.mean(fr))


def pick(gt, mask, ncl, n, seed):
    rng = np.random.default_rng(seed)
    rs, cs = [], []
    for k in range(1, ncl + 1):
        r, c = np.nonzero((gt == k) & mask)
        if len(r) == 0:
            continue
        p = rng.choice(len(r), min(n, len(r)), replace=False)
        rs.append(r[p]); cs.append(c[p])
    return np.concatenate(rs), np.concatenate(cs)


def train_eval(P, tr_rc, te_rc, seed, epochs=120):
    r, c = tr_rc
    Rt = torch.from_numpy(r).cuda(); Ct = torch.from_numpy(c).cuda()
    Yt = torch.from_numpy(P.gt[r, c] - 1).cuda()
    torch.manual_seed(seed)
    net = CVCNNb(P.ncl, cin=7).cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train(); n = len(r)
    for ep in range(epochs):
        pm = torch.randperm(n, device="cuda")
        for s in range(0, n, 128):
            b = pm[s:s + 128]
            xr, xi = P.grab(Rt[b], Ct[b])
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    er, ec = te_rc
    Re = torch.from_numpy(er).cuda(); Ce = torch.from_numpy(ec).cuda()
    pr = np.empty(len(er), np.int64)
    with torch.no_grad():
        for s in range(0, len(er), 4096):
            xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096])
            pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
    ok = pr == (P.gt[er, ec] - 1)
    del net; torch.cuda.empty_cache()
    return float(100 * ok.mean())


OUT = {}
for sc in ("flevoland", "sanfran", "ober"):
    P = MSPipe(sc, norm="equivariant", eval_cap=120_000)
    comp = field_components(P.gt, P.ncl)
    nfields = sum(comp[k][1] for k in comp)
    print("\n### %s   %d classes, %d distinct fields ###" % (sc, P.ncl, nfields), flush=True)
    rows = []
    for seed in SEEDS:
        trm, tem, _ = block_buffer_split(P.gt, block=BLOCK[sc], test_frac=0.7, seed=seed)
        te = np.nonzero(tem)
        rng = np.random.default_rng(seed)
        if len(te[0]) > 120_000:
            s = rng.choice(len(te[0]), 120_000, replace=False)
            te = (te[0][s], te[1][s])
        leak = (P.gt > 0) & (~tem)
        covA = coverage(P.gt, leak, comp, P.ncl)
        covB = coverage(P.gt, trm, comp, P.ncl)
        a = train_eval(P, pick(P.gt, leak, P.ncl, BUDGET[sc], seed), te, seed)
        b = train_eval(P, pick(P.gt, trm, P.ncl, BUDGET[sc], seed), te, seed)
        rows.append((seed, a, b, covA, covB))
        print("   seed%d  leaky=%6.2f  clean=%6.2f  gap=%+6.2f   field-coverage %.2f -> %.2f"
              % (seed, a, b, a - b, covA, covB), flush=True)
    A = np.array([r[1] for r in rows]); B = np.array([r[2] for r in rows])
    CB = np.array([r[4] for r in rows])
    OUT[sc] = dict(rows=rows, leaky=A.tolist(), clean=B.tolist(), covB=CB.tolist())
    print("   mean  leaky=%.2f+-%.2f  clean=%.2f+-%.2f  gap=%.2f+-%.2f"
          % (A.mean(), A.std(), B.mean(), B.std(), (A - B).mean(), (A - B).std()), flush=True)
    if len(set(np.round(CB, 3))) > 1:
        r = np.corrcoef(CB, B)[0, 1]
        print("   corr(field coverage, clean acc) = %+.2f" % r, flush=True)
    del P; torch.cuda.empty_cache()

json.dump(OUT, open("exp18_results.json", "w"), indent=1)
print("\nwritten to exp18_results.json")
