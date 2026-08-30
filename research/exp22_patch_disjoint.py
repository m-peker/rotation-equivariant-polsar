"""Exp. 22 - patch size, measured where the answer is not confounded.

Under the standard random split, accuracy and leakage both grow with patch size:
at 133 labels per class on Flevoland the fraction of test pixels whose patch
overlaps a training patch is 32.2, 58.2, 76.8 and 88.0 % for 7, 11, 15 and 19.
The observed gain from a larger patch therefore cannot be separated from the
larger overlap it creates, and the usual "15x15 is best" convention rests on a
measurement that cannot support it either way.

The block-and-buffer partition removes the overlap at every patch size -- the
buffer is set to (W+1)/2 so the guarantee holds for whatever W is under test --
so the comparison becomes clean. This is a question the standard protocol cannot
answer and ours can, which is the point.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from disjoint import block_buffer_split, verify_no_overlap
from exp21_ablation import EqAda

SCENE = "flevoland"
SEEDS = (0, 1, 2)
BUDGET = 133
BLOCK = 96
STORE = "exp22_results.json"
R = json.load(open(STORE)) if os.path.exists(STORE) else {}


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
    net = EqAda(P.ncl, cin=7, N=8).cuda()
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


print("%-7s %9s %9s %9s %9s" % ("patch", "buffer", "clean OA", "sd", "overlap"))
for W in (7, 11, 15, 19):
    tag = "W=%d" % W
    if tag in R:
        print("  %-5s cached" % tag, flush=True)
        continue
    P = MSPipe(SCENE, norm="equivariant", W=W, eval_cap=120_000)
    acc, ovl = [], []
    t0 = time.time()
    for sd in SEEDS:
        trm, tem, B = block_buffer_split(P.gt, W=W, block=BLOCK, test_frac=0.7, seed=sd)
        te = np.nonzero(tem)
        rng = np.random.default_rng(sd)
        if len(te[0]) > 120_000:
            k = rng.choice(len(te[0]), 120_000, replace=False)
            te = (te[0][k], te[1][k])
        tr = pick(P.gt, trm, P.ncl, BUDGET, sd)
        m = np.zeros(P.gt.shape, bool); m[tr[0], tr[1]] = True
        ovl.append(verify_no_overlap(m, tem, W=W))
        acc.append(train_eval(P, tr, te, sd))
    R[tag] = dict(W=W, buffer=int(B), oa=acc, sd=float(np.std(acc)),
                  overlap=int(max(ovl)))
    json.dump(R, open(STORE, "w"), indent=1)
    print("  %-5s %9d %9.2f %9.2f %9d   (%.0fs)"
          % (tag, B, float(np.mean(acc)), float(np.std(acc)), max(ovl),
             time.time() - t0), flush=True)
    del P; torch.cuda.empty_cache()

print("\nstored in " + STORE)
