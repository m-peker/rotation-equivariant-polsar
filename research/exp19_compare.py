"""Exp. 19 - comparison table against re-implemented literature baselines.

The method set follows Alkhatib (2025) so the table reads against the published
comparison: SVM, CV-MLP, CV-2DCNN, CV-3DCNN, CV-2D-3D, CV-ViT, CV-MsAtViT, plus
our discrete equivariant and steerable networks.

Every model is trained under identical conditions -- same representation, same
class-balanced budget, same seeds, same splits -- so the table isolates the model
and not the pipeline. Metrics are OA, AA and Kappa, matching the literature, and
we additionally report accuracy after a 45 deg polarimetric rotation of the test
data, which no published method has been evaluated under.

Resumable: each (scene, model) is stored in exp19_results.json as it finishes.
"""
import numpy as np, torch, torch.nn as nn, time, sys, json, os
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
from baselines import REGISTRY, count_macs
from cvmsatvit import CVMsAtViT
from equivariant import EqCVCNN_F
from steerable import SteerNet, field_stats
from polsar_data import load_scene
import pipe_ms

BUDGET = {"flevoland": 133, "sanfran": 400, "ober": 666}
SEEDS = (0, 1, 2)
STORE = "exp19_results.json"
THETAS = [0, 45]


class RawPipe(pipe_ms.MSPipe):
    """Steerable takes raw T3 and performs its own decomposition."""
    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        if th is not None:
            xr, xi = rot6_torch(xr, xi, th)
        return xr, xi


def kappa(cm):
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe)


def metrics(pred, true, ncl):
    cm = np.zeros((ncl, ncl), np.int64)
    np.add.at(cm, (true, pred), 1)
    oa = 100 * np.trace(cm) / cm.sum()
    per = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(ncl), where=cm.sum(1) > 0)
    return float(oa), float(100 * per.mean()), float(100 * kappa(cm))


def sample(P, n, seed):
    rng = np.random.default_rng(1000 + seed)
    tr = [rng.choice(np.nonzero(P.y == k)[0], min(n, int((P.y == k).sum())),
                     replace=False) for k in range(P.ncl)]
    tr = np.concatenate(tr)
    te = np.setdiff1d(np.arange(len(P.y)), tr)
    return tr, P.cap_eval(te)


def run_svm(P, seed):
    itr, ite = sample(P, BUDGET[P.scene], seed)
    m = P.W // 2
    def feats(idx):
        r = torch.from_numpy(P.lr[idx]).cuda(); c = torch.from_numpy(P.lc[idx]).cuda()
        out = []
        for s in range(0, len(idx), 8192):
            xr, xi = P.grab(r[s:s + 8192], c[s:s + 8192])
            v = torch.cat([xr[:, :, m, m], xi[:, :, m, m]], 1)
            out.append(v.cpu().numpy())
        return np.concatenate(out)
    # Kernel SVM prediction is linear in support vectors x test points; on a
    # 150k test set that dominates the whole table's runtime for a method that
    # is only present as a floor. Evaluate it on a 30k stratified subsample.
    rng = np.random.default_rng(seed)
    if len(ite) > 30_000:
        ite = rng.choice(ite, 30_000, replace=False)
    Xtr, Xte = feats(itr), feats(ite)
    sc = StandardScaler().fit(Xtr)
    clf = SVC(C=10, gamma="scale", cache_size=800)
    clf.fit(sc.transform(Xtr), P.y[itr])
    pr = clf.predict(sc.transform(Xte))
    return {0: metrics(pr, P.y[ite], P.ncl)}, 0, 0


def run_net(P, Praw, name, seed, epochs=120):
    pipe = Praw if name == "Steerable" else P
    itr, ite = sample(P, BUDGET[P.scene], seed)
    Rt = torch.from_numpy(P.lr[itr]).cuda(); Ct = torch.from_numpy(P.lc[itr]).cuda()
    Yt = torch.from_numpy(P.y[itr]).cuda()
    torch.manual_seed(seed)
    net = BUILD[name](P.ncl).cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(itr), device="cuda")
        for s in range(0, len(itr), 128):
            b = pm[s:s + 128]
            xr, xi = pipe.grab(Rt[b], Ct[b])
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        sch.step()
    net.eval()
    Re = torch.from_numpy(P.lr[ite]).cuda(); Ce = torch.from_numpy(P.lc[ite]).cuda()
    out = {}
    for td in THETAS:
        th = (torch.tensor(np.deg2rad(td), device="cuda", dtype=torch.float32)
              if td else None)
        pr = np.empty(len(ite), np.int64)
        with torch.no_grad():
            for s in range(0, len(ite), 4096):
                xr, xi = pipe.grab(Re[s:s + 4096], Ce[s:s + 4096], th)
                pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
        out[td] = metrics(pr, P.y[ite], P.ncl)
    npar = sum(p.numel() for p in net.parameters())
    del net; torch.cuda.empty_cache()
    return out, npar, MACS.get(name, 0)


R = json.load(open(STORE)) if os.path.exists(STORE) else {}
ORDER = ["SVM", "CV-MLP", "CV-2DCNN", "CV-3DCNN", "CV-2D-3D", "CV-ViT",
         "CV-MsAtViT", "Equivariant", "Steerable"]

for sc in ("flevoland", "sanfran", "ober"):
    todo = [m for m in ORDER if "%s|%s" % (sc, m) not in R]
    if not todo:
        print("### %s complete, skipped ###" % sc, flush=True)
        continue
    P = MSPipe(sc, norm="equivariant", eval_cap=150_000)
    Praw = RawPipe(sc, norm="equivariant", eval_cap=150_000)
    Xs, _, _ = load_scene(sc); ST = field_stats(Xs); del Xs
    BUILD = dict(REGISTRY)
    BUILD = {k: (lambda n, c=v: c(n, cin=7)) for k, v in BUILD.items()}
    BUILD["CV-MsAtViT"] = lambda n: CVMsAtViT(n, cin=7)
    BUILD["Equivariant"] = lambda n: EqCVCNN_F(n, cin=7, N=8)
    BUILD["Steerable"] = lambda n: SteerNet(n, stats=ST)
    globals()["BUILD"] = BUILD
    # Complexity once per scene, on CPU, away from the training loop.
    _xr = torch.randn(2, 7, 15, 15); _xi = torch.randn(2, 7, 15, 15)
    MACS = {}
    for k in BUILD:
        try:
            MACS[k] = count_macs(BUILD[k](P.ncl), _xr, _xi)
        except Exception as e:
            MACS[k] = 0
            print("  (MAC count failed for %s: %s)" % (k, type(e).__name__), flush=True)
    globals()["MACS"] = MACS
    print("\n### %s  budget=%d/class ###" % (sc, BUDGET[sc]), flush=True)
    print("%-13s %7s %7s %7s | %7s | %9s %10s"
          % ("model", "OA", "AA", "Kappa", "OA@45", "params", "MMACs"), flush=True)
    for name in todo:
        acc = {t: [] for t in THETAS}
        t0 = time.time()
        for sd in SEEDS:
            if name == "SVM":
                o, npar, mc = run_svm(P, sd)
            else:
                o, npar, mc = run_net(P, Praw, name, sd)
            for t in o:
                acc[t].append(o[t])
        A0 = np.array(acc[0])
        rot = float(np.array(acc[45])[:, 0].mean()) if acc[45] else float("nan")
        R["%s|%s" % (sc, name)] = dict(oa=A0[:, 0].tolist(), aa=A0[:, 1].tolist(),
                                       kappa=A0[:, 2].tolist(), oa45=rot,
                                       params=int(npar), macs=int(mc))
        json.dump(R, open(STORE, "w"), indent=1)
        print("%-13s %7.2f %7.2f %7.2f | %7.2f | %9d %10.2f  (%.0fs)"
              % (name, A0[:, 0].mean(), A0[:, 1].mean(), A0[:, 2].mean(), rot,
                 npar, mc / 1e6, time.time() - t0), flush=True)
    del P, Praw
    torch.cuda.empty_cache()

print("\nstored in " + STORE)
