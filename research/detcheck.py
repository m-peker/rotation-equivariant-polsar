"""How reproducible is a training run on this hardware?

Two tables report the same model from independent runs. After making the
evaluation subsample deterministic they agree to 0.04 pp, and the question is
whether that residual is a protocol difference we have missed or simply
run-to-run non-determinism on the GPU (cuDNN algorithm selection, atomics in
the backward pass). This measures it directly: identical seed, identical data,
three runs.
"""
import numpy as np, torch, torch.nn as nn, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from equivariant import EqCVCNN_F
from cvmsatvit import CVMsAtViT

P = MSPipe("flevoland", norm="equivariant", eval_cap=60_000)
rng = np.random.default_rng(1000)
tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                min(133, int((P.y == k).sum())), replace=False)
                     for k in range(P.ncl)])
te = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
Yt = torch.from_numpy(P.y[tr]).cuda()
Re = torch.from_numpy(P.lr[te]).cuda(); Ce = torch.from_numpy(P.lc[te]).cuda()


def once(cls, epochs):
    torch.manual_seed(0)
    net = cls().cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lf = nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), 128):
            b = pm[s:s + 128]
            xr, xi = P.grab(Rt[b], Ct[b])
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
    net.eval(); pr = np.empty(len(te), np.int64)
    with torch.no_grad():
        for s in range(0, len(te), 4096):
            xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096])
            pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
    del net; torch.cuda.empty_cache()
    return 100.0 * (pr == P.y[te]).mean()


for nm, cls, ep in [("Equivariant", lambda: EqCVCNN_F(P.ncl, cin=7, N=8), 40),
                    ("CV-MsAtViT", lambda: CVMsAtViT(P.ncl, cin=7), 40)]:
    t0 = time.time()
    v = [once(cls, ep) for _ in range(3)]
    print("%-12s same seed, three runs: %s   spread %.4f pp   (%.0fs)"
          % (nm, "  ".join("%.4f" % x for x in v), max(v) - min(v), time.time() - t0),
          flush=True)
print("\nobserved disagreement between Tables I and IV: 0.039 pp")
