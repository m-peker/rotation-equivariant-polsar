"""Run-to-run spread at the protocol actually used, and its cause.

The first probe used 40 epochs and found a 1.55 pp spread between identical
runs. If that also holds at the 120-epoch schedule of the real experiments then
many of the differences reported in the paper are inside the noise and must be
reported with intervals rather than as point comparisons. This measures:

  A. spread at 120 epochs + cosine, the protocol of Tables I and IV
  B. the same with deterministic kernels forced, to identify the cause
"""
import numpy as np, torch, torch.nn as nn, sys, time, os
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from equivariant import EqCVCNN_F

P = MSPipe("flevoland", norm="equivariant", eval_cap=60_000)
rng = np.random.default_rng(1000)
tr = np.concatenate([rng.choice(np.nonzero(P.y == k)[0],
                                min(133, int((P.y == k).sum())), replace=False)
                     for k in range(P.ncl)])
te = P.cap_eval(np.setdiff1d(np.arange(len(P.y)), tr))
Rt = torch.from_numpy(P.lr[tr]).cuda(); Ct = torch.from_numpy(P.lc[tr]).cuda()
Yt = torch.from_numpy(P.y[tr]).cuda()
Re = torch.from_numpy(P.lr[te]).cuda(); Ce = torch.from_numpy(P.lc[te]).cuda()


def once(epochs, cosine=True):
    torch.manual_seed(0)
    net = EqCVCNN_F(P.ncl, cin=7, N=8).cuda()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
           if cosine else None)
    lf = nn.CrossEntropyLoss(); net.train()
    for ep in range(epochs):
        pm = torch.randperm(len(tr), device="cuda")
        for s in range(0, len(tr), 128):
            b = pm[s:s + 128]
            xr, xi = P.grab(Rt[b], Ct[b])
            opt.zero_grad(); lf(net(xr, xi), Yt[b]).backward(); opt.step()
        if sch: sch.step()
    net.eval(); pr = np.empty(len(te), np.int64)
    with torch.no_grad():
        for s in range(0, len(te), 4096):
            xr, xi = P.grab(Re[s:s + 4096], Ce[s:s + 4096])
            pr[s:s + 4096] = net(xr, xi).argmax(1).cpu().numpy()
    del net; torch.cuda.empty_cache()
    return 100.0 * (pr == P.y[te]).mean()


def report(tag, vals, t0):
    v = np.array(vals)
    print("%-34s %s   spread %.3f  sd %.3f   (%.0fs)"
          % (tag, "  ".join("%.3f" % x for x in v), v.max() - v.min(), v.std(),
             time.time() - t0), flush=True)
    return v


t0 = time.time(); report("A. 120 epochs + cosine", [once(120) for _ in range(3)], t0)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
t0 = time.time(); report("B. same, deterministic kernels", [once(120) for _ in range(3)], t0)

print("""
If A is wide and B is tight, the spread is kernel non-determinism and can be
removed. If both are wide, the seed does not control the run and every number
in the paper needs an interval.""")
