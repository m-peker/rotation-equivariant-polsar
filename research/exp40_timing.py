"""Exp. 40 - full-scene inference time, which a reviewer asked for and the
complexity table did not carry.

No training: each model is built, warmed up, and run over every labelled pixel of
Flevoland at batch 4096, timed with CUDA synchronisation so the number is wall
time rather than queue time. Reported alongside the MAC counts, since the two do
not always agree -- a group-convolutional layer that is cheap in MACs can still
be slow if it moves more memory.
"""
import numpy as np, torch, json, os, sys, time
sys.path.insert(0, ".")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from pipe_ms import MSPipe
from polsar_lib import rot6_torch
from equivariant import EqCVCNN_F
from steerable import SteerNet, field_stats
from cvmsatvit import CVMsAtViT
from polsar_data import load_scene
from exp27_oac import CVCNNb

SCENE = "flevoland"
STORE = "exp40_results.json"


class RawPipe(MSPipe):
    def grab(s, r, c, th=None):
        rr = r[:, None, None] + s.off[None, :, None]
        cc = c[:, None, None] + s.off[None, None, :]
        xr = s.Pr[rr, cc].permute(0, 3, 1, 2).contiguous()
        xi = s.Pi[rr, cc].permute(0, 3, 1, 2).contiguous()
        return xr, xi


def time_model(net, pipe, R, C, bs=4096, warm=2):
    net.eval()
    with torch.no_grad():
        for _ in range(warm):
            net(*pipe.grab(R[:bs], C[:bs]))
        torch.cuda.synchronize()
        t0 = time.time()
        for s in range(0, len(R), bs):
            net(*pipe.grab(R[s:s + bs], C[s:s + bs]))
        torch.cuda.synchronize()
        return time.time() - t0


def main():
    out = json.load(open(STORE)) if os.path.exists(STORE) else {}
    P = MSPipe(SCENE, norm="equivariant", eval_cap=10 ** 9)
    Praw = RawPipe(SCENE, norm="equivariant", eval_cap=10 ** 9)
    Xs, _, _ = load_scene(SCENE); ST = field_stats(Xs); del Xs
    R = torch.from_numpy(P.lr).cuda(); C = torch.from_numpy(P.lc).cuda()
    n = len(R)
    print("Flevoland, %d labelled pixels, batch 4096\n" % n, flush=True)
    print("%-22s %10s %12s" % ("model", "seconds", "px/s"), flush=True)

    MODELS = [("CV-CNN", lambda: CVCNNb(P.ncl, cin=7), False),
              ("Equivariant (ours)", lambda: EqCVCNN_F(P.ncl, cin=7, N=8), False),
              ("Steerable (ours)", lambda: SteerNet(P.ncl, stats=ST), True),
              ("CV-MsAtViT", lambda: CVMsAtViT(P.ncl, cin=7), False)]
    for nm, mk, raw in MODELS:
        if nm in out:
            continue
        net = mk().cuda()
        t = time_model(net, Praw if raw else P, R, C)
        out[nm] = dict(seconds=t, pixels=n, px_per_s=n / t)
        json.dump(out, open(STORE, "w"), indent=1)
        print("%-22s %10.2f %12.0f" % (nm, t, n / t), flush=True)
        del net; torch.cuda.empty_cache()
    print("\nstored in " + STORE)


if __name__ == "__main__":
    main()
