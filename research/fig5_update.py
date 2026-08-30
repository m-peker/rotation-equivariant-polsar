"""Fig. 5 regenerated from a fresh measurement with the current representation.

The earlier version used deviations measured under the superseded per-channel
normalisation. Everything here is measured now, in float64, on real Flevoland
patches, and written to fig5_data.json so the figure is reproducible.

Runs on CPU so it does not contend with training jobs on the GPU.
"""
import json, sys, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
torch.set_default_dtype(torch.float64)
from polsar_data import load_scene
from polsar_lib import rot6_torch
from eqnorm import EqNorm
from equivariant import EqCVCNN, EqCVCNN_F
from steerable import SteerNet, field_stats

ANG = [0.7, 3.0, 5.0, 8.0, 11.25, 15.0, 18.0, 22.5, 26.0, 30.0, 37.0, 45.0,
       52.0, 58.9, 67.5, 76.2, 84.0]
GRID8 = {0.0, 22.5, 45.0, 67.5, 90.0}
W, M, NB = 15, 7, 96

X, gt, ncl = load_scene("flevoland")
Xp = np.pad(X, ((M, M), (M, M), (0, 0)))
r, c = np.nonzero(gt > 0)
idx = np.random.default_rng(0).choice(len(r), NB, replace=False)
xr = torch.tensor(np.stack([Xp[r[i]:r[i] + W, c[i]:c[i] + W].real
                            for i in idx]).transpose(0, 3, 1, 2).astype(np.float64))
xi = torch.tensor(np.stack([Xp[r[i]:r[i] + W, c[i]:c[i] + W].imag
                            for i in idx]).transpose(0, 3, 1, 2).astype(np.float64))
E = EqNorm(X).to("cpu")
ST = field_stats(X)


def measure(make, pre):
    torch.manual_seed(0)
    net = make().double().eval()
    out = []
    with torch.no_grad():
        y0 = net(*pre(xr, xi))
        s = float(y0.abs().max())
        for a in ANG:
            th = torch.tensor(np.deg2rad(a))
            rr, ii = rot6_torch(xr, xi, th)
            out.append(float((net(*pre(rr, ii)) - y0).abs().max()) / s)
    return out


MODELS = [("discrete-max", lambda: EqCVCNN(ncl, cin=7, N=8), lambda a, b: E(a, b),
           r"Discrete $N{=}8$, max readout", "#e67e22", "o"),
          ("discrete-fourier", lambda: EqCVCNN_F(ncl, cin=7, N=8), lambda a, b: E(a, b),
           r"Discrete $N{=}8$, Fourier readout", "#1f6fb4", "s"),
          ("steerable", lambda: SteerNet(ncl, stats=ST), lambda a, b: (a, b),
           "Steerable (continuous)", "#7b3fa0", "^")]

D = {"angles": ANG}
for key, make, pre, label, col, mk in MODELS:
    D[key] = measure(make, pre)
    on = [v for a, v in zip(ANG, D[key]) if a in GRID8]
    off = [v for a, v in zip(ANG, D[key]) if a not in GRID8]
    print("%-18s on-grid max %.2e   off-grid max %.2e"
          % (label, max(on) if on else float("nan"), max(off)))
json.dump(D, open("fig5_data.json", "w"), indent=1)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(3.5, 2.35))
for key, _, _, label, col, mk in MODELS:
    ax.semilogy(ANG, np.clip(D[key], 1e-17, None), color=col, marker=mk, ms=2.8,
                lw=1.0, label=label, mec="white", mew=0.4)
for g in (22.5, 45, 67.5):
    ax.axvline(g, color="#d5d7dc", lw=0.6, ls=(0, (1, 2)))
ax.set_xlabel(r"rotation angle $\theta$ (deg)")
ax.set_ylabel("relative deviation")
ax.set_ylim(1e-17, 1)
ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
ax.legend(frameon=False, loc="center left", handlelength=1.6)
ax.grid(axis="y", color="#d5d7dc", lw=0.5)
for ext in ("pdf", "png"):
    fig.savefig("paper_figs/fig5_equivariance." + ext)
plt.close(fig)
print("fig5 regenerated from measured data -> fig5_data.json")
