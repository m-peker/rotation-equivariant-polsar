"""Publication-ready figures for IEEE TGRS / ISPRS J. Photogramm.

Produces Fig. 1, 3, 4 only. Fig. 2, 5, 6 and 7 are generated from measured
result files by their own scripts so that they can never drift away from the
tables -- see make_all_figs.py for the full order.

Conventions: Times, single-column 3.5in / double-column 7.16in, 600 dpi,
vector PDF + high-res PNG, English labels, no in-axes titles (captions carry
the description, per journal style).
"""
import numpy as np, sys, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
sys.path.insert(0, ".")
from polsar_data import load_scene, bdist
from disjoint import block_buffer_split
from scipy.ndimage import binary_dilation

SC1, SC2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
K = dict(base="#c0392b", aug="#e67e22", eqd="#1f6fb4", eqf="#159c72",
         stl="#7b3fa0", grid="#d5d7dc", ink="#333333")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig("paper_figs/%s.%s" % (name, ext))
    plt.close(fig)
    print("  ", name)


X, gt, ncl = load_scene("flevoland")
d = bdist(gt)

# ---------------- Fig. 1  scene overview ----------------
def pauli(X):
    out = np.zeros(X.shape[:2] + (3,))
    for i, ch in enumerate((X[..., 1].real, X[..., 2].real, X[..., 0].real)):
        v = 10 * np.log10(np.clip(ch, 1e-8, None))
        lo, hi = np.percentile(v, [2, 98])
        out[..., i] = np.clip((v - lo) / (hi - lo), 0, 1)
    return out


fig, ax = plt.subplots(1, 3, figsize=(SC2, 2.05))
ax[0].imshow(pauli(X))
cmap = ListedColormap(["#f0f0ee"] + list(plt.cm.tab20(np.linspace(0, 1, 15))))
ax[1].imshow(gt, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 16.5), cmap.N))
im = ax[2].imshow(np.where(gt > 0, np.minimum(d, 15), np.nan), cmap="magma")
cb = plt.colorbar(im, ax=ax[2], fraction=0.036, pad=0.02)
cb.set_label("distance (pixels)", fontsize=7)
cb.ax.tick_params(labelsize=6)
for a, t in zip(ax, ["(a)", "(b)", "(c)"]):
    a.set_xticks([])
    a.set_yticks([])
    a.text(0.0, 1.02, t, transform=a.transAxes, va="bottom", fontsize=8)
save(fig, "fig1_scene")

# ---------------- Fig. 3  boundary-resolved error ----------------
lbl = ["1-2", "3-4", "5-7", "8-11", r"$\geq$12"]
acc = [94.95, 98.00, 98.97, 99.11, 99.65]
err = [72.5, 10.5, 7.1, 4.5, 5.4]
base = [23.1, 13.7, 16.9, 16.3, 30.0]
fig, ax = plt.subplots(1, 2, figsize=(SC2, 2.3))
ax[0].bar(lbl, [100 - a for a in acc], color=K["base"], width=0.6, zorder=3)
for i, a in enumerate(acc):
    ax[0].text(i, 100 - a + 0.1, "%.2f" % (100 - a), ha="center", fontsize=6.5,
               color=K["ink"])
ax[0].set_ylabel("error rate (%)")
ax[0].set_ylim(0, 6)
x = np.arange(5)
w = 0.38
ax[1].bar(x - w / 2, err, w, color=K["eqd"], label="errors", zorder=3)
ax[1].bar(x + w / 2, base, w, color="#b9bcc4", label="test population", zorder=3)
ax[1].set_xticks(x)
ax[1].set_xticklabels(lbl)
ax[1].set_ylabel("share (%)")
ax[1].legend(frameon=False, loc="upper right", handlelength=1.3)
ax[1].annotate(r"$3.14\times$", xy=(-w / 2, 72.5), xytext=(0.7, 62), fontsize=8,
               color=K["eqd"], arrowprops=dict(arrowstyle="->", color=K["eqd"], lw=0.7))
for i, a in enumerate(ax):
    a.set_xlabel("distance to class boundary (pixels)")
    a.grid(axis="y", color=K["grid"], lw=0.5, zorder=0)
    a.text(0.0, 1.045, "(%s)" % chr(97 + i), transform=a.transAxes, va="bottom", fontsize=8)
fig.subplots_adjust(wspace=0.28)
save(fig, "fig3_boundary")

# ---------------- Fig. 4  protocol leakage ----------------
rng = np.random.default_rng(0)
r, c = np.nonzero(gt > 0)
p = rng.permutation(len(r))
ntr = int(0.01 * len(r))
trm = np.zeros(gt.shape, bool)
trm[r[p[:ntr]], c[p[:ntr]]] = True
tem = np.zeros(gt.shape, bool)
tem[r[p[ntr:]], c[p[ntr:]]] = True
clash = binary_dilation(trm, structure=np.ones((15, 15), bool)) & tem
btr, bte, _ = block_buffer_split(gt, block=96, test_frac=0.7, seed=0)
sl = (slice(120, 320), slice(300, 500))
CTR, CTE, CCL = [0.13, 0.44, 0.71], [0.85, 0.87, 0.90], [0.75, 0.22, 0.17]
fig, ax = plt.subplots(1, 2, figsize=(SC2, 3.15))
for a, (tr_, te_, cl_) in zip(ax, [(trm, tem, clash), (btr, bte, None)]):
    v = np.full(gt.shape + (3,), 0.97)
    v[te_] = CTE
    if cl_ is not None:
        v[cl_] = CCL
    v[tr_] = CTR
    a.imshow(v[sl])
    a.set_xticks([])
    a.set_yticks([])
for a, t in zip(ax, ["(a) random split", "(b) block + buffer"]):
    a.set_xlabel(t, fontsize=7.5)
fig.legend(handles=[Patch(color=CTR, label="training"),
                    Patch(color=CTE, label="test, no overlap"),
                    Patch(color=CCL, label="test, patch overlaps a training patch")],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03),
           handlelength=1.3, columnspacing=1.4)
fig.subplots_adjust(wspace=0.06)
save(fig, "fig4_leakage")

print("done ->", os.path.abspath("paper_figs"))
