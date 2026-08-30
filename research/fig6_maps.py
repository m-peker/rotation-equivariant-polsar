"""Fig. 6 - full-scene classification maps under polarimetric rotation.

Layout: 3 model columns x 3 rows
  row 1  prediction at theta = 0 deg
  row 2  prediction at theta = 45 deg
  row 3  pixels whose predicted label CHANGED between the two angles
For an exactly equivariant network row 3 must be empty. That panel is the
claim, made visually undeniable.
"""
import numpy as np, os, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.insert(0, ".")
from polsar_data import load_scene

SC2 = 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})

MODELS = [("baseline", "Baseline CV-CNN"),
          ("rotAUG", "Baseline + rotation aug."),
          ("steerable", "Steerable equivariant")]
ANG = (0, 45)

missing = [f"figs/map_{m}_{a}.npy" for m, _ in MODELS for a in ANG
           if not os.path.exists("figs/map_%s_%s.npy" % (m, a))]
if missing:
    sys.exit("missing map files:\n  " + "\n  ".join(missing))

X, gt, ncl = load_scene("flevoland")
lr, lc = np.nonzero(gt > 0)
cmap = ListedColormap(["#f0f0ee"] + list(plt.cm.tab20(np.linspace(0, 1, 15))))
norm = BoundaryNorm(np.arange(-0.5, 16.5), cmap.N)

fig, ax = plt.subplots(3, 3, figsize=(SC2, 5.05))
for j, (key, label) in enumerate(MODELS):
    maps = {}
    for a in ANG:
        m = np.load("figs/map_%s_%s.npy" % (key, a))
        maps[a] = m
        oa = 100.0 * (m[lr, lc] == gt[lr, lc]).mean()
        r = ANG.index(a)
        ax[r, j].imshow(m, cmap=cmap, norm=norm, interpolation="nearest")
        ax[r, j].text(0.985, 0.02, "OA %.2f%%" % oa, transform=ax[r, j].transAxes,
                      ha="right", va="bottom", fontsize=6.6, color="white",
                      bbox=dict(fc="black", alpha=0.6, pad=1.5, lw=0))
    changed = maps[ANG[0]] != maps[ANG[1]]
    frac = 100.0 * changed.mean()
    vis = np.ones(gt.shape + (3,))
    vis[changed] = [0.75, 0.22, 0.17]
    ax[2, j].imshow(vis, interpolation="nearest")
    txt = "no pixel changed" if frac == 0 else "%.1f%% of pixels changed" % frac
    ax[2, j].text(0.5, 0.5, txt, transform=ax[2, j].transAxes, ha="center",
                  va="center", fontsize=7.5,
                  color=("#1b7a4b" if frac == 0 else "#333333"),
                  bbox=dict(fc="white", alpha=0.82, pad=2.4, lw=0))
    ax[0, j].set_title(label, fontsize=8, pad=4)

rows = [r"$\theta = 0^\circ$", r"$\theta = 45^\circ$", "changed labels"]
for i, rl in enumerate(rows):
    ax[i, 0].set_ylabel(rl, fontsize=8, labelpad=4)
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
        sp.set_color("#bbbbbb")
fig.subplots_adjust(wspace=0.03, hspace=0.03)
for ext in ("pdf", "png"):
    fig.savefig("paper_figs/fig6_maps." + ext)
plt.close(fig)
print("wrote paper_figs/fig6_maps.pdf / .png")

for key, label in MODELS:
    m0 = np.load("figs/map_%s_0.npy" % key)
    m45 = np.load("figs/map_%s_45.npy" % key)
    o0 = 100.0 * (m0[lr, lc] == gt[lr, lc]).mean()
    o45 = 100.0 * (m45[lr, lc] == gt[lr, lc]).mean()
    print("  %-22s OA(0)=%6.2f  OA(45)=%6.2f  drop=%6.2f  changed=%5.2f%%"
          % (label, o0, o45, o0 - o45, 100.0 * (m0 != m45).mean()))
