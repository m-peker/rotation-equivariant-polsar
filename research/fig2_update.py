"""Fig. 2 regenerated from exp15_results.json (the authoritative run).

The earlier version used numbers from the superseded per-channel normalisation
and disagreed with Table I. This reads the stored results so figure and table
cannot drift apart again.
"""
import json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC2 = 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
K = dict(base="#c0392b", aug="#e67e22", eqf="#159c72", msat="#7b3fa0",
         grid="#d5d7dc", ink="#333333")
TH = np.array([0, 10, 22.5, 45])
GRID = {0, 22.5, 45}

R = json.load(open("exp15_results.json"))
# Flevoland is reported at ten seeds in the paper; the figure must show the
# same numbers as the table or the two drift apart, which is what happened
# to an earlier version of this figure.
import os
if os.path.exists("exp31_results.json"):
    R.update(json.load(open("exp31_results.json")))
SERIES = [("flevoland|baseline no-aug",   "Baseline CV-CNN",            K["base"], "o"),
          ("flevoland|baseline rot-AUG",  "Baseline + rotation aug.",   K["aug"],  "s"),
          ("flevoland|Equivariant",    "Equivariant (ours)",         K["eqf"],  "D"),
          ("flevoland|CV-MsAtViT", "CV-MsAtViT [1]",             K["msat"], "v")]
missing = [k for k, _, _, _ in SERIES if k not in R]
if missing:
    sys.exit("missing arms: " + ", ".join(missing))

fig, ax = plt.subplots(1, 2, figsize=(SC2, 2.5))
for i, (ylo, yhi) in enumerate([(35, 100), (93.8, 98.4)]):
    for key, nm, c, m in SERIES:
        v = R[key]["mean"]
        if i == 1 and max(v) - min(v) > 10:      # collapsing models off the zoom panel
            continue
        ax[i].plot(TH, v, color=c, marker=m, ms=3.4, lw=1.1, label=nm,
                   mec="white", mew=0.5, clip_on=False, zorder=3)
    for g in GRID:
        ax[i].axvline(g, color=K["grid"], lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax[i].set_ylim(ylo, yhi)
    ax[i].set_xlim(-1.5, 46.5)
    ax[i].set_xticks([0, 10, 22.5, 45])
    ax[i].set_xticklabels(["0", "10", "22.5", "45"])
    ax[i].grid(axis="y", color=K["grid"], lw=0.5, zorder=0)
    ax[i].set_xlabel(r"polarimetric rotation $\theta$ (deg)")
    ax[i].text(0.0, 1.045, "(%s)" % "ab"[i], transform=ax[i].transAxes,
               va="bottom", fontsize=8)
ax[0].set_ylabel("overall accuracy (%)")

drop = R["flevoland|baseline no-aug"]["mean"][0] - R["flevoland|baseline no-aug"]["mean"][-1]
ax[0].annotate(r"$-$%.1f pp" % drop, xy=(45, R["flevoland|baseline no-aug"]["mean"][-1]),
               xytext=(26, 56), fontsize=7.5, color=K["base"],
               arrowprops=dict(arrowstyle="->", color=K["base"], lw=0.7))
dm = R["flevoland|CV-MsAtViT"]["mean"]
ax[0].annotate(r"$-$%.1f pp" % (dm[0] - dm[-1]), xy=(45, dm[-1]), xytext=(30, 72),
               fontsize=7.5, color=K["msat"],
               arrowprops=dict(arrowstyle="->", color=K["msat"], lw=0.7))

eqf = R["flevoland|Equivariant"]["mean"]
aug = R["flevoland|baseline rot-AUG"]["mean"]
ax[1].annotate("", xy=(33, eqf[0]), xytext=(33, aug[0]),
               arrowprops=dict(arrowstyle="<->", color=K["ink"], lw=0.7))
ax[1].text(33.8, (eqf[0] + aug[0]) / 2, "+%.2f pp" % (eqf[0] - aug[0]),
           fontsize=7, color=K["ink"], va="center")
ax[1].text(22.5, 97.55, "flat to within %.2f pp" % (max(eqf) - min(eqf)),
           ha="center", fontsize=6.6, color=K["eqf"])

h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.16), handlelength=1.6, columnspacing=1.3)
fig.subplots_adjust(wspace=0.26)
for ext in ("pdf", "png"):
    fig.savefig("paper_figs/fig2_rotation." + ext)
plt.close(fig)
print("fig2 regenerated from exp15_results.json")
for key, nm, _, _ in SERIES:
    v = R[key]["mean"]
    print("  %-26s %s  drop=%5.2f" % (nm, " ".join("%6.2f" % x for x in v), v[0] - min(v)))
