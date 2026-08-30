"""Fig. 8 - is the rotation we test for present in the data?

(a) Per-pixel polarisation orientation angle, estimated as the phase of the
    weight-4 component of Section II. The spread is wide.
(b) The same angle aggregated per class. The medians sit almost on top of one
    another, so the wide spread in (a) is variation within classes -- largely
    speckle in an angle that is weakly identifiable where |z_4| is small -- and
    not a systematic orientation offset between classes.

This is reported because it bounds our own claim: within these scenes rotation
invariance is not a source of accuracy, and the case for it rests on transfer
across acquisitions, which these three scenes cannot demonstrate.
"""
import json, numpy as np, matplotlib
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
COL = {"flevoland": "#c0392b", "sanfran": "#1f6fb4", "ober": "#159c72"}
NAME = {"flevoland": "Flevoland", "sanfran": "San Francisco",
        "ober": "Oberpfaffenhofen"}
MK = {"flevoland": "o", "sanfran": "s", "ober": "^"}
GREY = "#d5d7dc"

D = json.load(open("exp20_results.json"))
fig, ax = plt.subplots(1, 2, figsize=(SC2, 2.35))

edges = np.linspace(-22.5, 22.5, 61)
ctr = 0.5 * (edges[1:] + edges[:-1])
for sc in ["flevoland", "sanfran", "ober"]:
    h = np.array(D[sc]["hist"], float)
    h = h / h.sum()
    ax[0].plot(ctr, h, color=COL[sc], lw=1.2, label=NAME[sc])
    ax[0].fill_between(ctr, 0, h, color=COL[sc], alpha=0.10)
ax[0].set_xlabel(r"polarisation orientation angle (deg)")
ax[0].set_ylabel("density")
ax[0].set_xlim(-22.5, 22.5)
ax[0].legend(frameon=False, handlelength=1.3)
ax[0].grid(axis="y", color=GREY, lw=0.5)
ax[0].text(0.0, 1.05, "(a)", transform=ax[0].transAxes, va="bottom", fontsize=8)

for j, sc in enumerate(["flevoland", "sanfran", "ober"]):
    med = np.array(D[sc]["class_medians"])
    x = np.full(len(med), j) + np.linspace(-0.17, 0.17, len(med))
    ax[1].plot(x, med, MK[sc], color=COL[sc], ms=3.4, mec="white", mew=0.5,
               ls="none")
    ax[1].plot([j - 0.30, j + 0.30], [med.mean()] * 2, color=COL[sc], lw=0.9)
    ax[1].text(j, 12.6, r"$\sigma=%.2f^\circ$" % np.std(med), ha="center",
               fontsize=6.8, color=COL[sc])
ax[1].axhline(0, color=GREY, lw=0.6, ls=(0, (2, 2)))
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels([NAME[s] for s in ["flevoland", "sanfran", "ober"]],
                      fontsize=6.8)
ax[1].set_ylabel("per-class median angle (deg)")
ax[1].set_ylim(-14, 14)
ax[1].grid(axis="y", color=GREY, lw=0.5)
ax[1].text(0.0, 1.05, "(b)", transform=ax[1].transAxes, va="bottom", fontsize=8)

fig.subplots_adjust(wspace=0.30)
for ext in ("pdf", "png"):
    fig.savefig("paper_figs/fig8_poa." + ext)
plt.close(fig)
print("wrote paper_figs/fig8_poa.pdf / .png")
for sc in D:
    print("  %-16s per-pixel sd %5.2f deg   class-median sd %5.2f deg"
          % (NAME[sc], D[sc]["std"], D[sc]["class_median_spread"]))
