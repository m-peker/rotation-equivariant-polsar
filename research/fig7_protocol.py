"""Fig. 7 - what the standard protocol actually measures.

(a) Per-seed accuracy under the two training regimes on a fixed test partition.
    The leaky regime is tight (sigma 0.3-0.8); the spatially disjoint regime is
    wide (sigma 2.2-7.4). The apparent stability of published numbers is partly
    a property of the protocol, not of the methods.
(b) Clean-arm accuracy against the fraction of a class's distinct fields that
    the training partition happens to cover. The correlation shows that a
    substantial part of the gap is covariate shift - generalisation to unseen
    fields - and not leakage removal alone. We report it as such.

Reads exp18_results.json.
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

D = json.load(open("exp18_results.json"))
fig, ax = plt.subplots(1, 2, figsize=(SC2, 2.6))

# ---- (a) per-seed paired plot ----
for j, sc in enumerate(["flevoland", "sanfran", "ober"]):
    L = np.array(D[sc]["leaky"]); C = np.array(D[sc]["clean"])
    x0, x1 = j * 1.0 - 0.16, j * 1.0 + 0.16
    for a, b in zip(L, C):
        ax[0].plot([x0, x1], [a, b], color=GREY, lw=0.7, zorder=1)
    ax[0].plot([x0] * len(L), L, MK[sc], color=COL[sc], ms=3.6, mec="white",
               mew=0.5, zorder=3)
    ax[0].plot([x1] * len(C), C, MK[sc], color=COL[sc], ms=3.6, mfc="white",
               mec=COL[sc], mew=0.9, zorder=3)
    ax[0].text(j, 101.5, NAME[sc], ha="center", fontsize=7, color=COL[sc])
    ax[0].text(x1 + 0.10, C.mean(), r"$\sigma$=%.1f" % C.std(), fontsize=6.4,
               color=COL[sc], va="center")
    ax[0].text(x0 - 0.10, L.mean(), r"$\sigma$=%.1f" % L.std(), fontsize=6.4,
               color=COL[sc], va="center", ha="right")
ax[0].set_xticks([-0.16, 0.16, 0.84, 1.16, 1.84, 2.16])
ax[0].set_xticklabels(["adj.", "disj.", "adj.", "disj.", "adj.", "disj."],
                      fontsize=6.6)
ax[0].set_xlim(-0.55, 2.55)
ax[0].set_ylim(60, 100)
ax[0].set_ylabel("overall accuracy (%)")
ax[0].grid(axis="y", color=GREY, lw=0.5, zorder=0)
ax[0].text(0.0, 1.10, "(a)", transform=ax[0].transAxes, va="bottom", fontsize=8)

# ---- (b) coverage vs clean accuracy ----
# Within-scene trends only. Pooling across scenes would mix three different
# accuracy levels and destroy the relationship (pooled r = +0.06 against
# within-scene r of +0.66/+0.52/+0.34), so each scene gets its own fit.
rr = {}
for sc in ["flevoland", "sanfran", "ober"]:
    cov = np.array(D[sc]["covB"]); C = np.array(D[sc]["clean"])
    ax[1].plot(cov, C, MK[sc], color=COL[sc], ms=4, mec="white", mew=0.5,
               ls="none", label=NAME[sc], zorder=3)
    p = np.polyfit(cov, C, 1)
    xs = np.linspace(cov.min(), cov.max(), 20)
    ax[1].plot(xs, np.polyval(p, xs), color=COL[sc], lw=0.8, ls=(0, (4, 2)),
               alpha=0.75, zorder=2)
    rr[sc] = np.corrcoef(cov, C)[0, 1]
ax[1].text(0.035, 0.055,
           "\n".join(r"$r=%+.2f$" % rr[s] for s in ["flevoland", "sanfran", "ober"]),
           transform=ax[1].transAxes, fontsize=6.6, color="#333333",
           linespacing=1.35)
r = rr["flevoland"]
ax[1].set_xlabel("fraction of each class's fields seen in training")
ax[1].set_ylabel("overall accuracy (%)")
ax[1].grid(color=GREY, lw=0.5, zorder=0)
ax[1].legend(frameon=False, loc="upper left", handlelength=1.0)
ax[1].text(0.0, 1.10, "(b)", transform=ax[1].transAxes, va="bottom", fontsize=8)

fig.subplots_adjust(wspace=0.3)
for ext in ("pdf", "png"):
    fig.savefig("paper_figs/fig7_protocol." + ext)
plt.close(fig)
print("wrote paper_figs/fig7_protocol.pdf / .png")
print("  within-scene r: " + "  ".join("%s %+.2f" % (NAME[s_], rr[s_]) for s_ in rr))
for sc in D:
    L = np.array(D[sc]["leaky"]); C = np.array(D[sc]["clean"])
    print("  %-16s adjacent %.2f+-%.2f   disjoint %.2f+-%.2f   gap %.2f+-%.2f"
          % (NAME[sc], L.mean(), L.std(), C.mean(), C.std(),
             (L - C).mean(), (L - C).std()))
