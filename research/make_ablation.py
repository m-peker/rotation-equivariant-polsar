"""Emit the ablation table from exp21/exp22, plus the patch-size figure.

As with the other tables, nothing is typed by hand. The patch-size axis is drawn
from both protocols side by side, because the point of that row is that the two
protocols disagree.
"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../paper/tables"
os.makedirs(OUT, exist_ok=True)
A = json.load(open("exp21_results.json"))
B = json.load(open("exp22_results.json"))
# The patch sweep was re-run at ten seeds (Exp. 32) after a reviewer noted
# that at three the table supports no conclusion at all. Prefer it.
if os.path.exists("exp32_results.json"):
    B = json.load(open("exp32_results.json"))
    NSEED_PATCH = 10
else:
    NSEED_PATCH = 3
# These overlap fractions were hard-coded here until Exp. 39 measured them, at
# which point they turned out to be wrong and to disagree with the figure quoted
# in the prose. They are read from the measurement now, for the class-balanced
# budget this paper trains with.
_E39 = json.load(open("exp39_results.json"))
LEAK = {W: _E39["flevoland|balanced|W=%d" % W]["overlap"] for W in (7, 11, 15, 19)}


def g(tag, th="0"):
    return A[tag]["mean"][th]


L = [r"\begin{table}[t]",
     r"\caption{Ablation on Flevoland, 133 labels per class unless stated.",
     r"OA$_{10}$ is accuracy after a \ang{10} rotation, an angle off the",
     r"$N{=}8$ group grid, and OA$_{45}$ after an on-grid \ang{45} rotation.",
     r"The first four blocks vary the discrete network at three seeds; the last",
     r"varies the steerable network at five, so the two are comparable within a",
     r"block but not across.}",
     r"\label{tab:ablation}", r"\centering", r"\small",
     r"\begin{tabular}{llrrr}", r"\toprule",
     r"axis & setting & OA & OA$_{10}$ & OA$_{45}$ \\", r"\midrule"]

L.append(r"\multirow{3}{*}{orientations}")
for i, N in enumerate((4, 8, 16)):
    t = "A|N=%d" % N
    pre = "" if i == 0 else ""
    L.append("%s & $N=%d$ & %.2f & %.2f & %.2f \\\\"
             % (pre, N, g(t), g(t, "10"), g(t, "45")))
L.append(r"\midrule")

L.append(r"\multirow{2}{*}{readout}")
for i, (t, nm) in enumerate([("B|max", "max magnitude"),
                             ("B|fourier", "Fourier magnitudes")]):
    L.append(" & %s & %.2f & %.2f & %.2f \\\\" % (nm, g(t), g(t, "10"), g(t, "45")))
L.append(r"\midrule")

L.append(r"\multirow{2}{*}{normalisation}")
for t, nm in [("C|irrep", r"irreducible (ours)"),
              ("C|per-channel", r"per-channel")]:
    L.append(" & %s & %.2f & %.2f & %.2f \\\\" % (nm, g(t), g(t, "10"), g(t, "45")))
L.append(r"\midrule")

L.append(r"\multirow{5}{*}{labels/class}")
for bg in (10, 25, 50, 133, 300):
    t = "D|budget=%d" % bg
    L.append(" & %d & %.2f & %.2f & %.2f \\\\" % (bg, g(t), g(t, "10"), g(t, "45")))

# The steerable network's own components (Exp. 37). Separated because these are
# five seeds and vary the steerable construction rather than the discrete one;
# the numbers above vary the discrete network at three seeds.
if os.path.exists("exp37_results.json"):
    S37 = json.load(open("exp37_results.json"))
    rows = [("gate: invariant (ours)", "invariant gate"),
            ("gate: norm non-lin.", "norm non-lin."),
            ("gate: none (linear)", "no non-lin."),
            ("readout: no rel. phase", "no rel.\\ phase"),
            ("readout: invariants only", "invariants only")]
    have = [(k, nm) for k, nm in rows if k in S37]
    if have:
        L.append(r"\midrule")
        L.append(r"\multirow{%d}{*}{steerable}" % len(have))
        for k, nm in have:
            v = S37[k]["mean"]
            L.append(" & %s & %.2f & %.2f & %.2f \\\\" % (nm, v[0], v[1], v[2]))
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_ablation.tex"), "w").write("\n".join(L) + "\n")

# ---------------- patch size, both protocols ----------------
P = [r"\begin{table}[t]",
     r"\caption{Patch size under the two protocols. Under the random split",
     r"accuracy rises with the patch, but so does the fraction of test pixels",
     r"whose patch overlaps a training patch, and the two cannot be separated.",
     r"Under the block-and-buffer partition, where the buffer scales as $(W+1)/2$",
     r"and the overlap is zero at every size, the ordering does not survive and",
     r"no difference exceeds the seed spread. Block-and-buffer values are the mean "
     r"of NSEEDP seeds with their seed-to-seed deviation.}",
     r"\label{tab:patch}", r"\centering", r"\small",
     r"\begin{tabular}{lrrr}", r"\toprule",
     r"patch & random split & block\,+\,buffer & overlap \\", r"\midrule"]
for W in (7, 11, 15, 19):
    b = B["W=%d" % W]
    P.append(r"$%d\times%d$ & %.2f & %.2f\,{\tiny$\pm$%.2f} & \SI{%.1f}{\percent} \\"
             % (W, W, g("E|W=%d" % W), float(np.mean(b["oa"])), b["sd"], LEAK[W]))
P += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_patch.tex"), "w").write(
    ("\n".join(P) + "\n").replace("NSEEDP", str(NSEED_PATCH)))

print("wrote tab_ablation.tex and tab_patch.tex")
print("  normalisation, OA45: irrep %.2f vs per-channel %.2f  (drop %.2f)"
      % (g("C|irrep", "45"), g("C|per-channel", "45"),
         g("C|irrep", "45") - g("C|per-channel", "45")))
print("  readout, off-grid OA10: max %.2f vs Fourier %.2f"
      % (g("B|max", "10"), g("B|fourier", "10")))
print("  patch size: random split spans %.2f pp, disjoint spans %.2f pp with sd ~%.1f"
      % (g("E|W=19") - g("E|W=7"),
         max(np.mean(B["W=%d" % w]["oa"]) for w in (7, 11, 15, 19))
         - min(np.mean(B["W=%d" % w]["oa"]) for w in (7, 11, 15, 19)),
         np.mean([B["W=%d" % w]["sd"] for w in (7, 11, 15, 19)])))
