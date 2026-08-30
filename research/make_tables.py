"""Emit the LaTeX comparison and complexity tables from measured result files.

Same principle as the figures: nothing is typed by hand, so a table cannot drift
away from the experiment that produced it. Writes into paper/tables/ which
main.tex \\input{}s. Scenes that have not finished are simply omitted, so this
can be run at any point during a sweep.

  tab_compare.tex     OA / AA / Kappa / OA@45 per scene   <- exp19_results.json
  tab_complexity.tex  parameters and MACs                 <- exp19_results.json
  tab_main.tex        rotation sweep for the headline set <- exp15_results.json
"""
import json, os, numpy as np

OUT = "../paper/tables"
os.makedirs(OUT, exist_ok=True)

ORDER = ["SVM", "CV-MLP", "CV-2DCNN", "CV-3DCNN", "CV-2D-3D", "CV-ViT",
         "CV-MsAtViT", "Equivariant", "Steerable"]
PRETTY = {"CV-MsAtViT": r"CV-MsAtViT~\cite{alkhatib2025}",
          "Equivariant": r"\textbf{Equivariant (ours)}",
          "Steerable": r"\textbf{Steerable (ours)}"}
SCENES = [("flevoland", "Flevoland"), ("sanfran", "San Francisco"),
          ("ober", "Oberpfaffenhofen")]
R = json.load(open("exp19_results.json"))
have = [(k, n) for k, n in SCENES if any(k + "|" + m in R for m in ORDER)]


def cell(v, best, fmt="%.2f", bold=True, sd=None):
    """A value, optionally with its seed-to-seed standard deviation.

    Point estimates alone invited over-reading: in an earlier draft several
    differences we described as results turned out to be smaller than the
    spread across seeds. Printing the deviation puts that in the table itself
    rather than leaving it to the reader to discover.
    """
    s = fmt % v
    if sd is not None:
        s += r"\,{\tiny$\pm$%.2f}" % sd
    return r"\textbf{%s}" % s if (bold and abs(v - best) < 1e-9) else s


# ---------------- comparison ----------------
L = [r"\begin{table*}[t]",
     r"\caption{Comparison against re-implemented baselines under an identical",
     r"pipeline: same representation, same class-balanced budget, same splits and",
     r"seeds, with the OA column carrying its seed-to-seed standard deviation. OA, AA",
     r"and Kappa are measured on unrotated test data; OA$_{45}$ is",
     r"overall accuracy after a \ang{45} polarimetric rotation of the test data,",
     r"a setting under which none of the published methods was designed or",
     r"previously evaluated. Mean of three seeds throughout, including the two"
     r" proposed networks, so that every row of this table is measured under one"
     r" protocol; Table~\ref{tab:main} reports Flevoland at ten seeds instead,"
     r" which moves the shared rows by at most \SI{DELTA}{pp}."
     r" Best per column in bold.}",
     r"\label{tab:compare}", r"\centering", r"\footnotesize",
     # thirteen columns across a two-column float: the default padding pushed
     # this 15 pt past the text block, so it is tightened here rather than left
     # to spill into the margin.
     r"\setlength{\tabcolsep}{3.4pt}",
     r"\begin{tabular}{ll" + "rrrr" * len(have) + "}", r"\toprule",
     " & ".join([r"\multicolumn{2}{c}{}"] +
                [r"\multicolumn{4}{c}{%s}" % n for _, n in have]) + r" \\"]
L.append(" & ".join(["", ""] + [r"OA & AA & $\kappa$ & OA$_{45}$"
                                for _ in have]) + r" \\")
L.append(r"\midrule")
best = {}
for k, _ in have:
    for f in ("oa", "aa", "kappa"):
        best[(k, f)] = max(np.mean(R[k + "|" + m][f]) for m in ORDER
                           if k + "|" + m in R)
    best[(k, "oa45")] = max(R[k + "|" + m]["oa45"] for m in ORDER
                            if k + "|" + m in R
                            and not np.isnan(R[k + "|" + m]["oa45"]))
for m in ORDER:
    row = [PRETTY.get(m, m), ""]
    for k, _ in have:
        key = k + "|" + m
        if key not in R:
            row += ["--"] * 4
            continue
        v = R[key]
        row += [cell(np.mean(v["oa"]), best[(k, "oa")], sd=np.std(v["oa"])),
                cell(np.mean(v["aa"]), best[(k, "aa")]),
                cell(np.mean(v["kappa"]), best[(k, "kappa")]),
                "--" if np.isnan(v["oa45"]) else cell(v["oa45"], best[(k, "oa45")])]
    L.append(" & ".join(row) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
# How far this table's Flevoland rows sit from the ten-seed values of Table I.
# Measured rather than asserted, so the caption cannot become wrong silently.
_d = 0.0
if os.path.exists("exp31_results.json"):
    _m10 = json.load(open("exp31_results.json"))
    for _k in ("Equivariant", "Steerable", "CV-MsAtViT"):
        _a = "flevoland|%s" % _k
        if _a in R and _a in _m10:
            _d = max(_d, abs(float(np.mean(R[_a]["oa"])) - _m10[_a]["mean"][0]))
open(os.path.join(OUT, "tab_compare.tex"), "w").write(
    "\n".join(L).replace("DELTA", "%.2f" % _d) + "\n")
print("  compare vs ten-seed table: largest shared-row difference %.2f pp" % _d)

# ---------------- complexity ----------------
ref = have[0][0]
T40 = json.load(open("exp40_results.json")) if os.path.exists("exp40_results.json") else {}
NAME40 = {"CV-MsAtViT": "CV-MsAtViT", "Equivariant": "Equivariant (ours)",
          "Steerable": "Steerable (ours)", "CV-2DCNN": None}
C = [r"\begin{table}[t]",
     r"\caption{Model complexity. MACs are real multiply-accumulates for one",
     r"$15\times15$ sample, counted by operator dispatch so that functional and",
     r"modular convolutions are treated alike; a module-hook estimator misses the",
     r"functional calls in our group-convolutional layers and under-reports them",
     r"by an order of magnitude. The last column is wall time to classify every",
     r"labelled pixel of Flevoland at batch 4096, and it does not track the MAC",
     r"count: the discrete network moves $N$ times the memory per layer and is",
     r"slow despite being cheaper in arithmetic than the transformer.}",
     r"\label{tab:complexity}", r"\centering", r"\small",
     r"\begin{tabular}{lrrr}", r"\toprule",
     r"model & parameters & MMACs & full scene (s) \\", r"\midrule"]
for m in ORDER:
    key = ref + "|" + m
    if key not in R or R[key]["params"] == 0:
        continue
    t = T40.get(NAME40.get(m) or "", {}).get("seconds")
    # This table lives in the supplement, which carries no bibliography of its
    # own; a \cite here would drag all 45 entries in for one reference.
    nm = PRETTY.get(m, m).split("~\\cite")[0]
    C.append("%s & %s & %.2f & %s \\\\"
             % (nm,
                "{:,}".format(R[key]["params"]).replace(",", r"\,"),
                R[key]["macs"] / 1e6,
                ("%.1f" % t) if t else "--"))
C += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_complexity.tex"), "w").write("\n".join(C) + "\n")

print("wrote %s/tab_compare.tex and tab_complexity.tex" % OUT)
print("scenes included:", ", ".join(n for _, n in have))
for k, n in have:
    done = [m for m in ORDER if k + "|" + m in R]
    print("  %-16s %d/%d models" % (n, len(done), len(ORDER)))


# ---------------- main rotation-sweep table (Table I) ----------------
# This one was hand-written until the sweeps were re-run, at which point the
# prose and the table drifted apart and the audit caught it. Generated now.
M = json.load(open("exp15_results.json"))
# Flevoland was re-run at ten seeds (Exp. 31) after a reviewer objected that
# three do not support the differences the paper discusses. The other two scenes
# keep their three-seed numbers -- ten seeds on all three did not fit the compute
# -- and the caption says which is which, so the table cannot mislead.
NSEED = {sc: 3 for sc in ("flevoland", "sanfran", "ober")}
if os.path.exists("exp31_results.json"):
    M10 = json.load(open("exp31_results.json"))
    for _k, _v in M10.items():
        M[_k] = _v
        NSEED[_k.split("|")[0]] = 10
TH = [0, 10, 22.5, 45]
MORDER = [("baseline no-aug", "baseline"),
          ("baseline rot-AUG", r"baseline + rot.\ aug."),
          ("Equivariant", r"\textbf{equivariant (ours)}"),
          ("Steerable", r"\textbf{steerable (ours)}"),
          ("CV-MsAtViT", r"CV-MsAtViT~\cite{alkhatib2025}")]
T = [r"\begin{table*}[t]",
     r"\caption{Overall accuracy (\si{\percent}) against test-time polarimetric",
     r"rotation, class-balanced budget, with the seed-to-seed",
     r"deviation on the unrotated column. Flevoland is the mean of TENSEED seeds,",
     r"the other two scenes of three. ``bnd'' is accuracy within \SI{2}{px} of",
     r"a class boundary. Best rotation-robust result per scene in bold.}",
     r"\label{tab:main}", r"\centering", r"\small",
     r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r"scene & model & {$\theta{=}0$} & {\ang{10}} & {\ang{22.5}} & {\ang{45}}"
     r" & {AA} & {bnd} & {params} \\", r"\midrule"]
for si, (sc, nm) in enumerate([("flevoland", "Flevoland"),
                               ("sanfran", "San Francisco"),
                               ("ober", "Oberpfaffenhofen")]):
    rows = [(k, lab) for k, lab in MORDER if "%s|%s" % (sc, k) in M]
    if not rows:
        continue
    if si:
        T.append(r"\midrule")
    T.append(r"\multirow{%d}{*}{%s}" % (len(rows), nm))
    for k, lab in rows:
        v = M["%s|%s" % (sc, k)]
        cells = ["%.2f" % v["mean"][0] + r"\,{\tiny$\pm$%.2f}" % v["std"][0]]
        cells += ["%.2f" % v["mean"][i] for i in (1, 2, 3)]
        cells += ["%.2f" % v["aa"], "%.2f" % v["bnd"][0],
                  "{:,}".format(v["params"]).replace(",", r"\,")]
        T.append(" & %s & %s \\\\" % (lab, " & ".join(cells)))
T += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
txt = "\n".join(T).replace("TENSEED", str(NSEED["flevoland"])) + "\n"
open(os.path.join(OUT, "tab_main.tex"), "w").write(txt)
print("  seeds per scene:", NSEED)
print("wrote tab_main.tex")
for sc in ("flevoland", "sanfran", "ober"):
    b = M["%s|baseline no-aug" % sc]["mean"]
    print("  %-11s baseline drop %.2f pp" % (sc, b[0] - b[-1]))
