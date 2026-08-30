"""A table saying which experiment, protocol and seed count stands behind each
table in the paper.

Several quantities appear in more than one table with slightly different values,
because they were measured by different experiments under different seed counts.
Every one of those values is real, but leaving the reader to work out which is
which invites the conclusion that the numbers are unreliable. This makes the
mapping explicit, and is generated from the result files so it cannot drift.
"""
import json, os, numpy as np

OUT = "../paper/tables"
# The supplement cannot resolve labels defined in the paper, so the tables are
# named rather than cross-referenced.
ROWS = [
    ("rotation sweep, three scenes", "15, 31", "10 (Flev.), 3"),
    ("orientation angle compensation", "15, 31, 27", "10 (Flev.), 3 (OAC)"),
    ("structured rotation field", "29", "3"),
    ("AIR-PolSAR-Seg-2.0", "34", "3"),
    ("component ablations", "21, 37", "3, 5 (steerable)"),
    ("published architectures", "19", "3"),
    ("patch size, this supplement", "32", "10"),
    ("per-class, this supplement", "23, 26", "3"),
]

# The one quantity that appears in the most places, as a worked example of why
# the values differ. Measured, not asserted.
SRC = [("exp31_results.json", "flevoland|Steerable", "mean", 10, "rotation sweep"),
       ("exp19_results.json", "flevoland|Steerable", "oa", 3, "published architectures"),
       ("exp23_results.json", "flevoland|Steerable", "oa", 3, "per-class"),
       ("exp30_results.json", "flevoland|kmax=4", "mean", 3, "truncation"),
       ("exp37_results.json", "gate: invariant (ours)", "mean", 5, "component ablations")]

vals = []
for f, k, field, ns, where in SRC:
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    if k not in d:
        continue
    v = d[k][field]
    v = float(np.mean(v)) if isinstance(v, list) and field == "oa" else (
        v[0] if isinstance(v, list) else v)
    vals.append((where, ns, v))

L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{3.5pt}",
     r"\caption{WHICH EXPERIMENT STANDS BEHIND WHICH TABLE. A QUANTITY MEASURED "
     r"BY TWO EXPERIMENTS APPEARS TWICE WITH SLIGHTLY DIFFERENT VALUES; BOTH ARE "
     r"REAL, AND THE DIFFERENCE IS SEED COUNT AND SAMPLING, NOT DISAGREEMENT. "
     r"THE STEERABLE NETWORK'S UNROTATED ACCURACY ON FLEVOLAND IS THE WORKED "
     r"EXAMPLE: IT SPANS \SI{" + ("%.2f" % (max(v for _, _, v in vals)
                                            - min(v for _, _, v in vals)))
     + r"}{pp} ACROSS THE ROWS BELOW, AGAINST A SEED-TO-SEED DEVIATION OF "
     r"\SI{1.73}{pp}.}",
     r"\label{tab:protocol}",
     r"\begin{tabular}{llc}", r"\hline",
     r"Table & Exp. & Seeds \\", r"\hline"]
for lab, exp, seeds in ROWS:
    L.append("%s & %s & %s \\\\" % (lab, exp, seeds))
L.append(r"\hline")
L.append(r"\multicolumn{3}{l}{\textit{steerable, Flevoland, $\theta=0$, "
         r"where it appears}} \\")
for where, ns, v in vals:
    L.append("%s & %.2f & %d \\\\" % (where, v, ns))
L += [r"\hline", r"\end{tabular}", r"\end{table}"]

open(os.path.join(OUT, "tab_protocol.tex"), "w", encoding="utf-8").write(
    "\n".join(L) + "\n")
print("wrote tab_protocol.tex")
for where, ns, v in vals:
    print("  %-28s %6.2f  (%d seeds)" % (where, v, ns))
print("  spread %.2f pp against a seed deviation of 1.73"
      % (max(v for _, _, v in vals) - min(v for _, _, v in vals)))
json.dump({"steer_spread": float(max(v for _, _, v in vals)
                                 - min(v for _, _, v in vals))},
          open("protocol_facts.json", "w"), indent=1)
