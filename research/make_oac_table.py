"""The orientation-angle-compensation comparison table (Exp. 27 and Exp. 15).

Rows are methods, columns are test-time rotation angles. The point of the table
is not that our network wins everywhere -- it does not -- but that the three
routes to invariance fail in different places, and the reader should be able to
see where.
"""
import json, numpy as np, sys

E15 = json.load(open("exp15_results.json"))
# Table III reports Flevoland at ten seeds; this table shares four of its
# rows, and two tables in one paper must not disagree about the same
# quantity. The OAC rows themselves are three seeds, which the caption says.
import os
if os.path.exists("exp31_results.json"):
    E15.update(json.load(open("exp31_results.json")))
E27 = json.load(open("exp27_results.json"))
THETAS = [0, 10, 22.5, 45]
SCENES = [("flevoland", "Flevoland"), ("sanfran", "San Francisco"),
          ("ober", "Oberpfaffenhofen")]

ROWS = [("CV-CNN", "e15", "baseline no-aug"),
        ("CV-CNN + rot. aug.", "e15", "baseline rot-AUG"),
        ("CV-CNN + OAC (per pixel)", "e27", "OAC-pixel"),
        ("CV-CNN + OAC (per patch)", "e27", "OAC-patch"),
        ("CV-CNN + OAC (smoothed)", "e27", "OAC-smooth"),
        ("Equivariant (ours)", "e15", "Equivariant"),
        ("Steerable (ours)", "e15", "Steerable")]

missing = []
for sc, _ in SCENES:
    for _, src, key in ROWS:
        d = E15 if src == "e15" else E27
        if "%s|%s" % (sc, key) not in d:
            missing.append("%s|%s" % (sc, key))
if missing:
    sys.exit("not finished yet, missing:\n  " + "\n  ".join(missing))


def vals(sc, src, key):
    d = E15 if src == "e15" else E27
    return d["%s|%s" % (sc, key)]["mean"]


L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{3.5pt}",
     r"\caption{OVERALL ACCURACY (\%) AGAINST TEST-TIME ROTATION ANGLE, MEAN OF "
     r"THREE SEEDS FOR THE OAC ROWS AND OF TEN SEEDS ON FLEVOLAND FOR THE "
     r"ROWS SHARED WITH TABLE~\ref{tab:main}. OAC IS THE ORIENTATION ANGLE "
     r"COMPENSATION OF LEE AND "
     r"AINSWORTH APPLIED AS PREPROCESSING TO THE SAME BASELINE CNN, "
     r"CANONICALISING EACH PIXEL, EACH PATCH BY ITS CENTRE, OR EACH PATCH BY "
     r"ITS AVERAGED WEIGHT-4 COMPONENT. THE LAST COLUMN IS THE LARGEST DROP "
     r"FROM THE UNROTATED VALUE.}",
     r"\label{tab:oac}",
     r"\begin{tabular}{l" + "r" * (len(THETAS) + 1) + "}", r"\hline",
     "Method & " + " & ".join(r"\ang{%g}" % t for t in THETAS)
     + r" & worst \\", r"\hline"]

for sc, nm in SCENES:
    L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(THETAS) + 2, nm))
    body = [(lab, vals(sc, src, key)) for lab, src, key in ROWS]
    best = [max(v[i] for _, v in body) for i in range(len(THETAS))]
    drops = [v[0] - min(v) for _, v in body]
    bd = min(drops)
    for (lab, v), dr in zip(body, drops):
        cells = [(r"\textbf{%.2f}" if abs(x - best[i]) < 1e-9 else "%.2f") % x
                 for i, x in enumerate(v)]
        cells.append((r"\textbf{%.2f}" if abs(dr - bd) < 1e-9 else "%.2f") % dr)
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L.append(r"\hline")
L += [r"\end{tabular}", r"\end{table}"]

open("../paper/tables/tab_oac.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote paper/tables/tab_oac.tex")

# --- the facts the prose quotes -------------------------------------------
F = {}
for sc, _ in SCENES:
    base = vals(sc, "e15", "baseline no-aug")
    for lab, src, key in ROWS:
        v = vals(sc, src, key)
        F["%s|%s" % (sc, key)] = dict(clean=round(v[0], 2),
                                      worst=round(min(v), 2),
                                      drop=round(v[0] - min(v), 2),
                                      clean_vs_base=round(v[0] - base[0], 2))
json.dump(F, open("exp27_summary.json", "w"), indent=1)

print("\n%-12s %-26s %7s %7s %7s" % ("scene", "method", "clean", "worst", "drop"))
for sc, _ in SCENES:
    for lab, src, key in ROWS:
        f = F["%s|%s" % (sc, key)]
        print("%-12s %-26s %7.2f %7.2f %7.2f"
              % (sc, lab, f["clean"], f["worst"], f["drop"]))
