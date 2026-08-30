"""Tables for the three experiments added during revision: the branch-cut
measurement (Exp. 28), the spatially structured rotation (Exp. 29) and the
cross-acquisition transfer (Exp. 33).

As everywhere else, the numbers are read from the stored measurements; nothing
here is transcribed.
"""
import json, os, numpy as np

SC = [("flevoland", "Flevoland"), ("sanfran", "San Francisco"),
      ("ober", "Oberpfaffenhofen")]
OUT = "../paper/tables/"
facts = {}


def w(name, lines):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", name)


# ---------------------------------------------------------------- Exp. 28
E28 = json.load(open("exp28_results.json"))
ANG = ["10.0", "22.5", "45.0"]
MODE = [("pixel", "per pixel"), ("patch", "per patch"), ("smooth", "smoothed")]
L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\caption{THE BRANCH CUT, MEASURED. PERCENTAGE OF PIXELS WHOSE WEIGHT-2 "
     r"SECTOR ($T_{12}$, $T_{13}$) CHANGES SIGN AFTER CANONICALISATION, WHICH "
     r"IS WHAT A WRAPPED ESTIMATE DOES. THE INVARIANT CHANNEL IS UNAFFECTED AT "
     r"EVERY ANGLE (BELOW $10^{-7}$ RELATIVE).}",
     r"\label{tab:ambig}",
     r"\begin{tabular}{ll" + "r" * len(ANG) + "}", r"\hline",
     r"Scene & Canonicalisation & " + " & ".join(r"\ang{%g}" % float(a) for a in ANG)
     + r" \\", r"\hline"]
for sc, nm in SC:
    for i, (m, ml) in enumerate(MODE):
        lab = nm if i == 0 else ""
        v = [100 * E28[sc][m][a]["T12"]["flipped"] for a in ANG]
        L.append("%s & %s & %s \\\\" % (lab, ml, " & ".join("%.1f" % x for x in v)))
    L.append(r"\hline")
L += [r"\end{tabular}", r"\end{table}"]
w("tab_ambig.tex", L)
facts["flip45"] = [round(100 * E28[s][m]["45.0"]["T12"]["flipped"], 1)
                   for s, _ in SC for m, _ in MODE]
facts["flip45_min"] = min(facts["flip45"]); facts["flip45_max"] = max(facts["flip45"])
facts["flip10_max"] = round(max(100 * E28[s][m]["10.0"]["T12"]["flipped"]
                                for s, _ in SC for m, _ in MODE), 1)
facts["inv_max"] = max(E28[s][m][a]["T11"]["reldiff"]
                       for s, _ in SC for m, _ in MODE for a in ANG)

# ---------------------------------------------------------------- Exp. 29
E29 = json.load(open("exp29_results.json"))
AMPS = E29["flevoland|baseline"]["amps"]
ROWS = ["baseline", "baseline + rot. aug.", "CV-CNN + OAC", "Equivariant",
        "Steerable"]
LAB = {"baseline": "CV-CNN", "baseline + rot. aug.": "CV-CNN + rot. aug.",
       "CV-CNN + OAC": "CV-CNN + OAC", "Equivariant": "Equivariant (ours)",
       "Steerable": "Steerable (ours)"}
L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{4pt}",
     r"\caption{A SPATIALLY STRUCTURED ROTATION INSTEAD OF A CONSTANT ONE. "
     r"OVERALL ACCURACY (\%) AGAINST THE STANDARD DEVIATION OF A SMOOTH "
     r"RANDOM ROTATION FIELD WHOSE CORRELATION LENGTH IS MATCHED TO THE "
     r"ORIENTATION STRUCTURE MEASURED IN EACH SCENE. THE AMPLITUDE THE SCENES "
     r"THEMSELVES EXHIBIT IS GIVEN IN THE TEXT.}",
     r"\label{tab:field}",
     r"\begin{tabular}{l" + "r" * len(AMPS) + "}", r"\hline",
     r"Method & " + " & ".join(r"\ang{%g}" % a for a in AMPS) + r" \\", r"\hline"]
for sc, nm in SC:
    L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(AMPS) + 1, nm))
    for k in ROWS:
        v = E29["%s|%s" % (sc, k)]["mean"]
        best = max(E29["%s|%s" % (sc, r)]["mean"][-1] for r in ROWS)
        cells = [("\\textbf{%.2f}" if (i == len(v) - 1 and abs(x - best) < 1e-9)
                  else "%.2f") % x for i, x in enumerate(v)]
        L.append(LAB[k] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\hline")
L += [r"\end{tabular}", r"\end{table}"]
w("tab_field.tex", L)
facts["field_sd"] = {s: round(E29["%s|baseline" % s]["sd_measured"], 2) for s, _ in SC}
facts["field_sigma"] = {s: E29["%s|baseline" % s]["sigma"] for s, _ in SC}
for s, _ in SC:
    facts["drop20_%s" % s] = {
        k: round(E29["%s|%s" % (s, k)]["mean"][0] - E29["%s|%s" % (s, k)]["mean"][-1], 2)
        for k in ROWS}

# ---------------------------------------------------------------- Exp. 33
E33 = json.load(open("exp33_results.json"))
o = E33["orientation"]
L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\caption{TRANSFER BETWEEN THE TWO REAL ACQUISITIONS OF THE "
     r"OBERPFAFFENHOFEN REPEAT-PASS PAIR. TRAINED ON THE FIRST, EVALUATED ON "
     r"BOTH; NORMALISATION CONSTANTS COME FROM THE FIRST ONLY. THE MEASURED "
     r"ORIENTATION DIFFERENCE BETWEEN THE ACQUISITIONS IS "
     r"\SI{" + "%.2f" % o["median_abs"] + r"}{\degree} IN MEDIAN ABSOLUTE VALUE.}",
     r"\label{tab:crossacq}",
     r"\begin{tabular}{lrrr}", r"\hline",
     r"Method & Acq.~1 & Acq.~2 & $\Delta$ \\", r"\hline"]
for k in ROWS:
    if k not in E33:
        continue
    v = E33[k]
    L.append("%s & %.2f & %.2f & %.2f \\\\"
             % (LAB[k], v["acq1"], v["acq2"], v["acq1"] - v["acq2"]))
L += [r"\hline", r"\end{tabular}", r"\end{table}"]
w("tab_crossacq.tex", L)
facts["crossacq"] = {k: round(E33[k]["acq1"] - E33[k]["acq2"], 2)
                     for k in ROWS if k in E33}
facts["crossacq_orient"] = {a: round(o[a], 3) for a in
                            ("median_abs", "sd", "frac_gt5")}

json.dump(facts, open("revision_facts.json", "w"), indent=1)
print()
for k, v in facts.items():
    print("  %-18s %s" % (k, v))
