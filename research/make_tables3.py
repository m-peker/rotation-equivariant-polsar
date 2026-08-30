"""Tables for AIR-PolSAR-Seg-2.0 (Exp. 34) and the Clebsch-Gordan truncation
ablation (Exp. 30).
"""
import json, os, numpy as np

OUT = "../paper/tables/"
E34 = json.load(open("exp34_results.json"))
E30 = json.load(open("exp30_results.json"))
facts = {}

REG = [("air_gz", "Guangzhou"), ("air_sh", "Shanghai"), ("air_bj", "Beijing")]
ROWS = [("baseline", "CV-CNN"), ("baseline + rot. aug.", "CV-CNN + rot. aug."),
        ("CV-CNN + OAC", "CV-CNN + OAC"), ("Equivariant", "Equivariant (ours)"),
        ("Steerable", "Steerable (ours)"), ("CV-MsAtViT", "CV-MsAtViT")]
TH = [0, 10, 22.5, 45]


def w(name, lines):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", name)


# ------------------------------------------------------------------ AIR
L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{3.2pt}",
     r"\caption{AIR-PolSAR-SEG-2.0. OVERALL AND AVERAGE ACCURACY (\%) AT "
     r"$\theta=0$, AND OVERALL ACCURACY AT THE ROTATED ANGLES, MEAN OF THREE "
     r"SEEDS. THE CLASSES ARE HEAVILY IMBALANCED, SO AA IS THE INFORMATIVE "
     r"COLUMN; ON GUANGZHOU THE MAJORITY CLASS ALONE SCORES "
     r"\SI{56.37}{\percent} AND NO METHOD EXCEEDS IT, SO OA THERE ORDERS "
     r"NOTHING. THE ROTATION IS THE GROUP ACTION APPLIED BY US, NOT A PHYSICAL "
     r"ROTATION: THIS ARCHIVE IS L1A AND CARRIES NO CALIBRATION CONSTANTS.}",
     r"\label{tab:air}",
     r"\begin{tabular}{lrr" + "r" * (len(TH) - 1) + "}", r"\hline",
     r"Method & OA & AA & " + " & ".join(r"\ang{%g}" % t for t in TH[1:])
     + r" \\", r"\hline"]
for sc, nm in REG:
    L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(TH) + 2, nm))
    oa = {k: E34["%s|%s" % (sc, k)]["mean"][0] for k, _ in ROWS}
    aa = {k: E34["%s|%s" % (sc, k)]["aa"] for k, _ in ROWS}
    bo, ba = max(oa.values()), max(aa.values())
    for k, lab in ROWS:
        v = E34["%s|%s" % (sc, k)]["mean"]
        c = [(r"\textbf{%.2f}" if abs(v[0] - bo) < 1e-9 else "%.2f") % v[0],
             (r"\textbf{%.2f}" if abs(aa[k] - ba) < 1e-9 else "%.2f") % aa[k]]
        c += ["%.2f" % x for x in v[1:]]
        L.append(lab + " & " + " & ".join(c) + r" \\")
    L.append(r"\hline")
L += [r"\end{tabular}", r"\end{table}"]
w("tab_air.tex", L)

for sc, nm in REG:
    e = E34["%s|Equivariant" % sc]; m = E34["%s|CV-MsAtViT" % sc]
    facts["%s_oa_gap" % sc] = round(e["mean"][0] - m["mean"][0], 2)
    facts["%s_aa_gap" % sc] = round(e["aa"] - m["aa"], 2)
    facts["%s_msat_drop" % sc] = round(m["mean"][0] - min(m["mean"]), 2)
    facts["%s_eq_drop" % sc] = round(e["mean"][0] - min(e["mean"]), 2)
    facts["%s_st_drop" % sc] = round(E34["%s|Steerable" % sc]["mean"][0]
                                     - min(E34["%s|Steerable" % sc]["mean"]), 2)

# ----------------------------------------------------------------- kmax
L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\caption{CLEBSCH--GORDAN TRUNCATION. CARRYING THE PRODUCTS BEYOND "
     r"$k_{\max}=4$ OPENS WEIGHT CLASSES THE COHERENCY MATRIX DOES NOT "
     r"POPULATE. EQUIVARIANCE IS UNAFFECTED IN EVERY CASE---ALL THREE ARE FLAT "
     r"ACROSS THE FOUR ANGLES---BUT ACCURACY IS NOT. FLEVOLAND, THREE SEEDS.}",
     r"\label{tab:kmax}",
     r"\begin{tabular}{lrrr}", r"\hline",
     r"$k_{\max}$ & OA at $\theta=0$ & OA at \ang{45} & Parameters \\",
     r"\hline"]
for k in (4, 6, 8):
    v = E30["flevoland|kmax=%d" % k]
    L.append("$%d$ & %.2f & %.2f & \\num{%d} \\\\"
             % (k, v["mean"][0], v["mean"][-1], v["params"]))
L += [r"\hline", r"\end{tabular}", r"\end{table}"]
w("tab_kmax.tex", L)
for k in (4, 6, 8):
    v = E30["flevoland|kmax=%d" % k]
    facts["kmax%d" % k] = [round(v["mean"][0], 2), v["params"],
                           round(v["mean"][0] - min(v["mean"]), 4)]

json.dump(facts, open("air_kmax_facts.json", "w"), indent=1)
print()
for k, v in facts.items():
    print("  %-18s %s" % (k, v))
