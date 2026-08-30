"""Per-class accuracy tables, generated from exp23_results.json.

Two tables, both single column: Flevoland has fifteen classes and gets its own,
San Francisco and Oberpfaffenhofen share the second. Only the mean over the
three seeds is tabulated -- the spread is quoted in the caption, since a per-cell
+- would double the width for a number that is almost everywhere below 1 pp.
"""
import json, numpy as np, sys

R = json.load(open("exp23_results.json"))
CLASSES = {
    "flevoland": ["Water", "Forest", "Lucerne", "Grass", "Rapeseed", "Beet",
                  "Potatoes", "Peas", "Stem beans", "Bare soil", "Wheat",
                  "Wheat 2", "Wheat 3", "Barley", "Buildings"],
    "sanfran": ["Bare soil", "Mountain", "Water", "Urban", "Vegetation"],
    "ober": ["Built-up", "Wood land", "Open areas"],
}
NAMES = {"flevoland": "Flevoland", "sanfran": "San Francisco",
         "ober": "Oberpfaffenhofen"}
ARMS = ["baseline", "baseline + rot. aug.", "Equivariant", "Steerable",
        "CV-MsAtViT"]
HEAD = ["CV-CNN", r"CV-CNN$^{\dagger}$", "Ours (E)", "Ours (S)", "CV-MsAtViT"]

missing = ["%s|%s" % (s, a) for s in CLASSES for a in ARMS
           if "%s|%s" % (s, a) not in R]
if missing:
    sys.exit("not finished yet, missing:\n  " + "\n  ".join(missing))

BOLD, PLAIN = r"\textbf{%.2f}", "%.2f"


def row(label, vals):
    """One table line, best entry in bold."""
    b = int(np.argmax(vals))
    cells = [(BOLD if j == b else PLAIN) % x for j, x in enumerate(vals)]
    return label + " & " + " & ".join(cells) + r" \\"


def body(scene, sd_track):
    """Rows for one scene: one line per class, then OA and AA."""
    cols = [R["%s|%s" % (scene, a)] for a in ARMS]
    for c in cols:
        sd_track.append(max(c["per_sd"]))
    out = [row(cl, [c["per"][i] for c in cols])
           for i, cl in enumerate(CLASSES[scene])]
    out.append(r"\hline")
    for key, lab in (("oa", "OA"), ("aa", "AA")):
        out.append(row(r"\textit{%s}" % lab, [c[key] for c in cols]))
    return out


sd = []
allsd = [s for k, v in R.items() for s in v["per_sd"]]
SPREAD = (r"ACROSS THE THREE SEEDS THE PER-CLASS STANDARD DEVIATION HAS "
          r"MEDIAN \SI{%.2f}{PP} AND STAYS BELOW \SI{1}{PP} IN "
          r"\SI{%.0f}{\percent} OF CELLS; THE ONE EXCEPTION IS THE STEERABLE "
          r"NETWORK ON BEET, AT \SI{%.2f}{PP}."
          % (np.median(allsd), 100 * np.mean(np.array(allsd) < 1), max(allsd)))
COLS = "l" + "r" * len(ARMS)
HDR = "Class & " + " & ".join(HEAD) + r" \\"
OPEN = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}"]
GRID = [r"\begin{tabular}{%s}" % COLS, r"\hline", HDR, r"\hline"]
SHUT = [r"\hline", r"\end{tabular}", r"\end{table}"]

# ---- Table: Flevoland -----------------------------------------------------
t1 = OPEN + [
    r"\caption{PER-CLASS ACCURACY (\%) ON FLEVOLAND AT $\theta=0$, MEAN OF "
    r"THREE SEEDS. CV-CNN$^{\dagger}$ IS THE BASELINE TRAINED WITH ROTATION "
    r"AUGMENTATION; (E) AND (S) ARE THE EQUIVARIANT AND STEERABLE NETWORKS. "
    r"BEST IN EACH ROW IN BOLD. " + SPREAD + "}",
    r"\label{tab:perclass_flev}"] + GRID
t1 += body("flevoland", sd) + SHUT

# ---- Table: the other two scenes -----------------------------------------
t2 = OPEN + [
    r"\caption{PER-CLASS ACCURACY (\%) ON SAN FRANCISCO AND OBERPFAFFENHOFEN "
    r"AT $\theta=0$, MEAN OF THREE SEEDS. COLUMNS AS IN "
    r"TABLE~\ref{tab:perclass_flev}.}",
    r"\label{tab:perclass_rest}"] + GRID
for sc in ("sanfran", "ober"):
    t2.append(r"\multicolumn{%d}{l}{\textit{%s}} \\"
              % (len(ARMS) + 1, NAMES[sc]))
    t2 += body(sc, sd)
    if sc == "sanfran":
        t2.append(r"\hline")
t2 += SHUT

open("../paper/tables/tab_perclass.tex", "w", encoding="utf-8").write(
    "\n".join(t1) + "\n\n" + "\n".join(t2) + "\n")
print("wrote paper/tables/tab_perclass.tex")

# ---- the facts the prose quotes, so they cannot be hand-typed wrong -------
F = {}
for sc in CLASSES:
    for a in ARMS:
        c = R["%s|%s" % (sc, a)]
        w = int(np.argmin(c["per"]))
        F["%s|%s" % (sc, a)] = dict(worst_class=CLASSES[sc][w],
                                    worst=round(c["per"][w], 2),
                                    spread=round(max(c["per"]) - min(c["per"]), 2),
                                    oa=round(c["oa"], 2), aa=round(c["aa"], 2))
F["max_class_sd"] = round(float(max(sd)), 2)
json.dump(F, open("exp23_summary.json", "w"), indent=1)

print("largest per-class seed spread over all cells: %.2f pp" % max(sd))
print("\n%-16s %-22s %-12s %6s %7s" % ("scene", "arm", "worst class", "acc", "spread"))
for sc in CLASSES:
    for a in ARMS:
        f = F["%s|%s" % (sc, a)]
        print("%-16s %-22s %-12s %6.2f %7.2f"
              % (sc, a, f["worst_class"], f["worst"], f["spread"]))
