"""Adversarial consistency audit of the manuscript against the measurement files.

Every number asserted in main.tex should be traceable to a stored result. The
sweeps have been re-run several times, so this is the guard against a stale
figure surviving in the prose after the data underneath it changed -- which is
exactly what it caught the last time it ran.
"""
import json, re, sys, os, numpy as np

# Claims now live in two documents, and line breaks fall wherever the paragraph
# happens to wrap. Read both and collapse whitespace, so a pattern matches the
# sentence rather than the current line breaking.
_raw = open("../paper/main.tex", encoding="utf-8").read()
if os.path.exists("../paper/supp.tex"):
    _raw += "\n" + open("../paper/supp.tex", encoding="utf-8").read()
TEX = re.sub(r"\s+", " ", _raw)
E15 = json.load(open("exp15_results.json"))
# The main table takes Flevoland from the ten-seed re-run (Exp. 31) and the other
# two scenes from Exp. 15. The audit must read exactly what the table reads, or
# it certifies the prose against numbers the reader never sees -- which is the
# failure it exists to prevent.
if os.path.exists("exp31_results.json"):
    E15.update(json.load(open("exp31_results.json")))
E18 = json.load(open("exp18_results.json"))
E19 = json.load(open("exp19_results.json"))
E21 = json.load(open("exp21_results.json"))
E22 = json.load(open("exp22_results.json"))
# The patch sweep is reported at ten seeds (Exp. 32); check what is printed.
if os.path.exists("exp32_results.json"):
    E22 = json.load(open("exp32_results.json"))

issues, notes = [], []
add = lambda sev, m: (issues if sev == "X" else notes).append(m)


def claim(desc, pattern, actual, tol=0.15):
    """Compare a number PARSED FROM THE MANUSCRIPT against the measurement.

    An earlier version hard-coded what the paper was believed to say, so after
    the prose was corrected the audit went on reporting the old mismatch. The
    asserted value must come from the file, or the check drifts with the thing
    it is supposed to be checking.
    """
    m = re.search(pattern, TEX)
    if not m:
        print("  %-52s PATTERN NOT FOUND" % desc)
        add("!", "audit pattern no longer matches: " + desc)
        return
    asserted = float(m.group(1))
    d = abs(asserted - actual)
    flag = "X" if d > tol else " "
    print("  %-52s says %-7s data %-7s %s"
          % (desc, "%.2f" % asserted, "%.2f" % actual, flag))
    if d > tol:
        add("X", "%s: manuscript %.2f, data %.2f" % (desc, asserted, actual))


print("=" * 78)
print("1. CROSS-TABLE AGREEMENT  (Table I / exp15 vs Table IV / exp19)")
print("=" * 78)
worst = 0.0
E15_3 = json.load(open("exp15_results.json"))   # exp19 uc tohum: uc tohumla karsilastir
for sc in ("flevoland", "sanfran", "ober"):
    for m in ("Equivariant", "Steerable", "CV-MsAtViT"):
        k = "%s|%s" % (sc, m)
        if k in E15_3 and k in E19:
            a = E15_3[k]["mean"][0]; b = float(np.mean(E19[k]["oa"]))
            worst = max(worst, abs(a - b))
            print("  %-10s %-12s %7.2f %7.2f   delta %.3f" % (sc, m, a, b, abs(a - b)))
print("  worst %.3f pp   (run-to-run spread with a fixed seed is 0.075)" % worst)
if worst > 0.25:
    add("X", "cross-table disagreement %.2f pp exceeds run-to-run spread" % worst)

print()
print("=" * 78)
print("2. CLAIMS vs MEASUREMENTS")
print("=" * 78)
f = E15["flevoland|baseline no-aug"]["mean"]
a = E15["flevoland|baseline rot-AUG"]["mean"]
e = E15["flevoland|Equivariant"]
m = E15["flevoland|CV-MsAtViT"]["mean"]
claim("baseline drop, Flevoland (abstract)",
      r"loses \\SI\{([\d.]+)\}\{pp\} of overall accuracy", f[0] - f[-1])
claim("CV-MsAtViT drop (abstract)",
      r"attention transformer loses \\SI\{([\d.]+)\}\{pp\}", m[0] - m[-1])
claim("CV-MsAtViT drop (results)",
      r"\\SI\{([\d.]+)\}\{pp\} collapse", m[0] - m[-1])
claim("EQ over augmented baseline",
      r"equivariant network gains \\SI\{\+([\d.]+)\}\{pp\}", e["mean"][0] - a[0])
claim("EQ boundary margin",
      r"clears the same test on all three: \\SI\{\+([\d.]+)\}\{pp\}",
      e["bnd"][0] - E15["flevoland|baseline rot-AUG"]["bnd"][0])
claim("normalisation ablation, OA45 drop",
      r"costs \\SI\{([\d.]+)\}\{pp\} at \\ang\{45\}",
      E21["C|irrep"]["mean"]["45"] - E21["C|per-channel"]["mean"]["45"], tol=0.3)
claim("readout gap off-grid",
      r"gains \\SI\{([\d.]+)\}\{pp\} off the grid",
      E21["B|fourier"]["mean"]["10"] - E21["B|max"]["mean"]["10"])
claim("budget 300 accuracy",
      r"\\SI\{([\d.]+)\}\{\\percent\} at 300", E21["D|budget=300"]["mean"]["0"])
sp = max(np.mean(E22["W=%d" % w]["oa"]) for w in (7, 11, 15, 19)) - \
     min(np.mean(E22["W=%d" % w]["oa"]) for w in (7, 11, 15, 19))
# The manuscript no longer quotes the span; it asserts the span is smaller
# than the seed spread. Check that statement instead of a number.
sd_min = min(np.std(E22["W=%d" % w]["oa"]) for w in (7, 11, 15, 19))
ok_span = sp < sd_min
print("  %-52s span %.2f < seed sd %.2f  %s"
      % ("patch span below seed spread", sp, sd_min, " " if ok_span else "X"))
if not ok_span:
    add("X", "manuscript says the patch-size range is smaller than the seed "
             "spread; span %.2f, smallest sd %.2f" % (sp, sd_min))

# --- per-class claims (Sec. per-class accuracy, exp23) ---------------------
P23 = json.load(open("exp23_results.json"))
fb = np.array(P23["flevoland|baseline"]["per"])
fa = np.array(P23["flevoland|baseline + rot. aug."]["per"])
claim("Flevoland rapeseed loss under augmentation",
      r"rapeseed loses \\SI\{([\d.]+)\}\{pp\}", fb[4] - fa[4])
claim("Flevoland wheat 2 loss under augmentation",
      r"wheat~2 \\SI\{([\d.]+)\}\{pp\}", fb[11] - fa[11])
claim("Flevoland beet loss under augmentation",
      r"beet\s*\n?\\SI\{([\d.]+)\}\{pp\}", fb[5] - fa[5])
claim("baseline class spread, Flevoland",
      r"widens from \\num\{([\d.]+)\}", fb.max() - fb.min())
claim("augmented class spread, Flevoland",
      r"widens from \\num\{[\d.]+\} to\s*\n?\\SI\{([\d.]+)\}\{pp\}",
      fa.max() - fa.min())
claim("equivariant class spread, Flevoland",
      r"spread stays at \\SI\{([\d.]+)\}\{pp\}",
      float(np.ptp(np.array(P23["flevoland|Equivariant"]["per"]))))
worst_eq = max(float(max(np.array(P23["%s|baseline" % s]["per"])
                         - np.array(P23["%s|Equivariant" % s]["per"])))
               for s in ("flevoland", "sanfran", "ober"))
claim("worst equivariant per-class loss, any scene",
      r"loses more than\s*\n?\\SI\{([\d.]+)\}\{pp\}", worst_eq)
nhit = int((fb - fa > 2).sum())
if "seven of the fifteen classes" not in TEX or nhit != 7:
    add("X", "Flevoland classes losing >2pp under augmentation is %d, "
             "prose says seven" % nhit)
else:
    print("  %-52s says seven  data %d" % ("classes losing >2pp under aug.", nhit))

# --- per-class under rotation (exp26). These two statements about Buildings
# predated any stored measurement; exp26 is what makes them checkable. --------
P26 = json.load(open("exp26_results.json"))
rb0 = np.array(P26["baseline"]["0.0"]["per"])
rb45 = np.array(P26["baseline"]["45.0"]["per"])
ra0 = np.array(P26["baseline + rot. aug."]["0.0"]["per"])
ra45 = np.array(P26["baseline + rot. aug."]["45.0"]["per"])
re0 = np.array(P26["Equivariant"]["0.0"]["per"])
re45 = np.array(P26["Equivariant"]["45.0"]["per"])
claim("Buildings under rotation, baseline (Sec. fragility)",
      r"Buildings class of Flevoland[^.]*?drops to\s*\n?\\SI\{([\d.]+)\}\{\\percent\}",
      rb45[14])
claim("Buildings held at 100 (discussion)",
      r"Buildings class sits at\s*\n?\\SI\{([\d.]+)\}\{\\percent\}", rb0[14])
if not (ra45[14] == 100.0 and re45[14] == 100.0):
    add("X", "discussion says Buildings stays at 100 under rotation; data "
             "aug %.2f, equivariant %.2f" % (ra45[14], re45[14]))
n0 = int((rb45 == 0).sum()); n1 = int((rb45 < 1).sum()) - n0
ok0 = "sends five of the fifteen classes to" in TEX and n0 == 5
ok1 = "two more below" in TEX and n1 == 2
print("  %-52s says 5/+2   data %d/+%d %s"
      % ("classes at 0.00 and below 1 pct at 45 deg", n0, n1,
         " " if (ok0 and ok1) else "X"))
if not (ok0 and ok1):
    add("X", "class collapse counts at 45 deg are %d at zero and %d more "
             "below one; prose says five and two" % (n0, n1))
claim("beet under rotation", r"\\num\{94.90\}\$\\to\$\\num\{([\d.]+)\}", rb45[5])
claim("potatoes under rotation", r"\\num\{93.31\}\$\\to\$\\num\{([\d.]+)\}", rb45[6])
claim("augmented spread under rotation",
      r"within \\SI\{([\d.]+)\}\{pp\} of its unrotated value",
      float(np.abs(ra0 - ra45).max()))
d_eq = float(np.abs(re0 - re45).max())
print("  %-52s exact    data %.1e %s"
      % ("equivariant per-class identity at 45 deg", d_eq,
         " " if d_eq == 0 else "X"))
if d_eq != 0:
    add("X", "prose claims exact per-class identity, max diff %.2e" % d_eq)

sd23 = max(max(v["per_sd"]) for v in P23.values())
claim("largest per-class seed spread",
      r"seeds differ by\s*\n?\\SI\{([\d.]+)\}\{pp\}", sd23)


# --- orientation angle compensation (Sec. oac, exp27) ----------------------
if os.path.exists("exp27_results.json"):
    P27 = json.load(open("exp27_results.json"))
    def worst(sc, key, store=None):
        d = (store or P27)["%s|%s" % (sc, key)]["mean"]
        return d[0] - min(d)
    for sc, nm in (("flevoland", "Flevoland"), ("sanfran", "San Francisco"),
                   ("ober", "Oberpfaffenhofen")):
        b = E15["%s|baseline no-aug" % sc]["mean"]
        base_drop = b[0] - min(b)
        best_oac = min(worst(sc, "OAC-%s" % m) for m in ("pixel", "patch", "smooth"))
        claim("OAC best worst-case drop, %s" % nm,
              r"to \\SI\{([\d.]+)\}\{pp\} on %s" % nm.replace(" ", r"\s+"),
              best_oac)
    claim("smoothed OAC loss at 45 deg, Flevoland",
          r"still loses\s*\n?\\SI\{([\d.]+)\}\{pp\} at \\ang\{45\}",
          worst("flevoland", "OAC-smooth"))
    cl_pix = P27["flevoland|OAC-pixel"]["mean"][0]
    claim("per-pixel OAC clean cost, Flevoland",
          r"gives up \\SI\{([\d.]+)\}\{pp\} of clean accuracy",
          E15["flevoland|baseline no-aug"]["mean"][0] - cl_pix)
    # the two networks' own worst-case drops, quoted in the same paragraph
    eq = max((lambda m: m[0] - min(m))(E15["%s|Equivariant" % s]["mean"])
             for s in ("flevoland", "sanfran", "ober"))
    st = max((lambda m: m[0] - min(m))(E15["%s|Steerable" % s]["mean"])
             for s in ("flevoland", "sanfran", "ober"))
    claim("discrete network worst drop, any scene",
          r"to \\SI\{([\d.]+)\}\{pp\}\. The steerable network gives up", eq)
    print("  %-52s data %.2f  (prose: exactly 0.00)"
          % ("steerable worst drop, any scene", st))
    if st != 0.0:
        add("X", "prose claims the steerable network is exactly flat; "
                 "worst drop is %.4f" % st)


# --- AIR-PolSAR-Seg-2.0 (Sec. air, exp34) ---------------------------------
if os.path.exists("exp34_results.json"):
    P34 = json.load(open("exp34_results.json"))
    NM = {"air_gz": "Guangzhou", "air_sh": "Shanghai", "air_bj": "Beijing"}
    for sc, nm in NM.items():
        e = P34["%s|Equivariant" % sc]; m = P34["%s|CV-MsAtViT" % sc]
        claim("AIR %s: CV-MsAtViT rotation loss" % nm,
              r"loses \\num\{29.52\}, \\num\{27.42\} and\s*\n?\\SI\{([\d.]+)\}\{pp\}"
              if sc == "air_bj" else
              (r"loses \\num\{([\d.]+)\}, \\num\{27.42\}" if sc == "air_gz"
               else r"loses \\num\{29.52\}, \\num\{([\d.]+)\}"),
              m["mean"][0] - min(m["mean"]))
        st = P34["%s|Steerable" % sc]
        if st["mean"][0] - min(st["mean"]) != 0.0:
            add("X", "prose says the steerable net is flat on all three AIR "
                     "regions; %s drops %.2f" % (nm, st["mean"][0] - min(st["mean"])))
    eq_bj = P34["air_bj|Equivariant"]
    claim("AIR Beijing: equivariant off-grid loss",
          r"on Beijing it loses \\SI\{([\d.]+)\}\{pp\}",
          eq_bj["mean"][0] - min(eq_bj["mean"]))
    for sc, pat in (("air_gz", r"ahead by\s*\n?\\SI\{([\d.]+)\}\{pp\} on Guangzhou"),
                    ("air_sh", r"behind by \\SI\{([\d.]+)\}\{pp\} on Shanghai"),
                    ("air_bj", r"ahead by\s*\n?\\SI\{([\d.]+)\}\{pp\} on Beijing")):
        g = (P34["%s|Equivariant" % sc]["mean"][0]
             - P34["%s|CV-MsAtViT" % sc]["mean"][0])
        claim("AIR %s: gap to CV-MsAtViT" % NM[sc], pat, abs(g))

# --- Clebsch-Gordan truncation (Sec. kmax, exp30) -------------------------
if os.path.exists("exp30_results.json"):
    P30 = json.load(open("exp30_results.json"))
    for k, pat in ((6, r"to \\SI\{([\d.]+)\}\{\\percent\} and"),
                   (8, r"and \\SI\{([\d.]+)\}\{\\percent\} against"),
                   (4, r"against \\SI\{([\d.]+)\}\{\\percent\}, while raising")):
        claim("kmax=%d accuracy" % k, pat,
              P30["flevoland|kmax=%d" % k]["mean"][0])
    for k in (4, 6, 8):
        v = P30["flevoland|kmax=%d" % k]
        if v["mean"][0] - min(v["mean"]) != 0.0:
            add("X", "prose says every kmax variant is flat; kmax=%d drops %.3f"
                     % (k, v["mean"][0] - min(v["mean"])))

# --- patch overlap (Sec. proto, exp39). Both figures were unverifiable until
# Exp. 39: the prose number came from nowhere and the table column was a
# hard-coded dictionary, and the two disagreed with each other. -----------
if os.path.exists("exp39_results.json"):
    P39 = json.load(open("exp39_results.json"))
    claim("patch overlap, balanced budget, Flevoland W=15",
          r"\\SI\{([\d.]+)\}\{\\percent\} under the class-balanced budget",
          P39["flevoland|balanced|W=15"]["overlap"])
    claim("patch overlap, one-per-cent split, Flevoland W=15",
          r"\\SI\{([\d.]+)\}\{\\percent\} under the one-per-cent stratified",
          P39["flevoland|ratio1|W=15"]["overlap"])
    claim("patch overlap quoted in the abstract",
          r"\\SI\{([\d.]+)\}\{\\percent\} of Flevoland test pixels",
          P39["flevoland|balanced|W=15"]["overlap"], tol=0.06)

pe = E19["flevoland|Equivariant"]["params"]; pm = E19["flevoland|CV-MsAtViT"]["params"]
print("  %-52s ratio 1:%.1f" % ("params, equivariant vs CV-MsAtViT", pm / pe))

print()
print("=" * 78)
print("3. PROTOCOL")
print("=" * 78)
ratio = max(np.array(E18[s]["clean"]).std() / np.array(E18[s]["leaky"]).std() for s in E18)
print("  worst sigma ratio %.1f   (abstract: 'up to 25')" % ratio)
if abs(ratio - 25) > 4:
    add("X", "sigma ratio claim off: %.1f" % ratio)

print()
print("=" * 78)
print("4. COMPLETENESS")
print("=" * 78)
checks = [("per-class accuracy table", r"per-class"),
          ("training-ratio / budget sweep", r"labels per\s+class|label budget"),
          ("patch-size ablation", r"[Pp]atch size"),
          ("ablation of own components", r"subsection\{Ablation"),
          ("runtime / inference cost",
           r"inference time|runtime|throughput|training time|[Ww]all time"),
          ("code availability statement", r"code (is )?availab|github|will be released")]
for nm, pat in checks:
    ok = re.search(pat, TEX) is not None
    print("  %-34s %s" % (nm, "present" if ok else "MISSING"))
    if not ok:
        add("!", "missing: " + nm)
nref = TEX.count(r"\cite{") and open("../paper/refs.tex", encoding="utf-8").read().count(r"\bibitem")
todo = open("../paper/refs.tex", encoding="utf-8").read().count("VERIFY AUTHORS")
print("  %-34s %d  (%d need author completion)" % ("bibliography entries", nref, todo))
if todo:
    add("X", "%d bibliography entries still marked VERIFY AUTHORS" % todo)

print()
print("=" * 78)
print("BLOCKING (%d)" % len(issues))
for i in issues: print("  X  " + i)
print("WEAKNESS (%d)" % len(notes))
for n in notes: print("  !  " + n)
