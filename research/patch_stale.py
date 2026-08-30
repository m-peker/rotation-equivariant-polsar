"""Replace the hand-written Table I with the generated one and refresh the
numbers the audit flagged as stale after the sweeps were re-run.
"""
import re, json, numpy as np

M = json.load(open("exp15_results.json"))
f = M["flevoland|baseline no-aug"]["mean"]
m = M["flevoland|CV-MsAtViT"]["mean"]
e = M["flevoland|Equivariant"]["mean"]
a = M["flevoland|baseline rot-AUG"]["mean"]
sf = M["sanfran|baseline no-aug"]["mean"]
ob = M["ober|baseline no-aug"]["mean"]

p = "../paper/main.tex"
s = open(p, encoding="utf-8").read()

# 1. swap the hand-written table for the generated one
i = s.index(r"\begin{table*}[t]", s.index(r"\label{tab:main}") - 4000)
j = s.index(r"\end{table*}", i) + len(r"\end{table*}")
assert r"\label{tab:main}" in s[i:j], "did not isolate the main table"
s = s[:i] + r"\input{tables/tab_main}" + s[j:]

# 2. refresh the prose figures
SUB = [
    (r"loses \SI{54.7}{pp}", r"loses \SI{%.1f}{pp}" % (f[0] - f[-1])),
    (r"multiscale attention transformer loses \SI{47.8}{pp}",
     r"multiscale attention transformer loses \SI{%.1f}{pp}" % (m[0] - m[-1])),
    (r"\SI{47.8}{pp} collapse", r"\SI{%.1f}{pp} collapse" % (m[0] - m[-1])),
    (r"\SI{97.06}{\percent} to \SI{42.51}{\percent}",
     r"\SI{%.2f}{\percent} to \SI{%.2f}{\percent}" % (f[0], f[-1])),
    (r"(\SI{97.81}{\percent}), falls to \SI{50.02}{\percent}",
     r"(\SI{%.2f}{\percent}), falls to \SI{%.2f}{\percent}" % (m[0], m[-1])),
    (r"the corresponding drops are \SI{63.6}{pp} and \SI{60.7}{pp}",
     r"the corresponding drops are \SI{%.1f}{pp} and \SI{%.1f}{pp}"
     % (sf[0] - sf[-1], ob[0] - ob[-1])),
]
missed = []
for old, new in SUB:
    if old in s:
        s = s.replace(old, new)
    else:
        missed.append(old[:52])

open(p, "w", encoding="utf-8").write(s)
print("Table I now generated; %d prose figures refreshed" % (len(SUB) - len(missed)))
for x in missed:
    print("  NOT FOUND, check by hand: " + x)
print("  baseline drop  %.2f | CV-MsAtViT drop %.2f | SF %.2f | Ober %.2f"
      % (f[0] - f[-1], m[0] - m[-1], sf[0] - sf[-1], ob[0] - ob[-1]))
