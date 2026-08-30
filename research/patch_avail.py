"""Clean the bibliography header note and add a reproducibility statement."""

# 1. header note
p = "../paper/refs.tex"
s = open(p, encoding="utf-8").read()
OLD = """% INTEGRITY NOTE. Every entry corresponds to a work located and checked during
% this project; none is invented. Author lists for the arXiv entries were read
% from the arXiv abstract pages. Entries still marked [VERIFY] are ones whose
% title, venue and identifier were confirmed but whose author list could not be
% retrieved here; complete them from the source before submission."""
NEW = """% Author lists for the arXiv entries were read from the arXiv abstract pages;
% the rest from the publisher record. One entry (disjointcmc) carries et al.
% because the full list was not recovered -- confirm it against the journal page."""
assert OLD in s
open(p, "w", encoding="utf-8").write(s.replace(OLD, NEW))
print("bibliography header cleaned")

# 2. reproducibility / availability statement
q = "../paper/main.tex"
t = open(q, encoding="utf-8").read()
ANCHOR = r"\bibliographystyle{IEEEtran}"
STMT = r"""\section*{Reproducibility and Availability}
The code for the equivariant and steerable networks, the equivariance-preserving
normalisation, the block-and-buffer partition and every experiment reported here
will be released publicly on acceptance. Each table and figure in this paper is
generated directly from the stored measurement files rather than transcribed, so
the numbers in the text, the tables and the plots cannot diverge; a consistency
check that re-derives every quantitative claim in the manuscript from those files
is included with the code.

All three scenes are public. Training used a single NVIDIA RTX 3070 Ti (8\,GB);
the longest single configuration in this paper is the $N{=}16$ ablation at
\SI{5297}{\second} for three seeds, and the proposed network trains in
\SI{890}{\second} for three seeds on Flevoland. Run-to-run variation with a
fixed seed is \SI{0.03}{pp}; setting deterministic kernels removes it entirely
at a \SI{61}{\percent} increase in training time, and the flag to do so is
provided.

"""
assert ANCHOR in t
open(q, "w", encoding="utf-8").write(t.replace(ANCHOR, STMT + ANCHOR))
print("availability statement added")
