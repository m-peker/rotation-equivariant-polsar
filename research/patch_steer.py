"""Replace the steerable-vs-discrete passage with the three-scene picture.

The earlier text generalised a Flevoland-only number. Written as a script rather
than an inline edit because the passage contains LaTeX escapes that shell
heredocs mangle.
"""
OLD = """Third, the accuracy of the two proposed variants differs.
The discrete network reaches \\SI{97.15}{\\percent} against the steerable
network's \\SI{95.24}{\\percent} on Flevoland. Exact equivariance for
\\emph{all} angles is therefore not free: constraining the hypothesis space
continuously costs about \\SI{1.9}{pp} here relative to constraining it on a
grid of eight. We report both rather than only the stronger, because the two
occupy genuinely different points---the discrete variant is more accurate, the
steerable variant carries an unconditional guarantee and is what makes the
pixel-exact result of Fig.~\\ref{fig:maps} possible."""

NEW = """Third, the two proposed variants differ from each other, and the difference is
scene-dependent in a way worth stating precisely. Relative to the discrete
network, the steerable network scores \\num{-2.22}, \\num{-0.10} and
\\SI{+0.11}{pp} on Flevoland, San Francisco and Oberpfaffenhofen, with boundary
margins of \\num{-4.50}, \\num{-1.93} and \\SI{+0.83}{pp}. Exact equivariance for
\\emph{all} angles is therefore not free, but neither is it uniformly costly: the
penalty is confined to Flevoland and vanishes---indeed reverses---on the other
two. Those scenes carry 15, 5 and 3 classes respectively and the ordering
follows that, which suggests the cost of the stricter constraint grows with the
number of classes the representation must separate. With three scenes we offer
that as an observation rather than a law.

We report both variants rather than only the stronger, because they occupy
genuinely different points. The discrete network is the more accurate where the
label space is large; the steerable network carries an unconditional guarantee,
and it is what makes the pixel-exact result of Fig.~\\ref{fig:maps} possible at
an arbitrary angle rather than only on a grid."""

p = "../paper/main.tex"
s = open(p, encoding="utf-8").read()
assert OLD in s, "passage not found -- check for edits"
open(p, "w", encoding="utf-8").write(s.replace(OLD, NEW))
print("steerable passage updated to the three-scene result")
