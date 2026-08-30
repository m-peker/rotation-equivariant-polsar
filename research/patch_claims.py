"""Bring the accuracy claims in line with what three seeds actually support.

Two statements over-reached. The text said the equivariant network exceeds the
augmented baseline "on every scene", but on San Francisco the difference
(+0.09 pp) is inside the seed spread. And it read a trend into the steerable
penalty across three scenes when only one of the three differences is larger
than its interval.
"""
p = "../paper/main.tex"
s = open(p, encoding="utf-8").read()

OLD1 = """Table~\\ref{tab:main} gives the main comparison. The equivariant network exceeds
the rotation-augmented baseline on every scene---by \\num{2.53}, \\num{0.14} and
\\SI{0.43}{pp}---while using \\SI{58}{\\percent} of its parameters. The margin is
consistently and considerably larger at class boundaries: \\num{+4.22},
\\num{+3.25} and \\SI{+2.28}{pp}. This is the expected signature. Augmentation
buys approximate invariance by averaging over orientations during training,
which blurs precisely the pixels whose neighbourhood is not homogeneous;
architectural invariance costs nothing there."""

NEW1 = """Table~\\ref{tab:main} gives the main comparison. Against the
rotation-augmented baseline the equivariant network gains \\num{+2.48},
\\num{+0.09} and \\SI{+0.47}{pp} on the three scenes while using
\\SI{58}{\\percent} of its parameters. We treat a difference as meaningful only
when it exceeds twice the standard error of the two three-seed means; on that
test the Flevoland and Oberpfaffenhofen gains are meaningful and the San
Francisco one is not, so the honest statement is that the equivariant network
is better on two of the three scenes and indistinguishable on the third.

The margin at class boundaries is larger and clears the same test on all three:
\\num{+4.32}, \\num{+2.46} and \\SI{+1.88}{pp}. This is the expected signature.
Augmentation buys approximate invariance by averaging over orientations during
training, which blurs precisely the pixels whose neighbourhood is not
homogeneous; architectural invariance costs nothing there."""

OLD2 = """Third, the two proposed variants differ from each other, and the difference is
scene-dependent in a way worth stating precisely. Relative to the discrete
network, the steerable network scores \\num{-2.22}, \\num{-0.10} and
\\SI{+0.11}{pp} on Flevoland, San Francisco and Oberpfaffenhofen, with boundary
margins of \\num{-4.50}, \\num{-1.93} and \\SI{+0.83}{pp}. Exact equivariance for
\\emph{all} angles is therefore not free, but neither is it uniformly costly: the
penalty is confined to Flevoland and vanishes---indeed reverses---on the other
two. Those scenes carry 15, 5 and 3 classes respectively and the ordering
follows that, which suggests the cost of the stricter constraint grows with the
number of classes the representation must separate. With three scenes we offer
that as an observation rather than a law."""

NEW2 = """Third, the two proposed variants differ, but less consistently than a first
reading of the numbers suggests. Relative to the discrete network the steerable
network scores \\num{-2.22}, \\num{-0.10} and \\SI{+0.11}{pp} on the three scenes.
Only the first of those exceeds its interval: on Flevoland exact equivariance
for \\emph{all} angles costs a real \\SI{2.2}{pp}, and on San Francisco and
Oberpfaffenhofen the two variants are indistinguishable at three seeds. It is
tempting to read the ordering against the class counts of the three scenes
(15, 5 and 3), but with one measurable difference and two nulls we have no
basis for that and do not claim it."""

for old, new, tag in [(OLD1, NEW1, "comparison"), (OLD2, NEW2, "steerable")]:
    assert old in s, "passage not found: " + tag
    s = s.replace(old, new)

# note the significance convention once, in the setup section
OLD3 = """All results are
means over two seeds."""
NEW3 = """All results are
means over three seeds, and we report the seed-to-seed standard deviation with
the overall accuracy. Differences are called meaningful only when they exceed
twice the standard error of the two means being compared; run-to-run variation
with a fixed seed is \\SI{0.03}{pp} on this hardware and is therefore not the
limiting factor."""
if OLD3 in s:
    s = s.replace(OLD3, NEW3)
else:
    print("  note: seed sentence not found, add the significance convention by hand")

open(p, "w", encoding="utf-8").write(s)
print("claims brought in line with the three-seed intervals")
