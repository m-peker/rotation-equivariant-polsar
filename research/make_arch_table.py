"""Layer tables for the two proposed networks, generated from the modules.

A reviewer asked, correctly, that the architecture be reconstructible from the
paper. Writing the table by hand would let it drift from the code the moment a
width changes, so it is read off the instantiated modules instead: every channel
count and kernel size below is the one the measured networks actually used.
"""
import sys, torch, json
sys.path.insert(0, ".")
from steerable import SteerNet
from equivariant import EqCVCNN_F

NCL = 15          # Flevoland; only the final linear layer depends on this
W = 15            # patch size


def n(m):
    return sum(p.numel() for p in m.parameters())


net = SteerNet(NCL)
c0 = net.lift.p0.out_channels
c = net.lift.p2.r.out_channels
k = net.l1.k0.kernel_size[0]
head_in = net.head[2].in_features
hid = net.head[2].out_features

# The stage and operation columns are kept short deliberately: at \footnotesize
# this table has to fit one IEEE column, and longer wording pushed it 39 pt into
# the margin.
rows = [
    ("Input", "six complex components", "$6$", "$15^2$", "--"),
    ("Decomposition", "to $w_0$, $z_2$, $z_4$", "$4$, $2$, $1$", "$15^2$", "0"),
    ("Lift, $1\\times1$", "within class", "%d, %d, %d" % (c0, c, c), "$15^2$",
     "%d" % n(net.lift)),
    ("Block 1", "CG, $%d\\times%d$, gate" % (k, k),
     "%d, %d, %d" % (c0, c, c), "$15^2$", "%d" % n(net.l1)),
    ("Block 2", "CG, $%d\\times%d$, gate" % (k, k),
     "%d, %d, %d" % (c0, c, c), "$15^2$", "%d" % n(net.l2)),
    ("Avg.\\ pool", "per field", "--", "$7^2$", "0"),
    ("Block 3", "CG, $%d\\times%d$, gate" % (k, k),
     "%d, %d, %d" % (c0, c, c), "$7^2$", "%d" % n(net.l3)),
    ("Avg.\\ pool", "per field", "--", "$3^2$", "0"),
    ("Readout", "$w_0$, $|z_2|$, $|z_4|$, $z_4\\overline{z_2^{\\,2}}$",
     "%d" % (c0 + 4 * c), "$3^2$", "0"),
    ("Dropout, linear", "ReLU", "%d" % hid, "--", "%d" % n(net.head[2])),
    ("Linear", "classifier", "$C$", "--", "%d" % n(net.head[4])),
]

L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize",
     r"\setlength{\tabcolsep}{3.5pt}",
     r"\caption{THE STEERABLE NETWORK, LAYER BY LAYER. CHANNEL COUNTS ARE GIVEN "
     r"PER WEIGHT CLASS AS $(w_0, z_2, z_4)$ AND THE SPATIAL COLUMN AS $n^2$ "
     r"FOR AN $n\times n$ MAP; THE TWO HARMONIC CLASSES CARRY "
     r"COMPLEX CHANNELS. PARAMETER COUNTS ARE FOR $C{=}15$ CLASSES "
     r"(FLEVOLAND), TOTAL " + "%d" % n(net) + r".}",
     r"\label{tab:arch}",
     r"\begin{tabular}{llccr}", r"\hline",
     r"Stage & Operation & Channels & Spatial & Params \\", r"\hline"]
for a, b, c_, d, e in rows:
    L.append("%s & %s & %s & %s & %s \\\\" % (a, b, c_, d, e))
L += [r"\hline", r"\end{tabular}", r"\end{table}"]

open("../paper/tables/tab_arch.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote paper/tables/tab_arch.tex")

d = EqCVCNN_F(NCL, cin=7, N=8)
facts = dict(steer_params=n(net), steer_c0=c0, steer_c=c, steer_k=k,
             steer_head_in=head_in, steer_hidden=hid, steer_blocks=3,
             disc_params=n(d), disc_N=8)
json.dump(facts, open("arch_facts.json", "w"), indent=1)
for kk, vv in facts.items():
    print("  %-16s %s" % (kk, vv))
