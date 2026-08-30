"""Insert the ablation section into the manuscript."""

SEC = r"""\subsection{Ablation}
\label{sec:ablation}
Table~\ref{tab:ablation} varies the components of the proposed network on
Flevoland.

\emph{Orientations.} Sampling the group more finely does not pay: $N{=}8$ and
$N{=}16$ are indistinguishable (\num{97.38} against \SI{97.17}{\percent}) while
$N{=}16$ costs six times the training time. $N{=}4$ is already close. This is
consistent with the steerable result---once the representation is right, the
sampling rate is not where the accuracy is.

\emph{Readout.} On the group grid the two readouts agree. Away from it they do
not: at \ang{10} the maximum-magnitude readout gives \SI{96.59}{\percent}
against \SI{97.40}{\percent} for the Fourier magnitudes. The maximum is a
discontinuous function of the group axis---its $\arg\max$ jumps between
near-tied orientations---while the Fourier magnitudes are continuous, and the
\SI{0.81}{pp} gap is that discontinuity showing up as accuracy. It is the same
effect visible as a $3.6\times10^{-2}$ against $3.8\times10^{-3}$ deviation in
Fig.~\ref{fig:eqv}.

\emph{Normalisation.} This is the ablation that matters, and it is decisive. We
replace the irreducible normalisation with per-channel standardisation, adding
the log-power channel to the latter so that both carry seven channels and the
comparison isolates the commutation property rather than the channel count.
Unrotated accuracy is essentially unchanged (\num{97.07} against
\SI{97.31}{\percent}, within one standard error). Under rotation the network
collapses: \SI{62.18}{\percent} at \ang{45}, a loss of \SI{35.1}{pp}, from an
architecture that is equivariant by construction.

The practical reading is uncomfortable and worth stating plainly. An equivariant
architecture assembled behind a normalisation that does not commute with the
group is \emph{not} equivariant, and nothing in its unrotated accuracy reveals
the fact. The only way to notice is to test under rotation, which is not
currently done. We made this mistake ourselves in an earlier version of this
work and found it only by measuring the deviation directly.

\emph{Label budget.} Accuracy rises from \SI{83.29}{\percent} at 10 labels per
class to \SI{98.84}{\percent} at 300, and invariance holds exactly at every
budget. The last figure is worth noting against the claim in
Section~\ref{sec:compare} that we do not beat the strongest baseline: at the
budget used throughout this paper we do not, but the gap is a property of that
budget rather than of the method, and at 300 labels per class the equivariant
network reaches the range reported in the literature while remaining invariant.

\input{tables/tab_ablation}

\subsection{Patch size, and a question the standard protocol cannot answer}
\label{sec:patch}
The $15\times15$ patch is close to a convention in this literature. Under the
standard random split our own measurements appear to support it and to suggest
going further: accuracy rises monotonically from \SI{90.52}{\percent} at
$7\times7$ to \SI{97.96}{\percent} at $19\times19$. But the fraction of test
pixels whose patch overlaps a training patch rises along with it, from
\SI{32.2}{\percent} to \SI{88.0}{\percent} (Table~\ref{tab:patch}), because a
larger patch reaches further into the training set. The two effects are not
separable under that protocol, so the monotone curve cannot be read as evidence
about spatial context.

The block-and-buffer partition scales its buffer as $(W{+}1)/2$ and therefore
holds the overlap at zero for every patch size, which makes the comparison
clean. There the ordering does not survive: $19\times19$ falls below
$15\times15$, and the whole range spans \SI{4.83}{pp} against a seed-to-seed
deviation of about \SI{9.6}{pp} on this scene. We therefore cannot conclude that
larger patches help---and neither can the standard protocol, which returns a
confident-looking answer that its own leakage explains.

\input{tables/tab_patch}
"""

p = "../paper/main.tex"
s = open(p, encoding="utf-8").read()
anchor = r"\subsection{What the standard protocol measures}"
assert anchor in s, "anchor not found"
s = s.replace(anchor, SEC + "\n" + anchor)
open(p, "w", encoding="utf-8").write(s)
print("ablation and patch-size sections inserted")
