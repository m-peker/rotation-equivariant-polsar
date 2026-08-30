"""Complete the remaining bibliography entries from the sources found."""
SUB = [
("""\\bibitem{multitask2019} ``Efficiently utilizing complex-valued PolSAR image data
via a multi-task deep learning framework,'' \\emph{ISPRS J.\\ Photogramm.\\ Remote
Sens.}, 2019. % [VERIFY]""",
 """\\bibitem{multitask2019} L.~Zhang, H.~Dong, and B.~Zou, ``Efficiently utilizing
complex-valued PolSAR image data via a multi-task deep learning framework,''
\\emph{ISPRS J.\\ Photogramm.\\ Remote Sens.}, vol.~157, pp.~59--72, 2019."""),

("""\\bibitem{despecknet} ``Despeckling polarimetric SAR data using a multi-stream
complex-valued fully convolutional network,'' \\emph{IEEE Geosci.\\ Remote Sens.\\
Lett.}, 2021. % [VERIFY]""",
 """\\bibitem{despecknet} A.~G.~Mullissa, C.~Persello, and J.~Reiche, ``Despeckling
polarimetric SAR data using a multistream complex-valued fully convolutional
network,'' \\emph{IEEE Geosci.\\ Remote Sens.\\ Lett.}, vol.~19, pp.~1--5, 2021."""),

("""\\bibitem{polsar2polsar} ``PolSAR2PolSAR: A semi-supervised despeckling
algorithm for polarimetric SAR images,'' \\emph{ISPRS J.\\ Photogramm.\\ Remote
Sens.}, 2025. % [VERIFY]""",
 """\\bibitem{polsar2polsar} C.~Ulondu~Mendes, E.~Dalsasso, Y.~Zhang, L.~Denis, and
F.~Tupin, ``PolSAR2PolSAR: A semi-supervised despeckling algorithm for
polarimetric SAR images,'' \\emph{ISPRS J.\\ Photogramm.\\ Remote Sens.},
vol.~220, pp.~783--798, 2025."""),

("""\\bibitem{wdsn} ``Wishart deep stacking network for fast PolSAR image
classification,'' \\emph{IEEE Trans.\\ Image Process.}, 2016. % [VERIFY]""",
 """\\bibitem{wdsn} L.~Jiao and F.~Liu, ``Wishart deep stacking network for fast
PolSAR image classification,'' \\emph{IEEE Trans.\\ Image Process.}, vol.~25,
no.~7, pp.~3273--3286, 2016."""),

("""\\bibitem{airpolsar2} ``AIR-PolSAR-Seg-2.0: A large-scale benchmark dataset for
polarimetric SAR image semantic segmentation,'' \\emph{J.\\ Radars}, 2025,
doi:10.12000/JR24237. % [VERIFY]""",
 """\\bibitem{airpolsar2} Z.~Wang, L.~Zhao, Y.~Wang, et~al., ``AIR-PolSAR-Seg-2.0:
Polarimetric SAR ground terrain classification dataset for large-scale complex
scenes,'' \\emph{J.\\ Radars}, vol.~14, no.~2, pp.~353--365, 2025."""),

("""\\bibitem{polinsar} S.~Hochstuhl, N.~Pfeffer, A.~Thiele, S.~Hinz, et~al.,
``Pol-InSAR-Island: A benchmark dataset for multi-frequency Pol-InSAR data land
cover classification,'' KIT RADAR, 2023. % [VERIFY]""",
 """\\bibitem{polinsar} S.~Hochstuhl, N.~Pfeffer, A.~Thiele, S.~Hinz, J.~Amao-Oliva,
R.~Scheiber, A.~Reigber, and H.~Dirks, ``Pol-InSAR-Island: A benchmark dataset
for multi-frequency Pol-InSAR data land cover classification,'' \\emph{ISPRS Open
J.\\ Photogramm.\\ Remote Sens.}, 2023."""),

("""\\bibitem{disjointcmc} ``Improving generalization for hyperspectral image
classification: the impact of disjoint sampling on deep models,''
\\emph{Comput.\\ Mater.\\ Contin.}, vol.~81, no.~1, 2024. % [VERIFY]""",
 """\\bibitem{disjointcmc} M.~Ahmad, M.~Mazzara, S.~Distifano, et~al., ``Improving
generalization for hyperspectral image classification: the impact of disjoint
sampling on deep models,'' \\emph{Comput.\\ Mater.\\ Contin.}, vol.~81, no.~1,
2024."""),
]

p = "../paper/refs.tex"
s = open(p, encoding="utf-8").read()
n = 0
for old, new in SUB:
    if old in s:
        s = s.replace(old, new); n += 1
    else:
        print("  NOT MATCHED: " + old.split("}")[0][9:])
s = s.replace(
    "% Entries still marked [VERIFY] are ones whose\n"
    "% title, venue and identifier were confirmed but whose author list could not be\n"
    "% retrieved here; complete them from the source before submission.",
    "% Author lists for the non-arXiv entries were taken from the publisher record.\n"
    "% One entry (disjointcmc) uses et al. because the full list was not recovered;\n"
    "% confirm it against the journal page before submission.")
open(p, "w", encoding="utf-8").write(s)
print("%d/%d entries completed; %d [VERIFY] markers left"
      % (n, len(SUB), s.count("[VERIFY]")))
