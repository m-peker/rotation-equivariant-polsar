# Rotation-Equivariant Complex-Valued Networks for Polarimetric SAR Classification

Code and measurements for a study of one symmetry that PolSAR data provably
carries and that complex-valued classifiers ignore: rotation of the target about the radar
line of sight, which acts on the coherency matrix as the unitary conjugation

```
T  ->  U(t) T U(t)^T,     U(t) = [[1,       0,      0],
                                  [0,  cos 2t, sin 2t],
                                  [0, -sin 2t, cos 2t]]
```

**What the paper reports.** A representative complex-valued CNN loses 53 pp of
overall accuracy on Flevoland under a 45 degree rotation and relabels 58 % of the
scene; a published multiscale attention transformer loses 51 pp; the same
collapse appears on Gaofen-3 imagery from a different sensor. Working in the
circular-harmonic basis of that group gives a network that is exactly equivariant
at every angle, verified to 1e-14 in float64, whose classification map is bitwise
identical under rotation.

The input normalisation used throughout this literature does **not** commute with
the group. An architecturally equivariant network placed behind it loses 35 pp at
45 degrees while its unrotated accuracy is unchanged, so the failure is silent.
That is probably the most directly useful result here.

The paper then accounts for what invariance costs by each available route
(augmentation, orientation angle compensation, architecture), and measures two
properties of the standard evaluation protocol: on Flevoland, 98.0 % of test
pixels have patches overlapping a training patch under the class-balanced budget
and 99.6 % under the one-per-cent split common in this literature, and removing
that overlap grows the seed-to-seed spread by up to 25 times.

## Layout

```
research/
  polsar_lib.py      complex-valued layers, closed-form rotation of T
  eqnorm.py          equivariance-preserving normalisation
  equivariant.py     discrete group-convolutional network, FFT-diagonalised
  steerable.py       steerable network in the circular-harmonic basis
  steerable6.py      the same, with Clebsch-Gordan products to k_max = 6, 8
  cvmsatvit.py       reimplementation of CV-MsAtViT, used as a baseline
  baselines.py       CV-MLP, CV-2DCNN, CV-3DCNN, CV-2D-3D, CV-ViT
  disjoint.py        block-and-buffer partition, with overlap verification
  air2.py            AIR-PolSAR-Seg-2.0 reader: GF-3 SLC to coherency matrix
  exp*.py            the experiments; each writes exp*_results.json
  make_*.py fig*.py  generators for every table and figure in the paper
  audit.py           re-derives every quantitative claim in the manuscript
```

The manuscript itself is not in this repository. It is under review, and the
LaTeX sources, the generated tables and the built PDFs are kept locally until it
is published; they will be added here at that point. What is public is
everything needed to reproduce the measurements the manuscript reports.

## Reproducing

Python 3.9, PyTorch 2.7 with CUDA. The experiments were run on a single
RTX 3070 Ti (8 GB). Every experiment is resumable: interrupt it, re-run the same
command, and it continues from its stored results.

```
python exp15_table.py      # main rotation sweep, three scenes
python exp27_oac.py        # orientation angle compensation as a baseline
python exp34_air.py        # AIR-PolSAR-Seg-2.0
python audit.py            # check the manuscript against the measurements
```

**Data** is third-party and not redistributed here.

| scene | source |
| --- | --- |
| Flevoland, San Francisco, Oberpfaffenhofen | the usual public PolSAR distributions, as `T3` / `T6` ENVI products |
| AIR-PolSAR-Seg-2.0 | Journal of Radars data portal; registration required |

Put them under `Datasets/` and adjust the paths at the top of
`research/polsar_data.py` and `research/air2.py`. `research/air_cache.py` builds
the coherency matrices for the Gaofen-3 scenes once and caches them.

## The numbers

No number in the manuscript is typed by hand. Each table and figure is emitted by
a generator that reads the stored `exp*_results.json`, and `audit.py` parses the
numbers back out of the manuscript and compares them against those files. It
reads the asserted values *from the manuscript* rather than from a hard-coded
list, so it cannot drift with the thing it is supposed to check. (`audit.py`
therefore needs the manuscript sources, which are not public yet; the result
files it checks against are.)

This is not decoration. It caught a stale figure whose generator had been failing
silently for weeks, and three tables that disagreed with each other about the same
quantity after one scene was re-run at ten seeds.


## Attribution

`research/cvmsatvit.py` is our PyTorch reimplementation of CV-MsAtViT
(M. Q. Alkhatib, *Int. J. Appl. Earth Obs. Geoinf.*, vol. 137, art. 104412, 2025),
written so that it could be compared under the same pipeline as everything else.
It follows the published description including the split real/imaginary
attention. The original authors' code is theirs and is not redistributed here;
it is at <https://github.com/mqalkhatib/CV-MsAtViT>.

The irreducible decomposition these networks are built on is classical
polarimetry: that every entry of the coherency matrix is sinusoidal in the
rotation angle with a definite angular frequency is the uniform rotation theory of
Chen, Wang and Sato, *IEEE TGRS* 52(8), 2014, and the weight-4 phase is what makes
the orientation angle estimator of Lee and Ainsworth, *IEEE TGRS* 49(1), 2011,
work. The contribution here is the use of that decomposition as a steerable basis
for a network, the normalisation result, and the numerical verification.
