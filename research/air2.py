"""AIR-PolSAR-Seg-2.0 reader: GF-3 single-look complex to a coherency matrix.

Why this scene matters here. Every rotation experiment in this paper is either
applied by us or, in the Oberpfaffenhofen pair, real but small. The reviewers
ask for a larger, unsaturated, heterogeneous test bed, and this is one: three
GF-3 quad-pol acquisitions over Beijing, Shanghai and Guangzhou, tens of
millions of labelled pixels against the 768k of Flevoland.

What is distributed. The per-region archives contain four raw products,
``{tag}_{pol}.tiff``, each a two-band int16 image holding the real and imaginary
parts of the single-look complex scattering amplitude, pixel-aligned with an
RGB-coded ground truth in ``modified/``. The ``modified`` folder also carries
amp/pha/real/imgy renderings, but those are uint8 display products and are not
used here; the int16 pair is the measurement.

What we do with it. The scattering vector is assembled in the Pauli basis,

    k = [S_hh + S_vv,  S_hh - S_vv,  2 S_hv] / sqrt(2),

with S_hv taken as the average of the HV and VH channels, which a monostatic
sensor makes equal up to noise (measured coherence 0.86 on Guangzhou). The
coherency matrix is then the boxcar average of k k^H over an L x L window,
subsampled by L so that the looks are independent.

Calibration, stated rather than hidden. The archive ships no calibration
constants, and the channels are demonstrably not balanced. Reciprocity makes
S_hv = S_vh an identity for a monostatic sensor, yet the measured ratio is 1.90
in power on Guangzhou, and the measured phase between the two channels is
-139.9 deg on Guangzhou, +3.9 deg on Shanghai and +179.9 deg on Beijing, while
their coherence is 0.87 to 0.96. Two channels that coherent cannot differ
physically; the difference is instrumental. We correct it below using
reciprocity itself, which needs no external reference.

The co-polar imbalance between HH and VV cannot be fixed this way -- it needs
corner reflectors or a distributed-target model we do not have -- so it remains.
The consequence is that absolute polarimetric quantities on this scene, the
physical orientation angle among them, are not trustworthy, and we do not use
the scene for any such claim; Oberpfaffenhofen's acquisition pair carries that
part of the argument. What the scene is used for is unaffected: the group action
T -> U T U^T is applied by us to whatever Hermitian T we hold, and equivariance
under it is a property of the network, not of the radiometry.
"""
import numpy as np, os, glob, tifffile as tf

ROOT = ("c:/Users/musa.peker/Desktop/CV-MsAtViT-main/Datasets/"
        "AIR-PolSAR-Seg-2.0-New")

# RGB -> class index. The dataset paper names six categories and the palette
# below carries all six; white is the unlabelled border. Magenta appears only on
# Beijing, where it covers 7.34 per cent of the scene in one contiguous band, and
# an earlier version of this reader omitted it, which silently discarded a whole
# class from that region. Guangzhou and Shanghai contain five of the six.
PALETTE = {
    (255, 255, 0): 1,     # yellow
    (0, 255, 0): 2,       # green
    (0, 255, 255): 3,     # cyan
    (0, 0, 255): 4,       # blue
    (255, 0, 0): 5,       # red
    (255, 0, 255): 6,     # magenta -- Beijing only
    (255, 255, 255): 0,   # unlabelled
    (0, 0, 0): 0,
}
REGIONS = {"bj": "Beijing", "sh": "Shanghai", "gz": "Guangzhou"}


def region_dir(tag):
    hits = [d for d in glob.glob(os.path.join(ROOT, "*", "*"))
            if os.path.isdir(d) and "__MACOSX" not in d
            and os.path.exists(os.path.join(d, "%s_hh.tiff" % tag))]
    if not hits:
        raise FileNotFoundError("no directory holding %s_hh.tiff under %s"
                                % (tag, ROOT))
    return hits[0]


def _slc(path, sl=None):
    """One polarisation as complex64. The two int16 bands are real and imaginary."""
    a = tf.imread(path)
    if sl is not None:
        a = a[sl]
    return a[..., 0].astype(np.float32) + 1j * a[..., 1].astype(np.float32)


def load_gt(tag, sl=None, modified=True):
    d = region_dir(tag)
    pat = "*_gt_modify.tiff" if modified else "*_gt.tiff"
    f = sorted(glob.glob(os.path.join(d, "modified", pat)))
    if not f:
        f = sorted(glob.glob(os.path.join(d, "modified", "*_gt.tiff")))
    a = tf.imread(f[0])
    if sl is not None:
        a = a[sl]
    out = np.zeros(a.shape[:2], np.int64)
    for rgb, k in PALETTE.items():
        if k:
            out[(a[..., 0] == rgb[0]) & (a[..., 1] == rgb[1])
                & (a[..., 2] == rgb[2])] = k
    return out


def load_T3(tag, look=5, sl=None, verbose=True):
    """Coherency matrix and label map, both subsampled by `look`.

    Returns X (R, C, 6) complex64 in the same six-component convention as the
    other scenes, and gt (R, C) int64 with 0 for unlabelled.
    """
    d = region_dir(tag)
    S = {}
    for pol in ("hh", "hv", "vh", "vv"):
        S[pol] = _slc(os.path.join(d, "%s_%s.tiff" % (tag, pol)), sl)
    if verbose:
        p = {k: float(np.mean(np.abs(v) ** 2)) for k, v in S.items()}
        ref = p["hh"]
        print("  channel powers (relative to HH): "
              + ", ".join("%s %.3f" % (k.upper(), v / ref) for k, v in p.items()))

    # Cross-polar channel balancing, from reciprocity. A monostatic sensor has
    # S_hv = S_vh exactly, so any measured difference is instrumental. The
    # complex ratio c = <S_vh S_hv*> / <|S_hv|^2> is the relative gain and phase
    # of the VH channel; dividing it out puts both channels on one scale before
    # they are averaged. Without this the two are combined incoherently, and on
    # Beijing they are almost exactly antiphase, so the average would cancel.
    num = np.vdot(S["hv"].ravel(), S["vh"].ravel())
    den = np.vdot(S["hv"].ravel(), S["hv"].ravel())
    c = num / den
    S["vh"] = S["vh"] / c
    if verbose:
        print("  reciprocity: VH/HV gain %.3f, phase %+.1f deg -- corrected"
              % (abs(c), np.degrees(np.angle(c))))
    shv = 0.5 * (S["hv"] + S["vh"])
    r2 = np.float32(np.sqrt(2.0))
    k1 = (S["hh"] + S["vv"]) / r2
    k2 = (S["hh"] - S["vv"]) / r2
    k3 = r2 * shv
    del S

    def ml(a):
        """Boxcar average over look x look, subsampled by look."""
        R, C = a.shape
        R -= R % look; C -= C % look
        return a[:R, :C].reshape(R // look, look, C // look, look).mean((1, 3))

    X = np.empty(ml(k1.real).shape + (6,), np.complex64)
    X[..., 0] = ml(np.abs(k1) ** 2)
    X[..., 1] = ml(np.abs(k2) ** 2)
    X[..., 2] = ml(np.abs(k3) ** 2)
    X[..., 3] = ml((k1 * np.conj(k2)).real) + 1j * ml((k1 * np.conj(k2)).imag)
    X[..., 4] = ml((k1 * np.conj(k3)).real) + 1j * ml((k1 * np.conj(k3)).imag)
    X[..., 5] = ml((k2 * np.conj(k3)).real) + 1j * ml((k2 * np.conj(k3)).imag)
    del k1, k2, k3

    g = load_gt(tag, sl)
    R, C = X.shape[:2]
    g = g[:R * look, :C * look].reshape(R, look, C, look)
    # a looked pixel keeps its label only if the window is pure, so that the
    # label still describes the multi-looked measurement
    first = g[:, 0, :, 0]
    pure = (g == first[:, None, :, None]).all((1, 3))
    gt = np.where(pure, first, 0).astype(np.int64)
    if verbose:
        n = int((gt > 0).sum())
        print("  %s: %d x %d after %dx%d looks, %d labelled (%.1f %%), classes %s"
              % (REGIONS[tag], R, C, look, look, n, 100 * n / gt.size,
                 sorted(set(np.unique(gt).tolist()) - {0})))
    return X, gt
