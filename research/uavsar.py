"""UAVSAR PolSAR reader, and the terrain model for the orientation angle.

Why this exists. The weakest point of the paper is that our rotation experiments
apply U(theta) ourselves, and the orientation measurement on the three classic
scenes (Exp. 20) found no systematic class-level variation to be exposed to.
UAVSAR closes that gap because it ships, coregistered with the polarimetric
data, exactly the two quantities the physics needs: terrain slope and local
incidence angle. So we can predict the orientation angle from the terrain and
compare it against the angle measured from the polarimetry -- no labels needed.

Products used (all little-endian, geocoded, mutually coregistered):
  .grd   six cross products  HHHH, HVHV, VVVV (real4) and HHHV, HHVV, HVVV
         (complex8) -- these are the covariance matrix C3 in the lexicographic
         basis, which we rotate into the Pauli basis to get T3
  .slope two float32 per pixel: slope east, slope north
  .inc   float32, local incidence angle in radians
  .ann   ASCII key/value header with the array dimensions

Terrain model (Lee et al.): an azimuth slope tilts the polarisation basis by

    tan(theta) = tan(omega) / (sin(phi) - tan(gamma) cos(phi))

with omega the slope in the azimuth direction, gamma the slope in ground range
and phi the incidence angle. East/north slopes are rotated into azimuth/range
using the aircraft heading, which UAVSAR encodes in the file name.
"""
import numpy as np, os, re


# --------------------------------------------------------------------------
def read_ann(path):
    """Parse the ASCII annotation file into a dict of floats/strings."""
    out = {}
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.split(";")[0]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = re.sub(r"\s*\(.*?\)\s*", "", k).strip()
            v = v.strip()
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def grd_shape(ann):
    """Rows and columns of the ground-projected grid."""
    for kr, kc in [("grd_pwr.set_rows", "grd_pwr.set_cols"),
                   ("grd_mag.set_rows", "grd_mag.set_cols")]:
        if kr in ann and kc in ann:
            return int(ann[kr]), int(ann[kc])
    raise KeyError("grid size not found in annotation; keys present: "
                   + ", ".join(sorted(k for k in ann if "set_rows" in k)))


def heading_from_name(name):
    """Aircraft heading in degrees, read from the UAVSAR file name.

    Convention: site (6 chars) then heading (3 digits) + counter (2 digits),
    e.g. Dthvly_34501_08038_006_080731_L090HH_XX_01.grd -> 345 degrees.
    """
    m = re.search(r"_(\d{3})\d{2}_", os.path.basename(name))
    if not m:
        raise ValueError("cannot read heading from " + name)
    return float(m.group(1))


# --------------------------------------------------------------------------
def load_grd_T3(stem, rows, cols):
    """Read the six GRD cross products and return T3 as six components.

    The GRD products are the lexicographic covariance C3 with
        k_L = [S_hh, sqrt(2) S_hv, S_vv]
    and the Pauli vector is k_P = U k_L with

        U = 1/sqrt(2) * [[1, 0, 1], [1, 0, -1], [0, sqrt(2), 0]]

    so T3 = U C3 U^H. Expanding that on the six independent entries avoids
    building a 3x3 matrix per pixel.
    """
    def rd(suffix, dtype):
        f = "%s%s.grd" % (stem, suffix)
        a = np.fromfile(f, dtype=dtype)
        if a.size != rows * cols:
            raise ValueError("%s has %d values, expected %d"
                             % (os.path.basename(f), a.size, rows * cols))
        return a.reshape(rows, cols)

    hhhh = rd("HHHH", "<f4").astype(np.float64)
    hvhv = rd("HVHV", "<f4").astype(np.float64)      # = 2|S_hv|^2
    vvvv = rd("VVVV", "<f4").astype(np.float64)
    hhhv = rd("HHHV", "<c8").astype(np.complex128)
    hhvv = rd("HHVV", "<c8").astype(np.complex128)
    hvvv = rd("HVVV", "<c8").astype(np.complex128)

    # C3 entries: C11=|Shh|^2, C22=2|Shv|^2, C33=|Svv|^2,
    #             C12=sqrt2 Shh Shv*, C13=Shh Svv*, C23=sqrt2 Shv Svv*
    s2 = np.sqrt(2.0)
    C11, C22, C33 = hhhh, hvhv, vvvv
    C12, C13, C23 = s2 * hhhv, hhvv, s2 * hvvv

    T11 = 0.5 * (C11 + C33 + 2 * C13.real)
    T22 = 0.5 * (C11 + C33 - 2 * C13.real)
    T33 = C22
    T12 = 0.5 * (C11 - C33) - 1j * C13.imag
    T13 = (C12 + np.conj(C23)) / s2
    T23 = (C12 - np.conj(C23)) / s2

    X = np.empty((rows, cols, 6), np.complex64)
    X[..., 0] = T11; X[..., 1] = T22; X[..., 2] = T33
    X[..., 3] = T12; X[..., 4] = T13; X[..., 5] = T23
    return X


def load_slope(path, rows, cols):
    """East and north slope, two interleaved float32 per pixel."""
    a = np.fromfile(path, dtype="<f4")
    if a.size != rows * cols * 2:
        raise ValueError("slope file has %d values, expected %d"
                         % (a.size, rows * cols * 2))
    a = a.reshape(rows, cols, 2)
    return a[..., 0].astype(np.float64), a[..., 1].astype(np.float64)


def load_inc(path, rows, cols):
    """Local incidence angle, radians."""
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(rows, cols).astype(np.float64)


# --------------------------------------------------------------------------
def poa_from_terrain(slope_e, slope_n, inc, heading_deg):
    """Orientation angle predicted by the terrain, radians.

    East/north slopes are first rotated into the radar frame. Azimuth points
    along the heading; ground range is 90 degrees clockwise from it.
    """
    h = np.deg2rad(heading_deg)
    tan_omega = slope_e * np.sin(h) + slope_n * np.cos(h)      # azimuth slope
    tan_gamma = slope_e * np.cos(h) - slope_n * np.sin(h)      # range slope
    denom = np.sin(inc) - tan_gamma * np.cos(inc)
    return np.arctan2(tan_omega, denom)


def poa_from_polarimetry(X):
    """Orientation angle measured from the weight-4 component, radians.

    z_c = (T22 - T33)/2 + i Re T23  transforms as e^{-4 i theta} z_c, so the
    estimator is -arg(z_c)/4. This is the same quantity used in Exp. 20; the
    decomposition supplies it directly.
    """
    z = (X[..., 1].real - X[..., 2].real) / 2 + 1j * X[..., 5].real
    return -np.angle(z) / 4.0, np.abs(z)


def wrap_to(a, period):
    """Wrap angles into (-period/2, period/2]."""
    return (a + period / 2) % period - period / 2
