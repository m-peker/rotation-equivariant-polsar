"""Exp. 24 - is the rotation physically present? UAVSAR terrain test.

Exp. 20 measured the orientation angle on the three classic scenes and found no
systematic class-level variation, which bounds the paper's claim: the fragility
we measure is latent there. This closes the gap from the other side. UAVSAR
ships terrain slope and local incidence angle coregistered with the polarimetry,
so the orientation angle can be predicted from the terrain and compared against
the angle measured from the data. No labels are needed.

Two predictions, both falsifiable:
  1. the measured and predicted angles agree, over a range wide enough to matter
  2. the agreement is carried by the AZIMUTH component of the slope and not the
     range component, since only the former tilts the polarisation basis

If both hold, the rotation this paper is about is demonstrably present in real
imagery, driven by terrain, and not an artefact of our applying it ourselves.

Usage:
    python exp24_uavsar_poa.py <path to scene directory>

The directory should contain one UAVSAR ground-projected acquisition: the six
*.grd cross products, the .slope, .inc and .ann files.
"""
import numpy as np, sys, os, glob, json
sys.path.insert(0, ".")
from uavsar import (read_ann, grd_shape, heading_from_name, load_grd_T3,
                    load_slope, load_inc, poa_from_terrain,
                    poa_from_polarimetry, wrap_to)

if len(sys.argv) < 2:
    sys.exit(__doc__)
D = sys.argv[1]

ann = glob.glob(os.path.join(D, "*.ann"))
if not ann:
    sys.exit("no .ann file in " + D)
A = read_ann(ann[0])
rows, cols = grd_shape(A)
heading = heading_from_name(ann[0])
print("scene %s   %d x %d   heading %.0f deg" % (os.path.basename(ann[0]), rows, cols, heading))

hh = glob.glob(os.path.join(D, "*HHHH*.grd"))
if not hh:
    sys.exit("no HHHH grd file in " + D)
stem = hh[0][: hh[0].index("HHHH")]
X = load_grd_T3(stem, rows, cols)
se, sn = load_slope(glob.glob(os.path.join(D, "*.slope"))[0], rows, cols)
inc = load_inc(glob.glob(os.path.join(D, "*.inc"))[0], rows, cols)

meas, mag = poa_from_polarimetry(X)
pred = poa_from_terrain(se, sn, inc, heading)

span = X[..., 0].real + X[..., 1].real + X[..., 2].real
ok = np.isfinite(meas) & np.isfinite(pred) & (span > 0) & (inc > 0)
# the angle is only identifiable where the weight-4 component carries energy
ok &= mag > np.nanpercentile(mag[ok], 60)
# and the terrain model only bites where there is relief
relief = np.hypot(se, sn)
print("valid pixels %d   relief: median slope %.3f, 95th pct %.3f"
      % (ok.sum(), np.median(relief[ok]), np.percentile(relief[ok], 95)))

m = wrap_to(meas[ok], np.pi / 2)
p = wrap_to(pred[ok], np.pi / 2)
d = wrap_to(m - p, np.pi / 2)
r = np.corrcoef(m, p)[0, 1]
print()
print("  measured  sd %.2f deg" % np.degrees(m.std()))
print("  predicted sd %.2f deg" % np.degrees(p.std()))
print("  correlation measured vs predicted   r = %+.3f" % r)
print("  residual sd %.2f deg (against %.2f deg if the prediction were ignored)"
      % (np.degrees(d.std()), np.degrees(m.std())))

# prediction 2: the azimuth component should carry it, the range one should not
h = np.deg2rad(heading)
az = (se * np.sin(h) + sn * np.cos(h))[ok]
rg = (se * np.cos(h) - sn * np.sin(h))[ok]
print()
print("  corr(measured angle, azimuth slope) = %+.3f" % np.corrcoef(m, az)[0, 1])
print("  corr(measured angle, range slope)   = %+.3f" % np.corrcoef(m, rg)[0, 1])

# how much of the scene sees a shift large enough to matter
big = np.degrees(np.abs(m))
for t in (5, 10, 15):
    print("  |angle| > %2d deg on %5.1f %% of valid pixels" % (t, 100 * (big > t).mean()))

json.dump(dict(rows=rows, cols=cols, heading=heading, n=int(ok.sum()),
               r=float(r), sd_meas=float(np.degrees(m.std())),
               sd_pred=float(np.degrees(p.std())),
               sd_resid=float(np.degrees(d.std())),
               r_azimuth=float(np.corrcoef(m, az)[0, 1]),
               r_range=float(np.corrcoef(m, rg)[0, 1])),
          open("exp24_results.json", "w"), indent=1)
print("\nstored in exp24_results.json")
