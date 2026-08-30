"""Timestamped snapshot of everything that would be expensive to recompute.

Results live in JSON/NPY files that each experiment writes incrementally, so a
crash costs at most the arm in flight. This adds a second line of defence: a
dated copy of the result files, the figures and the manuscript, so that an
overwrite or a corrupted write is also recoverable.

Run: python snapshot.py        (idempotent, a few MB per call)
"""
import os, shutil, time, glob

STAMP = time.strftime("%Y%m%d_%H%M")
DST = os.path.join("snapshots", STAMP)
os.makedirs(DST, exist_ok=True)

PATTERNS = ["*.json", "*.log", "figs/map_*.npy",
            "paper_figs/*.pdf", "paper_figs/CAPTIONS.md",
            "../paper/main.tex", "../paper/main.pdf", "RESUME.md"]
n = 0
for pat in PATTERNS:
    for f in glob.glob(pat):
        sub = os.path.join(DST, os.path.dirname(pat).replace("..", "up"))
        os.makedirs(sub, exist_ok=True)
        shutil.copy2(f, os.path.join(sub, os.path.basename(f)))
        n += 1
size = sum(os.path.getsize(os.path.join(r, f))
           for r, _, fs in os.walk(DST) for f in fs)
print("snapshot -> %s   (%d files, %.1f MB)" % (DST, n, size / 1e6))

keep = sorted(glob.glob("snapshots/*"))
for old in keep[:-8]:                      # keep the last eight
    shutil.rmtree(old, ignore_errors=True)
    print("  pruned", old)
