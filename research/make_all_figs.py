"""Regenerate every paper figure, in dependency order.

Figures 2, 5, 6 and 7 read measured result files rather than hard-coded numbers,
so a figure can never disagree with the table it belongs to. Run this after any
experiment that changes those files.

  fig1, fig3, fig4  paper_figs.py      (scene geometry + stored measurements)
  fig2              fig2_update.py     <- exp15_results.json
  fig5              fig5_update.py     <- measures equivariance now, writes fig5_data.json
  fig6              fig6_maps.py       <- figs/map_*.npy
  fig7              fig7_protocol.py   <- exp18_results.json
"""
import subprocess, sys, os
PY = sys.executable
STEPS = [("paper_figs.py", None),
         ("fig2_update.py", "exp15_results.json"),
         ("fig5_update.py", None),
         ("fig6_maps.py", "figs/map_steerable_45.npy"),
         ("fig7_protocol.py", "exp18_results.json")]
for script, need in STEPS:
    if need and not os.path.exists(need):
        print("SKIP %-18s (missing %s)" % (script, need))
        continue
    print("RUN  %s" % script, flush=True)
    r = subprocess.run([PY, script], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        sys.exit("failed: " + script)
    for line in r.stdout.strip().splitlines()[-3:]:
        print("     " + line)
print("\nall figures regenerated")
