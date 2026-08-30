"""Run the two result-producing sweeps back to back, in a fixed order.

Both are resumable, so this can be stopped and restarted at any point. It exists
so that the whole table set is reproduced by one command after a change to the
pipeline, which is what forced this re-run: the evaluation subsample used to
depend on call order.
"""
import subprocess, sys, time
PY = sys.executable
for script, log in [("exp15_table.py", "exp15.log"),
                    ("exp19_compare.py", "exp19.log")]:
    t0 = time.time()
    print("=== %s ===" % script, flush=True)
    with open(log, "w") as fh:
        r = subprocess.run([PY, script], stdout=fh, stderr=subprocess.STDOUT)
    print("    exit %d after %.0f s" % (r.returncode, time.time() - t0), flush=True)
    if r.returncode:
        sys.exit("failed: see " + log)
print("both sweeps complete", flush=True)
