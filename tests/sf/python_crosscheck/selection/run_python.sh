#!/bin/bash
# Steps 2 and 3 (any machine with the athena_research repo and CuPy, e.g. della-vis2):
# exact Python sums on the snapshot, then the acceptance test.
# usage: run_python.sh OUTDIR [NAME]   (OUTDIR as written by run_athena.sh, possibly
#        copied: needs OUTDIR/athena/bin/sel.hydro_w.00001.bin and OUTDIR/athena/sf_*/)
# Results go to OUTDIR/NAME/ (default NAME=python): python.log, python_exact.npz,
# compare.txt. An existing OUTDIR/NAME is refused: a rerun uses a new NAME.
# env: PYTHON (default: the plot_env_py311_test env);
#      ATHENA_RESEARCH (athena_research checkout; PYREPO is accepted as an alias);
#      the reference is pinned by content: structure_functions.py and the modules it
#      and the reader use must have the SHA-256 hashes della validated (GitLab main has
#      them; any branch with identical files is accepted). The commit is recorded.
set -u
[ $# -ge 1 ] || { echo "usage: $0 OUTDIR [NAME]"; exit 2; }
OUT=$(cd "$1" && pwd)
RES=$OUT/${2:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-/scratch/gpfs/QUATAERT/rmohapatra/miniforge3/envs/plot_env_py311_test/bin/python3}
[ -x "$PY" ] || PY=python3
REPO=${ATHENA_RESEARCH:-${PYREPO:-/home/rm3395/analysis_RadShock/athena-research-rm}}
fail() { echo "SELECTION CROSSCHECK: FAIL ($*)"; exit 1; }
[ -e "$RES" ] && fail "$RES exists; use a new NAME"
head=$(git -C "$REPO" rev-parse HEAD 2>&1)
while read -r h f; do
  echo "$h  $REPO/$f" | sha256sum --quiet -c - > /dev/null 2>&1 ||
    fail "athena_research $f differs from the validated reference (sha256 $h)"
done <<'EOT'
504dea213010907eb8ba9ec130aecb246931a3750634aca5d7e50efa4e739b48 athena_research/operations/structure_functions.py
6a8c6ac63b7e75930a860d2dbc740bf02c614c4c5d9f70e77db64bfcd9388cbb athena_research/core/athena_data.py
ffcea5185d8fabedee80349499ece7884110ca06a1a7a3c8e636eaa5ef1aa6f1 athena_research/core/io_utils.py
f907b9c76de484c5337dd8e7f07634847203157134f1e0c30b1692df155b0364 athena_research/core/base.py
513d5f2e1e7fcc7abe4cdaa2f7eb72cc5f93c5559ddb1aa2016b1e20d7be7062 athena_research/core/utils.py
ed8b7b1c7227f9908a538eb1ee0ff16316bce1570d4b9a46d7b81691f44f82a4 athena_research/utils/meshblock_utils.py
9e7b4716b8e32058f275720f52b87a88c01247fb3f421906dc1c68c50a47cf3a athena_research/utils/batch_processing.py
a854cacb62193f5adbf8c563109c44c0f41c8bc7be2118b5c596a4b59b5e145d athena_research/core/data_functions.py
EOT
mkdir -p "$RES"
{ echo "athena_research=$REPO commit=$head"
  sha256sum "$REPO/athena_research/operations/structure_functions.py"
  "$PY" -c 'import sys, numpy, scipy; print(sys.executable, "numpy", numpy.__version__, "scipy", scipy.__version__)'
} > "$RES/reference.txt" 2>&1
t0=$SECONDS
"$PY" "$HERE/python_sel.py" "$OUT/athena/bin/sel.hydro_w.00001.bin" --pyrepo "$REPO" \
  --out "$RES/python_exact.npz" > "$RES/python.log" 2>&1
rc=$?
grep -v '^Warning\|^EOS\|^Using temperature' "$RES/python.log"
echo "python: exit $rc, $((SECONDS - t0)) s"
[ $rc -eq 0 ] || fail "python step"
"$PY" "$HERE/compare.py" "$OUT/athena" "$RES/python_exact.npz" > "$RES/compare.txt" 2>&1
rc=$?
cat "$RES/compare.txt"
exit $rc
