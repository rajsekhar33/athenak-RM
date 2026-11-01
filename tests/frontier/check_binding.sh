#!/bin/bash
# Device placement check with the real executable and the real launcher.
# usage: check_binding.sh NRANK ATHENA OUTDIR   (inside an allocation)
# Runs NRANK ranks of ATHENA (srun -l, the same SRUN_EXTRA / ATHENA_ARGS as the tests)
# with --kokkos-print-configuration on a tiny input (time/nlim=0). Every task first
# prints its host and device visibility and then execs ATHENA, so host, visibility and
# the Kokkos selection come from the same launched process.
# Device per rank: production style (--kokkos-map-device-id-by=mpi_rank) the index Kokkos
# marks "Selected"; --gpu-bind=closest style the visible device ($VISVAR, default
# ROCR_VISIBLE_DEVICES; CUDA_VISIBLE_DEVICES on della).
# PASS needs: one Athena rank banner with NRANK ranks (an MPI build, not NRANK serial
# copies); every rank one host, one Kokkos-selected device of backend REQUIRE_BACKEND
# (default HIP; none skips) and one device id; on every host pairwise distinct devices.
# The step is bounded (BIND_TIMEOUT, default 600 s, plus a Slurm step --time).
# OUTDIR must not exist or be empty (results are never overwritten).
set -u
NR=$1; BIN=$2; D=$3
SRC=$(cd "$(dirname "$0")/../.." && pwd)
AA=${ATHENA_ARGS---kokkos-map-device-id-by=mpi_rank}
VISVAR=${VISVAR:-ROCR_VISIBLE_DEVICES}
TO=${BIND_TIMEOUT:-600}
[ -e "$D" ] && [ -n "$(ls -A "$D" 2> /dev/null)" ] &&
  { echo "BINDING: FAIL ($D exists and is not empty; use a new directory)"; exit 1; }
mkdir -p "$D"
# any input works: the run stops after set-up (nlim = 0); the default build has no test
# problem, so use the built-in turbulence input
cat > "$D/in.athinput" <<EOT
<job>
basename = bind
<mesh>
nghost = 2
nx1 = 64
x1min = 0
x1max = 1
ix1_bc = periodic
ox1_bc = periodic
nx2 = 64
x2min = 0
x2max = 1
ix2_bc = periodic
ox2_bc = periodic
nx3 = $((16*NR))
x3min = 0
x3max = 1
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 64
nx2 = 64
nx3 = 16
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.3
nlim = 0
tlim = 1
<hydro>
eos = isothermal
iso_sound_speed = 1.0
reconstruct = plm
rsolver = hlle
<problem>
pgen_name = turbulence
EOT
cd "$D"
timeout --kill-after=60 "$TO" srun -l -n "$NR" --time=$(( (TO + 59)/60 + 1 )) ${SRUN_EXTRA:-} \
  bash -c 'echo "HOST $(hostname -s) VIS=${!1:-unset}"; shift; exec "$@"' bind "$VISVAR" \
  "$BIN" $AA --kokkos-print-configuration -i in.athinput > log 2>&1 ||
  { echo "BINDING: FAIL (launch failed or timed out, exit $?; see $D/log)"; exit 1; }
python3 -I - "$NR" "$AA" "${REQUIRE_BACKEND:-HIP}" <<'PY'
import collections, re, sys
nr, aa, backend = int(sys.argv[1]), sys.argv[2], sys.argv[3]
lab = r'\s*(?:\[\d+,)?(\d+)\]?(?:<stdout>)?:\s*'
host, vis, dev, arch = {}, {}, collections.defaultdict(set), set()
banner = []
for line in open('log', errors='replace'):
    m = re.match(lab + r'HOST (\S+) VIS=(\S+)', line)
    if m:
        host[int(m.group(1))], vis[int(m.group(1))] = m.group(2), m.group(3)
    m = re.match(lab + r'Kokkos::(\w+)\[ (\d+) \]\s*(.*?)\s*: Selected', line)
    if m:
        dev[int(m.group(1))].add((m.group(2), m.group(3)))
        arch.add('%s %s' % (m.group(2), m.group(4)))
    m = re.match(lab + r'Number of parallel ranks = (\d+)\s*$', line)
    if m:
        banner.append(int(m.group(2)))
print('backend/arch: %s' % ('; '.join(sorted(arch)) or 'none reported'))
print('rank banners: %s' % banner)
problems = []
if banner != [nr]:
    problems.append('rank banner %s, expected one with %d ranks (not an MPI run?)'
                    % (banner, nr))
mapping = {}
for r in range(nr):
    d = dev.get(r, set())
    if len(d) != 1:
        problems.append('rank %d: %d Kokkos-selected devices' % (r, len(d)))
    elif backend != 'none' and next(iter(d))[0].upper() != backend.upper():
        problems.append('rank %d: backend %s, expected %s' % (r, next(iter(d))[0], backend))
    if 'map-device-id-by' in aa:
        did = sorted(x[1] for x in d)[0] if len(d) == 1 else None
    else:
        v = vis.get(r)
        did = v if v and v != 'unset' and ',' not in v else None
    mapping[r] = (host.get(r), did)
per_host = collections.defaultdict(list)
for r, (h, d) in sorted(mapping.items()):
    per_host[h].append(d)
    print('rank %d: host %s device %s visible %s' % (r, h, d, vis.get(r)))
bad = [r for r, (h, d) in mapping.items() if h is None or d is None]
dup = sorted(h for h, ds in per_host.items() if len(set(ds)) != len(ds))
if bad:
    problems.append('ranks without a host or a single device: %s' % bad)
if dup:
    problems.append('hosts with shared devices: %s' % dup)
if problems:
    print('BINDING: FAIL (%s)' % '; '.join(problems))
    sys.exit(1)
print('BINDING: PASS (%d ranks on %d hosts, distinct %s devices per host)'
      % (nr, len(per_host), backend))
PY
