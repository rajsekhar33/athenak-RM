#!/bin/bash
# Writes a scaling input to stdout: driven turbulence (built-in pgen_name = turbulence,
# so a default AthenaK build with no PROBLEM works), with on-the-fly sf and/or spec
# outputs every DCYCLE cycles.
# usage: make_input.sh NX MB VARIANT NLIM DCYCLE
#   VARIANT: base (no sf/spec) | sf | spec | both
# env: PHYS=hydro|mhd (default hydro);
#      BUDGET=production|light (sf budgets, default production; see below);
#      SAMPLING=global|sqrtp|stress (how budgets change with NRANK, default global);
#      NRANK (needed unless SAMPLING=global); TRANSPORT=device|host (spec, default device);
#      SELECT=1 adds the RadShock-style temperature band selection to the sf outputs
# The chosen budgets go into the input; run_variants.sh records them in run_info.txt.
NX=$1; MB=$2; VAR=$3; NLIM=$4; DC=$5
PHYS=${PHYS:-hydro}
p=$PHYS
RS=hllc
if [ "$PHYS" = mhd ]; then BV="mhd_bcc1, mhd_bcc2, mhd_bcc3"; RS=hlld; fi
# sf budgets, max_order 3 and 100 log bins as the RadShock driver.
#   production: the calibrated order-3 settings of docs/structure_function_output.md
#     (high E1, balanced E2): ndraw_block 262144, anchors x samples 262144 x 262144,
#     nreplicate 8
#   light: a throughput setting, ndraw_block 16384, 262144 x 16384, nreplicate 4
# E2 pair work per rank is about A*S/(R*P) (A anchors, S samples, R replicates, P ranks):
#   global: A, S fixed -> E2 work per rank ~ 1/P (E1, selection etc. stay constant)
#   sqrtp:  A, S ~ sqrt(P/8) -> E2 work per rank about constant (gathered sample ~ sqrt P)
#   stress: A, S ~ P/8 -> E2 work per rank ~ P (an increasing-work stress test)
case ${BUDGET:-production} in
  production) ND=262144; NA=262144; NS=262144; NR=8 ;;
  light)      ND=16384;  NA=262144; NS=16384;  NR=4 ;;
  *) echo "make_input.sh: unknown BUDGET=$BUDGET" >&2; exit 1 ;;
esac
case ${SAMPLING:-global} in
  global) ;;
  sqrtp)
    : "${NRANK:?SAMPLING=sqrtp needs NRANK}"
    f=$(python3 -c "import math; print(math.sqrt($NRANK/8))")
    NA=$(python3 -c "print(max(1, round($NA*$f)))"); NS=$(python3 -c "print(max(1, round($NS*$f)))") ;;
  stress)
    : "${NRANK:?SAMPLING=stress needs NRANK}"
    NA=$((NA*NRANK/8)); NS=$((NS*NRANK/8)) ;;
  *) echo "make_input.sh: unknown SAMPLING=$SAMPLING" >&2; exit 1 ;;
esac
SEL=""
if [ "${SELECT:-0}" = 1 ]; then
  # the initial state has T = 1/gamma = 0.6; driving heats it. The band keeps most cells.
  SEL="select1_variable = temperature
select1_min    = 0.3
select1_max    = 3.0"
fi
cat <<EOT
<comment>
problem   = on-the-fly sf/spec scaling (variant $VAR)

<job>
basename  = scal

<mesh>
nghost    = 2
nx1       = $NX
x1min     = -0.5
x1max     = 0.5
ix1_bc    = periodic
ox1_bc    = periodic
nx2       = $NX
x2min     = -0.5
x2max     = 0.5
ix2_bc    = periodic
ox2_bc    = periodic
nx3       = $NX
x3min     = -0.5
x3max     = 0.5
ix3_bc    = periodic
ox3_bc    = periodic

<meshblock>
nx1       = $MB
nx2       = $MB
nx3       = $MB

<time>
evolution  = dynamic
integrator = rk2
cfl_number = 0.3
nlim       = $NLIM
tlim       = 100.0

<$p>
eos         = ideal
gamma       = 1.6666666666666667
reconstruct = plm
rsolver     = $RS

<problem>
pgen_name = turbulence

<turb_driving>
type  = $p
tcorr = 0.5
dedt  = 0.1
nlow  = 1
nhigh = 2
EOT
n=1
if [ "$VAR" = sf ] || [ "$VAR" = both ]; then
  if [ "$PHYS" = mhd ]; then
    cat <<EOT

<output$n>
file_type      = sf
id             = aniso
mode           = aniso_mhd
variables      = mhd_w_vx, mhd_w_vy, mhd_w_vz, $BV
weights        = ones, mhd_w_d
max_order      = 3
nbin           = 100
logscale       = true
ndraw_block    = $ND
nanchor_global = $NA
nsample_global = $NS
nreplicate     = $NR
timing         = true
dcycle         = $DC
EOT
    n=$((n+1))
  fi
  cat <<EOT

<output$n>
file_type      = sf
id             = scalar
mode           = scalar
variables      = ${p}_w_vx, ${p}_w_vy, ${p}_w_vz, ${p}_w_d, pressure, temperature
weights        = ones, ${p}_w_d
max_order      = 3
nbin           = 100
logscale       = true
histogram      = true
sparse         = true
ndraw_block    = $ND
nanchor_global = $NA
nsample_global = $NS
nreplicate     = $NR
timing         = true
dcycle         = $DC
$SEL

<output$((n+1))>
file_type      = sf
id             = longtrans
mode           = long_trans
variables      = ${p}_w_vx, ${p}_w_vy, ${p}_w_vz
weights        = ones
max_order      = 3
nbin           = 100
logscale       = true
histogram      = true
sparse         = true
ndraw_block    = $ND
nanchor_global = $NA
nsample_global = $NS
nreplicate     = $NR
timing         = true
dcycle         = $DC
$SEL
EOT
  n=$((n+2))
fi
if [ "$VAR" = spec ] || [ "$VAR" = both ]; then
  cat <<EOT

<output$n>
file_type  = spec
id         = scal
mode       = scalar
variables  = ${p}_w_d, ${p}_w_vx
weights    = ones, ${p}_w_d
nbin       = 256
transport  = ${TRANSPORT:-device}
dcycle     = $DC

<output$((n+1))>
file_type  = spec
id         = helm
mode       = helmholtz
variables  = ${p}_w_vx, ${p}_w_vy, ${p}_w_vz
nbin       = 200
transport  = ${TRANSPORT:-device}
dcycle     = $DC
EOT
fi
