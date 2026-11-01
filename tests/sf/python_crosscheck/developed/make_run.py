#!/usr/bin/env python3
"""Set up the developed-turbulence comparison run from an existing restart file.

usage: make_run.py SOURCE.rst RUNDIR [NCYC]

Writes RUNDIR/start.rst: SOURCE.rst with its parameter text replaced. Every non-output
block is kept verbatim (mesh, hydro, turbulence driving with the defaults recorded at run
time), <job> basename becomes 'dev', <time> runs NCYC (default 5) more cycles, and every
output block is replaced by OUTPUTS below, all written at the final cycle: a restart
file (the full-precision state for the references), a bin snapshot (its metadata for
athena_research), the two production sf outputs and the two production spec outputs.
The binary part after <par_end> is copied unchanged (AthenaK reads it from the file
position after <par_end>). RUNDIR/in.athinput holds the same parameter text.
"""
import os
import re
import sys

OUTPUTS = """<output1>
file_type = rst
dcycle    = {n}

<output2>
file_type = bin
variable  = hydro_w
dcycle    = {n}

<output3>
file_type      = sf
id             = scalar
mode           = scalar
variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz, hydro_w_d
weights        = ones, hydro_w_d
max_order      = 3
nbin           = 100
logscale       = true
histogram      = true
sparse         = true
ndraw_block    = 262144
nanchor_global = 262144
nsample_global = 262144
nreplicate     = 8
seed           = 7
timing         = true
dcycle         = {n}

<output4>
file_type      = sf
id             = longtrans
mode           = long_trans
variables      = hydro_w_vx, hydro_w_vy, hydro_w_vz
weights        = ones
max_order      = 3
nbin           = 100
logscale       = true
histogram      = true
sparse         = true
ndraw_block    = 262144
nanchor_global = 262144
nsample_global = 262144
nreplicate     = 8
seed           = 7
timing         = true
dcycle         = {n}

<output5>
file_type = spec
id        = scal
mode      = scalar
variables = hydro_w_d, hydro_w_vx
weights   = ones, hydro_w_d
nbin      = 256
dcycle    = {n}

<output6>
file_type = spec
id        = helm
mode      = helmholtz
variables = hydro_w_vx, hydro_w_vy, hydro_w_vz
nbin      = 200
dcycle    = {n}
"""


def main():
    src, run = sys.argv[1], sys.argv[2]
    ncyc = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    raw = open(src, "rb").read()
    end = raw.index(b"<par_end>")
    end = raw.index(b"\n", end) + 1
    text = raw[:end].decode("ascii")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
    from rst_compare import parse   # noqa: E402
    meta, _ = parse(src)
    ncycle0 = meta["ncycle"][2][0]
    blocks = re.split(r"(?m)^(?=<)", text[:text.index("<par_end>")])
    kept = []
    for b in blocks:
        name = b.split(">", 1)[0].lstrip("<") if b.startswith("<") else ""
        if name.startswith("output"):
            continue
        if name == "job":
            b = re.sub(r"(?m)^(basename\s*=\s*)\S+", r"\g<1>dev", b)
        if name == "time":
            b = re.sub(r"(?m)^(nlim\s*=\s*)\S+", r"\g<1>%d" % (ncycle0 + ncyc), b)
            b = re.sub(r"(?m)^(tlim\s*=\s*)\S+", r"\g<1>1.0e3", b)
        kept.append(b)
    # a cadence beyond the run: AthenaK writes every output once, at the final cycle
    par = "".join(kept).rstrip("\n") + "\n\n" + OUTPUTS.format(n=1000000)
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "in.athinput"), "w") as f:
        f.write(par)
    with open(os.path.join(run, "start.rst"), "wb") as f:
        f.write(par.encode("ascii") + b"<par_end>\n")
        f.write(raw[end:])
    print(f"{run}/start.rst: cycle {ncycle0} -> {ncycle0 + ncyc}, "
          f"time {meta['time'][2][0]}")


if __name__ == "__main__":
    main()
