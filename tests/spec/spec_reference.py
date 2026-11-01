#!/usr/bin/env python3
"""Check AthenaK on-the-fly spectra (file_type = spec) against spectra.py.
Rebuilds the fields of src/pgen/sf_unit_test.cpp from RUN_DIR/in.athinput and repeats
get_spectrum / get_spectrum_mb (weighted) / get_spectrum_helmholtz with full complex
numpy FFTs, using spectra.py's own _spectral_grid and _hamming_window and its crop rule
(core/utils.py get_refined_data). Also checks Parseval against the file's sumsq.
Usage: python3 tests/spec/spec_reference.py RUN_DIR [--athena-research PATH] [--rtol]
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sf"))
sys.path.insert(0, os.path.join(HERE, "..", "fft"))
from sf_reference import build_fields, read_header  # noqa: E402
from compare_pencil import read_mesh  # noqa: E402


def read_spec(fname):
    with open(fname, "rb") as f:
        meta = read_header(f, fname)
        payload = f.read()
    if len(payload) % 8:
        raise ValueError(f"{fname}: payload of {len(payload)} bytes is not float64")
    raw = np.frombuffer(payload, dtype=np.float64)
    nb, nv, nw = int(meta["nbin"]), int(meta["nvar"]), int(meta["nweight"])
    if meta["mode"] == "scalar":
        names = [("kbins", nb + 1), ("k", nb), ("nk", nb), ("E", nw * nv * nb),
                 ("P_DC", nw * nv), ("sumsq", nv)]
    else:
        names = [("kbins", nb + 1), ("k", nb), ("nk", nb), ("E_comp", nb), ("E_sol", nb),
                 ("P_DC", 1), ("sumsq", 1)]
    out, off = {}, 0
    for name, n in names:
        out[name] = raw[off:off + n]
        off += n
    if off != raw.size:
        raise ValueError(f"{fname}: payload has {raw.size} values, layout needs {off}")
    if meta["mode"] == "scalar":
        out["E"] = out["E"].reshape(nw, nv, nb)
        out["P_DC"] = out["P_DC"].reshape(nw, nv)
    return meta, out


def fields_from_rst(path, nx):
    """hydro_w_d and hydro_w_vx/vy/vz (z, y, x) of an isothermal-hydro restart file, as
    AthenaK's ConsToPrim forms them (vel = mom * (1/dens))."""
    sys.path.insert(0, os.path.join(HERE, ".."))
    from rst_compare import parse
    meta, arr = parse(path)
    ng, b1, b2, b3 = meta["mb_indcs"][2][:4]
    lloc = np.array(meta["lloc"][2]).reshape(-1, 4)
    n1, n2, n3 = b1 + 2 * ng, b2 + 2 * ng, b3 + 2 * ng
    act = (slice(None), slice(ng, ng + b3), slice(ng, ng + b2), slice(ng, ng + b1))
    u = [arr["hydro_u%d" % v].reshape(-1, n3, n2, n1)[act] for v in range(4)]
    di = 1.0 / u[0]
    blk = {"hydro_w_d": u[0], "hydro_w_vx": u[1] * di, "hydro_w_vy": u[2] * di,
           "hydro_w_vz": u[3] * di}
    out = {}
    for k, v in blk.items():
        g = np.empty((nx[2], nx[1], nx[0]))
        for m, (lx1, lx2, lx3, _) in enumerate(lloc):
            g[lx3 * b3:(lx3 + 1) * b3, lx2 * b2:(lx2 + 1) * b2,
              lx1 * b1:(lx1 + 1) * b1] = v[m]
        out[k] = g
    return out, meta["ncycle"][2][0]


class FakeAD:
    """The attributes of AthenaData that spectra.py's helpers read."""
    def __init__(self, nx, lo, hi):
        self.Nx1, self.Nx2, self.Nx3 = nx
        self.x1min, self.x2min, self.x3min = lo
        self.x1max, self.x2max, self.x3max = hi


def crop_range(ad, xyz):
    """Index range exactly as core/utils.py get_refined_data (int() truncation)."""
    f = [ad.Nx1 / (ad.x1max - ad.x1min), ad.Nx2 / (ad.x2max - ad.x2min),
         ad.Nx3 / (ad.x3max - ad.x3min)]
    mins = [ad.x1min, ad.x2min, ad.x3min]
    return [(int((xyz[2 * d] - mins[d]) * f[d]), int((xyz[2 * d + 1] - mins[d]) * f[d]))
            for d in range(3)]


def compressive_part(kx, ky, kz, k2, vx, vy, vz):
    """spectra.py _compressive_part with numpy (its xp may be CuPy)."""
    nonzero = k2 > 0
    inv_k2 = np.where(nonzero, 1.0 / np.where(nonzero, k2, 1.0), 0.0)
    scale = (kx * vx + ky * vy + kz * vz) * inv_k2
    return scale * kx, scale * ky, scale * kz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    # the athena-research-rm checkout (spectra.py); $ATHENA_RESEARCH overrides the default
    ap.add_argument("--athena-research",
                    default=os.environ.get(
                        "ATHENA_RESEARCH",
                        "/home/rm3395/analysis_RadShock/athena-research-rm"))
    ap.add_argument("--rtol", type=float, default=1e-10)
    # fields from an isothermal-hydro restart file of the same state instead of the
    # analytic sf_unit_test fields
    ap.add_argument("--rst", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.athena_research)
    from athena_research.operations import spectra as S

    meta0, nx, blocks = read_mesh(os.path.join(a.run, "in.athinput"))
    m = blocks["mesh"]
    lo = [float(m["x%dmin" % d]) for d in (1, 2, 3)]
    hi = [float(m["x%dmax" % d]) for d in (1, 2, 3)]
    ad = FakeAD(nx, lo, hi)
    rst_cycle = None
    if a.rst:
        fields, rst_cycle = fields_from_rst(a.rst, nx)
    else:
        noise_amp = float(blocks.get("problem", {}).get("noise", "1.0"))
        _, _, fields = build_fields(meta0, noise_amp)
        fields = {k: v.reshape(nx[2], nx[1], nx[0]) for k, v in fields.items()}

    ok = True
    files = sorted(glob.glob(os.path.join(a.run, "spec_*", "*.spec")))
    if not files:
        print("no .spec files")
        return 1
    for fname in files:
        meta, out = read_spec(fname)
        if rst_cycle is not None and int(meta["cycle"]) != rst_cycle:
            print("%s: cycle %s is not the restart cycle %d FAIL"
                  % (fname, meta["cycle"], rst_cycle))
            ok = False
            continue
        for k, v in out.items():
            if not np.all(np.isfinite(v)):
                print("%s: nonfinite values in %s FAIL" % (fname, k))
                ok = False
        strat = meta["strat"] == "1"
        skip = float(meta["skip"])
        xyz = (S._strat_xyz(ad, skip) if strat
               else [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
        (i0, i1), (j0, j1), (k0, k1) = crop_range(ad, xyz)
        crop = lambda q: q[k0:k1, j0:j1, i0:i1]  # noqa: E731
        shape = crop(fields["hydro_w_d"]).shape
        nb_in = int(meta["nbin"]) + (0 if meta["logscale"] == "1" else 1)
        kx, ky, kz, kbins, k_ = S._spectral_grid(ad, shape, nb_in,
                                                 meta["logscale"] == "1")
        kmag_corner = np.sqrt(np.max(np.abs(kx)) ** 2 + np.max(np.abs(ky)) ** 2 +
                              np.max(np.abs(kz)) ** 2)
        if meta["logscale"] == "1":
            # AthenaK keeps the final log edge inclusive (spectra.py's 10**log10(kmax)
            # can round below the corner |k|); everything else is spectra.py's grid
            kmax_ref = max(np.min((np.array(shape[::-1]) /
                                   np.array([float(v) for v in
                                             meta["length"].split(",")]))
                                  [np.array(shape[::-1]) > 1]), kmag_corner)
            kbins = kbins.copy()
            kbins[-1] = max(kbins[-1], kmax_ref)
            k_ = np.sqrt(kbins[:-1] * kbins[1:])
        kz3, ky3, kx3 = np.meshgrid(kz, ky, kx, indexing="ij")
        kk = np.sqrt(kx3 ** 2 + ky3 ** 2 + kz3 ** 2)
        win = S._hamming_window(shape[0])[:, None, None] if strat else 1.0
        N = np.prod(shape)
        nk = np.histogram(kk, bins=kbins)[0]
        # every non-DC mode must be binned: linear bins include DC, log bins exclude
        # only DC (strict sum rule; a dropped mode is a failure)
        msgs = []
        outside = (kk < kbins[0]) | (kk > kbins[-1])
        nondc_out = outside & (kk > 0)
        if np.any(nondc_out):
            msgs.append("UNBINNED non-DC modes: %d FAIL" % np.sum(nondc_out))
            ok = False

        def check(name, got, ref, scale=None):
            nonlocal ok
            scale = np.max(np.abs(ref)) if scale is None else scale
            err = np.max(np.abs(np.asarray(got) - ref)) / (scale if scale > 0 else 1.0)
            good = err <= a.rtol
            ok &= bool(good)
            msgs.append("%s %.1e%s" % (name, err, "" if good else " FAIL"))

        check("kbins", out["kbins"], kbins)
        check("k", out["k"], k_)
        check("nk", out["nk"], nk)
        if meta["mode"] == "scalar":
            vars_ = meta["variables"].split(",")
            wts = meta["weights"].split(",")
            for iv, v in enumerate(vars_):
                q = crop(fields[v]) * win
                P = np.abs(np.fft.fftn(q) / np.sqrt(N)) ** 2
                for iw, w in enumerate(wts):
                    Pw = P
                    if w != "ones":
                        Pw = P * np.abs(np.fft.fftn(crop(fields[w])) / np.sqrt(N)) ** 2
                    E = np.histogram(kk, bins=kbins, weights=Pw)[0]
                    check("E[%s,%s]" % (w, v), out["E"][iw, iv], E)
                    check("P_DC[%s,%s]" % (w, v), out["P_DC"][iw, iv], Pw[0, 0, 0],
                          scale=np.max(Pw))
                    if w == "ones":
                        dc = Pw[0, 0, 0] if meta["logscale"] == "1" else 0.0
                        check("parseval[%s]" % v, np.sum(out["E"][iw, iv]),
                              np.sum(q ** 2) - dc)
                check("sumsq[%s]" % v, out["sumsq"][iv], np.sum(q ** 2))
        else:
            comps = meta["variables"].split(",")
            F = [np.fft.fftn(crop(fields[c]) * win) / np.sqrt(N) for c in comps]
            cx, cy, cz = compressive_part(kx3, ky3, kz3, kk ** 2, *F)
            Kc = np.abs(cx) ** 2 + np.abs(cy) ** 2 + np.abs(cz) ** 2
            Ks = np.abs(F[0] - cx) ** 2 + np.abs(F[1] - cy) ** 2 + np.abs(F[2] - cz) ** 2
            check("E_comp", out["E_comp"], np.histogram(kk, bins=kbins, weights=Kc)[0])
            check("E_sol", out["E_sol"], np.histogram(kk, bins=kbins, weights=Ks)[0])
            tot = sum(np.sum((crop(fields[c]) * win) ** 2) for c in comps)
            check("sumsq", out["sumsq"][0], tot)
            check("P_DC", out["P_DC"][0], (Kc + Ks)[0, 0, 0], scale=np.max(Kc + Ks))
            par = tot - ((Kc + Ks)[0, 0, 0] if meta["logscale"] == "1" else 0.0)
            check("parseval", np.sum(out["E_comp"]) + np.sum(out["E_sol"]), par)
        print("%s shape(z,y,x)=%s: %s" % (os.path.relpath(fname, a.run), shape,
                                          ", ".join(msgs)))
    print("SPEC REFERENCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
