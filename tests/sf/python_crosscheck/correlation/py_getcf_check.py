"""Run the production Python get_cf (athena_research) on the sf_unit_test fields of an
AthenaK correlation output and compare its exhaustive sums with the AthenaK file and the
exact reference. usage: py_getcf_check.py SFFILE REFSCRIPT
(ATHENA_RESEARCH = athena-research-rm checkout)"""
import os
import sys
import importlib.util
import numpy as np
sys.path.insert(0, os.environ.get("ATHENA_RESEARCH",
                                  "/home/rm3395/analysis_RadShock/athena-research-rm"))
from athena_research.operations.correlation_functions import get_cf  # noqa: E402
spec = importlib.util.spec_from_file_location("ref", sys.argv[2])
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

meta, s = ref.read_sf(sys.argv[1])
meta["test_mask"] = "0"
pos, block, fields = ref.build_fields(meta, 1.0)
wts = ref.region_weights(meta, pos, fields)
nx = [int(v) for v in meta["mesh_nx"].split(",")]
nbx = [int(v) for v in meta["meshblock_nx"].split(",")]
xmin = [float(v) for v in meta["mesh_min"].split(",")]
L = [float(v) for v in meta["domain_length"].split(",")]
nb = [nx[d] // nbx[d] for d in range(3)]
nmb = nb[0] * nb[1] * nb[2]
names = meta["variables"].split(",")
A, B = names[0], names[-1]
arrs = {"A": fields[A], "B": fields[B]}
for iw, w in enumerate(wts):
    arrs["w%d" % iw] = w


def to_blocks(f):   # global (k, j, i) raveled -> (nmb, nz, ny, nx) in block order
    g = f.reshape(nx[2], nx[1], nx[0])
    out = np.empty((nmb, nbx[2], nbx[1], nbx[0]))
    geom = np.empty((nmb, 6))
    for bk in range(nb[2]):
        for bj in range(nb[1]):
            for bi in range(nb[0]):
                m = (bk * nb[1] + bj) * nb[0] + bi
                out[m] = g[bk*nbx[2]:(bk+1)*nbx[2], bj*nbx[1]:(bj+1)*nbx[1],
                           bi*nbx[0]:(bi+1)*nbx[0]]
                lo = [xmin[0] + bi*nbx[0]*L[0]/nx[0], xmin[1] + bj*nbx[1]*L[1]/nx[1],
                      xmin[2] + bk*nbx[2]*L[2]/nx[2]]
                geom[m] = [lo[0], lo[0] + nbx[0]*L[0]/nx[0],
                           lo[1], lo[1] + nbx[1]*L[1]/nx[1],
                           lo[2], lo[2] + nbx[2]*L[2]/nx[2]]
    return out, geom


blocks = {}
for k, v in arrs.items():
    blocks[k], geom = to_blocks(v)


class AD:
    n_mbs = nmb
    nx1, nx2, nx3 = nbx
    mb_geometry = geom
    x1min, x2min, x3min = xmin
    x1max, x2max, x3max = [xmin[d] + L[d] for d in range(3)]
    _header = {"mesh": {"ix%d_bc" % d: "periodic" for d in (1, 2, 3)}}

    def data(self, name, lo, hi):
        return blocks[name][lo:hi]


nbin = int(meta["nbin"])
P = int(meta["max_order"])
weights = ["w%d" % i for i in range(len(wts))]
res = get_cf(AD(), "A", "B" if A != B else None, weights=weights,
             npairs=float((nbx[0]*nbx[1]*nbx[2]*nmb)**2), nbins=nbin, max_order=P,
             log_bin_flag=meta["logscale"] == "1",
             nsamples_block_min=nbx[0]*nbx[1]*nbx[2],
             simultaneous_blocks=nmb)
edge_err = np.max(np.abs(res["edges"] - s["rbins"]) / s["rbins"].max())
print(f"edges max rel diff {edge_err:.1e}")
# exact reference
q = np.stack([fields[v] for v in names])
per = [True] * 3
Tx, Wx, Cx = ref.run_exhaustive(s["rbins"], pos, block, q, wts, per, L, "correlation", P)
Tx, Wx, Cx = Tx.sum(0), Wx.sum(0), Cx.sum(0)


def rel(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


print(f"Python cf vs reference T: {rel(res['cf'], Tx[:, 0]):.2e}")
print(f"Python weight_sum_bin vs reference W: {rel(res['weight_sum_bin'], Wx):.2e}")
print(f"Python num_bin_points/2 vs reference C: {rel(res['num_bin_points']/2, Cx):.2e}")
# AthenaK T/W (ratio of the summed numerator and denominator) against Python, with a
# delete-one-replicate jackknife keeping each replicate's (T, W) together; absolute
# errors are reported because the correlation can cross zero
R = s["T_er"].shape[1]
Tr = s["T_er"].sum(0)[:, :, 0]                       # (R, nw, P, nb)
Wr = s["W_er"].sum(0)[:, :, None, :]
with np.errstate(invalid="ignore", divide="ignore"):
    py = res["cf"] / res["weight_sum_bin"][:, None, :]
    ak = s["T"][:, 0] / s["W"][:, None, :]
    jk = (Tr.sum(0) - Tr) / (Wr.sum(0) - Wr)
    se = np.sqrt((R - 1) / R * ((jk - jk.mean(0)) ** 2).sum(0))
good = np.isfinite(py) & np.isfinite(ak) & (se > 0)
z = (ak[good] - py[good]) / se[good]
print(f"AthenaK T/W vs Python cf/weight_sum_bin (jackknife): "
      f"max|z| {np.abs(z).max():.2f}, "
      f"rms z {np.sqrt((z**2).mean()):.2f} over {good.sum()} entries "
      f"(entries share samples); "
      f"max abs err {np.abs(ak - py)[good].max():.2e}, "
      f"max |Python| {np.abs(py[good]).max():.2e}")
