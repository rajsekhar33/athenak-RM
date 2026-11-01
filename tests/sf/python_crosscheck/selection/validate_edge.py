#!/usr/bin/env python3
"""One-off validation of python_sel.edge_contributions (the numpy pair sums compare.py
subtracts for on-edge cells): on a small density band, treat every selected cell as
"on-edge", so edge_contributions must reproduce the full Python exhaustive sums
(get_sf_mb and get_sf_helmholtz_mb with the same 0/1 mask, sparse_data=True).
Needs CuPy for speed. Usage: python3 validate_edge.py SNAPSHOT.bin LO HI [PYREPO]
Result 2026-10-27 (np2 snapshot, band [1.07, 1.2], 4400 cells, 9677800 pairs): counts
identical, scalar sums max rel 4.5e-14, longitudinal 1.2e-14, transverse 6.9e-15.
"""
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4 else
                '/home/rm3395/analysis_RadShock/athena-research-rm')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from athena_research.core.athena_data import AthenaData  # noqa: E402
from athena_research.core.base import xp, asnumpy  # noqa: E402
import athena_research.operations.structure_functions as S  # noqa: E402
import python_sel  # noqa: E402

ad = AthenaData()
ad.load(sys.argv[1])
lo, hi = float(sys.argv[2]), float(sys.argv[3])


def mask(self, mbl=None, mbh=None):
    d = self.data('dens', mbl, mbh)
    return xp.where((d >= lo) & (d <= hi), 1.0, 0.0)


ad.data_func['m'] = mask
dens = asnumpy(ad.data('dens'))
sel = (dens >= lo) & (dens <= hi)
n = int(sel.sum())
npairs = float(ad.nx1 * ad.nx2 * ad.nx3 * ad.n_mbs) ** 2
_, sf, ws, nb = S.get_sf_mb(ad, ['velx', 'vely', 'velz'], weights=['m'], max_order=3,
                            npairs=npairs, nbins=24, sparse_data=True,
                            use_exhaustive=True)
_, c, s, ws2, nb2 = S.get_sf_helmholtz_mb(ad, 'vel', weights=['m'], max_order=3,
                                          npairs=npairs, nbins=24, sparse_data=True)
g = ad.mb_geometry[0, :]
L = [ad.x1max - ad.x1min, ad.x2max - ad.x2min, ad.x3max - ad.x3min]
rb, _ = S.radial_bin_edges((g[1] - g[0]) / ad.nx1,
                           np.sqrt(0.25 * sum(x * x for x in L)), 24, True)
e = python_sel.edge_contributions(ad, asnumpy, sel, sel, asnumpy(rb), L, 3)
print('cells', n, 'pairs', n * (n - 1) // 2, int(asnumpy(nb).sum()),
      int(e['edge_C'].sum()))
print('counts identical', np.array_equal(asnumpy(nb).ravel(), e['edge_C'].ravel()),
      np.array_equal(asnumpy(nb2).ravel(), e['edge_C'].ravel()))
with np.errstate(invalid='ignore', divide='ignore'):
    f = lambda A, B: np.nanmax(np.abs(A / B - 1))  # noqa: E731
    print('scalar max rel', f(e['edge_scalar_sf'][0], asnumpy(sf)[0]))
    print('long max rel', f(e['edge_longtrans_sf'][0, 0], asnumpy(c)[0]),
          'trans max rel', f(e['edge_longtrans_sf'][0, 1], asnumpy(s)[0]))
