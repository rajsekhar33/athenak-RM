//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file structure_function.cpp
//  \brief on-the-fly structure functions (file_type = sf). See structure_function.hpp
//  and docs/structure_function_output.md for the estimator contract.
//
//  Stored quantities (unordered-pair convention, as in structure_functions.py):
//    T[p,b] = 1/2 sum_{ordered valid pairs in bin b} |w2*q2 - w1*q1|^p
//    W[b]   = 1/2 sum (w1 + w2)       (Python weight_sum_bin)
//    C[b]   = 1/2 sum 1               (pair count; mean S_p = T/C)
//    n[b]   = raw accepted evaluations (Python num_bin_points)
//  T, W, C are importance-weighted estimates; each replicate estimates them for the full
//  population and the reported values are the replicate average.

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "outputs.hpp"
#include "structure_function.hpp"


namespace {

// role tags of the counter-based random streams (part of proposal_version=1)
constexpr uint64_t kStreamE1Radius = 1;
constexpr uint64_t kStreamE1Shell  = 2;
constexpr uint64_t kStreamE1Anchor = 3;
constexpr uint64_t kStreamE2Anchor = 4;
constexpr uint64_t kStreamE2Sample = 5;
constexpr uint64_t kStreamE1Mix    = 6;

//! splitmix64 finaliser (same constants as the Python/CUDA kernels)
KOKKOS_INLINE_FUNCTION
uint64_t SFMix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

//! counter-based key: no RNG state, independent of the simulation's RNGs
KOKKOS_INLINE_FUNCTION
uint64_t SFKey(uint64_t seed, uint64_t outnum, uint64_t stream, uint64_t owner,
               uint64_t rep, uint64_t draw) {
  uint64_t h = SFMix(seed);
  h = SFMix(h ^ outnum);
  h = SFMix(h ^ stream);
  h = SFMix(h ^ owner);
  h = SFMix(h ^ rep);
  return SFMix(h ^ draw);
}

KOKKOS_INLINE_FUNCTION
double SFUniform(uint64_t h) {
  return static_cast<double>(h >> 11) * (1.0/9007199254740992.0);  // [0,1)
}

//! np.searchsorted(edges, r, side='right') - 1; -1 or nbin when outside the edges
KOKKOS_INLINE_FUNCTION
int SFBinIndex(const DvceArray1D<double> &edges, int nbin, double r) {
  if (!(r >= edges(0))) return -1;
  if (r >= edges(nbin)) return nbin;
  int lo = 0, hi = nbin;   // edges(lo) <= r < edges(hi)
  while (hi - lo > 1) {
    int mid = (lo + hi)/2;
    if (edges(mid) <= r) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

//! difference of two global cell indices stored exactly as doubles
KOKKOS_INLINE_FUNCTION
int64_t SFCellOffset(double a, double b) {
  return static_cast<int64_t>(a) - static_cast<int64_t>(b);
}

//! what every pair kernel needs to turn one pair into accumulator contributions
struct SFPairSpec {
  int nq, nord, nbin, mode, nvar;
  SFWeightMap wmap;
  // increment histograms (optional): edges row (iw*nq + iq) holds nedge[iq] edges
  bool hist;
  int ncol;                            // columns per row = max edges + 1
  int nedge[kSFMaxFields];
  DvceArray2D<double> hedges;          // (nw*nq, ncol - 1)
  // aniso_mhd: families |l| [0, nbl), |l.b| [nbl, nbl+nbp), l_perp [nbl+nbp, ...)
  int nbl, nbp, nbq;
  double cos_prll_min, cos_perp_max;
  DvceArray1D<double> eprll, eperp;
};

//! increment-histogram column (Python _delta_bin_index): number of edges <= v, so 0 is
//! the underflow column and n the overflow column
KOKKOS_INLINE_FUNCTION
int SFHistCol(const DvceArray2D<double> &e, int row, int n, double v) {
  int lo = 0, hi = n;   // answer in [lo, hi]
  while (lo < hi) {
    int mid = (lo + hi)/2;
    if (e(row, mid) <= v) {lo = mid + 1;} else {hi = mid;}
  }
  return lo;
}

//! sink writing straight into the global accumulators of slot er = est*R + rep
struct SFGlobalSink {
  DvceArray1D<double> T, W, C;
  DvceArray1D<uint64_t> n;
  int64_t er;
  int nw, nq, nord, nbin;
  DvceArray1D<double> H;   // increment histograms, (er, w, q, bin, col)
  int ncol;
  KOKKOS_INLINE_FUNCTION
  void AddH(int iw, int iq, int b, int col, double v) const {
    Kokkos::atomic_add(&H(((((er*nw + iw)*nq + iq)*nbin + b)*ncol + col)), v);
  }
  KOKKOS_INLINE_FUNCTION
  void AddT(int iw, int iq, int p, int b, double v) const {
    Kokkos::atomic_add(&T(((((er*nw + iw)*nq + iq)*nord + p)*nbin + b)), v);
  }
  KOKKOS_INLINE_FUNCTION
  void AddWCN(int iw, int b, double w, double c) const {
    const int64_t i = (er*nw + iw)*nbin + b;
    Kokkos::atomic_add(&W(i), w);
    Kokkos::atomic_add(&C(i), c);
    Kokkos::atomic_add(&n(i), static_cast<uint64_t>(1));
  }
};

//! sink into a team-scratch histogram holding weights [w0, w0 + nwl) of one slot; its
//! layout is the slot's global layout restricted to those weights, so a flush is a
//! contiguous offset (see SFFlushScratch)
struct SFScratchSink {
  ScrArray1D<double> T, W, C;
  ScrArray1D<uint64_t> n;
  int w0, nq, nord, nbin;
  SFGlobalSink g;          // histograms are too large for scratch: global atomics
  KOKKOS_INLINE_FUNCTION
  void AddH(int iw, int iq, int b, int col, double v) const {
    g.AddH(iw, iq, b, col, v);
  }
  KOKKOS_INLINE_FUNCTION
  void AddT(int iw, int iq, int p, int b, double v) const {
    Kokkos::atomic_add(&T(((((iw - w0)*nq + iq)*nord + p)*nbin + b)), v);
  }
  KOKKOS_INLINE_FUNCTION
  void AddWCN(int iw, int b, double w, double c) const {
    const int i = (iw - w0)*nbin + b;
    Kokkos::atomic_add(&W(i), w);
    Kokkos::atomic_add(&C(i), c);
    Kokkos::atomic_add(&n(i), static_cast<uint64_t>(1));
  }
};

//! aniso_mhd contributions of one pair (structure_function_aniso_mhd_kernel): b = unit
//! vector of the unweighted pair-mean field (B1 + B2)/2, pair skipped if |mean B| <
//! 1e-30. Increments |dv|, |dv_perp|, |dB|, |dB_perp| of the weighted fields (perp to
//! b), binned by |l| (lbin, the caller's bin, may be nbl = beyond the last edge), by
//! |l.b| when cos(theta) > cos_prll_min, and by l_perp when cos(theta) <= cos_perp_max.
template <class Sink>
KOKKOS_INLINE_FUNCTION
void SFAddPairAniso(const Sink &sink, const SFPairSpec &s, int w0, int w1, int lbin,
                    double fac, const double *v1, const double *v2, double dx, double dy,
                    double dz) {
  double bx = 0.5*(v1[3] + v2[3]), by = 0.5*(v1[4] + v2[4]), bz = 0.5*(v1[5] + v2[5]);
  const double bmag = Kokkos::sqrt(bx*bx + by*by + bz*bz);
  if (bmag < 1e-30) return;
  bx /= bmag;
  by /= bmag;
  bz /= bmag;
  const double l_dot_b = dx*bx + dy*by + dz*bz;
  const double l_sq = dx*dx + dy*dy + dz*dz;
  const double l_mag = Kokkos::sqrt(l_sq);
  if (l_mag < 1e-30) return;
  const double l_prll = Kokkos::fabs(l_dot_b);
  const double l_perp = Kokkos::sqrt(Kokkos::fmax(0.0, l_sq - l_dot_b*l_dot_b));
  const double cth = Kokkos::fmin(1.0, Kokkos::fmax(0.0, l_prll/l_mag));
  int hit[3], nh = 0;
  if (lbin >= 0 && lbin < s.nbl) hit[nh++] = lbin;
  if (cth > s.cos_prll_min) {
    const int ib = SFBinIndex(s.eprll, s.nbp, l_prll);
    if (ib >= 0 && ib < s.nbp) hit[nh++] = s.nbl + ib;
  }
  if (cth <= s.cos_perp_max) {
    const int ib = SFBinIndex(s.eperp, s.nbq, l_perp);
    if (ib >= 0 && ib < s.nbq) hit[nh++] = s.nbl + s.nbp + ib;
  }
  if (nh == 0) return;
  for (int iw = w0; iw < w1; ++iw) {
    const int wf = s.wmap.field[iw];
    const double a1 = (wf < 0) ? 1.0 : v1[wf];
    const double a2 = (wf < 0) ? 1.0 : v2[wf];
    if (!(a1 > 0.0) || !(a2 > 0.0)) continue;
    double inc[4];
    for (int f = 0; f < 2; ++f) {   // f = 0: velocity, 1: magnetic field
      const double ux = a2*v2[3*f] - a1*v1[3*f];
      const double uy = a2*v2[3*f + 1] - a1*v1[3*f + 1];
      const double uz = a2*v2[3*f + 2] - a1*v1[3*f + 2];
      const double usq = ux*ux + uy*uy + uz*uz;
      const double ub = ux*bx + uy*by + uz*bz;
      inc[2*f] = Kokkos::sqrt(usq);
      inc[2*f + 1] = Kokkos::sqrt(Kokkos::fmax(0.0, usq - ub*ub));
    }
    for (int h = 0; h < nh; ++h) {
      sink.AddWCN(iw, hit[h], fac*(a1 + a2), fac);
      for (int iq = 0; iq < 4; ++iq) {
        double pw = inc[iq];
        for (int p = 0; p < s.nord; ++p) {
          sink.AddT(iw, iq, p, hit[h], fac*pw);
          pw *= inc[iq];
        }
      }
    }
  }
}

//! contributions of one sampled pair with importance factor fac, for weights [w0, w1).
//! v1, v2: all field values (variables first, then non-unit weights) at the endpoints;
//! (dx, dy, dz) = x2 - x1 with |.| = r, in separation bin 'bin'.
template <class Sink>
KOKKOS_INLINE_FUNCTION
void SFAddPair(const Sink &sink, const SFPairSpec &s, int w0, int w1, int bin,
               double fac, const double *v1, const double *v2, double dx, double dy,
               double dz, double r) {
  if (s.mode == 2) {
    SFAddPairAniso(sink, s, w0, w1, bin, fac, v1, v2, dx, dy, dz);
    return;
  }
  if (s.mode == 3) {
    // correlation (Python get_cf): raw products of the unweighted fields A = v[0] and
    // B = v[nvar-1], both orientations, pair weight w1*w2 entering once; W holds
    // 2*w1*w2, so T/W is Python's cf/weight_sum_bin
    const double ab = v1[0]*v2[s.nvar - 1];
    const double ba = v1[s.nvar - 1]*v2[0];
    for (int iw = w0; iw < w1; ++iw) {
      const int wf = s.wmap.field[iw];
      const double a1 = (wf < 0) ? 1.0 : v1[wf];
      const double a2 = (wf < 0) ? 1.0 : v2[wf];
      if (!(a1 > 0.0) || !(a2 > 0.0)) continue;
      const double pwt = fac*a1*a2;
      sink.AddWCN(iw, bin, 2.0*pwt, fac);
      double pab = ab, pba = ba;
      for (int p = 0; p < s.nord; ++p) {
        sink.AddT(iw, 0, p, bin, pwt*(pab + pba));
        pab *= ab;
        pba *= ba;
      }
    }
    return;
  }
  for (int iw = w0; iw < w1; ++iw) {
    const int wf = s.wmap.field[iw];
    const double a1 = (wf < 0) ? 1.0 : v1[wf];
    const double a2 = (wf < 0) ? 1.0 : v2[wf];
    if (!(a1 > 0.0) || !(a2 > 0.0)) continue;
    sink.AddWCN(iw, bin, fac*(a1 + a2), fac);
    double ux = 0.0, uy = 0.0, uz = 0.0, ul = 0.0;
    if (s.mode == 1) {
      ux = a2*v2[0] - a1*v1[0];
      uy = a2*v2[1] - a1*v1[1];
      uz = a2*v2[2] - a1*v1[2];
      ul = (ux*dx + uy*dy + uz*dz)/r;
    }
    for (int iq = 0; iq < s.nq; ++iq) {
      double incr;
      if (s.mode == 0) {
        incr = Kokkos::fabs(a2*v2[iq] - a1*v1[iq]);
      } else {
        incr = (iq == 0) ? Kokkos::fabs(ul) :
               Kokkos::sqrt(Kokkos::fmax(0.0, ux*ux + uy*uy + uz*uz - ul*ul));
      }
      double pw = incr;
      for (int p = 0; p < s.nord; ++p) {
        sink.AddT(iw, iq, p, bin, fac*pw);
        pw *= incr;
      }
      if (s.hist) {
        // the longitudinal histogram uses the signed dv.l/|l| (swap invariant)
        const double hv = (s.mode == 1 && iq == 0) ? ul : incr;
        const int row = iw*s.nq + iq;
        sink.AddH(iw, iq, bin, SFHistCol(s.hedges, row, s.nedge[iq], hv), fac);
      }
    }
  }
}

//! team-scratch histogram for weights [w0, w0 + nwl) of one slot
struct SFScratchHist {
  int nwl, nT, nW;   // weights held, T entries, W/C/n entries
  KOKKOS_INLINE_FUNCTION
  SFScratchHist(int nwl_, const SFPairSpec &s) :
      nwl(nwl_), nT(nwl_*s.nq*s.nord*s.nbin), nW(nwl_*s.nbin) {}
  std::size_t Bytes() const {
    return 3*ScrArray1D<double>::shmem_size(1) +   // alignment slack
           ScrArray1D<double>::shmem_size(nT) + 2*ScrArray1D<double>::shmem_size(nW) +
           ScrArray1D<uint64_t>::shmem_size(nW);
  }
};

//! allocate and zero the scratch histogram of a team; returns a sink into it
KOKKOS_INLINE_FUNCTION
SFScratchSink SFMakeScratchSink(const TeamMember_t &member, const SFScratchHist &h,
                                const SFPairSpec &s, int w0, const SFGlobalSink &g) {
  SFScratchSink k;
  k.g = g;
  k.T = ScrArray1D<double>(member.team_scratch(0), h.nT);
  k.W = ScrArray1D<double>(member.team_scratch(0), h.nW);
  k.C = ScrArray1D<double>(member.team_scratch(0), h.nW);
  k.n = ScrArray1D<uint64_t>(member.team_scratch(0), h.nW);
  k.w0 = w0;
  k.nq = s.nq;
  k.nord = s.nord;
  k.nbin = s.nbin;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, h.nT), [&](const int i) {
    k.T(i) = 0.0;
  });
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, h.nW), [&](const int i) {
    k.W(i) = 0.0;
    k.C(i) = 0.0;
    k.n(i) = 0;
  });
  member.team_barrier();
  return k;
}

//! add a team's scratch histogram (weights [w0, w0 + nwl) of slot er) to the global
//! accumulators; only touched entries cost a global atomic
KOKKOS_INLINE_FUNCTION
void SFFlushScratch(const TeamMember_t &member, const SFScratchSink &k,
                    const SFScratchHist &h, const SFGlobalSink &g) {
  member.team_barrier();
  const int64_t oT = (g.er*g.nw + k.w0)*static_cast<int64_t>(g.nq*g.nord*g.nbin);
  const int64_t oW = (g.er*g.nw + k.w0)*static_cast<int64_t>(g.nbin);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, h.nT), [&](const int i) {
    if (k.T(i) != 0.0) Kokkos::atomic_add(&g.T(oT + i), k.T(i));
  });
  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, h.nW), [&](const int i) {
    if (k.n(i) != 0) {
      Kokkos::atomic_add(&g.W(oW + i), k.W(i));
      Kokkos::atomic_add(&g.C(oW + i), k.C(i));
      Kokkos::atomic_add(&g.n(oW + i), k.n(i));
    }
  });
}

//! local MeshBlock m and 0-based block-interior cell (i, j, k) of region-local linear
//! index u (block-major over the local MeshBlocks' region boxes, x fastest)
KOKKOS_INLINE_FUNCTION
void SFDecodeRegion(const DvceArray1D<int64_t> &boxoff, const DvceArray2D<int> &box,
                    int nmb, int64_t u, int &m, int &i, int &j, int &k) {
  int lo = 0, hi = nmb - 1;   // block m with boxoff(m) <= u < boxoff(m+1)
  while (lo < hi) {
    int mid = (lo + hi + 1)/2;
    if (boxoff(mid) <= u) {lo = mid;} else {hi = mid - 1;}
  }
  m = lo;
  const int64_t loc = u - boxoff(m);
  const int bx = box(m, 1) - box(m, 0) + 1, by = box(m, 3) - box(m, 2) + 1;
  i = box(m, 0) + static_cast<int>(loc % bx);
  j = box(m, 2) + static_cast<int>((loc/bx) % by);
  k = box(m, 4) + static_cast<int>(loc/(static_cast<int64_t>(bx)*by));
}

//! E1 draw d of replicate t in local MeshBlock m: Chebyshev radius c ~ 1/c, a lag l
//! uniform on that integer cube shell, then an anchor uniform among the A(l) cells whose
//! partner x+l is in the block. Returns false when the draw yields no pair in a bin;
//! otherwise the 0-based block-interior endpoints, separation, bin and the importance
//! factor (1/2) A(l)/(n1 q(l)) (unordered-pair convention).
//! sparse: with probability 1/2 the anchor is instead uniform among the block's S
//! support cells (the draw is rejected when x+l leaves the region box). A pair that
//! contributes has both endpoints in the support and its anchor admissible, so its
//! proposal probability is q(l) (1/(2A) + 1/(2S)) whichever branch drew it; factor
//! (1/2) / (n1 q(l) (1/(2A) + 1/(2S))). The admissible branch keeps the large lags
//! covered; the support branch puts the other half of the draws on valid anchors.
struct SFE1Sampler {
  DvceArray1D<double> rbins, cdf;
  DvceArray2D<int> box;
  DualArray1D<RegionSize> size;
  DualArray1D<int> gid;
  int ndim, cmax, nbin, nrep, nx1, nx2, nx3;
  bool keep_over;   // aniso_mhd: |l| beyond the last edge may still bin in l_prll/perp
  int64_t ndraw;
  double harm;
  uint64_t seed, outnum;
  SFGeom geom;
  bool sparse = false;
  DvceArray1D<int64_t> supp, soff, boxoff;

  KOKKOS_INLINE_FUNCTION
  bool Draw(int m, int t, int64_t d, int &i1, int &j1, int &k1, int &i2, int &j2,
            int &k2, double &dx, double &dy, double &dz, double &r, int &bin,
            double &fac) const {
    const int64_t n1 = ndraw/nrep + ((t < ndraw % nrep) ? 1 : 0);
    if (d >= n1) return false;
    const uint64_t owner = static_cast<uint64_t>(gid.d_view(m));

    // Chebyshev radius c in [1, cmax] with p(c) = (1/c)/harm
    const double u = SFUniform(SFKey(seed, outnum, kStreamE1Radius, owner, t, d));
    int lo = 0, hi = cmax - 1;
    while (lo < hi) {
      int mid = (lo + hi)/2;
      if (cdf(mid) > u) {hi = mid;} else {lo = mid + 1;}
    }
    const int c = lo + 1;
    const double pc = (1.0/c)/harm;

    // lag uniform on the integer shell max|l_i| = c of the active dimensions
    const uint64_t hs = SFKey(seed, outnum, kStreamE1Shell, owner, t, d);
    int lx = 0, ly = 0, lz = 0;
    double nshell;
    if (ndim == 3) {
      const int64_t a = 2*c + 1, b = 2*c - 1;
      const int64_t nz = 2*a*a, ny = 2*a*b, nxf = 2*b*b;
      nshell = static_cast<double>(nz + ny + nxf);
      int64_t s = static_cast<int64_t>(hs % static_cast<uint64_t>(nz + ny + nxf));
      if (s < nz) {
        lz = (s < a*a) ? c : -c;
        s %= a*a;
        lx = static_cast<int>(s % a) - c;
        ly = static_cast<int>(s / a) - c;
      } else if (s < nz + ny) {
        s -= nz;
        ly = (s < a*b) ? c : -c;
        s %= a*b;
        lx = static_cast<int>(s % a) - c;
        lz = static_cast<int>(s / a) - (c - 1);
      } else {
        s -= nz + ny;
        lx = (s < b*b) ? c : -c;
        s %= b*b;
        ly = static_cast<int>(s % b) - (c - 1);
        lz = static_cast<int>(s / b) - (c - 1);
      }
    } else if (ndim == 2) {
      const int64_t a = 2*c + 1, b = 2*c - 1;
      nshell = static_cast<double>(2*a + 2*b);
      int64_t s = static_cast<int64_t>(hs % static_cast<uint64_t>(2*a + 2*b));
      if (s < 2*a) {
        ly = (s < a) ? c : -c;
        lx = static_cast<int>(s % a) - c;
      } else {
        s -= 2*a;
        lx = (s < b) ? c : -c;
        ly = static_cast<int>(s % b) - (c - 1);
      }
    } else {
      nshell = 2.0;
      lx = ((hs & 1ULL) == 0ULL) ? c : -c;
    }
    const double ql = pc/nshell;

    // admissible anchors: x and x+l both inside the block
    const int ax = nx1 - Kokkos::abs(lx), ay = nx2 - Kokkos::abs(ly);
    const int az = nx3 - Kokkos::abs(lz);
    if (ax <= 0 || ay <= 0 || az <= 0) return false;
    const int64_t acount = static_cast<int64_t>(ax)*ay*az;
    const uint64_t ha = SFKey(seed, outnum, kStreamE1Anchor, owner, t, d);
    int64_t scount = 0;
    if (sparse) {
      scount = soff(m+1) - soff(m);
      if (scount <= 0) return false;
    }
    if (sparse && SFUniform(SFKey(seed, outnum, kStreamE1Mix, owner, t, d)) < 0.5) {
      // anchor uniform among the block's support cells
      const int64_t u = supp(soff(m) + static_cast<int64_t>(
                                 ha % static_cast<uint64_t>(scount)));
      const int64_t loc = u - boxoff(m);
      const int bx = box(m, 1) - box(m, 0) + 1, by = box(m, 3) - box(m, 2) + 1;
      i1 = box(m, 0) + static_cast<int>(loc % bx);
      j1 = box(m, 2) + static_cast<int>((loc/bx) % by);
      k1 = box(m, 4) + static_cast<int>(loc/(static_cast<int64_t>(bx)*by));
    } else {
      const int64_t a = static_cast<int64_t>(ha % static_cast<uint64_t>(acount));
      i1 = static_cast<int>(a % ax) + Kokkos::max(0, -lx);
      j1 = static_cast<int>((a/ax) % ay) + Kokkos::max(0, -ly);
      k1 = static_cast<int>(a/(static_cast<int64_t>(ax)*ay)) + Kokkos::max(0, -lz);
    }
    i2 = i1 + lx;
    j2 = j1 + ly;
    k2 = k1 + lz;

    // both endpoints inside the region box (which lies inside the block)
    if (i1 < box(m, 0) || i1 > box(m, 1) || i2 < box(m, 0) || i2 > box(m, 1) ||
        j1 < box(m, 2) || j1 > box(m, 3) || j2 < box(m, 2) || j2 > box(m, 3) ||
        k1 < box(m, 4) || k1 > box(m, 5) || k2 < box(m, 4) || k2 > box(m, 5)) {
      return false;
    }

    // physical separation (minimum image on periodic axes)
    r = geom.Separation(lx, ly, lz, dx, dy, dz);
    bin = SFBinIndex(rbins, nbin, r);
    if (bin < 0 || (bin >= nbin && !keep_over)) return false;
    if (sparse) {
      fac = 0.5/(static_cast<double>(n1)*ql*(0.5/static_cast<double>(acount) +
                                             0.5/static_cast<double>(scount)));
    } else {
      fac = 0.5*static_cast<double>(acount)/(static_cast<double>(n1)*ql);
    }
    return true;
  }
};

SFPairSpec SFMakeSpec(int nq, int nord, int nbin, int mode, int nvar,
                      const std::vector<int> &weight_field) {
  SFPairSpec s;
  s.nq = nq;
  s.nord = nord;
  s.nbin = nbin;
  s.mode = mode;
  s.nvar = nvar;
  for (int iw = 0; iw < kSFMaxFields; ++iw) {
    s.wmap.field[iw] = (iw < static_cast<int>(weight_field.size())) ?
                       weight_field[iw] : -1;
    s.nedge[iw] = 0;
  }
  s.hist = false;
  s.ncol = 1;
  s.nbl = nbin;
  s.nbp = s.nbq = 0;
  s.cos_prll_min = s.cos_perp_max = 0.0;
  return s;
}

//! attach the increment-histogram configuration to a pair spec
void SFSetHist(SFPairSpec *s, bool hist, int ncol, const std::vector<int> &nedge,
               const DvceArray2D<double> &edges) {
  s->hist = hist;
  s->ncol = ncol;
  for (int i = 0; i < kSFMaxFields; ++i) {
    s->nedge[i] = (i < static_cast<int>(nedge.size())) ? nedge[i] : 0;
  }
  s->hedges = edges;
}

//! attach the aniso_mhd bin families to a pair spec
void SFSetAniso(SFPairSpec *s, int nbl, int nbp, int nbq, double theta_prll_max,
                double theta_perp_min, const DvceArray1D<double> &eprll,
                const DvceArray1D<double> &eperp) {
  s->nbl = nbl;
  s->nbp = nbp;
  s->nbq = nbq;
  s->cos_prll_min = std::cos(theta_prll_max);
  s->cos_perp_max = std::cos(theta_perp_min);
  s->eprll = eprll;
  s->eperp = eperp;
}

//! numpy.logspace(a, b, n) in decades: 10**(i*step + a), last point 10**b
std::vector<double> SFLogspace(double a, double b, int n) {
  std::vector<double> v(n);
  const double step = (n > 1) ? (b - a)/(n - 1) : 0.0;
  for (int i = 0; i < n; ++i) v[i] = std::pow(10.0, (i == n - 1) ? b : i*step + a);
  return v;
}

//! pairs per team: enough teams to fill the device, and at least 32 pairs per bin so
//! the scratch zero/flush stays a small fraction of the work
int64_t SFTeamChunk(int64_t total, int nbin) {
  constexpr int64_t kTargetTeams = 2048;
  return std::max<int64_t>(32*static_cast<int64_t>(nbin),
                           (total + kTargetTeams - 1)/kTargetTeams);
}

//! fatal-error helper
void SFFatal(const std::string &block, const std::string &msg) {
  std::cout << "### FATAL ERROR in sf output block '" << block << "'" << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//! positive whole-number parameter (sampling budgets), checked before any conversion
int64_t SFGetCount(ParameterInput *pin, const std::string &blk, const std::string &key,
                   double def) {
  const double v = GetOutputReal(pin, blk, key, def);
  if (!(v >= 1.0 && v <= 9.0e15) || v != std::floor(v)) {
    SFFatal(blk, key + " must be a whole number from 1 to 9e15");
  }
  return static_cast<int64_t>(v);
}

//! largest-remainder split of 'total' over strata with sizes n[s], reserving one unit
//! for every nonempty stratum first. Deterministic, so every rank computes the same.
std::vector<int64_t> SFAllocate(int64_t total, const std::vector<int64_t> &n) {
  std::size_t ns = n.size();
  std::vector<int64_t> out(ns, 0);
  int64_t nonempty = 0;
  long double ntot = 0.0L;
  for (std::size_t s = 0; s < ns; ++s) {
    if (n[s] > 0) {out[s] = 1; nonempty++; ntot += static_cast<long double>(n[s]);}
  }
  int64_t extra = total - nonempty;
  if (extra <= 0 || ntot <= 0.0L) return out;
  std::vector<std::pair<long double, std::size_t>> rem;
  int64_t used = 0;
  for (std::size_t s = 0; s < ns; ++s) {
    if (n[s] <= 0) continue;
    long double share =
        static_cast<long double>(extra)*static_cast<long double>(n[s])/ntot;
    int64_t base = static_cast<int64_t>(std::floor(share));
    out[s] += base;
    used += base;
    rem.emplace_back(share - static_cast<long double>(base), s);
  }
  std::stable_sort(rem.begin(), rem.end(),
                   [](const std::pair<long double, std::size_t> &a,
                      const std::pair<long double, std::size_t> &b) {
                     return a.first > b.first;
                   });
  for (std::size_t i = 0; i < rem.size() && used < extra; ++i, ++used) {
    out[rem[i].second] += 1;
  }
  return out;
}

//! ASCII header lines, then "header offset=<N>" giving the start of the binary payload
bool SFWriteHeader(FILE *pfile, const std::string &meta) {
  bool ok = (std::fwrite(meta.data(), 1, meta.size(), pfile) == meta.size());
  char dummy[64];
  int line_len = std::snprintf(dummy, sizeof(dummy), "header offset=%012ld\n", 0L);
  int64_t header_offset = static_cast<int64_t>(std::ftell(pfile)) + line_len;
  ok &= (std::fprintf(pfile, "header offset=%012lld\n",
                      static_cast<long long>(header_offset)) > 0);  // NOLINT
  return ok;
}

}  // namespace

//----------------------------------------------------------------------------------------
// ctor: reads and validates the sf input block, builds sampling strata and budgets

SFOutput::SFOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
    BaseTypeOutput(pin, pm, op),
    rbins_("sf_rbins", 1), cdf_c_("sf_cdf_c", 1),
    h_box_("sf_hbox", 1, 6), box_("sf_box", 1, 6),
    box_offset_("sf_box_offset", 1), h_box_offset_("sf_hbox_offset", 1),
    acc_T_("sf_T", 1), acc_W_("sf_W", 1), acc_C_("sf_C", 1), acc_n_("sf_n", 1),
    h_T_("sf_hT", 1), h_W_("sf_hW", 1), h_C_("sf_hC", 1), h_n_("sf_hn", 1),
    time_e1_(0.0), time_e2_(0.0), time_reduce_(0.0), time_support_(0.0),
    output_count_(0) {
  const std::string &blk = out_params.block_name;

  // ---- uniform grids only (first release) ----
  if (pm->multilevel) {
    SFFatal(blk, "sf output currently supports uniform grids only (no SMR/AMR)");
  }
  for (int f = 0; f < 6; ++f) {
    if (pm->mesh_bcs[f] == BoundaryFlag::shear_periodic) {
      SFFatal(blk, "shearing-periodic boundaries are not supported by sf output");
    }
  }

  // ---- mode, variables, weights ----
  std::string mode = pin->GetOrAddString(blk, "mode", "scalar");
  if (mode.compare("scalar") == 0) {
    mode_ = 0;
  } else if (mode.compare("long_trans") == 0) {
    mode_ = 1;
  } else if (mode.compare("aniso_mhd") == 0) {
    mode_ = 2;
  } else if (mode.compare("correlation") == 0) {
    mode_ = 3;
  } else {
    SFFatal(blk, "mode='" + mode + "' must be one of scalar, long_trans, aniso_mhd, "
                 "correlation");
  }
  var_names_ = SplitOutputList(pin->GetString(blk, "variables"));
  weight_names_ = SplitOutputList(pin->GetOrAddString(blk, "weights", "ones"));
  if (weight_names_.empty()) weight_names_.push_back("ones");
  nvar_ = static_cast<int>(var_names_.size());
  nweight_ = static_cast<int>(weight_names_.size());
  if (mode_ == 1 && nvar_ != 3) {
    SFFatal(blk, "mode=long_trans needs exactly three variables (vector components "
                 "along x1, x2, x3)");
  }
  if (mode_ == 2 && nvar_ != 6) {
    SFFatal(blk, "mode=aniso_mhd needs six variables: velocity x1, x2, x3 then "
                 "magnetic field x1, x2, x3 (e.g. mhd_w_vx, mhd_w_vy, mhd_w_vz, "
                 "mhd_bcc1, mhd_bcc2, mhd_bcc3)");
  }
  if (mode_ == 3 && nvar_ != 1 && nvar_ != 2) {
    SFFatal(blk, "mode=correlation needs one variable (autocorrelation <A(x)A(x+r)>) "
                 "or two (<A(x)B(x+r)>)");
  }
  // increments per pair: scalar: one per variable; long_trans: |dv_l|, dv_t;
  // aniso_mhd: |dv|, |dv_perp|, |dB|, |dB_perp| (perp to the pair-mean B direction);
  // correlation: one product
  nq_ = (mode_ == 0) ? nvar_ : ((mode_ == 1) ? 2 : ((mode_ == 2) ? 4 : 1));

  // outvars were loaded in variable_list order by BaseTypeOutput: one entry per name
  if (out_params.contains_derived) {
    SFFatal(blk, "derived variables are not yet supported by sf output (except the "
                 "computed variables temperature and pressure)");
  }
  for (std::size_t n = 0; n < outvars_per_name.size(); ++n) {
    if (outvars_per_name[n] != 1) {
      SFFatal(blk, "'" + out_params.variable_list[n] + "' gives " +
                   std::to_string(outvars_per_name[n]) + " field components here; every "
                   "sf variable/weight must name exactly one");
    }
  }
  if (static_cast<int>(outvars.size()) > kSFMaxFields) {
    SFFatal(blk, "too many distinct fields (max " + std::to_string(kSFMaxFields) + ")");
  }
  // the device weight map and the per-pair arrays hold kSFMaxFields entries; repeated
  // names (e.g. several "ones") do not add fields but do add weights
  if (nweight_ > kSFMaxFields || nvar_ > kSFMaxFields) {
    SFFatal(blk, "at most " + std::to_string(kSFMaxFields) + " weights and " +
                 std::to_string(kSFMaxFields) + " variables per sf output");
  }
  // every loaded field, by variable_list index (selection and weight sources)
  src_fields_.nfield = static_cast<int>(outvars.size());
  for (int n = 0; n < src_fields_.nfield; ++n) {
    src_fields_.arr[n] = *(outvars[n].data_ptr);
    src_fields_.index[n] = outvars[n].data_index;
  }
  auto loaded = [&](const std::string &name) -> int {
    for (std::size_t n = 0; n < out_params.variable_list.size(); ++n) {
      if (out_params.variable_list[n] == name) return static_cast<int>(n);
    }
    return -1;
  };
  gm1_ = 0.0;
  eos_mhd_ = false;
  // fields_: variables in slots 0..nvar-1 (computed ones read der_), then the distinct
  // non-unit weights that are not variables, in order of first appearance
  auto &mbi0 = pm->mb_indcs;
  const int c3 = (mbi0.nx3 > 1) ? mbi0.nx3 + 2*mbi0.ng : 1;
  const int c2 = (mbi0.nx2 > 1) ? mbi0.nx2 + 2*mbi0.ng : 1;
  const int c1 = mbi0.nx1 + 2*mbi0.ng;
  int nder = 0;
  for (const auto &v : var_names_) {
    if (v.compare("temperature") == 0 || v.compare("pressure") == 0) nder++;
  }
  if (nder > 0) {
    der_ = DvceArray5D<Real>("sf_der", pm->pmb_pack->nmb_thispack, nder, c3, c2, c1);
  }
  fields_.nfield = 0;
  for (int iv = 0; iv < nvar_; ++iv) {
    const std::string &v = var_names_[iv];
    const int slot = fields_.nfield++;
    if (v.compare("temperature") == 0 || v.compare("pressure") == 0) {
      SetupComputed(pm, v);
      fields_.arr[slot] = der_;
      fields_.index[slot] = static_cast<int>(der_kind_.size());
      der_kind_.push_back(v.compare("temperature") == 0 ? kSFTemperature : kSFPressure);
    } else {
      const int n = loaded(v);
      fields_.arr[slot] = src_fields_.arr[n];
      fields_.index[slot] = src_fields_.index[n];
    }
  }
  weight_field_.assign(nweight_, -1);
  weight_src_.assign(nweight_, -1);
  for (int iw = 0; iw < nweight_; ++iw) {
    const std::string &w = weight_names_[iw];
    if (w.compare("ones") == 0) continue;
    weight_src_[iw] = loaded(w);
    for (int iv = 0; iv < nvar_ && weight_field_[iw] < 0; ++iv) {
      if (var_names_[iv] == w) weight_field_[iw] = iv;
    }
    for (int jw = 0; jw < iw && weight_field_[iw] < 0; ++jw) {
      if (weight_names_[jw] == w) weight_field_[iw] = weight_field_[jw];
    }
    if (weight_field_[iw] < 0) {
      if (fields_.nfield >= kSFMaxFields) {
        SFFatal(blk, "too many distinct fields (max " + std::to_string(kSFMaxFields) +
                     ")");
      }
      const int slot = fields_.nfield++;
      fields_.arr[slot] = src_fields_.arr[weight_src_[iw]];
      fields_.index[slot] = src_fields_.index[weight_src_[iw]];
      weight_field_[iw] = slot;
    }
  }

  // ---- cell selection: select<n>_variable with select<n>_min <= value <= select<n>_max
  // (inclusive, defaults -inf/+inf). Each weight then reads its own sel_w_ component,
  // so variables come first in fields_ and the weights follow ----
  nsel_total_ = 0;
  time_select_ = 0.0;
  for (int c = 1; c <= 2; ++c) {
    const std::string key = "select" + std::to_string(c);
    if (!pin->DoesParameterExist(blk, key + "_variable")) {
      if (pin->DoesParameterExist(blk, key + "_min") ||
          pin->DoesParameterExist(blk, key + "_max")) {
        SFFatal(blk, key + "_min/" + key + "_max given without " + key + "_variable");
      }
      continue;
    }
    if (c == 2 && select_.nsel == 0) SFFatal(blk, "select2_* needs select1_*");
    const std::string name = pin->GetString(blk, key + "_variable");
    const int k = select_.nsel++;
    select_names_.push_back(name);
    select_.lo[k] = GetOutputReal(pin, blk, key + "_min", -INFINITY, true);
    select_.hi[k] = GetOutputReal(pin, blk, key + "_max", INFINITY, true);
    if (std::isnan(select_.lo[k]) || std::isnan(select_.hi[k]) ||
        !(select_.lo[k] <= select_.hi[k])) {
      SFFatal(blk, key + "_min must be <= " + key + "_max");
    }
    if (name.compare("temperature") == 0 || name.compare("pressure") == 0) {
      SetupComputed(pm, name);
      select_.field[k] = (name.compare("temperature") == 0) ? kSFTemperature :
                                                               kSFPressure;
    } else {
      select_.field[k] = loaded(name);
      if (select_.field[k] < 0) {
        SFFatal(blk, key + "_variable='" + name + "' is not a single output variable");
      }
    }
  }
  if (select_.nsel > 0) {
    if (nvar_ + nweight_ > kSFMaxFields) {
      SFFatal(blk, "with select*, variables + weights must be <= " +
                   std::to_string(kSFMaxFields));
    }
    auto &mbi = pm->mb_indcs;
    const int n3 = (mbi.nx3 > 1) ? mbi.nx3 + 2*mbi.ng : 1;
    const int n2 = (mbi.nx2 > 1) ? mbi.nx2 + 2*mbi.ng : 1;
    sel_w_ = DvceArray5D<Real>("sf_sel_w", pm->pmb_pack->nmb_thispack, nweight_, n3, n2,
                               mbi.nx1 + 2*mbi.ng);
    for (int iw = 0; iw < nweight_; ++iw) {
      fields_.arr[nvar_ + iw] = sel_w_;
      fields_.index[nvar_ + iw] = iw;
      weight_field_[iw] = nvar_ + iw;
    }
    fields_.nfield = nvar_ + nweight_;
  }

  // ---- geometry ----
  auto &ms = pm->mesh_size;
  len_[0] = ms.x1max - ms.x1min;
  len_[1] = ms.x2max - ms.x2min;
  len_[2] = ms.x3max - ms.x3min;
  periodic_[0] = (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::periodic);
  periodic_[1] = (pm->mesh_bcs[BoundaryFace::inner_x2] == BoundaryFlag::periodic);
  periodic_[2] = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
  geom_.per0 = periodic_[0];
  geom_.per1 = periodic_[1];
  geom_.per2 = periodic_[2];
  geom_.n0 = pm->mesh_indcs.nx1;
  geom_.n1 = pm->mesh_indcs.nx2;
  geom_.n2 = pm->mesh_indcs.nx3;
  geom_.h0 = static_cast<double>(len_[0])/static_cast<double>(geom_.n0);
  geom_.h1 = static_cast<double>(len_[1])/static_cast<double>(geom_.n1);
  geom_.h2 = static_cast<double>(len_[2])/static_cast<double>(geom_.n2);
  nactive_dim_ = pm->three_d ? 3 : (pm->two_d ? 2 : 1);
  auto &mbi = pm->mb_indcs;
  int nb[3] = {mbi.nx1, mbi.nx2, mbi.nx3};
  cmax_ = 0;
  for (int d = 0; d < nactive_dim_; ++d) cmax_ = std::max(cmax_, nb[d] - 1);

  // ---- separation bins: Python radial_bin_edges, defaults rmin=min dx, rmax=half
  // box diagonal ----
  max_order_ = pin->GetOrAddInteger(blk, "max_order", 10);
  nbin_prll_ = nbin_perp_ = 0;
  if (mode_ == 2) {
    // get_sf_aniso_mhd_mb defaults
    nbin_prll_ = pin->GetOrAddInteger(blk, "nbin_prll", 50);
    nbin_perp_ = pin->GetOrAddInteger(blk, "nbin_perp", 50);
    nbin_l_ = pin->GetOrAddInteger(blk, "nbin", std::max(nbin_prll_, nbin_perp_));
    theta_prll_max_ = GetOutputReal(pin, blk, "theta_prll_max", M_PI/18.0);
    theta_perp_min_ = GetOutputReal(pin, blk, "theta_perp_min", M_PI/2.0 - M_PI/18.0);
    if (nbin_prll_ < 1 || nbin_perp_ < 1) {
      SFFatal(blk, "nbin_prll and nbin_perp must be >= 1");
    }
    if (!(theta_prll_max_ >= 0.0 && theta_prll_max_ <= 0.5*M_PI) ||
        !(theta_perp_min_ >= 0.0 && theta_perp_min_ <= 0.5*M_PI)) {
      SFFatal(blk, "theta_prll_max and theta_perp_min must lie in [0, pi/2]");
    }
  } else {
    nbin_l_ = pin->GetOrAddInteger(blk, "nbin", 100);
    theta_prll_max_ = theta_perp_min_ = 0.0;
  }
  nbin_ = nbin_l_ + nbin_prll_ + nbin_perp_;
  logscale_ = pin->GetOrAddBoolean(blk, "logscale", true);
  if (max_order_ < 1 || nbin_l_ < 1) SFFatal(blk, "max_order and nbin must be >= 1");
  Real dxmin = ms.dx1;
  if (nactive_dim_ > 1) dxmin = std::min(dxmin, ms.dx2);
  if (nactive_dim_ > 2) dxmin = std::min(dxmin, ms.dx3);
  r_min_ = GetOutputReal(pin, blk, "r_min", 0.0);
  r_max_ = GetOutputReal(pin, blk, "r_max", 0.0);
  if (r_min_ <= 0.0) r_min_ = dxmin;
  if (r_max_ <= 0.0) {
    r_max_ = std::sqrt(0.25*(len_[0]*len_[0] + len_[1]*len_[1] + len_[2]*len_[2]));
  }
  if (!(r_max_ > r_min_)) SFFatal(blk, "r_max must exceed r_min");
  // Python radial_bin_edges; every aniso_mhd family uses the same [r_min, r_max]
  auto make_edges = [&](DvceArray1D<double> &edges, int nb) {
    edges = DvceArray1D<double>("sf_rbins", nb + 1);
    auto h = Kokkos::create_mirror_view(edges);
    for (int b = 0; b <= nb; ++b) {
      double f = static_cast<double>(b)/nb;
      h(b) = logscale_ ?
          std::pow(10.0, std::log10(r_min_) + f*(std::log10(r_max_)-std::log10(r_min_))) :
          f*r_max_;
    }
    if (logscale_) h(0) = r_min_*(1.0 - 1e-9);
    h(nb) = r_max_*(1.0 + 1e-9);
    Kokkos::deep_copy(edges, h);
  };
  make_edges(rbins_, nbin_l_);
  if (mode_ == 2) {
    make_edges(rbins_prll_, nbin_prll_);
    make_edges(rbins_perp_, nbin_perp_);
  }

  // ---- sampling budgets ----
  nrep_ = pin->GetOrAddInteger(blk, "nreplicate", 4);
  ndraw_block_ = SFGetCount(pin, blk, "ndraw_block", 4096.0);
  nsample_global_ = SFGetCount(pin, blk, "nsample_global", 65536.0);
  nanchor_global_ = SFGetCount(pin, blk, "nanchor_global", 1048576.0);
  seed_ = static_cast<uint64_t>(pin->GetOrAddInteger(blk, "seed", 0));
  sparse_ = pin->GetOrAddBoolean(blk, "sparse", false);
  nsupp_local_ = 0;
  halo_requested_ = pin->GetOrAddInteger(blk, "sf_halo", 0);
  halo_effective_ = 0;   // halo reuse (g > 0) is not implemented yet
  timing_ = pin->GetOrAddBoolean(blk, "timing", false);
  if (nrep_ < 1) SFFatal(blk, "nreplicate must be >= 1");
  if (ndraw_block_ < nrep_) SFFatal(blk, "ndraw_block must be >= nreplicate");
  // the E2 sample is gathered by one MPI_Allgatherv with int counts; check before any
  // sample is drawn or gathered
  if (nsample_global_*(4 + fields_.nfield) > INT32_MAX) {
    SFFatal(blk, "nsample_global x (4 + number of fields) must not exceed " +
                 std::to_string(INT32_MAX) +
                 " (one MPI_Allgatherv); reduce nsample_global");
  }
  if (halo_requested_ > 0 && global_variable::my_rank == 0) {
    std::cout << "### WARNING in sf output block '" << blk << "': sf_halo="
              << halo_requested_ << " requested, but halo reuse is not enabled yet; "
              << "using g=0" << std::endl;
  }

  // E1 radius distribution p(c) ~ 1/c, c = 1..cmax_
  harmonic_cmax_ = 0.0;
  for (int c = 1; c <= cmax_; ++c) harmonic_cmax_ += 1.0/c;
  Kokkos::realloc(cdf_c_, std::max(cmax_, 1));
  auto h_cdf = Kokkos::create_mirror_view(cdf_c_);
  double run = 0.0;
  for (int c = 1; c <= cmax_; ++c) {
    run += (1.0/c)/harmonic_cmax_;
    h_cdf(c-1) = run;
  }
  if (cmax_ >= 1) h_cdf(cmax_-1) = 1.0;
  Kokkos::deep_copy(cdf_c_, h_cdf);

#if MPI_PARALLEL_ENABLED
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
#endif
  BuildStrata(pm);
  if (!sparse_) {
    AllocateBudgets(ncell_region_rank_, false);
  } else if (std::accumulate(ncell_region_rank_.begin(), ncell_region_rank_.end(),
                             int64_t{0}) == 0) {
    // a sparse support may be empty at an output, but the region itself may not
    SFFatal(blk, "the sf region contains no cells");
  }

  // ---- accumulators ----
  std::size_t nT = 2ULL*nrep_*nweight_*nq_*max_order_*nbin_;
  std::size_t nW = 2ULL*nrep_*nweight_*nbin_;
  Kokkos::realloc(acc_T_, nT);
  Kokkos::realloc(acc_W_, nW);
  Kokkos::realloc(acc_C_, nW);
  Kokkos::realloc(acc_n_, nW);
  h_T_ = Kokkos::create_mirror_view(acc_T_);
  h_W_ = Kokkos::create_mirror_view(acc_W_);
  h_C_ = Kokkos::create_mirror_view(acc_C_);
  h_n_ = Kokkos::create_mirror_view(acc_n_);

  // ---- pair-kernel variant: team-scratch histograms with as many weights per pass as
  // fit in 48 KiB (a shared-memory budget every GPU offers), else global atomics ----
  std::string kernel = pin->GetOrAddString(blk, "kernel", "auto");
  if (kernel != "auto" && kernel != "team" && kernel != "global") {
    SFFatal(blk, "kernel='" + kernel + "' must be one of auto, team, global");
  }
  nw_pass_ = 0;
  if (kernel != "global") {
    const SFPairSpec spec =
        SFMakeSpec(nq_, max_order_, nbin_, mode_, nvar_, weight_field_);
    std::size_t smax = Kokkos::TeamPolicy<>(1, Kokkos::AUTO).scratch_size_max(0);
    smax = std::min<std::size_t>(smax, 48*1024);
    for (int n = nweight_; n >= 1; --n) {
      if (SFScratchHist(n, spec).Bytes() <= smax) {nw_pass_ = n; break;}
    }
    if (kernel == "team" && nw_pass_ == 0) {
      SFFatal(blk, "kernel=team: one weight's histogram (nq*max_order*nbin) does not "
                   "fit in team scratch; use kernel=auto or global");
    }
  }

  // ---- increment histograms: Python delta_pdf columns (underflow, bins, overflow);
  // default edges are structure_functions.py DEFAULT_NORMALIZED_(SIGNED_)DELTA_EDGES
  // times each variable's weighted standard deviation ----
  hist_ = pin->GetOrAddBoolean(blk, "histogram", false);
  if (hist_ && mode_ >= 2) {
    SFFatal(blk, "histogram = true is not available for mode = aniso_mhd or "
                 "correlation");
  }
  hist_ncol_ = 1;
  hist_nedge_.assign(nq_, 0);
  h_hist_edges_ = HostArray2D<double>("sf_hhedges", 1, 1);
  hist_edges_ = DvceArray2D<double>("sf_hedges", 1, 1);
  acc_H_ = DvceArray1D<double>("sf_H", 1);
  if (hist_) {
    hist_nbin_ = pin->GetOrAddInteger(blk, "hist_nbin", 96);
    hist_min_ = GetOutputReal(pin, blk, "hist_min", 1.0e-4);
    hist_max_ = GetOutputReal(pin, blk, "hist_max", 1.0e2);
    std::string scale = pin->GetOrAddString(blk, "hist_scale", "sigma");
    if (scale != "sigma" && scale != "absolute") {
      SFFatal(blk, "hist_scale='" + scale + "' must be one of sigma, absolute");
    }
    hist_sigma_scale_ = (scale == "sigma");
    if (hist_nbin_ < 2 || hist_nbin_ % 2 != 0 || !(hist_min_ > 0.0) ||
        !(hist_max_ > hist_min_)) {
      SFFatal(blk, "need an even hist_nbin >= 2 and 0 < hist_min < hist_max");
    }
    const double a = std::log10(hist_min_), b = std::log10(hist_max_);
    hist_tmpl_[0] = SFLogspace(a, b, hist_nbin_ + 1);
    std::vector<double> pos = SFLogspace(a, b, hist_nbin_/2 + 1);
    hist_tmpl_[1].clear();
    for (auto it = pos.rbegin(); it != pos.rend(); ++it) hist_tmpl_[1].push_back(-*it);
    for (double x : pos) hist_tmpl_[1].push_back(x);
    int nmax = 0;
    for (int iq = 0; iq < nq_; ++iq) {
      const int kind = (mode_ == 1 && iq == 0) ? 1 : 0;
      hist_nedge_[iq] = static_cast<int>(hist_tmpl_[kind].size());
      nmax = std::max(nmax, hist_nedge_[iq]);
    }
    hist_ncol_ = nmax + 1;
    h_hist_edges_ = HostArray2D<double>("sf_hhedges", nweight_*nq_, nmax);
    hist_edges_ = DvceArray2D<double>("sf_hedges", nweight_*nq_, nmax);
    acc_H_ = DvceArray1D<double>("sf_H", 2ULL*nrep_*nweight_*nq_*nbin_*hist_ncol_);
  }
  h_H_ = Kokkos::create_mirror_view(acc_H_);

  std::string dir_name = "sf_" + out_params.file_id;
  mkdir(dir_name.c_str(), 0775);
}

SFOutput::~SFOutput() {
#if MPI_PARALLEL_ENABLED
  MPI_Comm_free(&comm_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::BuildStrata()
//  \brief region index box of every local MeshBlock (same cell-centre test as the
//  kernels), its prefix sum, and the all-gathered region cell count of every rank.

void SFOutput::BuildStrata(Mesh *pm) {
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &mbi = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  Kokkos::realloc(h_box_, std::max(nmb, 1), 6);
  Kokkos::realloc(h_box_offset_, nmb + 1);
  h_box_offset_(0) = 0;
  const Real lo[3] = {out_params.x1_min, out_params.x2_min, out_params.x3_min};
  const Real hi[3] = {out_params.x1_max, out_params.x2_max, out_params.x3_max};
  for (int m = 0; m < nmb; ++m) {
    const Real xmin[3] = {size.h_view(m).x1min, size.h_view(m).x2min,
                          size.h_view(m).x3min};
    const Real xmax[3] = {size.h_view(m).x1max, size.h_view(m).x2max,
                          size.h_view(m).x3max};
    const int nx[3] = {mbi.nx1, mbi.nx2, mbi.nx3};
    int64_t ncell = 1;
    for (int d = 0; d < 3; ++d) {
      int first = nx[d], last = -1;
      for (int i = 0; i < nx[d]; ++i) {
        Real x = CellCenterX(i, nx[d], xmin[d], xmax[d]);
        if (x >= lo[d] && x <= hi[d]) {first = std::min(first, i); last = i;}
      }
      h_box_(m, 2*d) = first;
      h_box_(m, 2*d+1) = last;
      ncell *= (last >= first) ? (last - first + 1) : 0;
    }
    h_box_offset_(m+1) = h_box_offset_(m) + ncell;
  }
  ncell_region_local_ = h_box_offset_(nmb);
  Kokkos::realloc(box_, std::max(nmb, 1), 6);
  Kokkos::realloc(box_offset_, nmb + 1);
  Kokkos::deep_copy(box_, h_box_);
  Kokkos::deep_copy(box_offset_, h_box_offset_);

  ncell_region_rank_.assign(global_variable::nranks, 0);
#if MPI_PARALLEL_ENABLED
  int64_t mine = ncell_region_local_;
  MPI_Allgather(&mine, 1, MPI_INT64_T, ncell_region_rank_.data(), 1, MPI_INT64_T, comm_);
#else
  ncell_region_rank_[0] = ncell_region_local_;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::BuildSupport()
//  \brief sparse = true: compacted list of this rank's support cells (region cells where
//  some weight is > 0; every region cell if a weight is "ones"), in region-local linear
//  order, with the first entry of each local MeshBlock, and the support count of every
//  rank. Rebuilt at every output, since the weights evolve.

void SFOutput::BuildSupport(Mesh *pm) {
  const int nmb = pm->pmb_pack->nmb_thispack;
  auto &mbi = pm->mb_indcs;
  const int is = mbi.is, js = mbi.js, ks = mbi.ks;
  const int64_t nreg = ncell_region_local_;
  const auto box = box_;
  const auto boxoff = box_offset_;
  const SFFieldSet fields = fields_;
  SFWeightMap wm;
  bool all = false;
  for (int iw = 0; iw < nweight_; ++iw) {
    wm.field[iw] = weight_field_[iw];
    if (weight_field_[iw] < 0) all = true;
  }
  const int nw = nweight_;
  // the same test as the pair kernels: a cell takes part for weight w iff w > 0
  auto in_support = KOKKOS_LAMBDA(const int64_t u) -> int64_t {
    if (all) return 1;
    int m, i, j, k;
    SFDecodeRegion(boxoff, box, nmb, u, m, i, j, k);
    for (int iw = 0; iw < nw; ++iw) {
      const double a = static_cast<double>(fields.Get(wm.field[iw], m, k+ks, j+js, i+is));
      if (a > 0.0) return 1;
    }
    return 0;
  };
  int64_t count = 0;
  if (nreg > 0) {
    Kokkos::parallel_reduce("sf_supp_count",
    Kokkos::RangePolicy<DevExeSpace, int64_t>(0, nreg),
    KOKKOS_LAMBDA(const int64_t u, int64_t &n) {n += in_support(u);}, count);
  }
  Kokkos::realloc(supp_, std::max<int64_t>(count, 1));
  Kokkos::realloc(supp_off_, nmb + 1);
  auto supp = supp_;
  auto soff = supp_off_;
  if (count > 0) {
    Kokkos::parallel_scan("sf_supp_fill",
    Kokkos::RangePolicy<DevExeSpace, int64_t>(0, nreg),
    KOKKOS_LAMBDA(const int64_t u, int64_t &upd, const bool final) {
      const int64_t f = in_support(u);
      if (final && f) supp(upd) = u;
      upd += f;
    });
  }
  // first support entry of every MeshBlock: lower bound of its first region index
  Kokkos::parallel_for("sf_supp_off", Kokkos::RangePolicy<DevExeSpace>(0, nmb + 1),
  KOKKOS_LAMBDA(const int m) {
    const int64_t target = boxoff(m);
    int64_t lo = 0, hi = count;
    while (lo < hi) {
      const int64_t mid = (lo + hi)/2;
      if (supp(mid) < target) {lo = mid + 1;} else {hi = mid;}
    }
    soff(m) = lo;
  });
  h_supp_off_ = Kokkos::create_mirror_view_and_copy(HostMemSpace(), supp_off_);
  nsupp_local_ = count;
  nsupp_rank_.assign(global_variable::nranks, 0);
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&count, 1, MPI_INT64_T, nsupp_rank_.data(), 1, MPI_INT64_T, comm_);
#else
  nsupp_rank_[0] = count;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::SetupComputed()
//  \brief checks that "temperature"/"pressure" can be computed (Newtonian, single-fluid
//  hydro or mhd, eos = ideal) and stores gamma - 1 and which primitives to read

void SFOutput::SetupComputed(Mesh *pm, const std::string &what) {
  auto *pmbp = pm->pmb_pack;
  const std::string &blk = out_params.block_name;
  if (pmbp->pdyngr != nullptr || pmbp->pcoord->is_general_relativistic ||
      pmbp->pcoord->is_special_relativistic) {
    SFFatal(blk, what + " (computed field) supports Newtonian hydro/mhd only");
  }
  // two-fluid (ion-neutral) runs have one temperature per fluid: refuse to pick one
  if (pmbp->phydro != nullptr && pmbp->pmhd != nullptr) {
    SFFatal(blk, what + " (computed field) is ambiguous with both <hydro> and <mhd> "
                 "(two-fluid run); it supports single-fluid runs only");
  }
  EOS_Data *eos = (pmbp->phydro != nullptr) ? &(pmbp->phydro->peos->eos_data) :
                  ((pmbp->pmhd != nullptr) ? &(pmbp->pmhd->peos->eos_data) : nullptr);
  if (eos == nullptr || !eos->is_ideal) {
    SFFatal(blk, what + " (computed field) needs <hydro> or <mhd> with eos = ideal");
  }
  eos_mhd_ = (pmbp->phydro == nullptr);
  gm1_ = eos->gamma - 1.0;
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::ComputeFields()
//  \brief der_(m, n, k, j, i) = temperature (gamma-1) e/rho or pressure (gamma-1) e for
//  every interior cell. Called at every output before anything reads the fields.

void SFOutput::ComputeFields(Mesh *pm) {
  const int nder = static_cast<int>(der_kind_.size());
  if (nder == 0) return;
  auto &mbi = pm->mb_indcs;
  auto *pmbp = pm->pmb_pack;
  DvceArray5D<Real> w0 = eos_mhd_ ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  SFWeightMap kind;   // reuses the fixed-size int array for the per-component kind
  for (int n = 0; n < nder; ++n) kind.field[n] = der_kind_[n];
  const double gm1 = gm1_;
  auto der = der_;
  par_for("sf_computed", DevExeSpace(), 0, pmbp->nmb_thispack-1, mbi.ks, mbi.ke, mbi.js,
          mbi.je, mbi.is, mbi.ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const double p = gm1*static_cast<double>(w0(m, IEN, k, j, i));
    for (int n = 0; n < nder; ++n) {
      der(m, n, k, j, i) = (kind.field[n] == kSFTemperature) ?
          p/static_cast<double>(w0(m, IDN, k, j, i)) : p;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::ApplySelection()
//  \brief select*: sel_w_(m, iw, k, j, i) = (cell selected ? weight iw : 0) for every
//  region cell, and the number of selected region cells over all ranks. Called at every
//  output before anything reads the weights.

void SFOutput::ApplySelection(Mesh *pm) {
  const int nmb = pm->pmb_pack->nmb_thispack;
  auto &mbi = pm->mb_indcs;
  const int is = mbi.is, js = mbi.js, ks = mbi.ks;
  const int64_t nreg = ncell_region_local_;
  const auto box = box_;
  const auto boxoff = box_offset_;
  const SFFieldSet src = src_fields_;
  const SFSelect sel = select_;
  SFWeightMap wm;
  for (int iw = 0; iw < nweight_; ++iw) wm.field[iw] = weight_src_[iw];
  const int nw = nweight_;
  const double gm1 = gm1_;
  auto *pmbp = pm->pmb_pack;
  DvceArray5D<Real> w0;
  if (gm1 > 0.0) w0 = eos_mhd_ ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  auto selw = sel_w_;
  int64_t count = 0;
  if (nreg > 0) {
    Kokkos::parallel_reduce("sf_select",
    Kokkos::RangePolicy<DevExeSpace, int64_t>(0, nreg),
    KOKKOS_LAMBDA(const int64_t u, int64_t &n) {
      int m, i, j, k;
      SFDecodeRegion(boxoff, box, nmb, u, m, i, j, k);
      bool ok = true;
      for (int c = 0; c < sel.nsel; ++c) {
        double v;
        if (sel.field[c] >= 0) {
          v = static_cast<double>(src.Get(sel.field[c], m, k+ks, j+js, i+is));
        } else {
          v = gm1*static_cast<double>(w0(m, IEN, k+ks, j+js, i+is));
          if (sel.field[c] == kSFTemperature) {
            v /= static_cast<double>(w0(m, IDN, k+ks, j+js, i+is));
          }
        }
        ok = ok && (v >= sel.lo[c]) && (v <= sel.hi[c]);
      }
      for (int iw = 0; iw < nw; ++iw) {
        selw(m, iw, k+ks, j+js, i+is) = !ok ? 0.0 :
            ((wm.field[iw] < 0) ? 1.0 : src.Get(wm.field[iw], m, k+ks, j+js, i+is));
      }
      n += ok ? 1 : 0;
    }, count);
  }
  nsel_total_ = count;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&count, &nsel_total_, 1, MPI_INT64_T, MPI_SUM, comm_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::AllocateBudgets()
//  \brief per-replicate E2 sample and anchor quotas for every rank: at least one of each
//  per nonempty rank and replicate, the remainder proportional to the rank's cells
//  (region cells; support cells when sparse). Returns false if no rank has a cell
//  (allowed only for a sparse support, whose output is then empty).

bool SFOutput::AllocateBudgets(const std::vector<int64_t> &ncell_rank, bool allow_empty) {
  int nranks = global_variable::nranks;
  int64_t pplus = 0;
  for (int s = 0; s < nranks; ++s) {
    if (ncell_rank[s] > 0) pplus++;
  }
  if (pplus == 0) {
    if (allow_empty) return false;
    SFFatal(out_params.block_name, "the sf region contains no cells");
  }
  if (nsample_global_ < nrep_*pplus || nanchor_global_ < nrep_*pplus) {
    SFFatal(out_params.block_name, "nsample_global and nanchor_global must each be >= "
            "nreplicate * (number of ranks holding " +
            std::string(sparse_ ? "support" : "region") + " cells) = " +
            std::to_string(nrep_*pplus));
  }
  nsample_rt_.assign(nranks, std::vector<int64_t>(nrep_, 0));
  nanchor_rt_.assign(nranks, std::vector<int64_t>(nrep_, 0));
  for (int t = 0; t < nrep_; ++t) {
    int64_t mt = nsample_global_/nrep_ + ((t < nsample_global_ % nrep_) ? 1 : 0);
    int64_t at = nanchor_global_/nrep_ + ((t < nanchor_global_ % nrep_) ? 1 : 0);
    std::vector<int64_t> ms = SFAllocate(mt, ncell_rank);
    std::vector<int64_t> as = SFAllocate(at, ncell_rank);
    for (int s = 0; s < nranks; ++s) {
      nsample_rt_[s][t] = ms[s];
      nanchor_rt_[s][t] = as[s];
    }
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::LoadOutputData()
//  \brief zero accumulators, run E1 and E2, reduce to rank 0

void SFOutput::LoadOutputData(Mesh *pm) {
  Kokkos::deep_copy(acc_T_, 0.0);
  Kokkos::deep_copy(acc_W_, 0.0);
  Kokkos::deep_copy(acc_C_, 0.0);
  Kokkos::deep_copy(acc_n_, 0);
  attempts_[0] = attempts_[1] = 0;
  ComputeFields(pm);
  if (select_.nsel > 0) {
    Kokkos::Timer tsel;
    ApplySelection(pm);
    Kokkos::fence();
    time_select_ = tsel.seconds();
  }
  if (hist_) {
    Kokkos::deep_copy(acc_H_, 0.0);
    BuildHistEdges(pm);
  }

  // sparse: support cells and budgets of this output (an empty support samples nothing)
  bool any = true;
  if (sparse_) {
    Kokkos::Timer tsupp;
    BuildSupport(pm);
    any = AllocateBudgets(nsupp_rank_, true);
    time_support_ = tsupp.seconds();
  }

  Kokkos::Timer timer;
  if (any) SampleWithinBlocks(pm);
  Kokkos::fence();
  time_e1_ = timer.seconds();
  timer.reset();
  if (any) SampleAcrossBlocks(pm);
  Kokkos::fence();
  time_e2_ = timer.seconds();
  timer.reset();
  ReduceAccumulators();
  time_reduce_ = timer.seconds();
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::SampleWithinBlocks()
//  \brief E1: ordered pairs inside one MeshBlock, drawn lag-first by SFE1Sampler.
//  Team kernel: each team accumulates a chunk of draws of one replicate into a scratch
//  histogram and flushes it once; one pass per group of weights that fits in scratch.
//  Global kernel (fallback, kernel = global): atomics straight into the accumulators.

void SFOutput::SampleWithinBlocks(Mesh *pm) {
  auto &mbi = pm->mb_indcs;
  const int nmb = pm->pmb_pack->nmb_thispack;
  // attempts: every draw in a block holding region (sparse: support) cells
  int nmb_region = 0;
  for (int m = 0; m < nmb; ++m) {
    if (sparse_) {
      if (h_supp_off_(m+1) > h_supp_off_(m)) nmb_region++;
    } else if (h_box_(m, 1) >= h_box_(m, 0) && h_box_(m, 3) >= h_box_(m, 2) &&
               h_box_(m, 5) >= h_box_(m, 4)) {
      nmb_region++;
    }
  }
  attempts_[0] = static_cast<uint64_t>(ndraw_block_)*nmb_region;
  if (cmax_ < 1 || nmb_region == 0) return;

  const int is = mbi.is, js = mbi.js, ks = mbi.ks;
  SFE1Sampler smp;
  smp.rbins = rbins_;
  smp.cdf = cdf_c_;
  smp.box = box_;
  smp.size = pm->pmb_pack->pmb->mb_size;
  smp.gid = pm->pmb_pack->pmb->mb_gid;
  smp.ndim = nactive_dim_;
  smp.cmax = cmax_;
  smp.nbin = nbin_l_;
  smp.keep_over = (mode_ == 2);
  smp.nrep = nrep_;
  smp.nx1 = mbi.nx1;
  smp.nx2 = mbi.nx2;
  smp.nx3 = mbi.nx3;
  smp.ndraw = ndraw_block_;
  smp.harm = harmonic_cmax_;
  smp.seed = seed_;
  smp.outnum = static_cast<uint64_t>(out_params.file_number);
  smp.geom = geom_;
  smp.sparse = sparse_;
  if (sparse_) {
    smp.supp = supp_;
    smp.soff = supp_off_;
    smp.boxoff = box_offset_;
  }
  const SFFieldSet fields = fields_;
  const int nfield = fields_.nfield;
  SFPairSpec spec = SFMakeSpec(nq_, max_order_, nbin_, mode_, nvar_, weight_field_);
  SFSetHist(&spec, hist_, hist_ncol_, hist_nedge_, hist_edges_);
  SFSetAniso(&spec, nbin_l_, nbin_prll_, nbin_perp_, theta_prll_max_, theta_perp_min_,
             rbins_prll_, rbins_perp_);
  SFGlobalSink gsink{acc_T_, acc_W_, acc_C_, acc_n_, 0, nweight_, nq_, max_order_,
                     nbin_, acc_H_, hist_ncol_};
  const int nrep = nrep_, nw = nweight_;
  const int64_t nd_max = (ndraw_block_ + nrep_ - 1)/nrep_;
  const int64_t nper = static_cast<int64_t>(nmb)*nd_max;   // draw slots per replicate

  if (nw_pass_ == 0) {
    Kokkos::parallel_for("sf_e1_global",
    Kokkos::RangePolicy<DevExeSpace, int64_t>(0, nrep*nper),
    KOKKOS_LAMBDA(const int64_t idx) {
      const int t = static_cast<int>(idx/nper);
      const int64_t q = idx % nper;
      const int m = static_cast<int>(q/nd_max);
      int i1, j1, k1, i2, j2, k2, bin;
      double dx, dy, dz, r, fac;
      if (!smp.Draw(m, t, q % nd_max, i1, j1, k1, i2, j2, k2, dx, dy, dz, r, bin, fac)) {
        return;
      }
      double v1[kSFMaxFields], v2[kSFMaxFields];
      for (int n = 0; n < nfield; ++n) {
        v1[n] = fields.Get(n, m, k1 + ks, j1 + js, i1 + is);
        v2[n] = fields.Get(n, m, k2 + ks, j2 + js, i2 + is);
      }
      SFGlobalSink g = gsink;
      g.er = t;
      SFAddPair(g, spec, 0, nw, bin, fac, v1, v2, dx, dy, dz, r);
    });
    return;
  }

  const int64_t chunk = SFTeamChunk(nrep*nper, nbin_);
  const int64_t nchunk = (nper + chunk - 1)/chunk;
  for (int w0 = 0; w0 < nw; w0 += nw_pass_) {
    const int w1 = std::min(nw, w0 + nw_pass_);
    const SFScratchHist hist(w1 - w0, spec);
    Kokkos::TeamPolicy<> policy(DevExeSpace(), static_cast<int>(nrep*nchunk),
                                Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(hist.Bytes()));
    Kokkos::parallel_for("sf_e1_team", policy,
    KOKKOS_LAMBDA(const TeamMember_t &member) {
      const int t = static_cast<int>(member.league_rank()/nchunk);
      const int64_t q0 = (member.league_rank() % nchunk)*chunk;
      const int64_t q1 = (q0 + chunk < nper) ? q0 + chunk : nper;
      SFGlobalSink g = gsink;
      g.er = t;
      const SFScratchSink k = SFMakeScratchSink(member, hist, spec, w0, g);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(member, q0, q1),
      [&](const int64_t q) {
        const int m = static_cast<int>(q/nd_max);
        int i1, j1, k1, i2, j2, k2, bin;
        double dx, dy, dz, r, fac;
        if (!smp.Draw(m, t, q % nd_max, i1, j1, k1, i2, j2, k2, dx, dy, dz, r, bin,
                      fac)) {
          return;
        }
        double v1[kSFMaxFields], v2[kSFMaxFields];
        for (int n = 0; n < nfield; ++n) {
          v1[n] = fields.Get(n, m, k1 + ks, j1 + js, i1 + is);
          v2[n] = fields.Get(n, m, k2 + ks, j2 + js, i2 + is);
        }
        SFAddPair(k, spec, w0, w1, bin, fac, v1, v2, dx, dy, dz, r);
      });
      SFFlushScratch(member, k, hist, g);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::SampleAcrossBlocks()
//  \brief E2: ordered pairs in different MeshBlocks. Every rank draws (per replicate)
//  anchors and samples uniformly from its region cells (sparse: its support cells)
//  with independent streams; the
//  samples are all-gathered (host-staged) and every local anchor is paired with every
//  sample of the same replicate. Factor (N_r N_s)/(n_rt m_st); same-block pairs (E1's
//  population, including self-pairs) are skipped.

void SFOutput::SampleAcrossBlocks(Mesh *pm) {
  const int nranks = global_variable::nranks;
  const int myrank = global_variable::my_rank;
  const int nfield = fields_.nfield;
  const int ncol = 4 + nfield;   // global i, j, k, stratum factor, fields...
  auto &mbi = pm->mb_indcs;
  const int is = mbi.is, js = mbi.js, ks = mbi.ks;
  const int nx1 = mbi.nx1, nx2 = mbi.nx2, nx3 = mbi.nx3;
  const int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &gid = pm->pmb_pack->pmb->mb_gid;
  const SFFieldSet fields = fields_;
  const auto box = box_;
  const auto boxoff = box_offset_;
  // population of the local draws: region cells, or the support cells when sparse
  const int64_t nreg = sparse_ ? nsupp_local_ : ncell_region_local_;
  const bool sparse = sparse_;
  const auto supp = supp_;
  const uint64_t seed = seed_, outnum = static_cast<uint64_t>(out_params.file_number);

  // local draws for one stream: records (count, ncol) and block gids (count)
  // global index of each MeshBlock's first cell (uniform mesh)
  DvceArray2D<int64_t> cell0("sf_cell0", std::max(nmb, 1), 3);
  {
    auto h_cell0 = Kokkos::create_mirror_view(cell0);
    auto &ms = pm->mesh_size;
    const double h[3] = {geom_.h0, geom_.h1, geom_.h2};
    for (int m = 0; m < nmb; ++m) {
      const double x0[3] = {size.h_view(m).x1min - ms.x1min,
                            size.h_view(m).x2min - ms.x2min,
                            size.h_view(m).x3min - ms.x3min};
      for (int d = 0; d < 3; ++d) h_cell0(m, d) = std::llround(x0[d]/h[d]);
    }
    Kokkos::deep_copy(cell0, h_cell0);
  }
  auto draw_local = [&](uint64_t stream, const std::vector<int64_t> &count_t,
                        DvceArray2D<double> &rec, DvceArray1D<int64_t> &rgid) {
    int64_t ntot = 0;
    std::vector<int64_t> off(nrep_ + 1, 0);
    for (int t = 0; t < nrep_; ++t) {off[t+1] = off[t] + count_t[t];}
    ntot = off[nrep_];
    Kokkos::realloc(rec, std::max<int64_t>(ntot, 1), ncol);
    Kokkos::realloc(rgid, std::max<int64_t>(ntot, 1));
    if (ntot == 0 || nreg == 0) return;
    DvceArray1D<int64_t> d_off("sf_off", nrep_ + 1);
    auto h_off = Kokkos::create_mirror_view(d_off);
    for (int t = 0; t <= nrep_; ++t) h_off(t) = off[t];
    Kokkos::deep_copy(d_off, h_off);
    DvceArray1D<double> d_fac("sf_fac", nrep_);
    auto h_fac = Kokkos::create_mirror_view(d_fac);
    for (int t = 0; t < nrep_; ++t) {
      h_fac(t) = (count_t[t] > 0) ?
                 static_cast<double>(nreg)/static_cast<double>(count_t[t]) : 0.0;
    }
    Kokkos::deep_copy(d_fac, h_fac);
    const int nrep = nrep_;
    auto rec_ = rec;
    auto rgid_ = rgid;
    Kokkos::parallel_for("sf_e2_draw", Kokkos::RangePolicy<DevExeSpace, int64_t>(0, ntot),
    KOKKOS_LAMBDA(const int64_t idx) {
      int t = 0;
      while (t < nrep - 1 && idx >= d_off(t+1)) ++t;
      const int64_t dd = idx - d_off(t);
      const uint64_t h = SFKey(seed, outnum, stream, static_cast<uint64_t>(myrank), t,
                               dd);
      int64_t u = static_cast<int64_t>(h % static_cast<uint64_t>(nreg));
      if (sparse) u = supp(u);
      int m, i, j, k;
      SFDecodeRegion(boxoff, box, nmb, u, m, i, j, k);
      // global cell indices, exact in double (below 2^53)
      rec_(idx, 0) = static_cast<double>(cell0(m, 0) + i);
      rec_(idx, 1) = static_cast<double>(cell0(m, 1) + j);
      rec_(idx, 2) = static_cast<double>(cell0(m, 2) + k);
      rec_(idx, 3) = d_fac(t);
      for (int n = 0; n < fields.nfield; ++n) {
        rec_(idx, 4 + n) = static_cast<double>(fields.Get(n, m, k + ks, j + js, i + is));
      }
      rgid_(idx) = static_cast<int64_t>(gid.d_view(m));
    });
  };

  // ---- anchors (local only) ----
  std::vector<int64_t> nanc_t(nrep_), nsam_t(nrep_);
  for (int t = 0; t < nrep_; ++t) {
    nanc_t[t] = nanchor_rt_[myrank][t];
    nsam_t[t] = nsample_rt_[myrank][t];
  }
  DvceArray2D<double> anc("sf_anc", 1, 1);
  DvceArray1D<int64_t> anc_gid("sf_anc_gid", 1);
  draw_local(kStreamE2Anchor, nanc_t, anc, anc_gid);

  // ---- samples: local draws, then host-staged all-gather ordered (rank, rep) ----
  DvceArray2D<double> sam_loc("sf_sam_loc", 1, 1);
  DvceArray1D<int64_t> sam_loc_gid("sf_sam_loc_gid", 1);
  draw_local(kStreamE2Sample, nsam_t, sam_loc, sam_loc_gid);
  Kokkos::fence();

  std::vector<int64_t> nsam_rank(nranks, 0);
  for (int s = 0; s < nranks; ++s) {
    for (int t = 0; t < nrep_; ++t) nsam_rank[s] += nsample_rt_[s][t];
  }
  int64_t nsam_all = std::accumulate(nsam_rank.begin(), nsam_rank.end(), int64_t{0});
  HostArray2D<double> h_all("sf_h_all", std::max<int64_t>(nsam_all, 1), ncol);
  HostArray1D<int64_t> h_all_gid("sf_h_all_gid", std::max<int64_t>(nsam_all, 1));
  {
    auto h_loc = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sam_loc);
    auto h_loc_gid = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sam_loc_gid);
#if MPI_PARALLEL_ENABLED
    std::vector<int> cnt_d(nranks), dsp_d(nranks), cnt_i(nranks), dsp_i(nranks);
    int64_t run = 0;
    for (int s = 0; s < nranks; ++s) {
      int64_t cd = nsam_rank[s]*ncol;
      if (cd > INT32_MAX || (run + nsam_rank[s])*ncol > INT32_MAX) {
        SFFatal(out_params.block_name, "nsample_global too large for a single "
                "MPI_Allgatherv (int counts); reduce it");
      }
      cnt_d[s] = static_cast<int>(cd);
      dsp_d[s] = static_cast<int>(run*ncol);
      cnt_i[s] = static_cast<int>(nsam_rank[s]);
      dsp_i[s] = static_cast<int>(run);
      run += nsam_rank[s];
    }
    MPI_Allgatherv(h_loc.data(), cnt_d[myrank], MPI_DOUBLE, h_all.data(),
                   cnt_d.data(), dsp_d.data(), MPI_DOUBLE, comm_);
    MPI_Allgatherv(h_loc_gid.data(), cnt_i[myrank], MPI_INT64_T, h_all_gid.data(),
                   cnt_i.data(), dsp_i.data(), MPI_INT64_T, comm_);
#else
    for (int64_t n = 0; n < nsam_all; ++n) {
      for (int c = 0; c < ncol; ++c) h_all(n, c) = h_loc(n, c);
      h_all_gid(n) = h_loc_gid(n);
    }
#endif
  }
  // reorder to replicate-major; stratum factor N_s/m_st already in column 3
  HostArray2D<double> h_sam("sf_h_sam", std::max<int64_t>(nsam_all, 1), ncol);
  HostArray1D<int64_t> h_sam_gid("sf_h_sam_gid", std::max<int64_t>(nsam_all, 1));
  std::vector<int64_t> sam_off(nrep_ + 1, 0);
  {
    std::vector<int64_t> rank_start(nranks + 1, 0);
    for (int s = 0; s < nranks; ++s) rank_start[s+1] = rank_start[s] + nsam_rank[s];
    int64_t w = 0;
    for (int t = 0; t < nrep_; ++t) {
      sam_off[t] = w;
      for (int s = 0; s < nranks; ++s) {
        int64_t src = rank_start[s];
        for (int tt = 0; tt < t; ++tt) src += nsample_rt_[s][tt];
        for (int64_t n = 0; n < nsample_rt_[s][t]; ++n, ++w) {
          for (int c = 0; c < ncol; ++c) h_sam(w, c) = h_all(src + n, c);
          h_sam_gid(w) = h_all_gid(src + n);
        }
      }
    }
    sam_off[nrep_] = w;
  }
  auto sam = Kokkos::create_mirror_view_and_copy(DevMemSpace(), h_sam);
  auto sam_gid = Kokkos::create_mirror_view_and_copy(DevMemSpace(), h_sam_gid);

  // ---- pair kernels, one replicate at a time (same kernel variants as E1) ----
  SFPairSpec spec = SFMakeSpec(nq_, max_order_, nbin_, mode_, nvar_, weight_field_);
  SFSetHist(&spec, hist_, hist_ncol_, hist_nedge_, hist_edges_);
  SFSetAniso(&spec, nbin_l_, nbin_prll_, nbin_perp_, theta_prll_max_, theta_perp_min_,
             rbins_prll_, rbins_perp_);
  SFGlobalSink gsink{acc_T_, acc_W_, acc_C_, acc_n_, 0, nweight_, nq_, max_order_,
                     nbin_, acc_H_, hist_ncol_};
  const int nrep = nrep_, nw = nweight_;
  const SFGeom geom = geom_;
  const auto rbins = rbins_;
  const int nbin = nbin_l_;
  const bool keep_over = (mode_ == 2);
  int64_t anc_start = 0;
  uint64_t natt = 0;
  for (int t = 0; t < nrep; ++t) {
    const int64_t na = nanc_t[t];
    const int64_t ns = sam_off[t+1] - sam_off[t];
    const int64_t a0 = anc_start, s0 = sam_off[t];
    anc_start += na;
    if (na == 0 || ns == 0) continue;
    natt += static_cast<uint64_t>(na)*static_cast<uint64_t>(ns);
    const int64_t npair = na*ns;
    gsink.er = nrep + t;
    if (nw_pass_ == 0) {
      Kokkos::parallel_for("sf_e2_global",
      Kokkos::RangePolicy<DevExeSpace, int64_t>(0, npair),
      KOKKOS_LAMBDA(const int64_t idx) {
        const int64_t ia = a0 + idx/ns;
        const int64_t jb = s0 + idx % ns;
        if (anc_gid(ia) == sam_gid(jb)) return;   // same block: E1 population
        double dx, dy, dz;
        const double r = geom.Separation(SFCellOffset(sam(jb, 0), anc(ia, 0)),
                                         SFCellOffset(sam(jb, 1), anc(ia, 1)),
                                         SFCellOffset(sam(jb, 2), anc(ia, 2)),
                                         dx, dy, dz);
        const int bin = SFBinIndex(rbins, nbin, r);
        if (bin < 0 || (bin >= nbin && !keep_over)) return;
        SFAddPair(gsink, spec, 0, nw, bin, 0.5*anc(ia, 3)*sam(jb, 3), &anc(ia, 4),
                  &sam(jb, 4), dx, dy, dz, r);
      });
      continue;
    }
    const int64_t chunk = SFTeamChunk(npair, nbin_);
    const int64_t nchunk = (npair + chunk - 1)/chunk;
    for (int w0 = 0; w0 < nw; w0 += nw_pass_) {
      const int w1 = std::min(nw, w0 + nw_pass_);
      const SFScratchHist hist(w1 - w0, spec);
      Kokkos::TeamPolicy<> policy(DevExeSpace(), static_cast<int>(nchunk), Kokkos::AUTO);
      policy.set_scratch_size(0, Kokkos::PerTeam(hist.Bytes()));
      Kokkos::parallel_for("sf_e2_team", policy,
      KOKKOS_LAMBDA(const TeamMember_t &member) {
        const int64_t p0 = member.league_rank()*chunk;
        const int64_t p1 = (p0 + chunk < npair) ? p0 + chunk : npair;
        const SFScratchSink k = SFMakeScratchSink(member, hist, spec, w0, gsink);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, p0, p1),
        [&](const int64_t idx) {
          const int64_t ia = a0 + idx/ns;
          const int64_t jb = s0 + idx % ns;
          if (anc_gid(ia) == sam_gid(jb)) return;   // same block: E1 population
          double dx, dy, dz;
          const double r = geom.Separation(SFCellOffset(sam(jb, 0), anc(ia, 0)),
                                           SFCellOffset(sam(jb, 1), anc(ia, 1)),
                                           SFCellOffset(sam(jb, 2), anc(ia, 2)),
                                           dx, dy, dz);
          const int bin = SFBinIndex(rbins, nbin, r);
          if (bin < 0 || (bin >= nbin && !keep_over)) return;
          SFAddPair(k, spec, w0, w1, bin, 0.5*anc(ia, 3)*sam(jb, 3), &anc(ia, 4),
                    &sam(jb, 4), dx, dy, dz, r);
        });
        SFFlushScratch(member, k, hist, gsink);
      });
    }
  }
  attempts_[1] = natt;
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::BuildHistEdges()
//  \brief increment-histogram edges of this output: the template times, per (weight,
//  q), the weighted standard deviation over the valid cells (region, weight > 0), as
//  structure_functions.py masked_increment_scales but with a centered two-pass variance,
//  rounded to 12 significant digits. long_trans rows use the 1D velocity dispersion
//  sqrt((s_x^2 + s_y^2 + s_z^2)/3) of the raw component variances. The final scale falls
//  back to 1 only for exactly constant data (global min == max of every contributing
//  component), an empty selection or nonfinite data. With hist_scale = absolute the
//  template is used as is.

void SFOutput::BuildHistEdges(Mesh *pm) {
  const int nvar = nvar_, nw = nweight_;
  std::vector<double> sig(static_cast<std::size_t>(nw)*nvar, 1.0);
  std::vector<double> varies(static_cast<std::size_t>(nw)*nvar, 0.0);
  if (hist_sigma_scale_) {
    auto &mbi = pm->mb_indcs;
    const int is = mbi.is, js = mbi.js, ks = mbi.ks;
    const int nmb = pm->pmb_pack->nmb_thispack;
    const auto box = box_;
    const SFFieldSet fields = fields_;
    // centered two-pass weighted variance: pass 0 gives the weighted mean, pass 1
    // sum w (q - mean)^2 (no cancellation for a large mean, e.g. a uniform boost)
    const std::size_t nr = static_cast<std::size_t>(nw)*nvar;
    std::vector<double> sum_w(nr, 0.0), mean(nr, 0.0), sum_c(nr, 0.0);
    // exact constancy test: global min and max of q over the valid cells (the two-pass
    // variance of a constant field is round-off, not 0, once the mean is a parallel sum)
    std::vector<double> qmin(nr, 0.0), qmax(nr, 0.0);
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<double> buf(2*nr, 0.0);
      for (int iw = 0; iw < nw; ++iw) {
        const int wf = weight_field_[iw];
        for (int n = 0; n < nvar; ++n) {
          const std::size_t r = static_cast<std::size_t>(iw)*nvar + n;
          const double mu = mean[r];
          double a0 = 0.0, a1 = 0.0;
          Kokkos::parallel_reduce("sf_hist_sigma",
          Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
              {nmb, mbi.nx3, mbi.nx2, mbi.nx1}),
          KOKKOS_LAMBDA(int m, int k, int j, int i, double &b0, double &b1) {
            if (i < box(m, 0) || i > box(m, 1) || j < box(m, 2) || j > box(m, 3) ||
                k < box(m, 4) || k > box(m, 5)) return;
            const double w = (wf < 0) ? 1.0 :
                static_cast<double>(fields.Get(wf, m, k + ks, j + js, i + is));
            if (!(w > 0.0)) return;
            const double q = static_cast<double>(fields.Get(n, m, k+ks, j+js, i+is));
            b0 += w;
            b1 += (pass == 0) ? w*q : w*(q - mu)*(q - mu);
          }, a0, a1);
          buf[2*r] = a0;
          buf[2*r + 1] = a1;
          if (pass == 0) {
            double lo = DBL_MAX, hi = -DBL_MAX;
            Kokkos::parallel_reduce("sf_hist_range",
            Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                {nmb, mbi.nx3, mbi.nx2, mbi.nx1}),
            KOKKOS_LAMBDA(int m, int k, int j, int i, double &l, double &h) {
              if (i < box(m, 0) || i > box(m, 1) || j < box(m, 2) || j > box(m, 3) ||
                  k < box(m, 4) || k > box(m, 5)) return;
              const double w = (wf < 0) ? 1.0 :
                  static_cast<double>(fields.Get(wf, m, k + ks, j + js, i + is));
              if (!(w > 0.0)) return;
              const double q = static_cast<double>(fields.Get(n, m, k+ks, j+js, i+is));
              l = (q < l) ? q : l;
              h = (q > h) ? q : h;
            }, Kokkos::Min<double>(lo), Kokkos::Max<double>(hi));
            qmin[r] = lo;
            qmax[r] = hi;
          }
        }
      }
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE,
                    MPI_SUM, comm_);
      if (pass == 0) {
        MPI_Allreduce(MPI_IN_PLACE, qmin.data(), static_cast<int>(nr), MPI_DOUBLE,
                      MPI_MIN, comm_);
        MPI_Allreduce(MPI_IN_PLACE, qmax.data(), static_cast<int>(nr), MPI_DOUBLE,
                      MPI_MAX, comm_);
      }
#endif
      for (std::size_t r = 0; r < nr; ++r) {
        sum_w[r] = buf[2*r];
        if (pass == 0) {
          mean[r] = (sum_w[r] > 0.0) ? buf[2*r + 1]/sum_w[r] : 0.0;
        } else {
          sum_c[r] = buf[2*r + 1];
        }
      }
    }
    for (std::size_t r = 0; r < nr; ++r) {
      sig[r] = (sum_w[r] > 0.0) ? sum_c[r]/sum_w[r] : 0.0;   // variance for now
      varies[r] = (qmax[r] > qmin[r]) ? 1.0 : 0.0;
    }
  }
  // scale per (weight, q): the standard deviation (long_trans: of the three components
  // combined, sqrt of their mean variance), rounded to 12 significant digits as Python;
  // the fallback 1 applies only to the final scale (constant field, empty selection or
  // nonfinite data), never to a single component
  hist_sigma_.assign(static_cast<std::size_t>(nw)*nq_, 1.0);
  for (int iw = 0; iw < nw; ++iw) {
    for (int iq = 0; iq < nq_; ++iq) {
      double sg = 1.0;
      if (hist_sigma_scale_) {
        double var, ms;   // ms > 0: some contributing component varies
        if (mode_ == 0) {
          var = sig[static_cast<std::size_t>(iw)*nvar + iq];
          ms = varies[static_cast<std::size_t>(iw)*nvar + iq];
        } else {
          var = ms = 0.0;
          for (int c = 0; c < 3; ++c) {
            var += sig[static_cast<std::size_t>(iw)*nvar + c];
            ms += varies[static_cast<std::size_t>(iw)*nvar + c];
          }
          var /= 3.0;
          ms /= 3.0;
        }
        char b[40];
        std::snprintf(b, sizeof(b), "%.12g", std::sqrt(std::max(var, 0.0)));
        sg = std::atof(b);
        // constant data (no contributing component varies) has no scale; any resolved
        // variation is kept however small relative to the mean
        if (!(std::isfinite(sg) && sg > 0.0 && ms > 0.0)) sg = 1.0;
      }
      hist_sigma_[static_cast<std::size_t>(iw)*nq_ + iq] = sg;
      const std::vector<double> &tm = hist_tmpl_[(mode_ == 1 && iq == 0) ? 1 : 0];
      for (int e = 0; e < hist_ncol_ - 1; ++e) {
        h_hist_edges_(iw*nq_ + iq, e) =
            (e < static_cast<int>(tm.size())) ? tm[e]*sg : 0.0;
      }
    }
  }
  Kokkos::deep_copy(hist_edges_, h_hist_edges_);
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::ReduceAccumulators()
//  \brief host-staged sum of all accumulators to rank 0 (device-pointer collectives are
//  not relied on; see pdf.cpp)

void SFOutput::ReduceAccumulators() {
  Kokkos::deep_copy(h_T_, acc_T_);
  Kokkos::deep_copy(h_W_, acc_W_);
  Kokkos::deep_copy(h_C_, acc_C_);
  Kokkos::deep_copy(h_n_, acc_n_);
  if (hist_) Kokkos::deep_copy(h_H_, acc_H_);
#if MPI_PARALLEL_ENABLED
  bool root = (global_variable::my_rank == 0);
  // counts are chunked below INT_MAX (histograms can be large)
  constexpr std::size_t kChunk = std::size_t{1} << 28;
  auto reduce_d = [&](double *p, std::size_t n) {
    for (std::size_t o = 0; o < n; o += kChunk) {
      const int c = static_cast<int>(std::min(kChunk, n - o));
      MPI_Reduce(root ? MPI_IN_PLACE : p + o, p + o, c, MPI_DOUBLE, MPI_SUM, 0, comm_);
    }
  };
  reduce_d(h_T_.data(), h_T_.size());
  reduce_d(h_W_.data(), h_W_.size());
  reduce_d(h_C_.data(), h_C_.size());
  if (hist_) reduce_d(h_H_.data(), h_H_.size());
  for (std::size_t o = 0; o < h_n_.size(); o += kChunk) {
    const int c = static_cast<int>(std::min(kChunk, h_n_.size() - o));
    MPI_Reduce(root ? MPI_IN_PLACE : h_n_.data() + o, h_n_.data() + o, c, MPI_UINT64_T,
               MPI_SUM, 0, comm_);
  }
  MPI_Reduce(root ? MPI_IN_PLACE : attempts_, attempts_, 2, MPI_UINT64_T, MPI_SUM, 0,
             comm_);
  double tloc[3] = {time_e1_, time_e2_, time_reduce_}, tmax[3], tmin[3];
  MPI_Reduce(tloc, tmax, 3, MPI_DOUBLE, MPI_MAX, 0, comm_);
  MPI_Reduce(tloc, tmin, 3, MPI_DOUBLE, MPI_MIN, 0, comm_);
  if (timing_ && root) {
    std::printf("sf[%s] output %d: E1 %.4e..%.4e s, E2 %.4e..%.4e s (min..max over "
                "ranks)\n", out_params.file_id.c_str(), out_params.file_number,
                tmin[0], tmax[0], tmin[1], tmax[1]);
  }
#else
  if (timing_) {
    std::printf("sf[%s] output %d: E1 %.4e s, E2 %.4e s\n", out_params.file_id.c_str(),
                out_params.file_number, time_e1_, time_e2_);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void SFOutput::WriteOutputFile()
//  \brief rank 0 writes sf_<id>/<basename>.NNNNN.sf: ASCII key=value header, then
//  float64/uint64 arrays in the order listed under 'payload' in the header.

void SFOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  if (global_variable::my_rank == 0) {
    const std::size_t nTrep = static_cast<std::size_t>(nweight_)*nq_*max_order_*nbin_;
    const std::size_t nWrep = static_cast<std::size_t>(nweight_)*nbin_;
    // replicate average of (E1 + E2)
    std::vector<double> T(nTrep, 0.0), Wc(nWrep, 0.0), Cc(nWrep, 0.0);
    for (int e = 0; e < 2; ++e) {
      for (int t = 0; t < nrep_; ++t) {
        std::size_t oT = (static_cast<std::size_t>(e)*nrep_ + t)*nTrep;
        std::size_t oW = (static_cast<std::size_t>(e)*nrep_ + t)*nWrep;
        for (std::size_t n = 0; n < nTrep; ++n) T[n] += h_T_(oT + n)/nrep_;
        for (std::size_t n = 0; n < nWrep; ++n) {
          Wc[n] += h_W_(oW + n)/nrep_;
          Cc[n] += h_C_(oW + n)/nrep_;
        }
      }
    }
    auto h_rbins = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rbins_);

    char number[16];
    std::snprintf(number, sizeof(number), "%05d", out_params.file_number);
    std::string fname = "sf_" + out_params.file_id + "/" + out_params.file_basename +
                        "." + number + ".sf";
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "wb")) == nullptr) {
      SFFatal(out_params.block_name, "output file '" + fname + "' could not be opened");
    }
    auto join = [](const std::vector<std::string> &v) {
      std::string s;
      for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
      return s;
    };
    auto fmt = [](double x) {
      char b[40];
      std::snprintf(b, sizeof(b), "%.17g", x);
      return std::string(b);
    };
    int64_t ncell_total = 0;
    for (auto n : ncell_region_rank_) ncell_total += n;
    std::string meta;
    meta += "Athena sf format_version=1\n";
    meta += "proposal_version=1\n";
    meta += "time=" + fmt(pm->time) + "\n";
    meta += "cycle=" + std::to_string(pm->ncycle) + "\n";
    meta += "file_number=" + std::to_string(out_params.file_number) + "\n";
    const char *mode_name[4] = {"scalar", "long_trans", "aniso_mhd", "correlation"};
    meta += "mode=" + std::string(mode_name[mode_]) + "\n";
    meta += "variables=" + join(var_names_) + "\n";
    meta += "weights=" + join(weight_names_) + "\n";
    meta += "nvar=" + std::to_string(nvar_) + "\n";
    meta += "nweight=" + std::to_string(nweight_) + "\n";
    meta += "nq=" + std::to_string(nq_) + "\n";
    meta += "max_order=" + std::to_string(max_order_) + "\n";
    meta += "nbin=" + std::to_string(nbin_) + "\n";
    if (mode_ == 3) {
      meta += "products=w1*w2*((A1*B2)^p+(B1*A2)^p),W=2*w1*w2,A=" + var_names_[0] +
              ",B=" + var_names_[nvar_ - 1] + "\n";
    }
    if (mode_ == 2) {
      meta += "increments=dv,dv_perp,dB,dB_perp\n";
      meta += "bin_families=l,l_prll,l_perp\n";
      meta += "nbin_l=" + std::to_string(nbin_l_) + "\n";
      meta += "nbin_prll=" + std::to_string(nbin_prll_) + "\n";
      meta += "nbin_perp=" + std::to_string(nbin_perp_) + "\n";
      meta += "theta_prll_max=" + fmt(theta_prll_max_) + "\n";
      meta += "theta_perp_min=" + fmt(theta_perp_min_) + "\n";
    }
    meta += "logscale=" + std::to_string(logscale_ ? 1 : 0) + "\n";
    meta += "r_min=" + fmt(r_min_) + "\n";
    meta += "r_max=" + fmt(r_max_) + "\n";
    meta += "periodic=" + std::to_string(periodic_[0]) + "," +
            std::to_string(periodic_[1]) + "," + std::to_string(periodic_[2]) + "\n";
    meta += "domain_length=" + fmt(len_[0]) + "," + fmt(len_[1]) + "," + fmt(len_[2]) +
            "\n";
    meta += "region=" + fmt(out_params.x1_min) + "," + fmt(out_params.x1_max) + "," +
            fmt(out_params.x2_min) + "," + fmt(out_params.x2_max) + "," +
            fmt(out_params.x3_min) + "," + fmt(out_params.x3_max) + "\n";
    meta += "ncell_region=" + std::to_string(ncell_total) + "\n";
    meta += "sparse=" + std::to_string(sparse_ ? 1 : 0) + "\n";
    if (sparse_) {
      int64_t nsupp_total = 0;
      for (auto n : nsupp_rank_) nsupp_total += n;
      meta += "ncell_support=" + std::to_string(nsupp_total) + "\n";
      int nrank_supp = 0;
      for (auto n : nsupp_rank_) nrank_supp += (n > 0) ? 1 : 0;
      meta += "nrank_support=" + std::to_string(nrank_supp) + "\n";
    }
    if (select_.nsel > 0) meta += "nselect=" + std::to_string(select_.nsel) + "\n";
    for (int c = 0; c < select_.nsel; ++c) {
      meta += "select" + std::to_string(c + 1) + "=" + select_names_[c] + "," +
              fmt(select_.lo[c]) + "," + fmt(select_.hi[c]) + "\n";
    }
    if (select_.nsel > 0) meta += "ncell_selected=" + std::to_string(nsel_total_) + "\n";
    meta += "mesh_nx=" + std::to_string(pm->mesh_indcs.nx1) + "," +
            std::to_string(pm->mesh_indcs.nx2) + "," +
            std::to_string(pm->mesh_indcs.nx3) + "\n";
    meta += "meshblock_nx=" + std::to_string(pm->mb_indcs.nx1) + "," +
            std::to_string(pm->mb_indcs.nx2) + "," + std::to_string(pm->mb_indcs.nx3) +
            "\n";
    meta += "mesh_min=" + fmt(pm->mesh_size.x1min) + "," + fmt(pm->mesh_size.x2min) +
            "," + fmt(pm->mesh_size.x3min) + "\n";
    meta += "nranks=" + std::to_string(global_variable::nranks) + "\n";
    meta += "nreplicate=" + std::to_string(nrep_) + "\n";
    meta += "ndraw_block=" + std::to_string(ndraw_block_) + "\n";
    meta += "nsample_global=" + std::to_string(nsample_global_) + "\n";
    meta += "nanchor_global=" + std::to_string(nanchor_global_) + "\n";
    meta += "seed=" + std::to_string(seed_) + "\n";
    meta += "halo_requested=" + std::to_string(halo_requested_) + "\n";
    meta += "halo_effective=" + std::to_string(halo_effective_) + "\n";
    meta += "kernel=" + (nw_pass_ > 0 ? "team,weights_per_pass=" +
                         std::to_string(nw_pass_) : std::string("global")) + "\n";
    meta += "e1_lag_distribution=chebyshev_shell_p_c_proportional_to_1_over_c\n";
    meta += "attempts_e1=" + std::to_string(attempts_[0]) + "\n";
    meta += "attempts_e2=" + std::to_string(attempts_[1]) + "\n";
    meta += "time_e1=" + fmt(time_e1_) + "\n";
    meta += "time_e2=" + fmt(time_e2_) + "\n";
    if (sparse_) meta += "time_support=" + fmt(time_support_) + "\n";
    if (select_.nsel > 0) meta += "time_select=" + fmt(time_select_) + "\n";
    const uint16_t one = 1;
    const bool little = (*reinterpret_cast<const unsigned char *>(&one) == 1);
    meta += "dtype=float64 except n_acc uint64\n";
    meta += std::string("byte_order=") + (little ? "little" : "big") + "\n";
    meta += std::string(mode_ == 2 ?
                        "payload=rbins[nbin_l+1],rbins_prll[nbin_prll+1],"
                        "rbins_perp[nbin_perp+1]," : "payload=rbins[nbin+1],") +
            "T[nweight,nq,max_order,nbin],W[nweight,nbin],"
            "C[nweight,nbin],T_er[2,nreplicate,nweight,nq,max_order,nbin],"
            "W_er[2,nreplicate,nweight,nbin],C_er[2,nreplicate,nweight,nbin],"
            "n_acc_er[2,nreplicate,nweight,nbin]" +
            std::string(hist_ ?
                        ",hist_edges[nweight,nq,hist_ncol-1],hist_sigma[nweight,nq],"
                        "H[nweight,nq,nbin,hist_ncol],"
                        "H_er[2,nreplicate,nweight,nq,nbin,hist_ncol]" : "") + "\n";
    meta += "histogram=" + std::to_string(hist_ ? 1 : 0) + "\n";
    if (hist_) {
      std::string ne;
      for (int iq = 0; iq < nq_; ++iq) {
        ne += (iq ? "," : "") + std::to_string(hist_nedge_[iq]);
      }
      meta += "hist_ncol=" + std::to_string(hist_ncol_) + "\n";
      meta += "hist_nedge=" + ne + "\n";
      meta += "hist_template=log10_" + fmt(hist_min_) + "_" + fmt(hist_max_) + "_nbin" +
              std::to_string(hist_nbin_) + (mode_ == 1 ? "_q0_signed" : "") + "\n";
      meta += "hist_scale=" + std::string(hist_sigma_scale_ ? "sigma" : "absolute") +
              "\n";
      meta += "hist_columns=0_underflow,k_edges[k-1]<=v<edges[k],nedge_overflow;"
              "importance_weighted_pair_counts\n";
    }
    meta += "estimator_index=0:E1_within_block,1:E2_cross_block\n";
    // every write is counted; a short write or stream error is fatal before the output
    // counter advances (rank 0 writes; its exit aborts the MPI job)
    bool wok = SFWriteHeader(pfile, meta);
    auto put = [&](const void *ptr, std::size_t size, std::size_t n) {
      wok &= (std::fwrite(ptr, size, n, pfile) == n);
    };
    put(h_rbins.data(), sizeof(double), nbin_l_ + 1);
    if (mode_ == 2) {
      auto hp = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rbins_prll_);
      auto hq = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rbins_perp_);
      put(hp.data(), sizeof(double), nbin_prll_ + 1);
      put(hq.data(), sizeof(double), nbin_perp_ + 1);
    }
    put(T.data(), sizeof(double), T.size());
    put(Wc.data(), sizeof(double), Wc.size());
    put(Cc.data(), sizeof(double), Cc.size());
    put(h_T_.data(), sizeof(double), h_T_.size());
    put(h_W_.data(), sizeof(double), h_W_.size());
    put(h_C_.data(), sizeof(double), h_C_.size());
    put(h_n_.data(), sizeof(uint64_t), h_n_.size());
    if (hist_) {
      const std::size_t nHrep = static_cast<std::size_t>(nweight_)*nq_*nbin_*hist_ncol_;
      std::vector<double> H(nHrep, 0.0);
      for (int e = 0; e < 2; ++e) {
        for (int t = 0; t < nrep_; ++t) {
          std::size_t o = (static_cast<std::size_t>(e)*nrep_ + t)*nHrep;
          for (std::size_t n = 0; n < nHrep; ++n) H[n] += h_H_(o + n)/nrep_;
        }
      }
      put(h_hist_edges_.data(), sizeof(double), h_hist_edges_.size());
      put(hist_sigma_.data(), sizeof(double), hist_sigma_.size());
      put(H.data(), sizeof(double), H.size());
      put(h_H_.data(), sizeof(double), h_H_.size());
    }
    wok &= (std::ferror(pfile) == 0);
    if (std::fclose(pfile) != 0 || !wok) {
      SFFatal(out_params.block_name, "error writing or closing '" + fname + "'");
    }
  }

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
