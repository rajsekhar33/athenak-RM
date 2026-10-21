#ifndef MESH_PROLONGATION_HPP_
#define MESH_PROLONGATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prolongation.hpp
//! \brief prolongation operators for cell-centered and face-centered variables,
//! implemented as inline functions so they can be used both in Bval and AMR functions.

#include <limits>

#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "z4c/z4c.hpp"

// Shared ideal-Hydro conservative sibling policy for ghosts and active AMR.
// Callers select Newtonian ideal Hydro with viscosity STS; not MHD/CT.
KOKKOS_INLINE_FUNCTION
bool HydroProlongAdmissible(const Real u[5], const EOS_Data &eos) {
  const Real largest = std::numeric_limits<Real>::max();
  for (int v = 0; v < 5; ++v) {
    if (!(u[v] >= -largest && u[v] <= largest)) return false;
  }
  if (!(u[IDN] > 0.0)) return false;
  HydCons1D cell;
  cell.d = u[IDN]; cell.mx = u[IM1]; cell.my = u[IM2];
  cell.mz = u[IM3]; cell.e = u[IEN];
  const Real thermal = cell.e - 0.5*(1.0/cell.d)*
      (SQR(cell.mx) + SQR(cell.my) + SQR(cell.mz));
  if (!(thermal > 0.0 && thermal <= largest)) return false;
  HydPrim1D primitive;
  bool df = false, ef = false, tf = false;
  SingleC2P_IdealHyd(cell, eos, primitive, df, ef, tf);
  const Real temperature = (eos.gamma-1.0)*primitive.e/primitive.d;
  return !df && !ef && !tf && primitive.e > 0.0 && primitive.e <= largest &&
      temperature > 0.0 && temperature <= largest &&
      primitive.vx >= -largest && primitive.vx <= largest &&
      primitive.vy >= -largest && primitive.vy <= largest &&
      primitive.vz >= -largest && primitive.vz <= largest;
}

KOKKOS_INLINE_FUNCTION
bool HydroProlongSiblingsAdmissible(const Real parent[5], const Real child[8][5],
    const int nc, const Real theta, const EOS_Data &eos, const Real cushion = 0.0) {
  for (int c = 0; c < nc; ++c) {
    Real trial[5];
    for (int v = 0; v < 5; ++v) {
      trial[v] = theta == 1.0 ? child[c][v] :
          parent[v] + theta*(child[c][v]-parent[v]);
    }
    trial[IEN] -= cushion;
    if (!HydroProlongAdmissible(trial, eos)) return false;
  }
  return true;
}

// One common contraction preserves every sibling conserved mean. The exact
// production EOS is evaluated on copies: it must not invoke any of its floors.
// Admissible children take the theta=1 fast path without being rewritten.
KOKKOS_INLINE_FUNCTION
Real HydroProlongTheta(const Real parent[5], const Real child[8][5],
                    const int nc, const EOS_Data &eos) {
  if (!HydroProlongAdmissible(parent, eos)) return -1.0;
  const Real largest = std::numeric_limits<Real>::max();
  for (int c = 0; c < nc; ++c) {
    for (int v = 0; v < 5; ++v) {
      if (!(child[c][v] >= -largest && child[c][v] <= largest)) return -1.0;
    }
  }
  if (HydroProlongSiblingsAdmissible(parent, child, nc, 1.0, eos)) return 1.0;
  // Separate kernels/compilers may contract the kinetic-energy arithmetic
  // differently. Keep limited states several rounding units inside ALL EOS
  // constraints, not merely theta several epsilons below the root (which gives
  // no thermal margin when thermal energy is tiny relative to total energy).
  // Checking a lower-energy COPY is a stricter test, never an energy update.
  // Reduce the cushion if the valid parent itself is very near an EOS floor.
  Real cushion = 64.0*std::numeric_limits<Real>::epsilon()*fabs(parent[IEN]);
  Real safe_parent[5];
  for (int v = 0; v < 5; ++v) safe_parent[v] = parent[v];
  for (int iter = 0; iter < std::numeric_limits<Real>::digits+2; ++iter) {
    safe_parent[IEN] = parent[IEN]-cushion;
    if (HydroProlongAdmissible(safe_parent, eos)) break;
    cushion *= 0.5;
  }
  safe_parent[IEN] = parent[IEN]-cushion;
  if (!HydroProlongAdmissible(safe_parent, eos)) cushion = 0.0;
  Real lo = 0.0, hi = 1.0;
  for (int iter = 0; iter < std::numeric_limits<Real>::digits+2; ++iter) {
    const Real mid = 0.5*(lo+hi);
    if (HydroProlongSiblingsAdmissible(parent, child, nc, mid, eos, cushion)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  // Retreat from the roundoff-sensitive boundary, then recheck the very same
  // arithmetic used to write the cells. This changes slopes, never total energy
  // alone. The valid parent is the safe zero-slope endpoint.
  Real retreat = 128.0*std::numeric_limits<Real>::epsilon();
  for (int iter = 0; iter < std::numeric_limits<Real>::digits+2; ++iter) {
    const Real theta = lo*(1.0-retreat);
    if (HydroProlongSiblingsAdmissible(parent, child, nc, theta, eos, cushion)) return theta;
    // Floating-point admissibility is not exactly monotone at the root. Expand
    // the small retreat before using the zero-slope endpoint; an immediate
    // theta=0 fallback can create a macroscopic backend-dependent flux change.
    retreat = fmin(1.0, 2.0*retreat);
  }
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongCC()
//! \brief 2nd-order (piecewise-linear) prolongation operator for cell-centered variables

KOKKOS_INLINE_FUNCTION
void ProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi,
               const bool multi_d, const bool three_d,
               const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  // calculate x1-gradient using the min-mod limiter
  Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
  Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  // calculate x2-gradient using the min-mod limiter
  Real dvar2 = 0.0;
  if (multi_d) {
    dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
    dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // calculate x1-gradient using the min-mod limiter
  Real dvar3 = 0.0;
  if (three_d) {
    dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
    dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // interpolate to the finer grid
  a(m,v,fk,fj,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
  a(m,v,fk,fj,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
  if (multi_d) {
    a(m,v,fk,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
    a(m,v,fk,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
  }
  if (three_d) {
    a(m,v,fk+1,fj  ,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj  ,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX1Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X1-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX1Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d, const bool three_d,
                   const DvceArray4D<Real> &cbx1f, const DvceArray4D<Real> &bx1f) {
  // Prolongate b.x1f (v=0) by interpolating in x2/x3
  Real dvar2 = 0.0;
  if (multi_d) {
    Real dl = cbx1f(m,k,j  ,i) - cbx1f(m,k,j-1,i);
    Real dr = cbx1f(m,k,j+1,i) - cbx1f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  Real dvar3 = 0.0;
  if (three_d) {
    Real dl = cbx1f(m,k  ,j,i) - cbx1f(m,k-1,j,i);
    Real dr = cbx1f(m,k+1,j,i) - cbx1f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx1f(m,fk,fj,fi) = cbx1f(m,k,j,i) - dvar2 - dvar3;
  if (multi_d) {
    bx1f(m,fk,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 - dvar3;
  }
  if (three_d) {
    bx1f(m,fk+1,fj  ,fi) = cbx1f(m,k,j,i) - dvar2 + dvar3;
    bx1f(m,fk+1,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX2Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X2-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX2Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool three_d,
                   const DvceArray4D<Real> &cbx2f, const DvceArray4D<Real> &bx2f) {
  // Prolongate b.x2f (v=1) by interpolating in x1/x3
  Real dl = cbx2f(m,k,j,i  ) - cbx2f(m,k,j,i-1);
  Real dr = cbx2f(m,k,j,i+1) - cbx2f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar3 = 0.0;
  if (three_d) {
    dl = cbx2f(m,k  ,j,i) - cbx2f(m,k-1,j,i);
    dr = cbx2f(m,k+1,j,i) - cbx2f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx2f(m,fk  ,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 - dvar3;
  bx2f(m,fk  ,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 - dvar3;
  if (three_d) {
    bx2f(m,fk+1,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 + dvar3;
    bx2f(m,fk+1,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX3Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X3-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX3Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d,
                   const DvceArray4D<Real> &cbx3f, const DvceArray4D<Real> &bx3f) {
  // Prolongate b.x3f (v=2) by interpolating in x1/x2
  Real dl = cbx3f(m,k,j,i  ) - cbx3f(m,k,j,i-1);
  Real dr = cbx3f(m,k,j,i+1) - cbx3f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar2 = 0.0;
  if (multi_d) {
    dl = cbx3f(m,k,j  ,i) - cbx3f(m,k,j-1,i);
    dr = cbx3f(m,k,j+1,i) - cbx3f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx3f(m,fk,fj  ,fi  ) = cbx3f(m,k,j,i) - dvar1 - dvar2;
  bx3f(m,fk,fj  ,fi+1) = cbx3f(m,k,j,i) + dvar1 - dvar2;
  if (multi_d) {
    bx3f(m,fk,fj+1,fi  ) = cbx3f(m,k,j,i) - dvar1 + dvar2;
    bx3f(m,fk,fj+1,fi+1) = cbx3f(m,k,j,i) + dvar1 + dvar2;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongInternalFC()
//! \brief 2nd-order prolongation operator for face-centered variables on internal edges
//! of new fine cells within one coarse cell using divergence-preserving interpolation
//! scheme of Toth & Roe, JCP 180, 736 (2002).

KOKKOS_INLINE_FUNCTION
void ProlongFCInternal(const int m, const int fk, const int fj, const int fi,
                       const bool three_d, const DvceFaceFld4D<Real> &b) {
  // Prolongate internal fields in 3D
  if (three_d) {
    Real Uxx  = 0.0, Vyy  = 0.0, Wzz  = 0.0;
    Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
    for (int jj=0; jj<2; jj++) {
      int jsgn = 2*jj - 1;
      int fjj  = fj + jj, fjp = fj + 2*jj;
      for (int ii=0; ii<2; ii++) {
        int isgn = 2*ii - 1;
        int fii = fi + ii, fip = fi + 2*ii;
        Uxx += isgn*(jsgn*(b.x2f(m,fk  ,fjp,fii) + b.x2f(m,fk+1,fjp,fii)) +
                          (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)));

        Vyy += jsgn*(     (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)) +
                     isgn*(b.x1f(m,fk  ,fjj,fip) + b.x1f(m,fk+1,fjj,fip)));

        Wzz +=       isgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip)) +
                     jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));

        Uxyz += isgn*jsgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip));
        Vxyz += isgn*jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));
        Wxyz += isgn*jsgn*(b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii));
      }
    }
    Uxx *= 0.125;  Vyy *= 0.125;  Wzz *= 0.125;
    Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

    b.x1f(m,fk  ,fj  ,fi+1) = 0.5*(b.x1f(m,fk  ,fj  ,fi  ) + b.x1f(m,fk  ,fj  ,fi+2))
                            + Uxx - Vxyz - Wxyz;
    b.x1f(m,fk  ,fj+1,fi+1) = 0.5*(b.x1f(m,fk  ,fj+1,fi  ) + b.x1f(m,fk  ,fj+1,fi+2))
                            + Uxx - Vxyz + Wxyz;
    b.x1f(m,fk+1,fj  ,fi+1) = 0.5*(b.x1f(m,fk+1,fj  ,fi  ) + b.x1f(m,fk+1,fj  ,fi+2))
                            + Uxx + Vxyz - Wxyz;
    b.x1f(m,fk+1,fj+1,fi+1) = 0.5*(b.x1f(m,fk+1,fj+1,fi  ) + b.x1f(m,fk+1,fj+1,fi+2))
                            + Uxx + Vxyz + Wxyz;
    b.x2f(m,fk  ,fj+1,fi  ) = 0.5*(b.x2f(m,fk  ,fj  ,fi  ) + b.x2f(m,fk  ,fj+2,fi  ))
                            + Vyy - Uxyz - Wxyz;
    b.x2f(m,fk  ,fj+1,fi+1) = 0.5*(b.x2f(m,fk  ,fj  ,fi+1) + b.x2f(m,fk  ,fj+2,fi+1))
                            + Vyy - Uxyz + Wxyz;
    b.x2f(m,fk+1,fj+1,fi  ) = 0.5*(b.x2f(m,fk+1,fj  ,fi  ) + b.x2f(m,fk+1,fj+2,fi  ))
                            + Vyy + Uxyz - Wxyz;
    b.x2f(m,fk+1,fj+1,fi+1) = 0.5*(b.x2f(m,fk+1,fj  ,fi+1) + b.x2f(m,fk+1,fj+2,fi+1))
                            + Vyy + Uxyz + Wxyz;
    b.x3f(m,fk+1,fj  ,fi  ) = 0.5*(b.x3f(m,fk+2,fj  ,fi  ) + b.x3f(m,fk  ,fj  ,fi  ))
                            + Wzz - Uxyz - Vxyz;
    b.x3f(m,fk+1,fj  ,fi+1) = 0.5*(b.x3f(m,fk+2,fj  ,fi+1) + b.x3f(m,fk  ,fj  ,fi+1))
                            + Wzz - Uxyz + Vxyz;
    b.x3f(m,fk+1,fj+1,fi  ) = 0.5*(b.x3f(m,fk+2,fj+1,fi  ) + b.x3f(m,fk  ,fj+1,fi  ))
                            + Wzz + Uxyz - Vxyz;
    b.x3f(m,fk+1,fj+1,fi+1) = 0.5*(b.x3f(m,fk+2,fj+1,fi+1) + b.x3f(m,fk  ,fj+1,fi+1))
                            + Wzz + Uxyz + Vxyz;

  // Prolongate internal fields in 2D
  } else {
    Real tmp1 = 0.25*(b.x2f(m,fk,fj+2,fi+1) - b.x2f(m,fk,fj,  fi+1)
                    - b.x2f(m,fk,fj+2,fi  ) + b.x2f(m,fk,fj,  fi  ));
    Real tmp2 = 0.25*(b.x1f(m,fk,fj,  fi  ) - b.x1f(m,fk,fj,  fi+2)
                    - b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2));
    b.x1f(m,fk,fj  ,fi+1) = 0.5*(b.x1f(m,fk,fj,  fi  ) + b.x1f(m,fk,fj,  fi+2)) + tmp1;
    b.x1f(m,fk,fj+1,fi+1) = 0.5*(b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2)) + tmp1;
    b.x2f(m,fk,fj+1,fi  ) = 0.5*(b.x2f(m,fk,fj,  fi  ) + b.x2f(m,fk,fj+2,fi  )) + tmp2;
    b.x2f(m,fk,fj+1,fi+1) = 0.5*(b.x2f(m,fk,fj,  fi+1) + b.x2f(m,fk,fj+2,fi+1)) + tmp2;
  }
  return;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real ProlongInterpolation(const int m, const int v, int k, int j, int i,
                            const int nx1, const int nx2, const int nx3,
                            const bool offsetk, const bool offsetj, const bool offseti,
                        const DvceArray5D<Real> &ca, const DualArray3D<Real> &weights) {
  // interpolated value at new grid point
  Real ivals = 0;

  for (int kk=0; kk<NGHOST+1; kk++) {
    for (int jj=0; jj<NGHOST+1; jj++) {
      for (int ii=0; ii<NGHOST+1; ii++) {
        int wghti = (offseti) ? NGHOST-ii : ii;
        int wghtj = (offsetj) ? NGHOST-jj : jj;
        int wghtk = (offsetk) ? NGHOST-kk : kk;
        ivals += weights.d_view(wghtk,wghtj,wghti)*ca(m,v,
                    k-NGHOST/2+kk,j-NGHOST/2+jj,i-NGHOST/2+ii);
      }
    }
  }

  return ivals;
}

//----------------------------------------------------------------------------------------
//! \fn HighOrderProlongCC()
//! \brief high-order prolongation operator for cell-centered variables

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void HighOrderProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi, const int nx1, const int nx2,
               const int nx3, const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a,
               const DualArray3D<Real> &weights) {
  // stencil size for interpolator
  a(m,v,fk  ,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false,false, ca, weights);
  a(m,v,fk  ,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false, true, ca, weights);
  a(m,v,fk  ,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true,false, ca, weights);
  a(m,v,fk  ,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true, true, ca, weights);
  a(m,v,fk+1,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false,false, ca, weights);
  a(m,v,fk+1,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false, true, ca, weights);
  a(m,v,fk+1,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true,false, ca, weights);
  a(m,v,fk+1,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true, true, ca, weights);
  return;
}

#endif // MESH_PROLONGATION_HPP_

