//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction_spitzer.cpp
//! \brief Explicit pure-Spitzer conduction with harmonic face conductivity.

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/conduction_closure.hpp"
#include "diffusion/conduction_saturated.hpp"

// Interpolate K(theta)=K0*theta^(5/2) harmonically at each face. The SAME face
// coefficient multiplies the normal and limited transverse theta gradients;
// no direction-dependent nonlinear thermal-potential differences are used.
// A truly zero endpoint K gives zero Spitzer flux at that face. This is the
// harmonic closure's cold-endpoint limit, not a numerical temperature floor.
void Conduction::AddHeatFluxSpitzer(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                                  DvceFaceFld5D<Real> &flx) {
  const Real k0 = spitzer_kappa0_code;
  if (!alpha_spitzer || k0 == 0.0) return;
  const Real gm1 = eos.gamma-1.0;
  FailIfInvalid(!eos.is_ideal || !conduction::IsFinitePositive(gm1) ||
                !conduction::IsFinitePositive(k0),
                "Spitzer equation of state/coefficient");
  const bool aligned = spitzer_field_aligned;
  FailIfInvalid(aligned && pmy_pack->pmhd == nullptr, "Spitzer magnetic geometry");
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int nmb = pmy_pack->nmb_thispack;
  const bool mc = use_mc_limiter;
  auto w = w0;
  auto size = pmy_pack->pmb->mb_size;
  DvceArray5D<Real> bcc;
  if (aligned) bcc = pmy_pack->pmhd->bcc0;
  for (int dir=0; dir<ndim; ++dir) {
    const int di = dir == 0, dj = dir == 1, dk = dir == 2;
    const int ni = ind.nx1+di, nj = ind.nx2+dj, nk = ind.nx3+dk;
    const int nji = ni*nj, nface = nji*nk;
    auto f = dir == 0 ? flx.x1f : (dir == 1 ? flx.x2f : flx.x3f);
    DvceArray4D<Real> bf;
    if (aligned) {
      bf = dir == 0 ? pmy_pack->pmhd->b0.x1f :
           (dir == 1 ? pmy_pack->pmhd->b0.x2f : pmy_pack->pmhd->b0.x3f);
    }
    int invalid_faces = 0;
    Kokkos::parallel_reduce("conduct_spitzer_harmonic",
      Kokkos::RangePolicy<>(DevExeSpace(),0,nmb*nface),
    KOKKOS_LAMBDA(const int idx, int &invalid) {
      const int m = idx/nface, cell = idx-m*nface;
      const int kk = cell/nji, jj = (cell-kk*nji)/ni, ii = cell-kk*nji-jj*ni;
      const int k = ks+kk, j = js+jj, i = is+ii;
      const int il = i-di, jl = j-dj, kl = k-dk;
      const Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
      Real b[3] = {0.0,0.0,0.0};
      if (aligned) {
        for (int d=0; d<3; ++d) {
          b[d] = d == dir ? bf(m,k,j,i) :
                 0.5*bcc(m,d,k,j,i)+0.5*bcc(m,d,kl,jl,il);
          if (!conduction::IsFinite(b[d])) { ++invalid; return; }
        }
        const Real scale = fmax(fabs(b[0]),fmax(fabs(b[1]),fabs(b[2])));
        // A zero field still consumes/validates the thermal stencil below.
        if (scale > 0.0) {
          for (int d=0; d<3; ++d) b[d] /= scale;
          const Real norm = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
          for (int d=0; d<3; ++d) b[d] /= norm;
        }
      }
      const Real tr = conduction::SpitzerTemperature(w(m,IDN,k,j,i),w(m,IEN,k,j,i),gm1);
      const Real tl = conduction::SpitzerTemperature(w(m,IDN,kl,jl,il),
                                                     w(m,IEN,kl,jl,il),gm1);
      if (!conduction::IsFiniteNonNegative(tl) ||
          !conduction::IsFiniteNonNegative(tr)) { ++invalid; return; }
      const Real kappal = conduction::SpitzerConductivity(tl,k0);
      const Real kappar = conduction::SpitzerConductivity(tr,k0);
      if (!conduction::IsFiniteNonNegative(kappal) ||
          !conduction::IsFiniteNonNegative(kappar) ||
          (tl > 0.0 && kappal == 0.0) || (tr > 0.0 && kappar == 0.0)) {
        ++invalid;
        return;
      }
      const Real kface = conduction::HarmonicMeanPositive(kappal,kappar);
      if (!conduction::IsFiniteNonNegative(kface) ||
          !conduction::IsFinitePositive(dx[dir])) { ++invalid; return; }
      Real g[3] = {0.0,0.0,0.0};
      g[dir] = conduction::SpitzerTemperatureSlope(tl,tr,dx[dir]);
      if (!conduction::IsFinite(g[dir])) { ++invalid; return; }
      if (aligned) {
        for (int d=0; d<ndim; ++d) {
          if (d == dir) continue;
          const int ti = d == 0, tj = d == 1, tk = d == 2;
          const Real rp = conduction::SpitzerTemperature(w(m,IDN,k+tk,j+tj,i+ti),
                                                       w(m,IEN,k+tk,j+tj,i+ti),gm1);
          const Real rm = conduction::SpitzerTemperature(w(m,IDN,k-tk,j-tj,i-ti),
                                                       w(m,IEN,k-tk,j-tj,i-ti),gm1);
          const Real lp = conduction::SpitzerTemperature(w(m,IDN,kl+tk,jl+tj,il+ti),
                                                       w(m,IEN,kl+tk,jl+tj,il+ti),gm1);
          const Real lm = conduction::SpitzerTemperature(w(m,IDN,kl-tk,jl-tj,il-ti),
                                                       w(m,IEN,kl-tk,jl-tj,il-ti),gm1);
          if (!conduction::IsFiniteNonNegative(rp) ||
              !conduction::IsFiniteNonNegative(rm) ||
              !conduction::IsFiniteNonNegative(lp) ||
              !conduction::IsFiniteNonNegative(lm) ||
              !conduction::IsFinitePositive(dx[d])) { ++invalid; return; }
          // These are temperature slopes, not potential slopes. Positive
          // homogeneity permits limiting gradients before the common K_face.
          g[d] = conduction::SpitzerLimited4(
              conduction::SpitzerTemperatureSlope(tr,rp,dx[d]),
              conduction::SpitzerTemperatureSlope(rm,tr,dx[d]),
              conduction::SpitzerTemperatureSlope(tl,lp,dx[d]),
              conduction::SpitzerTemperatureSlope(lm,tl,dx[d]),mc);
          if (!conduction::IsFinite(g[d])) { ++invalid; return; }
        }
        // Exactly perpendicular or zero face field: no flux, but no invalid
        // thermal input was hidden by this exit. Retain all three B components.
        if (b[dir] == 0.0) return;
      }
      Real flux = -conduction::SpitzerProduct3(kface,g[dir],1.0);
      if (aligned) {
        const Real scale = fmax(fabs(g[0]),fmax(fabs(g[1]),fabs(g[2])));
        Real projected = 0.0;
        if (scale > 0.0) {
          for (int d=0; d<ndim; ++d) projected += b[d]*(g[d]/scale);
        }
        // Reuse the robust generic product without forming K_face*scale first.
        flux = -conduction::SaturatedProduct5(kface,scale,b[dir],projected,1.0);
      }
      const Real updated = f(m,IEN,k,j,i)+flux;
      if (!conduction::IsFinite(flux) || !conduction::IsFinite(updated)) {
        ++invalid;
        return;
      }
      f(m,IEN,k,j,i) = updated;
    }, Kokkos::Sum<int>(invalid_faces));
    // Every rank calls the coordinated check in every direction, including
    // local zero-error/zero-block cases. This is rejection, not rollback/retry.
    FailIfInvalid(invalid_faces,"Spitzer face thermal state, geometry, or flux");
  }
}
