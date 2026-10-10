//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction_saturated.cpp
//! \brief Explicit Cowie-McKee saturation of the complete conductive face vector.

#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/conduction_saturated.hpp"

void Conduction::AddHeatFluxSaturated(const DvceArray5D<Real> &w0,
    const EOS_Data &eos, DvceFaceFld5D<Real> &flx) {
  const Real gm1 = eos.gamma-1.0;
  const Real k0 = alpha_spitzer ? spitzer_kappa0_code : 0.0;
  const bool aligned = alpha_aniso > 0.0 || (k0 > 0.0 && spitzer_field_aligned);
  FailIfInvalid(!eos.is_ideal || !conduction::IsFinitePositive(gm1) ||
                (aligned && pmy_pack->pmhd == nullptr), "saturated conduction setup");
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const conduction::SaturatedFaceParameters p = {gm1,alpha_iso,alpha_aniso,k0,
      saturation_phi,ndim,spitzer_field_aligned,use_mc_limiter};
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int nmb = pmy_pack->nmb_thispack;
  auto w = w0;
  auto size = pmy_pack->pmb->mb_size;
  DvceArray5D<Real> bcc;
  if (aligned) bcc = pmy_pack->pmhd->bcc0;
  for (int dir=0; dir<ndim; ++dir) {
    const int ni = ind.nx1+(dir == 0), nj = ind.nx2+(dir == 1);
    const int nk = ind.nx3+(dir == 2), nji = ni*nj, nface = nji*nk;
    auto f = dir == 0 ? flx.x1f : (dir == 1 ? flx.x2f : flx.x3f);
    DvceArray4D<Real> bf;
    if (aligned) {
      bf = dir == 0 ? pmy_pack->pmhd->b0.x1f :
          (dir == 1 ? pmy_pack->pmhd->b0.x2f : pmy_pack->pmhd->b0.x3f);
    }
    int invalid_faces = 0;
    Kokkos::parallel_reduce("conduct_saturated", Kokkos::RangePolicy<>(
        DevExeSpace(),0,nmb*nface),
    KOKKOS_LAMBDA(const int idx, int &invalid) {
      const int m = idx/nface, cell = idx-m*nface;
      const int k = ks+cell/nji, j = js+(cell%nji)/ni, i = is+cell%ni;
      const Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
      const auto face = conduction::SaturatedFace(w,bcc,bf,p,dx,dir,m,k,j,i);
      if (!face.valid) { ++invalid; return; }
      const Real updated = f(m,IEN,k,j,i)+face.normal_flux;
      if (!conduction::IsFinite(updated)) { ++invalid; return; }
      // Only conduction is suppressed; existing hyperbolic flux is untouched.
      f(m,IEN,k,j,i) = updated;
    }, Kokkos::Sum<int>(invalid_faces));
    FailIfInvalid(invalid_faces,"saturated face thermal state, geometry, or flux");
  }
}

// The larger of the donor-pressure derivative rate and outgoing-energy drain
// rate, summed over actual adjacent faces of each cell. The caller combines
// this inverse timestep with the UNSUPPRESSED diffusion row bound. The drain
// check bounds conduction alone; it does not promise coupled-MHD positivity.
// With donor_only, omit the forward-Euler drain restriction: this returns only
// the nonaccelerated donor-pressure rate needed by guarded nonlinear STS.
Real Conduction::SaturationTimeStep(const DvceArray5D<Real> &w0,
                                    const EOS_Data &eos, bool donor_only) const {
  const Real gm1 = eos.gamma-1.0;
  const Real k0 = alpha_spitzer ? spitzer_kappa0_code : 0.0;
  const bool aligned = alpha_aniso > 0.0 || (k0 > 0.0 && spitzer_field_aligned);
  FailIfInvalid(!eos.is_ideal || !conduction::IsFinitePositive(gm1) ||
                (aligned && pmy_pack->pmhd == nullptr), "saturated timestep setup");
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const conduction::SaturatedFaceParameters p = {gm1,alpha_iso,alpha_aniso,k0,
      saturation_phi,ndim,spitzer_field_aligned,use_mc_limiter};
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int nmb = pmy_pack->nmb_thispack;
  auto w = w0;
  auto size = pmy_pack->pmb->mb_size;
  DvceArray5D<Real> bcc;
  if (aligned) bcc = pmy_pack->pmhd->bcc0;
  DvceArray4D<Real> b1, b2, b3;
  if (aligned) {
    b1 = pmy_pack->pmhd->b0.x1f;
    b2 = pmy_pack->pmhd->b0.x2f;
    b3 = pmy_pack->pmhd->b0.x3f;
  }
  const int ni = ind.nx1, nj = ind.nx2, nji = ni*nj, ncell = nji*ind.nx3;
  Real rate = 0.0;
  int invalid_cells = 0;
  Kokkos::parallel_reduce("saturated_donor_dt", Kokkos::RangePolicy<>(
      DevExeSpace(),0,nmb*ncell),
  KOKKOS_LAMBDA(const int idx, Real &max_rate, int &invalid) {
    const int m = idx/ncell, cell = idx-m*ncell;
    const int k = ks+cell/nji, j = js+(cell%nji)/ni, i = is+cell%ni;
    const Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
    const Real eint = w(m,IEN,k,j,i);
    if (!conduction::IsFiniteNonNegative(eint)) { ++invalid; return; }
    Real advective = 0.0, drain = 0.0;
    for (int dir=0; dir<ndim; ++dir) {
      auto bf = dir == 0 ? b1 : (dir == 1 ? b2 : b3);
      const auto left = conduction::SaturatedFace(w,bcc,bf,p,dx,dir,m,k,j,i);
      const auto right = conduction::SaturatedFace(w,bcc,bf,p,dx,dir,m,
          k+(dir == 2),j+(dir == 1),i+(dir == 0));
      if (!left.valid || !right.valid) { ++invalid; return; }
      // Evaluate divisions separately to avoid overflowing a sum of speeds.
      const Real al = left.donor_speed/dx[dir];
      const Real ar = right.donor_speed/dx[dir];
      if (!conduction::IsFiniteNonNegative(al) ||
          !conduction::IsFiniteNonNegative(ar) ||
          (left.donor_speed > 0.0 && al == 0.0) ||
          (right.donor_speed > 0.0 && ar == 0.0)) { ++invalid; return; }
      advective += al+ar;
      if (!donor_only) {
        const Real outward[2] = {fmax(-left.normal_flux,0.0),
                                 fmax(right.normal_flux,0.0)};
        for (int side=0; side<2; ++side) {
          if (outward[side] == 0.0) continue;
          if (eint == 0.0) { ++invalid; return; }
          // Log scaling avoids either dx*eint or outward/dx overflowing first.
          const Real contribution = exp(log(outward[side])-log(dx[dir])-log(eint));
          if (!conduction::IsFinitePositive(contribution)) { ++invalid; return; }
          drain += contribution;
        }
      }
    }
    const Real local_rate = donor_only ? advective : fmax(advective,drain);
    if (!conduction::IsFiniteNonNegative(local_rate)) { ++invalid; return; }
    max_rate = fmax(max_rate,local_rate);
  }, Kokkos::Max<Real>(rate), Kokkos::Sum<int>(invalid_cells));
  if (nmb == 0) rate = 0.0;
  // One combined kernel visits every direction; every rank joins this check,
  // including zero-error and empty-pack ranks.
  FailIfInvalid(invalid_cells || !conduction::IsFiniteNonNegative(rate),
                "saturated donor-pressure or outgoing-energy timestep rate");
  const Real bound = rate == 0.0 ?
      static_cast<Real>(std::numeric_limits<float>::max()) : 1.0/rate;
  FailIfInvalid(!conduction::IsFinitePositive(bound),
                "unrepresentable saturated donor-pressure timestep");
  return bound;
}
