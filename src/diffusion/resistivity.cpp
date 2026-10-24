//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.cpp
//  \brief Implements functions for Resistivity class.

#include <float.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string> // string

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "coordinates/coordinates.hpp"
#include "units/units.hpp"
#include "resistivity.hpp"
#include "current_density.hpp"

namespace {

// Athena++ FieldDiffusion::OhmicEMF places eta at an edge by arithmetic
// averaging of the adjacent cell coefficients. Use the same eta for induction
// and the conservative Poynting flux. Collapsed dimensions have no neighbors.
KOKKOS_INLINE_FUNCTION
Real SpitzerCellEta(const DvceArray5D<Real> &w, const int m, const int k,
                    const int j, const int i, const Real gm1, const Real eta0) {
  const Real temp = gm1*w(m,IEN,k,j,i)/w(m,IDN,k,j,i);
  const Real eta = exp(log(eta0)-1.5*log(temp));
  if (!std::isfinite(temp) || temp <= 0.0 || !std::isfinite(eta) || eta <= 0.0) {
    Kokkos::abort("Spitzer resistivity requires finite positive temperature/diffusivity");
  }
  return eta;
}

KOKKOS_INLINE_FUNCTION
Real SpitzerEdgeEta(const DvceArray5D<Real> &w, const int m, const int k,
                    const int j, const int i, const int axis, const bool multi_d,
                    const bool three_d, const Real gm1, const Real eta0) {
  const int ni = (axis != 1) ? 2 : 1;
  const int nj = (axis != 2 && multi_d) ? 2 : 1;
  const int nk = (axis != 3 && three_d) ? 2 : 1;
  Real eta = 0.0;
  for (int dk=0; dk<nk; ++dk) {
    for (int dj=0; dj<nj; ++dj) {
      for (int di=0; di<ni; ++di) {
        eta += SpitzerCellEta(w, m, k-dk, j-dj, i-di, gm1, eta0)/(ni*nj*nk);
      }
    }
  }
  return eta;
}

parabolic::DiffusionSelection ParseResistivityIntegrator(ParameterInput *pin) {
  std::string integrator = pin->GetOrAddString("mhd", "resistivity_integrator",
                                               "explicit");
  if (integrator == "explicit") {
    return parabolic::DiffusionSelection::explicit_only;
  }
  if (integrator == "sts") {
    return parabolic::DiffusionSelection::sts_only;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<mhd>/resistivity_integrator = '" << integrator
            << "' must be 'explicit' or 'sts'" << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

namespace resistivity {
Real SpitzerCoefficientCode(Real eta0_cgs, Real length_cgs, Real time_cgs,
                            Real temperature_cgs) {
  if (!std::isfinite(eta0_cgs) || eta0_cgs <= 0.0 ||
      !std::isfinite(length_cgs) || length_cgs <= 0.0 ||
      !std::isfinite(time_cgs) || time_cgs <= 0.0 ||
      !std::isfinite(temperature_cgs) || temperature_cgs <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  const long double logarithm = std::log(static_cast<long double>(eta0_cgs))
      + std::log(static_cast<long double>(time_cgs))
      - 2.0L*std::log(static_cast<long double>(length_cgs))
      - 1.5L*std::log(static_cast<long double>(temperature_cgs));
  return static_cast<Real>(std::exp(logarithm));
}
} // namespace resistivity

//----------------------------------------------------------------------------------------
// ctor: also calls Resistivity base class constructor

Resistivity::Resistivity(MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  // Read non-ideal MHD coefficients (if any). A non-zero value enables the term.
  eta_ohm = pin->GetOrAddReal("mhd","eta_ohm",0.0);
  eta_ad  = pin->GetOrAddReal("mhd","eta_ad",0.0);
  mode = ParseResistivityIntegrator(pin);
  eta_spitzer = pin->GetOrAddBoolean("mhd", "eta_spitzer", false);
  const bool has_code = pin->DoesParameterExist("mhd", "spitzer_eta0_code");
  const bool has_cgs = pin->DoesParameterExist("mhd", "spitzer_eta0_cgs");
  if (!eta_spitzer && (has_code || has_cgs)) {
    std::cerr << "Spitzer resistivity parameters require eta_spitzer=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (eta_spitzer) {
    if (has_code == has_cgs || eta_ohm != 0.0 || eta_ad != 0.0 ||
        pin->GetString("mhd", "eos") != "ideal" ||
        pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic ||
        pmy_pack->pcoord->is_dynamical_relativistic) {
      std::cerr << "Spitzer resistivity requires Newtonian ideal MHD, exactly one "
                << "spitzer_eta0_code/cgs, and zero eta_ohm/eta_ad" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (has_code) {
      spitzer_eta0_code = pin->GetReal("mhd", "spitzer_eta0_code");
    } else {
      auto *pu = pmy_pack->punit;
      if (pu == nullptr) {
        std::cerr << "spitzer_eta0_cgs requires an explicit units block" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      spitzer_eta0_code = resistivity::SpitzerCoefficientCode(
          pin->GetReal("mhd", "spitzer_eta0_cgs"), pu->length_cgs(),
          pu->time_cgs(), pu->temperature_cgs());
    }
    if (!std::isfinite(spitzer_eta0_code) || spitzer_eta0_code <= 0.0) {
      std::cerr << "Spitzer resistivity coefficient must be finite and positive "
                << "in code units" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real cfl = pin->GetReal("time", "cfl_number");
    const std::string rk = pin->GetOrAddString("time", "integrator", "rk2");
    if (!std::isfinite(cfl) || cfl <= 0.0 || cfl > 1.0 ||
        (rk != "rk1" && rk != "rk2" && rk != "rk3")) {
      std::cerr << "Spitzer resistivity requires 0 < cfl_number <= 1 and "
                << "SSP rk1, rk2 or rk3" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (mode == parabolic::DiffusionSelection::sts_only &&
      (!std::isfinite(eta_ad) || eta_ad != 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Ambipolar diffusion is not supported with STS; use explicit "
              << "integration" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mode == parabolic::DiffusionSelection::sts_only && !eta_spitzer &&
      (!std::isfinite(eta_ohm) || eta_ohm <= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "STS requires positive constant Ohmic resistivity" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// Resistivity destructor

Resistivity::~Resistivity() {
}

// Inspect raw U and current CT fields without changing state or EOS counters.
// Match the production ideal-MHD EOS, including its density/temperature floors.
Resistivity::RawState Resistivity::InspectRawState(const DvceArray5D<Real> &u,
    const EOS_Data &eos, const DvceFaceFld4D<Real> &b, bool include_ghosts) const {
  const auto &idx = pmy_pack->pmesh->mb_indcs;
  const int is = idx.is, ie = idx.ie, js = idx.js, je = idx.je;
  const int ks = idx.ks, ke = idx.ke;
  const int il = include_ghosts ? 0 : is;
  const int jl = include_ghosts ? 0 : js;
  const int kl = include_ghosts ? 0 : ks;
  const int ni = idx.nx1+(include_ghosts ? 2*idx.ng : 0);
  const int nj = idx.nx2+(include_ghosts && idx.nx2 > 1 ? 2*idx.ng : 0);
  const int nk = idx.nx3+(include_ghosts && idx.nx3 > 1 ? 2*idx.ng : 0);
  const int nji = ni*nj, nkji = nji*nk;
  auto b1 = b.x1f, b2 = b.x2f, b3 = b.x3f;
  const auto eos_data = eos;
  const bool valid_eos = eos.is_ideal && std::isfinite(eos.gamma) && eos.gamma > 1.0;
  const bool multi_d = pmy_pack->pmesh->multi_d, three_d = pmy_pack->pmesh->three_d;
  const Real fac = three_d ? 1.0/6.0 : (multi_d ? 0.25 : 0.5);
  const Real eta0 = spitzer_eta0_code;
  const bool spitzer = eta_spitzer;
  const Real constant_eta = eta_ohm;
  auto size = pmy_pack->pmb->mb_size;
  int invalid = 0, invalid_active = 0;
  Real diffusion_dt = std::numeric_limits<float>::max();
  Kokkos::parallel_reduce("resistivity_raw_state", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*nkji),
  KOKKOS_LAMBDA(const int n, int &bad, int &bad_active, Real &min_dt) {
    const int m = n/nkji, rem = n-m*nkji;
    const int k = rem/nji+kl, j = (rem%nji)/ni+jl, i = rem%ni+il;
    MHDCons1D cell;
    cell.d = u(m,IDN,k,j,i);
    cell.mx = u(m,IM1,k,j,i);
    cell.my = u(m,IM2,k,j,i);
    cell.mz = u(m,IM3,k,j,i);
    cell.e = u(m,IEN,k,j,i);
    const Real faces[6] = {b1(m,k,j,i), b1(m,k,j,i+1),
                          b2(m,k,j,i), b2(m,k,j+1,i),
                          b3(m,k,j,i), b3(m,k+1,j,i)};
    bool invalid_cell = !valid_eos || !(cell.d > 0.0) || !std::isfinite(cell.d) ||
        !std::isfinite(cell.mx) || !std::isfinite(cell.my) ||
        !std::isfinite(cell.mz) || !std::isfinite(cell.e);
    for (int a=0; a<6; ++a) invalid_cell |= !std::isfinite(faces[a]);
    cell.bx = 0.5*(faces[0]+faces[1]);
    cell.by = 0.5*(faces[2]+faces[3]);
    cell.bz = 0.5*(faces[4]+faces[5]);
    if (!invalid_cell) {
      const Real thermal = cell.e-0.5*(SQR(cell.mx)+SQR(cell.my)+SQR(cell.mz))/cell.d
          - 0.5*(SQR(cell.bx)+SQR(cell.by)+SQR(cell.bz));
      invalid_cell = !std::isfinite(thermal) || thermal <= 0.0;
    }
    if (!invalid_cell) {
      HydPrim1D primitive;
      bool df = false, ef = false, tf = false;
      SingleC2P_IdealMHD(cell, eos_data, primitive, df, ef, tf);
      const Real temp = (eos_data.gamma-1.0)*primitive.e/primitive.d;
      invalid_cell = df || ef || tf || !std::isfinite(temp) || temp <= 0.0 ||
          !std::isfinite(primitive.vx) || !std::isfinite(primitive.vy) ||
          !std::isfinite(primitive.vz);
      if (!invalid_cell) {
        const Real eta = spitzer ? exp(log(eta0)-1.5*log(temp)) : constant_eta;
        invalid_cell = !std::isfinite(eta) || eta <= 0.0;
        // Match NewTimeStep's edge-coefficient stencil. More distant ghosts
        // still receive the EOS admissibility check above.
        if (!invalid_cell && i >= is-1 && i <= ie+1 &&
            j >= js-multi_d && j <= je+multi_d &&
            k >= ks-three_d && k <= ke+three_d) {
          min_dt = fmin(min_dt, fac*SQR(size.d_view(m).dx1)/eta);
          if (multi_d) min_dt = fmin(min_dt, fac*SQR(size.d_view(m).dx2)/eta);
          if (three_d) min_dt = fmin(min_dt, fac*SQR(size.d_view(m).dx3)/eta);
        }
      }
    }
    if (invalid_cell) {
      ++bad;
      if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
        ++bad_active;
      }
    }
  }, Kokkos::Sum<int>(invalid), Kokkos::Sum<int>(invalid_active),
     Kokkos::Min<Real>(diffusion_dt));
#if MPI_PARALLEL_ENABLED
  int invalid_global[2] = {invalid, invalid_active};
  MPI_Allreduce(MPI_IN_PLACE, invalid_global, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  invalid = invalid_global[0];
  invalid_active = invalid_global[1];
  MPI_Allreduce(MPI_IN_PLACE, &diffusion_dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  return RawState{invalid == 0, invalid_active == 0, diffusion_dt};
}

//----------------------------------------------------------------------------------------
//! \fn void AddResistiveEMFs()
//! \brief Wrapper function that adds non-ideal electric fields to the corner-centered
//! EMF. Adds the Ohmic contribution if eta_ohm != 0 and the ambipolar contribution if
//! eta_ad != 0. Ohmic diffusion may use the Spitzer temperature law.

void Resistivity::AddResistiveEMFs(const DvceFaceFld4D<Real> &b0,
    DvceEdgeFld4D<Real> &efld) {
  if (eta_ohm != 0.0 || eta_spitzer) {
    AddEMFOhmic(b0, efld);
  }
  if (eta_ad != 0.0) {
    AddEMFConstantAmbipolar(b0, efld);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddResistiveFluxes()
//! \brief Wrapper function that adds non-ideal energy (Poynting) fluxes to the energy
//! flux. Adds the Ohmic contribution if eta_ohm != 0 and the ambipolar contribution if
//! eta_ad != 0. Ohmic diffusion may use the Spitzer temperature law.

void Resistivity::AddResistiveFluxes(const DvceFaceFld4D<Real> &b0,
    DvceFaceFld5D<Real> &flx) {
  if (eta_ohm != 0.0 || eta_spitzer) {
    AddFluxOhmic(b0, flx);
  }
  if (eta_ad != 0.0) {
    AddFluxConstantAmbipolar(b0, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddEMFOhmic()
//  \brief Adds electric field from Ohmic resistivity to corner-centered electric field
//  Using Ohm's Law to compute the electric field:  E + (v x B) = \eta J, then
//    E_{inductive} = - (v x B)  [computed in the MHD Riemann solver]
//    E_{resistive} = \eta J     [computed in this function]

void Resistivity::AddEMFOhmic(const DvceFaceFld4D<Real> &b0,
    DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  const bool spitzer = eta_spitzer;
  const Real eta0 = spitzer_eta0_code;
  const Real gm1 = spitzer ? pmy_pack->pmhd->peos->eos_data.gamma-1.0 : 0.0;
  auto w = pmy_pack->pmhd->w0;

  //---- 1-D problem:
  //  copy face-centered E-fields to edges and return.
  //  Note e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]

  if (pmy_pack->pmesh->one_d) {
    // capture class variables for the kernels
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, js, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        const Real eta = spitzer ? SpitzerEdgeEta(w, m, ks, js, i, 2,
                                                   false, false, gm1, eta0) : eta_o;
        e2(m,ks,  js  ,i) += eta*j2(i);
        e2(m,ke+1,js  ,i) += eta*j2(i);
        e3(m,ks  ,js  ,i) += eta*j3(i);
        e3(m,ks  ,je+1,i) += eta*j3(i);
      });
    });
    return;
  }

  //---- 2-D problem:
  if (pmy_pack->pmesh->two_d) {
    // capture class variables for the kernels
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm2", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        const Real eta1 = spitzer ? SpitzerEdgeEta(w, m, ks, j, i, 1,
                                                    true, false, gm1, eta0) : eta_o;
        const Real eta2 = spitzer ? SpitzerEdgeEta(w, m, ks, j, i, 2,
                                                    true, false, gm1, eta0) : eta_o;
        const Real eta3 = spitzer ? SpitzerEdgeEta(w, m, ks, j, i, 3,
                                                    true, false, gm1, eta0) : eta_o;
        e1(m,ks,  j,i) += eta1*j1(i);
        e1(m,ke+1,j,i) += eta1*j1(i);
        e2(m,ks,  j,i) += eta2*j2(i);
        e2(m,ke+1,j,i) += eta2*j2(i);
        e3(m,ks  ,j,i) += eta3*j3(i);
      });
    });
    return;
  }

  //---- 3-D problem:

  // capture class variables for the kernels
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto eta_o = eta_ohm;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  par_for_outer("ohm3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

    CurrentDensity(member, m, k, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

    // Add E_{resistive} = \eta J to corner-centered electric fields
    par_for_inner(member, is, ie+1, [&](const int i) {
      const Real eta1 = spitzer ? SpitzerEdgeEta(w, m, k, j, i, 1,
                                                  true, true, gm1, eta0) : eta_o;
      const Real eta2 = spitzer ? SpitzerEdgeEta(w, m, k, j, i, 2,
                                                  true, true, gm1, eta0) : eta_o;
      const Real eta3 = spitzer ? SpitzerEdgeEta(w, m, k, j, i, 3,
                                                  true, true, gm1, eta0) : eta_o;
      e1(m,k,j,i) += eta1*j1(i);
      e2(m,k,j,i) += eta2*j2(i);
      e3(m,k,j,i) += eta3*j3(i);
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn AddResistiveFluxConstantResist()
//  \brief Adds Poynting flux from Ohmic resistivity to energy flux
//  Total energy equation is dE/dt = - Div(F) where F = (E X B) = \eta (J X B)

void Resistivity::AddFluxOhmic(const DvceFaceFld4D<Real> &b,
                                        DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  const bool spitzer = eta_spitzer;
  const Real eta0 = spitzer_eta0_code;
  const Real gm1 = spitzer ? pmy_pack->pmhd->peos->eos_data.gamma-1.0 : 0.0;
  auto w = pmy_pack->pmhd->w0;
  Real qa = spitzer ? 0.25 : 0.25*eta_ohm;

  //------------------------------
  // energy fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("ohm_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j2k   = -(b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1;
    Real j2kp1 = -(b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j,i-1))/size.d_view(m).dx1;

    Real j3j   = (b.x2f(m,k,j  ,i) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1;
    Real j3jp1 = (b.x2f(m,k,j+1,i) - b.x2f(m,k,j+1,i-1))/size.d_view(m).dx1;

    if (multi_d) {
      j3j   -= (b.x1f(m,k,j  ,i) - b.x1f(m,k,j-1,i))/size.d_view(m).dx2;
      j3jp1 -= (b.x1f(m,k,j+1,i) - b.x1f(m,k,j  ,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      j2k   += (b.x1f(m,k  ,j,i) - b.x1f(m,k-1,j,i))/size.d_view(m).dx3;
      j2kp1 += (b.x1f(m,k+1,j,i) - b.x1f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    if (spitzer) {
      j2k   *= SpitzerEdgeEta(w, m, k, j, i, 2, multi_d, three_d, gm1, eta0);
      j2kp1 *= SpitzerEdgeEta(w, m, k+three_d, j, i, 2, multi_d, three_d, gm1, eta0);
      j3j   *= SpitzerEdgeEta(w, m, k, j, i, 3, multi_d, three_d, gm1, eta0);
      j3jp1 *= SpitzerEdgeEta(w, m, k, j+multi_d, i, 3, multi_d, three_d, gm1, eta0);
    }
    // flx1 = (E X B)_{1} =  ((\eta J) X B)_{1} = \eta (J2*B3 - J3*B2)
    flx1(m,IEN,k,j,i) += qa*(j2k  *(b.x3f(m,k  ,j  ,i) + b.x3f(m,k  ,j  ,i-1)) +
                             j2kp1*(b.x3f(m,k+1,j  ,i) + b.x3f(m,k+1,j  ,i-1)) -
                             j3j  *(b.x2f(m,k  ,j  ,i) + b.x2f(m,k  ,j  ,i-1)) -
                             j3jp1*(b.x2f(m,k  ,j+1,i) + b.x2f(m,k  ,j+1,i-1)));
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //------------------------------
  // energy fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("ohm_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1k   = (b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2;
    Real j1kp1 = (b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j-1,i))/size.d_view(m).dx2;

    Real j3i   = (b.x2f(m,k,j,i  ) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i  ) - b.x1f(m,k,j-1,i  ))/size.d_view(m).dx2;
    Real j3ip1 = (b.x2f(m,k,j,i+1) - b.x2f(m,k,j  ,i  ))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i+1) - b.x1f(m,k,j-1,i+1))/size.d_view(m).dx2;

    if (three_d) {
      j1k   -= (b.x2f(m,k  ,j,i) - b.x2f(m,k-1,j,i))/size.d_view(m).dx3;
      j1kp1 -= (b.x2f(m,k+1,j,i) - b.x2f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    if (spitzer) {
      j3i   *= SpitzerEdgeEta(w, m, k, j, i, 3, multi_d, three_d, gm1, eta0);
      j3ip1 *= SpitzerEdgeEta(w, m, k, j, i+1, 3, multi_d, three_d, gm1, eta0);
      j1k   *= SpitzerEdgeEta(w, m, k, j, i, 1, multi_d, three_d, gm1, eta0);
      j1kp1 *= SpitzerEdgeEta(w, m, k+three_d, j, i, 1, multi_d, three_d, gm1, eta0);
    }
    // E2 = \eta (J X B)_{2} = \eta (J3*B1 - J1*B3)
    flx2(m,IEN,k,j,i) += qa*(j3i  *(b.x1f(m,k  ,j,i  ) + b.x1f(m,k  ,j-1,i  )) +
                             j3ip1*(b.x1f(m,k  ,j,i+1) + b.x1f(m,k  ,j-1,i+1)) -
                             j1k  *(b.x3f(m,k  ,j,i  ) + b.x3f(m,k  ,j-1,i  )) -
                             j1kp1*(b.x3f(m,k+1,j,i  ) + b.x3f(m,k+1,j-1,i  )));
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //------------------------------
  // energy fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("ohm_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1j   = (b.x3f(m,k,j  ,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j  ,i) - b.x2f(m,k-1,j  ,i))/size.d_view(m).dx3;
    Real j1jp1 = (b.x3f(m,k,j+1,i) - b.x3f(m,k  ,j  ,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j+1,i) - b.x2f(m,k-1,j+1,i))/size.d_view(m).dx3;

    Real j2i   = -(b.x3f(m,k,j,i  ) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i  ) - b.x1f(m,k-1,j,i  ))/size.d_view(m).dx3;
    Real j2ip1 = -(b.x3f(m,k,j,i+1) - b.x3f(m,k  ,j,i  ))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i+1) - b.x1f(m,k-1,j,i+1))/size.d_view(m).dx3;

    if (spitzer) {
      j1j   *= SpitzerEdgeEta(w, m, k, j, i, 1, multi_d, three_d, gm1, eta0);
      j1jp1 *= SpitzerEdgeEta(w, m, k, j+1, i, 1, multi_d, three_d, gm1, eta0);
      j2i   *= SpitzerEdgeEta(w, m, k, j, i, 2, multi_d, three_d, gm1, eta0);
      j2ip1 *= SpitzerEdgeEta(w, m, k, j, i+1, 2, multi_d, three_d, gm1, eta0);
    }
    // E2 = \eta (J X B)_{2} = \eta (J1*B2 - J2*B1)
    flx3(m,IEN,k,j,i) += qa*(j1j  *(b.x2f(m,k,j  ,i  ) + b.x2f(m,k-1,j  ,i  )) +
                             j1jp1*(b.x2f(m,k,j+1,i  ) + b.x2f(m,k-1,j+1,i  )) -
                             j2i  *(b.x1f(m,k,j  ,i  ) + b.x1f(m,k-1,j  ,i  )) -
                             j2ip1*(b.x1f(m,k,j  ,i+1) + b.x1f(m,k-1,j  ,i+1)));
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Resistivity::NewTimeStep()
//! \brief Compute new time step for non-ideal MHD (Ohmic + ambipolar).
//! Ohmic:     dt <= fac * dx^2 / eta_ohm                  (constant diffusivity)
//! Spitzer:   dt <= fac * dx^2 / eta(T) (includes the neighboring edge stencil)
//! Ambipolar: dt <= fac * dx^2 / (eta_ohm + eta_ad * B^2) (diffusivity varies in space)
//! When ambipolar diffusion is active the limit is evaluated PER CELL (each cell pairs
//! its own dx with its own total diffusivity eta_ohm + eta_ad*B^2, and the global min is
//! taken), matching Athena++ FieldDiffusion::NewDiffusionDt and AthenaK's own spatially-
//! varying module Conduction::NewTimeStep. This avoids pairing the global-min dx with the
//! global-max B^2, which over-restricts dt under mesh refinement.

void Resistivity::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  // non-ideal MHD timestep on MeshBlock(s) in this pack
  dtnew = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mb_size;
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  auto &multi_d = pmy_pack->pmesh->multi_d;
  auto &three_d = pmy_pack->pmesh->three_d;

  if (eta_spitzer) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int il = indcs.is-1, iu = indcs.ie+1;
    const int jl = multi_d ? indcs.js-1 : indcs.js;
    const int ju = multi_d ? indcs.je+1 : indcs.je;
    const int kl = three_d ? indcs.ks-1 : indcs.ks;
    const int ku = three_d ? indcs.ke+1 : indcs.ke;
    const Real gm1 = eos_data.gamma-1.0, eta0 = spitzer_eta0_code;
    Kokkos::parallel_reduce("spitzer_resist_dt",
        Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>(
            {0, kl, jl, il}, {pmy_pack->nmb_thispack, ku+1, ju+1, iu+1}),
        KOKKOS_LAMBDA(int m, int k, int j, int i, Real &min_dt) {
      const Real eta = SpitzerCellEta(w0, m, k, j, i, gm1, eta0);
      min_dt = fmin(min_dt, fac*SQR(size.d_view(m).dx1)/eta);
      if (multi_d) min_dt = fmin(min_dt, fac*SQR(size.d_view(m).dx2)/eta);
      if (three_d) min_dt = fmin(min_dt, fac*SQR(size.d_view(m).dx3)/eta);
    }, Kokkos::Min<Real>(dtnew));
    return;
  }

  // Ohmic diffusivity is a CONSTANT coefficient: when no (spatially-varying) ambipolar
  // term is active, the limit is simply fac*dx^2/eta_ohm and a cheap host loop suffices
  // (matching Viscosity::NewTimeStep). Keeping this branch separate leaves the eta_ad==0
  // (AD-off) behaviour byte-identical to before.
  if (eta_ad == 0.0) {
    if (eta_ohm > 0.0) {
      for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
        dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/eta_ohm);
        if (multi_d) {
          dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/eta_ohm);
        }
        if (three_d) {
          dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/eta_ohm);
        }
      }
    }
    return;
  }

  // Ambipolar diffusivity eta_ad*B^2 varies in space, so evaluate the limit per cell.
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &bcc0 = pmy_pack->pmhd->bcc0;
  Real eta_o = eta_ohm;
  Real eta_a = eta_ad;

  // find smallest dx^2/(eta_ohm + eta_ad*B^2) over all cells, then scale by fac
  Kokkos::parallel_reduce("resist_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real eta = eta_o + eta_a*(SQR(bcc0(m,IBX,k,j,i)) + SQR(bcc0(m,IBY,k,j,i))
                            + SQR(bcc0(m,IBZ,k,j,i)));
    if (eta > 0.0) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx1)/eta);
      if (multi_d) {
        min_dt = fmin(min_dt, SQR(size.d_view(m).dx2)/eta);
      }
      if (three_d) {
        min_dt = fmin(min_dt, SQR(size.d_view(m).dx3)/eta);
      }
    }
  }, Kokkos::Min<Real>(dtnew));
  dtnew *= fac;

  return;
}
