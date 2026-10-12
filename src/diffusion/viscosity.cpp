//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.cpp
//  \brief Implements functions for Viscosity class. This includes isotropic shear
//  viscosity in a Newtonian fluid (in which stress is proportional to shear).
//  Viscosity may be added to Hydro and/or MHD independently.

#include <float.h>
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <string> // string

// Athena++ headers
#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "viscosity.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
bool ViscosityFinite(const Real value) {
  const Real largest = std::numeric_limits<Real>::max();
  return value >= -largest && value <= largest;
}

// Apply the production EOS to local copies only. Isothermal states have no
// energy component, so return before attempting to access one.
KOKKOS_INLINE_FUNCTION
bool ViscosityRawCellAdmissible(const DvceArray5D<Real> &u, const EOS_Data &eos,
    const bool mhd, const DvceArray4D<Real> &b1, const DvceArray4D<Real> &b2,
    const DvceArray4D<Real> &b3, const int m, const int k, const int j, const int i) {
  MHDCons1D cell;
  cell.d = u(m,IDN,k,j,i);
  cell.mx = u(m,IM1,k,j,i);
  cell.my = u(m,IM2,k,j,i);
  cell.mz = u(m,IM3,k,j,i);
  if (!(cell.d > 0.0) || !ViscosityFinite(cell.d) ||
      !ViscosityFinite(cell.mx) || !ViscosityFinite(cell.my) ||
      !ViscosityFinite(cell.mz)) return false;
  cell.bx = cell.by = cell.bz = 0.0;
  if (mhd) {
    const Real bx0 = b1(m,k,j,i), bx1 = b1(m,k,j,i+1);
    const Real by0 = b2(m,k,j,i), by1 = b2(m,k,j+1,i);
    const Real bz0 = b3(m,k,j,i), bz1 = b3(m,k+1,j,i);
    if (!ViscosityFinite(bx0) || !ViscosityFinite(bx1) ||
        !ViscosityFinite(by0) || !ViscosityFinite(by1) ||
        !ViscosityFinite(bz0) || !ViscosityFinite(bz1)) return false;
    // Match ConsToPrim arithmetic and use the current face-centered field.
    cell.bx = 0.5*(bx0+bx1);
    cell.by = 0.5*(by0+by1);
    cell.bz = 0.5*(bz0+bz1);
  }
  const Real bsq = SQR(cell.bx)+SQR(cell.by)+SQR(cell.bz);
  if (!ViscosityFinite(bsq)) return false;
  const Real density_floor = mhd ? fmax(eos.dfloor, bsq/eos.sigma_max) : eos.dfloor;
  if (!ViscosityFinite(density_floor) || cell.d < density_floor) return false;
  const Real di = 1.0/cell.d;
  if (!ViscosityFinite(di*cell.mx) || !ViscosityFinite(di*cell.my) ||
      !ViscosityFinite(di*cell.mz)) return false;
  if (!eos.is_ideal) return true;

  cell.e = u(m,IEN,k,j,i);
  const Real kinetic = 0.5*di*(SQR(cell.mx)+SQR(cell.my)+SQR(cell.mz));
  const Real thermal = cell.e-kinetic-0.5*bsq;
  if (!ViscosityFinite(cell.e) || !ViscosityFinite(thermal) || !(thermal > 0.0)) {
    return false;
  }
  HydPrim1D primitive;
  bool df = false, ef = false, tf = false;
  if (mhd) {
    SingleC2P_IdealMHD(cell, eos, primitive, df, ef, tf);
  } else {
    HydCons1D hydro;
    hydro.d = cell.d; hydro.mx = cell.mx; hydro.my = cell.my;
    hydro.mz = cell.mz; hydro.e = cell.e;
    SingleC2P_IdealHyd(hydro, eos, primitive, df, ef, tf);
  }
  const Real temperature = (eos.gamma-1.0)*primitive.e/primitive.d;
  return !df && !ef && !tf && ViscosityFinite(primitive.e) && primitive.e > 0.0 &&
         ViscosityFinite(primitive.vx) && ViscosityFinite(primitive.vy) &&
         ViscosityFinite(primitive.vz) && ViscosityFinite(temperature) &&
         temperature > 0.0;
}

// Unlike atof, reject a malformed token instead of silently disabling viscosity
// or accepting a numerical prefix. Keep this validation local to viscosity.
Real ReadViscosityCoefficient(const std::string &block, const std::string &name,
                             ParameterInput *pin) {
  const std::string token = pin->GetOrAddString(block, name, "0.0");
  const char *begin = token.c_str();
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(begin, &end);
  const int parse_errno = errno;
  const bool parsed_number = (end != begin);
  while (std::isspace(static_cast<unsigned char>(*end))) ++end;
  if (!parsed_number || *end != '\0' || parse_errno == ERANGE ||
      !std::isfinite(value) || value < 0.0 ||
      value > std::numeric_limits<Real>::max() ||
      (value > 0.0 && value < std::numeric_limits<Real>::denorm_min())) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Viscosity coefficients must be finite and nonnegative: <"
              << block << ">/" << name << " = '" << token << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return static_cast<Real>(value);
}

parabolic::DiffusionSelection ParseViscosityIntegrator(const std::string &block,
                                                        ParameterInput *pin) {
  std::string integrator =
      pin->GetOrAddString(block, "viscosity_integrator", "explicit");
  if (integrator == "explicit") {
    return parabolic::DiffusionSelection::explicit_only;
  }
  if (integrator == "sts") {
    return parabolic::DiffusionSelection::sts_only;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<" << block << ">/viscosity_integrator = '" << integrator
            << "' must be 'explicit' or 'sts'" << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
// ctor:
// Note first argument passes string ("hydro" or "mhd") denoting in which class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.

Viscosity::Viscosity(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  // Read parameters for viscosity (if any)
  nu_iso = ReadViscosityCoefficient(block, "nu_iso", pin);
  nu_aniso = ReadViscosityCoefficient(block, "nu_aniso", pin);
  mode = ParseViscosityIntegrator(block, pin);
  if (nu_aniso != 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Anisotropic viscosity is not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nu_iso > 0.0 &&
      (pmy_pack->pcoord->is_special_relativistic ||
       pmy_pack->pcoord->is_general_relativistic ||
       pmy_pack->pcoord->is_dynamical_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Viscosity requires Newtonian physics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mode == parabolic::DiffusionSelection::sts_only && nu_iso <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "STS requires positive constant isotropic viscosity" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// Viscosity destructor

Viscosity::~Viscosity() {
}

//----------------------------------------------------------------------------------------
// Check trial states before EOS conversion can conceal a failed STS stage.
// Include all fine-grid ghost cells when requested: ConToPrim visits this full
// extent, even at corners not used by the viscous flux. Distinguish active-cell
// failures so callers can identify a boundary/prolongation-only rejection.
Viscosity::RawState Viscosity::InspectRawState(const DvceArray5D<Real> &u,
    const EOS_Data &eos, const DvceFaceFld4D<Real> *b, bool include_ghosts) const {
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
  const bool mhd = b != nullptr;
  DvceArray4D<Real> b1, b2, b3;
  if (mhd) { b1 = b->x1f; b2 = b->x2f; b3 = b->x3f; }
  const auto eos_data = eos;
  int invalid = 0, invalid_active = 0;
  const bool valid_eos = !eos.is_ideal ||
                        (std::isfinite(eos.gamma) && eos.gamma > 1.0);
  Kokkos::parallel_reduce("viscosity_raw_state", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*nkji),
  KOKKOS_LAMBDA(const int n, int &bad, int &bad_active) {
    const int m = n/nkji, rem = n-m*nkji;
    const int k = rem/nji+kl;
    const int j = (rem%nji)/ni+jl;
    const int i = rem%ni+il;
    if (!valid_eos || !ViscosityRawCellAdmissible(
        u, eos_data, mhd, b1, b2, b3, m, k, j, i)) {
      ++bad;
      if (i >= is && i <= ie && j >= js && j <= je && k >= ks && k <= ke) {
        ++bad_active;
      }
    }
  }, Kokkos::Sum<int>(invalid), Kokkos::Sum<int>(invalid_active));
  if (!valid_eos) { invalid = 1; invalid_active = 1; }
#if MPI_PARALLEL_ENABLED
  int invalid_global[2] = {invalid, invalid_active};
  MPI_Allreduce(MPI_IN_PLACE, invalid_global, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  invalid = invalid_global[0];
  invalid_active = invalid_global[1];
#endif
  return RawState{invalid == 0, invalid_active == 0};
}

//----------------------------------------------------------------------------------------
//! \fn void AddViscousFluxes()
//! \brief Wrapper function that adds viscous fluxes for different types of viscosity
//! to face-centered fluxes of conserved variables
//! Currently only isotropic viscosity with constant coefficient implemented

void Viscosity::AddViscousFluxes(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  if (nu_iso != 0.0) {
    AddViscousFluxIso(w0, eos, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddViscousFluxIso
//  \brief Adds viscous fluxes to face-centered fluxes of conserved variables

void Viscosity::AddViscousFluxIso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real nu_iso_ = nu_iso;

  // fluxes in x1-direction
  int scr_level = 0;
  size_t scr_size = (ScrArray1D<Real>::shmem_size(ncells1)) * 3;
  auto flx1 = flx.x1f;

  par_for_outer("visc1",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [2(dVx/dx)-(2/3)dVx/dx, dVy/dx, dVz/dx]
    par_for_inner(member, is, ie+1, [&](const int i) {
      fvx(i) = 4.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k,j,i-1))/(3.0*size.d_view(m).dx1);
      fvy(i) =     (w0(m,IVY,k,j,i) - w0(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      fvz(i) =     (w0(m,IVZ,k,j,i) - w0(m,IVZ,k,j,i-1))/size.d_view(m).dx1;
    });

    // In 2D/3D Add [(-2/3)dVy/dy, dVx/dy, 0]
    if (multi_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        fvx(i) -= ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k,j+1,i-1)) -
                   (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j-1,i-1)))/(6.0*size.d_view(m).dx2);
        fvy(i) += ((w0(m,IVX,k,j+1,i) + w0(m,IVX,k,j+1,i-1)) -
                   (w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j-1,i-1)))/(4.0*size.d_view(m).dx2);
      });
    }

    // In 3D Add [(-2/3)dVz/dz, 0,  dVx/dz]
    if (three_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        fvx(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j,i-1)) -
                   (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j,i-1)))/(6.0*size.d_view(m).dx3);
        fvz(i) += ((w0(m,IVX,k+1,j,i) + w0(m,IVX,k+1,j,i-1)) -
                   (w0(m,IVX,k-1,j,i) + w0(m,IVX,k-1,j,i-1)))/(4.0*size.d_view(m).dx3);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie+1, [&](const int i) {
      Real nud = 0.5*nu_iso_*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
      flx1(m,IVX,k,j,i) -= nud*fvx(i);
      flx1(m,IVY,k,j,i) -= nud*fvy(i);
      flx1(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx1(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->one_d) {return;}

  // fluxes in x2-direction
  auto flx2 = flx.x2f;

  par_for_outer("visc2",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [(dVx/dy+dVy/dx), 2(dVy/dy)-(2/3)(dVx/dx+dVy/dy), dVz/dy]
    par_for_inner(member, is, ie, [&](const int i) {
      fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k,j-1,i  ))/size.d_view(m).dx2 +
              ((w0(m,IVY,k,j,i+1) + w0(m,IVY,k,j-1,i+1)) -
               (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j-1,i-1)))/(4.0*size.d_view(m).dx1);
      fvy(i) = (w0(m,IVY,k,j,i) - w0(m,IVY,k,j-1,i))*4.0/(3.0*size.d_view(m).dx2) -
              ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k,j-1,i+1)) -
               (w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j-1,i-1)))/(6.0*size.d_view(m).dx1);
      fvz(i) = (w0(m,IVZ,k,j,i  ) - w0(m,IVZ,k,j-1,i  ))/size.d_view(m).dx2;
    });

    // In 3D Add [0, (-2/3)dVz/dz, dVy/dz]
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        fvy(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j-1,i)) -
                   (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j-1,i)))/(6.0*size.d_view(m).dx3);
        fvz(i) += ((w0(m,IVY,k+1,j,i) + w0(m,IVY,k+1,j-1,i)) -
                   (w0(m,IVY,k-1,j,i) + w0(m,IVY,k-1,j-1,i)))/(4.0*size.d_view(m).dx3);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real nud = 0.5*nu_iso_*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
      flx2(m,IVX,k,j,i) -= nud*fvx(i);
      flx2(m,IVY,k,j,i) -= nud*fvy(i);
      flx2(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx2(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->two_d) {return;}

  // fluxes in x3-direction
  auto flx3 = flx.x3f;

  par_for_outer("visc3",DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 2(dVz/dz)-(2/3)(dVx/dx+dVy/dy+dVz/dz)]
    par_for_inner(member, is, ie, [&](const int i) {
      fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k-1,j,i  ))/size.d_view(m).dx3 +
              ((w0(m,IVZ,k,j,i+1) + w0(m,IVZ,k-1,j,i+1)) -
               (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k-1,j,i-1)))/(4.0*size.d_view(m).dx1);
      fvy(i) = (w0(m,IVY,k,j,i  ) - w0(m,IVY,k-1,j,i  ))/size.d_view(m).dx3 +
              ((w0(m,IVZ,k,j+1,i) + w0(m,IVZ,k-1,j+1,i)) -
               (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k-1,j-1,i)))/(4.0*size.d_view(m).dx2);
      fvz(i) = (w0(m,IVZ,k,j,i) - w0(m,IVZ,k-1,j,i))*4.0/(3.0*size.d_view(m).dx3) -
              ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k-1,j,i+1)) -
               (w0(m,IVX,k,j,i-1) + w0(m,IVX,k-1,j,i-1)))/(6.0*size.d_view(m).dx1) -
              ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k-1,j+1,i)) -
               (w0(m,IVY,k,j-1,i) + w0(m,IVY,k-1,j-1,i)))/(6.0*size.d_view(m).dx2);
    });

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real nud = 0.5*nu_iso_*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
      flx3(m,IVX,k,j,i) -= nud*fvx(i);
      flx3(m,IVY,k,j,i) -= nud*fvy(i);
      flx3(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx3(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k-1,j,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k-1,j,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddViscousFluxAniso
//  \brief Currently no-op function, to be added later

void Viscosity::AddViscousFluxAniso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Viscosity::NewTimeStep()
//! \brief Compute new time step for viscosity

void Viscosity::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  // viscous timestep on MeshBlock(s) in this pack for constant isotropic viscosity
  dtnew = std::numeric_limits<float>::max();
  if (nu_iso <= 0.0) return;
  auto size = pmy_pack->pmb->mb_size;
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    Real inv_dx2_sum = 1.0/SQR(size.h_view(m).dx1);
    Real inv_dx2_max = inv_dx2_sum;
    if (pmy_pack->pmesh->multi_d) {
      Real inv_dx2 = 1.0/SQR(size.h_view(m).dx2);
      inv_dx2_sum += inv_dx2;
      inv_dx2_max = std::max(inv_dx2_max, inv_dx2);
    }
    if (pmy_pack->pmesh->three_d) {
      Real inv_dx2 = 1.0/SQR(size.h_view(m).dx3);
      inv_dx2_sum += inv_dx2;
      inv_dx2_max = std::max(inv_dx2_max, inv_dx2);
    }
    Real rate = 2.0*nu_iso*(inv_dx2_sum + inv_dx2_max/3.0);
    dtnew = std::min(dtnew, 1.0/rate);
  }
  return;
}
