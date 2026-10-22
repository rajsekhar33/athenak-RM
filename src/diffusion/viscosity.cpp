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

// Monotonized central limiter for Parrish et al. (2012), Appendix A4-A5.
// Test signs without products; form the mean without overflowing a+b.
KOKKOS_INLINE_FUNCTION
Real ViscosityMC(const Real a, const Real b) {
  if (!((a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0))) return 0.0;
  const Real lo = fmin(fabs(a), fabs(b));
  const Real hi = fmax(fabs(a), fabs(b));
  const Real mean = 0.5*lo+0.5*hi;
  const Real limited = lo < 0.5*mean ? 2.0*lo : mean;
  return a > 0.0 ? limited : -limited;
}

KOKKOS_INLINE_FUNCTION
bool ViscosityFinite(const Real value) {
  const Real largest = std::numeric_limits<Real>::max();
  return value >= -largest && value <= largest;
}

// Trace-free symmetric velocity gradient. normal=-1 denotes a cell center;
// otherwise (k,j,i) is the cell on the positive side of that face. Normal
// differences and centered transverse averages match the existing face strain.
// Composing its Laplacian below needs two primitive ghost cells, not a new halo.
KOKKOS_INLINE_FUNCTION
void Hyper4Strain(const DvceArray5D<Real> &w, const int m, const int k,
    const int j, const int i, const int ndim, const int normal,
    const Real inv_dx[3], Real strain[3][3]) {
  Real grad[3][3] = {};
  const int ni = (normal == 0), nj = (normal == 1), nk = (normal == 2);
  for (int a=0; a<3; ++a) {
    for (int d=0; d<ndim; ++d) {
      const int di = (d == 0), dj = (d == 1), dk = (d == 2);
      if (d == normal) {
        grad[a][d] = (w(m,IVX+a,k,j,i)-w(m,IVX+a,k-nk,j-nj,i-ni))*inv_dx[d];
      } else if (normal < 0) {
        grad[a][d] = 0.5*inv_dx[d]*(w(m,IVX+a,k+dk,j+dj,i+di)-
                                   w(m,IVX+a,k-dk,j-dj,i-di));
      } else {
        grad[a][d] = 0.25*inv_dx[d]*(
            (w(m,IVX+a,k+dk,j+dj,i+di)+w(m,IVX+a,k-nk+dk,j-nj+dj,i-ni+di))-
            (w(m,IVX+a,k-dk,j-dj,i-di)+w(m,IVX+a,k-nk-dk,j-nj-dj,i-ni-di)));
      }
    }
  }
  const Real trace_third = (grad[0][0]+grad[1][1]+grad[2][2])/3.0;
  for (int a=0; a<3; ++a) {
    for (int b=0; b<3; ++b) {
      strain[a][b] = 0.5*(grad[a][b]+grad[b][a]);
      if (a == b) strain[a][b] -= trace_third;
    }
  }
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
  mu_hyper4 = ReadViscosityCoefficient(block, "mu_hyper4", pin);
  mode = ParseViscosityIntegrator(block, pin);
  if (nu_aniso > 0.0 && (block != "mhd" || pmy_pack->pmesh->multilevel)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Braginskii validation currently requires uniform-grid MHD; "
              << "hydro and refinement are not supported" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if ((nu_iso > 0.0 || nu_aniso > 0.0 || mu_hyper4 > 0.0) &&
      (pmy_pack->pcoord->is_special_relativistic ||
       pmy_pack->pcoord->is_general_relativistic ||
       pmy_pack->pcoord->is_dynamical_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Viscosity requires Newtonian physics" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mu_hyper4 > 0.0) {
    const auto *mesh = pmy_pack->pmesh;
    const int ndim = mesh->three_d ? 3 : (mesh->multi_d ? 2 : 1);
    bool supported_boundaries = true;
    for (int face=0; face<2*ndim; ++face) {
      const auto bc = mesh->mesh_bcs[face];
      supported_boundaries = supported_boundaries &&
          (bc == BoundaryFlag::periodic || bc == BoundaryFlag::outflow);
    }
    if (pin->GetString(block, "eos") != "ideal" || mesh->multilevel ||
        !supported_boundaries || mesh->mb_indcs.ng < 2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "mu_hyper4 requires ideal-gas, uniform Newtonian Hydro/MHD "
                << "with periodic/outflow boundaries and nghost >= 2; "
                << "refinement is not supported" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // STS advances hyperstress separately from the FOFC-protected fluid step.
    // Explicit diffusion shares the flux array that FOFC replaces; keep that
    // combination disabled until its transport fallback is handled explicitly.
    if (mode == parabolic::DiffusionSelection::explicit_only &&
        pin->GetOrAddBoolean(block, "fofc", false)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "mu_hyper4 with fofc=true requires viscosity_integrator=sts"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (mode == parabolic::DiffusionSelection::sts_only &&
      nu_iso <= 0.0 && nu_aniso <= 0.0 && mu_hyper4 <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "STS requires a positive constant viscosity coefficient" << std::endl;
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
//! Isotropic entry point shared by hydro and MHD. The MHD caller separately
//! adds Braginskii fluxes with explicit current-stage magnetic fields.

void Viscosity::AddViscousFluxes(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  if (nu_iso != 0.0) {
    AddViscousFluxIso(w0, eos, flx);
  }
  if (mu_hyper4 != 0.0) {
    AddViscousFluxHyper4(w0, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \brief Fourth-order constant-dynamic hyperstress and ideal-gas energy flux.
// Continuum multipolar work balance: Necas, Novotny & Silhavy (1989),
// eqs. 2.12/2.26. This centered FV adaptation is not a copied published scheme.
// tau_ij=-2*mu*Lap(S_ij), J_j=2*mu*S_ab*partial_j(S_ab), with summed a,b.
// Add -tau_ij to momentum flux and -(v_i*tau_ij+J_j) to total-energy flux.
// The periodic force is globally dissipative. The matched semidiscrete heat
// is nonnegative in 1D; multidimensional local heating positivity is NOT claimed.
// Outflow retains the ordinary extrapolated ghosts for interior faces and
// imposes zero hyperviscous momentum/work flux at physical domain faces.
// Advective, conductive and ordinary-viscosity boundary fluxes are unchanged.
void Viscosity::AddViscousFluxHyper4(const DvceArray5D<Real> &w0,
    DvceFaceFld5D<Real> &flx) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ndim = pmy_pack->pmesh->three_d ? 3 :
                   (pmy_pack->pmesh->multi_d ? 2 : 1);
  const int nmb = pmy_pack->nmb_thispack;
  const Real mu = mu_hyper4;
  auto size = pmy_pack->pmb->mb_size;
  auto mb_bcs = pmy_pack->pmb->mb_bcs.d_view;
  for (int normal=0; normal<ndim; ++normal) {
    auto flux = normal == 0 ? flx.x1f : (normal == 1 ? flx.x2f : flx.x3f);
    const int ni = (normal == 0), nj = (normal == 1), nk = (normal == 2);
    par_for("hyper4_flux", DevExeSpace(), 0, nmb-1, ks, ke+nk, js, je+nj, is, ie+ni,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int face_index = ni*i+nj*j+nk*k;
      const int inner = ni*is+nj*js+nk*ks;
      const int outer = ni*(ie+1)+nj*(je+1)+nk*(ke+1);
      if ((face_index == inner && mb_bcs(m,2*normal) == BoundaryFlag::outflow) ||
          (face_index == outer && mb_bcs(m,2*normal+1) == BoundaryFlag::outflow)) {
        return;
      }
      const Real inv_dx[3] = {1.0/size.d_view(m).dx1,
                             ndim > 1 ? 1.0/size.d_view(m).dx2 : 0.0,
                             ndim > 2 ? 1.0/size.d_view(m).dx3 : 0.0};
      Real sf[3][3], sl[3][3], sr[3][3];
      Hyper4Strain(w0, m, k, j, i, ndim, normal, inv_dx, sf);
      Real lap_strain[3] = {};
      for (int d=0; d<ndim; ++d) {
        const int di = (d == 0), dj = (d == 1), dk = (d == 2);
        Real sp[3][3], sm[3][3];
        Hyper4Strain(w0, m, k+dk, j+dj, i+di, ndim, normal, inv_dx, sp);
        Hyper4Strain(w0, m, k-dk, j-dj, i-di, ndim, normal, inv_dx, sm);
        for (int a=0; a<3; ++a) {
          lap_strain[a] += (sp[a][normal]-2.0*sf[a][normal]+sm[a][normal])*
                          SQR(inv_dx[d]);
        }
      }
      Hyper4Strain(w0, m, k-nk, j-nj, i-ni, ndim, -1, inv_dx, sl);
      Hyper4Strain(w0, m, k, j, i, ndim, -1, inv_dx, sr);
      Real work = 0.0;
      for (int a=0; a<3; ++a) {
        const Real traction = -2.0*mu*lap_strain[a];
        flux(m,IVX+a,k,j,i) -= traction;
        work += 0.5*(w0(m,IVX+a,k,j,i)+w0(m,IVX+a,k-nk,j-nj,i-ni))*traction;
      }
      for (int a=0; a<3; ++a) {
        for (int b=0; b<3; ++b) {
          work += 2.0*mu*sf[a][b]*(sr[a][b]-sl[a][b])*inv_dx[normal];
        }
      }
      flux(m,IEN,k,j,i) -= work;
    });
  }
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
//  \brief Centered parallel Braginskii stress and conservative face work.
// DIAGNOSTIC centered transverse reconstruction: Kempf, Rincon & Clerc (2023),
// Appendix B, https://arxiv.org/abs/2306.12460. This is a local candidate, not
// an unconditional positivity guarantee. Conservative face stress/work retain
// the Athena-C form; no pairwise work veto or heating redistribution is added.
// Explicit and RKL2 uniform-grid MHD integration are under validation.
// The constructor continues to reject mesh refinement.

void Viscosity::AddViscousFluxAniso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    const DvceFaceFld4D<Real> &bface, const DvceArray5D<Real> &bcc,
    DvceFaceFld5D<Real> &flx) {
  if (nu_aniso == 0.0) return;
  const auto &idx = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const auto size = pmy_pack->pmb->mb_size;
  const Real nu = nu_aniso;
  const bool energy = eos.is_ideal;
  for (int dir=0; dir<ndim; ++dir) {
    const auto normal_b = dir == 0 ? bface.x1f : (dir == 1 ? bface.x2f : bface.x3f);
    auto flux = dir == 0 ? flx.x1f : (dir == 1 ? flx.x2f : flx.x3f);
    par_for("braginskii_face_flux", DevExeSpace(), 0, pmy_pack->nmb_thispack-1,
      idx.ks, idx.ke+(dir == 2), idx.js, idx.je+(dir == 1),
      idx.is, idx.ie+(dir == 0),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int il = i-(dir == 0), jl = j-(dir == 1), kl = k-(dir == 2);
      const Real dx[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
      Real b[3];
      for (int c=0; c<3; ++c) {
        b[c] = c == dir ? normal_b(m,k,j,i) :
            0.5*bcc(m,c,k,j,i)+0.5*bcc(m,c,kl,jl,il);
      }
      // Scale before normalization to avoid squaring very large/small fields.
      const Real scale = fmax(fabs(b[0]),fmax(fabs(b[1]),fabs(b[2])));
      // Diagnostic convention only: no directional stress at exactly zero B.
      // This does not model a physical transition to unmagnetized viscosity.
      if (scale == 0.0) return;
      for (int c=0; c<3; ++c) b[c] /= scale;
      const Real norm = sqrt(SQR(b[0])+SQR(b[1])+SQR(b[2]));
      for (int c=0; c<3; ++c) b[c] /= norm;
      Real grad[3][3] = {};
      for (int c=0; c<3; ++c) {
        const int v = IVX+c;
        grad[c][dir] = (w0(m,v,k,j,i)-w0(m,v,kl,jl,il))/dx[dir];
        for (int d=0; d<ndim; ++d) {
          if (d == dir) continue;
          const int ti = d == 0, tj = d == 1, tk = d == 2;
          // Centered asymmetric face gradient, without nonlinear limiting.
          // IDEFIX's Cartesian nolimiter form; not a positivity guarantee.
          const Real upper = 0.5*(w0(m,v,k+tk,j+tj,i+ti)-w0(m,v,k,j,i))+
              0.5*(w0(m,v,kl+tk,jl+tj,il+ti)-w0(m,v,kl,jl,il));
          const Real lower = 0.5*(w0(m,v,k,j,i)-w0(m,v,k-tk,j-tj,i-ti))+
              0.5*(w0(m,v,kl,jl,il)-w0(m,v,kl-tk,jl-tj,il-ti));
          grad[c][d] = (0.5*upper+0.5*lower)/dx[d];
        }
      }
      Real strain = -(grad[0][0]+grad[1][1]+grad[2][2])/3.0;
      for (int c=0; c<3; ++c) {
        for (int d=0; d<ndim; ++d) strain += b[c]*b[d]*grad[c][d];
      }
      const Real mu = nu*(0.5*w0(m,IDN,k,j,i)+0.5*w0(m,IDN,kl,jl,il));
      Real work = 0.0;
      for (int c=0; c<3; ++c) {
        const Real stress = mu*strain*(3.0*b[c]*b[dir]-(c == dir ? 1.0 : 0.0));
        flux(m,IM1+c,k,j,i) -= stress;
        work += (0.5*w0(m,IVX+c,k,j,i)+0.5*w0(m,IVX+c,kl,jl,il))*stress;
      }
      if (energy) flux(m,IEN,k,j,i) -= work;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Viscosity::NewTimeStep()
//! \brief Compute new time step for viscosity

void Viscosity::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  // Preserve the previous constant-kinematic arithmetic when mu_hyper4=0.
  // Constant-dynamic hyperviscosity additionally depends on current density.
  dtnew = std::numeric_limits<float>::max();
  if (nu_iso <= 0.0 && nu_aniso <= 0.0 && mu_hyper4 <= 0.0) return;
  Real rho_min = std::numeric_limits<Real>::max();
  if (mu_hyper4 > 0.0) {
    const auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2;
    const int nji = nx1*nx2, ncell = nji*indcs.nx3;
    int invalid = 0;
    Kokkos::parallel_reduce("hyper4_density_dt", Kokkos::RangePolicy<>(
        DevExeSpace(), 0, pmy_pack->nmb_thispack*ncell),
    KOKKOS_LAMBDA(const int n, Real &minimum, int &bad) {
      const int m = n/ncell, cell = n-m*ncell;
      const int k = ks+cell/nji, j = js+(cell%nji)/nx1, i = is+cell%nx1;
      const Real density = w0(m,IDN,k,j,i);
      if (!ViscosityFinite(density) || !(density > 0.0)) {
        ++bad;
      } else {
        minimum = fmin(minimum, density);
      }
    }, Kokkos::Min<Real>(rho_min), Kokkos::Sum<int>(invalid));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (invalid != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << ": mu_hyper4 requires finite positive active density" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  auto size = pmy_pack->pmb->mb_size;
  int invalid_hyper_dt = 0;
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
    if (nu_aniso > 0.0) {
      // Berlok et al. (2020), eq. 52: 2*d*nu/dx_min^2; include the
      // longitudinal eigenvalue factor 4/3. Mesh::NewTimeStep applies CFL.
      // Add rates when both viscosities act, rather than taking separate minima.
      const int ndim = pmy_pack->pmesh->three_d ? 3 :
                       (pmy_pack->pmesh->multi_d ? 2 : 1);
      rate += (8.0/3.0)*ndim*nu_aniso*inv_dx2_max;
    }
    if (mu_hyper4 > 0.0) {
      // Conservative norm bound for the composed centered hyperstress:
      // lambda_max <= (4/3)*(mu/rho_min)*(4*sum_d dx_d^-2)^2.
      // This rate is lambda_max/2, i.e. the inverse forward-Euler limit.
      // The minimum is local to this pack; Mesh reduces the timestep over ranks.
      const Real hyper_rate = (32.0/3.0)*(mu_hyper4/rho_min)*SQR(inv_dx2_sum);
      rate += hyper_rate;
      if (!std::isfinite(hyper_rate) || !(hyper_rate > 0.0) ||
          !std::isfinite(rate) || !(1.0/rate > 0.0)) {
        invalid_hyper_dt = 1;
      }
    }
    dtnew = std::min(dtnew, 1.0/rate);
  }
  if (mu_hyper4 > 0.0) {
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &invalid_hyper_dt, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (invalid_hyper_dt != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << ": unrepresentable mu_hyper4 timestep" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  return;
}
