//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.cpp
//! \brief Implements functions for Conduction class. This includes isotropic thermal
//! conduction, in which heat flux is proportional to negative local temperature gradient.
//! Conduction may be added to Hydro and/or MHD independently.

#include <float.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "conduction.hpp"
#include "conduction_closure.hpp"
#include "units/units.hpp"

namespace {

parabolic::DiffusionSelection ParseConductivityIntegrator(const std::string &block,
                                                           ParameterInput *pin) {
  std::string integrator =
      pin->GetOrAddString(block, "conductivity_integrator", "explicit");
  if (integrator == "explicit") {
    return parabolic::DiffusionSelection::explicit_only;
  }
  if (integrator == "sts") {
    return parabolic::DiffusionSelection::sts_only;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<" << block << ">/conductivity_integrator = '" << integrator
            << "' must be 'explicit' or 'sts'" << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

// VanLeer Limiter which takes 2 slopes
KOKKOS_INLINE_FUNCTION
Real VLL2State(const Real a, const Real b) {
  if (a*b > 0) {
    return 2.0*a*b/(a+b);
  } else {
    return 0.0;
  }
}

// VanLeer Limiter which takes 4 slopes
KOKKOS_INLINE_FUNCTION
Real VLL4State(const Real a, const Real b, const Real c, const Real d) {
  return VLL2State(VLL2State(a,b), VLL2State(c,d));
}

// Monotonized-central limiter: Sharma & Hammett (2007), equation 18.
// Keep the van-Leer default available for matched numerical-diffusion controls.
KOKKOS_INLINE_FUNCTION
Real MCL2State(const Real a, const Real b) {
  if ((a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0)) {
    Real limited = fmin(0.5*fabs(a)+0.5*fabs(b),2.0*fmin(fabs(a),fabs(b)));
    return a > 0.0 ? limited : -limited;
  }
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \brief Conduction constructor
// Note first argument passes string ("hydro" or "mhd") denoting in wihch class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.
// Note that the coefficients of thermal conduction, alpha_iso, etc., correspond to
// diffusivities. The conductivity is kappa = (dens)*alpha, and the energy flux
// q = -kappa * (dT/dx) = - alpha * d * *dT/dx)

Conduction::Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  dt_transport = dtnew;
  // Read parameters for thermal diffusivity (if any)
  alpha_iso = pin->GetOrAddReal(block,"alpha_iso", 0.0);
  alpha_aniso = pin->GetOrAddReal(block,"alpha_aniso", 0.0);
  alpha_spitzer = pin->GetOrAddBoolean(block,"alpha_spitzer", false);
  std::string limiter = pin->GetOrAddString(block,"anisotropic_limiter","vanleer");
  if (limiter != "vanleer" && limiter != "mc") {
    std::cerr << "anisotropic_limiter must be vanleer or mc" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  use_mc_limiter = limiter == "mc";
  if (!std::isfinite(alpha_iso) || !std::isfinite(alpha_aniso) ||
      alpha_iso < 0.0 || alpha_aniso < 0.0 || (alpha_aniso > 0.0 && block != "mhd")) {
    std::cerr << "Conduction requires finite nonnegative alpha coefficients; "
              << "anisotropic conduction requires MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string saturation =
      pin->GetOrAddString(block, "conduction_saturation", "none");
  if (saturation != "none" && saturation != "cowie_mckee") {
    std::cerr << "conduction_saturation must be none or cowie_mckee" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  saturated = saturation == "cowie_mckee";
  if (!saturated && pin->DoesParameterExist(block, "saturation_phi")) {
    std::cerr << "saturation_phi requires conduction_saturation=cowie_mckee"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (saturated) {
    saturation_phi = pin->GetOrAddReal(block, "saturation_phi", 0.3);
    if (!std::isfinite(saturation_phi) || saturation_phi < 0.0) {
      std::cerr << "saturation_phi must be finite and nonnegative" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // Preserve the legacy disabled sentinel. Physical saturation is selected
  // above and depends on the local gas state, not an absolute flux ceiling.
  q_limit = pin->GetOrAddReal(block,"q_limit",
                     static_cast<Real>(std::numeric_limits<float>::max()));
  // Saturation is a separate closure, not an implicit conductivity ceiling.
  // GetOrAddReal serializes legacy default values with six significant digits.
  // Accept its rounded float-max sentinel as well, including on restart.
  if (!std::isfinite(q_limit) || q_limit < static_cast<Real>(3.40282e38)) {
    std::cerr << "q_limit must be finite and at least the legacy disabled "
              << "sentinel (3.40282e38); select physical saturation with "
              << "conduction_saturation=cowie_mckee" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  mode = ParseConductivityIntegrator(block, pin);
  const bool has_code = pin->DoesParameterExist(block, "spitzer_kappa0_code");
  const bool has_cgs = pin->DoesParameterExist(block, "spitzer_kappa0_cgs");
  const bool has_geometry = pin->DoesParameterExist(block, "spitzer_geometry");
  if (!alpha_spitzer && (has_code || has_cgs || has_geometry)) {
    std::cerr << "Spitzer parameters require alpha_spitzer=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (alpha_spitzer) {
    if (has_code == has_cgs) {
      std::cerr << "Spitzer requires exactly one of spitzer_kappa0_code and "
                << "spitzer_kappa0_cgs" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (has_code) {
      spitzer_kappa0_code = pin->GetReal(block, "spitzer_kappa0_code");
    } else {
      if (pmy_pack->punit == nullptr) {
        std::cerr << "spitzer_kappa0_cgs requires an explicit units block" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      auto *pu = pmy_pack->punit;
      conduction::ConductivityUnits units{pu->pressure_cgs(), pu->velocity_cgs(),
                                          pu->length_cgs(), pu->temperature_cgs()};
      const Real k0_cgs = pin->GetReal(block, "spitzer_kappa0_cgs");
      spitzer_kappa0_code = conduction::SpitzerCoefficientCode(k0_cgs, units);
      if (k0_cgs > 0.0 && spitzer_kappa0_code == 0.0) {
        std::cerr << "Spitzer coefficient conversion underflows code precision"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Do not add the derived code coefficient to ParameterInput: restart
      // must retain exactly the originally selected coefficient convention.
    }
    if (!std::isfinite(spitzer_kappa0_code) || spitzer_kappa0_code < 0.0) {
      std::cerr << "Spitzer coefficient must be finite and nonnegative" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (block == "mhd" && !has_geometry) {
      std::cerr << "MHD Spitzer requires explicit spitzer_geometry selection"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const std::string geometry =
        pin->GetOrAddString(block, "spitzer_geometry", "isotropic");
    if (geometry != "isotropic" && geometry != "field_aligned") {
      std::cerr << "spitzer_geometry must be isotropic or field_aligned" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    spitzer_field_aligned = geometry == "field_aligned";
    if (spitzer_field_aligned && block != "mhd") {
      std::cerr << "Field-aligned Spitzer conduction requires MHD" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (alpha_spitzer || saturated) {
    const Real cfl = pin->GetReal("time", "cfl_number");
    if (!std::isfinite(cfl) || cfl <= 0.0 || cfl > 1.0) {
      std::cerr << "Spitzer/saturation requires finite 0 < cfl_number <= 1"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const std::string rk = pin->GetOrAddString("time", "integrator", "rk2");
    if (rk != "rk1" && rk != "rk2" && rk != "rk3") {
      std::cerr << "Spitzer/saturation requires SSP rk1, rk2 or rk3 for the "
                << "non-STS part of the update" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // This energy-flux closure is Newtonian, not a relativistic heat-flux model.
  if ((alpha_aniso > 0.0 || alpha_spitzer || saturated) &&
      (pmy_pack->pcoord->is_special_relativistic ||
      pmy_pack->pcoord->is_general_relativistic ||
      pmy_pack->pcoord->is_dynamical_relativistic)) {
    std::cerr << "Anisotropic/nonlinear conduction requires Newtonian physics"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mode == parabolic::DiffusionSelection::sts_only &&
      !(alpha_iso > 0.0 || alpha_aniso > 0.0 || spitzer_kappa0_code > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "STS requires positive thermal conduction"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Conduction destructor

Conduction::~Conduction() {
}

// Coordinate failures before any rank enters the next exchange. In particular,
// a bad face on one rank must not leave its neighbors waiting for flux messages.
void Conduction::FailIfInvalid(int local_count, const char *where) const {
  int invalid = local_count;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (invalid != 0) {
    std::cerr << "### FATAL ERROR: nonlinear conduction: " << where << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }
}

// Inspect raw conserved states before EOS conversion can repair them. Include
// the consumed one-cell stencil halo, but not unused triple-corner ghost cells.
// EOS conversion is performed only on local copies, with the current CT field.
void Conduction::ValidateRawState(const DvceArray5D<Real> &u0, const EOS_Data &eos_data,
                                  const DvceFaceFld4D<Real> *b) const {
  if (!saturated && (!alpha_spitzer || spitzer_kappa0_code == 0.0)) return;
  const auto state = InspectRawState(u0, eos_data, b);
  FailIfInvalid(state.invalid,
                "raw conserved state is invalid or would require an EOS floor");
}

Conduction::ThermalState Conduction::InspectRawState(const DvceArray5D<Real> &u0,
    const EOS_Data &eos_data, const DvceFaceFld4D<Real> *b, bool active_only) const {
  FailIfInvalid(!eos_data.is_ideal || !std::isfinite(eos_data.gamma) ||
                eos_data.gamma <= 1.0, "invalid ideal-gas EOS");
  const auto &idx = pmy_pack->pmesh->mb_indcs;
  const bool multi = pmy_pack->pmesh->multi_d;
  const bool three = pmy_pack->pmesh->three_d;
  const bool edges = saturated || spitzer_field_aligned || alpha_aniso > 0.0;
  const int is = idx.is, ie = idx.ie, js = idx.js, je = idx.je;
  const int ks = idx.ks, ke = idx.ke;
  const int ni = idx.nx1+2, nj = idx.nx2+(multi ? 2 : 0);
  const int nk = idx.nx3+(three ? 2 : 0);
  const int nji = nj*ni, nkji = nk*nji;
  const bool mhd = b != nullptr;
  DvceArray4D<Real> b1, b2, b3;
  if (mhd) { b1 = b->x1f; b2 = b->x2f; b3 = b->x3f; }
  const auto eos = eos_data;
  int invalid = 0;
  Real tmin = std::numeric_limits<Real>::max(), tmax = 0.0;
  Kokkos::parallel_reduce("spitzer_raw_state", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*nkji),
  KOKKOS_LAMBDA(const int n, int &bad, Real &minimum, Real &maximum) {
    const int m = n/nkji, rem = n-m*nkji;
    const int k = rem/nji + ks-(three ? 1 : 0);
    const int j = (rem%nji)/ni + js-(multi ? 1 : 0);
    const int i = rem%ni + is-1;
    const int outside = (i < is || i > ie)+(j < js || j > je)+(k < ks || k > ke);
    if (outside > (active_only ? 0 : (edges ? 2 : 1))) return;
    HydCons1D u;
    u.d = u0(m,IDN,k,j,i);
    u.mx = u0(m,IM1,k,j,i); u.my = u0(m,IM2,k,j,i); u.mz = u0(m,IM3,k,j,i);
    u.e = u0(m,IEN,k,j,i);
    if (!conduction::IsFinitePositive(u.d) || !conduction::IsFinite(u.e) ||
        !conduction::IsFinite(u.mx) || !conduction::IsFinite(u.my) ||
        !conduction::IsFinite(u.mz)) { ++bad; return; }
    bool df = false, ef = false, tf = false;
    HydPrim1D w;
    if (mhd) {
      MHDCons1D um;
      um.d = u.d; um.mx = u.mx; um.my = u.my; um.mz = u.mz; um.e = u.e;
      const Real bx0 = b1(m,k,j,i), bx1 = b1(m,k,j,i+1);
      const Real by0 = b2(m,k,j,i), by1 = b2(m,k,j+1,i);
      const Real bz0 = b3(m,k,j,i), bz1 = b3(m,k+1,j,i);
      if (!conduction::IsFinite(bx0) || !conduction::IsFinite(bx1) ||
          !conduction::IsFinite(by0) || !conduction::IsFinite(by1) ||
          !conduction::IsFinite(bz0) || !conduction::IsFinite(bz1)) { ++bad; return; }
      um.bx = 0.5*bx0+0.5*bx1; um.by = 0.5*by0+0.5*by1; um.bz = 0.5*bz0+0.5*bz1;
      SingleC2P_IdealMHD(um, eos, w, df, ef, tf);
    } else {
      SingleC2P_IdealHyd(u, eos, w, df, ef, tf);
    }
    const Real temp = conduction::SpitzerTemperature(u.d, w.e, eos.gamma-1.0);
    if (df || ef || tf || !conduction::IsFiniteNonNegative(w.e) ||
        !conduction::IsFinite(w.vx) || !conduction::IsFinite(w.vy) ||
        !conduction::IsFinite(w.vz) || !conduction::IsFiniteNonNegative(temp)) {
      ++bad;
      return;
    }
    minimum = fmin(minimum, temp);
    maximum = fmax(maximum, temp);
  }, Kokkos::Sum<int>(invalid), Kokkos::Min<Real>(tmin), Kokkos::Max<Real>(tmax));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  return ThermalState{invalid, tmin, tmax};
}

bool Conduction::UsesNonlinearSTS() const {
  return mode == parabolic::DiffusionSelection::sts_only &&
         (saturated || (alpha_spitzer && spitzer_kappa0_code > 0.0));
}

void Conduction::BeginSTSTrial(const DvceArray5D<Real> &u,
    const DvceArray5D<Real> &w, DvceArray5D<Real> &snapshot) {
  // AMR may resize the owner arrays between completed cycles.
  bool resize = false;
  for (int d=0; d<5; ++d) resize |= sts_w0.extent(d) != w.extent(d);
  if (resize) Kokkos::realloc(sts_w0, w.extent(0), w.extent(1), w.extent(2),
                            w.extent(3), w.extent(4));
  Kokkos::deep_copy(DevExeSpace(), snapshot, u);
  Kokkos::deep_copy(DevExeSpace(), sts_w0, w);
}

void Conduction::RestoreSTSTrial(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
    const DvceArray5D<Real> &snapshot) {
  Kokkos::deep_copy(DevExeSpace(), u, snapshot);
  Kokkos::deep_copy(DevExeSpace(), w, sts_w0);
}

bool Conduction::CheckSTSTransport(const DvceArray5D<Real> &w,
    const EOS_Data &eos_data, Real interval) const {
  if (!saturated) return true;
  Real bound = SaturationTimeStep(w, eos_data, true);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &bound, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  bound *= pmy_pack->pmesh->cfl_no;
  return std::isfinite(interval) && interval > 0.0 &&
         interval <= bound*(1.0+64.0*std::numeric_limits<Real>::epsilon());
}

// Positive harmonic face conductivities are <= K(Tmax). For scalar isotropic
// forward Euler this bounds the current-state convex-combination row sum;
// derivatives of the temperature-dependent coefficients are not needed for
// that argument. Check every SSP stage, not just the start of the RK step.
// The factor four per dimension for limited field-aligned transverse terms is
// deliberately conservative, not a general nonlinear stability theorem.
Real Conduction::SpitzerTimeStep(const DvceArray5D<Real> &w0,
                                 const EOS_Data &eos_data,
                                 bool include_saturation) const {
  const auto &idx = pmy_pack->pmesh->mb_indcs;
  const bool multi = pmy_pack->pmesh->multi_d, three = pmy_pack->pmesh->three_d;
  const int ndim = three ? 3 : (multi ? 2 : 1);
  const bool edges = saturated || spitzer_field_aligned || alpha_aniso > 0.0;
  const int is = idx.is, ie = idx.ie, js = idx.js, je = idx.je;
  const int ks = idx.ks, ke = idx.ke;
  const int ni = idx.nx1+2, nj = idx.nx2+(multi ? 2 : 0);
  const int nk = idx.nx3+(three ? 2 : 0);
  const int nji = nj*ni, nkji = nk*nji;
  const Real gm1 = eos_data.gamma-1.0;
  int invalid = 0;
  Real tmax = 0.0;
  Kokkos::parallel_reduce("spitzer_temperature_cap", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*nkji),
  KOKKOS_LAMBDA(const int n, Real &maximum, int &bad) {
    const int m = n/nkji, rem = n-m*nkji;
    const int k = rem/nji + ks-(three ? 1 : 0);
    const int j = (rem%nji)/ni + js-(multi ? 1 : 0);
    const int i = rem%ni + is-1;
    const int outside = (i < is || i > ie)+(j < js || j > je)+(k < ks || k > ke);
    if (outside > (edges ? 2 : 1)) return;
    const Real temp = conduction::SpitzerTemperature(
        w0(m,IDN,k,j,i), w0(m,IEN,k,j,i), gm1);
    if (!conduction::IsFiniteNonNegative(temp)) { ++bad; return; }
    maximum = fmax(maximum, temp);
  }, Kokkos::Max<Real>(tmax), Kokkos::Sum<int>(invalid));
  FailIfInvalid(invalid, "invalid thermal state in timestep stencil");
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  const Real kcap = conduction::SpitzerConductivity(tmax, spitzer_kappa0_code);
  // Positive conductivity underflow is not a physical zero coefficient.
  // Reject it consistently with the face operator instead of dropping dt.
  FailIfInvalid(!std::isfinite(kcap) || kcap < 0.0 ||
                (spitzer_kappa0_code > 0.0 && tmax > 0.0 && kcap == 0.0),
                "unrepresentable conductivity cap");
  // Whole-vector saturation couples transverse gradients even for isotropic
  // transport. Retain an unsuppressed diffusion budget, not only a cap speed.
  const Real alpha0 = saturated ? 4.0*ndim*(alpha_iso+alpha_aniso) :
                                  alpha_iso+4.0*ndim*alpha_aniso;
  const Real kfactor = (saturated || spitzer_field_aligned) ? 4.0*ndim : 1.0;
  const auto size = pmy_pack->pmb->mb_size;
  const int nx1 = idx.nx1, nx2 = idx.nx2, nx3 = idx.nx3;
  const int active_nji = nx1*nx2, active_nkji = active_nji*nx3;
  Real bound = std::numeric_limits<float>::max();
  invalid = 0;
  Kokkos::parallel_reduce("spitzer_dt", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*active_nkji),
  KOKKOS_LAMBDA(const int n, Real &minimum, int &bad) {
    const int m = n/active_nkji, rem = n-m*active_nkji;
    const int k = rem/active_nji+ks, j = (rem%active_nji)/nx1+js, i = rem%nx1+is;
    const Real rho = w0(m,IDN,k,j,i);
    const Real heat_capacity_inverse = gm1/rho;
    const bool active_diffusion = alpha0 > 0.0 || kcap > 0.0;
    if (active_diffusion &&
        (!conduction::IsFinitePositive(heat_capacity_inverse) ||
         heat_capacity_inverse < conduction::kMinNormal)) {
      ++bad;
      return;
    }
    const Real dx[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
    Real inverse_square = 0.0, row_sum = 0.0;
    for (int d=0; d<ndim; ++d) {
      if (!conduction::IsFinitePositive(dx[d])) { ++bad; return; }
      const Real dsq = 1.0/SQR(dx[d]);
      // A finite dx can overflow when squared. Do not let its zero reciprocal
      // silently erase a representable K/dx^2 rate for a large conductivity.
      if (active_diffusion &&
          (!conduction::IsFinitePositive(dsq) || dsq < conduction::kMinNormal)) {
        ++bad;
        return;
      }
      inverse_square += dsq;
      const int di = d == 0, dj = d == 1, dk = d == 2;
      // A lost density-weighted term can be rescued later by alpha/rho.
      // Reject that unsupported intermediate range before accumulating it.
      if (alpha0 > 0.0) {
        const Real row_term = 0.5*(w0(m,IDN,k-dk,j-dj,i-di)+2.0*rho+
                                   w0(m,IDN,k+dk,j+dj,i+di))*dsq;
        if (!conduction::IsFinitePositive(row_term) ||
            row_term < conduction::kMinNormal) { ++bad; return; }
      }
      row_sum += 0.5*(w0(m,IDN,k-dk,j-dj,i-di)+2.0*rho+
                       w0(m,IDN,k+dk,j+dj,i+di))*dsq;
    }
    // Check each active term separately: a positive contribution from the
    // other closure must not mask zero/subnormal loss before multiplication
    // by inverse heat capacity. Keep ordinary rate arithmetic unchanged.
    if (alpha0 > 0.0) {
      const Real alpha_term = alpha0*row_sum;
      if (!conduction::IsFinitePositive(alpha_term) ||
          alpha_term < conduction::kMinNormal) { ++bad; return; }
    }
    if (kcap > 0.0) {
      const Real spitzer_term = 2.0*kfactor*kcap*inverse_square;
      if (!conduction::IsFinitePositive(spitzer_term) ||
          spitzer_term < conduction::kMinNormal) { ++bad; return; }
    }
    const Real rate = heat_capacity_inverse*
        (alpha0*row_sum+2.0*kfactor*kcap*inverse_square);
    if (!conduction::IsFiniteNonNegative(rate) ||
        (active_diffusion && rate < conduction::kMinNormal)) { ++bad; return; }
    if (rate > 0.0) minimum = fmin(minimum, 1.0/rate);
  }, Kokkos::Min<Real>(bound), Kokkos::Sum<int>(invalid));
  FailIfInvalid(invalid || !std::isfinite(bound) || bound <= 0.0,
                "invalid explicit conduction timestep");
  if (saturated && include_saturation) {
    // Saturation's donor-pressure response is hyperbolic. Add its rate; it
    // must never be counted as an independently STS-accelerated diffusion rate.
    const Real adv_bound = SaturationTimeStep(w0, eos_data);
    const Real rate = 1.0/bound+1.0/adv_bound;
    FailIfInvalid(!std::isfinite(rate) || rate < 0.0,
                  "invalid combined diffusion/saturation rate");
    bound = rate > 0.0 ? 1.0/rate : std::numeric_limits<float>::max();
  }
  return bound;
}

void Conduction::CheckExplicitStage(const DvceArray5D<Real> &w0,
                                    const EOS_Data &eos_data) {
  if (!saturated && (!alpha_spitzer || spitzer_kappa0_code == 0.0)) return;
  // In SSP RK1/2/3 every constituent forward-Euler interval is dt, including
  // RK2 stage 2 (beta/gam0 = 1). Never change the committed mesh dt mid-step.
  // Mesh::NewTimeStep already applies the requested CFL safety factor when
  // selecting dt. Here check the current forward-Euler bound itself: harmless
  // stage heating/compression may use a little of that reserved margin.
  const Real bound = SpitzerTimeStep(w0, eos_data);
  const Real dt = pmy_pack->pmesh->dt;
  const Real tolerance = 64.0*std::numeric_limits<Real>::epsilon();
  FailIfInvalid(!std::isfinite(bound) || bound <= 0.0 ||
                !std::isfinite(dt) || dt <= 0.0 || dt > bound*(1.0+tolerance),
                "current SSP stage exceeds the temperature-dependent CFL bound");
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxes()
//! \brief Wrapper function that adds heat fluxes for different types of thermal
//! conduction to face-centered fluxes of conserved variables

void Conduction::AddHeatFluxes(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  if (saturated) {
    AddHeatFluxSaturated(w0, eos, flx);
    return;
  }
  if (alpha_iso != 0) {
    AddHeatFluxIso(w0, eos, flx);
  }
  if (alpha_aniso != 0) {
    AddHeatFluxAniso(w0, eos, flx);
  }
  if (alpha_spitzer) {
    AddHeatFluxSpitzer(w0, eos, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxIso()
//! \brief Adds isotropic heat flux computed using constant conductivity to face-centered
//! fluxes of conserved variables

void Conduction::AddHeatFluxIso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  Real gm1 = eos.gamma-1.0;
  Real &alpha_ = alpha_iso;

  // fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j,i  )/w0(m,IDN,k,j,i  );
    Real templ = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx1;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
    flx1(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->one_d) {
    return;
  }

  // fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j  ,i)/w0(m,IDN,k,j  ,i);
    Real templ = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx2;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
    flx2(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->two_d) {
    return;
  }

  // fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k  ,j,i)/w0(m,IDN,k  ,j,i);
    Real templ = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx3;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
    flx3(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxAniso()
//! \brief Conservative constant-alpha field-aligned heat flux. Transverse slopes
//! use the limited asymmetric method of Sharma & Hammett (2007), equations 5,17.
//! alpha_aniso is ADDITIONAL parallel diffusion above alpha_iso. Exactly zero
//! face field gives zero anisotropic flux; no arbitrary weak-field threshold.

void Conduction::AddHeatFluxAniso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const int nmb1 = pmy_pack->nmb_thispack-1;
  auto w = w0;
  auto bcc = pmy_pack->pmhd->bcc0;
  auto size = pmy_pack->pmb->mb_size;
  const Real gm1 = eos.gamma-1.0, alpha = alpha_aniso;
  const bool mc = use_mc_limiter;
  for (int dir=0; dir<ndim; ++dir) {
    auto f = dir == 0 ? flx.x1f : (dir == 1 ? flx.x2f : flx.x3f);
    auto bf = dir == 0 ? pmy_pack->pmhd->b0.x1f :
              (dir == 1 ? pmy_pack->pmhd->b0.x2f : pmy_pack->pmhd->b0.x3f);
    const int di = dir == 0, dj = dir == 1, dk = dir == 2;
    par_for("conduct_aniso", DevExeSpace(), 0,nmb1, ind.ks,ind.ke+dk,
            ind.js,ind.je+dj, ind.is,ind.ie+di,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int il=i-di, jl=j-dj, kl=k-dk;
      Real b[3];
      for (int d=0; d<3; ++d) {
        b[d] = d == dir ? bf(m,k,j,i) :
               0.5*bcc(m,d,k,j,i)+0.5*bcc(m,d,kl,jl,il);
      }
      // Normalize all three components, including Bz in a 2-D calculation.
      // Scaling first avoids underflow/overflow when field magnitudes vary.
      Real scale = fmax(fabs(b[0]),fmax(fabs(b[1]),fabs(b[2])));
      if (scale == 0.0) return;
      for (int d=0; d<3; ++d) b[d] /= scale;
      Real norm = sqrt(SQR(b[0])+SQR(b[1])+SQR(b[2]));
      for (int d=0; d<3; ++d) b[d] /= norm;
      Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
      Real tr = gm1*w(m,IEN,k,j,i)/w(m,IDN,k,j,i);
      Real tl = gm1*w(m,IEN,kl,jl,il)/w(m,IDN,kl,jl,il);
      Real grad = b[dir]*(tr-tl)/dx[dir];
      for (int d=0; d<ndim; ++d) {
        if (d == dir) continue;
        int ti=d == 0, tj=d == 1, tk=d == 2;
        Real rp = gm1*w(m,IEN,k+tk,j+tj,i+ti)/w(m,IDN,k+tk,j+tj,i+ti);
        Real rm = gm1*w(m,IEN,k-tk,j-tj,i-ti)/w(m,IDN,k-tk,j-tj,i-ti);
        Real lp = gm1*w(m,IEN,kl+tk,jl+tj,il+ti)/w(m,IDN,kl+tk,jl+tj,il+ti);
        Real lm = gm1*w(m,IEN,kl-tk,jl-tj,il-ti)/w(m,IDN,kl-tk,jl-tj,il-ti);
        Real transverse = mc ? MCL2State(MCL2State(rp-tr,tr-rm),MCL2State(lp-tl,tl-lm)) :
                               VLL4State(rp-tr,tr-rm,lp-tl,tl-lm);
        grad += b[d]*transverse/dx[d];
      }
      // Harmonic interpolation of rho*alpha handles conductivity contrasts.
      Real lo = fmin(w(m,IDN,k,j,i),w(m,IDN,kl,jl,il));
      Real hi = fmax(w(m,IDN,k,j,i),w(m,IDN,kl,jl,il));
      Real rhof = 2.0*lo/(1.0+lo/hi);
      f(m,IEN,k,j,i) -= alpha*rhof*b[dir]*grad;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TempDependentHeatFlux()
//! \brief Adds heat flux to face-centered fluxes of conserved variables with
//! temperature-dependent conductivity


//----------------------------------------------------------------------------------------
//! \fn void Conduction::NewTimeStep()
//! \brief Compute new time step for thermal conduction.

void Conduction::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  dt_transport = static_cast<Real>(std::numeric_limits<float>::max());
  if (saturated || (alpha_spitzer && spitzer_kappa0_code > 0.0)) {
    dtnew = SpitzerTimeStep(w0, eos_data, !UsesNonlinearSTS());
    if (UsesNonlinearSTS() && saturated) {
      dt_transport = SaturationTimeStep(w0, eos_data, true);
    }
    return;
  }
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());

  // The active constant-alpha operator uses code units only. Do not dereference
  // punit here: a units block is optional and these conversions were unused.

  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &w0_ = w0;
  auto &multi_d = pmy_pack->pmesh->multi_d;
  auto &three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  Real gm1 = eos_data.gamma-1.0;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  // Deliberately conservative explicit bound for the limited cross derivatives.
  // |b_i b_j| <= 1 and 1/(dx_i dx_j) <= (1/dx_i^2+1/dx_j^2)/2;
  // allow a factor four per dimension for the nested limited slopes. Arithmetic
  // face densities below bound the harmonic density in the anisotropic flux.
  // This bound is not a proof of nonlinear STS or refined-mesh acceptance.
  Real alpha0 = alpha_iso + 4.0*ndim*alpha_aniso;

  // find smallest timestep for thermal conduction in each cell
  // Note loop over all cells needed even for constant conductivity
  Kokkos::parallel_reduce("cond_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real row_sum = 0.5*(w0_(m,IDN,k,j,i-1) + 2.0*w0_(m,IDN,k,j,i) +
                        w0_(m,IDN,k,j,i+1))/SQR(size.d_view(m).dx1);
    if (multi_d) {
      row_sum += 0.5*(w0_(m,IDN,k,j-1,i) + 2.0*w0_(m,IDN,k,j,i) +
                     w0_(m,IDN,k,j+1,i))/SQR(size.d_view(m).dx2);
    }
    if (three_d) {
      row_sum += 0.5*(w0_(m,IDN,k-1,j,i) + 2.0*w0_(m,IDN,k,j,i) +
                     w0_(m,IDN,k+1,j,i))/SQR(size.d_view(m).dx3);
    }

    Real rate = alpha0*gm1*row_sum/w0_(m,IDN,k,j,i);
    if (rate > 0.0) {
      min_dt = fmin(min_dt, 1.0/rate);
    }
  }, Kokkos::Min<Real>(dtnew));

  return;
}
