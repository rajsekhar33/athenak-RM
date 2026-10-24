//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file diffusion.cpp
//! \brief problem generator for tests of the diffusion modules (viscosity, resistivity,
//! thermal conduction).  Sets up a Gaussian pulse that spreads under pure diffusion.
//! The pulse can vary along one, two, or three coordinate axes (selected with the
//! spread_x1/x2/x3 flags), giving an isotropic n-dimensional Gaussian whose amplitude
//! decays as 1/(1+4*D*t)^(n/2).  For thermal conduction the pulse is in the gas
//! temperature/pressure; for viscosity the pulse is in a single (transverse) velocity
//! component selected with vel_comp; for resistivity the pulse is in a single
//! (transverse) magnetic-field component (also selected with vel_comp: 1->Bx, 2->By,
//! 3->Bz, stored on the corresponding cell face and uniform along its own axis so that
//! div(B)=0).  Must be run in kinematic mode.  Errors in the final solution are computed
//! from the analytic profile at final time in DiffusionErrors().  Resistivity runs use
//! the <mhd> block; conduction/viscosity use the <hydro> block. The optional
//! resistivity_profile=helix instead evolves a force-free field with dynamical gas,
//! providing an analytic check of Ohmic decay and Joule heating.

// C headers

// C++ headers
#include <cmath>      // sqrt(), pow(), exp()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <string>     // c_str()

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/resistivity.hpp"

// Prototype for function to compute errors in solution at end of run
void DiffusionErrors(ParameterInput *pin, Mesh *pm);
// Prototype for user-defined BCs
void GaussianProfileBCs(Mesh *pm);

// Anonymous namespace used to prevent name collisions outside of this file
namespace {
// global variable to control computation of initial conditions versus errors
bool set_initial_conditions = true;
// input parameters passed to the initialization kernel and user-defined BC function
struct DiffusionVariables {
  bool conduction_test, viscosity_test, resistivity_test;
  bool spread_x1, spread_x2, spread_x3;
  int vel_comp;                 // 1/2/3 -> velocity (viscosity) or B (resistivity) compt
  Real amp, x10, x20, x30;      // amplitude and Gaussian centers
};

DiffusionVariables diffvars;

// Circular transverse shear supplies isotropic heating and general initial-RHS
// tests. With rho=1 and a constant transverse magnetic field, isotropic viscosity
// gives V'=-k^2*mu(T)*V. The functional harness integrates this ODE.
void ViscousHelix(ParameterInput *pin, Mesh *pm, const bool restart) {
  auto *pack = pm->pmb_pack;
  auto *ph = pack->phydro;
  auto *pb = pack->pmhd;
  auto *pvisc = ph != nullptr ? ph->pvisc : (pb != nullptr ? pb->pvisc : nullptr);
  const int axis = pin->GetOrAddInteger("problem", "helix_axis", 1);
  const int mode = pin->GetOrAddInteger("problem", "helix_mode", 1);
  const Real amp = pin->GetOrAddReal("problem", "amp", 0.5);
  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real pres = pin->GetOrAddReal("problem", "pres", 0.1);
  const Real bg = pin->GetOrAddReal("problem", "helix_bfield", 0.0);
  const Real bx = pin->GetOrAddReal("problem", "helix_bx", axis%3 == 0 ? bg : 0.0);
  const Real by = pin->GetOrAddReal("problem", "helix_by", axis%3 == 1 ? bg : 0.0);
  const Real bz = pin->GetOrAddReal("problem", "helix_bz", axis%3 == 2 ? bg : 0.0);
  const Real magnetic_amp = pin->GetOrAddReal("problem", "helix_magnetic_amp", 0.0);
  const Real bulk[3] = {pin->GetOrAddReal("problem", "helix_bulk_v1", 0.0),
                        pin->GetOrAddReal("problem", "helix_bulk_v2", 0.0),
                        pin->GetOrAddReal("problem", "helix_bulk_v3", 0.0)};
  const Real drho = pin->GetOrAddReal("problem", "helix_density_amp", 0.0);
  if ((ph == nullptr && pb == nullptr) || (ph != nullptr && pb != nullptr) ||
      pvisc == nullptr || axis < 1 || axis > 3 || mode < 1 ||
      (axis == 2 && !pm->multi_d) || (axis == 3 && !pm->three_d) ||
      !std::isfinite(amp) || amp <= 0.0 || !std::isfinite(rho) || rho <= 0.0 ||
      !std::isfinite(pres) || pres <= 0.0 || !std::isfinite(bg) ||
      !std::isfinite(bx) || !std::isfinite(by) || !std::isfinite(bz) ||
      !std::isfinite(magnetic_amp) || magnetic_amp < 0.0 ||
      !std::isfinite(bulk[0]) || !std::isfinite(bulk[1]) || !std::isfinite(bulk[2]) ||
      !std::isfinite(drho) || fabs(drho) >= 1.0 ||
      pin->GetString("time", "evolution") != "dynamic") {
    Kokkos::abort("Viscous helix needs one fluid, viscosity and positive finite data");
  }
  const auto eos = ph != nullptr ? ph->peos->eos_data : pb->peos->eos_data;
  if (!eos.is_ideal || (ph != nullptr && (bx != 0.0 || by != 0.0 || bz != 0.0))) {
    Kokkos::abort("Viscous helix requires ideal gas and MHD for a magnetic field");
  }
  if (magnetic_amp > 0.0 && (!pb || bx != 0.0 || by != 0.0 || bz != 0.0)) {
    Kokkos::abort("Coupled helix requires MHD and zero uniform magnetic background");
  }
  for (const char *face : {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc",
                           "ix3_bc", "ox3_bc"}) {
    if (pin->GetString("mesh", face) != "periodic") {
      Kokkos::abort("Viscous helix requires periodic boundaries");
    }
  }
  if (restart) return;
  const auto idx = pm->mb_indcs;
  const auto domain = pm->mesh_size;
  const Real xmin = axis == 1 ? domain.x1min : (axis == 2 ? domain.x2min : domain.x3min);
  const Real xmax = axis == 1 ? domain.x1max : (axis == 2 ? domain.x2max : domain.x3max);
  const Real twopi = 2.0*std::acos(-1.0), wave = twopi*mode/(xmax-xmin);
  auto size = pack->pmb->mb_size;
  auto u = ph != nullptr ? ph->u0 : pb->u0;
  const int nvars = ph != nullptr ? ph->nhydro+ph->nscalars : pb->nmhd+pb->nscalars;
  DvceArray4D<Real> b1, b2, b3;
  const bool mhd = pb != nullptr;
  if (mhd) { b1 = pb->b0.x1f; b2 = pb->b0.x2f; b3 = pb->b0.x3f; }
  const bool multi_d = pm->multi_d, three_d = pm->three_d;
  par_for("pgen_viscous_helix", DevExeSpace(), 0, pack->nmb_thispack-1,
          idx.ks, idx.ke, idx.js, idx.je, idx.is, idx.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x[3] = {
        CellCenterX(i-idx.is,idx.nx1,size.d_view(m).x1min,size.d_view(m).x1max),
        CellCenterX(j-idx.js,idx.nx2,size.d_view(m).x2min,size.d_view(m).x2max),
        CellCenterX(k-idx.ks,idx.nx3,size.d_view(m).x3min,size.d_view(m).x3max)};
    Real modulation = cos(twopi*(x[0]-domain.x1min)/(domain.x1max-domain.x1min));
    if (multi_d) {
      modulation *= cos(twopi*(x[1]-domain.x2min)/(domain.x2max-domain.x2min));
    }
    if (three_d) {
      modulation *= cos(twopi*(x[2]-domain.x3min)/(domain.x3max-domain.x3min));
    }
    const Real density = rho*(1.0+drho*modulation);
    const Real phase = wave*(x[axis-1]-xmin);
    for (int n=0; n<nvars; ++n) u(m,n,k,j,i) = 0.0;
    u(m,IDN,k,j,i) = density;
    u(m,IM1+axis%3,k,j,i) = density*amp*cos(phase);
    u(m,IM1+(axis+1)%3,k,j,i) = density*amp*sin(phase);
    for (int a=0; a<3; ++a) u(m,IM1+a,k,j,i) += density*bulk[a];
    Real magnetic[3] = {bx, by, bz};
    magnetic[axis%3] += magnetic_amp*cos(phase);
    magnetic[(axis+1)%3] += magnetic_amp*sin(phase);
    const Real kinetic = (bulk[0] == 0.0 && bulk[1] == 0.0 && bulk[2] == 0.0) ?
        0.5*density*SQR(amp) :
        0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+SQR(u(m,IM3,k,j,i)))/density;
    u(m,IEN,k,j,i) = pres/(eos.gamma-1.0)+kinetic+
                    0.5*(SQR(magnetic[0])+SQR(magnetic[1])+SQR(magnetic[2]));
    if (mhd) {
      b1(m,k,j,i) = magnetic[0];
      b2(m,k,j,i) = magnetic[1];
      b3(m,k,j,i) = magnetic[2];
      if (i == idx.ie) b1(m,k,j,i+1) = magnetic[0];
      if (j == idx.je) b2(m,k,j+1,i) = magnetic[1];
      if (k == idx.ke) b3(m,k+1,j,i) = magnetic[2];
    }
  });
}

//----------------------------------------------------------------------------------------
//! \brief Periodic force-free helix: curl(B) is parallel to B, so the gas stays at
//! rest while magnetic energy becomes heat. Total energy is spatially constant.
//! Transverse face values are independent of their own face-normal coordinates.

void ResistiveHelix(ParameterInput *pin, Mesh *pm, const bool restart) {
  auto *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  const int axis = pin->GetOrAddInteger("problem", "helix_axis", 1);
  const int mode = pin->GetOrAddInteger("problem", "helix_mode", 1);
  const Real amp = pin->GetOrAddReal("problem", "amp", 1.0);
  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real pres = pin->GetOrAddReal("problem", "pres", 1.0);
  const Real density_amp = pin->GetOrAddReal("problem", "helix_density_amp", 0.0);
  const std::string evolution = pin->GetString("time", "evolution");
  if (pmhd == nullptr || pmbp->phydro != nullptr || pmhd->presist == nullptr ||
      !pmhd->peos->eos_data.is_ideal ||
      (evolution != "dynamic" && evolution != "kinematic") ||
      axis < 1 || axis > 3 || mode < 1 ||
      !std::isfinite(amp) || amp <= 0.0 ||
      !std::isfinite(rho) || rho <= 0.0 ||
      !std::isfinite(pres) || pres <= 0.0 ||
      !std::isfinite(density_amp) || fabs(density_amp) >= 1.0) {
    std::cout << "### FATAL ERROR: resistive helix requires ideal MHD, "
              << "resistivity, a valid axis/mode and positive finite state" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const bool spitzer = pmhd->presist->eta_spitzer;
  const Real eta = spitzer ? pmhd->presist->spitzer_eta0_code/pow(pres/rho, 1.5)
                           : pmhd->presist->eta_ohm;
  if (!std::isfinite(eta) || eta <= 0.0 || pmhd->presist->eta_ad != 0.0 ||
      (axis == 2 && !pm->multi_d) || (axis == 3 && !pm->three_d)) {
    std::cout << "### FATAL ERROR: resistive helix requires positive "
              << "Ohmic diffusion and a resolved helix axis" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (const char *face : {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc",
                           "ix3_bc", "ox3_bc"}) {
    if (pin->GetString("mesh", face) != "periodic") {
      std::cout << "### FATAL ERROR: resistive helix requires periodic boundaries"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (restart) return;

  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const auto domain = pm->mesh_size;
  const Real xmin = (axis == 1) ? domain.x1min :
                    ((axis == 2) ? domain.x2min : domain.x3min);
  const Real xmax = (axis == 1) ? domain.x1max :
                    ((axis == 2) ? domain.x2max : domain.x3max);
  const Real wave = 2.0*std::acos(-1.0)*mode/(xmax - xmin);
  const Real twopi = 2.0*std::acos(-1.0);
  const Real amplitude = spitzer ? amp : amp*std::exp(-eta*SQR(wave)*pm->time);
  const Real energy = pres/(pmhd->peos->eos_data.gamma - 1.0) + 0.5*SQR(amp);
  auto &size = pmbp->pmb->mb_size;
  auto &b = (set_initial_conditions) ? pmhd->b0 : pmhd->b1;
  auto &u = (set_initial_conditions) ? pmhd->u0 : pmhd->u1;
  const int nvars = pmhd->nmhd + pmhd->nscalars;
  par_for("pgen_resist_helix", DevExeSpace(), 0, pmbp->nmb_thispack-1,
          ks, ke, js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                        size.d_view(m).x1max);
    if (axis == 2) {
      x = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    } else if (axis == 3) {
      x = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    }
    const Real bc = amplitude*cos(wave*(x-xmin));
    const Real bs = amplitude*sin(wave*(x-xmin));
    const Real bx = (axis == 1) ? 0.0 : ((axis == 2) ? bs : bc);
    const Real by = (axis == 2) ? 0.0 : ((axis == 3) ? bs : bc);
    const Real bz = (axis == 3) ? 0.0 : ((axis == 1) ? bs : bc);
    b.x1f(m,k,j,i) = bx;
    b.x2f(m,k,j,i) = by;
    b.x3f(m,k,j,i) = bz;
    if (i == ie) b.x1f(m,k,j,i+1) = bx;
    if (j == je) b.x2f(m,k,j+1,i) = by;
    if (k == ke) b.x3f(m,k+1,j,i) = bz;
    for (int n=0; n<nvars; ++n) u(m,n,k,j,i) = 0.0;
    Real modulation = 1.0;
    if (density_amp != 0.0) {
      const Real x1 = CellCenterX(i-is, indcs.nx1, size.d_view(m).x1min,
                                  size.d_view(m).x1max);
      modulation = cos(twopi*(x1-domain.x1min)/(domain.x1max-domain.x1min));
      if (indcs.nx2 > 1) {
        const Real x2 = CellCenterX(j-js, indcs.nx2, size.d_view(m).x2min,
                                    size.d_view(m).x2max);
        modulation *= cos(twopi*(x2-domain.x2min)/(domain.x2max-domain.x2min));
      }
      if (indcs.nx3 > 1) {
        const Real x3 = CellCenterX(k-ks, indcs.nx3, size.d_view(m).x3min,
                                    size.d_view(m).x3max);
        modulation *= cos(twopi*(x3-domain.x3min)/(domain.x3max-domain.x3min));
      }
    }
    u(m,IDN,k,j,i) = rho*(1.0+density_amp*modulation);
    u(m,IEN,k,j,i) = energy;
  });
}

//----------------------------------------------------------------------------------------
//! \fn DiffusionGaussian
//! \brief Device-callable helper returning the scalar amplitude of the isotropic n-D
//! radial Gaussian pulse at (x1,x2,x3) and time, spreading with diffusivity "coef".

KOKKOS_INLINE_FUNCTION
Real DiffusionGaussian(const DiffusionVariables dv, const Real coef, const Real time,
                       const Real x1, const Real x2, const Real x3) {
  Real ndim = (dv.spread_x1 ? 1.0 : 0.0)
            + (dv.spread_x2 ? 1.0 : 0.0)
            + (dv.spread_x3 ? 1.0 : 0.0);
  Real spread = 1.0 + 4.0*coef*time;
  Real r2 = 0.0;
  if (dv.spread_x1) r2 += SQR(x1 - dv.x10);
  if (dv.spread_x2) r2 += SQR(x2 - dv.x20);
  if (dv.spread_x3) r2 += SQR(x3 - dv.x30);
  return (dv.amp/pow(spread, 0.5*ndim))*exp(-r2/spread);
}

//----------------------------------------------------------------------------------------
//! \fn DiffusionConsState
//! \brief Device-callable helper that returns the analytic conserved state of the
//! Gaussian-pulse diffusion solution at position (x1,x2,x3) and time. "coef" is the
//! relevant diffusivity (nu_iso for viscosity, alpha_iso for conduction), "gamma" the
//! adiabatic index.  Results are written into cons[IDN..IEN].

KOKKOS_INLINE_FUNCTION
void DiffusionConsState(const DiffusionVariables dv, const Real coef, const Real gamma,
                        const Real time, const Real x1, const Real x2, const Real x3,
                        Real cons[5]) {
  Real g = DiffusionGaussian(dv, coef, time, x1, x2, x3);

  Real gm1 = gamma - 1.0;
  Real rho = 1.0;
  Real m1 = 0.0, m2 = 0.0, m3 = 0.0;
  Real p0 = 1.0/gamma;       // background pressure (used by viscosity test)
  if (dv.conduction_test) {
    p0 = g;                  // Gaussian temperature/pressure pulse
  }
  if (dv.viscosity_test) {
    if (dv.vel_comp == 1) {
      m1 = rho*g;
    } else if (dv.vel_comp == 2) {
      m2 = rho*g;
    } else {
      m3 = rho*g;
    }
  }
  cons[IDN] = rho;
  cons[IM1] = m1;
  cons[IM2] = m2;
  cons[IM3] = m3;
  cons[IEN] = p0/gm1 + 0.5*(SQR(m1) + SQR(m2) + SQR(m3))/rho;
}

} // end anonymous namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::Diffusion()
//! \brief Sets initial conditions for diffusion tests

void ProblemGenerator::Diffusion(ParameterInput *pin, const bool restart) {
  const std::string visc_profile =
      pin->GetOrAddString("problem", "viscosity_profile", "gaussian");
  if (visc_profile == "helix") {
    ViscousHelix(pin, pmy_mesh_, restart);
    pgen_final_func = nullptr;
    return;
  }
  if (visc_profile != "gaussian") Kokkos::abort("Unknown viscosity_profile");
  const std::string profile = pin->GetOrAddString("problem", "resistivity_profile",
                                                "gaussian");
  if (profile == "helix") {
    ResistiveHelix(pin, pmy_mesh_, restart);
    // Nonlinear decay uses the independent energy/induction ODE in the
    // functional harness; the constant-eta exponential is inapplicable there.
    pgen_final_func = pmy_mesh_->pmb_pack->pmhd->presist->eta_spitzer ?
                      nullptr : DiffusionErrors;
    return;
  }
  if (profile != "gaussian") {
    std::cout << "### FATAL ERROR: unknown resistivity_profile '" << profile
              << "' (expected gaussian or helix)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string evolution_t = pin->GetString("time","evolution");
  if (evolution_t.compare("kinematic") != 0) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "Diffusion tests must be run in kinematic mode" << std::endl;
    exit(EXIT_FAILURE);
  }
  // set diffusion errors function
  pgen_final_func = DiffusionErrors;
  // user-defined BC
  user_bcs_func = GaussianProfileBCs;
  // Analytic ghost assignments have no mutable state or external side effects.
  user_bcs_retry_safe = true;

  // Reconstruct boundary/error parameters on fresh starts and restarts alike.
  diffvars.amp = pin->GetOrAddReal("problem", "amp", 1.e-6);
  diffvars.x10 = pin->GetOrAddReal("problem", "x10", 0.0);
  diffvars.x20 = pin->GetOrAddReal("problem", "x20", 0.0);
  diffvars.x30 = pin->GetOrAddReal("problem", "x30", 0.0);
  diffvars.conduction_test = pin->GetBoolean("problem", "conduction_test");
  diffvars.viscosity_test = pin->GetBoolean("problem", "viscosity_test");
  diffvars.resistivity_test = pin->GetBoolean("problem", "resistivity_test");
  diffvars.spread_x1 = pin->GetOrAddBoolean("problem", "spread_x1", true);
  diffvars.spread_x2 = pin->GetOrAddBoolean("problem", "spread_x2", false);
  diffvars.spread_x3 = pin->GetOrAddBoolean("problem", "spread_x3", false);
  diffvars.vel_comp = pin->GetOrAddInteger("problem", "vel_comp", 2);
  int ntests = (diffvars.conduction_test ? 1 : 0)
             + (diffvars.viscosity_test ? 1 : 0)
             + (diffvars.resistivity_test ? 1 : 0);
  if (ntests != 1) {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
            << "Exactly one of conduction_test/viscosity_test/resistivity_test must be "
            << "set true (got " << ntests << ")" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Restart has already restored the evolved fields. Re-enroll callbacks and
  // their parameters above, but do not overwrite that state with fresh ICs.
  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;
  Real time = pmbp->pmesh->time;
  auto dv = diffvars;

  // Initialize Hydro variables -------------------------------
  if (pmbp->phydro != nullptr) {
    if (dv.conduction_test && pmbp->phydro->pcond == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Conduction not defined in Hydro input block" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (dv.viscosity_test && pmbp->phydro->pvisc == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Viscosity not defined in Hydro input block" << std::endl;
      exit(EXIT_FAILURE);
    }

    EOS_Data &eos = pmbp->phydro->peos->eos_data;
    if (!(eos.is_ideal)) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Diffusion test requires ideal EOS in Hydro block" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real gamma = eos.gamma;
    // Effective diffusivity of the pulse for the active test.  For viscosity the
    // transverse momentum diffuses with the kinematic viscosity nu_iso.  For conduction
    // the heat flux uses T = (gamma-1)*eint/rho = p/rho, giving an energy/pressure pulse
    // that (with rho=1) diffuses with D = (gamma-1) * alpha_iso.
    Real coef = 0.0;
    if (dv.conduction_test) coef = (gamma-1.0)*pmbp->phydro->pcond->alpha_iso;
    if (dv.viscosity_test) coef = pmbp->phydro->pvisc->nu_iso;

    // compute solution in u1 register. For initial conditions, set u1 -> u0.
    auto &u1 = (set_initial_conditions)? pmbp->phydro->u0 : pmbp->phydro->u1;

    par_for("pgen_diff", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real cons[5];
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u1(m,IDN,k,j,i) = cons[IDN];
      u1(m,IM1,k,j,i) = cons[IM1];
      u1(m,IM2,k,j,i) = cons[IM2];
      u1(m,IM3,k,j,i) = cons[IM3];
      u1(m,IEN,k,j,i) = cons[IEN];
    });
  } // End initialization of Hydro variables

  // Initialize MHD variables (resistivity test) --------------
  if (pmbp->pmhd != nullptr) {
    if (!dv.resistivity_test) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "MHD diffusion test only supports the resistivity test" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (pmbp->pmhd->presist == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Resistivity (mhd/eta_ohm) not defined in MHD input block"
              << std::endl;
      exit(EXIT_FAILURE);
    }
    EOS_Data &eos = pmbp->pmhd->peos->eos_data;
    if (!(eos.is_ideal)) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " << __LINE__ << std::endl
              << "Diffusion test requires ideal EOS in MHD block" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real gamma = eos.gamma;
    Real gm1 = gamma - 1.0;
    // The transverse magnetic field diffuses with the Ohmic diffusivity eta_ohm.
    Real coef = pmbp->pmhd->presist->eta_ohm;
    int bcomp = dv.vel_comp;          // 1->Bx, 2->By, 3->Bz
    Real p0 = 1.0/gamma;              // uniform background gas pressure

    // face- and cell-centered fields go to b0/b1 depending on whether these are ICs
    auto &b = (set_initial_conditions)? pmbp->pmhd->b0 : pmbp->pmhd->b1;
    auto &bcc0 = pmbp->pmhd->bcc0;
    auto &w0 = pmbp->pmhd->w0;

    // The pulse lives in a single B component that is uniform along its own coordinate
    // axis (which is never a spread axis), so the staggered face value equals the
    // Gaussian evaluated at the cell-centered spread coordinates and div(B)=0.
    par_for("pgen_resist_b", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m,int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

      Real g = DiffusionGaussian(dv, coef, time, x1v, x2v, x3v);
      b.x1f(m,k,j,i) = (bcomp == 1) ? g : 0.0;
      b.x2f(m,k,j,i) = (bcomp == 2) ? g : 0.0;
      b.x3f(m,k,j,i) = (bcomp == 3) ? g : 0.0;
      if (i == ie) {
        b.x1f(m,k,j,i+1) = (bcomp == 1) ? g : 0.0;
      }
      if (j == je) {
        b.x2f(m,k,j+1,i) = (bcomp == 2) ? g : 0.0;
      }
      if (k == ke) {
        b.x3f(m,k+1,j,i) = (bcomp == 3) ? g : 0.0;
      }

      // cell-centered field and primitive state (rho=1, v=0, uniform pressure)
      bcc0(m,IBX,k,j,i) = (bcomp == 1) ? g : 0.0;
      bcc0(m,IBY,k,j,i) = (bcomp == 2) ? g : 0.0;
      bcc0(m,IBZ,k,j,i) = (bcomp == 3) ? g : 0.0;
      w0(m,IDN,k,j,i) = 1.0;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IEN,k,j,i) = p0/gm1;
    });

    // convert primitives + cell-centered B to conserved variables in u0/u1
    auto &u = (set_initial_conditions)? pmbp->pmhd->u0 : pmbp->pmhd->u1;
    pmbp->pmhd->peos->PrimToCons(w0, bcc0, u, is, ie, js, je, ks, ke);
  } // End initialization of MHD variables
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DiffusionErrors_()
//! \brief Computes errors in diffusion solution by calling initialization function
//! again to compute initial condictions, and subtracting current solution from ICs, and
//! outputs errors to file.

void DiffusionErrors(ParameterInput *pin, Mesh *pm) {
  // calculate reference solution by calling pgen again.  Solution stored in second
  // register u1/b1 when flag is false.
  set_initial_conditions = false;
  pm->pgen->Diffusion(pin, false);
  pm->pgen->OutputErrors(pin, pm);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn GaussianProfileBCs
//! \brief Sets boundary conditions to the time-dependent analytic Gaussian profile on
//! all (user) boundary faces of the computational domain.

void GaussianProfileBCs(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &size = pm->pmb_pack->pmb->mb_size;

  // The analytic-profile user BCs are only used by the hydro (conduction/viscosity)
  // tests.  Resistivity (MHD) runs use outflow BCs, since the field is negligible at the
  // (wide) domain boundaries, so nothing to do here when hydro is absent.
  auto *phydro = pm->pmb_pack->phydro;
  if (phydro == nullptr) return;
  EOS_Data &eos = phydro->peos->eos_data;
  Real gamma = eos.gamma;
  auto &u0 = phydro->u0;

  // capture variables for the kernel
  auto dv = diffvars;
  Real time = pm->time;
  Real coef = 0.0;
  if (dv.conduction_test && phydro->pcond != nullptr) {
    coef = (gamma-1.0)*phydro->pcond->alpha_iso;
  }
  if (dv.viscosity_test && phydro->pvisc != nullptr) coef = phydro->pvisc->nu_iso;

  // X1 boundaries
  par_for("diffbc_x1", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real cons[5];

    // inner x1
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,i) = cons[IDN];
      u0(m,IM1,k,j,i) = cons[IM1];
      u0(m,IM2,k,j,i) = cons[IM2];
      u0(m,IM3,k,j,i) = cons[IM3];
      u0(m,IEN,k,j,i) = cons[IEN];
    }
    // outer x1
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      Real x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,(ie+i+1)) = cons[IDN];
      u0(m,IM1,k,j,(ie+i+1)) = cons[IM1];
      u0(m,IM2,k,j,(ie+i+1)) = cons[IM2];
      u0(m,IM3,k,j,(ie+i+1)) = cons[IM3];
      u0(m,IEN,k,j,(ie+i+1)) = cons[IEN];
    }
  });
  if (pm->one_d) return;

  // X2 boundaries
  par_for("diffbc_x2", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(ng-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
    Real cons[5];

    // inner x2
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,i) = cons[IDN];
      u0(m,IM1,k,j,i) = cons[IM1];
      u0(m,IM2,k,j,i) = cons[IM2];
      u0(m,IM3,k,j,i) = cons[IM3];
      u0(m,IEN,k,j,i) = cons[IEN];
    }
    // outer x2
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      Real x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,(je+j+1),i) = cons[IDN];
      u0(m,IM1,k,(je+j+1),i) = cons[IM1];
      u0(m,IM2,k,(je+j+1),i) = cons[IM2];
      u0(m,IM3,k,(je+j+1),i) = cons[IM3];
      u0(m,IEN,k,(je+j+1),i) = cons[IEN];
    }
  });
  if (pm->two_d) return;

  // X3 boundaries
  par_for("diffbc_x3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real cons[5];

    // inner x3
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,k,j,i) = cons[IDN];
      u0(m,IM1,k,j,i) = cons[IM1];
      u0(m,IM2,k,j,i) = cons[IM2];
      u0(m,IM3,k,j,i) = cons[IM3];
      u0(m,IEN,k,j,i) = cons[IEN];
    }
    // outer x3
    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      Real x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);
      DiffusionConsState(dv, coef, gamma, time, x1v, x2v, x3v, cons);
      u0(m,IDN,(ke+k+1),j,i) = cons[IDN];
      u0(m,IM1,(ke+k+1),j,i) = cons[IM1];
      u0(m,IM2,(ke+k+1),j,i) = cons[IM2];
      u0(m,IM3,(ke+k+1),j,i) = cons[IM3];
      u0(m,IEN,(ke+k+1),j,i) = cons[IEN];
    }
  });
  return;
}
