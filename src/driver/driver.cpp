//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driver.cpp
//  \brief implementation of functions in class Driver

#include <iostream>
#include <iomanip>    // std::setprecision()
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string> // string
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "eos/eos.hpp"
#include "coordinates/coordinates.hpp"
#include "outputs/outputs.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "ion-neutral/ion-neutral.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/srcterms.hpp"
#include "srcterms/turb_driver.hpp"
#include "driver.hpp"
#include "gravity/gravity.hpp"
#include "utils/utils.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

// A fluid checkpoint must contain the NEXT counters for all outputs in the
// batch, including particle checkpoints written later in that batch. Reserve
// only ParameterInput values; writers still advance their own counters once.
void ReserveOutputCounters(const std::vector<BaseTypeOutput*> &outputs,
                           ParameterInput *pin, Real time) {
  for (auto *out : outputs) {
    const auto &op = out->out_params;
    pin->SetInteger(op.block_name, "file_number", op.file_number+1);
    pin->SetReal(op.block_name, "last_time",
                 op.last_time < 0 ? time : op.last_time+op.dt);
  }
}

[[noreturn]] void DriverFatalError(const char *file, int line, const std::string &msg) {
  std::cout << "### FATAL ERROR in " << file << " at line " << line << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

Conduction *NonlinearSTSConduction(Mesh *pm) {
  auto *ph = pm->pmb_pack->phydro;
  auto *pmhd = pm->pmb_pack->pmhd;
  if (ph != nullptr && ph->pcond != nullptr && ph->pcond->UsesNonlinearSTS()) {
    return ph->pcond;
  }
  if (pmhd != nullptr && pmhd->pcond != nullptr && pmhd->pcond->UsesNonlinearSTS()) {
    return pmhd->pcond;
  }
  return nullptr;
}

Viscosity *STSViscosity(Mesh *pm) {
  auto *ph = pm->pmb_pack->phydro;
  auto *pmhd = pm->pmb_pack->pmhd;
  if (ph != nullptr && ph->has_sts_viscosity) return ph->pvisc;
  if (pmhd != nullptr && pmhd->has_sts_viscosity) return pmhd->pvisc;
  return nullptr;
}

Resistivity *STSSpitzerResistivity(Mesh *pm) {
  auto *pmhd = pm->pmb_pack->pmhd;
  if (pmhd != nullptr && pmhd->has_sts_resistivity && pmhd->presist->eta_spitzer) {
    return pmhd->presist;
  }
  return nullptr;
}

bool UsesFullStateSTS(Mesh *pm) {
  if (STSViscosity(pm) != nullptr || STSSpitzerResistivity(pm) != nullptr) return true;
  if (NonlinearSTSConduction(pm) != nullptr) {
    int processes = 0;
    for (const auto &process : pm->pmb_pack->parabolic_processes) {
      if (process.UsesSTS()) ++processes;
    }
    return processes > 1;
  }
  return false;
}

template<typename View>
void SaveDiffusionArray(View &saved, const View &current) {
  bool resize = false;
  for (int d=0; d<View::rank; ++d) resize |= saved.extent(d) != current.extent(d);
  if (resize) {
    saved = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, "diffusion_rollback"),
                 current.layout());
  }
  if (current.size() > 0) Kokkos::deep_copy(DevExeSpace(), saved, current);
}

[[noreturn]] void GuardedSTSFatal(const char *reason) {
  std::cerr << "### FATAL ERROR: guarded diffusion STS: " << reason << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

[[noreturn]] void NonlinearSTSFatal(const char *reason) {
  std::cerr << "### FATAL ERROR: nonlinear STS: " << reason << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

} // namespace

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters
//
// First, define each time-integrator by setting weights for each step of the algorithm
// and the CFL number stability limit when coupled to the single-stage spatial operator.
// Currently, the explicit, multistage time-integrators must be expressed as 2S-type
// algorithms as in Ketcheson (2010) Algorithm 3, which incudes 2N (Williamson) and 2R
// (van der Houwen) popular 2-register low-storage RK methods. The 2S-type integrators
// depend on a bidiagonally sparse Shu-Osher representation; at each stage l:
//
//    U^{l} = a_{l,l-2}*U^{l-2} + a_{l-1}*U^{l-1}
//          + b_{l,l-2}*dt*Div(F_{l-2}) + b_{l,l-1}*dt*Div(F_{l-1}),
//
// where U^{l-1} and U^{l-2} are previous stages and a_{l,l-2}, a_{l,l-1}=(1-a_{l,l-2}),
// and b_{l,l-2}, b_{l,l-1} are weights that are different for each stage and
// integrator. Previous timestep U^{0} = U^n is given, and the integrator solves
// for U^{l} for 1 <= l <= nstages.
//
// The 2x RHS evaluations of Div(F) and source terms per stage is avoided by adding
// another weighted average / caching of these terms each stage. The API and framework
// is extensible to three register 3S* methods, although none are currently implemented.

// Notation: exclusively using "stage", equivalent in lit. to "substage" or "substep"
// (infrequently "step"), to refer to the intermediate values of U^{l} between each
// "timestep" = "cycle" in explicit, multistage methods.

Driver::Driver(ParameterInput *pin, Mesh *pmesh, Real wtlim, Kokkos::Timer* ptimer) :
  impl_src("ru",1,1,1,1,1,1),
  tlim(-1.0),
  nlim(-1),
  ndiag(1),
  pwall_clock_(ptimer),
  wall_time(wtlim),
  nmb_updated_(0),
  npart_updated_(0),
  lb_efficiency_(0) {
  // set time-evolution option (no default)
  {
    std::string evolution_t = pin->GetString("time","evolution");
    if (evolution_t.compare("static") == 0) {
      time_evolution = TimeEvolution::tstatic;  // cannot use 'static' (keyword);
    } else if (evolution_t.compare("kinematic") == 0) {
      time_evolution = TimeEvolution::kinematic;
    } else if (evolution_t.compare("dynamic") == 0) {
      time_evolution = TimeEvolution::dynamic;
    } else {
      std::cout<<"### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
               <<"<hydro> evolution = '"<< evolution_t <<"' not implemented"<< std::endl;
      std::exit(EXIT_FAILURE);
    }
  } // extra brace to limit scope of string

  // read <time> parameters controlling driver if run requires time-evolution
  if (time_evolution != TimeEvolution::tstatic) {
    integrator = pin->GetOrAddString("time", "integrator", "rk2");
    tlim = pin->GetReal("time", "tlim");
    nlim = pin->GetOrAddInteger("time", "nlim", -1);
    ndiag = pin->GetOrAddInteger("time", "ndiag", 1);

    if (integrator == "rk1") {
      // RK1: first-order Runge-Kutta / the forward Euler (FE) method
      nimp_stages = 0;
      nexp_stages = 1;
      cfl_limit = 1.0;
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;
    } else if (integrator == "rk2") {
      // Heun's method / SSPRK (2,2): Gottlieb (2009) equation 3.1
      // Optimal (in error bounds) explicit two-stage, second-order SSPRK
      nimp_stages = 0;
      nexp_stages = 2;
      cfl_limit = 1.0;  // c_eff = c/nstages = 1/2 (Gottlieb (2009), pg 271)
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;

      gam0[1] = 0.5;
      gam1[1] = 0.5;
      beta[1] = 0.5;
    } else if (integrator == "rk3") {
      // SSPRK (3,3): Gottlieb (2009) equation 3.2
      // Optimal (in error bounds) explicit three-stage, third-order SSPRK
      nimp_stages = 0;
      nexp_stages = 3;
      cfl_limit = 1.0;  // c_eff = c/nstages = 1/3 (Gottlieb (2009), pg 271)
      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.0;

      gam0[1] = 0.25;
      gam1[1] = 0.75;
      beta[1] = 0.25;

      gam0[2] = 2.0/3.0;
      gam1[2] = 1.0/3.0;
      beta[2] = 2.0/3.0;
    } else if (integrator == "rk4") {
      // RK4()4[2S] from Table 2 of Ketcheson (2010)
      // Non-SSP, explicit four-stage, fourth-order RK
      // Stability properties are similar to classical (non-SSP) RK4
      // (but ~2x L2 principal error norm).
      // Refer to Colella (2011) for linear stability analysis of constant coeff.
      nimp_stages = 0;
      nexp_stages = 4;

      // Colella (2011) eq 101; 1st order flux is most severe constraint
      cfl_limit = 1.3925;

      gam0[0] = 0.0;
      gam1[0] = 1.0;
      beta[0] = 1.193743905974738;

      gam0[1] = 0.121098479554482;
      gam1[1] = 0.721781678111411;
      beta[1] = 0.099279895495783;

      gam0[2] = -3.843833699660025;
      gam1[2] = 2.121209265338722;
      beta[2] = 1.131678018054042;

      gam0[3] = 0.546370891121863;
      gam1[3] = 0.198653035682705;
      beta[3] = 0.310665766509336;

      delta[0] = 1.0;
      delta[1] = 0.217683334308543;
      delta[2] = 1.065841341361089;
      delta[3] = 0.0;
    } else if (integrator == "imex2") {
      // IMEX-SSP2(3,2,2): Pareschi & Russo (2005) Table III.
      // two-stage explicit, three-stage implicit, second-order ImEx
      // Note explicit steps identical to RK2
      nimp_stages = 3;
      nexp_stages = 2;
      cfl_limit = 1.0;
      gam0[0] = 1.0;
      gam1[0] = 0.0;
      beta[0] = 1.0;

      gam0[1] = 0.5;
      gam1[1] = 0.5;
      beta[1] = 0.5;

      a_twid[0][0] = -1.0;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;

      a_twid[1][0] = 0.5;
      a_twid[1][1] = 0.0;
      a_twid[1][2] = 0.0;

      a_twid[2][0] = 0.0;
      a_twid[2][1] = 0.25;
      a_twid[2][2] = 0.25;
      a_impl = 0.5;
    } else if (integrator == "imex2+") {
      // IMEX(4,3,2): Krapp et al. (2024, arXiv:2310.04435), Eq.30.
      // three-stage explicit, four-stage implicit, second-order ImEx
      // two implicit stages added, adapting Athenak's overall architecture
      // Note explicit steps may not reduce to RK2 based on the parameters chosen
      nimp_stages = 4;
      nexp_stages = 3;
      cfl_limit = 1.0;
      gamma = 1.707106781186547;   //1+1/sqrt(2)
      gam0[0] = 1.0;
      gam1[0] = 0.0;
      beta[0] = gamma;

      gam0[1] = (2.0*gamma-1.0)/(2.0*gamma*gamma);
      gam1[1] = (1.0-(2.0*gamma-1.0)/(2.0*gamma*gamma));
      beta[1] = 1.0/(2.0*gamma);

      gam0[2] = 1.0;
      gam1[2] = 0.0;
      beta[2] = 0.0;

      a_twid[0][0] = 0.0;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;
      a_twid[0][3] = 0.0;

      a_twid[1][0] = 0.0;
      a_twid[1][1] = 0.0;
      a_twid[1][2] = 0.0;
      a_twid[1][3] = 0.0;

      a_twid[2][0] = 0.0;
      a_twid[2][1] = 0.0;
      a_twid[2][2] = (1.0-2.0*gamma*gamma)/2.0/gamma;
      a_twid[2][3] = 0.0;

      a_twid[3][0] = 0.0;
      a_twid[3][1] = 0.0;
      a_twid[3][2] = 0.0;
      a_twid[3][3] = 0.0;

      a_impl = gamma;
    } else if (integrator == "imex3") {
      // IMEX-SSP3(4,3,3): Pareschi & Russo (2005) Table VI.
      // three-stage explicit, four-stage implicit, third-order ImEx
      // Note explicit steps identical to RK3
      nimp_stages = 4;
      nexp_stages = 3;
      cfl_limit = 1.0;
      gam0[0] = 1.0;
      gam1[0] = 0.0;
      beta[0] = 1.0;

      gam0[1] = 0.25;
      gam1[1] = 0.75;
      beta[1] = 0.25;

      gam0[2] = 2.0/3.0;
      gam1[2] = 1.0/3.0;
      beta[2] = 2.0/3.0;

      Real a = 0.24169426078821;
      Real b = 0.06042356519705;
      Real e = 0.12915286960590;
      a_twid[0][0] = -2.0*a;
      a_twid[0][1] = 0.0;
      a_twid[0][2] = 0.0;
      a_twid[0][3] = 0.0;

      a_twid[1][0] = a;
      a_twid[1][1] = 1.0 - 2.0*a;
      a_twid[1][2] = 0.0;
      a_twid[1][3] = 0.0;

      a_twid[2][0] = b;
      a_twid[2][1] = e - ((1.0-a)/4.0);
      a_twid[2][2] = 0.5 - b - e - 1.25*a;
      a_twid[2][3] = 0.0;

      a_twid[3][0] = (-2.0/3.0)*b;
      a_twid[3][1] = (1.0 - 4.0*e)/6.0;
      a_twid[3][2] = (4.0*(b + e + a) - 1.0)/6.0;
      a_twid[3][3] = 2.0*(1.0 - a)/3.0;
      a_impl = a;
    // Error, unrecognized integrator name.
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
         << std::endl << "integrator=" << integrator << " not implemented. "
         << "Valid choices are [rk1,rk2,rk3,rk4,imex2,imex3]." << std::endl;
      exit(EXIT_FAILURE);
    }

    ValidateSTSConfiguration(pmesh);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Driver::ResetSTSController()
//! \brief Clear cycle-local STS controller state.

void Driver::ResetSTSController() {
  sts.enabled = false;
  sts.nonlinear = false;
  sts.full_state_retry = false;
  sts.attempt_failed = false;
  sts.temperature_min = sts.temperature_max = 0.0;
  sts.rejection_reason = "none";
  sts.integrator = parabolic::STSIntegrator::none;
  sts.sweep = STSSweep::none;
  sts.dt_cycle = 0.0;
  sts.dt_sweep = 0.0;
  sts.dt_parabolic_min = std::numeric_limits<float>::max();
  sts.nstages = 0;
  sts.current_stage = 0;
  sts.coeffs = parabolic::RKL2Coefficients{};
}

//----------------------------------------------------------------------------------------
//! \fn Driver::ValidateSTSConfiguration()
//! \brief Validate global STS settings against registered diffusion processes.

void Driver::ValidateSTSConfiguration(Mesh *pm) {
  int nsts_processes = 0;
  const parabolic::ParabolicProcessDescriptor *first_sts_process = nullptr;
  bool has_hydro_sts = false;
  bool has_mhd_sts = false;

  for (const auto &process : pm->pmb_pack->parabolic_processes) {
    if (!process.UsesSTS()) {
      continue;
    }
    ++nsts_processes;
    if (first_sts_process == nullptr) {
      first_sts_process = &process;
    }
    has_hydro_sts |= process.owner == parabolic::ParabolicProcessOwner::hydro;
    has_mhd_sts |= process.owner == parabolic::ParabolicProcessOwner::mhd;
  }

  if (pm->sts_integrator == parabolic::STSIntegrator::none) {
    if (first_sts_process != nullptr) {
      DriverFatalError(__FILE__, __LINE__,
                       "Parabolic process '" + first_sts_process->name +
                       "' selects STS, but <time>/sts_integrator = none");
    }
    return;
  }

  if (nsts_processes == 0) {
    DriverFatalError(__FILE__, __LINE__,
                     "<time>/sts_integrator = rkl2 requires at least one active "
                     "diffusion process with *_integrator = sts");
  }
  if (pm->pmb_pack->pionn != nullptr) {
    DriverFatalError(__FILE__, __LINE__,
                     "STS is not supported with <ion-neutral> physics");
  }
  if (pm->pmb_pack->prad != nullptr) {
    DriverFatalError(__FILE__, __LINE__,
                     "Hydro/MHD STS diffusion is not supported with <radiation> physics");
  }
  if (pm->pmb_pack->pcoord->is_special_relativistic ||
      pm->pmb_pack->pcoord->is_general_relativistic ||
      pm->pmb_pack->pcoord->is_dynamical_relativistic) {
    DriverFatalError(__FILE__, __LINE__,
                     "STS diffusion currently supports only Newtonian Hydro and MHD");
  }

  hydro::Hydro *phydro = pm->pmb_pack->phydro;
  if (has_hydro_sts && phydro != nullptr &&
      (phydro->porb_u != nullptr || phydro->psbox_u != nullptr)) {
    DriverFatalError(__FILE__, __LINE__,
                     "Hydro STS is not supported with shearing-box or orbital advection");
  }

  mhd::MHD *pmhd = pm->pmb_pack->pmhd;
  if (has_mhd_sts && pmhd != nullptr &&
      (pmhd->porb_u != nullptr || pmhd->porb_b != nullptr ||
       pmhd->psbox_u != nullptr || pmhd->psbox_b != nullptr)) {
    DriverFatalError(__FILE__, __LINE__,
                     "MHD STS is not supported with shearing-box or orbital advection");
  }
  if (NonlinearSTSConduction(pm) != nullptr) {
    if (phydro != nullptr && pmhd != nullptr) {
      NonlinearSTSFatal("nonlinear conduction requires a single fluid owner");
    }
    // Conduction alone retains its U/W snapshot and thermal extremum check.
    // Joint sweeps use the full-state guard and its code-enrolled callbacks.
    if (pm->pgen->user_bcs && !UsesFullStateSTS(pm)) {
      NonlinearSTSFatal("retryable nonlinear conduction does not support user BCs");
    }
    if (pm->pzoom != nullptr) {
      NonlinearSTSFatal("retryable nonlinear conduction does not support cyclic zoom");
    }
    if (pm->pmr != nullptr && pm->pmr->prolong_prims) {
      NonlinearSTSFatal("nonlinear STS requires conserved-variable prolongation");
    }
  }
  if (UsesFullStateSTS(pm)) {
    if (phydro != nullptr && pmhd != nullptr) {
      GuardedSTSFatal("full-state STS requires a single fluid owner");
    }
    if (STSViscosity(pm) != nullptr && pmhd != nullptr && pmhd->has_sts_resistivity &&
        !pmhd->peos->eos_data.is_ideal) {
      GuardedSTSFatal("joint viscosity/resistivity STS requires ideal-gas MHD");
    }
    if (pm->pgen->user_bcs && !pm->pgen->user_bcs_retry_safe) {
      GuardedSTSFatal("full-state STS requires code-enrolled retry-safe user BCs");
    }
    if (pm->pgen->user_dt && !pm->pgen->user_dt_retry_safe) {
      GuardedSTSFatal("full-state STS requires code-enrolled retry-safe user dt");
    }
    if (pm->pzoom != nullptr) {
      GuardedSTSFatal("full-state STS does not support cyclic zoom");
    }
    if (pm->pmr != nullptr && pm->pmr->prolong_prims) {
      GuardedSTSFatal("full-state STS requires conserved-variable prolongation");
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Driver::RefreshSTSCycleState()
//! \brief Refresh cycle-local STS controller data from the Mesh timestep budget.

void Driver::RefreshSTSCycleState(Mesh *pm) {
  ResetSTSController();
  sts.integrator = pm->sts_integrator;
  sts.dt_cycle = pm->dt;
  sts.dt_parabolic_min = pm->dt_parabolic_sts;

  if (sts.integrator == parabolic::STSIntegrator::none || sts.dt_cycle <= 0.0 ||
      sts.dt_parabolic_min >= std::numeric_limits<float>::max()) {
    return;
  }

  sts.dt_sweep = 0.5*sts.dt_cycle;
  sts.nstages = parabolic::ComputeRKL2StageCount(sts.dt_sweep,
                                                 sts.dt_parabolic_min);
  sts.enabled = sts.nstages > 0;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::BeginSTSSweep()
//! \brief Initialize the controller for one pre- or post-RK parabolic sweep.

void Driver::BeginSTSSweep(Mesh *pm, STSSweep sweep) {
  RefreshSTSCycleState(pm);
  if (sts.enabled) {
    sts.sweep = sweep;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Driver::SetSTSStage()
//! \brief Set the coefficients for one RKL2 stage.

void Driver::SetSTSStage(int stage) {
  sts.current_stage = stage;
  sts.coeffs = parabolic::ComputeRKL2Coefficients(stage, sts.nstages);
}

//----------------------------------------------------------------------------------------
//! \fn Driver::EndSTSSweep()
//! \brief Clear sweep-local state.

void Driver::EndSTSSweep() {
  sts.sweep = STSSweep::none;
  sts.current_stage = 0;
  sts.dt_sweep = 0.0;
  sts.coeffs = parabolic::RKL2Coefficients{};
  sts.nonlinear = false;
  sts.full_state_retry = false;
  sts.attempt_failed = false;
}

void Driver::DiffusionSnapshot::Save(Mesh *pm) {
  auto *ph = pm->pmb_pack->phydro;
  auto *pmhd = pm->pmb_pack->pmhd;
  SaveDiffusionArray(u, ph != nullptr ? ph->u0 : pmhd->u0);
  SaveDiffusionArray(w, ph != nullptr ? ph->w0 : pmhd->w0);
  if (pm->multilevel) {
    SaveDiffusionArray(coarse_u, ph != nullptr ? ph->coarse_u0 : pmhd->coarse_u0);
    SaveDiffusionArray(coarse_w, ph != nullptr ? ph->coarse_w0 : pmhd->coarse_w0);
  }
  if (pmhd != nullptr) {
    SaveDiffusionArray(bcc, pmhd->bcc0);
    SaveDiffusionArray(b1, pmhd->b0.x1f);
    SaveDiffusionArray(b2, pmhd->b0.x2f);
    SaveDiffusionArray(b3, pmhd->b0.x3f);
    if (pm->multilevel) {
      SaveDiffusionArray(coarse_b1, pmhd->coarse_b0.x1f);
      SaveDiffusionArray(coarse_b2, pmhd->coarse_b0.x2f);
      SaveDiffusionArray(coarse_b3, pmhd->coarse_b0.x3f);
    }
  }
}

void Driver::DiffusionSnapshot::Restore(Mesh *pm) const {
  auto *ph = pm->pmb_pack->phydro;
  auto *pmhd = pm->pmb_pack->pmhd;
  Kokkos::deep_copy(DevExeSpace(), ph != nullptr ? ph->u0 : pmhd->u0, u);
  Kokkos::deep_copy(DevExeSpace(), ph != nullptr ? ph->w0 : pmhd->w0, w);
  if (pm->multilevel) {
    Kokkos::deep_copy(DevExeSpace(), ph != nullptr ? ph->coarse_u0 : pmhd->coarse_u0,
                      coarse_u);
    Kokkos::deep_copy(DevExeSpace(), ph != nullptr ? ph->coarse_w0 : pmhd->coarse_w0,
                      coarse_w);
  }
  if (pmhd != nullptr) {
    Kokkos::deep_copy(DevExeSpace(), pmhd->bcc0, bcc);
    Kokkos::deep_copy(DevExeSpace(), pmhd->b0.x1f, b1);
    Kokkos::deep_copy(DevExeSpace(), pmhd->b0.x2f, b2);
    Kokkos::deep_copy(DevExeSpace(), pmhd->b0.x3f, b3);
    if (pm->multilevel) {
      Kokkos::deep_copy(DevExeSpace(), pmhd->coarse_b0.x1f, coarse_b1);
      Kokkos::deep_copy(DevExeSpace(), pmhd->coarse_b0.x2f, coarse_b2);
      Kokkos::deep_copy(DevExeSpace(), pmhd->coarse_b0.x3f, coarse_b3);
    }
  }
}

void Driver::RestoreFullStateSTSTrial(Mesh *pm) {
  diffusion_trial_.Restore(pm);
}

// Called before restriction, and again after regenerated halos but before EOS.
// The collective failure decision is identical on every rank. Do not require a
// maximum-temperature envelope: viscosity and resistivity heat the fluid.
void Driver::CheckFullStateSTSState(Mesh *pm, bool include_ghosts) {
  if (!sts.full_state_retry) return;
  if (!sts.attempt_failed) {
    auto *ph = pm->pmb_pack->phydro;
    auto *pmhd = pm->pmb_pack->pmhd;
    bool admissible = true, active_admissible = true;
    Real viscous_dt = std::numeric_limits<Real>::max();
    Real resistive_dt = std::numeric_limits<Real>::max();
    Real conductive_dt = std::numeric_limits<Real>::max();
    bool check_diffusion_budget = false;
    auto *pvisc = STSViscosity(pm);
    auto *presist = STSSpitzerResistivity(pm);
    auto *pcond = NonlinearSTSConduction(pm);
    if (presist == nullptr && pvisc == nullptr && pmhd != nullptr &&
        pmhd->has_sts_resistivity) presist = pmhd->presist;
    if (pvisc != nullptr) {
      const auto state = pvisc->InspectRawState(
          ph != nullptr ? ph->u0 : pmhd->u0,
          ph != nullptr ? ph->peos->eos_data : pmhd->peos->eos_data,
          ph != nullptr ? nullptr : &pmhd->b0, include_ghosts);
      admissible = state.admissible;
      active_admissible = state.active_admissible;
      if (pvisc->mu_spitzer) {
        viscous_dt = state.diffusion_dt;
        check_diffusion_budget = true;
      }
    }
    if (presist != nullptr) {
      const auto state = presist->InspectRawState(pmhd->u0, pmhd->peos->eos_data,
                                                 pmhd->b0, include_ghosts);
      admissible &= state.admissible;
      active_admissible &= state.active_admissible;
      resistive_dt = state.diffusion_dt;
      check_diffusion_budget = true;
    }
    if (pcond != nullptr) {
      if (admissible) {
        conductive_dt = pcond->RawStageTimeStep(
            ph != nullptr ? ph->u0 : pmhd->u0,
            ph != nullptr ? ph->w0 : pmhd->w0,
            ph != nullptr ? ph->peos->eos_data : pmhd->peos->eos_data,
            ph != nullptr ? nullptr : &pmhd->b0, include_ghosts);
      } else {
        conductive_dt = 0.0;
      }
      check_diffusion_budget = true;
    }
    if (check_diffusion_budget) {
      // Sum inverse bounds of every selected operator, replacing nonlinear
      // coefficients by their raw-stage bounds. Saturation's nonaccelerated
      // donor cap is checked separately before each current-stage flux.
      Real inverse_dt = 0.0;
      for (const auto &process : pm->pmb_pack->parabolic_processes) {
        if (!process.UsesSTS()) continue;
        Real dt = process.ExplicitDt();
        if (pvisc != nullptr && pvisc->mu_spitzer &&
            process.explicit_dt_ptr == &pvisc->dtnew) dt = viscous_dt;
        if (presist != nullptr && process.explicit_dt_ptr == &presist->dtnew) {
          dt = resistive_dt;
        }
        if (pcond != nullptr && process.explicit_dt_ptr == &pcond->dtnew) {
          dt = conductive_dt;
        }
        if (!std::isfinite(dt) || dt <= 0.0) {
          inverse_dt = std::numeric_limits<Real>::infinity();
        } else if (dt < std::numeric_limits<float>::max()) {
          inverse_dt += 1.0/dt;
        }
      }
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &inverse_dt, 1, MPI_ATHENA_REAL,
                    MPI_MAX, MPI_COMM_WORLD);
#endif
      // The RKL2 interval for s stages is at most (s*s+s-2)/4 times
      // the combined explicit bound. Every rank rejects the same interval.
      const Real stages = static_cast<Real>(sts.nstages);
      const Real capacity = 0.25*(stages*stages+stages-2.0)*pm->cfl_no/inverse_dt;
      if (admissible && sts.dt_sweep > 0.0 &&
          (!std::isfinite(inverse_dt) || sts.dt_sweep >
           capacity*(1.0+512.0*std::numeric_limits<Real>::epsilon()))) {
        sts.attempt_failed = true;
        sts.rejection_reason = "stage_diffusion_bound";
      }
    }
    if (!admissible) {
      sts.attempt_failed = true;
      sts.rejection_reason = active_admissible ? "raw_halo" : "raw_active";
    }
  }
  if (sts.attempt_failed) RestoreFullStateSTSTrial(pm);
}

// RKL2 is not SSP: a stable momentum/field recurrence can produce negative raw
// thermal energy. Reject the full U/W/B trial, drain exchanges,
// then halve its interval. This is an admissibility guard, not an error estimator.
void Driver::ExecuteGuardedSTSSweep(Mesh *pm, STSSweep sweep) {
  auto *ph = pm->pmb_pack->phydro;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto *pcond = NonlinearSTSConduction(pm);
  const char *label = pcond != nullptr ? "transport_sts" :
      (STSSpitzerResistivity(pm) != nullptr ? "resistivity_sts" : "viscosity_sts");
  EndSTSSweep();
  diffusion_trial_.Save(pm);
  sts.full_state_retry = true;
  sts.sweep = sweep;
  sts.current_stage = 1;
  CheckFullStateSTSState(pm);
  if (sts.attempt_failed) GuardedSTSFatal("invalid initial active state");
  // Also guard the initial ghost refresh; shortening a trial cannot repair
  // inadmissible interpolation of an unchanged starting state.
  InitBoundaryValuesAndPrimitives(pm);
  if (sts.attempt_failed) GuardedSTSFatal("invalid initial boundary stencil");

  const Real duration = 0.5*pm->dt;
  Real elapsed = 0.0;
  while (elapsed < duration) {
    if (ph != nullptr) {
      ph->RecomputeTimeStepFromCurrentState(this, false);
    } else {
      pmhd->RecomputeTimeStepFromCurrentState(this, false);
    }
    pm->RefreshSTSParabolicTimeStep();
    BeginSTSSweep(pm, sweep);
    sts.enabled = true;
    sts.full_state_retry = true;
    sts.sweep = sweep;
    diffusion_trial_.Save(pm);
    Real interval = duration-elapsed;
    if (pcond != nullptr && pcond->saturated) {
      Real donor_bound = pcond->dt_transport;
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &donor_bound, 1, MPI_ATHENA_REAL,
                    MPI_MIN, MPI_COMM_WORLD);
#endif
      interval = std::min(interval, pm->cfl_no*donor_bound);
    }
    int retries = 0;
    while (true) {
      if (!std::isfinite(interval) || interval <= 0.0 ||
          elapsed+interval <= elapsed || retries > 32) {
        GuardedSTSFatal("trial cannot make progress within 32 interval halvings");
      }
      sts.dt_sweep = interval;
      sts.nstages = parabolic::ComputeRKL2StageCount(interval, pm->dt_parabolic_sts);
      sts.attempt_failed = false;
      sts.rejection_reason = "none";
      int checked_stages = 0;
      for (int stage=1; stage<=sts.nstages; ++stage) {
        SetSTSStage(stage);
        ExecuteTaskList(pm, "before_parabolic_stagen", stage);
        ExecuteTaskList(pm, "parabolic_stagen", stage);
        ExecuteTaskList(pm, "after_parabolic_stagen", stage);
        ++checked_stages;
        // All posted communication must finish before retrying on every rank.
        if (sts.attempt_failed) break;
      }
      if (global_variable::my_rank == 0) {
        const auto precision = std::cout.precision();
        std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10)
                  << label << " cycle=" << pm->ncycle
                  << " sweep=" << (sweep == STSSweep::pre ? "pre" : "post")
                  << " interval=" << interval << " stages=" << sts.nstages
                  << " checked_stages=" << checked_stages
                  << " status=" << (sts.attempt_failed ? "retry" : "accepted")
                  << " diffusion_bound=" << pm->dt_parabolic_sts
                  << " reason=" << sts.rejection_reason << std::endl;
        std::cout.precision(precision);
      }
      if (!sts.attempt_failed) break;
      // Restore again after the completed exchange/prolongation, including CT
      // faces and coarse buffers. No rejected ghost data may survive a retry.
      RestoreFullStateSTSTrial(pm);
      interval *= 0.5;
      ++retries;
    }
    elapsed += interval;
    // Repeated retry subintervals can leave less than one rounding error of
    // the original duration. Do not launch a new recurrence for that residue.
    if (duration-elapsed <= 8.0*std::numeric_limits<Real>::epsilon()*duration) {
      elapsed = duration;
    }
  }
  EndSTSSweep();
  // Only the completed post-sweep updates source/user timestep constraints.
  if (sweep == STSSweep::post) {
    if (ph != nullptr) {
      (void) ph->NewTimeStep(this, nexp_stages);
    } else {
      (void) pmhd->NewTimeStep(this, nexp_stages);
    }
  }
}

void Driver::ExecutePreSTSSweep(Mesh *pm) {
  if (!UsesFullStateSTS(pm)) {
    ExecuteSTSSweep(pm, STSSweep::pre);
    return;
  }
  diffusion_cycle_.Save(pm);
  const char *label = STSSpitzerResistivity(pm) != nullptr ?
                      "resistivity_sts_cfl" : "viscosity_sts_cfl";
  const Real user_dt_start = pm->pgen->dtnew;
  for (int attempt=0; attempt<=32; ++attempt) {
    ExecuteSTSSweep(pm, STSSweep::pre);
    auto *ph = pm->pmb_pack->phydro;
    auto *pmhd = pm->pmb_pack->pmhd;
    if (ph != nullptr) {
      ph->RecomputeTimeStepFromCurrentState(this, false);
    } else {
      pmhd->RecomputeTimeStepFromCurrentState(this, false);
    }
    Real bound = pm->cfl_no*(ph != nullptr ? ph->dtnew : pmhd->dtnew);
    // Built-in source timestep evaluation only updates dtnew: it neither
    // applies sources nor calls user code. Heating can tighten this bound.
    // Recompute for every candidate before accepting its first Strang half.
    auto *psrc = ph != nullptr ? ph->psrc : pmhd->psrc;
    if (psrc != nullptr) {
      if (ph != nullptr) {
        psrc->NewTimeStep(ph->w0, ph->peos->eos_data);
      } else {
        psrc->NewTimeStep(pmhd->w0, pmhd->peos->eos_data);
      }
      bound = std::min(bound, pm->cfl_no*psrc->dtnew);
    }
    // Only code-enrolled, repeat-safe setters may observe retryable states.
    // Validate before taking a minimum so NaNs on one rank cannot be hidden.
    if (pm->pgen->user_dt) {
      (pm->pgen->user_time_step_func)(pm);
      const Real user_bound = pm->pgen->dtnew;
      int invalid = !std::isfinite(user_bound) || user_bound <= 0.0;
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
      if (invalid) GuardedSTSFatal("invalid refreshed user timestep");
      bound = std::min(bound, pm->cfl_no*user_bound);
    }
    // Co-selected explicit diffusion also consumes the heated fluid state.
    for (const auto &process : pm->pmb_pack->parabolic_processes) {
      if (!process.UsesSTS()) bound = std::min(bound, pm->cfl_no*process.ExplicitDt());
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &bound, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
    if (!std::isfinite(bound) || bound <= 0.0) {
      GuardedSTSFatal("invalid refreshed fluid timestep");
    }
    const bool retry = pm->dt > bound*(1.0+64.0*std::numeric_limits<Real>::epsilon());
    const Real next_dt = retry ? std::min(0.5*pm->dt, bound) : pm->dt;
    if (global_variable::my_rank == 0) {
      const auto precision = std::cout.precision();
      std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10)
                << label << " cycle=" << pm->ncycle << " attempt=" << attempt
                << " dt=" << pm->dt << " bound=" << bound
                << " status=" << (retry ? "retry" : "accepted")
                << " next_dt=" << next_dt << std::endl;
      std::cout.precision(precision);
    }
    if (!retry) return;
    // Shortening only the fluid step would leave the first Strang half at the
    // wrong duration. Restore the cycle start before repeating that half.
    diffusion_cycle_.Restore(pm);
    if (pm->pgen->user_dt) pm->pgen->dtnew = user_dt_start;
    pm->dt = next_dt;
    if (pm->time+pm->dt <= pm->time) GuardedSTSFatal("cycle retry makes no progress");
    RefreshSTSCycleState(pm);
  }
  GuardedSTSFatal("fluid CFL retry exceeded 32 reductions");
}

// A nonlinear residual is reevaluated at every RKL stage. Freeze only the
// recurrence coefficients within a trial, not conductivity or saturated flux.
// The row-budget stage count is not a nonlinear stability theorem: check raw
// states before floors/restriction and retry an entire trial at half interval.
// There is no time-error estimator here; accuracy needs time-refinement tests.
void Driver::ExecuteSTSSweep(Mesh *pm, STSSweep sweep) {
  if (pm->sts_integrator == parabolic::STSIntegrator::none) return;
  if (UsesFullStateSTS(pm)) {
    ExecuteGuardedSTSSweep(pm, sweep);
    return;
  }
  auto *pcond = NonlinearSTSConduction(pm);
  if (pcond == nullptr) {
    // Preserve the original constant-coefficient scheduling and arithmetic.
    if (sweep == STSSweep::post) pm->RefreshSTSParabolicTimeStep();
    if (sweep == STSSweep::pre && !sts.enabled) return;
    BeginSTSSweep(pm, sweep);
    if (sts.enabled) {
      for (int stage=1; stage<=sts.nstages; ++stage) {
        SetSTSStage(stage);
        ExecuteTaskList(pm, "before_parabolic_stagen", stage);
        ExecuteTaskList(pm, "parabolic_stagen", stage);
        ExecuteTaskList(pm, "after_parabolic_stagen", stage);
      }
      EndSTSSweep();
    }
    return;
  }

  auto *ph = pm->pmb_pack->phydro;
  auto *pmhd = pm->pmb_pack->pmhd;
  auto &u = ph != nullptr ? ph->u0 : pmhd->u0;
  auto &w = ph != nullptr ? ph->w0 : pmhd->w0;
  auto &snapshot = ph != nullptr ? ph->u_sts0 : pmhd->u_sts0;
  const auto &eos = ph != nullptr ? ph->peos->eos_data : pmhd->peos->eos_data;
  const DvceFaceFld4D<Real> *b = ph != nullptr ? nullptr : &pmhd->b0;

  // WorkInLoop and sources may have changed U since the last primitive
  // conversion. Refresh from current U before BOTH split halves. Existing
  // initialization exchanges complete all sends/receives before returning.
  EndSTSSweep();
  InitBoundaryValuesAndPrimitives(pm);
  const Real duration = 0.5*pm->dt;
  Real elapsed = 0.0;
  while (elapsed < duration) {
    pcond->ValidateRawState(u, eos, b);
    pcond->NewTimeStep(w, eos);
    pm->RefreshSTSParabolicTimeStep();
    BeginSTSSweep(pm, sweep);
    // A vanishing operator still advances the requested interval with the
    // minimum RKL recurrence; do not silently skip a split half.
    sts.enabled = true;
    sts.nonlinear = true;
    sts.sweep = sweep;
    const auto start = pcond->InspectRawState(u, eos, b);
    sts.temperature_min = start.minimum;
    sts.temperature_max = start.maximum;
    Real donor_bound = pcond->dt_transport;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &donor_bound, 1, MPI_ATHENA_REAL,
                  MPI_MIN, MPI_COMM_WORLD);
#endif
    Real interval = std::min(duration-elapsed, pm->cfl_no*donor_bound);
    pcond->BeginSTSTrial(u, w, snapshot);
    int retries = 0;
    while (true) {
      if (!std::isfinite(interval) || interval <= 0.0 ||
          elapsed+interval <= elapsed || retries > 32) {
        NonlinearSTSFatal("trial cannot make progress within 32 interval halvings");
      }
      sts.dt_sweep = interval;
      sts.nstages = parabolic::ComputeRKL2StageCount(interval, pm->dt_parabolic_sts);
      sts.attempt_failed = false;
      sts.rejection_reason = "none";
      int checked_stages = 0;
      for (int stage=1; stage<=sts.nstages; ++stage) {
        SetSTSStage(stage);
        ExecuteTaskList(pm, "before_parabolic_stagen", stage);
        ExecuteTaskList(pm, "parabolic_stagen", stage);
        ExecuteTaskList(pm, "after_parabolic_stagen", stage);
        ++checked_stages;
        // Only exit after all posted exchanges/clears complete on all ranks.
        if (sts.attempt_failed) break;
      }
      if (global_variable::my_rank == 0) {
        const auto precision = std::cout.precision();
        std::cout << std::setprecision(std::numeric_limits<Real>::max_digits10)
                  << "nonlinear_sts cycle=" << pm->ncycle
                  << " sweep=" << (sweep == STSSweep::pre ? "pre" : "post")
                  << " interval=" << interval << " stages=" << sts.nstages
                  << " checked_stages=" << checked_stages
                  << " status=" << (sts.attempt_failed ? "retry" : "accepted")
                  << " donor_bound=" << pm->cfl_no*donor_bound
                  << " diffusion_bound=" << pm->dt_parabolic_sts
                  << " reason=" << sts.rejection_reason << std::endl;
        std::cout.precision(precision);
      }
      if (!sts.attempt_failed) break;
      // Also restore after boundary exchange: no failed trial may leak ghost
      // changes or a reconstructed primitive state into the next attempt.
      pcond->RestoreSTSTrial(u, w, snapshot);
      interval *= 0.5;
      ++retries;
    }
    elapsed += interval;
  }
  EndSTSSweep();
}

//----------------------------------------------------------------------------------------
//! \fn Driver::ExecuteTaskList()
//! \brief Perform tasks over all MeshBlocks for the TaskList specified by string "tl".
//! Integer argument "stage" can be used to indicate at which step in overall algorithm
//! these tasks are to be performed, e.g. which stage of a multi-stage RK integrator.

void Driver::ExecuteTaskList(Mesh *pm, std::string tl, int stage) {
  MeshBlockPack* pmbp = pm->pmb_pack;
  for (int p=0; p<(pm->nmb_packs_thisrank); ++p) {
    if (!(pmbp->tl_map[tl]->Empty())) {pmbp->tl_map[tl]->Reset();}
  }
  int npack_left = (pm->nmb_packs_thisrank);
  while (npack_left > 0) {
    if (pmbp->tl_map[tl]->Empty()) {
      npack_left--;
    } else {
      if (!pmbp->tl_map[tl]->IsComplete()) {
        auto status = pmbp->tl_map[tl]->DoAvailable(this, stage);
        if (status == TaskListStatus::complete) { npack_left--; }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Driver::Initialize()
// Tasks to be performed before execution of Driver, such as setting ghost zones (BCs),
//  outputting ICs, and computing initial time step

void Driver::Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool res_flag) {
  const Real saved_next_dt = pmesh->dt;
  const Real saved_dtold = pmesh->dtold;
  //---- Step 1.  Set conserved variables in ghost zones for all physics
  InitBoundaryValuesAndPrimitives(pmesh);

  //---- Step 2.  Compute time step (if problem involves time evolution)
  hydro::Hydro *phydro = pmesh->pmb_pack->phydro;
  mhd::MHD *pmhd = pmesh->pmb_pack->pmhd;
  radiation::Radiation *prad = pmesh->pmb_pack->prad;
  z4c::Z4c *pz4c = pmesh->pmb_pack->pz4c;
  TurbulenceDriver *pturb = pmesh->pmb_pack->pturb;
  if (time_evolution != TimeEvolution::tstatic) {
    if (phydro != nullptr) {
      (void) pmesh->pmb_pack->phydro->NewTimeStep(this, nexp_stages);
    }
    if (pmhd != nullptr) {
      (void) pmesh->pmb_pack->pmhd->NewTimeStep(this, nexp_stages);
    }
    if (prad != nullptr) {
      (void) pmesh->pmb_pack->prad->NewTimeStep(this, nexp_stages);
    }
    if (pz4c != nullptr) {
      (void) pmesh->pmb_pack->pz4c->NewTimeStep(this, nexp_stages);
    }
    if (pturb != nullptr) {
      (void) pmesh->pmb_pack->pturb->NewTimeStep(this, nexp_stages);
    }

    pmesh->NewTimeStep(tlim);
    if (res_flag && pmesh->restart_next_dt) {
      // This dt was already selected at a complete checkpoint boundary. Reapply
      // tighter current constraints, but do not grant another growth allowance
      // or lose a saved particle accuracy cap during restart initialization.
      pmesh->dt = std::min(pmesh->dt, saved_next_dt);
      pmesh->dtold = saved_dtold;
    }
    RefreshSTSCycleState(pmesh);
  }
  pmesh->checkpoint_ready = true;

  //---- Step 3.  Cycle through output Types and load data / write files.
  if (!res_flag) { // only write outputs at the beginning of the run
    ReserveOutputCounters(pout->pout_list, pin, pmesh->time);
    for (auto &out : pout->pout_list) {
      out->LoadOutputData(pmesh);
      out->WriteOutputFile(pmesh, pin);
    }
  }

  //---- Step 4.  Initialize various counters, timers, etc.
  run_time_.reset();
  nmb_updated_ = 0;

  // allocate memory for stiff source terms with ImEx integrators
  // only implemented for ion-neutral two fluid for now
  ion_neutral::IonNeutral *pionn = pmesh->pmb_pack->pionn;
  if (pionn != nullptr) {
    if (nimp_stages == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "IonNeutral MHD can only be run with ImEx integrators."
          << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int nmb = std::max((pmesh->pmb_pack->nmb_thispack), (pmesh->nmb_maxperrank));
    auto &indcs = pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(impl_src, nimp_stages, nmb, 8, ncells3, ncells2, ncells1);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn Driver::Execute()
//! \brief Executes "main loop" by running all relevant task lists over all MeshBlockPacks
//! until a relevant stopping criteria is found (e.g. t > tlim). Calls AMR driver, and
//! performs outputs. Updates counters like (ncycle, time, etc.)

void Driver::Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool wdflag) {
  if (global_variable::my_rank == 0) {
    std::cout << "\nSetup complete, executing task list(s)...\n" << std::endl;
  }

  if (time_evolution == TimeEvolution::tstatic) {
    std::cout << "\nStatic time evolution selected, solving steady-state problem...\n"
              << std::endl;
    // TODO(@user): add work for time static problems here
  } else {
    Real elapsed_time = -1.;
    if (wall_time > 0.) {
      elapsed_time = UpdateWallClock();
    }
    while ((pmesh->time < tlim) && (pmesh->ncycle < nlim || nlim < 0) &&
           (elapsed_time < wall_time)) {
      pmesh->checkpoint_ready = false;
      if (global_variable::my_rank == 0) {OutputCycleDiagnostics(pmesh);}
      if (wdflag) {WatchDog(0);}

      ExecutePreSTSSweep(pmesh);

      // Execute TaskLists
      // Work before time integrator indicated by "0" in stage
      ExecuteTaskList(pmesh, "before_timeintegrator", 0);
      // time-integrator tasks for each stage of integrator
      for (int stage=1; stage<=(nexp_stages); ++stage) {
        ExecuteTaskList(pmesh, "before_stagen", stage);
        // solve gravity at each RK stage so the potential is consistent
        // with the current density (required for 2nd-order accuracy)
        if (pmesh->pmb_pack->pgrav != nullptr)
            {pmesh->pmb_pack->pgrav->pmgd->Solve(this, stage);}
        ExecuteTaskList(pmesh, "stagen", stage);
        ExecuteTaskList(pmesh, "after_stagen", stage);
      }

      // Work after time integrator indicated by "1" in stage
      ExecuteTaskList(pmesh, "after_timeintegrator", 1);

      // Diffusion coefficients may depend on the post-RK state. Refresh the global
      // stability bound before choosing the second Strang-split half-sweep.
      ExecuteSTSSweep(pmesh, STSSweep::post);

      // Work outside of TaskLists:
      // increment time, ncycle, etc.
      pmesh->time = pmesh->time + pmesh->dt;
      pmesh->ncycle++;
      pmesh->dt_last_completed = pmesh->dt;
      nmb_updated_ += pmesh->nmb_total;
      npart_updated_ += pmesh->nprtcl_total;
      // load balancing efficiency
      if (global_variable::nranks > 1) {
        int minnmb = std::numeric_limits<int>::max();
        for (int i=0; i<global_variable::nranks; ++i) {
          minnmb = std::min(minnmb, pmesh->nmb_eachrank[i]);
        }
        lb_efficiency_ += static_cast<float>(minnmb*(global_variable::nranks))/
            static_cast<float>(pmesh->nmb_total);
      }

      // Keep analysis outputs here (some consume completed-step fluxes). Stage
      // only the list of due checkpoints, not their old-topology data.
      pmesh->checkpoint_ready = false;
      std::vector<BaseTypeOutput*> due_checkpoints;
      // Test for/make outputs
      for (auto &out : pout->pout_list) {
        // compare at floating point (32-bit) precision to reduce effect of round off
        float time_32 = static_cast<float>(pmesh->time);
        float next_32 = static_cast<float>(out->out_params.last_time+out->out_params.dt);
        float tlim_32 = static_cast<float>(tlim);
        float elapsed_time_32 = static_cast<float>(UpdateWallClock());
        float next_wall_32 = static_cast<float>(out->out_params.last_wall_time+
                                                 out->out_params.dt_wall);
        int &dcycle_ = out->out_params.dcycle;

        if (((out->out_params.dt > 0.0) && ((time_32 >= next_32) && (time_32<tlim_32))) ||
            ((out->out_params.dt_wall > 0.0) && ((elapsed_time_32 >= next_wall_32))) ||
            ((dcycle_ > 0) && ((pmesh->ncycle)%(dcycle_) == 0)) ) {
          // update the elapsed time
          out->out_params.last_wall_time = (Real) elapsed_time_32;

          if (out->out_params.file_type == "rst" || out->out_params.file_type == "prst") {
            due_checkpoints.push_back(out);
          } else {
            out->LoadOutputData(pmesh);
            out->WriteOutputFile(pmesh, pin);
          }
        }
      }

      // AMR
      if (pmesh->adaptive) {pmesh->pmr->AdaptiveMeshRefinement(this, pin);}
      // compute new timestep AFTER all Meshblocks refined/derefined
      pmesh->NewTimeStep(tlim);
      RefreshSTSCycleState(pmesh);
      pmesh->checkpoint_ready = true;
      ReserveOutputCounters(due_checkpoints, pin, pmesh->time);
      for (auto *out : due_checkpoints) {
        out->LoadOutputData(pmesh);
        out->WriteOutputFile(pmesh, pin);
      }

      // Update wall clock time if needed.
      if (wall_time > 0.) {
        elapsed_time = UpdateWallClock();
      }
    }  // end while
  }    // end of (time_evolution != tstatic) clause
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::Finalize()
//! \brief Tasks to be performed after execution of Driver, such as making final output
//!  and printing diagnostic messages

void Driver::Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {
  ReserveOutputCounters(pout->pout_list, pin, pmesh->time);
  // cycle through output Types and load data / write files
  //  This design allows for asynchronous outputs to implemented in the future.
  for (auto &out : pout->pout_list) {
    out->LoadOutputData(pmesh);
    out->WriteOutputFile(pmesh, pin);
  }

  // call any problem specific functions to do work after main loop
  if (pmesh->pgen->pgen_final_func != nullptr) {
    (pmesh->pgen->pgen_final_func)(pin, pmesh);
  }

  float exe_time = run_time_.seconds();

  if (time_evolution != TimeEvolution::tstatic) {
#if MPI_PARALLEL_ENABLED
    // Collect number of MeshBlocks communicated during load balancing across all ranks
    if (pmesh->adaptive) {
      MPI_Allreduce(MPI_IN_PLACE, &(pmesh->pmr->nmb_sent_thisrank), 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);
    }
#endif
    if (global_variable::my_rank == 0) {
      // Print diagnostic messages related to the end of the simulation
      OutputCycleDiagnostics(pmesh);
      if (pmesh->ncycle == nlim) {
        std::cout << std::endl << "Terminating on cycle limit" << std::endl;
      } else if (pmesh->time >= tlim) {
        std::cout << std::endl << "Terminating on time limit" << std::endl;
      } else {
        std::cout << std::endl << "Terminating on wall clock limit" << std::endl;
      }

      std::cout << "time=" << pmesh->time << " cycle=" << pmesh->ncycle << std::endl;
      std::cout << "tlim=" << tlim << " nlim=" << nlim << std::endl;

      if (pmesh->adaptive) {
        std::cout << std::endl << "Current number of MeshBlocks = " << pmesh->nmb_total
          << std::endl << pmesh->pmr->nmb_created << " MeshBlocks created, "
          << pmesh->pmr->nmb_deleted << " deleted by AMR" << std::endl;
#if MPI_PARALLEL_ENABLED
        std::cout << pmesh->pmr->nmb_sent_thisrank << " communicated for load balancing, "
          <<"load balancing efficiency = " << (lb_efficiency_/pmesh->ncycle) << std::endl;
#endif
      }

      // Calculate and print the zone-cycles/cpu-second
      // Note the need for 64-bit integers since nmb_updated can easily exceed 2^32.
      std::uint64_t zonecycles = nmb_updated_ *
                                 static_cast<uint64_t>(pmesh->NumberOfMeshBlockCells());
      float zcps = static_cast<float>(zonecycles) / exe_time;
      float pups = static_cast<float>(npart_updated_) / exe_time;

      std::cout << std::endl << "MeshBlock-cycles = " << nmb_updated_ << std::endl;
      std::cout << "cpu time used  = " << exe_time << std::endl;
      std::cout << "zone-cycles/cpu_second = " << zcps << std::endl;
      std::cout << "particle-updates/cpu_second = " << pups << std::endl;
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::OutputCycleDiagnostics()
//! \brief Simple function to print diagnostics every 'ndiag' cycles to stdout

void Driver::OutputCycleDiagnostics(Mesh *pm) {
//  const int dtprcsn = std::numeric_limits<Real>::max_digits10 - 1;
  const int dtprcsn = 6;
  if (pm->ncycle % ndiag == 0) {
    Real elapsed = pwall_clock_->seconds();
    std::cout << "elapsed=" << std::scientific << std::setprecision(dtprcsn) << elapsed
              << " cycle=" << pm->ncycle
              << " time=" << pm->time << " dt=" << pm->dt << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::UpdateWallClock()
//! \brief Update and sync the wall clock across all MPI ranks. This is necessary because
//! the different MPI ranks may 1) initialize their timers at slightly different times,
//! and 2) may reach the end of a loop to update their timers at slightly different times.
//! This may result in a weird problem where one or more ranks have timers that fall
//! slightly below the wall clock time while others determine that it's time to quit.

Real Driver::UpdateWallClock() {
  Real tnow = 0.0;
  if (global_variable::my_rank == 0) {
    tnow = pwall_clock_->seconds();
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&tnow, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  return tnow;
}

//----------------------------------------------------------------------------------------
//! \fn Driver::InitBoundaryValuesAndPrimitives()
//! \brief Sets boundary conditions on conserved and initializes primitives.  Used both
//! on initialization, and when new MBs created with AMR.

void Driver::InitBoundaryValuesAndPrimitives(Mesh *pm) {
  // Note: with MPI, sends on ALL MBs must be complete before receives execute

  // Initialize Z4c
  z4c::Z4c *pz4c = pm->pmb_pack->pz4c;
  if (pz4c != nullptr) {
    (void) pz4c->RestrictU(this, 0);
    (void) pz4c->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) pz4c->SendU(this, 0);
    (void) pz4c->ClearSend(this, -1);
    (void) pz4c->ClearRecv(this, -1);
    (void) pz4c->RecvU(this, 0);
    (void) pz4c->Z4cBoundaryRHS(this, 0);
    (void) pz4c->Prolongate(this, 0); // coarse grid BCs and prolongation
    (void) pz4c->ApplyPhysicalBCs(this, 0); // fine grid BCs
  }

  // Initialize HYDRO: ghost zones and primitive variables (everywhere)
  // includes communications for shearing box boundaries
  hydro::Hydro *phydro = pm->pmb_pack->phydro;
  if (phydro != nullptr) {
    // following functions return a TaskStatus, but it is ignored so cast to (void)
    (void) phydro->RestrictU(this, 0);
    (void) phydro->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) phydro->SendU(this, 0);
    (void) phydro->ClearSend(this, -1); // stage = -1 only clear SendU
    (void) phydro->ClearRecv(this, -1); // stage = -1 only clear RecvU
    (void) phydro->RecvU(this, 0);
    (void) phydro->SendU_Shr(this, 0);
    (void) phydro->ClearSend(this, -4); // stage = -4 only clear SendU_Shr
    (void) phydro->ClearRecv(this, -4); // stage = -4 only clear RecvU_Shr
    (void) phydro->RecvU_Shr(this, 0);
    (void) phydro->Prolongate(this, 0); // coarse grid BCs and prolongation
    (void) phydro->ApplyPhysicalBCs(this, 0); // fine grid BCs
    (void) phydro->ConToPrim(this, 0);
  }

  // Initialize MHD: ghost zones and primitive variables (everywhere)
  // includes communications for shearing box boundaries
  mhd::MHD *pmhd = pm->pmb_pack->pmhd;
  dyngr::DynGRMHD *pdyngr = pm->pmb_pack->pdyngr;
  if (pmhd != nullptr) {
    (void) pmhd->RestrictU(this, 0);
    (void) pmhd->RestrictB(this, 0);
    (void) pmhd->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) pmhd->SendU(this, 0);
    (void) pmhd->SendB(this, 0);
    (void) pmhd->ClearSend(this, -1); // stage = -1 only clear SendU, SendB
    (void) pmhd->ClearRecv(this, -1); // stage = -1 only clear RecvU, RecvB
    (void) pmhd->RecvU(this, 0);
    (void) pmhd->RecvB(this, 0);
    (void) pmhd->SendU_Shr(this, 0);
    (void) pmhd->SendB_Shr(this, 0);
    (void) pmhd->ClearSend(this, -4); // stage = -4 only clear SendU_Shr, SendB_Shr
    (void) pmhd->ClearRecv(this, -4); // stage = -4 only clear RecvU_Shr, SendB_Shr
    (void) pmhd->RecvU_Shr(this, 0);
    (void) pmhd->RecvB_Shr(this, 0);
    (void) pmhd->Prolongate(this, 0); // coarse grid BCs and prolongation
    (void) pmhd->ApplyPhysicalBCs(this, 0); // fine grid BCs
    if (pdyngr == nullptr) {
      (void) pmhd->ConToPrim(this, 0);
    } else {
      if (pz4c != nullptr) {
        (void) pz4c->ConvertZ4cToADM(this, 0);
      }
      (void) pdyngr->ConToPrim(this, 0);
    }
  }

  // Initialize radiation: ghost zones and intensity (everywhere)
  // DOES NOT include communications for shearing box boundaries
  radiation::Radiation *prad = pm->pmb_pack->prad;
  if (prad != nullptr) {
    (void) prad->RestrictI(this, 0);
    (void) prad->InitRecv(this, -1);  // stage < 0 suppresses InitFluxRecv
    (void) prad->SendI(this, 0);
    (void) prad->ClearSend(this, -1);
    (void) prad->ClearRecv(this, -1);
    (void) prad->RecvI(this, 0);
    (void) prad->Prolongate(this, 0); // coarse grid BCs and prolongation
    (void) prad->ApplyPhysicalBCs(this, 0); // fine grid BCs
  }

  return;
}
