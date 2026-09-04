//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file turb_ti.cpp
//! \brief Problem generator for turbulence simulations with cooling thermal instability
//! - written by Rajsekhar Mohapatra 2022-2025
//!
//! The code includes functions to:
//!  - Determine a suitable time step for cooling and turbulence updates.
//!  - Apply explicit source terms to models, such as cooling, density ceilings,
//!    equilibrium heating, and temperature ceilings.
//!  - Ensure that the simulation remains stable by subcycling through cooling
//!    steps where necessary and compensating for unbalanced cooling via
//!    equilibrium heating.
//!  - Diagnose and print out key simulation metrics, including minimum and
//!    maximum values of density, velocity, temperature, and internal energy.
//!
//! Main functionalities are divided into the following sections:
//!  - UserTimeStep: Determines the new time step based on the minimum
//!    cooling time across mesh cells.
//!  - AddUserSrcs: Dispatches calls to additional user-defined source terms.
//!  - AddISMCooling: Implements ISM (Interstellar Medium) cooling with optional
//!    subcycling for computational stability.
//!  - ApplyTempCeiling: Imposes a temperature ceiling, preventing unphysically
//!    high temperatures.
//!  - AddEquHeating: Applies equilibrium heating to counteract cooling that is
//!    not otherwise balanced, helping maintain thermal equilibrium.
//!  - Diagnostic: Collects and prints global minimum and maximum field values,
//!    total mass, energy, and rates of cooling and heating at diagnostic
//!    intervals.
//!
//! Reusability and Extension:
//!  - Although specific to turbulence or ISM cooling modeling, individual
//!    source-term functions can be adapted for other physics processes requiring
//!    explicit time-step constraints and subcycling. 
// ----------------------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <random>
#include <iomanip>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
// #include "srcterms/ismcooling_zsol0_30.hpp"
#include "srcterms/ismcooling_zinterp.hpp"
#include "globals.hpp"
#include "units/units.hpp"
#include "utils/random.hpp"
#include "srcterms/turb_driver.hpp"
#include "turb_init.hpp"
#include "mag_init.hpp"
#include "densfluc_init.hpp"
// #include "srcterms/TurbGen.h"

#include <Kokkos_Random.hpp>

#define NREDUCTION_SLAB 24 // number of slabs into which the z-direction is decomposed
KOKKOS_INLINE_FUNCTION
Real pow_dev(const Real x, const Real y) {
  return ::pow(x, y); // uses pow/powf from <cmath>; available on device if compiled with device math
}
namespace {
struct pgen_turb {
  int  ndiag;
  Real T_cold;          // temperature cutoff for defining cold phase gas
  Real T_hot;           // temperature cutoff for defining hot phase gas
  Real x_h;             // fraction of hydrogen
  bool cool_subc;       // toggle cooling subcycling on/off
  Real cool_subfac;     // ratio between cooling time and dt used for time stepping
                        // - used in user time step function 
  bool use_temp_floor_cool; // turn on a floor for the radiative cooling
  Real T_floor_cool;    // switch off radiative cooling below for T<T_floor_cool
  bool use_temp_ceiling_cool; // turn on a ceiling for the radiative cooling
  Real T_ceiling_cool;  // temperature ceiling for cooling
  Real t_ceiling_start; // time after which temperature ceiling on cooling is applied
  bool use_temp_ceiling;// toggle temperature ceiling on/off
  Real T_ceiling;       // maximum allowed temperature in the system
  bool use_dens_ceiling;// toggle density ceiling for cooling on/off
  bool use_equ_heating; // toggle equilibrium heating on/off
  bool equ_thot_only;   // option to balance cooling of gas only above T=T_hot
  int  heat_cycle;      // equilibrium heating rate updated every ncycles
  int  heat_weight;     // 1 for density-weighted heating, 0 for volume heating
  Real heat_fraction;   // fraction of net cooling to be balanced by equilibrium heating
  bool use_equ_heat_ceiling; // Ceiling on maximum heating rate due to equilibrium heating
  bool const_ratio_equ_heating; // Sets equ_heating to cooling*(1-hc_ratio) after each update
  bool const_ratio_equ_heating_L0; // Sets equ_heating to cooling-turb_heating after each update
  Real max_equ_heat_rate; // set a maximum heating rate for the equ heating
  Real tot_vol;         // store total volume here
  Real tot_mass;        // store total mass here
  Real tot_coolrate;    // store total cooling rate
  Real tot_turbheatrate;// store total turbulent heatinging rate
  Real tot_equheatrate; // store total equilibrium heating rate
  Real dens_ceiling;    // switch off cooling for when density exceeds this value
  Real rho0;            // value of density at z=0, t=0
  Real pres0;           // value of pressure at z=0, t=0
  Real metallicity;     // metallicity of the gas
  Real hrate_euv;       // heating rate by extragalactic UV background
  Real hc_ratio;        // ratio between heating due to turbulence and net cooling at t=0
};
  pgen_turb* pti_turb = new pgen_turb();

void AddUserSrcs(Mesh *pm, const Real bdt);
void UserTimeStep(Mesh *pm);
void UserWorkInLoop(Mesh *pm);
void Diagnostic(Mesh *pm);
void AddISMCooling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void ApplyTempCeiling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void AddEquHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data);
void TurbTiFinalWork(ParameterInput *pin, Mesh *pm);
void UserHistOutput(HistoryData *pdata, Mesh *pm);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Problem Generator for nonlinear thermal instability

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (global_variable::my_rank == 0) std::cout << "Pgen start" << std::endl;
  user_srcs_func = AddUserSrcs;
  user_hist_func = UserHistOutput;
  user_time_step_func = UserTimeStep;
  user_work_in_loop_func = UserWorkInLoop;
  pgen_final_func = TurbTiFinalWork;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
       << "Supernovae-thermal instability problem generator can only be run with Hydro and/or MHD, "
       << "but no <hydro> or <mhd> block in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nmb1 = (pmbp->nmb_thispack-1);

  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;

  EOS_Data &eos = (pmbp->pmhd != nullptr) ?
                pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;

  // Get temperature in Kelvin
  Real temp_cgs = pin->GetOrAddReal("problem","temp",3e6);
  Real temp_0 = temp_cgs/pmbp->punit->temperature_cgs();
  pti_turb->T_cold = pin->GetOrAddReal("problem","temp_cold",2e4)/pmbp->punit->temperature_cgs();
  pti_turb->T_hot = pin->GetOrAddReal("problem","temp_hot",1e6)/pmbp->punit->temperature_cgs();
  pti_turb->x_h = pin->GetOrAddReal("problem","x_h",0.715);
  pti_turb->hc_ratio = pin->GetOrAddReal("problem","hc_ratio", 1.02);
  pti_turb->metallicity = pin->GetOrAddReal("problem","metallicity", 0.3);
  Real metallicity = pti_turb->metallicity;
  pti_turb->use_temp_floor_cool = pin->GetOrAddBoolean("problem","use_temp_floor_cool", false);
  if (pti_turb->use_temp_floor_cool) pti_turb->T_floor_cool = pin->GetOrAddReal("problem","temp_floor_cool", 2e4)/pmbp->punit->temperature_cgs();
  else pti_turb->T_floor_cool = eos.tfloor; // If user has not set a floor on the cooling, pass the eos floor
  pti_turb->use_temp_ceiling_cool = pin->GetOrAddBoolean("problem","use_temp_ceiling_cool", false);
  if (pti_turb->use_temp_ceiling_cool) {
    pti_turb->T_ceiling_cool = pin->GetOrAddReal("problem","temp_ceiling_cool", -1.0)/pmbp->punit->temperature_cgs();
    pti_turb->t_ceiling_start = pin->GetOrAddReal("problem","t_ceiling_start", 0.0);
  }
  else {
    pti_turb->T_ceiling_cool = -1.0; // If user has not set a floor on the cooling, pass -1.0
    pti_turb->t_ceiling_start = 0.0;
  }
  // Would need to change the above for mhd

  pti_turb->use_temp_ceiling = pin->GetOrAddBoolean("problem","use_temp_ceiling", false);
  if (pti_turb->use_temp_ceiling) pti_turb->T_ceiling = pin->GetOrAddReal("problem","temp_ceiling", 1e9)/pmbp->punit->temperature_cgs();
  pti_turb->cool_subc = pin->GetOrAddBoolean("problem","cool_subc",false);
  pti_turb->cool_subfac = pin->GetOrAddReal("problem","cool_subfac", 1.0);
  pti_turb->use_dens_ceiling = pin->GetOrAddBoolean("problem","use_dens_ceiling_cool", false);
  if (pti_turb->use_dens_ceiling){
    Real n0_ceiling = pin->GetOrAddReal("problem","n0_ceiling_cool", 20.0);
    pti_turb->dens_ceiling = n0_ceiling*pmbp->punit->mu()*pmbp->punit->atomic_mass_unit_cgs/pmbp->punit->density_cgs();
  }
  else{
    pti_turb->dens_ceiling = FLT_MAX;
  } 
  pti_turb->ndiag = pin->GetOrAddInteger("problem","ndiag",-1);

  Real number_density = pin->GetOrAddReal("problem", "n0", 0.1);
  Real rho_0 = number_density*pmbp->punit->mu()*
               pmbp->punit->atomic_mass_unit_cgs/pmbp->punit->density_cgs();
  pti_turb->rho0 = rho_0;
  int64_t seed_perturb = pin->GetOrAddInteger("problem","seed_perturb",-1);
  Real sigma_perturb = pin->GetOrAddReal("problem","sigma_perturb",0.0);

  pti_turb->use_equ_heating = pin->GetOrAddBoolean("problem","use_equ_heating", false);
  // Switch off equilibrium heating if hc ratio is greater than 1
  Real equ_ceiling_factor = 1.0;
  if(pti_turb->use_equ_heating ==true){
    if (global_variable::my_rank == 0){
      std::cout << "Equilibrium heating turned on." << std::endl;
    }
    pti_turb->heat_cycle = pin->GetOrAddInteger("problem","equ_heat_cycle", 10); // number of cycles in which we update the equilibrium heating rate
    pti_turb->equ_thot_only = pin->GetOrAddBoolean("problem","equ_thot_only", false); // Equilibrium heating will only balance cooling of hot gas
    pti_turb->heat_weight = pin->GetOrAddInteger("problem","heat_weight",1); // 1 for density-weighted heating, 0 for volume heating
    pti_turb->heat_fraction = pin->GetOrAddReal("problem","heat_fraction",1.0); // fraction of net cooling (cooling-turb heating) to be balanced by equilibrium heating
    pti_turb->const_ratio_equ_heating = pin->GetOrAddBoolean("problem","const_ratio_equ_heating", false); // equilibrium heating is set to cooling*(1-hc_ratio) after each update
    pti_turb->const_ratio_equ_heating_L0 = pin->GetOrAddBoolean("problem","const_ratio_equ_heating_L0", false); // equilibrium heating is set to cooling-heating_sn after each update
    pti_turb->use_equ_heat_ceiling = pin->GetOrAddBoolean("problem","use_equ_heat_ceiling", false); // Ceiling on maximum heating rate due to equilibrium heating 
    if (pti_turb->use_equ_heat_ceiling) equ_ceiling_factor = pin->GetOrAddReal("problem","equ_ceiling_factor",10.0); // Ratio between maximum equlibrium heating rate and initial cooling rate
  }
  Real pres_0 = number_density*pmbp->punit->k_boltzmann_cgs*temp_cgs/pmbp->punit->pressure_cgs();
  pti_turb->pres0 = pres_0;
  Real T_floor_cool_cgs = pti_turb->T_floor_cool*pmbp->punit->temperature_cgs();
  Real T_ceiling_cool_cgs = pti_turb->T_ceiling_cool*pmbp->punit->temperature_cgs();

  if (global_variable::my_rank == 0) std::cout << "Input values scanned" << std::endl;

  // Initialize Hydro/MHD variables -------------------------------
  if (pmbp->phydro != nullptr || pmbp->pmhd != nullptr) {

    Real totcoolrate = 0.0;
    Real totvol = 0.0;
    Real totmass = 0.0;
    if (global_variable::my_rank == 0) std::cout << "Now calculating total initial cooling rate" << std::endl;

    Real n_unit = pmbp->punit->density_cgs()/pmbp->punit->mu()
                  /pmbp->punit->atomic_mass_unit_cgs;
    Real mu = pmbp->punit->mu();
    Real x_h = pti_turb->x_h;
    Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()
                        /n_unit/n_unit;
    Real heating_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/n_unit;
    Real heating_rate = pti_turb->hrate_euv;
    seed_perturb += global_variable::my_rank;
    auto &size = pmbp->pmb->mb_size;
    // Now calculate total cooling rate using a global sum
    Kokkos::parallel_reduce("PgenTotCoolRate", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &totcoolrate_, Real &totvol_, Real &totmass_) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      k += ks;
      j += js;
      Real rho = rho_0;
      if(sigma_perturb > FLT_MIN){
        int64_t seed_perturb_loc = seed_perturb*nmb1-m;
        rho *= (1.0 +  sigma_perturb * RanGaussian((int64_t*)(&seed_perturb_loc)));
      } 
      Real lambda_cooling = SQR(x_h*mu)*ISMCoolFn(temp_cgs, T_floor_cool_cgs, T_ceiling_cool_cgs, metallicity)/cooling_unit;
      Real gamma_heating = heating_rate/heating_unit;
      Real dvol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      totvol_ += dvol;
      totmass_ += rho*dvol;
      totcoolrate_ += dvol * rho * (rho * lambda_cooling - gamma_heating);
    }, Kokkos::Sum<Real>(totcoolrate),
       Kokkos::Sum<Real>(totvol), 
       Kokkos::Sum<Real>(totmass));

#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &totcoolrate, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &totvol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &totmass,1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif 
    pti_turb->tot_coolrate = totcoolrate;
    pti_turb->max_equ_heat_rate = equ_ceiling_factor * totcoolrate / totvol;


  

    // Print info
    if (global_variable::my_rank == 0) {
      std::cout << "============== Check Initialization ===============" << std::endl;
      std::cout << "  rho_0  (code) = " << rho_0  << std::endl;
      std::cout << "  pres_0 (code) = " << pres_0 << std::endl;
      std::cout << "  total mass (code) = " << totmass << std::endl;
      std::cout << "  total volume (code) = " << totvol << std::endl;
      std::cout << "  total cooling rate (code) = " << totcoolrate << std::endl;
      std::cout << "  mu = " << pmbp->punit->mu() << std::endl;
      std::cout << "  metallicity = " << pti_turb->metallicity << std::endl;
      std::cout << "  temperature (c.g.s) = " << temp_cgs << std::endl;
      std::cout << "  temperature (code) = " << temp_0 << std::endl;
      std::cout << "  temperature floor for cooling (c.g.s) = " << T_floor_cool_cgs << std::endl;
      std::cout << "  temperature ceiling for cooling (c.g.s) = " << T_ceiling_cool_cgs << std::endl;
      std::cout << "  temperature ceiling implementation starts at time (code) = " << pti_turb->t_ceiling_start << std::endl;
      std::cout << "  cooling function (c.g.s) = " << SQR(x_h*mu)*ISMCoolFn(temp_cgs, T_floor_cool_cgs, T_ceiling_cool_cgs, metallicity) << std::endl;
    }
    // End print info


  }

  if (restart) return;

  auto &u0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  Real gm1 = eos.gamma - 1.0;
  // Set initial conditions
  par_for("pgen_turb", DevExeSpace(), 0,nmb1,0,n3m1,0,n2m1,0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {

    u0(m,IDN,k,j,i) = rho_0;
    if(sigma_perturb > FLT_MIN){
      int64_t seed_perturb_loc = seed_perturb*nmb1-m;
      u0(m,IDN,k,j,i) *= (1.0 +  sigma_perturb * RanGaussian((int64_t*)(&seed_perturb_loc)));
    } 
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    if (eos.is_ideal) {
      u0(m,IEN,k,j,i) = pres_0/gm1;
    }
  });
  if ((pmbp->pmhd != nullptr)){
    auto &b0_ = pmbp->pmhd->b0;
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &w0 = pmbp->pmhd->w0;

    // First initialize fields to zero
    par_for("pgen_bondi_bfield", DevExeSpace(), 0,nmb1,0,n3m1,0,n2m1,0,n1m1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0_.x1f(m,k,j,i) = 0.0;
      if (i==n1m1) b0_.x1f(m,k,j,i+1) = 0.0;
      b0_.x2f(m,k,j,i) = 0.0;
      if (j==n2m1) b0_.x2f(m,k,j+1,i) = 0.0;
      b0_.x3f(m,k,j,i) = 0.0;
      if (k==n3m1) b0_.x3f(m,k+1,j,i) = 0.0;
      bcc0_(m,IBX,k,j,i) = 0.0;
      bcc0_(m,IBY,k,j,i) = 0.0;
      bcc0_(m,IBZ,k,j,i) = 0.0;
    });

    // First do a conservative to primitive conversion - this is needed for the turbulent magnetic field initialization
    pmbp->pmhd->peos->ConsToPrim(u0, b0_, w0, bcc0_, false, 0, n1m1, 0, n2m1, 0, n3m1);

    // Add turbulent magnetic fields
    for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
      if (it->block_name.compare(0, 8, "mag_init") == 0) {
        if (global_variable::my_rank == 0) std::cout << "Now adding turbulent B fields" << std::endl;
        MagInit *pmaginit;
        pmaginit = new MagInit(pmbp, pin);
        if (global_variable::my_rank == 0) std::cout << "New MagIni created." << std::endl;
        pmaginit->InitializeAVecModes(1);
        if (global_variable::my_rank == 0) std::cout << "Modes initialized" << std::endl;
        pmaginit->InitMagField(1);
        if (global_variable::my_rank == 0) std::cout << "Toroidal/Poloidal/Turbulent B fields added." << std::endl;
        delete pmaginit;
      }
    } // end turb_mhd block
    // Add uniform magnetic field
    Real b_ini = pin->GetOrAddReal("problem","b_ini_uni",0.0);
    Real plasma_beta = pin->GetOrAddReal("problem","plasma_beta_uni",0.0);
    int dir_b_ini = pin->GetOrAddInteger("problem","dir_b_ini",2); // default is z-direction
    if(b_ini!=0.0 || plasma_beta >0.0){ // For adding uniform magnetic field
      if(plasma_beta>0.0){
        b_ini = sqrt(2.0*pres_0/plasma_beta);
      }
      par_for("pgen_bfield", DevExeSpace(),0,nmb1,0,n3m1,0,n2m1,0,n1m1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        if(dir_b_ini==0) {
          b0_.x1f(m,k,j,i) = b_ini;
          if (i==ie) b0_.x1f(m,k,j,i+1) = b_ini;
        }
        else if(dir_b_ini==1){
          b0_.x2f(m,k,j,i) = b_ini;
          if (j==je) b0_.x2f(m,k,j+1,i) = b_ini;
        }
        else if(dir_b_ini==2){
          b0_.x3f(m,k,j,i) = b_ini;
          if (k==ke) b0_.x3f(m,k+1,j,i) = b_ini;
        }
        if (eos.is_ideal) {
          u0(m,IEN,k,j,i) += 0.5*b_ini*b_ini;
        }
      });
      if (global_variable::my_rank == 0) std::cout << "Uniform B fields added." << std::endl;
    } // end uniform magnetic field

    pmbp->pmhd->peos->ConsToPrim(u0, b0_, w0, bcc0_, false, 0, n1m1, 0, n2m1, 0, n3m1);
  }
  // Initialise the turbulent initial velocities module
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 9, "turb_init") == 0) {
      TurbulenceInit *pturb_init;
      pturb_init = new TurbulenceInit(it->block_name,pmbp, pin);
      pturb_init->InitializeModes(1);
      pturb_init->AddForcing(1);
      delete pturb_init;
    }
  }
  // Initialise the turbulent initial density fluctuations module
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 9, "dens_init") == 0) {
      if (global_variable::my_rank == 0) std::cout << "Now adding turbulent density fluctuations" << std::endl;
      DensFlucInit *pdensfluc;
      pdensfluc = new DensFlucInit(pmbp, pin);
      if (global_variable::my_rank == 0) std::cout << "New DensFlucInit created." << std::endl;
      pdensfluc->InitializeLogDensFlucModes(1);
      if (global_variable::my_rank == 0) std::cout << "Modes initialized" << std::endl;
      pdensfluc->AddDensFluc(1);
      if (global_variable::my_rank == 0) std::cout << "Turbulent density fluctuations added." << std::endl;
      delete pdensfluc;

      auto &size = pmbp->pmb->mb_size;
      Real m0=0., m1=0., m2=0.;

      if (global_variable::my_rank == 0) std::cout << "Now calculating the amplitude of density perturbations" << std::endl;
      // Now calculate the amplitude of density fluctuations and check if they match input parameters
      Kokkos::parallel_reduce("PgenDensFluc", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &sum_vol, Real &sum_logdens, Real &sum_logdens_sq) {
        // compute n,k,j,i indices of thread
        int m = (idx)/nkji;
        int k = (idx - m*nkji)/nji;
        int j = (idx - m*nkji - k*nji)/nx1;
        int i = (idx - m*nkji - k*nji - j*nx1) + is;
        k += ks;
        j += js;
        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        Real dvol = dx1*dx2*dx3;
        Real logdens_fluc = log(u0(m,IDN,k,j,i)/rho_0);
        sum_vol += dvol;
        sum_logdens += dvol*logdens_fluc;
        sum_logdens_sq += dvol*SQR(logdens_fluc);
      }, Kokkos::Sum<Real>(m0), Kokkos::Sum<Real>(m1), Kokkos::Sum<Real>(m2));
  #if MPI_PARALLEL_ENABLED
      Real m_sum[3] = {m0,m1,m2};
      Real gm_sum[3];
      MPI_Allreduce(m_sum, gm_sum, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      m0 = gm_sum[0];
      m1 = gm_sum[1];
      m2 = gm_sum[2];
  #endif
      Real sigma_s = sqrt(m2/m0 - SQR(m1/m0));
      if (global_variable::my_rank == 0) std::cout << "sigma_s = " << sigma_s << std::endl;
    }
  } // end turb_densfluc block
// 
  return;
}

namespace {
//----------------------------------------------------------------------------------------
// ! \fn UserTimeStep
// ! \brief Sets the time step for cooling, and turbulence updates
// ----------------------------------------------------------------------------------------

void UserTimeStep(Mesh *pm){
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  auto pmbp = pm->pmb_pack;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real dtnew_loc = static_cast<Real>(std::numeric_limits<float>::max());
  Real time = pm->time;

  auto &w0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  EOS_Data &eos = (pmbp->pmhd != nullptr) ?
                pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  
  Real gamma = eos.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = pti_turb->hrate_euv;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_unit = pmbp->punit->density_cgs()/pmbp->punit->mu()
                /pmbp->punit->atomic_mass_unit_cgs;
  Real mu = pmbp->punit->mu();
  Real x_h = pti_turb->x_h;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()
                      /n_unit;

  Real T_floor_cool_cgs = pti_turb->T_floor_cool*temp_unit;
  Real T_ceiling_cool_cgs = -1.0;
  if(time > pti_turb->t_ceiling_start) T_ceiling_cool_cgs = pti_turb->T_ceiling_cool*temp_unit;
  Real metallicity = pti_turb->metallicity;

  // For applying density ceiling
  bool use_dens_ceiling = pti_turb->use_dens_ceiling;
  Real dens_ceiling = pti_turb->dens_ceiling;

  // find smallest (e/cooling_rate) in each cell
  Kokkos::parallel_reduce("srcterms_cooling_newdt",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    // temperature in cgs unit
    Real temp = temp_unit*w0(m,IEN,k,j,i)/w0(m,IDN,k,j,i)*gm1;
    Real eint = w0(m,IEN,k,j,i);

    Real lambda_cooling = SQR(x_h*mu)*ISMCoolFn(temp, T_floor_cool_cgs, T_ceiling_cool_cgs, metallicity)/cooling_unit;
    Real gamma_heating = heating_rate/heating_unit;

    // add a tiny number
    // Add the weighting function here - maybe have a boolean flag to switch it on
    Real cooling_heating = FLT_MIN + fabs(w0(m,IDN,k,j,i) *
                            (w0(m,IDN,k,j,i) * lambda_cooling - gamma_heating));

    if(use_dens_ceiling && (w0(m,IDN,k,j,i) > dens_ceiling)) cooling_heating = FLT_MIN;

    min_dt = fmin((eint/cooling_heating), min_dt);
  }, Kokkos::Min<Real>(dtnew_loc));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &dtnew_loc, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif 
  pm->pgen->dtnew = dtnew_loc;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddUserSrcs()
//! \brief Add User Source Terms
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddUserSrcs(Mesh *pm, const Real bdt) {

  MeshBlockPack *pmbp = pm->pmb_pack;
  DvceArray5D<Real> &u0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  const DvceArray5D<Real> &w0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  const EOS_Data &eos = (pmbp->pmhd != nullptr) ?
                pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  if (pti_turb->cool_subc) {
    AddISMCooling(pm,bdt,u0,w0,eos);
  }
  if (pti_turb->use_equ_heating){
    AddEquHeating(pm,bdt,u0,w0,eos);
  }
  if (pti_turb->use_temp_ceiling) ApplyTempCeiling(pm,bdt,u0,w0,eos);
  return;
}




// ----------------------------------------------------------------------------------------
// ! \fn void SourceTerms::AddISMCooling()
// ! \brief Add explict ISM cooling and heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddISMCooling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &size = pm->pmb_pack->pmb->mb_size;
  Real beta = bdt/pm->dt;
  Real time = pm->time;
  Real cfl_no = pm->cfl_no;
  auto &eos = eos_data;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  Real heating_rate = pti_turb->hrate_euv;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_unit = pmbp->punit->density_cgs()/pmbp->punit->mu()
                /pmbp->punit->atomic_mass_unit_cgs;
  Real mu = pmbp->punit->mu();
  Real x_h = pti_turb->x_h;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()
                      /n_unit/n_unit;
  Real heating_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()/n_unit;

  // For applying density ceiling
  bool use_dens_ceiling = pti_turb->use_dens_ceiling;
  Real dens_ceiling = pti_turb->dens_ceiling;

  Real T_floor_cool_cgs = pti_turb->T_floor_cool*temp_unit;
  Real T_ceiling_cool_cgs = -1.0;
  if(time > pti_turb->t_ceiling_start) T_ceiling_cool_cgs = pti_turb->T_ceiling_cool*temp_unit;
  Real metallicity = pti_turb->metallicity;
  Real tot_vol=0.0;
  Real totcoolrate = 0.0;

  int nsubcycle=0, nsubcycle_count=0;
  Kokkos::parallel_reduce("cooling", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum0, int &sum1, Real &totvol_, Real &totcoolrate_) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dens = w0(m,IDN,k,j,i);
    Real temp = w0(m,IEN,k,j,i)/dens*gm1;
    Real eint = w0(m,IEN,k,j,i);

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    Real gamma_heating = heating_rate/heating_unit;
    bool sub_cycling = true;
    bool sub_cycling_used = false;
    Real bdt_now = 0.0;
    Real etot_old = u0(m,IEN,k,j,i);
    Real etot_new = etot_old;
    while (sub_cycling) {
      Real lambda_cooling = SQR(x_h*mu)*ISMCoolFn(temp*temp_unit, T_floor_cool_cgs, T_ceiling_cool_cgs, metallicity)/cooling_unit;
      Real cooling_heating =  dens * (dens * lambda_cooling - gamma_heating);

      // Do not cool if density is above the density ceiling value
      if(use_dens_ceiling && (w0(m,IDN,k,j,i) > dens_ceiling)) cooling_heating = FLT_MIN;
      
      Real dt_cool = (eint/(FLT_MIN + fabs(cooling_heating)));
      Real bdt_cool = beta*cfl_no*dt_cool;
      if (bdt_now+bdt_cool<bdt) {
        u0(m,IEN,k,j,i) -= bdt_cool * cooling_heating;

        // compute new temperature and internal energy
        
        // load single state conserved variables
        HydCons1D u;
        u.d  = u0(m,IDN,k,j,i);
        u.mx = u0(m,IM1,k,j,i);
        u.my = u0(m,IM2,k,j,i);
        u.mz = u0(m,IM3,k,j,i);
        u.e  = u0(m,IEN,k,j,i);

        // call c2p function
        // (inline function in ideal_c2p_hyd.hpp file)
        HydPrim1D w;
        bool dfloor_used=false, efloor_used=false, tfloor_used=false;
        SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);
        dens = w.d;
        temp = gm1*w.e/w.d;
        eint = w.e;
        sub_cycling_used = true;
        sum1++;
      } else {
        u0(m,IEN,k,j,i) -= (bdt-bdt_now) * cooling_heating;
        sub_cycling = false;
      }
      bdt_now += bdt_cool;
    }
    etot_new = u0(m,IEN,k,j,i);
    totvol_ += vol;
    totcoolrate_ += vol*fabs(etot_old-etot_new)/(bdt+1e-20);
    if (sub_cycling_used) {
      sum0++;
    }
  }, Kokkos::Sum<int>(nsubcycle), 
     Kokkos::Sum<int>(nsubcycle_count),
     Kokkos::Sum<Real>(tot_vol),
     Kokkos::Sum<Real>(totcoolrate));
#if MPI_PARALLEL_ENABLED
  int* pnsubcycle = &(nsubcycle);
  int* pnsubcycle_count = &(nsubcycle_count);
  MPI_Allreduce(MPI_IN_PLACE, pnsubcycle, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pnsubcycle_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tot_vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &totcoolrate, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  pti_turb->tot_coolrate = totcoolrate;
  if (global_variable::my_rank == 0) {
    if (pti_turb->ndiag>0 && pm->ncycle % pti_turb->ndiag == 0) {
      if (nsubcycle>0 || nsubcycle_count >0) {
        std::cout << " nsubcycle_cell=" << nsubcycle << std::endl
                  << " nsubcycle_count=" << nsubcycle_count << std::endl;
      }
      std::cout << " Subcycling cooling implementation is active." << std::endl;
    }
  }
  return;
}


// ----------------------------------------------------------------------------------------
// ! \fn void SourceTerms::ApplyTempCeiling()
// ! \brief Set a temperature ceiling as specified in the input file.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void ApplyTempCeiling(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  auto pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real beta = bdt/pm->dt;
  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;
  int nceiling_count=0;
  Real T_ceiling_ = pti_turb->T_ceiling;
  Kokkos::parallel_reduce("Temp_ceiling", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real dens=1.0, temp = 1.0, eint = 1.0;
    Real eint_ceiling = 1.0; Real deint = 0.0;
    dens = w0(m,IDN,k,j,i);
    temp = w0(m,IEN,k,j,i)/dens*gm1;
    eint = w0(m,IEN,k,j,i);
    if(temp>T_ceiling_) {
      eint_ceiling = T_ceiling_ * dens/gm1;
      deint = eint_ceiling - eint;
      sum ++;
      // Now update internal energy
      u0(m,IEN,k,j,i) = u0(m,IEN,k,j,i) + beta * deint;
    }
  }, Kokkos::Sum<int>(nceiling_count));
  // Can print a statement here if temperature ceiling is applied
  #if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &nceiling_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    if (pti_turb->ndiag>0 && pm->ncycle % pti_turb->ndiag == 0) {
      if (nceiling_count >0) {
        std::cout << " Temperature ceiling implementation is active." << std::endl;
        std::cout << " nceiling_count=" << nceiling_count << std::endl;
      }
    }
  }
  return;
}


// ----------------------------------------------------------------------------------------
// ! \fn void SourceTerms::AddEquHeating()
// ! \brief Add heating source terms in the energy equations.
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars

void AddEquHeating(Mesh *pm, const Real bdt, DvceArray5D<Real> &u0,
                   const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real gm1 = eos_data.gamma - 1.0;
  Real time = pm->time;
  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_unit = pmbp->punit->density_cgs()/pmbp->punit->mu()
                /pmbp->punit->atomic_mass_unit_cgs;
  Real mu = pmbp->punit->mu();
  Real x_h = pti_turb->x_h;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()
                      /n_unit/n_unit;
  Real T_floor_cool_cgs = pti_turb->T_floor_cool*temp_unit;
  Real T_ceiling_cool_cgs = -1.0;
  if(time > pti_turb->t_ceiling_start) T_ceiling_cool_cgs = pti_turb->T_ceiling_cool*temp_unit;
  Real metallicity = pti_turb->metallicity;
  Real beta = bdt/pm->dt; // Ratio between RK time step and code time step

  bool const_ratio_equ_heating = pti_turb->const_ratio_equ_heating; // Sets equ_heating to cooling*(1-hc_ratio) after each update
  bool const_ratio_equ_heating_L0 = pti_turb->const_ratio_equ_heating_L0; // Sets equ_heating to cooling-heating_sn after each update
  Real hc_ratio = pti_turb->hc_ratio; // ratio between heating due to equ_heating and net cooling at t=0
  Real q_max = pti_turb->max_equ_heat_rate; // Maximum heating rate per volume
  int weight = pti_turb->heat_weight;

  Real tot_vol = 0.0;
  Real tot_mass = 0.0;
  Real tot_weight = 0.0;
  Real totcoolrate = 0.0;
  Real totturbheatrate = 0.0;
  auto force_ = pmbp->pturb->force;

  // find smallest (e/cooling_rate) in each cell
  Kokkos::parallel_reduce("Equ_heat", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &totvol_, Real &totm_, Real &totweight_, Real &totcoolrate_, Real &totturbheatrate_) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    // temperature in cgs unit
    Real dens = w0(m,IDN,k,j,i);
    Real temp = w0(m,IEN,k,j,i)/dens*gm1;

    //use temperature in cgs unit
    Real lambda_cooling = SQR(x_h*mu)*ISMCoolFn(temp*temp_unit, T_floor_cool_cgs, T_ceiling_cool_cgs, metallicity)/cooling_unit;
    totcoolrate_ += vol * dens * dens * lambda_cooling;

    Real a1 = force_(m,0,k,j,i);
    Real a2 = force_(m,1,k,j,i);
    Real a3 = force_(m,2,k,j,i);

    totvol_ += vol;
    totm_ += dens*vol;
    totweight_ += pow_dev(dens, weight)*vol;
    Real turb_heating = vol*dens*(a1*w0(m,IVX,k,j,i) + a2*w0(m,IVY,k,j,i) + a3*w0(m,IVZ,k,j,i));
    totturbheatrate_ += turb_heating;
  }, Kokkos::Sum<Real>(tot_vol),
     Kokkos::Sum<Real>(tot_mass),
     Kokkos::Sum<Real>(tot_weight),
     Kokkos::Sum<Real>(totcoolrate),
     Kokkos::Sum<Real>(totturbheatrate));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &tot_vol, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tot_mass, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tot_weight, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &totcoolrate, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &totturbheatrate, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  pti_turb->tot_mass = tot_mass;
  pti_turb->tot_turbheatrate = totturbheatrate;

  Real heat_fraction = pti_turb->heat_fraction;
  Real q_compensate_per_wt = heat_fraction * (totcoolrate - totturbheatrate) / tot_weight;
  // total cooling normalised by the weight
  q_compensate_per_wt = fmax(q_compensate_per_wt, 0.0);
  Real q_equ_tot = 0.0; // to calculate total equilibrium heating rate

  // For calculating the cooling rate at t=0
  Real pres_0 = pti_turb->pres0;
  Real rho_0  = pti_turb->rho0;
  Real n_0 = rho_0*pmbp->punit->density_cgs()/(pmbp->punit->mu()*pmbp->punit->atomic_mass_unit_cgs);
  Real temp_cgs_0 = pres_0*pmbp->punit->pressure_cgs()/(n_0*pmbp->punit->k_boltzmann_cgs);
  
  
  Kokkos::parallel_reduce("add_and_sum_qheat", Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &q_equ_tot_) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    Real q_comp_per_wt = q_compensate_per_wt;
    Real dens = w0(m,IDN,k,j,i);
    Real q_heating = q_comp_per_wt * pow_dev(dens, weight);
    // Make sure this heating rate is less than the maximum heating rate/volume
    if(const_ratio_equ_heating) q_heating = fmax(fmin( (1.0-hc_ratio) * q_heating, q_max), 0.0);
    else if(const_ratio_equ_heating_L0){
      Real q_cooling0 = rho_0*rho_0*SQR(x_h*mu)*ISMCoolFn(temp_cgs_0, T_floor_cool_cgs, T_ceiling_cool_cgs, metallicity)/cooling_unit;
      Real q_cooling0_per_wt = q_cooling0/pow_dev(rho_0, weight);
      q_comp_per_wt = q_comp_per_wt-hc_ratio*q_cooling0_per_wt;
      q_heating = q_comp_per_wt * pow_dev(dens, weight);
      q_heating = fmax(fmin(q_heating, q_max), 0.0);
    } 
    u0(m,IEN,k,j,i) += bdt * q_heating;
    q_equ_tot_ += bdt * q_heating * vol;
  }, Kokkos::Sum<Real>(q_equ_tot));

  #if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &q_equ_tot, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  #endif


  pti_turb->tot_equheatrate = q_equ_tot/bdt; // Now store this in the global variable

  if (pm->ncycle % pti_turb->ndiag == 0 && pm->ncycle % pti_turb->heat_cycle == 0 && global_variable::my_rank == 0 && beta > 0.99){
    std::cout << "tot_cool_rate =  " << totcoolrate    << std::endl;
    std::cout << "equ_heat_rate =  " << q_equ_tot/bdt << std::endl;
  }

  return;
}
// ----------------------------------------------------------------------------------------
// ! \fn void UserWorkInLoop()
// ! \brief Function called in hydro or mhd tasks in "after_timeintegrator" stage
void UserWorkInLoop(Mesh *pm) {
  if (pti_turb->ndiag>0 && pm->ncycle % pti_turb->ndiag == 0) {
    Diagnostic(pm);
  }
}

//----------------------------------------------------------------------------------------
// ! \fn void UserWorkInLoop::Diagnostic()
// ! \brief Compute volume and mass averages and print them to the screen
void Diagnostic(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real time = pm->time;

  EOS_Data &eos_data = (pmbp->pmhd != nullptr) ?
                  pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;

  auto &w0 = (pmbp->pmhd != nullptr) ? pmbp->pmhd->w0 : pmbp->phydro->w0;

  bool is_mhd = (pmbp->pmhd != nullptr);
  DvceArray5D<Real> bcc0; // Initialize as an empty view
  if(is_mhd) bcc0 = pmbp->pmhd->bcc0;

  // For calculating cooling rate

  Real temp_unit = pmbp->punit->temperature_cgs();
  Real n_unit = pmbp->punit->density_cgs()/pmbp->punit->mu()
                /pmbp->punit->atomic_mass_unit_cgs;
  Real mu = pmbp->punit->mu();
  Real x_h = pti_turb->x_h;
  Real cooling_unit = pmbp->punit->pressure_cgs()/pmbp->punit->time_cgs()
                      /n_unit/n_unit;
  Real T_floor_cool_cgs = pti_turb->T_floor_cool*temp_unit;
  Real T_ceiling_cool_cgs = -1.0;
  if(time > pti_turb->t_ceiling_start) T_ceiling_cool_cgs = pti_turb->T_ceiling_cool*temp_unit;
  Real metallicity = pti_turb->metallicity;

  Real gamma = eos_data.gamma;
  Real gm1 = gamma - 1.0;

  Real dtnew = std::numeric_limits<Real>::max();

  Real min_dens = std::numeric_limits<Real>::max();
  Real min_vtot = std::numeric_limits<Real>::max();
  Real min_temp = std::numeric_limits<Real>::max();
  Real min_eint = std::numeric_limits<Real>::max();
  Real max_dens = std::numeric_limits<Real>::min();
  Real max_vtot = std::numeric_limits<Real>::min();
  Real max_temp = std::numeric_limits<Real>::min();
  Real max_eint = std::numeric_limits<Real>::min();
  Real tot_mass = 0.;
  Real tot_eint = 0.;
  Real totcoolrate = 0.0;
  Real totturbheatrate = 0.0;
  Real tot_vsq = 0.0;
  Real tot_bsq = 0.0;
  Real tot_vol = 0.0;
  Real tot_mach_sq = 0.0;
  Real tot_machalfven_sq = 0.0;

  auto force_ = pmbp->pturb->force;

  // find smallest (e/cooling_rate) in each cell
  Kokkos::parallel_reduce("diagnostic", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt, Real &min_d, Real &min_v, Real &min_t,
  Real &min_e, Real &max_d, Real &max_v, Real &max_t, Real &max_e, Real &tot_m,
  Real &tot_e, Real &totcoolrate_, Real &totturbheatrate_, Real &totvol_, 
  Real &totvsq_, Real &totbsq_, Real &tot_mach_sq_, Real &tot_machalfven_sq_) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real dx = fmin(fmin(size.d_view(m).dx1,size.d_view(m).dx2),size.d_view(m).dx3);
    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    // temperature in cgs unit
    Real dens = w0(m,IDN,k,j,i);
    Real temp = w0(m,IEN,k,j,i)/dens*gm1;
    Real eint = w0(m,IEN,k,j,i);

    Real vtot = sqrt(SQR(w0(m,IVX,k,j,i))+SQR(w0(m,IVY,k,j,i))+SQR(w0(m,IVZ,k,j,i)));

    min_dt = fmin(dx/sqrt(gamma*temp), min_dt);
    min_d = fmin(dens, min_d);
    min_v = fmin(vtot,min_v);
    min_t = fmin(temp, min_t);
    min_e = fmin(eint, min_e);
    max_d = fmax(dens, max_d);
    max_v = fmax(vtot,max_v);
    max_t = fmax(temp, max_t);
    max_e = fmax(eint, max_e);
    tot_m += dens*vol;
    tot_e += w0(m,IEN,k,j,i)*vol;

    //use temperature in cgs unit
    Real lambda_cooling = SQR(x_h*mu)*ISMCoolFn(temp*temp_unit, T_floor_cool_cgs, T_ceiling_cool_cgs, metallicity)/cooling_unit;
    totcoolrate_ += vol * dens * dens * lambda_cooling;

    Real a1 = force_(m,0,k,j,i);
    Real a2 = force_(m,1,k,j,i);
    Real a3 = force_(m,2,k,j,i);

    Real turb_heating = vol*dens*(a1*w0(m,IVX,k,j,i) + a2*w0(m,IVY,k,j,i) + a3*w0(m,IVZ,k,j,i));
    totturbheatrate_ += turb_heating;
    totvol_ += vol;
    totvsq_ += vol*SQR(vtot);
    tot_mach_sq_ += vol*SQR(vtot)/(gamma*eint*gm1/dens);

    if(is_mhd){
      Real btot = sqrt(SQR(bcc0(m,IBX,k,j,i))+SQR(bcc0(m,IBY,k,j,i))+SQR(bcc0(m,IBZ,k,j,i)));
      totbsq_ += vol*SQR(btot);
      tot_machalfven_sq_ += vol*SQR(vtot)*dens/SQR(btot);
    }

  }, Kokkos::Min<Real>(dtnew),
     Kokkos::Min<Real>(min_dens),
     Kokkos::Min<Real>(min_vtot),
     Kokkos::Min<Real>(min_temp),
     Kokkos::Min<Real>(min_eint),
     Kokkos::Max<Real>(max_dens),
     Kokkos::Max<Real>(max_vtot),
     Kokkos::Max<Real>(max_temp),
     Kokkos::Max<Real>(max_eint),
     Kokkos::Sum<Real>(tot_mass), 
     Kokkos::Sum<Real>(tot_eint),
     Kokkos::Sum<Real>(totcoolrate),
     Kokkos::Sum<Real>(totturbheatrate),
     Kokkos::Sum<Real>(tot_vol),
     Kokkos::Sum<Real>(tot_vsq),
     Kokkos::Sum<Real>(tot_bsq),
     Kokkos::Sum<Real>(tot_mach_sq),
     Kokkos::Sum<Real>(tot_machalfven_sq));
  Real dt_hyd  = (pmbp->pmhd != nullptr) ? pmbp->pmhd->dtnew : pmbp->phydro->dtnew;
  auto *pcond = (is_mhd) ? pmbp->pmhd->pcond : pmbp->phydro->pcond;
  Real dt_cond = (pcond != nullptr) ? pcond->dtnew : FLT_MAX;
  Real dt_src  = FLT_MAX;
  auto *psrc = (is_mhd) ? pmbp->pmhd->psrc : pmbp->phydro->psrc;
  if (psrc != nullptr) {
    dt_src  = psrc->dtnew;
  }
  Real dt_user  = FLT_MAX;
  if (pmbp->pmesh->pgen->user_dt) {
    dt_user = pmbp->pmesh->pgen->dtnew;
  }
#if MPI_PARALLEL_ENABLED
  Real m_min[9] = {dtnew,min_dens,min_vtot,min_temp,min_eint,dt_hyd,dt_cond,dt_src,dt_user};
  Real m_max[4] = {max_dens,max_vtot,max_temp,max_eint};
  Real gm_min[9];
  Real gm_max[4];
  Real loc_sum[9] = {tot_mass,tot_eint,totcoolrate,totturbheatrate,tot_vol,tot_vsq,tot_bsq,tot_mach_sq,tot_machalfven_sq};
  Real glob_sum[9];
  MPI_Allreduce(m_min, gm_min, 9, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(m_max, gm_max, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(loc_sum, glob_sum, 9, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  dtnew = gm_min[0];
  min_dens = gm_min[1];
  min_vtot = gm_min[2];
  min_temp = gm_min[3];
  min_eint = gm_min[4];
  dt_hyd   = gm_min[5];
  dt_cond  = gm_min[6];
  dt_src   = gm_min[7];
  dt_user  = gm_min[8];
  max_dens = gm_max[0];
  max_vtot = gm_max[1];
  max_temp = gm_max[2];
  max_eint = gm_max[3];

  tot_mass = glob_sum[0];
  tot_eint = glob_sum[1];
  totcoolrate = glob_sum[2];
  totturbheatrate = glob_sum[3];
  tot_vol = glob_sum[4];
  tot_vsq = glob_sum[5];
  tot_bsq = glob_sum[6];
  tot_mach_sq = glob_sum[7];
  tot_machalfven_sq = glob_sum[8];
#endif
  pti_turb->tot_mass = tot_mass;
  pti_turb->tot_coolrate = totcoolrate;
  pti_turb->tot_turbheatrate = totturbheatrate;
  Real vrms = sqrt(tot_vsq/tot_vol);
  Real brms = sqrt(tot_bsq/tot_vol);
  Real plasma_beta = tot_eint*gm1*2.0/tot_bsq;
  Real mach_rms = sqrt(tot_mach_sq/tot_vol);
  Real machalfven_rms = sqrt(tot_machalfven_sq/tot_vol);
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(16)  
              << " min_d=" << min_dens*pmbp->punit->density_cgs()     << " max_d=" << max_dens*pmbp->punit->density_cgs()     << std::endl
              << " min_T=" << min_temp*pmbp->punit->temperature_cgs() << " max_T=" << max_temp*pmbp->punit->temperature_cgs() << std::endl
              << " min_v_cgs=" << min_vtot*pmbp->punit->velocity_cgs()    << " max_v_cgs=" << max_vtot*pmbp->punit->velocity_cgs()    << std::endl
              << " min_v=" << min_vtot    << " max_v=" << max_vtot    << std::endl
              << " min_e=" << min_eint << " max_e=" << max_eint << std::endl
              << " tot_m=" << tot_mass << " tot_e=" << tot_eint << std::endl
              << " dt_temp=" << dtnew   << " dt_hyd=" << dt_hyd << std::endl
              << " dt_cond=" << dt_cond << " dt_src=" << dt_src << std::endl
              << " dt_user=" << dt_user << " totcoolrate=" << totcoolrate << std::endl
              << " totturbheatrate=" << totturbheatrate << " v_rms=" << vrms << std::endl
              << " b_rms=" << brms << " plasma_beta=" << plasma_beta << std::endl
              << " mach_rms=" << mach_rms << " machalfven_rms=" << machalfven_rms << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn UserHistOutput
//
//! @brief Computes and stores user-defined history data for turbulence diagnostics.
//!
//! This function updates the provided \p pdata structure by calculating several
//! statistical properties of the flow and, optionally, magnetic fields. The results are
//! organized into arrays that track integrated quantities of hot and cold gas based on
//! specified temperature thresholds.
// 
//! @param[in,out] pdata Pointer to a HistoryData structure where results will be stored.
//! @param[in]     pm    Pointer to a Mesh object representing the domain over which
//!                      data are collected.
//! 
//! The function begins by allocating a fixed number of history variables and checking
//! against the maximum permitted value. It then labels each variable, indicating its
//! physical meaning (e.g., mass, volume, velocity fluctuations, etc.). The routine
//! distinguishes between hot and cold gas regions using predefined temperature cutoffs,
//! thus storing separate integrals of relevant physical quantities for different thermal
//! phases.
// 
//! If magnetohydrodynamics (MHD) is enabled, additional computations are performed for
//! magnetic field magnitudes and Alfvenic quantities (e.g., Alfven Mach number and plasma
//! beta). The resulting data are aggregated via a parallel reduction over all mesh
//! blocks, ensuring that final integrated values reflect the entire computational domain.
//! 
//! The last few elements in the history arrays store cooling and heating rates computed
//! externally. These values are only meaningful on the root rank, where the partial
//! results are finally consolidated.
// ----------------------------------------------------------------------------------------
void UserHistOutput(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  bool is_mhd = (pmbp->pmhd != nullptr);
  const EOS_Data &eos = (is_mhd) ?
                  pmbp->pmhd->peos->eos_data : pmbp->phydro->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  Real gamma = eos.gamma;
  // capture class variables for kernel
  auto &w0 = (is_mhd) ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  DvceArray5D<Real> bcc0; // Initialize as an empty view
  if(is_mhd) bcc0 = pmbp->pmhd->bcc0;

  auto &size = pmbp->pmb->mb_size;
  int nsum0 = 18; // All hydro variables to be summed
  int nsum1 = (is_mhd) ? 18 : 0; // All mhd variables to be summed
  int nsum = nsum0 + nsum1;
  int nother_hist = 3; // history variables that are not global sums
  int nuser = nsum+nother_hist;
  pdata->nhist = nuser;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  if ((nsum0 > NREDUCTION_VARIABLES) || (nsum1 > NREDUCTION_VARIABLES)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified nsum larger than"
              << " NREDUCTION_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }

  int nhi=0;
  pdata->label[nhi++] = "Vol_tot "; //0
  pdata->label[nhi++] = "Vol_hot ";
  pdata->label[nhi++] = "Vol_cold ";
  pdata->label[nhi++] = "Mtot ";
  pdata->label[nhi++] = "Mtot_hot ";
  pdata->label[nhi++] = "Mtot_cold "; //5
  pdata->label[nhi++] = "V_sq_vw ";
  pdata->label[nhi++] = "V_sq_hot_vw ";
  pdata->label[nhi++] = "V_sq_cold_vw ";
  pdata->label[nhi++] = "V_sq_mw ";
  pdata->label[nhi++] = "V_sq_hot_mw "; //10
  pdata->label[nhi++] = "V_sq_cold_mw ";
  pdata->label[nhi++] = "Mach_sq_vw ";
  pdata->label[nhi++] = "Mach_sq_hot_vw ";
  pdata->label[nhi++] = "Mach_sq_cold_vw ";
  pdata->label[nhi++] = "Mach_sq_mw "; //15
  pdata->label[nhi++] = "Mach_sq_hot_mw ";
  pdata->label[nhi++] = "Mach_sq_cold_mw ";
  if(is_mhd){
    pdata->label[nhi++] = "M_A_sq_vw ";
    pdata->label[nhi++] = "M_A_sq_hot_vw ";
    pdata->label[nhi++] = "M_A_sq_cold_vw "; //20
    pdata->label[nhi++] = "M_A_sq_mw ";
    pdata->label[nhi++] = "M_A_sq_hot_mw ";
    pdata->label[nhi++] = "M_A_sq_cold_mw ";
    pdata->label[nhi++] = "B_sq_vw ";
    pdata->label[nhi++] = "B_sq_hot_vw "; //25
    pdata->label[nhi++] = "B_sq_cold_vw ";
    pdata->label[nhi++] = "B_sq_mw ";
    pdata->label[nhi++] = "B_sq_hot_mw ";
    pdata->label[nhi++] = "B_sq_cold_mw ";
    pdata->label[nhi++] = "Plasma_beta_vw "; //30
    pdata->label[nhi++] = "Plasma_beta_hot_vw ";
    pdata->label[nhi++] = "Plasma_beta_cold_vw ";
    pdata->label[nhi++] = "Plasma_beta_mw ";
    pdata->label[nhi++] = "Plasma_beta_hot_mw ";
    pdata->label[nhi++] = "Plasma_beta_cold_mw "; //35
  }
  pdata->label[nhi++] = "Coolr ";
  pdata->label[nhi++] = "Edot_turb ";
  pdata->label[nhi++] = "Edot_equ "; //38 - mhd, 20 - hydro


  // loop over all MeshBlocks in this pack
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb0, sum_this_mb1;
  // Initialize hdata arrays to 0
  for (int n=0; n<NREDUCTION_VARIABLES; ++n) {
    sum_this_mb0.the_array[n] = 0.0; 
    sum_this_mb1.the_array[n] = 0.0;
  }
  Real T_cold = pti_turb->T_cold;
  Real T_hot = pti_turb->T_hot;

  Kokkos::parallel_reduce("UserHistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum0, array_sum::GlobalSum &mb_sum1) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real dens=1.0, temp = 1.0, eint = 1.0;

    dens = w0(m,IDN,k,j,i);
    Real velx = w0(m,IVX,k,j,i);
    Real vely = w0(m,IVY,k,j,i);
    Real velz = w0(m,IVZ,k,j,i);

    temp = w0(m,IEN,k,j,i)/dens*gm1;
    eint = w0(m,IEN,k,j,i);

    array_sum::GlobalSum hvars0, hvars1;

    for (int index = 0; index < NREDUCTION_VARIABLES; index++) {
      hvars0.the_array[index] = 0.0;
      hvars1.the_array[index] = 0.0;
    }

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    // volume
    hvars0.the_array[0] += vol;
    // mass
    Real mass = dens*vol;
    hvars0.the_array[3] += mass;

    // velocity squared
    Real vtot_sq = SQR(velx)+SQR(vely)+SQR(velz);
    hvars0.the_array[6] += vtot_sq*vol; // vw
    hvars0.the_array[9] += vtot_sq*mass; // mw

    // Mach squared
    Real mach_sq = vtot_sq/(gamma*eint*gm1/dens);
    hvars0.the_array[12] += mach_sq*vol; // vw
    hvars0.the_array[15] += mach_sq*mass; // mw
    
    // Hot gas variables
    if (temp > T_hot) {
      // Hot gas volume
      hvars0.the_array[1] += vol;
      // Hot gas mass
      hvars0.the_array[4] += dens*vol; 
      // Hot gas velocity squared
      hvars0.the_array[7] += vtot_sq*vol; // vw
      hvars0.the_array[10] += vtot_sq*mass; // mw
      // Hot gas Mach squared
      hvars0.the_array[13] += mach_sq*vol; // vw
      hvars0.the_array[16] += mach_sq*mass; // mw
    }
    // Cold gas variables
    if (temp < T_cold) {
      // Cold gas volume
      hvars0.the_array[2] += vol;
      // Cold gas mass
      hvars0.the_array[5] += dens*vol;
      // Cold gas velocity squared
      hvars0.the_array[8] += vtot_sq*vol; // vw
      hvars0.the_array[11] += vtot_sq*mass; // mw
      // Cold gas Mach squared
      hvars0.the_array[14] += mach_sq*vol; // vw
      hvars0.the_array[17] += mach_sq*mass; // mw
    }
    // Zero out the rest of the variables
    for (int index=nsum0; index<NREDUCTION_VARIABLES; index++) hvars0.the_array[index] = 0.0;

    // Now if B-fields are present
    if(is_mhd){
      // Alfven Mach number squared
      Real bx = bcc0(m,IBX,k,j,i);
      Real by = bcc0(m,IBY,k,j,i);
      Real bz = bcc0(m,IBZ,k,j,i);
      Real btot_sq = SQR(bx)+SQR(by)+SQR(bz);
      btot_sq = fmax(btot_sq, 1.e-20);
      Real machalfven_sq = vtot_sq*dens/btot_sq; 
      // Alfven Mach squared
      hvars1.the_array[0] += machalfven_sq*vol; // vw
      hvars1.the_array[3] += machalfven_sq*mass; // mw
      // Magnetic field squared
      hvars1.the_array[6] += btot_sq*vol; // vw
      hvars1.the_array[9] += btot_sq*mass; // mw
      // Plasma beta
      hvars1.the_array[12] += eint*gm1*2.0/btot_sq*vol; // vw
      hvars1.the_array[15] += eint*gm1*2.0/btot_sq*mass; // mw
      // Hot gas variables
      if (temp > T_hot) {
        // Hot gas Alfven Mach squared
        hvars1.the_array[1] += machalfven_sq*vol; // vw
        hvars1.the_array[4] += machalfven_sq*mass; // mw
        // Hot gas magnetic field squared
        hvars1.the_array[7] += btot_sq*vol; // vw
        hvars1.the_array[10] += btot_sq*mass; // mw
        // Hot gas plasma beta
        hvars1.the_array[13] += eint*gm1*2.0/btot_sq*vol; // vw
        hvars1.the_array[16] += eint*gm1*2.0/btot_sq*mass; // mw
      }
      // Cold gas variables
      if (temp < T_cold) {
        // Cold gas Alfven Mach squared
        hvars1.the_array[2] += machalfven_sq*vol; // vw
        hvars1.the_array[5] += machalfven_sq*mass; // mw
        // Cold gas magnetic field squared
        hvars1.the_array[8] += btot_sq*vol; // vw
        hvars1.the_array[11] += btot_sq*mass; // mw
        // Cold gas plasma beta
        hvars1.the_array[14] += eint*gm1*2.0/btot_sq*vol; // vw
        hvars1.the_array[17] += eint*gm1*2.0/btot_sq*mass; // mw
      }
      else{
        for (int index=0; index<nsum1; index++) hvars1.the_array[index] = 0.0;
      }
      // Zero out the rest of the variables
      for (int index=nsum1; index<NREDUCTION_VARIABLES; index++) hvars1.the_array[index] = 0.0;
    }
    // sum into parallel reduce
    mb_sum0 += hvars0;
    mb_sum1 += hvars1;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb0),
     Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb1));

  // store data into hdata array
  // store data into hdata array
  for (int n=0; n<nsum; ++n) {
    pdata->hdata[n] = (n < nsum0) ? sum_this_mb0.the_array[n] : sum_this_mb1.the_array[n-nsum0];
  }

  if (global_variable::my_rank == 0){
    // Cooling rate, turbulent and equ heating rate are already calculated in the diagnostics function
    pdata->hdata[nsum]   = pti_turb->tot_coolrate;
    pdata->hdata[nsum+1] = pti_turb->tot_turbheatrate;
    if(pti_turb->use_equ_heating) pdata->hdata[nsum+2] = pti_turb->tot_equheatrate;
    else pdata->hdata[nsum+2] = 0.0;
  }
  else{
    for (int index=nsum; index<nuser; index++) pdata->hdata[index] = 0.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TurbTiInitWork()
//! \brief Performs final cleanup tasks for turbulence pgen.
//!
//! This function deallocates resources used for turbulence pgen.
//!
//! \param pin Reference to the ParameterInput object for configuration data.
//! \param pm  Reference to the Mesh object representing the simulation domain.
// ----------------------------------------------------------------------------------------
void TurbTiFinalWork(ParameterInput *, Mesh *) {
  delete pti_turb;
}

} //namespace