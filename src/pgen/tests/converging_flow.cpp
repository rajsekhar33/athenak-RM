//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file converging_flow.cpp
//! \brief Converging/diverging flow test for Lagrangian tracer particles.  Use with
//! evolution=kinematic, hydro/rsolver=advect, eos=isothermal.
//!
//! Sets a fixed linear velocity profile v1(x1) = -strain_rate*x1 (strain_rate>0 gives
//! flow converging on x1=0; strain_rate<0 gives flow diverging away from x1=0) on top
//! of a smooth density perturbation rho0(x1) = 1 + amplitude*cos(2*pi*x1/Lx). Density
//! evolves via the code's normal advective flux update; UserWorkInLoop resets the
//! momentum every cycle to rho*v1_prescribed(x1) using the just-updated density, so the
//! velocity field never self-advects away from the prescribed profile (decoupling it
//! from the pressureless-Burgers dynamics the plain advect solver would otherwise give).
//! ix1_bc/ox1_bc must be outflow (not periodic): a linear v(x1) profile is not itself
//! periodic, and outflow keeps the analytic solution below valid across the whole
//! domain since mass only moves inward (converging case) or outward through open
//! boundaries (diverging case), with no artificial wraparound source/sink.
//!
//! Because v1 is held fixed, continuity has an exact solution via characteristics
//! dx1/dt = -strain_rate*x1  =>  x1(t) = x1_0*exp(-strain_rate*t), and mass conservation
//! along each characteristic gives
//!   rho(x1,t) = rho0(x1*exp(strain_rate*t)) * exp(strain_rate*t)
//! -- i.e. the initial perturbation is advected inward (or outward) and compressed (or
//! stretched) in wavelength while its amplitude (and the background) grows (or decays)
//! by the same exp(strain_rate*t) factor. Correctly-implemented Lagrangian tracers
//! should reproduce this same profile in their number density, since they represent
//! fixed mass parcels responding to the identical velocity/flux field the gas obeys.

#include <cmath>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

namespace {
Real strain_rate_global = 0.0;
void ConvergingFlowWorkInLoop(Mesh *pm);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets up the converging/diverging flow test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "Converging-flow test requires a <hydro> block"
       << " (MHD not supported)." << std::endl;
    exit(EXIT_FAILURE);
  }

  Real strain_rate = pin->GetOrAddReal("problem", "strain_rate", 1.0);
  Real amplitude = pin->GetOrAddReal("problem", "amplitude", 0.2);
  strain_rate_global = strain_rate;
  user_work_in_loop_func = ConvergingFlowWorkInLoop;

  if (restart) return;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1;

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real kwave = 2.0*M_PI/x1size;

  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;

  par_for("pgen_converging_flow", DevExeSpace(), 0,(pmbp->nmb_thispack-1),
  ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real dens = 1.0 + amplitude*cos(kwave*x1v);
    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = -strain_rate*x1v*dens;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
  });

  auto &u0_ = pmbp->phydro->u0;
  if (pmbp->ppart != nullptr) {
    InitializeLagrangianParticles(pin, u0_);
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void ConvergingFlowWorkInLoop()
//! \brief Resets momentum to rho*v1_prescribed(x1) every cycle, using the density just
//! updated by the advective flux, so the velocity field never self-evolves away from
//! the prescribed linear profile.
void ConvergingFlowWorkInLoop(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1;

  Real strain_rate = strain_rate_global;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;

  par_for("converging_flow_workinloop", DevExeSpace(), 0,(pmbp->nmb_thispack-1),
  ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    u0(m,IM1,k,j,i) = -strain_rate*x1v*u0(m,IDN,k,j,i);
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
  });
  return;
}
} // namespace
