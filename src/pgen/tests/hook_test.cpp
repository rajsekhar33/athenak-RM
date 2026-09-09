//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hook_test.cpp
//! \brief minimal validation pgen for the UserWorkInLoop/user_dt hook mechanism.
//! Uniform hydro medium at rest; UserTimeStep provides a fixed raw timestep cap
//! (to which the global CFL factor is subsequently applied), and UserWorkInLoop
//! counts how many times it is called per rank. Both are exposed
//! via a custom history output so a completed run's .hst file can confirm the
//! hooks actually fired every cycle and the timestep was actually clamped.

#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {
struct pgen_hook_test {
  Real test_dt;
  int workinloop_calls;
};
pgen_hook_test *phook = nullptr;

void UserTimeStep(Mesh *pm);
void UserWorkInLoop(Mesh *pm);
void UserHistOutput(HistoryData *pdata, Mesh *pm);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::HookTest()
//! \brief Minimal problem for validating UserWorkInLoop/user_dt hooks.

void ProblemGenerator::HookTest(ParameterInput *pin, const bool restart) {
  phook = new pgen_hook_test();
  phook->test_dt = pin->GetOrAddReal("problem", "test_dt", 1.0e-3);
  phook->workinloop_calls = 0;

  user_time_step_func = UserTimeStep;
  user_work_in_loop_func = UserWorkInLoop;
  user_hist_func = UserHistOutput;

  if (restart) return;

  Real dens = pin->GetOrAddReal("problem", "dens", 1.0);
  Real pres = pin->GetOrAddReal("problem", "pres", 1.0);

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;

  if (pmbp->phydro != nullptr) {
    auto &u0 = pmbp->phydro->u0;
    Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
    par_for("hook_test_hydro", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = pres/gm1;
    });
  }

  if (pmbp->pmhd != nullptr) {
    auto &u0 = pmbp->pmhd->u0;
    auto &b0 = pmbp->pmhd->b0;
    Real gm1 = pmbp->pmhd->peos->eos_data.gamma - 1.0;
    par_for("hook_test_mhd", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m,IDN,k,j,i) = dens;
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      b0.x1f(m,k,j,i) = 0.0;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
      if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
      if (k==ke) {b0.x3f(m,k+1,j,i) = 0.0;}
      u0(m,IEN,k,j,i) = pres/gm1;
    });
  }

  return;
}

#if USER_PROBLEM_ENABLED
// Preserve the custom-pgen entry point for existing -D PROBLEM=tests/hook_test builds.
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  HookTest(pin, restart);
}
#endif

namespace {
//----------------------------------------------------------------------------------------
//! \fn void UserTimeStep()
//! \brief provides a raw timestep cap deliberately below the hydro/mhd limit;
//! Mesh::NewTimeStep applies the configured CFL factor to it.
void UserTimeStep(Mesh *pm) {
  pm->pgen->dtnew = phook->test_dt;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void UserWorkInLoop()
//! \brief counts calls; runs once per cycle in the "after_timeintegrator" task list.
void UserWorkInLoop(Mesh *pm) {
  phook->workinloop_calls += 1;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void UserHistOutput()
//! \brief exposes the workinloop call counter in the history output.
void UserHistOutput(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 1;
  pdata->label[0] = "workinloop_calls";
  pdata->hdata[0] = static_cast<Real>(phook->workinloop_calls);
}
} // namespace
