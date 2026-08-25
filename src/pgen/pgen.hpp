#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.hpp
//  \brief definitions for ProblemGenerator class

#include <functional>
#include <memory>
#include <vector>

#include "athena.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "parameter_input.hpp"

// forward declarations for the particle-restart helper below
struct RegionSize;
struct RegionIndcs;
enum class BoundaryFlag;

using ProblemFinalizeFnPtr = void (*)(ParameterInput *pin, Mesh *pm);
using UserBoundaryFnPtr = void (*)(Mesh* pm);
using UserSrctermFnPtr = void (*)(Mesh* pm, const Real bdt);
using UserTimestepFnPtr = void (*)(Mesh* pm);
using UserRefinementFnPtr = void (*)(MeshBlockPack* pmbp);
using UserHistoryFnPtr = void (*)(HistoryData *pdata, Mesh *pm);
using UserWorkInLoopFnPtr = void (*)(Mesh* pm);

//----------------------------------------------------------------------------------------
//! \class ProblemGenerator

class ProblemGenerator {
 public:
  // constructor for new problems
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh);
  // constructor for restarts
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh, IOWrapper resfile,
                   bool single_file_per_rank=false, IOWrapper *prestartfile=nullptr);
  ~ProblemGenerator() = default;

  // true if user BCs are specified on any face
  bool user_bcs;

  // true if user srcterms are specified
  bool user_srcs;

  // true if user dt is specified
  bool user_dt;
  // store user_dt in dtnew here
  Real dtnew = 1e10;

  // true if user workinloop is specified
  bool user_work_in_loop;

  // true if user history outputs are specified
  bool user_hist;

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  // function pointer for final work after main loop (e.g. compute errors).  Called by
  // Driver::Finalize()
  ProblemFinalizeFnPtr pgen_final_func=nullptr;
  // function pointer for user-enrolled BCs.  Called in ApplyPhysicalBCs in task list
  UserBoundaryFnPtr user_bcs_func=nullptr;
  UserSrctermFnPtr user_srcs_func=nullptr;
  UserTimestepFnPtr user_time_step_func=nullptr;
  UserRefinementFnPtr user_ref_func=nullptr;
  UserHistoryFnPtr user_hist_func=nullptr;
  UserWorkInLoopFnPtr user_work_in_loop_func=nullptr;

  // predefined problem generator functions (default test suite)
  void CallProblemGenerator(ParameterInput *pin, bool is_restart);
  void Advection(ParameterInput *pin, const bool restart);
  void AlfvenWave(ParameterInput *pin, const bool restart);
  void BondiAccretion(ParameterInput *pin, const bool restart);
  void CShock(ParameterInput *pin, const bool restart);
  void DivBAMR(ParameterInput *pin, const bool restart);
  void Diffusion(ParameterInput *pin, const bool restart);
  void LinearWave(ParameterInput *pin, const bool restart);
  void LWImplode(ParameterInput *pin, const bool restart);
  void Monopole(ParameterInput *pin, const bool restart);
  void MRI3d(ParameterInput *pin, const bool restart);
  void OrszagTang(ParameterInput *pin, const bool restart);
  void ShockTube(ParameterInput *pin, const bool restart);
  void Shwave(ParameterInput *pin, const bool restart);
  void RadiationLinearWave(ParameterInput *pin, const bool restart);
  void RadiationBeam(ParameterInput *pin, const bool restart);
  void Z4cBoostedPuncture(ParameterInput *pin, const bool restart);
  void Z4cLinearWave(ParameterInput *pin, const bool restart);
  void SelfGravity(ParameterInput *pin, const bool restart);
  void BinaryGravity(ParameterInput *pin, const bool restart);
  void BECollapse(ParameterInput *pin, const bool restart);

  // predefined problem generator functions for unit tests
  void EOSCompose(ParameterInput *pin, const bool restart);
  void GaussLegendre(ParameterInput *pin, const bool restart);

  // Generic error output function (using difference u0-u1)
  void OutputErrors(ParameterInput *pin, Mesh *pm);

  // template for user-specified problem generator
  void UserProblem(ParameterInput *pin, const bool restart);

  // Restores particles read from a particle restart file into prtcl_rdata/idata,
  // matching each by PGID to the MeshBlock now owning it on this rank
  void InitializeParticlesFromRestart(
      int64_t nparticles_thispack,
      const DualArray2D<Real>& part_data,
      DvceArray2D<Real>& pr,
      DvceArray2D<int>& pi,
      bool snap_to_cell_center,
      bool multi_d,
      bool three_d,
      const DvceArray1D<RegionSize>& mbsize,
      const DvceArray1D<int>& mblev,
      int gids,
      int nmb,
      const RegionIndcs& indcs,
      const DvceArray2D<BoundaryFlag>& mb_bcs,
      Real min_rad);

  // Reusable fresh-run Lagrangian tracer particle initializer (mass- or volume-
  // weighted); see InitializeLagrangianParticles() in pgen.cpp for pgen usage
  void InitializeLagrangianParticles(ParameterInput *pin, const DvceArray5D<Real>& u0);

 private:
  bool single_file_per_rank; // for restart file naming
  Mesh* pmy_mesh_;
};

#endif // PGEN_PGEN_HPP_
