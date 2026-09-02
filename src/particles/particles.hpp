#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.hpp
//  \brief definitions for Particles class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

#include <Kokkos_Random.hpp>

// forward declarations

// constants that enumerate ParticlesPusher options
enum class ParticlesPusher {drift, leap_frog, lagrangian_tracer, lagrangian_mc, ito_2};

// constants that enumerate ParticleTypes
enum class ParticleType {cosmic_ray, lagrangian_tracer};

//----------------------------------------------------------------------------------------
//! \struct ParticlesTaskIDs
//  \brief container to hold TaskIDs of all particles tasks

struct ParticlesTaskIDs {
  TaskID push;
  TaskID newgid;
  TaskID count;
  TaskID irecv;
  TaskID sendp;
  TaskID recvp;
  TaskID csend;
  TaskID crecv;
  TaskID mradj;
  TaskID newdt;
};

namespace particles {

//----------------------------------------------------------------------------------------
//! \class Particles

class Particles {
  friend class ParticlesBoundaryValues;
 public:
  Particles(MeshBlockPack *ppack, ParameterInput *pin);
  ~Particles();

  // data
  ParticleType particle_type;
  int nprtcl_thispack;             // number of particles this MeshBlockPack
  int nrdata, nidata;
//  DvceArray1D<int>  prtcl_gid;     // GID of MeshBlock containing each par
//  DvceArray2D<Real> prtcl_pos;     // positions
//  DvceArray2D<Real> prtcl_vel;     // velocities
  DvceArray2D<Real> prtcl_rdata;   // real number properties each particle (x,v,etc.)
  DvceArray2D<int>  prtcl_idata;   // integer properties each particle (gid, tag, etc.)
  Real dtnew;

  // excise zone (used by some pgens): particles within this radius of the origin are
  // not updated. -1 disables the excision entirely.
  Real min_radius;
  // base seed for deterministic per-particle random draws; defaults (in the
  // constructor) to a value that varies by MeshBlockPack GID
  int64_t random_seed;

  ParticlesPusher pusher;

  // Moseley, Teyssier & Abel 2026 (arXiv:2604.23041) Eq 63: a stronger, accuracy-
  // motivated (not stability-motivated) locality bound |u_i|*dt/h < 1/4 for the
  // Ito-2 tracer's drift, in addition to the existing Cplus>1 stability guard in
  // PushIto2. Off by default (matches prior behavior; the paper itself notes
  // enforcing it "makes little difference to the results"); enabled via
  // <particles>/ito2_enforce_locality_dt = true. No effect for any pusher other
  // than ito_2.
  bool ito2_enforce_locality_dt;

  // Boundary communication buffers and functions for particles
  ParticlesBoundaryValues *pbval_part;

  // container to hold names of TaskIDs
  ParticlesTaskIDs id;

  // functions...
  void ReallocateParticles(int new_nprtcl_thispack);
  void CreateParticleTags(ParameterInput *pin);
  void AssembleTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  TaskStatus Push(Driver *pdriver, int stage);
  TaskStatus NewGID(Driver *pdriver, int stage);
  TaskStatus SendCnt(Driver *pdriver, int stage);
  TaskStatus InitRecv(Driver *pdriver, int stage);
  TaskStatus SendP(Driver *pdriver, int stage);
  TaskStatus RecvP(Driver *pdriver, int stage);
  TaskStatus ClearSend(Driver *pdriver, int stage);
  TaskStatus ClearRecv(Driver *pdriver, int stage);
  TaskStatus AdjustMeshRefinement(Driver *pdriver, int stage);
  TaskStatus NewTimeStep(Driver *pdriver, int stage);

  // per-pusher update functions, called from Push()
  void PushDrift();
  void PushLagrangianTracer();
  void PushLagrangianMC();
  void PushIto2();

 private:
  MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this Particles
  Kokkos::Random_XorShift64_Pool<> rand_pool64;
};

} // namespace particles
#endif // PARTICLES_PARTICLES_HPP_
