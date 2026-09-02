//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  // Default: this particle set never constrains the driver's timestep. Set
  // unconditionally here (not just in the lagrangian_mc/ito_2/lagrangian_tracer
  // pusher branch below) so every particle_type/pusher combination -- including
  // cosmic_ray/drift, which never overwrites this -- leaves dtnew well-defined.
  // Mesh::NewTimeStep now reads ppart->dtnew unconditionally for any Particles
  // instance, so this must never be left uninitialized.
  dtnew = std::numeric_limits<float>::max();
  ito2_enforce_locality_dt = false;

  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // base seed for deterministic per-particle random draws. Deliberately a FIXED
  // constant, not rank/gids-dependent: LagrangianMCUniform01/StatelessUniform01
  // already hash in the particle's own (globally unique) tag, so a rank-varying
  // base seed adds no real decorrelation value. A rank-varying default (e.g.
  // -1-gids, tried initially) breaks restart-exactness: restart files embed the
  // resolved ParameterInput block, and on restart every rank's GetOrAddInteger()
  // call below finds "particles/random_seed" already present (typically from
  // whichever rank's block got embedded) and reuses THAT single resolved value
  // instead of recomputing its own per-rank default -- so continuous (each rank
  // computing gids-based -1-gids fresh) and a restarted run (every rank reusing
  // one embedded rank's value) silently use different seeds on most ranks. A
  // fixed constant sidesteps this entirely: every rank always resolves to the
  // same value, whether freshly defaulted or inherited from another rank's
  // embedded parameter block.
  random_seed = pin->GetOrAddInteger("particles","random_seed",-1);

  // read number of particles per cell, and calculate number of particles this pack
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

  // compute number of particles as real number, since ppc can be < 1
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
  // then cast to integer
  nprtcl_thispack = static_cast<int>(r_npart);

  // select particle type
  {
    std::string ptype = pin->GetString("particles","particle_type");
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
    } else if (ptype.compare("lagrangian_tracer") == 0 ||
               ptype.compare("mass_tracer") == 0 ||
               ptype.compare("lagrangian_mc") == 0) {
      particle_type = ParticleType::lagrangian_tracer;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type = '" << ptype << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // select pusher algorithm
  {
    std::string ppush = pin->GetString("particles","pusher");
    if (ppush.compare("drift") == 0) {
      pusher = ParticlesPusher::drift;
      if (particle_type == ParticleType::lagrangian_tracer) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Particle pusher 'drift' not allowed for lagrangian_mc"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (ppush.compare("lagrangian_mc") == 0 ||
               ppush.compare("ito_2") == 0 ||
               ppush.compare("ito2") == 0 ||
               ppush.compare("ito") == 0 ||
               ppush.compare("lagrangian_tracer") == 0 ||
               ppush.compare("classical_lagrangian") == 0 ||
               ppush.compare("classical") == 0) {
      if (particle_type != ParticleType::lagrangian_tracer) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Particle pusher '" << ppush
                  << "' requires particle_type = lagrangian_tracer"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // driver inherits its timestep from the fluid; particle dt never binds
      dtnew = std::numeric_limits<float>::max();
      if (ppush.compare("lagrangian_mc") == 0) {
        pusher = ParticlesPusher::lagrangian_mc;
      } else if (ppush.compare("ito_2") == 0 ||
                 ppush.compare("ito2") == 0 ||
                 ppush.compare("ito") == 0) {
        pusher = ParticlesPusher::ito_2;
      } else {
        pusher = ParticlesPusher::lagrangian_tracer;
      }
      // lagrangian_mc/ito_2 move particles using the fluid's saved density flux;
      // classical (CIC velocity interpolation) doesn't need it
      if (pusher == ParticlesPusher::lagrangian_mc || pusher == ParticlesPusher::ito_2) {
        if (ppack->pmhd != nullptr) {
          ppack->pmhd->SetSaveUFlxIdn();
        } else if (ppack->phydro != nullptr) {
          ppack->phydro->SetSaveUFlxIdn();
        }
      }
      // Moseley+2026 Eq 63 locality bound |u_i|*dt/h < 1/4, ito_2 only; off by
      // default, see the member declaration in particles.hpp for the rationale.
      ito2_enforce_locality_dt = pin->GetOrAddBoolean("particles",
                                                        "ito2_enforce_locality_dt",
                                                        false);
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher must be specified in <particles> block"
                <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // set dimensions of particle arrays. Note particles only work in 2D/3D
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particles only work in 2D/3D, but 1D problem initialized" <<std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (particle_type) {
    case ParticleType::cosmic_ray:
      {
        // save particle position then velocity for compiler optimizations even though
        // 2d runs will not require all six real entries
        nrdata = 6;
        nidata = 2;
        break;
      }
    case ParticleType::lagrangian_tracer:
      {
        nrdata = 3;
        nidata = 4;  // gid, ptag, lastmove, lastlevel (rseed computed on-the-fly)
        // lastmove:
        //  if >= 0 => encodes current-zone parity (i_isodd,j_isodd,k_isodd) * 8, plus
        //             the MC-pusher's last-crossed-face code
        //  if -1   => freeze particle and perform no updates or position checks
        //  if -2   => remove from domain at next chance (not yet implemented)
        break;
      }
    default:
      break;
  }
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  rand_pool64 = Kokkos::Random_XorShift64_Pool<>(random_seed);

  // allocate boundary object
  min_radius = -1;
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
}

//----------------------------------------------------------------------------------------
// ReallocateParticles()
// Update particle arrays and particle-count bookkeeping for a new number of particles
// in this pack. Does NOT preserve any existing particle data -- callers (e.g. a fresh-run
// pgen initializer or the restart reader) are responsible for filling prtcl_rdata/idata
// afterward.

void Particles::ReallocateParticles(int new_nprtcl_thispack) {
  nprtcl_thispack = new_nprtcl_thispack;
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);
}

//----------------------------------------------------------------------------------------
// CreateParticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  std::string assign = pin->GetOrAddString("particles","assign_tag","index_order");

  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (assign.compare("index_order") == 0) {
    int tagstart = 0;
    for (int n=1; n<=global_variable::my_rank; ++n) {
      tagstart += pmy_pack->pmesh->nprtcl_eachrank[n-1];
    }

    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = tagstart + p;
    });

  // tags are assigned sequentially across ranks
  } else if (assign.compare("rank_order") == 0) {
    int myrank = global_variable::my_rank;
    int nranks = global_variable::nranks;
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      pi(PTAG,p) = myrank + nranks*p;
    });

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assignment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles
