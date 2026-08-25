//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.cpp
//! \brief Implementation of constructors and functions in class ProblemGenerator.
//! Default constructor calls problem generator function, while  constructor for restarts
//! reads data from restart file, as well as re-initializing problem-specific data.

#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include "athena.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/z4c.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
#include "particles/particles.hpp"
#include "coordinates/cell_locations.hpp"
#include "pgen.hpp"

namespace {
// Deterministic hashed draw (splitmix64-derived) used by InitializeLagrangianParticles()
// for fresh-run particle placement, so the initial particle count/positions per zone are
// reproducible given the same pos_init_seed -- not used for exact-restart continuity,
// which comes from the particle restart file instead (see InitializeParticlesFromRestart).
KOKKOS_INLINE_FUNCTION
Real PGenUniform01(const int64_t base_seed, const int gid, const int zone_index,
                   const int particle_index, const int stream) {
  uint64_t z = static_cast<uint64_t>(base_seed);
  z += static_cast<uint64_t>(gid) * 0x9e3779b97f4a7c15ULL;
  z += static_cast<uint64_t>(zone_index) * 0xbf58476d1ce4e5b9ULL;
  z += static_cast<uint64_t>(particle_index + 1) * 0x94d049bb133111ebULL;
  z += static_cast<uint64_t>(stream) * 0x4f1bbcdcBfa540abULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return static_cast<Real>(z >> 11) *
         static_cast<Real>(1.0 / 9007199254740992.0);
}
} // namespace


//----------------------------------------------------------------------------------------
// default constructor, calls pgen function.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm) :
    user_bcs(false),
    user_srcs(false),
    user_hist(false),
    user_dt(false),
    pmy_mesh_(pm) {
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }

  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_dt   = pin->GetOrAddBoolean("problem","user_dt",false);
  user_work_in_loop   = pin->GetOrAddBoolean("problem","user_work_in_loop",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);

  // second argument false since this IS NOT a restart
  CallProblemGenerator(pin, false);

  // Check that user defined BCs were enrolled if needed
  if (user_bcs) {
    if (user_bcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User BCs specified in <mesh> block, but not enrolled "
                << "by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
    // Warn: user BCs are applied to the fine arrays only. The coarse-array path used by
    // prolongation is not filled by user BCs (see UserBoundaryFnPtr in pgen.hpp), so a
    // fine/coarse boundary that coincides with a physical (user) boundary can
    // prolongate from unfilled coarse ghosts and trigger C2P failures etc.
    // Keep refinement away from user-boundary faces.
    if (pm->multilevel && global_variable::my_rank == 0) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "User-defined boundary conditions are combined with SMR/AMR. User BCs "
                << "are only applied to the fine arrays, not the coarse arrays used "
                << "for prolongation. Avoid refining at user-boundary faces, otherwise "
                << "prolongation may read unfilled coarse ghost zones (e.g. C2P fail.)."
                << std::endl;
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, but "
                << "not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined dt function is enrolled if needed
  if (user_dt) {
    if (user_time_step_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "User time step function specified in <problem> block, but "
                << "not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
// constructor for restarts
// When called, data needed to rebuild mesh has been read from restart file by
// Mesh::BuildTreeFromRestart() function. This constructor reads from the restart file and
// initializes all the dependent variables (u0,b0,etc) stored in each Physics class. It
// also calls ProblemGenerator::SetProblemData() function to set any user-defined BCs,
// and any data necessary for restart runs to continue correctly.

ProblemGenerator::ProblemGenerator(ParameterInput *pin, Mesh *pm, IOWrapper resfile,
                                   bool single_file_per_rank, IOWrapper *prestartfile) :
    user_bcs(false),
    user_srcs(false),
    user_hist(false),
    pmy_mesh_(pm) {
  // check for user-defined boundary conditions
  for (int dir=0; dir<6; ++dir) {
    if (pm->mesh_bcs[dir] == BoundaryFlag::user) {
      user_bcs = true;
    }
  }
  user_srcs = pin->GetOrAddBoolean("problem","user_srcs",false);
  user_hist = pin->GetOrAddBoolean("problem","user_hist",false);
  user_dt = pin->GetOrAddBoolean("problem","user_dt",false);
  user_work_in_loop = pin->GetOrAddBoolean("problem","user_work_in_loop",false);

  // get spatial dimensions of arrays, including ghost zones
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nout1 = indcs.nx1 + 2*(indcs.ng);
  int nout2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;
  // calculate total number of CC variables
  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  adm::ADM* padm = pm->pmb_pack->padm;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  radiation::Radiation* prad=pm->pmb_pack->prad;
  TurbulenceDriver* pturb=pm->pmb_pack->pturb;
  int nrad = 0, nhydro = 0, nmhd = 0, nforce = 6, nadm = 0, nz4c = 0;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }
  if (pz4c != nullptr) {
    nz4c = pz4c->nz4c;
  } else if (padm != nullptr) {
    nadm = padm->nadm;
  }

  // root process reads z4c last_output_time and tracker data
  if (pz4c != nullptr) {
    Real last_output_time = 0.0;
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_Reals(&last_output_time, 1,single_file_per_rank) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "z4c::last_output_time data size read from restart "
                  << "file is incorrect, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      MPI_Bcast(&last_output_time, sizeof(Real), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    pz4c->last_output_time = last_output_time;

    for (auto &pt : pz4c->ptracker) {
      Real pos[3];
      if (global_variable::my_rank == 0 || single_file_per_rank) {
        if (resfile.Read_Reals(&pos[0], 3, single_file_per_rank) != 3) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "compact object tracker data size read from restart "
                    << "file is incorrect, restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
      }
#if MPI_PARALLEL_ENABLED
      if (!single_file_per_rank) {
        MPI_Bcast(&pos[0], 3*sizeof(Real), MPI_CHAR, 0, MPI_COMM_WORLD);
      }
#endif
      pt->SetPos(&pos[0]);
    }
  }

  if (pturb != nullptr && pturb->restart_forcing) {
    // root process reads size the random seed
    char *rng_data = new char[sizeof(RNG_State)];
    // the master process reads the variables data
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_bytes(rng_data, 1, sizeof(RNG_State), single_file_per_rank)
          != sizeof(RNG_State)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "RNG data size read from restart file is incorrect, "
                  << "restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      // then broadcast the RNG information
      MPI_Bcast(rng_data, sizeof(RNG_State), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    std::memcpy(&(pturb->rstate), &(rng_data[0]), sizeof(RNG_State));
    delete[] rng_data;

    // n_turb_updates_yet must be read back as-written, not recomputed from
    // pm->time: InitializeModes only runs once per cycle (using that
    // cycle's pre-step time), so with CFL-limited steps the count of
    // update-windows actually processed by the time the restart file was
    // written can be less than floor(pm->time/dt_turb_update)+1. Restoring
    // the exact saved value (rather than that formula) is what makes
    // InitializeModes' replay loop below correctly resume -- replaying
    // exactly the updates that hadn't happened yet, no more and no fewer --
    // instead of either replaying stale history on top of the freshly
    // restored force/rstate, or skipping updates that were actually still
    // due.
    int n_turb_updates_yet = 0;
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_bytes(&n_turb_updates_yet, 1, sizeof(int), single_file_per_rank)
          != sizeof(int)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "n_turb_updates_yet data size read from restart "
                  << "file is incorrect, restart file is broken." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      MPI_Bcast(&n_turb_updates_yet, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    pturb->n_turb_updates_yet = n_turb_updates_yet;
  }

  // root process reads size of CC and FC data arrays from restart file
  IOWrapperSizeT variablesize = sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    if (resfile.Read_bytes(variabledata, 1, variablesize, single_file_per_rank)
        != variablesize) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Variable data size read from restart file is incorrect, "
                << "restart file is broken." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
#if MPI_PARALLEL_ENABLED
  // then broadcast the datasize information
  if (!single_file_per_rank) {
    MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif
  IOWrapperSizeT data_size;
  std::memcpy(&data_size, &(variabledata[0]), sizeof(IOWrapperSizeT));

  // calculate total number of CC variables
  IOWrapperSizeT headeroffset=0;
  // master process gets file offset
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    headeroffset = resfile.GetPosition(single_file_per_rank);
  }
#if MPI_PARALLEL_ENABLED
  // then broadcasts it
  if (!single_file_per_rank) {
    MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  IOWrapperSizeT data_size_ = 0;
  if (phydro != nullptr) {
    data_size_ += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
  }
  if (pmhd != nullptr) {
    data_size_ += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    data_size_ += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    data_size_ += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    data_size_ += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
  }
  if (prad != nullptr) {
    data_size_ += nout1*nout2*nout3*nrad*sizeof(Real);   // rad i0
  }
  if (pturb != nullptr && pturb->restart_forcing) {
    data_size_ += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
  }
  if (pz4c != nullptr) {
    data_size_ += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
  } else if (padm != nullptr) {
    data_size_ += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
  }

  if (data_size_ != data_size) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "CC data size read from restart file not equal to size "
              << "of Hydro, MHD, Rad, and/or Z4c arrays, restart file is broken."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // read CC data into host array
  IOWrapperSizeT offset_myrank = headeroffset;
  if (!single_file_per_rank) {
    offset_myrank += data_size_ * pm->gids_eachrank[global_variable::my_rank];
  }
  IOWrapperSizeT myoffset = offset_myrank;

  HostArray5D<Real> ccin("rst-cc-in", 1, 1, 1, 1, 1);
  HostFaceFld4D<Real> fcin("rst-fc-in", 1, 1, 1, 1);

  // calculate max/min number of MeshBlocks across all ranks
  int noutmbs_max = pm->nmb_eachrank[0];
  int noutmbs_min = pm->nmb_eachrank[0];
  for (int i=0; i<(global_variable::nranks); ++i) {
    noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
    noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
  }

  if (phydro != nullptr) {
    Kokkos::realloc(ccin, nmb, nhydro, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC hydro data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(phydro->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nhydro*sizeof(Real); // hydro u0
    myoffset = offset_myrank;
  }

  if (pmhd != nullptr) {
    Kokkos::realloc(ccin, nmb, nmhd, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
            != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC mhd data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nmhd*sizeof(Real);   // mhd u0
    myoffset = offset_myrank;

    Kokkos::realloc(fcin.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::realloc(fcin.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::realloc(fcin.x3f, nmb, nout3+1, nout2, nout1);
    // read FC data into host array, again one MeshBlock at a time
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to write, so write collectively
      if (m < noutmbs_min) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        size_t fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at_all(x1fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at_all(x2fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at_all(x3fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      } else if (m < pm->nmb_thisrank) {
        // get ptr to x1-face field
        auto x1fptr = Kokkos::subview(fcin.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        size_t fldcnt = x1fptr.size();

        if (resfile.Read_Reals_at(x1fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x1f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x2-face field
        auto x2fptr = Kokkos::subview(fcin.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x2fptr.size();

        if (resfile.Read_Reals_at(x2fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x2f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        // get ptr to x3-face field
        auto x3fptr = Kokkos::subview(fcin.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
        fldcnt = x3fptr.size();

        if (resfile.Read_Reals_at(x3fptr.data(), fldcnt, myoffset,
                                      single_file_per_rank) != fldcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Input b0.x3f field not read correctly from rst file, "
                << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += fldcnt*sizeof(Real);

        myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x1f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x1f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x2f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x2f);
    Kokkos::deep_copy(Kokkos::subview(pmhd->b0.x3f, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL), fcin.x3f);
    offset_myrank += (nout1+1)*nout2*nout3*sizeof(Real);    // mhd b0.x1f
    offset_myrank += nout1*(nout2+1)*nout3*sizeof(Real);    // mhd b0.x2f
    offset_myrank += nout1*nout2*(nout3+1)*sizeof(Real);    // mhd b0.x3f
    myoffset = offset_myrank;
  }

  if (prad != nullptr) {
    Kokkos::realloc(ccin, nmb, nrad, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC rad data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(prad->i0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nrad*sizeof(Real);   // radiation i0
    myoffset = offset_myrank;
  }

  if (pturb != nullptr && pturb->restart_forcing) {
    Kokkos::realloc(ccin, nmb, nforce, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC turb data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    // Kokkos cannot deep_copy directly between a Host view and a Cuda view
    // when either side is a non-contiguous subview -- slicing the channel
    // (2nd) dimension of ccin makes it non-contiguous on a GPU build, and
    // there's no common execution space that can walk a non-contiguous
    // layout across the Host/Cuda boundary. So pull each half into its own
    // contiguous host buffer with a host-to-host copy first (same space,
    // slicing is fine there), then transfer each whole buffer to the device.
    HostArray5D<Real> force_host("force_host", nmb, 3, nout3, nout2, nout1);
    HostArray5D<Real> force_tmp1_host("force_tmp1_host", nmb, 3, nout3, nout2, nout1);
    Kokkos::deep_copy(force_host, Kokkos::subview(ccin, Kokkos::ALL, std::make_pair(0,3),
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(force_tmp1_host, Kokkos::subview(ccin, Kokkos::ALL,
                      std::make_pair(3,6), Kokkos::ALL, Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(Kokkos::subview(pturb->force, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), force_host);
    Kokkos::deep_copy(Kokkos::subview(pturb->force_tmp1, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), force_tmp1_host);
    offset_myrank += nout1*nout2*nout3*nforce*sizeof(Real); // forcing
    myoffset = offset_myrank;
  }

  if (pz4c != nullptr) {
    Kokkos::realloc(ccin, nmb, nz4c, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC z4c data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(pz4c->u0, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nz4c*sizeof(Real);   // z4c u0
    myoffset = offset_myrank;

    // We also need to reinitialize the ADM data.
    pz4c->Z4cToADM(pmy_mesh_->pmb_pack);
  } else if (padm != nullptr) {
    Kokkos::realloc(ccin, nmb, nadm, nout3, nout2, nout1);
    for (int m=0;  m<noutmbs_max; ++m) {
      // every rank has a MB to read, so read collectively
      if (m < noutmbs_min) {
        // get ptr to cell-centered MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;

      // some ranks are finished writing, so use non-collective write
      } else if (m < pm->nmb_thisrank) {
        // get ptr to MeshBlock data
        auto mbptr = Kokkos::subview(ccin, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                     Kokkos::ALL);
        size_t mbcnt = mbptr.size();
        if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset,
                                      single_file_per_rank) != mbcnt) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "CC adm data not read correctly from rst file, "
                    << "restart file is broken." << std::endl;
          exit(EXIT_FAILURE);
        }
        myoffset += data_size;
      }
    }
    Kokkos::deep_copy(Kokkos::subview(padm->u_adm, std::make_pair(0,nmb), Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), ccin);
    offset_myrank += nout1*nout2*nout3*nadm*sizeof(Real);   // adm u_adm
    myoffset = offset_myrank;
  }

  // call problem generator again to re-initialize data, fn ptrs, as needed
  // second argument true since this IS a restart
  CallProblemGenerator(pin, true);

  // Particles are restored from a separate particle restart file (-p <file>), not from
  // resfile above -- this keeps the fluid and particle restart formats independent.
  if (pm->pmb_pack->ppart != nullptr) {
    if (prestartfile == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particles are enabled but no particle restart file "
                << "is specified." << std::endl
                << "Use -p <file> command line option." << std::endl;
      exit(EXIT_FAILURE);
    }

    if (global_variable::my_rank == 0) {
      std::cout << "Starting particle restart... " << std::endl;
    }

    // header is 2 x int64_t: a magic number (42) and the total particle count
    char *headerdata = new char[8];
    int64_t magic_number = 0;
    int64_t number_particles = 0;

    if (prestartfile->Read_bytes_at(headerdata, 8, 1, 0, single_file_per_rank) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Cannot read header byte from particle restart file" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::memcpy(&magic_number, headerdata, 8);
    if (magic_number != 42) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Particle restart file magic number is not 42, got "
                << magic_number << std::endl;
      exit(EXIT_FAILURE);
    }

    if (prestartfile->Read_bytes_at(headerdata, 8, 1, 8, single_file_per_rank) != 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Cannot read number of particles from particle restart file"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    std::memcpy(&number_particles, headerdata, 8);
    if (global_variable::my_rank == 0) {
      std::cout << "Found " << number_particles << " particles in restart file"
                << std::endl;
    }
    delete[] headerdata;

    // read the data - gid, tag, plastmove, x, y, z (all as Real/double)
    Real *gid_data = new Real[number_particles];
    Real *tag_data = new Real[number_particles];
    Real *plastmove_data = new Real[number_particles];
    Real *X_data = new Real[number_particles];
    Real *Y_data = new Real[number_particles];
    Real *Z_data = new Real[number_particles];

    std::size_t data_offset = 2 * sizeof(int64_t);  // after the 2 int64_t header fields

    if (
      (prestartfile->Read_Reals_at(gid_data, number_particles, data_offset,
                                    single_file_per_rank) != number_particles) ||
      (prestartfile->Read_Reals_at(tag_data, number_particles,
                                    data_offset + number_particles*sizeof(Real),
                                    single_file_per_rank) != number_particles) ||
      (prestartfile->Read_Reals_at(plastmove_data, number_particles,
                                    data_offset + 2*number_particles*sizeof(Real),
                                    single_file_per_rank) != number_particles) ||
      (prestartfile->Read_Reals_at(X_data, number_particles,
                                    data_offset + 3*number_particles*sizeof(Real),
                                    single_file_per_rank) != number_particles) ||
      (prestartfile->Read_Reals_at(Y_data, number_particles,
                                    data_offset + 4*number_particles*sizeof(Real),
                                    single_file_per_rank) != number_particles) ||
      (prestartfile->Read_Reals_at(Z_data, number_particles,
                                    data_offset + 5*number_particles*sizeof(Real),
                                    single_file_per_rank) != number_particles)
     ) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "Cannot read particle data from particle restart file" << std::endl;
      exit(EXIT_FAILURE);
    }

    // figure out how many particles belong to this rank (matched by PGID)
    int64_t nparticles_thispack = 0;
    for (int64_t i=0; i<number_particles; i++) {
      int gid = static_cast<int>(gid_data[i]);
      for (int m=0; m<pm->pmb_pack->nmb_thispack; ++m) {
        if (pm->pmb_pack->pmb->mb_gid.h_view(m) == gid) {
          nparticles_thispack += 1;
        }
      }
    }

    pm->pmb_pack->ppart->ReallocateParticles(nparticles_thispack);

    // populate dual array to add particles to device; order: gid, tag, plastmove, x, y, z
    DualArray2D<Real> part_data("particle_syncdata", nparticles_thispack, 6);
    int pidx = 0;
    for (int64_t i=0; i<number_particles; i++) {
      int gid = static_cast<int>(gid_data[i]);
      for (int m=0; m<pm->pmb_pack->nmb_thispack; ++m) {
        if (pm->pmb_pack->pmb->mb_gid.h_view(m) == gid) {
          part_data.h_view(pidx,0) = gid_data[i];
          part_data.h_view(pidx,1) = tag_data[i];
          part_data.h_view(pidx,2) = plastmove_data[i];  // preserve frozen/deleted status
          part_data.h_view(pidx,3) = X_data[i];
          part_data.h_view(pidx,4) = Y_data[i];
          part_data.h_view(pidx,5) = Z_data[i];
          pidx += 1;
        }
      }
    }
    delete[] gid_data;
    delete[] tag_data;
    delete[] plastmove_data;
    delete[] X_data;
    delete[] Y_data;
    delete[] Z_data;

    part_data.template modify<HostMemSpace>();
    part_data.template sync<DevExeSpace>();

    auto &pr = pm->pmb_pack->ppart->prtcl_rdata;
    auto &pi = pm->pmb_pack->ppart->prtcl_idata;
    bool snap_to_cell_center =
      (pm->pmb_pack->ppart->pusher == ParticlesPusher::lagrangian_mc);

    InitializeParticlesFromRestart(
        nparticles_thispack,
        part_data,
        pr,
        pi,
        snap_to_cell_center,
        pm->pmb_pack->pmesh->multi_d,
        pm->pmb_pack->pmesh->three_d,
        pm->pmb_pack->pmb->mb_size.d_view,
        pm->pmb_pack->pmb->mb_lev.d_view,
        pm->pmb_pack->gids,
        pm->pmb_pack->nmb_thispack,
        pm->pmb_pack->pmesh->mb_indcs,
        pm->pmb_pack->pmb->mb_bcs.d_view,
        pm->pmb_pack->ppart->min_radius
    );
  } // end if restart for particles

  // Check that user defined BCs were enrolled if needed
  if (user_bcs) {
    if (user_bcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User BCs specified in <mesh> block, but not enrolled "
                << "during restart by SetProblemData()." << std::endl;
      exit(EXIT_FAILURE);
    }
    // See note in the non-restart constructor: user BCs only fill the fine arrays, so
    // refinement at user-boundary faces can prolongate from unfilled coarse ghost zones.
    if (pm->multilevel && global_variable::my_rank == 0) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "User-defined boundary conditions are combined with SMR/AMR. User BCs "
                << "are only applied to the fine arrays, not the coarse arrays used "
                << "for prolongation. Avoid refining at user-boundary faces, otherwise "
                << "prolongation may read unfilled coarse ghost zones (e.g. C2P fail.)."
                << std::endl;
    }
  }
  // Check that user defined srcterms were enrolled if needed
  if (user_srcs) {
    if (user_srcs_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User SRCs specified in <problem> block, but not "
                << "enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined history outputs were enrolled if needed
  if (user_hist) {
    if (user_hist_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "User history output specified in <problem> block, "
                << "but not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  // Check that user defined dt function is enrolled if needed
  if (user_dt) {
    if (user_time_step_func == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "User time step function specified in <problem> block, but "
                << "not enrolled by UserProblem()." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::OutputErrors()
//! \brief Generic function for computing the L1 and L-infty difference between solutions
//! stored in the u0 and u1 registers, and outputting them to an error file.  This is
//! used for linear wave convergence tests, for example.
//! Function requires appropriate solutions already stored in u0 and u1.

void ProblemGenerator::OutputErrors(ParameterInput *pin, Mesh *pm) {
  Real l1_err[16];
  Real linfty_err=0.0;
  int nvars=0,nprev=0;

  // capture class variables for kernel
  auto &indcs = pm->mb_indcs;
  int &nx1 = indcs.nx1;
  int &nx2 = indcs.nx2;
  int &nx3 = indcs.nx3;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // compute errors for Hydro  -----------------------------------------------------------
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro;

    auto &is_ideal_ = pmbp->phydro->peos->eos_data.is_ideal;
    auto &u0_ = pmbp->phydro->u0;
    auto &u1_ = pmbp->phydro->u1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("L1-err",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (is_ideal_) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n] = sum_this_mb.the_array[n];
    }
    nprev += nvars;
  }

  // compute errors for MHD  -------------------------------------------------------------
  if (pmbp->pmhd != nullptr) {
    nvars = pmbp->pmhd->nmhd + 3;  // include 3-compts of cell-centered B in errors
    auto &is_ideal_ = pmbp->pmhd->peos->eos_data.is_ideal;

    int bindx;
    if (is_ideal_) {
      bindx = 5;
    } else {
      bindx = 4;
    }

    auto &u0_ = pmbp->pmhd->u0;
    auto &u1_ = pmbp->pmhd->u1;
    auto &b0_ = pmbp->pmhd->b0;
    auto &b1_ = pmbp->pmhd->b1;

    const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    array_sum::GlobalSum sum_this_mb;
    Kokkos::parallel_reduce("L1-err-Sums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum, Real &max_err) {
      // compute n,k,j,i indices of thread
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;

      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

      // conserved variables:
      array_sum::GlobalSum evars;
      evars.the_array[IDN] = vol*fabs(u0_(m,IDN,k,j,i) - u1_(m,IDN,k,j,i));
      max_err = fmax(max_err, evars.the_array[IDN]);
      evars.the_array[IM1] = vol*fabs(u0_(m,IM1,k,j,i) - u1_(m,IM1,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM1]);
      evars.the_array[IM2] = vol*fabs(u0_(m,IM2,k,j,i) - u1_(m,IM2,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM2]);
      evars.the_array[IM3] = vol*fabs(u0_(m,IM3,k,j,i) - u1_(m,IM3,k,j,i));
      max_err = fmax(max_err, evars.the_array[IM3]);
      if (is_ideal_) {
        evars.the_array[IEN] = vol*fabs(u0_(m,IEN,k,j,i) - u1_(m,IEN,k,j,i));
        max_err = fmax(max_err, evars.the_array[IEN]);
      }

      // cell-centered B
      Real bcc0 = 0.5*(b0_.x1f(m,k,j,i) + b0_.x1f(m,k,j,i+1));
      Real bcc1 = 0.5*(b1_.x1f(m,k,j,i) + b1_.x1f(m,k,j,i+1));
      evars.the_array[bindx] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+1]);

      bcc0 = 0.5*(b0_.x2f(m,k,j,i) + b0_.x2f(m,k,j+1,i));
      bcc1 = 0.5*(b1_.x2f(m,k,j,i) + b1_.x2f(m,k,j+1,i));
      evars.the_array[bindx+1] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+2]);

      bcc0 = 0.5*(b0_.x3f(m,k,j,i) + b0_.x3f(m,k+1,j,i));
      bcc1 = 0.5*(b1_.x3f(m,k,j,i) + b1_.x3f(m,k+1,j,i));
      evars.the_array[bindx+2] = vol*fabs(bcc0 - bcc1);
      max_err = fmax(max_err, evars.the_array[IEN+3]);

      // fill rest of the_array with zeros, if narray < NREDUCTION_VARIABLES
      for (int n=nvars; n<NREDUCTION_VARIABLES; ++n) {
        evars.the_array[n] = 0.0;
      }

      // sum into parallel reduce
      mb_sum += evars;
    }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb), Kokkos::Max<Real>(linfty_err));

    // store data into l1_err array
    for (int n=0; n<nvars; ++n) {
      l1_err[n+nprev] = sum_this_mb.the_array[n];
    }
    nprev += nvars;
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_err, nprev, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &linfty_err, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

  // normalize errors by number of cells
  Real vol=  (pmbp->pmesh->mesh_size.x1max - pmbp->pmesh->mesh_size.x1min)
            *(pmbp->pmesh->mesh_size.x2max - pmbp->pmesh->mesh_size.x2min)
            *(pmbp->pmesh->mesh_size.x3max - pmbp->pmesh->mesh_size.x3min);
  for (int i=0; i<nprev; ++i) l1_err[i] = l1_err[i]/vol;
  linfty_err /= vol;

  // compute rms error
  Real rms_err = 0.0;
  for (int i=0; i<nprev; ++i) {
    rms_err += SQR(l1_err[i]);
  }
  rms_err = std::sqrt(rms_err);

  // root process opens output file and writes out errors
  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Error output file could not be opened" <<std::endl;
        std::exit(EXIT_FAILURE);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3   Ncycle   RMS-L1       L-infty       ");
      if (pmbp->phydro != nullptr) {
        std::fprintf(pfile,"d_L1          M1_L1         M2_L1         M3_L1         ");
        if (pmbp->phydro->peos->eos_data.is_ideal) {
          std::fprintf(pfile,"E_L1          ");
        }
      }
      if (pmbp->pmhd != nullptr) {
        std::fprintf(pfile,"d_L1          M1_L1         M2_L1         M3_L1         ");
        if (pmbp->pmhd->peos->eos_data.is_ideal) {
          std::fprintf(pfile,"E_L1          ");
        }
        std::fprintf(pfile,"B1_L1         B2_L1         B3_L1");
      }
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%04d", pmbp->pmesh->mesh_indcs.nx1);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx2);
    std::fprintf(pfile, "  %04d", pmbp->pmesh->mesh_indcs.nx3);
    std::fprintf(pfile, "  %05d  %e %e", pmbp->pmesh->ncycle, rms_err, linfty_err);
    for (int i=0; i<nprev; ++i) {
      std::fprintf(pfile, "  %e", l1_err[i]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::CallProblemGenerator()
//! \brief selects one of the default problem generators compiled automatically with
//! the source code depending on input string in <problem> block ELSE selects a
//! user-defined problem generator function compiled with the code.

void ProblemGenerator::CallProblemGenerator(ParameterInput *pin, bool is_restart) {
#if USER_PROBLEM_ENABLED
  // call user-defined problem generator (if USER_PROBLEM_ENABLED macro defined at build)
  UserProblem(pin, is_restart);
#else
  // else read name of built-in pgen from <problem> block in input file, and call
  std::string pgen_fun_name = pin->GetOrAddString("problem", "pgen_name", "none");

  if (pgen_fun_name.compare("advection") == 0) {
    Advection(pin, is_restart);
  } else if (pgen_fun_name.compare("cpaw") == 0) {
    AlfvenWave(pin, is_restart);
  } else if (pgen_fun_name.compare("gr_bondi") == 0) {
    BondiAccretion(pin, is_restart);
  } else if (pgen_fun_name.compare("cshock") == 0) {
    CShock(pin, is_restart);
  } else if (pgen_fun_name.compare("divb_amr") == 0) {
    DivBAMR(pin, is_restart);
  } else if (pgen_fun_name.compare("diffusion") == 0) {
    Diffusion(pin, is_restart);
  } else if (pgen_fun_name.compare("linear_wave") == 0) {
    LinearWave(pin, is_restart);
  } else if (pgen_fun_name.compare("implode") == 0) {
    LWImplode(pin, is_restart);
  } else if (pgen_fun_name.compare("gr_monopole") == 0) {
    Monopole(pin, is_restart);
  } else if (pgen_fun_name.compare("mri3d") == 0) {
    MRI3d(pin, is_restart);
  } else if (pgen_fun_name.compare("orszag_tang") == 0) {
    OrszagTang(pin, is_restart);
  } else if (pgen_fun_name.compare("rad_linear_wave") == 0) {
    RadiationLinearWave(pin, is_restart);
  } else if (pgen_fun_name.compare("rad_beam") == 0) {
    RadiationBeam(pin, is_restart);
  } else if (pgen_fun_name.compare("shock_tube") == 0) {
    ShockTube(pin, is_restart);
  } else if (pgen_fun_name.compare("shwave") == 0) {
    Shwave(pin, is_restart);
  } else if (pgen_fun_name.compare("z4c_boosted_puncture") == 0) {
    Z4cBoostedPuncture(pin, is_restart);
  } else if (pgen_fun_name.compare("z4c_linear_wave") == 0) {
    Z4cLinearWave(pin, is_restart);
  } else if (pgen_fun_name.compare("diffusion") == 0) {
    Diffusion(pin, is_restart);
  } else if (pgen_fun_name.compare("gravity") == 0) {
    SelfGravity(pin, is_restart);
  } else if (pgen_fun_name.compare("binary_gravity") == 0) {
    BinaryGravity(pin, is_restart);
  } else if (pgen_fun_name.compare("be_collapse") == 0) {
    BECollapse(pin, is_restart);

  // pre-defined unit tests
  } else if (pgen_fun_name.compare("eos_compose") == 0) {
    EOSCompose(pin, is_restart);
  } else if (pgen_fun_name.compare("gauss_legendre") == 0) {
    GaussLegendre(pin, is_restart);

  } else {
    // name not set on command line or input file, print warning and quit
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Problem generator name could not be found in <problem> block in input file"
        << std::endl
        << "and it was not set by -D PROBLEM option on cmake command line during build"
        << std::endl
        << "Rerun cmake with -D PROBLEM=file to specify custom problem generator file"
        << std::endl;;
    std::exit(EXIT_FAILURE);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::InitializeLagrangianParticles()
//! \brief Mass- or volume-weighted fresh-run particle placement for Lagrangian tracer
//! pgens. Not used on restart -- particles are restored from the particle restart file
//! instead (see InitializeParticlesFromRestart() below).

void ProblemGenerator::InitializeLagrangianParticles(ParameterInput *pin,
                                                     const DvceArray5D<Real>& u0) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppart == nullptr) {
    return;
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &mbsize = pmbp->pmb->mb_size;
  auto &mblev = pmbp->pmb->mb_lev;
  auto gids = pmbp->gids;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  bool uniform_by_volume = pin->GetOrAddBoolean("particles","uniform_by_volume",false);
  bool random_positions = pin->GetOrAddBoolean("particles","random_positions",true);
  int64_t pos_init_seed = pin->GetOrAddInteger("particles","pos_init_seed",280496);

  Real total_weight = 0.0;
  Kokkos::parallel_reduce("pgen_lagrangian_particle_weight",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &weight_sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
    Real zone_weight = uniform_by_volume ? vol : u0(m,IDN,k,j,i)*vol;
    weight_sum += zone_weight;
  }, total_weight);

  Real total_weight_thispack = total_weight;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &total_weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (total_weight <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Cannot initialize Lagrangian particles with "
              << "non-positive total weight." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real target_nparticles = pin->GetOrAddReal("particles","target_count",100000.0);
  Real weight_per_particle = total_weight / target_nparticles;

  DualArray2D<int> nparticles_per_zone("partperzone", nmkji,2);
  par_for("lagrangian_particle_count", DevExeSpace(), 0,nmkji-1,
  KOKKOS_LAMBDA(int idx) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
    Real zone_weight = uniform_by_volume ? vol : u0(m,IDN,k,j,i)*vol;
    Real nppc = zone_weight / weight_per_particle;
    int nparticles = static_cast<int>(nppc);
    Real frac = nppc - static_cast<Real>(nparticles);
    if (PGenUniform01(pos_init_seed, gids + m, idx, 0, 0) < frac) {
      nparticles += 1;
    }
    nparticles_per_zone.d_view(idx,0) = nparticles;
  });

  nparticles_per_zone.template modify<DevExeSpace>();
  nparticles_per_zone.template sync<HostMemSpace>();
  int nparticles_thispack = 0;
  for (int i=0; i<nmkji; ++i) {
    nparticles_per_zone.h_view(i,1) = nparticles_thispack;
    nparticles_thispack += nparticles_per_zone.h_view(i,0);
  }
  nparticles_per_zone.template modify<HostMemSpace>();
  nparticles_per_zone.template sync<DevMemSpace>();

  if (global_variable::my_rank == 0) {
    std::cout << "total particle initialization weight across domain: " << total_weight
              <<  ", total weight in pack: " << total_weight_thispack
              << ", target nparticles: " << target_nparticles
              << ", nparticles in pack: " << nparticles_thispack << std::endl;
  }

  pmbp->ppart->ReallocateParticles(nparticles_thispack);
  auto &pr = pmbp->ppart->prtcl_rdata;
  auto &pi = pmbp->ppart->prtcl_idata;

  // lagrangian_mc jumps are pure +-dx displacements between cell centers, so its
  // particles must start exactly at a cell center -- restart snaps them back to
  // the cell center too (see InitializeParticlesFromRestart), and initial
  // placement has to match that invariant or the sub-cell offset acquired here
  // would be silently discarded on the first restart, diverging from a
  // continuous run.
  bool snap_to_cell_center = (pmbp->ppart->pusher == ParticlesPusher::lagrangian_mc);

  par_for("lagrangian_part_init", DevExeSpace(), 0,nmkji-1,
  KOKKOS_LAMBDA(int idx) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    int nparticles_in_zone = nparticles_per_zone.d_view(idx,0);
    int starting_index = nparticles_per_zone.d_view(idx,1);

    for (int p=0; p<nparticles_in_zone; ++p) {
      int pidx = p + starting_index;
      bool draw_random = random_positions && !snap_to_cell_center;
      Real rx = draw_random ? PGenUniform01(pos_init_seed, gids + m, idx, p, 1) - 0.5
                             : 0.0;
      Real ry = draw_random ? PGenUniform01(pos_init_seed, gids + m, idx, p, 2) - 0.5
                             : 0.0;
      Real rz = draw_random ? PGenUniform01(pos_init_seed, gids + m, idx, p, 3) - 0.5
                             : 0.0;

      pi(PGID,pidx) = gids + m;
      pi(PLASTMOVE,pidx) = 0;
      pi(PLASTLEVEL,pidx) = mblev.d_view(m);
      pr(IPX,pidx) = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min,
                                 mbsize.d_view(m).x1max) + rx*mbsize.d_view(m).dx1;
      pr(IPY,pidx) = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min,
                                 mbsize.d_view(m).x2max) + ry*mbsize.d_view(m).dx2;
      pr(IPZ,pidx) = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min,
                                 mbsize.d_view(m).x3max) + rz*mbsize.d_view(m).dx3;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::InitializeParticlesFromRestart()
//! \brief Restores particles read from a particle restart file into prtcl_rdata/idata,
//! matching each by PGID to the MeshBlock now owning it on this rank. Kept as its own
//! (non-member-lambda) function since a Kokkos device lambda directly inside the restart
//! constructor otherwise implicitly captures 'this'.

void ProblemGenerator::InitializeParticlesFromRestart(
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
    Real min_rad) {
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1;
  int nx2 = indcs.nx2;
  int nx3 = indcs.nx3;

  const bool snap_positions = snap_to_cell_center;

  par_for("part_init", DevExeSpace(), 0, nparticles_thispack-1,
  KOKKOS_LAMBDA(int idx) {
    // part_data order: gid(0), tag(1), plastmove(2), x(3), y(4), z(5)
    int pgid = static_cast<int>(part_data.d_view(idx,0));
    int m = pgid - gids;

    pi(PGID,idx) = pgid;
    pi(PTAG,idx) = static_cast<int>(part_data.d_view(idx,1));
    pi(PLASTMOVE,idx) = static_cast<int>(part_data.d_view(idx,2));  // frozen/deleted status

    // particle's PGID doesn't belong to this rank -- shouldn't happen (the caller only
    // includes particles matched to a local MeshBlock), but guard against it anyway
    if (m < 0 || m >= nmb) {
      pi(PLASTMOVE,idx) = -1;
      return;
    }

    pi(PLASTLEVEL,idx) = mblev(m);

    Real x = part_data.d_view(idx,3);
    Real y = part_data.d_view(idx,4);
    Real z = part_data.d_view(idx,5);

    // length of MeshBlock in each direction
    Real lx = (mbsize(m).x1max - mbsize(m).x1min);
    Real ly = (mbsize(m).x2max - mbsize(m).x2min);
    Real lz = (mbsize(m).x3max - mbsize(m).x3min);

    // integer offset of particle relative to center of MeshBlock (-1,0,+1)
    int ix = static_cast<int>((x - mbsize(m).x1min + lx)/lx) - 1;
    int iy = static_cast<int>((y - mbsize(m).x2min + ly)/ly) - 1;
    int iz = static_cast<int>((z - mbsize(m).x3min + lz)/lz) - 1;

    bool check_boundary = (
        mb_bcs(m, BoundaryFace::inner_x3) == BoundaryFlag::user && iz < 0)
    || ( mb_bcs(m, BoundaryFace::outer_x3) == BoundaryFlag::user && iz > 0)
    || ( mb_bcs(m, BoundaryFace::inner_x2) == BoundaryFlag::user && iy < 0)
    || ( mb_bcs(m, BoundaryFace::outer_x2) == BoundaryFlag::user && iy > 0)
    || ( mb_bcs(m, BoundaryFace::inner_x1) == BoundaryFlag::user && ix < 0)
    || ( mb_bcs(m, BoundaryFace::outer_x1) == BoundaryFlag::user && ix > 0)
    || ( sqrt(SQR(x) + SQR(y) + SQR(z)) < min_rad);

    int ip = (x - mbsize(m).x1min)/mbsize(m).dx1 + is;
    int jp = js;
    int kp = ks;
    if (multi_d) {
      jp = (y - mbsize(m).x2min)/mbsize(m).dx2 + js;
    }
    if (three_d) {
      kp = (z - mbsize(m).x3min)/mbsize(m).dx3 + ks;
    }
    if (check_boundary
        || ip < is || ip >= (nx1+is)
        || jp < js || jp >= (nx2+js)
        || kp < ks || kp >= (nx3+ks)) {
      pi(PLASTMOVE,idx) = -1;
      pr(IPX,idx) = x;
      pr(IPY,idx) = y;
      pr(IPZ,idx) = z;
    } else if (snap_positions) {
      pr(IPX,idx) = CellCenterX(ip-is, nx1, mbsize(m).x1min, mbsize(m).x1max);
      pr(IPY,idx) = CellCenterX(jp-js, nx2, mbsize(m).x2min, mbsize(m).x2max);
      pr(IPZ,idx) = CellCenterX(kp-ks, nx3, mbsize(m).x3min, mbsize(m).x3max);
    } else {
      pr(IPX,idx) = x;
      pr(IPY,idx) = y;
      pr(IPZ,idx) = z;
    }
  });
}
