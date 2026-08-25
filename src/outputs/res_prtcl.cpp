//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file res_prtcl.cpp
//! \brief writes particle restart data in binary format.
//! Binary format consists of:
//!   - Header: 2 int64_t values (magic number 42, particle count in this file)
//!   - Data: 6 double/Real arrays (all as Real/double to avoid type issues)
//!     - gid[nparticles]      : MeshBlock global ID (double)
//!     - tag[nparticles]      : Particle tag (double)
//!     - plastmove[nparticles]: Last move status (double, -1 = frozen/deleted)
//!     - x[nparticles]        : X position (double)
//!     - y[nparticles]        : Y position (double)
//!     - z[nparticles]        : Z position (double)
//! Note: Random seeds are NOT stored - they are computed deterministically from
//! tag+ncycle
//! Data is written either to one shared MPI-IO file or to one local file per MPI rank.

#include <sys/stat.h>  // mkdir
#include <vector>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

ParticleRestartOutput::ParticleRestartOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output
  mkdir("prst",0775);
  bool single_file_per_rank = op.single_file_per_rank;
  if (single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "prst/rank_%08d/",
                  global_variable::my_rank);
    mkdir(rank_dir, 0775);
  }
}

//----------------------------------------------------------------------------------------
// ParticleRestartOutput::LoadOutputData()
// Copies real and integer particle data to host for outputs

void ParticleRestartOutput::LoadOutputData(Mesh *pm) {
  particles::Particles *pp = pm->pmb_pack->ppart;
  npout_thisrank = pm->nprtcl_thisrank;
  npout_total = pm->nprtcl_total;

  Kokkos::realloc(outpart_rdata, pp->nrdata, npout_thisrank);
  Kokkos::realloc(outpart_idata, pp->nidata, npout_thisrank);

  // Create mirror view on device of host view of output particle real/int data
  auto d_outpart_rdata = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(),
                                                    outpart_rdata);
  auto d_outpart_idata = Kokkos::create_mirror_view(Kokkos::DefaultHostExecutionSpace(),
                                                    outpart_idata);
  // Copy particle data into device mirrors
  Kokkos::deep_copy(d_outpart_rdata, pp->prtcl_rdata);
  Kokkos::deep_copy(d_outpart_idata, pp->prtcl_idata);
  // Copy particle data from device mirror to host output array
  Kokkos::deep_copy(outpart_rdata, d_outpart_rdata);
  Kokkos::deep_copy(outpart_idata, d_outpart_idata);
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleRestartOutput::WriteOutputFile(Mesh *pm)
//! \brief Writes particle restart data in binary format.
//!
//! Binary file format:
//!  1. Header: 2 x int64_t values
//!     - Magic number (42) for format identification
//!     - Number of particles in the file (global for shared files, local for per-rank
//!       files)
//!  2. Data arrays (all as double/Real):
//!     - gid[nparticles]      : MeshBlock global ID for each particle (double)
//!     - tag[nparticles]      : Particle tag (double)
//!     - plastmove[nparticles]: Last move status (double, -1 = frozen/deleted)
//!     - x[nparticles]        : X position (double)
//!     - y[nparticles]        : Y position (double)
//!     - z[nparticles]        : Z position (double)
//!  Note: Random seeds are NOT stored - computed from tag + ncycle

void ParticleRestartOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // create filename: "prst/file_basename"."file_id"."XXXXX".prtclrst
  // or "prst/rank_YYYYYYY/file_basename"."file_id"."XXXXX".prtclrst for
  // single_file_per_rank
  // where XXXXX = 5-digit file_number
  // where YYYYYYY = 8-digit rank number
  bool single_file_per_rank = out_params.single_file_per_rank;
  int npout_file = single_file_per_rank ? npout_thisrank : npout_total;
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  if (single_file_per_rank) {
    // Generate filename with rank-specific directory
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d/", global_variable::my_rank);
    fname.assign("prst/");
    fname.append(rank_dir);
    fname.append(out_params.file_basename);
    if (!out_params.file_id.empty()) {
      fname.append(".");
      fname.append(out_params.file_id);
    }
    fname.append(".");
    if (out_params.gid >= 0) {
      fname.append(std::to_string(out_params.gid));
      fname.append(".");
    }
    fname.append(number);
    fname.append(".prtclrst");
  } else {
    // Original single-file behavior
    fname.assign("prst/");
    fname.append(out_params.file_basename);
    if (!out_params.file_id.empty()) {
      fname.append(".");
      fname.append(out_params.file_id);
    }
    fname.append(".");
    if (out_params.gid >= 0) {
      fname.append(std::to_string(out_params.gid));
      fname.append(".");
    }
    fname.append(number);
    fname.append(".prtclrst");
  }

  IOWrapper partfile;
  std::size_t header_offset = 0;
  partfile.Open(fname.c_str(), IOWrapper::FileMode::write, single_file_per_rank);

  // Write header: magic number (42) and particle count in this file (as int64_t)
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    int64_t header[2];
    header[0] = 42;  // magic number
    header[1] = static_cast<int64_t>(npout_file);

    // Write as raw bytes (2 int64_t values = 16 bytes)
    if (partfile.Write_any_type(header, 2*sizeof(int64_t), "byte",
                                 single_file_per_rank) != 2*sizeof(int64_t)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "header not written correctly to particle restart file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  header_offset = 2 * sizeof(int64_t);

  // Calculate offsets for the shared MPI-IO file. Per-rank files store each rank's
  // particles contiguously starting immediately after the header.
  std::vector<int> rank_offset(global_variable::nranks, 0);
  int npout_min = npout_thisrank;
  if (!single_file_per_rank) {
    npout_min = pm->nprtcl_eachrank[0];
    for (int n=1; n<global_variable::nranks; ++n) {
      rank_offset[n] = rank_offset[n-1] + pm->nprtcl_eachrank[n-1];
      npout_min = std::min(npout_min, pm->nprtcl_eachrank[n]);
    }
  }

  // Allocate temporary array for output (all data as Real/double)
  double *rdata = new double[npout_thisrank];
  std::size_t real_datasize = sizeof(double);

  // Write GID data (as double)
  {
    for (int p=0; p<npout_thisrank; ++p) {
      rdata[p] = static_cast<double>(outpart_idata(PGID,p));
    }

    std::size_t myoffset = header_offset +
                            rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(rdata, npout_min, myoffset, "double",
                                       single_file_per_rank) != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "GID data not written correctly to particle restart file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(rdata + npout_min, nremain, myoffset, "double",
                                     single_file_per_rank) != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "GID data not written correctly to particle restart file"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_file*real_datasize;
  }

  // Write TAG data (as double)
  {
    for (int p=0; p<npout_thisrank; ++p) {
      rdata[p] = static_cast<double>(outpart_idata(PTAG,p));
    }

    std::size_t myoffset = header_offset +
                            rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(rdata, npout_min, myoffset, "double",
                                       single_file_per_rank) != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "TAG data not written correctly to particle restart file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(rdata + npout_min, nremain, myoffset, "double",
                                     single_file_per_rank) != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "TAG data not written correctly to particle restart file"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_file*real_datasize;
  }

  // Write PLASTMOVE data (as double) - preserves frozen/deleted particle status
  {
    for (int p=0; p<npout_thisrank; ++p) {
      rdata[p] = static_cast<double>(outpart_idata(PLASTMOVE,p));
    }

    std::size_t myoffset = header_offset +
                            rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(rdata, npout_min, myoffset, "double",
                                       single_file_per_rank) != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl
          << "PLASTMOVE data not written correctly to particle restart file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(rdata + npout_min, nremain, myoffset, "double",
                                     single_file_per_rank) != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "PLASTMOVE data not written correctly to particle restart file"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_file*real_datasize;
  }

  // Write X position data (real)
  {
    for (int p=0; p<npout_thisrank; ++p) {
      rdata[p] = static_cast<double>(outpart_rdata(IPX,p));
    }

    std::size_t myoffset = header_offset +
                            rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(rdata, npout_min, myoffset, "double",
                                       single_file_per_rank) != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "X position not written correctly to particle restart file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(rdata + npout_min, nremain, myoffset, "double",
                                     single_file_per_rank) != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "X position not written correctly to particle restart file"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_file*real_datasize;
  }

  // Write Y position data (real)
  {
    for (int p=0; p<npout_thisrank; ++p) {
      if (pm->multi_d) {
        rdata[p] = static_cast<double>(outpart_rdata(IPY,p));
      } else {
        rdata[p] = pm->mesh_size.x2min;
      }
    }

    std::size_t myoffset = header_offset +
                            rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(rdata, npout_min, myoffset, "double",
                                       single_file_per_rank) != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Y position not written correctly to particle restart file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(rdata + npout_min, nremain, myoffset, "double",
                                     single_file_per_rank) != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Y position not written correctly to particle restart file"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_file*real_datasize;
  }

  // Write Z position data (real)
  {
    for (int p=0; p<npout_thisrank; ++p) {
      if (pm->three_d) {
        rdata[p] = static_cast<double>(outpart_rdata(IPZ,p));
      } else {
        rdata[p] = pm->mesh_size.x3min;
      }
    }

    std::size_t myoffset = header_offset +
                            rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(rdata, npout_min, myoffset, "double",
                                       single_file_per_rank) != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Z position not written correctly to particle restart file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(rdata + npout_min, nremain, myoffset, "double",
                                     single_file_per_rank) != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Z position not written correctly to particle restart file"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // Close the output file and clean up
  partfile.Close(single_file_per_rank);
  delete[] rdata;

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  return;
}
