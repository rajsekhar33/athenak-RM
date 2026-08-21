//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file profile.cpp
//  \brief writes on-the-fly spatial profile output data
//  Bins cells by a spatial coordinate (x1, x2, x3, or radius r, either spherical or
//  cylindrical about a user-specified center) and accumulates the volume- or
//  mass-weighted mean of a single user-specified variable in each bin. This reuses the
//  same ScatterView-based binning/reduction machinery as the PDF output (pdf.cpp), but
//  bins by a cell's coordinate rather than by the value of the profiled variable itself.
//
//  Unlike PDF output (which stores raw weighted counts and leaves normalization to
//  post-processing), the profile's mean is computed here, host-side, right after the
//  MPI reduce, so the written file is immediately plot-ready.

#include <sys/stat.h>  // mkdir

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/z4c.hpp"
#include "outputs.hpp"

// ScatterView is not part of Kokkos core interface
#include "Kokkos_ScatterView.hpp"

//----------------------------------------------------------------------------------------
// Constructor: also calls BaseTypeOutput base class constructor

ProfileOutput::ProfileOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op), profile_data(op.nbin) {
  // create directory for this profile's output
  std::string dir_name;
  dir_name.assign("prof_");
  dir_name.append(op.file_id);
  mkdir(dir_name.c_str(), 0775);

  profile_data.mass_weighted = op.mass_weighted;
  profile_data.logscale = op.logscale;

  // throw an error if the user tries to use logscale with a negative bin_min
  if (op.logscale && op.bin_min <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "logscale is true but bin_min <= 0.0" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Create bin edges for the profile coordinate. Create mirror view on host, populate,
  // then copy back to device.
  auto bins_host = Kokkos::create_mirror_view(profile_data.bins);
  if (op.logscale) {
    Real logbin_min = std::log10(op.bin_min);
    Real logbin_max = std::log10(op.bin_max);
    for (int i = 0; i <= op.nbin; i++) {
      bins_host(i) = std::pow(10, logbin_min + i * (logbin_max - logbin_min) / op.nbin);
    }
  } else {
    Real bin_step = (op.bin_max - op.bin_min) / op.nbin;
    for (int i = 0; i <= op.nbin; i++) {
      bins_host(i) = op.bin_min + i * bin_step;
    }
  }
  Kokkos::deep_copy(profile_data.bins, bins_host);
  Kokkos::fence();

  profile_data.step_size = op.logscale ?
                       (std::log10(op.bin_max) - std::log10(op.bin_min)) / op.nbin :
                       (op.bin_max - op.bin_min) / op.nbin;

  // Resolve the coordinate/radial-type/axis strings into enums once, here, so the
  // per-cell kernel in LoadOutputData never does string comparisons.
  if (op.coord_axis.compare("x1") == 0) {
    profile_data.coord_mode = 0;
  } else if (op.coord_axis.compare("x2") == 0) {
    profile_data.coord_mode = 1;
  } else if (op.coord_axis.compare("x3") == 0) {
    profile_data.coord_mode = 2;
  } else {
    profile_data.coord_mode = 3;  // "r", validated in outputs.cpp
  }

  if (profile_data.coord_mode == 3) {
    profile_data.cylindrical = (op.radial_type.compare("cylindrical") == 0);
    if (profile_data.cylindrical) {
      if (op.cyl_axis.compare("x1") == 0) {
        profile_data.cyl_axis_mode = 0;
      } else if (op.cyl_axis.compare("x2") == 0) {
        profile_data.cyl_axis_mode = 1;
      } else {
        profile_data.cyl_axis_mode = 2;
      }
    }
    profile_data.xc = op.xc;
    profile_data.yc = op.yc;
    profile_data.zc = op.zc;
  }

  profile_data.result_ = DvceArray2D<Real>("result", 2, op.nbin+2);
  profile_data.scatter_result = Kokkos::Experimental::ScatterView<Real **, LayoutWrapper>(
    profile_data.result_
  );
}

//----------------------------------------------------------------------------------------
//! \fn void ProfileOutput::LoadOutputData()
//  \brief bins cells by the chosen coordinate and accumulates weighted sums of the
//  profiled variable and of the weight itself, in each bin.

void ProfileOutput::LoadOutputData(Mesh *pm) {
  // Calculate derived variable, if required
  if (out_params.contains_derived) {
    ComputeDerivedVariable(out_params.variable, pm);
  }

  // Pointer for initial determination
  DvceArray5D<Real> *u0_ptr = nullptr;

  if (pm->pmb_pack->phydro != nullptr) {
    u0_ptr = &(pm->pmb_pack->phydro->u0);
  } else if (pm->pmb_pack->pmhd != nullptr) {
    u0_ptr = &(pm->pmb_pack->pmhd->u0);
  } else if (pm->pmb_pack->pz4c != nullptr) {
    u0_ptr = &(pm->pmb_pack->pz4c->u0);
  }

  if (u0_ptr == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "No physics module found" << std::endl;
    exit(EXIT_FAILURE);
  }
  DvceArray5D<Real> &u0_ = *u0_ptr;

  // capture class variables for kernel
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;

  auto result  = profile_data.result_;
  auto scatter = profile_data.scatter_result;

  int nmb = pm->pmb_pack->nmb_thispack;
  int nx1 = indcs.nx1 + 2*indcs.ng;
  int nx2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
  int nx3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;

  // Copy MeshBlock data (the single profiled variable) from host to device
  DvceArray5D<Real> outvars_device("outvars_device", outvars.size(), nmb, nx3, nx2, nx1);
  for (std::size_t i = 0; i < outvars.size(); ++i) {
    auto d_slice = Kokkos::subview(*(outvars[i].data_ptr),
      Kokkos::ALL(), outvars[i].data_index, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    auto d_target_slice = Kokkos::subview(outvars_device, i,
      Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
    Kokkos::deep_copy(d_target_slice, d_slice);
  }
  Kokkos::fence();

  // Reset ScatterView and result array from previous output
  scatter.reset();
  Kokkos::deep_copy(result, 0);
  Kokkos::fence();

  // Capture the necessary data from profile_data
  auto bins = profile_data.bins;
  Real step_size = profile_data.step_size;
  int nbin_ = profile_data.nbin;
  bool logscale = profile_data.logscale;
  bool mass_weighted = profile_data.mass_weighted;
  int coord_mode = profile_data.coord_mode;
  bool cylindrical = profile_data.cylindrical;
  int cyl_axis_mode = profile_data.cyl_axis_mode;
  Real xc = profile_data.xc;
  Real yc = profile_data.yc;
  Real zc = profile_data.zc;

  // Unpack region-restriction bounds into plain locals before the kernel: out_params
  // (an OutputParameters, which holds std::string members) must never be captured into
  // a KOKKOS_LAMBDA.
  Real x1_min = out_params.x1_min; Real x1_max = out_params.x1_max;
  Real x2_min = out_params.x2_min; Real x2_max = out_params.x2_max;
  Real x3_min = out_params.x3_min; Real x3_max = out_params.x3_max;

  par_for("prof", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1min = size.d_view(m).x1min; Real x1max = size.d_view(m).x1max;
    Real x2min = size.d_view(m).x2min; Real x2max = size.d_view(m).x2max;
    Real x3min = size.d_view(m).x3min; Real x3max = size.d_view(m).x3max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // spatial region restriction: no-op when the user didn't set any bounds, since
    // they then default to the full mesh extent
    if (x1v < x1_min || x1v > x1_max ||
        x2v < x2_min || x2v > x2_max ||
        x3v < x3_min || x3v > x3_max) {
      return;
    }

    // compute the coordinate value to bin by. This is a uniform branch (every thread
    // takes the same path, since coord_mode is fixed for the whole kernel launch), so
    // there is no warp-divergence cost.
    Real coord_val;
    if (coord_mode == 0) {
      coord_val = x1v;
    } else if (coord_mode == 1) {
      coord_val = x2v;
    } else if (coord_mode == 2) {
      coord_val = x3v;
    } else {
      if (!cylindrical) {
        coord_val = sqrt(SQR(x1v-xc) + SQR(x2v-yc) + SQR(x3v-zc));
      } else if (cyl_axis_mode == 0) {
        coord_val = sqrt(SQR(x2v-yc) + SQR(x3v-zc));
      } else if (cyl_axis_mode == 1) {
        coord_val = sqrt(SQR(x1v-xc) + SQR(x3v-zc));
      } else {
        coord_val = sqrt(SQR(x1v-xc) + SQR(x2v-yc));
      }
    }

    // bin lookup: same three-way (underflow/overflow/interior) structure as PDF output
    int bin = -1;
    if (coord_val < bins(0)) {
      bin = 0;
    } else if (coord_val >= bins(nbin_)) {
      bin = nbin_ + 1;
    } else if (!logscale) {
      bin = static_cast<int>((coord_val - bins(0)) / step_size) + 1;
    } else {
      bin = static_cast<int>(std::log10(coord_val / bins(0)) / step_size) + 1;
    }

    Real var_val = outvars_device(0, m, k, j, i);
    Real weight = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
    weight *= mass_weighted == false ? 1.0 : u0_(m, IDN, k, j, i);

    auto res = scatter.access();
    res(0, bin) += weight;
    res(1, bin) += weight*var_val;
  });

  // "reduce" results from scatter view to original view.
  Kokkos::Experimental::contribute(result, scatter);
  Kokkos::fence();

  // Now reduce over ranks. result is a DvceArray2D (device/GPU memory) -- MPI_Reduce
  // must not be called directly on its pointer, since this MPI build/collective path
  // is not reliably CUDA-aware across nodes (confirmed via crash: passing the device
  // pointer directly segfaults deep inside the MPI collective's internal buffer
  // handling, both with HCOLL and with OpenMPI's AVX-optimized reduction operator,
  // since both treat it as ordinary host memory). Reduce through a host mirror
  // instead, then copy the (root-rank-correct) result back to the device array so
  // WriteOutputFile's own mirror+deep_copy of result_ picks up the reduced values.
#if MPI_PARALLEL_ENABLED
  auto result_host = Kokkos::create_mirror_view(result);
  Kokkos::deep_copy(result_host, result);
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, result_host.data(), result_host.size(),
                                   MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(result_host.data(), result_host.data(), result_host.size(),
                                   MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  Kokkos::deep_copy(result, result_host);
#endif
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void WriteHeaderWithOffset()
//  \brief writes an ASCII metadata block (already newline-terminated lines
//  concatenated in `meta`), followed by a "header offset=<N>" line giving the exact
//  byte at which the raw binary payload below it begins. Follows the same
//  ASCII-header-plus-binary-payload convention used elsewhere in this codebase (see
//  src/outputs/binary.cpp), so a reader just scans lines until it sees this key,
//  seeks to that byte, and reads raw native-endian Reals from there.

void WriteHeaderWithOffset(FILE *pfile, const std::string &meta) {
  std::fwrite(meta.data(), 1, meta.size(), pfile);
  char dummy[64];
  int line_len = std::snprintf(dummy, sizeof(dummy), "header offset=%012ld\n", 0L);
  long header_offset = std::ftell(pfile) + line_len;
  std::fprintf(pfile, "header offset=%012ld\n", header_offset);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProfileOutput::WriteOutputFile()
//  \brief writes the bin-edges header file (once, binary) and one binary profile file
//  per output dump. See PDFOutput::WriteOutputFile for why binary (not ASCII) is used.

void ProfileOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // only the master rank writes the file
  if (global_variable::my_rank == 0) {
    // Write header, if it has not been written already
    if (!(profile_data.bins_written)) {
      std::string fname;
      fname.assign("prof_");
      fname.append(out_params.file_id);
      fname.append("/");
      fname.append(out_params.file_basename);
      fname.append(".bins.prof");

      FILE *pfile;
      if ((pfile = std::fopen(fname.c_str(),"wb")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
        exit(EXIT_FAILURE);
      }

      std::string meta;
      meta += "Athena profile bins version=1.0\n";
      meta += "size of Real=" + std::to_string(sizeof(Real)) + "\n";
      meta += "coord_axis=" + out_params.coord_axis + "\n";
      meta += "variable=" + outvars[0].label + "\n";
      meta += "nbin=" + std::to_string(profile_data.nbin) + "\n";
      meta += "logscale=" + std::to_string(profile_data.logscale ? 1 : 0) + "\n";
      WriteHeaderWithOffset(pfile, meta);

      auto bins_host = Kokkos::create_mirror_view(profile_data.bins);
      Kokkos::deep_copy(bins_host, profile_data.bins);
      Kokkos::fence();
      std::fwrite(bins_host.data(), sizeof(Real), profile_data.nbin+1, pfile);
      std::fclose(pfile);
      profile_data.bins_written = true;
    }

    // create filename: "prof_"+"file_id"/file_basename" + "." + XXXXX + ".prof"
    std::string fname;
    char number[6];
    std::snprintf(number, sizeof(number), "%05d", out_params.file_number);
    fname.assign("prof_");
    fname.append(out_params.file_id);
    fname.append("/");
    fname.append(out_params.file_basename);
    fname.append(".");
    fname.append(number);
    fname.append(".prof");

    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"wb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
    }

    auto result_host = Kokkos::create_mirror_view(profile_data.result_);
    Kokkos::deep_copy(result_host, profile_data.result_);

    int nbin_tot = profile_data.nbin+2;

    char time_str[32];
    std::snprintf(time_str, sizeof(time_str), "%.17g", pm->time);
    std::string meta;
    meta += "time=" + std::string(time_str) + "\n";
    meta += "nbin=" + std::to_string(profile_data.nbin) + "\n";
    meta += "rows=sum_weight,sum_weight_times_var,mean\n";
    WriteHeaderWithOffset(pfile, meta);

    // row 0: sum of weight per bin, row 1: sum of weight*variable per bin. result_ has
    // LayoutRight (row-major), so these two rows are one contiguous 2*nbin_tot block.
    std::fwrite(result_host.data(), sizeof(Real), 2*nbin_tot, pfile);

    // row 2: weighted mean per bin, computed here (host-side, post-MPI-reduce), with a
    // 0.0 fallback for empty bins so the mean is always well-defined
    std::vector<Real> mean_row(nbin_tot);
    for (int n=0; n<nbin_tot; ++n) {
      Real sum_w = result_host(0, n);
      mean_row[n] = (sum_w > 0.0) ? result_host(1, n)/sum_w : 0.0;
    }
    std::fwrite(mean_row.data(), sizeof(Real), nbin_tot, pfile);
    std::fclose(pfile);
  }

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
