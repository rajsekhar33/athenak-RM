//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bin_prtcl.cpp
//! \brief writes particle data with grid quantities in binary format for analysis.
//! Binary format consists of:
//!   - Header: metadata (int64_t and double values)
//!   - Particle data arrays (all as double for consistency)
//! Data over multiple MeshBlocks and MPI ranks is written to a single file using MPI-IO.

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
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

ParticleBinaryOutput::ParticleBinaryOutput(ParameterInput *pin, Mesh *pm,
                                           OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output
  mkdir("pbin",0775);

  // Parse variable specification
  ParseVariableSpec(pin, pm);
}

//----------------------------------------------------------------------------------------
// ParseVariableSpec: Parse user-specified variables for output
// Supports predefined sets (hydro_w, mhd_w_bcc) and custom lists

void ParticleBinaryOutput::ParseVariableSpec(ParameterInput *pin, Mesh *pm) {
  bool is_mhd = (pm->pmb_pack->pmhd != nullptr);
  std::string variable = out_params.variable;

  // Trim whitespace from variable string
  variable.erase(0, variable.find_first_not_of(" \t"));
  variable.erase(variable.find_last_not_of(" \t") + 1);

  // Debug: print the detected variable
  if (global_variable::my_rank == 0) {
    std::cout << "Particle binary output: detected variable = '" << variable << "'"
              << std::endl;
  }

  var_names.clear();

  // Check for predefined variable sets
  if (variable.compare("hydro_w") == 0) {
    // All primitive variables: density, pressure, velocities
    var_names.push_back("dens");
    var_names.push_back("pres");
    var_names.push_back("velx");
    var_names.push_back("vely");
    var_names.push_back("velz");
  } else if (variable.compare("mhd_w") == 0) {
    // MHD primitives without B-field
    var_names.push_back("dens");
    var_names.push_back("pres");
    var_names.push_back("velx");
    var_names.push_back("vely");
    var_names.push_back("velz");
  } else if (variable.compare("mhd_w_bcc") == 0) {
    // MHD primitives with B-field
    var_names.push_back("dens");
    var_names.push_back("pres");
    var_names.push_back("velx");
    var_names.push_back("vely");
    var_names.push_back("velz");
    var_names.push_back("bx");
    var_names.push_back("by");
    var_names.push_back("bz");
  } else if (variable.compare("prtcl_minimal") == 0) {
    // Just density
    var_names.push_back("dens");
  } else if (variable.compare("prtcl_dens_temp") == 0) {
    // Density and temperature
    var_names.push_back("dens");
    var_names.push_back("temp");
  } else if (variable.compare("prtcl_dens_pres_temp") == 0) {
    // Density, pressure, temperature
    var_names.push_back("dens");
    var_names.push_back("pres");
    var_names.push_back("temp");
  } else if (variable.compare("prtcl_default") == 0 || variable.empty()) {
    // Default: density, eint/temp, velocities, B-field (if MHD)
    var_names.push_back("dens");
    var_names.push_back("eint_or_temp");
    var_names.push_back("velx");
    var_names.push_back("vely");
    var_names.push_back("velz");
    if (is_mhd) {
      var_names.push_back("bx");
      var_names.push_back("by");
      var_names.push_back("bz");
    }
  } else if (variable.compare("prtcl_custom") == 0) {
    // Custom list - read from prtcl_vars parameter
    std::string custom_vars = pin->GetString(out_params.block_name, "prtcl_vars");

    // Parse comma-separated list
    std::stringstream ss(custom_vars);
    std::string varname;
    while (std::getline(ss, varname, ',')) {
      // Trim whitespace
      varname.erase(0, varname.find_first_not_of(" \t"));
      varname.erase(varname.find_last_not_of(" \t") + 1);
      if (!varname.empty()) {
        var_names.push_back(varname);
      }
    }
  } else {
    // Treat as single variable name
    var_names.push_back(variable);
  }

  ngriddata = var_names.size();

  // Print info about selected variables
  if (global_variable::my_rank == 0) {
    std::cout << "Particle binary output '" << out_params.block_name
              << "' will include " << ngriddata << " grid variables:" << std::endl;
    for (int i = 0; i < ngriddata; ++i) {
      std::cout << "  [" << i << "] " << var_names[i] << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
// ParticleBinaryOutput::LoadOutputData()
// Copies particle data and interpolates grid data to particle locations

void ParticleBinaryOutput::LoadOutputData(Mesh *pm) {
  particles::Particles *pp = pm->pmb_pack->ppart;
  npout_thisrank = pm->nprtcl_thisrank;
  npout_total = pm->nprtcl_total;

  Kokkos::realloc(outpart_rdata, pp->nrdata, npout_thisrank);
  Kokkos::realloc(outpart_idata, pp->nidata, npout_thisrank);

  // Copy particle data directly from device to host
  Kokkos::deep_copy(outpart_rdata, pp->prtcl_rdata);
  Kokkos::deep_copy(outpart_idata, pp->prtcl_idata);

  // Allocate grid data array based on number of requested variables
  Kokkos::realloc(outpart_griddata, ngriddata, npout_thisrank);

  // Create device view for grid data interpolation
  auto d_outpart_griddata = Kokkos::create_mirror_view(DevMemSpace(),
                                                       outpart_griddata);

  // Get grid data
  auto &pr = pp->prtcl_rdata;
  auto &pi = pp->prtcl_idata;
  auto &mbsize = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int nmb = pm->pmb_pack->nmb_thispack;

  // Get primitive variables (w0) and B-field (bcc0) from hydro or MHD
  bool is_mhd = (pm->pmb_pack->pmhd != nullptr);
  DvceArray5D<Real> w0, bcc0;
  if (is_mhd) {
    w0 = pm->pmb_pack->pmhd->w0;
    bcc0 = pm->pmb_pack->pmhd->bcc0;
  } else {
    w0 = pm->pmb_pack->phydro->w0;
  }

  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;

  // Create device-accessible array of variable type codes for device kernel
  // Use negative values to distinguish special variables from athena.hpp indices
  Kokkos::View<int*, Kokkos::HostSpace> h_var_codes("var_codes_host", ngriddata);
  for (int v=0; v<ngriddata; ++v) {
    const std::string &varname = var_names[v];
    if (varname == "dens") {
      h_var_codes(v) = IDN;  // Use athena.hpp index
    } else if (varname == "pres" || varname == "eint") {
      h_var_codes(v) = IEN;  // IEN holds eint
    } else if (varname == "temp") {
      h_var_codes(v) = -1;  // Special: pressure/density
    } else if (varname == "eint_or_temp") {
      h_var_codes(v) = -2;  // Special: eint
    } else if (varname == "velx") {
      h_var_codes(v) = IVX;  // Use athena.hpp index
    } else if (varname == "vely") {
      h_var_codes(v) = IVY;
    } else if (varname == "velz") {
      h_var_codes(v) = IVZ;
    } else if (varname == "bx") {
      h_var_codes(v) = IBX;  // Use athena.hpp index
    } else if (varname == "by") {
      h_var_codes(v) = IBY;
    } else if (varname == "bz") {
      h_var_codes(v) = IBZ;
    } else if (varname == "bmag") {
      h_var_codes(v) = -3;  // Special: sqrt(bx^2 + by^2 + bz^2)
    } else {
      h_var_codes(v) = -999;  // Unknown variable
    }
  }

  // Copy variable codes to device
  auto var_codes = Kokkos::create_mirror_view_and_copy(DevMemSpace(), h_var_codes);
  int n_vars = ngriddata;

  // Capture mb_gid for device kernel
  auto &mb_gid = pm->pmb_pack->pmb->mb_gid;

  // Interpolate grid data to particle positions (all on device)
  par_for("interpolate_grid_bin", DevExeSpace(), 0, (npout_thisrank-1),
  KOKKOS_LAMBDA(const int p) {
    // Get particle position and MeshBlock ID
    Real x1 = pr(IPX, p);
    Real x2 = pr(IPY, p);
    Real x3 = pr(IPZ, p);
    int pgid = pi(PGID, p);

    // Find MeshBlock containing this particle
    int m = -1;
    for (int mb=0; mb<nmb; ++mb) {
      if (mb_gid.d_view(mb) == pgid) {
        m = mb;
        break;
      }
    }

    if (m >= 0) {
      // Get MeshBlock bounds
      Real &x1min = mbsize.d_view(m).x1min;
      Real &x1max = mbsize.d_view(m).x1max;
      Real &x2min = mbsize.d_view(m).x2min;
      Real &x2max = mbsize.d_view(m).x2max;
      Real &x3min = mbsize.d_view(m).x3min;
      Real &x3max = mbsize.d_view(m).x3max;

      // Find cell indices containing particle
      int i = CellCenterIndex(x1, indcs.nx1, x1min, x1max) + is;
      int j = CellCenterIndex(x2, indcs.nx2, x2min, x2max) + js;
      int k = CellCenterIndex(x3, indcs.nx3, x3min, x3max) + ks;

      // Clamp indices to valid range
      i = (i < is) ? is : ((i >= is+indcs.nx1) ? is+indcs.nx1-1 : i);
      j = (j < js) ? js : ((j >= js+indcs.nx2) ? js+indcs.nx2-1 : j);
      k = (k < ks) ? ks : ((k >= ks+indcs.nx3) ? ks+indcs.nx3-1 : k);

      // Get commonly needed grid values
      Real density = w0(m, IDN, k, j, i);
      Real pressure = w0(m, IEN, k, j, i);

      // Compute each requested variable
      for (int v=0; v<n_vars; ++v) {
        Real value = 0.0;
        int code = var_codes(v);

        // Positive codes: direct indices from athena.hpp (IDN, IEN, IVX, etc.)
        // Negative codes: derived quantities requiring computation
        if (code >= 0) {
          // Direct access to w0 or bcc0 arrays using athena.hpp indices
          if (code == IDN) {
            value = density;
          } else if (code == IEN) {
            value = pressure;  // IEN holds eint
          } else if (code == IVX || code == IVY || code == IVZ) {
            value = w0(m, code, k, j, i);
          } else if (code == IBX || code == IBY || code == IBZ) {
            if (is_mhd) value = bcc0(m, code, k, j, i);
          }
        } else {
          // Derived quantities
          if (code == -1) {  // temp
            value = pressure / density;
          } else if (code == -2) {  // eint_or_temp
            value = pressure;  // always eint
          } else if (code == -3) {  // bmag
            if (is_mhd) {
              Real bx = bcc0(m, IBX, k, j, i);
              Real by = bcc0(m, IBY, k, j, i);
              Real bz = bcc0(m, IBZ, k, j, i);
              value = sqrt(bx*bx + by*by + bz*bz);
            }
          }
        }

        d_outpart_griddata(v, p) = value;
      }
    } else {
      // Particle not found in local MeshBlocks, set to zero
      for (int v=0; v<n_vars; ++v) {
        d_outpart_griddata(v, p) = 0.0;
      }
    }
  });

  // Copy grid data from device to host
  Kokkos::deep_copy(outpart_griddata, d_outpart_griddata);
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleBinaryOutput::WriteOutputFile(Mesh *pm)
//! \brief Writes particle data with grid quantities in binary format.
//!
//! Binary file format:
//!  1. Header (variable size):
//!     - int64_t magic_number = 43  (identifies binary particle data format)
//!     - int64_t total_particles    (total number of particles across all ranks)
//!     - int64_t nrdata            (number of real data fields per particle)
//!     - int64_t nidata            (number of integer data fields per particle)
//!     - int64_t ngriddata         (number of grid data fields per particle)
//!     - double time               (simulation time)
//!     - double dt                 (timestep)
//!     - int64_t ncycle            (cycle number)
//!     - char[16] var_names[ngriddata]  (variable names, 16 chars each, null-padded)
//!  2. Particle real data (nrdata arrays of nprtcl doubles each):
//!     - x, y, z positions (for all particles)
//!     - vx, vy, vz velocities (for cosmic rays, nrdata=6)
//!     - interpolated velocities (for Lagrangian MC, if requested in grid variables)
//!  3. Particle integer data (nidata arrays of nprtcl int32_t each):
//!     - gid, tag, etc.
//!  4. Grid data at particle locations (ngriddata arrays of nprtcl doubles each):
//!     - user-specified variables (see var_names)

void ParticleBinaryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // create filename: "pbin/file_basename"."file_id"."XXXXX".prtclbin
  std::string fname;
  char number[6];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);

  fname.assign("pbin/");
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
  fname.append(".prtclbin");

  IOWrapper partfile;
  std::size_t header_offset = 0;
  partfile.Open(fname.c_str(), IOWrapper::FileMode::write);

  particles::Particles *pp = pm->pmb_pack->ppart;

  // Write header (variable size: 64 bytes + 16*ngriddata bytes)
  if (global_variable::my_rank == 0) {
    int64_t header_int[5];
    header_int[0] = 43;  // magic number for binary particle data
    header_int[1] = static_cast<int64_t>(npout_total);
    header_int[2] = static_cast<int64_t>(pp->nrdata);
    header_int[3] = static_cast<int64_t>(pp->nidata);
    header_int[4] = static_cast<int64_t>(ngriddata);

    double header_real[3];
    header_real[0] = pm->time;
    header_real[1] = pm->dt;
    header_real[2] = static_cast<double>(pm->ncycle);

    // Write integer header (5 int64_t = 40 bytes)
    if (partfile.Write_any_type(header_int, 5*sizeof(int64_t), "byte")
        != 5*sizeof(int64_t)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "header not written correctly to particle binary file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Write real header (3 double = 24 bytes)
    if (partfile.Write_any_type(header_real, 3*sizeof(double), "byte")
        != 3*sizeof(double)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "header not written correctly to particle binary file"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Write variable names (16 bytes each, null-padded)
    for (int i = 0; i < ngriddata; ++i) {
      char varname[16];
      std::memset(varname, 0, 16);
      std::strncpy(varname, var_names[i].c_str(), 15);  // Leave space for null terminator
      if (partfile.Write_any_type(varname, 16, "byte") != 16) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "variable name not written correctly to particle binary file"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  // 64 + 16*ngriddata bytes
  header_offset = 5*sizeof(int64_t) + 3*sizeof(double) + 16*ngriddata;

  // Calculate offset for each rank's data
  std::vector<int> rank_offset(global_variable::nranks, 0);
  int npout_min = pm->nprtcl_eachrank[0];
  for (int n=1; n<global_variable::nranks; ++n) {
    rank_offset[n] = rank_offset[n-1] + pm->nprtcl_eachrank[n-1];
    npout_min = std::min(npout_min, pm->nprtcl_eachrank[n]);
  }

  std::size_t real_datasize = sizeof(double);
  std::size_t int_datasize = sizeof(int32_t);

  // Allocate temporary arrays
  double *rdata = new double[npout_thisrank];
  int32_t *idata = new int32_t[npout_thisrank];

  // Write particle real data (positions and stored velocities from prtcl_rdata)
  for (int n=0; n<pp->nrdata; ++n) {
    for (int p=0; p<npout_thisrank; ++p) {
      rdata[p] = static_cast<double>(outpart_rdata(n,p));
    }

    std::size_t myoffset = header_offset +
                           rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(&(rdata[0]), npout_min, myoffset, "double")
          != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "rdata[" << n << "] not written correctly"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(&(rdata[npout_min]), nremain, myoffset, "double")
            != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "rdata[" << n << "] not written correctly"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_total*real_datasize;
  }

  // Write particle integer data (gid, tag, etc.)
  for (int n=0; n<pp->nidata; ++n) {
    for (int p=0; p<npout_thisrank; ++p) {
      idata[p] = static_cast<int32_t>(outpart_idata(n,p));
    }

    std::size_t myoffset = header_offset +
                           rank_offset[global_variable::my_rank]*int_datasize;

    if (partfile.Write_any_type_at_all(&(idata[0]), npout_min, myoffset, "int")
          != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "idata[" << n << "] not written correctly"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += int_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(&(idata[npout_min]), nremain, myoffset, "int")
            != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "idata[" << n << "] not written correctly"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_total*int_datasize;
  }

  // Write grid data at particle locations
  for (int n=0; n<ngriddata; ++n) {
    for (int p=0; p<npout_thisrank; ++p) {
      rdata[p] = static_cast<double>(outpart_griddata(n,p));
    }

    std::size_t myoffset = header_offset +
                           rank_offset[global_variable::my_rank]*real_datasize;

    if (partfile.Write_any_type_at_all(&(rdata[0]), npout_min, myoffset, "double")
          != npout_min) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "griddata[" << n << "] not written correctly"
          << std::endl;
      std::exit(EXIT_FAILURE);
    }

    myoffset += real_datasize*npout_min;
    int nremain = npout_thisrank - npout_min;
    if (nremain > 0) {
      if (partfile.Write_any_type_at(&(rdata[npout_min]), nremain, myoffset, "double")
            != nremain) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "griddata[" << n << "] not written correctly"
            << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    header_offset += npout_total*real_datasize;
  }

  // close the output file and clean up
  partfile.Close();
  delete[] rdata;
  delete[] idata;

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
