//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outputs.cpp
//! \brief implements Outputs class constructor
//!
//! The number and types of outputs are all controlled by the number and values of
//! parameters specified in <output[n]> blocks in the input file.  Each output block must
//! be labelled by a unique integer "n".  Following the convention of the parser
//! implemented in the ParameterInput class, a second output block with the same integer
//! "n" of an earlier block will silently overwrite the values read by the first block.
//! Numbering of output blocks does not need to be consecutive, and blocks may appear
//! in any order in the input file. A new output type will be created for each and every
//! <output[n]> block in the input file.
//!
//! Required parameters that must be specified in an <output[n]> block are:
//!   - variable  = [list of currently implemented strings for specifing output variables
//!                  is defined at start of outputs.hpp file]
//!   - file_type = tab,vtk,hst,bin,rst,prst,pbin
//!   - dt        = problem time between outputs
//!   - dt_wall   = wall time between outputs in seconds
//!
//! EXAMPLE of an <output[n]> block for a TAB dump:
//!   <output3>
//!   file_type   = tab       # Tabular data dump
//!   variable    = prim      # variables to be output
//!   data_format = %12.5e    # Optional data format string
//!   dt          = 0.01      # time increment between outputs
//!   slice_x2    = 0.0       # slice at x2
//!   slice_x3    = 0.0       # slice at x3
//!
//! Each <output[n]> block will result in a new node being created in a linked list of
//! BaseTypeOutput stored in the Outputs class.  During a simulation, outputs are made
//! when the simulation time satisfies the criteria implemented in the Driver class.
//!
//! To implement a new output type, write a new BaseTypeOutput derived class and construct
//! an object of this class in the Outputs constructor at the location indicated by the
//! comment text: 'NEW_OUTPUT_TYPES'.
//========================================================================================

#include <algorithm>  // std::find
#include <cmath>      // std::isfinite
#include <cstdio>
#include <cstdlib>
#include <cstring>    // strcmp
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>   // std::string, to_string()
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"
#include "structure_function.hpp"
#include "spectrum.hpp"

namespace {

//----------------------------------------------------------------------------------------
//! \fn void ParseRegionRestriction()
//  \brief reads optional per-axis spatial region-restriction bounds (x1_min/x1_max, etc.)
//  shared by the "pdf" and "prof" output types. Each bound independently defaults to the
//  full mesh extent (i.e. no restriction) when not specified in the input file, so the
//  per-cell mask check in the output kernels is always correct with no extra "enabled"
//  flags needed.

void ParseRegionRestriction(ParameterInput *pin, Mesh *pm, OutputParameters *opar) {
  opar->x1_min = pin->GetOrAddReal(opar->block_name, "x1_min", pm->mesh_size.x1min);
  opar->x1_max = pin->GetOrAddReal(opar->block_name, "x1_max", pm->mesh_size.x1max);
  opar->x2_min = pin->GetOrAddReal(opar->block_name, "x2_min", pm->mesh_size.x2min);
  opar->x2_max = pin->GetOrAddReal(opar->block_name, "x2_max", pm->mesh_size.x2max);
  opar->x3_min = pin->GetOrAddReal(opar->block_name, "x3_min", pm->mesh_size.x3min);
  opar->x3_max = pin->GetOrAddReal(opar->block_name, "x3_max", pm->mesh_size.x3max);
}

//----------------------------------------------------------------------------------------
//! \fn void ValidateBinConfig()
//  \brief shared validation for a "pdf" or "prof" output block's bin configuration,
//  called once per binned axis (the primary axis, and again for a PDF's optional second
//  axis). Catches bad input before any bin array is allocated or a step size computed,
//  rather than letting a zero/negative bin count or non-finite/inverted range reach
//  allocation, division, or a logarithm downstream.

void ValidateBinConfig(const std::string &block_name, const std::string &axis_label,
                        int nbin, Real bin_min, Real bin_max, bool logscale) {
  if (nbin <= 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output block '" << block_name << "' has " << axis_label
        << "nbin=" << nbin << " <= 0" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!std::isfinite(bin_min) || !std::isfinite(bin_max)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output block '" << block_name << "' has non-finite "
        << axis_label << "bin_min=" << bin_min << " or " << axis_label
        << "bin_max=" << bin_max << std::endl;
    exit(EXIT_FAILURE);
  }
  if (bin_min >= bin_max) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output block '" << block_name << "' has " << axis_label
        << "bin_min=" << bin_min << " >= " << axis_label << "bin_max=" << bin_max
        << std::endl;
    exit(EXIT_FAILURE);
  }
  if (logscale && bin_min <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << axis_label << "logscale is true but " << axis_label
        << "bin_min <= 0.0" << std::endl;
    exit(EXIT_FAILURE);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn std::vector<std::string> SplitOutputList()
//  \brief splits a comma-separated input value ("a, b, c") into trimmed, non-empty names

std::vector<std::string> SplitOutputList(const std::string &value) {
  std::vector<std::string> names;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::size_t first = item.find_first_not_of(" \t");
    if (first == std::string::npos) continue;
    std::size_t last = item.find_last_not_of(" \t");
    names.push_back(item.substr(first, last - first + 1));
  }
  return names;
}

//----------------------------------------------------------------------------------------
//! \fn Real GetOutputReal()
//  \brief real parameter with a default stored at full precision: GetOrAddReal writes
//  defaults with 6 significant digits, so a restart (which re-reads the parameter text)
//  would continue with e.g. pi/18 -> 0.174533 or 1048576 -> 1048580. The whole value
//  must be one number, finite unless allow_inf (never NaN); anything else is fatal.

Real GetOutputReal(ParameterInput *pin, const std::string &blk, const std::string &key,
                   double def, bool allow_inf) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.17g", def);
  const std::string s = pin->GetOrAddString(blk, key, b);
  char *end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  while (*end == ' ' || *end == '\t') ++end;
  if (end == s.c_str() || *end != '\0' || std::isnan(v) ||
      (!allow_inf && !std::isfinite(v))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output block '" << blk << "': " << key << " = '" << s
        << "' is not a " << (allow_inf ? "number" : "finite number") << std::endl;
    exit(EXIT_FAILURE);
  }
  return v;
}

//----------------------------------------------------------------------------------------
// Outputs constructor

Outputs::Outputs(ParameterInput *pin, Mesh *pm) {
  // loop over input block names.  Find those that start with "output", read parameters,
  // and add to linked list of BaseTypeOutputs.

  int num_hst=0, num_rst=0, num_log=0; // count # of hst,rst,log outputs
  std::vector<std::string> sf_spec_dirs;  // "<type>_<id>" of sf/spec outputs (unique)
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 6, "output") == 0) {
      OutputParameters opar;  // define temporary OutputParameters struct

      // extract integer number of output block.  Save name and number
      std::string outn = it->block_name.substr(6); // 6 because counting starts at 0!
      opar.block_number = atoi(outn.c_str());
      opar.block_name.assign(it->block_name);

      // set time of last output, time or cycles between outputs
      // when last_time < 0, then outputs will always be made
      opar.last_time = pin->GetOrAddReal(opar.block_name,"last_time", -1.0);
      if (pin->DoesParameterExist(opar.block_name,"dcycle")) {
        opar.dcycle = pin->GetInteger(opar.block_name,"dcycle");
        opar.dt = 0.0;
      } else if (pin->DoesParameterExist(opar.block_name,"dt")) {
        opar.dt = pin->GetReal(opar.block_name,"dt");
        opar.dcycle = 0;
      } else {
        opar.dt = 0.0;
        opar.dcycle = 0;
      }

      if (pin->DoesParameterExist(opar.block_name,"dt_wall")) {
        opar.dt_wall = pin->GetReal(opar.block_name,"dt_wall");
        opar.last_wall_time = 0.0;
      } else {
        opar.dt_wall=0.0;
      }
      // Only add output if at least one cadence is positive.
      if (opar.dcycle == 0 && opar.dt <= 0.0 && opar.dt_wall <= 0.0) continue;

      // set file number, basename, and format
      opar.file_number = pin->GetOrAddInteger(opar.block_name,"file_number",0);
      opar.file_basename = pin->GetString("job","basename");
      opar.file_type = pin->GetString(opar.block_name,"file_type");

      // set output variable and optional file id (default is output variable name)
      // but only for those output types that use them
      if (opar.file_type.compare("hst") != 0 &&
          opar.file_type.compare("rst") != 0 &&
          opar.file_type.compare("prst") != 0 &&
          opar.file_type.compare("pbin") != 0 &&
          opar.file_type.compare("log") != 0 &&
          opar.file_type.compare("trk") != 0 &&
          opar.file_type.compare("sf") != 0 &&
          opar.file_type.compare("spec") != 0) {
        opar.variable = pin->GetString(opar.block_name, "variable");
        opar.file_id = pin->GetOrAddString(opar.block_name,"id",opar.variable);
      }
      // pbin's variable is optional -- it falls back to its own particle-output
      // default field set when omitted (see bin_prtcl.cpp)
      if (opar.file_type.compare("pbin") == 0 &&
          pin->DoesParameterExist(opar.block_name, "variable")) {
        opar.variable = pin->GetString(opar.block_name, "variable");
        opar.file_id = pin->GetOrAddString(opar.block_name,"id",opar.variable);
      }

      // read ghost cell option
      opar.include_gzs = pin->GetOrAddBoolean(opar.block_name, "ghost_zones", false);

      // read MeshBlock ID (if specified)
      opar.gid = pin->GetOrAddInteger(opar.block_name, "gid", -1);
      if (opar.gid >= 0 && pm->nmb_total == 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Cannot specify MeshBlock ID in output block '"
            << opar.block_name << "' when there is only one" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (opar.gid > (pm->nmb_total-1)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "MeshBlock gid=" << opar.gid << " in output block '"
            << opar.block_name << "' exceeds total number of MeshBlocks" << std::endl;
        exit(EXIT_FAILURE);
      }

      // read slicing options.  Check that slice is within mesh
      if (pin->DoesParameterExist(opar.block_name,"slice_x1")) {
        Real x1 = pin->GetReal(opar.block_name,"slice_x1");
        if (x1 >= pm->mesh_size.x1min && x1 < pm->mesh_size.x1max) {
          opar.slice_x1 = x1;
          opar.slice1 = true;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x1=" << x1 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      } else {
        opar.slice1 = false;
      }

      if (pin->DoesParameterExist(opar.block_name,"slice_x2")) {
        Real x2 = pin->GetReal(opar.block_name,"slice_x2");
        if (x2 >= pm->mesh_size.x2min && x2 < pm->mesh_size.x2max) {
          opar.slice_x2 = x2;
          opar.slice2 = true;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x2=" << x2 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      } else {
        opar.slice2 = false;
      }

      if (pin->DoesParameterExist(opar.block_name,"slice_x3")) {
        Real x3 = pin->GetReal(opar.block_name,"slice_x3");
        if (x3 >= pm->mesh_size.x3min && x3 < pm->mesh_size.x3max) {
          opar.slice_x3 = x3;
          opar.slice3 = true;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Slice at x3=" << x3 << " in output block '"
              << opar.block_name << "' is out of range of Mesh" << std::endl;
          exit(EXIT_FAILURE);
        }
      } else {
        opar.slice3 = false;
      }

      // read ghost cell option
      opar.include_gzs = pin->GetOrAddBoolean(opar.block_name, "ghost_zones", false);

      // read MeshBlock ID (if specified)
      opar.gid = pin->GetOrAddInteger(opar.block_name, "gid", -1);
      if (opar.gid >= 0 && pm->nmb_total == 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Cannot specify MeshBlock ID in output block '"
            << opar.block_name << "' when there is only one" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (opar.gid > (pm->nmb_total-1)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "MeshBlock gid=" << opar.gid << " in output block '"
            << opar.block_name << "' exceeds total number of MeshBlocks" << std::endl;
        exit(EXIT_FAILURE);
      }

      // set output variable and optional file id (default is output variable name)
      if (opar.file_type.compare("hst") != 0 &&
          opar.file_type.compare("rst") != 0 &&
          opar.file_type.compare("prst") != 0 &&
          opar.file_type.compare("pbin") != 0 &&
          opar.file_type.compare("log") != 0 &&
          opar.file_type.compare("sf") != 0 &&
          opar.file_type.compare("spec") != 0) {
        opar.variable = pin->GetString(opar.block_name, "variable");
        opar.file_id = pin->GetOrAddString(opar.block_name,"id",opar.variable);
      }

      // check that pdf/prof variables are single variables
      // raise error if variable = mhd_w, mhd_u, hydro_w, hydro_u
      if (opar.file_type.compare("pdf") == 0 || opar.file_type.compare("prof") == 0) {
        if (opar.variable.compare("mhd_w") == 0 ||
            opar.variable.compare("mhd_u") == 0 ||
            opar.variable.compare("hydro_w") == 0 ||
            opar.variable.compare("hydro_u") == 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Output block '" << opar.block_name
              << "' cannot output variable '" << opar.variable << "'."
              << " The variable must be a single variable not a variable group"
              << std::endl;
          exit(EXIT_FAILURE);
        }
      }

      // set optional boolean to output only user-defined history variables
      if (opar.file_type.compare("hst") == 0) {
        opar.user_hist_only =pin->GetOrAddBoolean(opar.block_name,"user_hist_only",false);
        if (opar.user_hist_only && !(pm->pgen->user_hist)) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User-history file requested in output block '"
              << opar.block_name << "', but <problem>/user_hist is not true" << std::endl;
          exit(EXIT_FAILURE);
        }
      }

      // set optional data format string used in formatted writes
      opar.data_format = pin->GetOrAddString(opar.block_name, "data_format", "%12.5e");
      opar.data_format.insert(0, " "); // prepend with blank to separate columns

      // Construct new BaseTypeOutput according to file format
      // NEW_OUTPUT_TYPES: Add block to construct new types here
      BaseTypeOutput *pnode;
      if (opar.file_type.compare("tab") == 0) {
        pnode = new FormattedTableOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("hst") == 0) {
        pnode = new HistoryOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
        num_hst++;
      } else if (opar.file_type.compare("log") == 0) {
        pnode = new EventLogOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
        num_log++;
      } else if (opar.file_type.compare("vtk") == 0) {
        pnode = new MeshVTKOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("pvtk") == 0) {
        pnode = new ParticleVTKOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("prst") == 0) {
        opar.single_file_per_rank = pin->GetOrAddBoolean(opar.block_name,
          "single_file_per_rank", false);
        pnode = new ParticleRestartOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("pbin") == 0) {
        pnode = new ParticleBinaryOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("trk") == 0) {
        pnode = new TrackedParticleOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("cbin") == 0) {
        opar.single_file_per_rank = pin->GetOrAddBoolean(opar.block_name,
          "single_file_per_rank", false);
        opar.coarsen_factor = pin->GetInteger(opar.block_name,"coarsen_factor");
        opar.compute_moments = pin->GetOrAddBoolean(opar.block_name,
          "compute_moments", false);
        pnode = new CoarsenedBinaryOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("pdf") == 0) {
        opar.bin_min = pin->GetReal(opar.block_name,"bin_min");
        opar.bin_max = pin->GetReal(opar.block_name,"bin_max");
        opar.nbin = pin->GetInteger(opar.block_name,"nbin");
        opar.logscale = pin->GetOrAddBoolean(opar.block_name,"logscale",true);
        opar.mass_weighted = pin->GetOrAddBoolean(opar.block_name,"mass_weighted",false);
        ValidateBinConfig(opar.block_name, "", opar.nbin, opar.bin_min, opar.bin_max,
                          opar.logscale);
        // check and set second variable option.
        if (pin->DoesParameterExist(opar.block_name,"variable_2")) {
          opar.variable_2 = pin->GetString(opar.block_name, "variable_2");
          opar.bin2_min = pin->GetOrAddReal(opar.block_name,"bin2_min",0);
          opar.bin2_max = pin->GetOrAddReal(opar.block_name,"bin2_max",1);
          opar.nbin2 = pin->GetOrAddInteger(opar.block_name,"nbin2",0);
          opar.logscale2 = pin->GetOrAddBoolean(opar.block_name,"logscale2",true);
          ValidateBinConfig(opar.block_name, "2nd-axis ", opar.nbin2, opar.bin2_min,
                            opar.bin2_max, opar.logscale2);
        } else {
          opar.variable_2 = "";
          opar.bin2_min = 0;
          opar.bin2_max = 1;
          opar.nbin2 = 0;
          opar.logscale2 = true;
        }
        ParseRegionRestriction(pin, pm, &opar);
        pnode = new PDFOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("prof") == 0) {
        opar.bin_min = pin->GetReal(opar.block_name,"bin_min");
        opar.bin_max = pin->GetReal(opar.block_name,"bin_max");
        opar.nbin = pin->GetInteger(opar.block_name,"nbin");
        opar.logscale = pin->GetOrAddBoolean(opar.block_name,"logscale",true);
        opar.mass_weighted = pin->GetOrAddBoolean(opar.block_name,"mass_weighted",false);
        ValidateBinConfig(opar.block_name, "", opar.nbin, opar.bin_min, opar.bin_max,
                          opar.logscale);

        // coordinate to bin cells by: x1, x2, x3, or r (radial)
        opar.coord_axis = pin->GetString(opar.block_name,"coord_axis");
        if (opar.coord_axis.compare("x1") != 0 && opar.coord_axis.compare("x2") != 0 &&
            opar.coord_axis.compare("x3") != 0 && opar.coord_axis.compare("r") != 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Profile output block '" << opar.block_name
              << "' has invalid coord_axis='" << opar.coord_axis << "'."
              << " Must be one of x1, x2, x3, r" << std::endl;
          exit(EXIT_FAILURE);
        }

        if (opar.coord_axis.compare("r") == 0) {
          opar.radial_type =
              pin->GetOrAddString(opar.block_name,"radial_type","spherical");
          if (opar.radial_type.compare("spherical") != 0 &&
              opar.radial_type.compare("cylindrical") != 0) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Profile output block '" << opar.block_name
                << "' has invalid radial_type='" << opar.radial_type << "'."
                << " Must be one of spherical, cylindrical" << std::endl;
            exit(EXIT_FAILURE);
          }
          if (opar.radial_type.compare("cylindrical") == 0) {
            opar.cyl_axis = pin->GetOrAddString(opar.block_name,"cyl_axis","x3");
            if (opar.cyl_axis.compare("x1") != 0 && opar.cyl_axis.compare("x2") != 0 &&
                opar.cyl_axis.compare("x3") != 0) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Profile output block '" << opar.block_name
                  << "' has invalid cyl_axis='" << opar.cyl_axis << "'."
                  << " Must be one of x1, x2, x3" << std::endl;
              exit(EXIT_FAILURE);
            }
          }
          opar.xc = pin->GetOrAddReal(opar.block_name,"xc",0.0);
          opar.yc = pin->GetOrAddReal(opar.block_name,"yc",0.0);
          opar.zc = pin->GetOrAddReal(opar.block_name,"zc",0.0);
        }

        ParseRegionRestriction(pin, pm, &opar);
        pnode = new ProfileOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("sf") == 0 ||
                 opar.file_type.compare("spec") == 0) {
        // variables, then any non-unit weights, each loaded once into outvars
        std::vector<std::string> vars =
            SplitOutputList(pin->GetString(opar.block_name, "variables"));
        std::vector<std::string> wts =
            SplitOutputList(pin->GetOrAddString(opar.block_name, "weights", "ones"));
        opar.variable_list = vars;
        // sf variables "temperature" and "pressure" are computed by the sf output from
        // the primitives, so they are not loaded (nor are such select<n>_variable)
        auto sf_computed = [](const std::string &v) {
          return v.compare("temperature") == 0 || v.compare("pressure") == 0;
        };
        if (opar.file_type.compare("sf") == 0) {
          opar.variable_list.clear();
          for (const auto &v : vars) {
            if (!sf_computed(v)) opar.variable_list.push_back(v);
          }
        }
        for (const auto &w : wts) {
          if (w.compare("ones") != 0 &&
              std::find(opar.variable_list.begin(), opar.variable_list.end(), w) ==
              opar.variable_list.end()) {
            opar.variable_list.push_back(w);
          }
        }
        // sf cell selection: select<n>_variable fields are loaded too
        if (opar.file_type.compare("sf") == 0) {
          for (const char *key : {"select1_variable", "select2_variable"}) {
            if (!pin->DoesParameterExist(opar.block_name, key)) continue;
            std::string s = pin->GetString(opar.block_name, key);
            if (!sf_computed(s) &&
                std::find(opar.variable_list.begin(), opar.variable_list.end(), s) ==
                opar.variable_list.end()) {
              opar.variable_list.push_back(s);
            }
          }
        }
        // only computed fields: load the density so the output has a variable list
        if (opar.variable_list.empty() && !vars.empty()) {
          opar.variable_list.push_back(pin->DoesBlockExist("mhd") ? "mhd_w_d" :
                                                                    "hydro_w_d");
        }
        if (vars.empty()) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Output block '" << opar.block_name
              << "' (" << opar.file_type << ") needs a non-empty 'variables' list"
              << std::endl;
          exit(EXIT_FAILURE);
        }
        for (const auto &v : opar.variable_list) {
          if (v.compare("mhd_w") == 0 || v.compare("mhd_u") == 0 ||
              v.compare("hydro_w") == 0 || v.compare("hydro_u") == 0 ||
              v.compare("mhd_bcc") == 0) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Output block '" << opar.block_name << "' ("
                << opar.file_type << "): '"
                << v << "' is a variable group; list single variables" << std::endl;
            exit(EXIT_FAILURE);
          }
        }
        opar.variable = opar.variable_list[0];
        // default id = block name, so several blocks of one type never share a
        // directory; two blocks with the same explicit type and id are rejected
        opar.file_id = pin->GetOrAddString(opar.block_name, "id", opar.block_name);
        std::string dir = opar.file_type + "_" + opar.file_id;
        if (std::find(sf_spec_dirs.begin(), sf_spec_dirs.end(), dir) !=
            sf_spec_dirs.end()) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Output block '" << opar.block_name << "' writes to '"
              << dir << "/', which another " << opar.file_type
              << " output already uses; give it a distinct 'id'" << std::endl;
          exit(EXIT_FAILURE);
        }
        sf_spec_dirs.push_back(dir);
        // both types read the whole mesh themselves: the generic slice, MeshBlock-ID and
        // ghost-zone options would be silently ignored, so reject them
        if (opar.slice1 || opar.slice2 || opar.slice3 || opar.gid >= 0 ||
            opar.include_gzs) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Output block '" << opar.block_name << "' ("
              << opar.file_type << ") does not support slice_x1/2/3, gid or ghost_zones"
              << std::endl;
          exit(EXIT_FAILURE);
        }
        if (opar.file_type.compare("spec") == 0) {
          // spec transforms the whole mesh (z crop only, via strat/skip). A bound equal
          // to the mesh extent is accepted: restart files of earlier versions hold them.
          const char *bkey[6] = {"x1_min", "x1_max", "x2_min", "x2_max", "x3_min",
                                 "x3_max"};
          const Real bmesh[6] = {pm->mesh_size.x1min, pm->mesh_size.x1max,
                                 pm->mesh_size.x2min, pm->mesh_size.x2max,
                                 pm->mesh_size.x3min, pm->mesh_size.x3max};
          for (int b = 0; b < 6; ++b) {
            if (pin->DoesParameterExist(opar.block_name, bkey[b]) &&
                pin->GetReal(opar.block_name, bkey[b]) != bmesh[b]) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Output block '" << opar.block_name << "' (spec): "
                  << bkey[b] << " is not supported; spec transforms the whole mesh "
                  << "(use strat = true and skip to crop x3)" << std::endl;
              exit(EXIT_FAILURE);
            }
          }
          for (int f = 0; f < 6; ++f) {
            if (pm->mesh_bcs[f] == BoundaryFlag::shear_periodic) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Output block '" << opar.block_name << "' (spec): "
                  << "shearing-periodic boundaries are not supported" << std::endl;
              exit(EXIT_FAILURE);
            }
          }
        } else {
          ParseRegionRestriction(pin, pm, &opar);
        }
        if (opar.file_type.compare("sf") == 0) {
          pnode = new SFOutput(pin,pm,opar);
        } else {
          pnode = new SpectrumOutput(pin,pm,opar);
        }
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("bin") == 0) {
        opar.single_file_per_rank = pin->GetOrAddBoolean(opar.block_name,
          "single_file_per_rank", false);
        pnode = new MeshBinaryOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("cart") == 0) {
        pnode = new CartesianGridOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("sph") == 0) {
        pnode = new SphericalSurfaceOutput(pin,pm,opar);
        pout_list.insert(pout_list.begin(),pnode);
      } else if (opar.file_type.compare("rst") == 0) {
      // Add restarts to the tail end of BaseTypeOutput list, so file counters for other
      // output types are up-to-date in restart file
        opar.single_file_per_rank = pin->GetOrAddBoolean(opar.block_name,
          "single_file_per_rank", false);
        pnode = new RestartOutput(pin,pm,opar);
        pout_list.push_back(pnode);
        num_rst++;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Unrecognized file format = '" << opar.file_type
            << "' in output block '" << opar.block_name << "'" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  // check there were no more than one history, event log, or restart files requested
  if (num_hst > 1 || num_rst > 1 || num_log > 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "More than one history, event log, or restart output block found in "
              << "input file" << std::endl;
    exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// destructor

Outputs::~Outputs() {
  // Must manually delete memory assigned to each OutputType object stored in pout_list
  for (BaseTypeOutput* pnode : pout_list) {
    delete pnode;
  }
  pout_list.clear();
}
