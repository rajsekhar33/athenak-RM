#ifndef OUTPUTS_AREA_HPP_
#define OUTPUTS_AREA_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file area.hpp
//! \brief Multiscale isosurface areas of an output scalar on a uniform Cartesian mesh.

#include <memory>

#include "outputs.hpp"

struct AreaOutputState;

class AreaOutput : public BaseTypeOutput {
 public:
  AreaOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  ~AreaOutput();
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;

 private:
  std::unique_ptr<AreaOutputState> state_;
};

#endif  // OUTPUTS_AREA_HPP_
