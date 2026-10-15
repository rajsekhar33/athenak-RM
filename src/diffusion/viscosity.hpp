#ifndef DIFFUSION_VISCOSITY_HPP_
#define DIFFUSION_VISCOSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.hpp
//  \brief Contains data and functions that implement various formulations for
//  viscosity: isotropic Navier-Stokes shear and explicit uniform-grid MHD
//  Braginskii stress. Anisotropic integration is under development/validation.

#include <string>

#include "athena.hpp"
#include "diffusion/sts_types.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

//----------------------------------------------------------------------------------------
//! \class Viscosity
//  \brief data and functions that implement viscosity in Hydro and MHD

class Viscosity {
 public:
  Viscosity(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Viscosity();

  // data
  Real dtnew;
  Real nu_iso;      // coefficient of isotropic kinematic shear viscosity
  Real nu_aniso;    // coefficient of anisotropic kinematic shear viscosity
  parabolic::DiffusionSelection mode = parabolic::DiffusionSelection::explicit_only;

  // function to add viscous fluxes to Hydro and/or MHD fluxes
  void AddViscousFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                        DvceFaceFld5D<Real> &f);
  void AddViscousFluxIso(const DvceArray5D<Real> &w,const EOS_Data &eos,
                         DvceFaceFld5D<Real> &f);
  // Explicit fields prevent reading centered B from a different RK stage.
  // Explicit and RKL2 use the same uniform-grid operator; refinement is rejected.
  void AddViscousFluxAniso(const DvceArray5D<Real> &w, const EOS_Data &eos,
      const DvceFaceFld4D<Real> &b, const DvceArray5D<Real> &bcc,
      DvceFaceFld5D<Real> &f);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

  struct RawState {
    bool admissible;
    bool active_admissible;
  };
  // Inspect without repairing conserved variables or incrementing EOS counters.
  // With include_ghosts, cover the complete fine-grid EOS conversion extent.
  // For MHD, use current CT faces rather than potentially stale centered fields.
  RawState InspectRawState(const DvceArray5D<Real> &u, const EOS_Data &eos,
      const DvceFaceFld4D<Real> *b = nullptr, bool include_ghosts = false) const;

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_VISCOSITY_HPP_
