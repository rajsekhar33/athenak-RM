#ifndef DIFFUSION_CONDUCTION_HPP_
#define DIFFUSION_CONDUCTION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.hpp
//! \brief Contains data and functions that implement various formulations for conduction.
//  Constant isotropic and slope-limited anisotropic conduction are implemented.
//  Nonlinear STS uses checked, retryable RKL2 sweeps, alone or with other transport.

#include <string>

#include "athena.hpp"
#include "diffusion/sts_types.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class Conduction
//! \brief data and functions that implement thermal conduction in Hydro and MHD

class Conduction {
 public:
  Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Conduction();

  // data
  Real dtnew;
  Real dt_transport;  // CFL-free saturation donor bound; never STS-accelerated
  Real alpha_iso;       // isotropic thermal diffusivity
  Real alpha_aniso;     // anisotropic thermal diffusivity
  bool alpha_spitzer;   // switch to turn on Spitzer conductivity
  Real spitzer_kappa0_code = 0.0;  // K(theta) = K0 theta^(5/2), theta = p/rho
  bool spitzer_field_aligned = false;
  bool saturated = false;
  Real saturation_phi = 0.3;  // dimensionless Cowie-McKee uncertainty factor
  Real q_limit;              // legacy disabled sentinel, not a physical cap
  parabolic::DiffusionSelection mode = parabolic::DiffusionSelection::explicit_only;

  // functions
  void AddHeatFluxes(const DvceArray5D<Real> &w, const EOS_Data &eos,
                     DvceFaceFld5D<Real> &f);
  void AddHeatFluxIso(const DvceArray5D<Real> &w, const EOS_Data &eos,
                      DvceFaceFld5D<Real> &f);
  void AddHeatFluxAniso(const DvceArray5D<Real> &w, const EOS_Data &eos,
                        DvceFaceFld5D<Real> &f);
  void AddHeatFluxSpitzer(const DvceArray5D<Real> &w, const EOS_Data &eos,
                          DvceFaceFld5D<Real> &f);
  void AddHeatFluxSaturated(const DvceArray5D<Real> &w, const EOS_Data &eos,
                            DvceFaceFld5D<Real> &f);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);
  void CheckExplicitStage(const DvceArray5D<Real> &w, const EOS_Data &eos_data);
  void ValidateRawState(const DvceArray5D<Real> &u, const EOS_Data &eos_data,
                        const DvceFaceFld4D<Real> *b = nullptr) const;
  struct ThermalState {
    int invalid;
    Real minimum, maximum;
  };
  bool UsesNonlinearSTS() const;
  ThermalState InspectRawState(const DvceArray5D<Real> &u, const EOS_Data &eos_data,
                               const DvceFaceFld4D<Real> *b = nullptr,
                               bool active_only = false) const;
  void BeginSTSTrial(const DvceArray5D<Real> &u, const DvceArray5D<Real> &w,
                     DvceArray5D<Real> &snapshot);
  void RestoreSTSTrial(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
                       const DvceArray5D<Real> &snapshot);
  bool CheckSTSTransport(const DvceArray5D<Real> &w, const EOS_Data &eos_data,
                         Real interval) const;
  // Public because NVCC requires enclosing extended-lambda methods to be public.
  Real SpitzerTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data,
                       bool include_saturation = true, bool retryable = false) const;
  // Current raw-stage coefficient bound, using scratch primitives only.
  Real RawStageTimeStep(const DvceArray5D<Real> &u, const DvceArray5D<Real> &w,
                        const EOS_Data &eos_data, const DvceFaceFld4D<Real> *b,
                        bool include_ghosts);
  Real SaturationTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data,
                          bool donor_only = false) const;

 private:
  MeshBlockPack* pmy_pack;
  bool use_mc_limiter = false;
  DvceArray5D<Real> sts_w0;
  DvceArray5D<Real> sts_stage_w;
  void FailIfInvalid(int local_count, const char *where) const;
};
#endif // DIFFUSION_CONDUCTION_HPP_
