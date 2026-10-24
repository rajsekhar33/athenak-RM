#ifndef DIFFUSION_RESISTIVITY_HPP_
#define DIFFUSION_RESISTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.hpp
//  \brief Contains data and functions that implement various non-ideal MHD (resistive)
//  processes, such as Ohmic diffusion and ambipolar diffusion. TODO(@user): Hall effect.
//
//  In <mhd>, eta_ohm enables constant Ohmic diffusion, eta_ad enables ambipolar
//  diffusion, and eta_spitzer selects temperature-dependent Ohmic diffusion.
//  The Spitzer normalization is supplied in code units or converted from CGS.

#include "athena.hpp"
#include "diffusion/sts_types.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"

namespace resistivity {
// eta_m = eta0 T^(-3/2); T is p/rho in code units.
Real SpitzerCoefficientCode(Real eta0_cgs, Real length_cgs, Real time_cgs,
                            Real temperature_cgs);
} // namespace resistivity

//----------------------------------------------------------------------------------------
//! \class Resistivity
//  \brief data and functions that implement various non-ideal MHD physics (Ohmic
//  resistivity and ambipolar diffusion)

class Resistivity {
 public:
  Resistivity(MeshBlockPack *pp, ParameterInput *pin);
  ~Resistivity();

  // data
  Real dtnew;
  Real eta_ohm;   // Ohmic resistivity coefficient (0 => Ohmic term off)
  Real eta_ad;    // ambipolar diffusion coefficient (0 => ambipolar term off)
  bool eta_spitzer = false;
  Real spitzer_eta0_code = 0.0;  // eta_m = eta0*(p/rho)^(-3/2)
  parabolic::DiffusionSelection mode =
      parabolic::DiffusionSelection::explicit_only;

  // wrapper functions: add non-ideal E-Field and energy (Poynting) flux. Each dispatches
  // to the Ohmic and/or ambipolar implementations depending on which coefficients are on.
  void AddResistiveEMFs(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddResistiveFluxes(const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx);
  void NewTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);

  struct RawState {
    bool admissible;
    bool active_admissible;
    Real diffusion_dt;
  };
  // Inspect current conserved energy and CT faces before any EOS floor.
  RawState InspectRawState(const DvceArray5D<Real> &u, const EOS_Data &eos,
      const DvceFaceFld4D<Real> &b, bool include_ghosts = false) const;

  // Implementation kernels. These MUST be public: each launches Kokkos par_for kernels,
  // nvcc forbids an extended __host__ __device__ lambda inside a private/protected member
  // function (the CUDA build fails otherwise; the CPU/Serial build does not catch this).
  // Matches main, where the Ohmic implementations are also public.
  // Ohmic resistivity, constant or Spitzer: E = eta_m * J.
  void AddEMFOhmic(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddFluxOhmic(const DvceFaceFld4D<Real> &b, DvceFaceFld5D<Real> &flx);

  // Ambipolar diffusion (constant coefficient): E = eta_ad * [B^2 J - (J.B) B].
  void AddEMFConstantAmbipolar(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void AddFluxConstantAmbipolar(const DvceFaceFld4D<Real> &b0, DvceFaceFld5D<Real> &flx);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_RESISTIVITY_HPP_
