//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hyperviscosity_probe.cpp
// Test-only hyperviscous face-flux and coupled ideal-gas circular-shear probe.
// No isolated integrator: evolution, when requested, uses the production driver.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "diffusion/viscosity.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {
void Require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "Hyperviscosity probe: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

Real Sinc(Real x) { return x == 0.0 ? 1.0 : std::sin(x)/x; }

void DumpState(ParameterInput *pin, Mesh *mesh, const std::string &suffix) {
  auto *pack = mesh->pmb_pack;
  const bool magnetic = pack->pmhd != nullptr;
  auto udev = magnetic ? pack->pmhd->u0 : pack->phydro->u0;
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), udev);
  const int n1 = u.extent_int(4), n2 = u.extent_int(3), n3 = u.extent_int(2);
  DvceFaceFld4D<Real> b("probe_dump_b", 1, n3, n2, n1);
  if (magnetic) {
    b = pack->pmhd->b0;
  } else {
    Kokkos::deep_copy(b.x1f, 0.0);
    Kokkos::deep_copy(b.x2f, 0.0);
    Kokkos::deep_copy(b.x3f, 0.0);
  }
  auto bx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b.x1f);
  auto by = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b.x2f);
  auto bz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b.x3f);
  const auto &idx = mesh->mb_indcs;
  const auto size = pack->pmb->mb_size.h_view(0);
  const std::string name = pin->GetString("problem", "state_output")+suffix;
  std::ofstream out(name);
  out << std::setprecision(17) << "# time " << mesh->time << '\n'
      << "# cycle " << mesh->ncycle << " dt " << mesh->dt << '\n';
  const auto &ec = mesh->ecounter;
  out << "# events " << ec.nfofc << ' ' << ec.neos_dfloor << ' '
      << ec.neos_efloor << ' ' << ec.neos_tfloor << ' ' << ec.neos_fail << '\n';
  for (int k=idx.ks; k<=idx.ke; ++k) for (int j=idx.js; j<=idx.je; ++j) {
    for (int i=idx.is; i<=idx.ie; ++i) {
      out << k << ' ' << j << ' ' << i;
      for (int v=0; v<5; ++v) out << ' ' << u(0,v,k,j,i);
      out << ' ' << 0.5*(bx(0,k,j,i)+bx(0,k,j,i+1));
      out << ' ' << 0.5*(by(0,k,j,i)+by(0,k,j+1,i));
      out << ' ' << 0.5*(bz(0,k,j,i)+bz(0,k+1,j,i));
      Real divb = (bx(0,k,j,i+1)-bx(0,k,j,i))/size.dx1;
      if (mesh->multi_d) divb += (by(0,k,j+1,i)-by(0,k,j,i))/size.dx2;
      if (mesh->three_d) divb += (bz(0,k+1,j,i)-bz(0,k,j,i))/size.dx3;
      out << ' ' << divb << '\n';
    }
  }
  Require(out.good(), "state dump failed");
  if (magnetic) {
    std::ofstream faces(name+".faces");
    faces << std::setprecision(17) << "# time " << mesh->time << '\n'
          << "# cycle " << mesh->ncycle << " dt " << mesh->dt << '\n';
    for (int d=0; d<3; ++d) {
      auto field = d == 0 ? bx : (d == 1 ? by : bz);
      for (int k=idx.ks; k<=idx.ke+(d == 2); ++k) {
        for (int j=idx.js; j<=idx.je+(d == 1); ++j) {
          for (int i=idx.is; i<=idx.ie+(d == 0); ++i) {
            faces << d << ' ' << k << ' ' << j << ' ' << i << ' '
                  << field(0,k,j,i) << '\n';
          }
        }
      }
    }
    Require(faces.good(), "CT face dump failed");
  }
}

void FinalState(ParameterInput *pin, Mesh *mesh) { DumpState(pin, mesh, ".final"); }
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Reuse production initialization for uniform block/rank/restart checks.
  // Its ideal-wave error callback is not an oracle for dissipative evolution.
  if (pin->GetOrAddBoolean("problem", "delegate_linear_wave", false)) {
    LinearWave(pin, restart);
    pgen_final_func = nullptr;
    return;
  }
  auto *pack = pmy_mesh_->pmb_pack;
  Require(!restart && global_variable::nranks == 1 && pack->nmb_thispack == 1,
          "fresh one-block one-rank input required");
  Require((pack->pmhd != nullptr) != (pack->phydro != nullptr),
          "exactly one Hydro or MHD owner required");
  const bool magnetic = pack->pmhd != nullptr;
  const bool evolve = pin->GetOrAddBoolean("problem", "circular_shear", false);
  const bool linear_wave = pin->GetOrAddBoolean("problem", "linear_wave", false);
  const bool variable_rho = pin->GetOrAddBoolean("problem", "variable_rho", false);
  Require(evolve || linear_wave || pin->GetInteger("time", "nlim") == 0,
          "operator probe requires nlim=0");
  Require(!linear_wave || (!evolve && !pmy_mesh_->multi_d && !variable_rho),
          "linear-wave fixture requires one dimension and exclusive mode");
  Require(!evolve || !variable_rho,
          "circular shear requires constant density");
  auto *eos = magnetic ? pack->pmhd->peos : pack->phydro->peos;
  Require(eos->eos_data.is_ideal, "ideal gas required");
  auto udev = magnetic ? pack->pmhd->u0 : pack->phydro->u0;
  auto wdev = magnetic ? pack->pmhd->w0 : pack->phydro->w0;
  auto u = Kokkos::create_mirror(udev);
  auto w = Kokkos::create_mirror(wdev);
  Require(u.extent_int(1) == 5, "five fluid variables without scalars required");
  if (linear_wave) {
    // Test-only FV conserved fixture; no eigenproblem or evolution is implemented here.
    // Transverse CT faces share the exact x-cell average supplied in the fixture.
    const int n1 = u.extent_int(4);
    const int n2 = u.extent_int(3), n3 = u.extent_int(2);
    Require(n2 == 1 && n3 == 1, "linear fixture expects inactive transverse axes");
    std::ifstream in(pin->GetString("problem", "wave_fixture"));
    std::string magic;
    int count = 0;
    in >> magic >> count;
    Require(magic == "HYPER4_LINEAR_FIXTURE_V1" && count == n1,
            "invalid linear-wave fixture header");
    DvceFaceFld4D<Real> b("linear_fixture_b", 1, n3, n2, n1);
    if (magnetic) b = pack->pmhd->b0;
    auto bx = Kokkos::create_mirror_view(b.x1f);
    auto by = Kokkos::create_mirror_view(b.x2f);
    auto bz = Kokkos::create_mirror_view(b.x3f);
    Real bx_constant = 0.0;
    for (int i=0; i<n1; ++i) {
      int index = -1;
      Real values[8];
      in >> index;
      for (int c=0; c<8; ++c) in >> values[c];
      Require(in.good() && index == i, "truncated or misindexed linear fixture");
      for (int c=0; c<8; ++c) {
        Require(std::isfinite(values[c]), "nonfinite linear fixture");
      }
      if (i == 0) bx_constant = values[5];
      Require(values[5] == bx_constant, "linear fixture Bx must be constant");
      Require(magnetic || (values[5] == 0.0 && values[6] == 0.0 && values[7] == 0.0),
              "Hydro linear fixture must have zero magnetic field");
      const Real density = values[0];
      Require(density > 0.0, "nonpositive linear fixture density");
      Real kinetic = 0.0, magnetic_energy = 0.0;
      for (int c=0; c<5; ++c) u(0,c,0,0,i) = values[c];
      w(0,IDN,0,0,i) = density;
      for (int c=0; c<3; ++c) {
        w(0,IVX+c,0,0,i) = values[c+1]/density;
        kinetic += 0.5*SQR(values[c+1])/density;
        magnetic_energy += 0.5*SQR(values[c+5]);
      }
      const Real internal = values[4]-kinetic-magnetic_energy;
      Require(internal > 0.0, "nonpositive linear fixture internal energy");
      w(0,IEN,0,0,i) = internal;
      bx(0,0,0,i) = bx_constant;
      for (int j=0; j<2; ++j) by(0,0,j,i) = values[6];
      for (int k=0; k<2; ++k) bz(0,k,0,i) = values[7];
    }
    bx(0,0,0,n1) = bx_constant;
    std::string extra;
    Require(!(in >> extra), "unexpected trailing linear fixture data");
    Kokkos::deep_copy(udev, u);
    Kokkos::deep_copy(wdev, w);
    if (magnetic) {
      Kokkos::deep_copy(pack->pmhd->b0.x1f, bx);
      Kokkos::deep_copy(pack->pmhd->b0.x2f, by);
      Kokkos::deep_copy(pack->pmhd->b0.x3f, bz);
    }
    pgen_final_func = FinalState;
    DumpState(pin, pmy_mesh_, ".initial");
    return;
  }
  const auto &idx = pmy_mesh_->mb_indcs;
  const auto size = pack->pmb->mb_size.h_view(0);
  const int ndim = pmy_mesh_->three_d ? 3 : (pmy_mesh_->multi_d ? 2 : 1);
  const int starts[3] = {idx.is, idx.js, idx.ks};
  const Real low[3] = {size.x1min, size.x2min, size.x3min};
  const Real lengths[3] = {size.x1max-size.x1min, size.x2max-size.x2min,
                           size.x3max-size.x3min};
  const Real dx[3] = {size.dx1, size.dx2, size.dx3};
  const Real pi = std::acos(-1.0), gm1 = eos->eos_data.gamma-1.0;
  const Real amp = pin->GetOrAddReal("problem", "amplitude", 0.1);
  int modes[2][3] = {
    {1,1,1}, {2,-1,-1}
  };
  const Real amplitudes[2][3] = {
    {0.2,-0.3,0.4}, {-0.1,0.2,0.15}
  };
  Real circle_e1[3] = {0.0,1.0,0.0}, circle_e2[3] = {0.0,0.0,1.0};
  Real background_b[3] = {0.0,0.3,0.0};
  if (evolve) {
    // Exact continuum circular shear: e1,e2 perpendicular to k, and constant
    // B perpendicular to k. Advection and ideal induction vanish. Hyper4-only
    // A(t)=A(0)*exp[-mu_hyper4*|k|^4*t/rho]; heating is uniform, continuum J=0.
    // Oblique Cartesian discrete derivatives need not preserve k.v=0 exactly;
    // their O(dx^2) longitudinal leakage is a convergence diagnostic, not a
    // continuum-null assertion or a reason to project the native solution.
    Require(std::isfinite(amp) && amp > 0.0, "finite positive amplitude required");
    Real normal[3] = {0.0,0.0,0.0};
    Real norm_squared = 0.0;
    for (int d=0; d<3; ++d) {
      modes[0][d] = pin->GetOrAddInteger("problem", "circular_mode"+
                                        std::to_string(d+1), d < ndim ? 1 : 0);
      Require(d < ndim || modes[0][d] == 0, "inactive circular modes must be zero");
      if (d < ndim) {
        normal[d] = 2.0*pi*modes[0][d]/lengths[d];
        norm_squared += SQR(normal[d]);
      }
    }
    Require(std::isfinite(norm_squared) && norm_squared > 0.0,
            "finite nonzero circular wave vector required");
    const Real norm = std::sqrt(norm_squared);
    for (int d=0; d<3; ++d) normal[d] /= norm;
    int seed = 0;
    for (int d=1; d<3; ++d) {
      if (std::abs(normal[d]) < std::abs(normal[seed])) seed = d;
    }
    Real transverse_squared = 0.0;
    for (int d=0; d<3; ++d) {
      circle_e1[d] = (d == seed ? 1.0 : 0.0)-normal[seed]*normal[d];
      transverse_squared += SQR(circle_e1[d]);
    }
    const Real transverse_norm = std::sqrt(transverse_squared);
    for (int d=0; d<3; ++d) {
      circle_e1[d] /= transverse_norm;
      background_b[d] = 0.3*circle_e1[d];
    }
    circle_e2[0] = normal[1]*circle_e1[2]-normal[2]*circle_e1[1];
    circle_e2[1] = normal[2]*circle_e1[0]-normal[0]*circle_e1[2];
    circle_e2[2] = normal[0]*circle_e1[1]-normal[1]*circle_e1[0];
  }
  const Real magnetic_energy = magnetic ? 0.5*(SQR(background_b[0])+
                               SQR(background_b[1])+SQR(background_b[2])) : 0.0;
  // Every allocated ghost cell is analytic; no communication is relied on
  // for the initial isolated kernel call. Kernel rho and velocity are prescribed
  // cell averages; their product is not claimed to be exact average(rho*v).
  for (int k=0; k<u.extent_int(2); ++k) for (int j=0; j<u.extent_int(3); ++j) {
    for (int i=0; i<u.extent_int(4); ++i) {
      const int coordinates[3] = {i,j,k};
      Real phases[2] = {0.0,0.0}, averages[2] = {1.0,1.0};
      for (int a=0; a<2; ++a) for (int d=0; d<ndim; ++d) {
        const Real wave = 2.0*pi*modes[a][d]/lengths[d];
        phases[a] += wave*(low[d]+(coordinates[d]-starts[d]+0.5)*dx[d]);
        averages[a] *= Sinc(0.5*wave*dx[d]);
      }
      Real density = 1.0, velocity[3] = {0.0,0.0,0.0};
      if (variable_rho) density += 0.2*averages[0]*std::cos(phases[0]);
      if (evolve) {
        for (int c=0; c<3; ++c) {
          velocity[c] = amp*averages[0]*(circle_e1[c]*std::cos(phases[0])+
                                        circle_e2[c]*std::sin(phases[0]));
        }
      } else {
        for (int c=0; c<3; ++c) for (int a=0; a<2; ++a) {
          velocity[c] += amplitudes[a][c]*averages[a]*std::cos(phases[a]);
        }
      }
      Real kinetic = 0.0;
      u(0,IDN,k,j,i) = w(0,IDN,k,j,i) = density;
      for (int c=0; c<3; ++c) {
        w(0,IVX+c,k,j,i) = velocity[c];
        u(0,IM1+c,k,j,i) = density*velocity[c];
        kinetic += 0.5*density*SQR(velocity[c]);
      }
      // Circular shear has exact constant FV total energy, not kinetic energy
      // reconstructed from averaged momentum. The positive variance remains
      // in cell-reconstructed internal energy and is part of the oracle.
      const Real energy = 1.0/gm1+(evolve ? 0.5*SQR(amp) : kinetic)+magnetic_energy;
      u(0,IEN,k,j,i) = energy;
      w(0,IEN,k,j,i) = energy-kinetic-magnetic_energy;
    }
  }
  Kokkos::deep_copy(udev, u);
  Kokkos::deep_copy(wdev, w);
  if (magnetic) {
    Kokkos::deep_copy(pack->pmhd->b0.x1f, background_b[0]);
    Kokkos::deep_copy(pack->pmhd->b0.x2f, background_b[1]);
    Kokkos::deep_copy(pack->pmhd->b0.x3f, background_b[2]);
  }
  pgen_final_func = FinalState;
  DumpState(pin, pmy_mesh_, ".initial");
  if (!evolve) {
    DvceFaceFld5D<Real> flux("probe_flux", 1, 5, u.extent_int(2),
                            u.extent_int(3), u.extent_int(4));
    Kokkos::deep_copy(flux.x1f, 0.0);
    Kokkos::deep_copy(flux.x2f, 0.0);
    Kokkos::deep_copy(flux.x3f, 0.0);
    Viscosity viscosity(magnetic ? "mhd" : "hydro", pack, pin);
    viscosity.AddViscousFluxes(wdev, eos->eos_data, flux);
    viscosity.NewTimeStep(wdev, eos->eos_data);
    std::ofstream out(pin->GetString("problem", "flux_output"));
    out << std::setprecision(17) << "# dtnew " << viscosity.dtnew << '\n';
    for (int d=0; d<ndim; ++d) {
      auto field = d == 0 ? flux.x1f : (d == 1 ? flux.x2f : flux.x3f);
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field);
      for (int k=idx.ks; k<=idx.ke+(d == 2); ++k) {
        for (int j=idx.js; j<=idx.je+(d == 1); ++j) {
          for (int i=idx.is; i<=idx.ie+(d == 0); ++i) {
            out << d << ' ' << k << ' ' << j << ' ' << i;
            for (int v=0; v<5; ++v) out << ' ' << host(0,v,k,j,i);
            out << '\n';
          }
        }
      }
    }
    Require(out.good(), "face flux dump failed");
  }
}
