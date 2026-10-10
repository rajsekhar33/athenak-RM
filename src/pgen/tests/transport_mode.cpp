//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file transport_mode.cpp
//! \brief Periodic thermal Fourier mode on a fixed Hydro/MHD background. Test-only pgen.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "mhd/rsolvers/advect_mhd.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "pgen/tests/transport_patch.hpp"
#include "pgen/tests/transport_equilibrium.hpp"
#include "pgen/tests/transport_operator.hpp"
#include "pgen/tests/transport_saturated_operator.hpp"
#include "pgen/tests/transport_gaussian.hpp"
#include "pgen/tests/transport_helical.hpp"
#include "pgen/tests/transport_coupled.hpp"
#include "pgen/tests/transport_sawtooth.hpp"

namespace {
// Exercise the real device solver with stale (poisoned) energy flux. The test
// covers all directions, both upwind branches, zero velocity and isothermal EOS.
// Keep this inside the test pgen so CPU/GPU use the same executable and kernels.
void CheckAdvectionFlux() {
  constexpr int nc = 18;
  DvceArray5D<Real> left("adv_left",nc,5,1,1,1), right("adv_right",nc,5,1,1,1);
  DvceArray5D<Real> bl("adv_bl",nc,3,1,1,1), br("adv_br",nc,3,1,1,1);
  DvceArray5D<Real> flux("adv_flux",nc,5,1,1,1);
  DvceArray4D<Real> bx("adv_bx",nc,1,1,1), ey("adv_ey",nc,1,1,1);
  DvceArray4D<Real> ez("adv_ez",nc,1,1,1);
  Kokkos::deep_copy(left, 0.0);
  Kokkos::deep_copy(right, 0.0);
  Kokkos::deep_copy(bl, 0.0);
  Kokkos::deep_copy(br, 0.0);
  Kokkos::deep_copy(bx, 0.0);
  Kokkos::deep_copy(flux, 123.0);
  Kokkos::parallel_for("check_advection_flux", Kokkos::RangePolicy<DevExeSpace>(0,nc),
  KOKKOS_LAMBDA(const int m) {
    int dir = (m/3)%3;
    Real vel = (m%3-1)*0.5;
    EOS_Data eos{};
    eos.is_ideal = m < 9;
    left(m,IDN,0,0,0) = right(m,IDN,0,0,0) = 1.0;
    left(m,IVX+dir,0,0,0) = right(m,IVX+dir,0,0,0) = vel;
    left(m,IEN,0,0,0) = 2.0;
    right(m,IEN,0,0,0) = 3.0;
    if (dir == 0) {
      mhd::Advect<IVX>(eos,m,0,0,0,0,0,0,left,right,bl,br,bx,flux,ey,ez);
    } else if (dir == 1) {
      mhd::Advect<IVY>(eos,m,0,0,0,0,0,0,left,right,bl,br,bx,flux,ey,ez);
    } else {
      mhd::Advect<IVZ>(eos,m,0,0,0,0,0,0,left,right,bl,br,bx,flux,ey,ez);
    }
  });
  auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), flux);
  for (int m=0; m<nc; ++m) {
    Real vel = (m%3-1)*0.5;
    Real expected = m < 9 ? (vel < 0.0 ? 3.0 : 2.0)*vel : 123.0;
    if (result(m,IEN,0,0,0) != expected) {
      std::cerr << "transport advection energy-flux check FAILED: " << m << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (global_variable::my_rank == 0) {
    std::cout << "transport advection flux checks: 18 PASS" << std::endl;
  }
}

struct Mode {
  Real kx, ky, kz, rho, temp, amp, bx, by, bz;
};

Mode ReadMode(ParameterInput *pin, Mesh *pm) {
  auto *pack = pm->pmb_pack;
  const bool is_mhd = pack->pmhd != nullptr;
  if ((!is_mhd && pack->phydro == nullptr) ||
      pin->GetString("time", "evolution") != "kinematic") {
    std::cerr << "transport_mode requires kinematic Hydro or MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int nx = pin->GetOrAddInteger("problem", "mode_x", 1);
  int ny = pin->GetOrAddInteger("problem", "mode_y", 0);
  int nz = pin->GetOrAddInteger("problem", "mode_z", 0);
  if ((nx == 0 && ny == 0 && nz == 0) ||
      (!pm->multi_d && ny != 0) || (!pm->three_d && nz != 0)) {
    std::cerr << "transport_mode requires a nonzero mode on active axes" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &s = pm->mesh_size;
  Mode v;
  v.kx = 2.0*M_PI*nx/(s.x1max-s.x1min);
  v.ky = 2.0*M_PI*ny/(s.x2max-s.x2min);
  v.kz = 2.0*M_PI*nz/(s.x3max-s.x3min);
  v.rho = pin->GetOrAddReal("problem", "rho0", 1.0);
  v.temp = pin->GetOrAddReal("problem", "temp0", 1.0);
  v.amp = pin->GetOrAddReal("problem", "amplitude", 0.01);
  v.bx = is_mhd ? pin->GetOrAddReal("problem", "bx", 1.0) : 0.0;
  v.by = is_mhd ? pin->GetOrAddReal("problem", "by", 0.0) : 0.0;
  v.bz = is_mhd ? pin->GetOrAddReal("problem", "bz", 0.0) : 0.0;
  const auto &eos = is_mhd ? pack->pmhd->peos->eos_data : pack->phydro->peos->eos_data;
  if (!eos.is_ideal || !(v.rho > 0.0) ||
      !(v.temp > std::abs(v.amp)) || !std::isfinite(v.rho) ||
      !std::isfinite(v.temp) || !std::isfinite(v.amp) ||
      !std::isfinite(v.bx) || !std::isfinite(v.by) || !std::isfinite(v.bz)) {
    std::cerr << "transport_mode requires finite positive ideal-gas data" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return v;
}

// Measure the solution only. Analytic evolution and acceptance live in Python,
// independently of the C++ transport operator and this initialization.
void MeasureMode(ParameterInput *pin, Mesh *pm) {
  Mode v = ReadMode(pin, pm);
  auto *pack = pm->pmb_pack;
  const bool is_mhd = pack->pmhd != nullptr;
  auto device_u = is_mhd ? pack->pmhd->u0 : pack->phydro->u0;
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device_u);
  auto &ind = pm->mb_indcs;
  auto size = pack->pmb->mb_size.h_view;
  const auto &eos = is_mhd ? pack->pmhd->peos->eos_data : pack->phydro->peos->eos_data;
  Real gm1 = eos.gamma-1.0;
  Real sums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  Real tmin = std::numeric_limits<Real>::max();
  Real tmax = -std::numeric_limits<Real>::max();
  Real benergy = 0.5*(SQR(v.bx)+SQR(v.by)+SQR(v.bz));
  int invalid = 0;
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto s = size(m);
    Real vol = s.dx1*s.dx2*s.dx3;
    for (int k=ind.ks; k<=ind.ke; ++k) {
      for (int j=ind.js; j<=ind.je; ++j) {
        for (int i=ind.is; i<=ind.ie; ++i) {
          Real x = CellCenterX(i-ind.is, ind.nx1, s.x1min, s.x1max);
          Real y = CellCenterX(j-ind.js, ind.nx2, s.x2min, s.x2max);
          Real z = CellCenterX(k-ind.ks, ind.nx3, s.x3min, s.x3max);
          Real rho = u(m,IDN,k,j,i);
          Real kinetic = 0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+
                             SQR(u(m,IM3,k,j,i)))/rho;
          Real temp = gm1*(u(m,IEN,k,j,i)-kinetic-benergy)/rho;
          if (!std::isfinite(temp) || !(rho > 0.0)) {
            ++invalid;
            continue;
          }
          Real phase = v.kx*x+v.ky*y+v.kz*z;
          sums[0] += vol;
          sums[1] += vol*temp;
          sums[2] += vol*(temp-v.temp)*std::cos(phase);
          sums[3] += vol*(temp-v.temp)*std::sin(phase);
          sums[4] += vol*u(m,IEN,k,j,i);
          tmin = std::fmin(tmin, temp);
          tmax = std::fmax(tmax, temp);
        }
      }
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (invalid || !(sums[0] > 0.0)) {
    std::cerr << "transport_mode invalid final state" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::string path = pin->GetString("job", "basename") + ".mode.dat";
    FILE *file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file, "# time mean_temp cosine sine min_temp max_temp total_energy\n");
    std::fprintf(file, "%.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
                 pm->time, sums[1]/sums[0], 2.0*sums[2]/sums[0],
                 2.0*sums[3]/sums[0], tmin, tmax, sums[4]);
    if (std::fclose(file) != 0) {
      std::exit(EXIT_FAILURE);
    }
  }
}
} // namespace

void ProblemGenerator::TransportMode(ParameterInput *pin, const bool restart) {
  std::string profile=pin->GetOrAddString("problem","profile","fourier");
  const bool is_mhd = pmy_mesh_->pmb_pack->pmhd != nullptr;
  if (!is_mhd && profile != "fourier") {
    std::cerr << "transport_mode profiles other than fourier require MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (profile == "saturated_sawtooth") {
    user_srcs = true;
    user_srcs_func = transport_stages::Check;
    pgen_final_func = transport_sawtooth::Finalize;
    transport_sawtooth::Initialize(pin, pmy_mesh_, restart);
    return;
  }
  if (profile=="coupled") {
    user_srcs=user_work_in_loop=true;
    user_srcs_func=transport_coupled::RawState;
    user_work_in_loop_func=transport_coupled::Monitor;
    pgen_final_func=transport_coupled::Finalize;
    transport_coupled::Initialize(pin,pmy_mesh_,restart);
    return;
  }
  if (profile=="helical") {
    user_srcs=true;
    user_srcs_func=transport_stages::Check;
    pgen_final_func=transport_helical::Finalize;
    transport_helical::Initialize(pin,pmy_mesh_,restart);
    return;
  }
  if (profile=="gaussian" || profile=="front") {
    user_srcs=true;
    user_srcs_func=transport_stages::Check;
    pgen_final_func=transport_gaussian::Finalize;
    transport_gaussian::Initialize(pin,pmy_mesh_,restart);
    return;
  }
  if (profile=="sovinec" || profile=="resistance") {
    user_srcs=user_work_in_loop=true;
    user_bcs_func=transport_equilibrium::Boundary;
    user_srcs_func=transport_equilibrium::Source;
    user_work_in_loop_func=transport_equilibrium::Monitor;
    pgen_final_func=transport_equilibrium::Finalize;
    transport_equilibrium::Initialize(pin,pmy_mesh_,restart);
    return;
  }
  if (pin->GetOrAddString("problem","profile","fourier") != "fourier") {
    user_srcs=true;
    user_srcs_func=transport_stages::Check;
    user_work_in_loop=true;
    user_work_in_loop_func=transport_patch::MonitorPatch;
    pgen_final_func=transport_patch::FinalizePatch;
    transport_patch::InitializePatch(pin,pmy_mesh_,restart);
    return;
  }
  Mode v = ReadMode(pin, pmy_mesh_);
  const bool check_operator = pin->GetOrAddBoolean("problem", "check_conduction_operator",
                                                 false);
  const bool check_advection =
      pin->GetOrAddBoolean("problem", "check_advection_flux", false);
  const bool check_saturated = pin->GetOrAddBoolean("problem", "check_saturated_operator",
                                                  false);
  if (!is_mhd && (check_operator || check_advection || check_saturated)) {
    std::cerr << "transport_mode operator and advection diagnostics require MHD"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (check_operator) {
    transport_operator::Check(pmy_mesh_, restart);
  }
  if (check_saturated) {
    transport_saturated_operator::Check(pin, pmy_mesh_, restart);
  }
  if (check_advection) {
    CheckAdvectionFlux();
  }
  pgen_final_func = MeasureMode;
  // Register diagnostics on restart, but never reinitialize the saved solution.
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  auto &ind = pmy_mesh_->mb_indcs;
  auto size = pack->pmb->mb_size;
  auto u = is_mhd ? pack->pmhd->u0 : pack->phydro->u0;
  DvceArray4D<Real> bx, by, bz;
  if (is_mhd) {
    bx = pack->pmhd->b0.x1f;
    by = pack->pmhd->b0.x2f;
    bz = pack->pmhd->b0.x3f;
  }
  int is=ind.is, js=ind.js, ks=ind.ks;
  int nx=ind.nx1, ny=ind.nx2, nz=ind.nx3;
  const auto &eos = is_mhd ? pack->pmhd->peos->eos_data : pack->phydro->peos->eos_data;
  Real gm1 = eos.gamma-1.0;
  par_for("transport_initial", DevExeSpace(), 0, pack->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    auto s = size.d_view(m);
    Real x = CellCenterX(i-is, nx, s.x1min, s.x1max);
    Real y = CellCenterX(j-js, ny, s.x2min, s.x2max);
    Real z = CellCenterX(k-ks, nz, s.x3min, s.x3max);
    Real temp = v.temp+v.amp*cos(v.kx*x+v.ky*y+v.kz*z);
    u(m,IDN,k,j,i) = v.rho;
    u(m,IM1,k,j,i) = u(m,IM2,k,j,i) = u(m,IM3,k,j,i) = 0.0;
    u(m,IEN,k,j,i) = v.rho*temp/gm1+0.5*(SQR(v.bx)+SQR(v.by)+SQR(v.bz));
    if (is_mhd) {
      bx(m,k,j,i) = v.bx;
      by(m,k,j,i) = v.by;
      bz(m,k,j,i) = v.bz;
      if (i == is+nx-1) bx(m,k,j,i+1) = v.bx;
      if (j == js+ny-1) by(m,k,j+1,i) = v.by;
      if (k == ks+nz-1) bz(m,k+1,j,i) = v.bz;
    }
  });
}
