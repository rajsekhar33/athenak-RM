//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file transport_refinement.cpp
//! \brief Test-only thermal mode with observable refine/derefine topology events.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
struct RefinementMode {
  Real kx, ky, kz, rho, temp, amp, bx, by, bz;
  int period;
  Real rho_right = 0.0, rho_mid = 0.0;
  bool same_density = true;
};
RefinementMode cfg;
// Test-only callbacks share this basename across initialization and finalization.
std::string diagnostic_basename;  // NOLINT(runtime/string)
int observed_topology;
bool trace_started = false;

KOKKOS_INLINE_FUNCTION
Real Sinc(Real x) {
  return x == 0.0 ? 1.0 : sin(x)/x;
}

// Record the conserved state and CT faces directly. Analysis and acceptance are
// deliberately external to the problem generator. The raw table covers every
// active cell and both faces of every component, including collapsed directions.
void Measure(Mesh *pm, const char *event, bool dump = false) {
  auto *pack = pm->pmb_pack;
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->u0);
  auto b1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                               pack->pmhd->b0.x1f);
  auto b2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                               pack->pmhd->b0.x2f);
  auto b3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                               pack->pmhd->b0.x3f);
  const auto &ind = pm->mb_indcs;
  const auto size = pack->pmb->mb_size.h_view;
  const Real gm1 = pack->pmhd->peos->eos_data.gamma-1.0;
  Real sums[4] = {0.0, 0.0, 0.0, 0.0};  // volume, mass, energy, rho*T
  Real extrema[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Real tmin = std::numeric_limits<Real>::max();
  Real tmax = -tmin;
  int invalid = 0, nrefined = 0, max_level = 0;
  FILE *cells = nullptr;
  if (dump) {
    const std::string path = diagnostic_basename + ".refinement_cells.rank" +
                             std::to_string(global_variable::my_rank) + ".dat";
    cells = std::fopen(path.c_str(), "w");
    if (!cells) Kokkos::abort("cannot open transport refinement cell table");
    std::fprintf(cells, "# gid level i j k x y z dx dy dz rho m1 m2 m3 energy "
                       "b1lo b1hi b2lo b2hi b3lo b3hi\n");
  }
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto s = size(m);
    const int level = pack->pmb->mb_lev.h_view(m)-pm->root_level;
    max_level = std::max(max_level, level);
    nrefined += level > 0;
    const Real vol = s.dx1*s.dx2*s.dx3;
    for (int k=ind.ks; k<=ind.ke; ++k) {
      for (int j=ind.js; j<=ind.je; ++j) {
        for (int i=ind.is; i<=ind.ie; ++i) {
          const Real rho = u(m,IDN,k,j,i);
          const Real m1 = u(m,IM1,k,j,i), m2 = u(m,IM2,k,j,i);
          const Real m3 = u(m,IM3,k,j,i), energy = u(m,IEN,k,j,i);
          const Real bx = 0.5*(b1(m,k,j,i)+b1(m,k,j,i+1));
          const Real by = 0.5*(b2(m,k,j,i)+b2(m,k,j+1,i));
          const Real bz = 0.5*(b3(m,k,j,i)+b3(m,k+1,j,i));
          const Real kin = 0.5*(SQR(m1)+SQR(m2)+SQR(m3))/rho;
          const Real mag = 0.5*(SQR(bx)+SQR(by)+SQR(bz));
          const Real temp = gm1*(energy-kin-mag)/rho;
          if (!(rho > 0.0) || !(temp > 0.0) || !std::isfinite(temp) ||
              !std::isfinite(energy) || !std::isfinite(mag)) ++invalid;
          sums[0] += vol;
          sums[1] += vol*rho;
          sums[2] += vol*energy;
          sums[3] += vol*rho*temp;
          tmin = std::fmin(tmin, temp);
          tmax = std::fmax(tmax, temp);
          // Departure from the initial density profile, not a zero-error
          // requirement when the fixture uses coupled dynamic MHD.
          Real initial_rho = cfg.rho;
          if (!cfg.same_density) {
            const Real x = CellCenterX(i-ind.is, ind.nx1, s.x1min, s.x1max);
            initial_rho = x >= cfg.rho_mid ? cfg.rho_right : cfg.rho;
          }
          extrema[0] = std::fmax(extrema[0], std::abs(rho-initial_rho));
          extrema[1] = std::fmax(extrema[1], std::abs(m1));
          extrema[1] = std::fmax(extrema[1], std::abs(m2));
          extrema[1] = std::fmax(extrema[1], std::abs(m3));
          extrema[2] = std::fmax(extrema[2], std::abs(b1(m,k,j,i)-cfg.bx));
          extrema[2] = std::fmax(extrema[2], std::abs(b1(m,k,j,i+1)-cfg.bx));
          extrema[3] = std::fmax(extrema[3], std::abs(b2(m,k,j,i)-cfg.by));
          extrema[3] = std::fmax(extrema[3], std::abs(b2(m,k,j+1,i)-cfg.by));
          extrema[4] = std::fmax(extrema[4], std::abs(b3(m,k,j,i)-cfg.bz));
          extrema[4] = std::fmax(extrema[4], std::abs(b3(m,k+1,j,i)-cfg.bz));
          Real divb = (b1(m,k,j,i+1)-b1(m,k,j,i))/s.dx1;
          if (pm->multi_d) divb += (b2(m,k,j+1,i)-b2(m,k,j,i))/s.dx2;
          if (pm->three_d) divb += (b3(m,k+1,j,i)-b3(m,k,j,i))/s.dx3;
          extrema[5] = std::fmax(extrema[5], std::abs(divb));
          if (cells) {
            const Real x = CellCenterX(i-ind.is, ind.nx1, s.x1min, s.x1max);
            const Real y = CellCenterX(j-ind.js, ind.nx2, s.x2min, s.x2max);
            const Real z = CellCenterX(k-ind.ks, ind.nx3, s.x3min, s.x3max);
            std::fprintf(cells, "%d %d %d %d %d", pack->gids+m, level,
                         i-ind.is, j-ind.js, k-ind.ks);
            const Real row[] = {x, y, z, s.dx1, s.dx2, s.dx3, rho, m1, m2, m3, energy,
                                b1(m,k,j,i), b1(m,k,j,i+1), b2(m,k,j,i),
                                b2(m,k,j+1,i), b3(m,k,j,i), b3(m,k+1,j,i)};
            for (Real value : row) std::fprintf(cells, " %.17e", value);
            std::fprintf(cells, "\n");
          }
        }
      }
    }
  }
  if (cells && std::fclose(cells) != 0) Kokkos::abort("cell table close failed");
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, extrema, 6, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nrefined, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max_level, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    const std::string path = diagnostic_basename + ".refinement_trace.dat";
    FILE *trace = std::fopen(path.c_str(), trace_started ? "a" : "w");
    if (!trace) Kokkos::abort("cannot open transport refinement trace");
    if (!trace_started) {
      std::fprintf(trace, "# event cycle time topology nmb nrefined max_level volume "
                         "mass energy rhoT min_temp max_temp rho_error momentum_max "
                         "b1_error b2_error b3_error divb_max invalid\n");
    }
    std::fprintf(trace, "%s %d %.17e %d %d %d %d", event, pm->ncycle, pm->time,
                 pm->GetAMRLoadBalanceUpdateSeq(), pm->nmb_total, nrefined, max_level);
    for (Real value : sums) std::fprintf(trace, " %.17e", value);
    std::fprintf(trace, " %.17e %.17e", tmin, tmax);
    for (Real value : extrema) std::fprintf(trace, " %.17e", value);
    std::fprintf(trace, " %d\n", invalid);
    if (std::fclose(trace) != 0) Kokkos::abort("refinement trace close failed");
    trace_started = true;
  }
  if (invalid) Kokkos::abort("invalid transport refinement state");
}

// This callback runs after AMR transfer, CT repair and boundary refresh. A topology
// counter identifies those calls without confusing ordinary RK timestep calls with
// completed cycles. The callback imposes no additional timestep restriction.
void ObserveTopology(Mesh *pm) {
  pm->pgen->dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  if (observed_topology != pm->GetAMRLoadBalanceUpdateSeq()) {
    Measure(pm, "after_amr");
    observed_topology = pm->GetAMRLoadBalanceUpdateSeq();
  }
}

void Refine(MeshBlockPack *pack) {
  auto *pm = pack->pmesh;
  Measure(pm, "before_amr");
  auto &flag = pm->pmr->refine_flag;
  // Cycle zero stays coarse. Alternate a lower corner, no refinement, an upper
  // corner, and no refinement. This forces both genuine topology operations.
  const int phase = pm->ncycle > 0 ? ((pm->ncycle-1)/cfg.period)%4 : -1;
  const auto domain = pm->mesh_size;
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto s = pack->pmb->mb_size.h_view(m);
    const bool upper = phase == 2;
    const Real xmid = 0.5*(domain.x1min+domain.x1max);
    const Real ymid = 0.5*(domain.x2min+domain.x2max);
    const Real zmid = 0.5*(domain.x3min+domain.x3max);
    const bool region = (upper ? s.x1min >= xmid : s.x1max <= xmid) &&
        (!pm->multi_d || (upper ? s.x2min >= ymid : s.x2max <= ymid)) &&
        (!pm->three_d || (upper ? s.x3min >= zmid : s.x3max <= zmid));
    const int level = pack->pmb->mb_lev.h_view(m)-pm->root_level;
    const bool wanted = (phase == 0 || phase == 2) && region;
    flag.h_view(pack->gids+m) = wanted ? (level < 1 ? 1 : 0) : -1;
  }
  flag.template modify<HostMemSpace>();
  flag.template sync<DevExeSpace>();
}

void Finalize(ParameterInput *pin, Mesh *pm) {
  (void) pin;
  Measure(pm, "final", true);
}
} // namespace

void ProblemGenerator::TransportRefinement(ParameterInput *pin, const bool restart) {
  auto *pm = pmy_mesh_;
  auto *pack = pm->pmb_pack;
  const std::string evolution = pin->GetString("time", "evolution");
  if (pack->pmhd == nullptr || !pack->pmhd->peos->eos_data.is_ideal ||
      pack->pcoord->is_special_relativistic ||
      pack->pcoord->is_general_relativistic ||
      pack->pcoord->is_dynamical_relativistic ||
      (evolution != "kinematic" && evolution != "dynamic")) {
    Kokkos::abort(
        "transport_refinement requires kinematic or dynamic Newtonian ideal MHD");
  }
  const auto domain = pm->mesh_size;
  const int nx = pin->GetOrAddInteger("problem", "mode_x", 1);
  const int ny = pin->GetOrAddInteger("problem", "mode_y", 1);
  const int nz = pin->GetOrAddInteger("problem", "mode_z", 0);
  if ((!pm->multi_d && ny != 0) || (!pm->three_d && nz != 0)) {
    Kokkos::abort("transport_refinement mode uses an inactive dimension");
  }
  cfg = {2.0*M_PI*nx/(domain.x1max-domain.x1min),
         2.0*M_PI*ny/(domain.x2max-domain.x2min),
         2.0*M_PI*nz/(domain.x3max-domain.x3min),
         pin->GetOrAddReal("problem", "rho0", 1.0),
         pin->GetOrAddReal("problem", "temp0", 1.0),
         pin->GetOrAddReal("problem", "amplitude", 0.1),
         pin->GetOrAddReal("problem", "bx", 1.0),
         pin->GetOrAddReal("problem", "by", 0.5),
         pin->GetOrAddReal("problem", "bz", 0.25),
         pin->GetOrAddInteger("problem", "amr_period", 3)};
  // Do not insert a new default key into existing inputs/restart headers.
  cfg.rho_right = pin->DoesParameterExist("problem", "rho_right") ?
      pin->GetReal("problem", "rho_right") : cfg.rho;
  cfg.rho_mid = 0.5*domain.x1min+0.5*domain.x1max;
  cfg.same_density = cfg.rho_right == cfg.rho;
  if (!(cfg.rho > 0.0) || !(cfg.temp > std::abs(cfg.amp)) || cfg.period < 1 ||
      !std::isfinite(cfg.rho) || !std::isfinite(cfg.temp) ||
      !std::isfinite(cfg.amp) || !std::isfinite(cfg.bx) ||
      !std::isfinite(cfg.by) || !std::isfinite(cfg.bz)) {
    Kokkos::abort("transport_refinement requires finite positive thermal data");
  }
  if (!(cfg.rho_right > 0.0) || !std::isfinite(cfg.rho_right) ||
      (!cfg.same_density && pm->mesh_indcs.nx1 % 2 != 0)) {
    Kokkos::abort("transport_refinement density step requires positive finite density "
                  "and an even root nx1 for a cell-aligned midpoint");
  }
  diagnostic_basename = pin->GetString("job", "basename");
  observed_topology = pm->GetAMRLoadBalanceUpdateSeq();
  user_ref_func = Refine;
  user_dt = true;
  user_time_step_func = ObserveTopology;
  pgen_final_func = Finalize;
  if (restart) {
    Measure(pm, "restart");
    return;  // preserve restored cell-centered and CT fields
  }
  const auto ind = pm->mb_indcs;
  const auto size = pack->pmb->mb_size;
  const auto u = pack->pmhd->u0;
  const auto b = pack->pmhd->b0;
  const auto v = cfg;
  const Real gm1 = pack->pmhd->peos->eos_data.gamma-1.0;
  par_for("transport_refinement_initial", DevExeSpace(), 0, pack->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto s = size.d_view(m);
    const Real x = CellCenterX(i-ind.is, ind.nx1, s.x1min, s.x1max)-domain.x1min;
    const Real y = CellCenterX(j-ind.js, ind.nx2, s.x2min, s.x2max)-domain.x2min;
    const Real z = CellCenterX(k-ind.ks, ind.nx3, s.x3min, s.x3max)-domain.x3min;
    const Real average = Sinc(0.5*v.kx*s.dx1)*Sinc(0.5*v.ky*s.dx2)*
                         Sinc(0.5*v.kz*s.dx3);
    const Real temp = v.temp+v.amp*average*cos(v.kx*x+v.ky*y+v.kz*z);
    u(m,IDN,k,j,i) = v.rho;
    u(m,IM1,k,j,i) = u(m,IM2,k,j,i) = u(m,IM3,k,j,i) = 0.0;
    u(m,IEN,k,j,i) = v.rho*temp/gm1+0.5*(SQR(v.bx)+SQR(v.by)+SQR(v.bz));
    // Keep the old same-density initializer arithmetic untouched. Only the
    // new optional step overwrites density and its consistent thermal energy.
    if (!v.same_density) {
      const Real x_abs = CellCenterX(i-ind.is, ind.nx1, s.x1min, s.x1max);
      const Real rho = x_abs >= v.rho_mid ? v.rho_right : v.rho;
      u(m,IDN,k,j,i) = rho;
      u(m,IEN,k,j,i) = rho*temp/gm1+0.5*(SQR(v.bx)+SQR(v.by)+SQR(v.bz));
    }
    b.x1f(m,k,j,i) = v.bx;
    b.x2f(m,k,j,i) = v.by;
    b.x3f(m,k,j,i) = v.bz;
    if (i == ind.ie) b.x1f(m,k,j,i+1) = v.bx;
    if (j == ind.je) b.x2f(m,k,j+1,i) = v.by;
    if (k == ind.ke) b.x3f(m,k+1,j,i) = v.bz;
  });
  Measure(pm, "initial");
}
