#ifndef PGEN_TESTS_TRANSPORT_SAWTOOTH_HPP_
#define PGEN_TESTS_TRANSPORT_SAWTOOTH_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only periodic saturated sawtooth using production conduction and STS.
// No thermal evolution is implemented in this fixture.

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/tests/transport_stages.hpp"

namespace transport_sawtooth {
KOKKOS_INLINE_FUNCTION
Real Initial(Real x) {
  return 10.0 + 20.0 * (x / 10.0 - floor(x / 10.0 + 0.5));
}

inline void Dump(ParameterInput *pin, Mesh *pm, const char *label) {
  auto *pack = pm->pmb_pack;
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->u0);
  auto b1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->b0.x1f);
  auto b2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->b0.x2f);
  auto b3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->b0.x3f);
  const auto ind = pm->mb_indcs;
  const auto sizes = pack->pmb->mb_size.h_view;
  std::string path = pin->GetString("job", "basename") + ".sawtooth." + label + ".dat";
  FILE *f = std::fopen(path.c_str(), "w");
  if (!f) Kokkos::abort("cannot open sawtooth raw table");
  std::fprintf(f, "# cycle=%d time=%.17e dt=%.17e\n", pm->ncycle, pm->time, pm->dt);
  std::fprintf(f, "# gid level i j k x y z dx dy dz rho m1 m2 m3 energy "
                  "b1lo b1hi b2lo b2hi b3lo b3hi\n");
  for (int m = 0; m < pack->nmb_thispack; ++m) {
    const auto s = sizes(m);
    for (int i = ind.is; i <= ind.ie; ++i) {
      const int j = ind.js, k = ind.ks;
      const Real x = CellCenterX(i - ind.is, ind.nx1, s.x1min, s.x1max);
      std::fprintf(f, "%d 0 %d 0 0", pack->gids + m, i - ind.is);
      const Real values[] = {
          x, 0.5, 0.5, s.dx1, s.dx2, s.dx3, u(m, IDN, k, j, i),
          u(m, IM1, k, j, i), u(m, IM2, k, j, i), u(m, IM3, k, j, i),
          u(m, IEN, k, j, i), b1(m, k, j, i), b1(m, k, j, i + 1),
          b2(m, k, j, i), b2(m, k, j + 1, i), b3(m, k, j, i), b3(m, k + 1, j, i)};
      for (Real v : values) std::fprintf(f, " %.17e", v);
      std::fprintf(f, "\n");
    }
  }
  if (std::fclose(f) != 0) Kokkos::abort("sawtooth raw table close failed");
}

inline void Finalize(ParameterInput *pin, Mesh *pm) {
  if (pm->ncycle > transport_stages::first_cycle) transport_stages::Write(pin, pm);
  Dump(pin, pm, "final");
}

inline void Initialize(ParameterInput *pin, Mesh *pm, bool restart) {
  auto *pack = pm->pmb_pack;
  const auto domain = pm->mesh_size;
  const int n = pm->mesh_indcs.nx1;
  if (global_variable::nranks != 1 || !pack->pmhd || pack->phydro ||
      pm->multi_d || pm->multilevel || pack->pmhd->nscalars != 0 ||
      !pack->pmhd->peos->eos_data.is_ideal || pack->pmhd->peos->eos_data.gamma != 2.0 ||
      pin->GetString("time", "evolution") != "kinematic" ||
      pin->GetString("time", "integrator") != "rk2" ||
      pin->GetString("mhd", "rsolver") != "advect" ||
      domain.x1min != -10.0 || domain.x1max != 10.0 ||
      domain.x2min != 0.0 || domain.x2max != 1.0 ||
      domain.x3min != 0.0 || domain.x3max != 1.0 ||
      (n != 100 && n != 200 && n != 400) || pm->mb_indcs.nx1 != 100 ||
      pin->GetReal("mhd", "alpha_iso") != 40.0 ||
      pin->GetReal("mhd", "alpha_aniso") != 0.0 ||
      pin->GetBoolean("mhd", "alpha_spitzer") ||
      pin->GetString("mhd", "conduction_saturation") != "cowie_mckee" ||
      pin->GetReal("mhd", "saturation_phi") != 0.3) {
    Kokkos::abort("sawtooth fixture requires the pinned 1D periodic scalar setup");
  }
  for (const char *key : {"ix1_bc", "ox1_bc", "ix2_bc", "ox2_bc", "ix3_bc", "ox3_bc"}) {
    if (pin->GetString("mesh", key) != "periodic") {
      Kokkos::abort("sawtooth fixture requires periodic boundaries");
    }
  }
  // These check only ordinary RK stages. Production nonlinear-STS checks and
  // accepted/retried stage logs remain separate evidence, not double-counted.
  transport_stages::Reset(pin, pm, 0.0, 20.0);
  if (restart) {
    Dump(pin, pm, "restart");
    return;
  }
  auto ind = pm->mb_indcs;
  auto sizes = pack->pmb->mb_size;
  auto u = pack->pmhd->u0;
  auto b = pack->pmhd->b0;
  par_for("saturated_sawtooth_init", DevExeSpace(), 0, pack->nmb_thispack - 1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto s = sizes.d_view(m);
    const Real x = CellCenterX(i - ind.is, ind.nx1, s.x1min, s.x1max);
    u(m, IDN, k, j, i) = 1.0;
    u(m, IM1, k, j, i) = u(m, IM2, k, j, i) = u(m, IM3, k, j, i) = 0.0;
    // At all allowed N, jumps at +/-5 lie on faces. The center value is the
    // exact cell average; gamma2 and rho1 give internal energy=T.
    u(m, IEN, k, j, i) = Initial(x) + 0.5;
    b.x1f(m, k, j, i) = 1.0;
    b.x2f(m, k, j, i) = b.x3f(m, k, j, i) = 0.0;
    if (i == ind.ie) b.x1f(m, k, j, i + 1) = 1.0;
    if (j == ind.je) b.x2f(m, k, j + 1, i) = 0.0;
    if (k == ind.ke) b.x3f(m, k + 1, j, i) = 0.0;
  });
  Dump(pin, pm, "initial");
}
}  // namespace transport_sawtooth
#endif  // PGEN_TESTS_TRANSPORT_SAWTOOTH_HPP_
