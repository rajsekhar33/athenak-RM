//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file marching_cubes.cpp
//! \brief Standalone execution-space test for the marching-cubes area utility.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "utils/marching_cubes.hpp"

namespace {

constexpr Real kTolerance = 1.0e-5;

bool NearlyEqual(Real actual, Real expected) {
  return std::abs(actual - expected) <= kTolerance*(1.0 + std::abs(expected));
}

int Check(const char *name, Real actual, Real expected) {
  if (NearlyEqual(actual, expected)) return 0;
  std::cerr << name << ": expected " << expected << ", got " << actual << '\n';
  return 1;
}

} // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    using marching_cubes::CellSpacing;
    using marching_cubes::ComputeArea;
    using marching_cubes::Cube;

    const Cube no_surface(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    const Cube fully_inside(-1.0, -1.0, -1.0, -1.0,
                            -1.0, -1.0, -1.0, -1.0);
    const Cube x_plane(-0.5, 0.5, 0.5, -0.5,
                       -0.5, 0.5, 0.5, -0.5);
    const Cube y_plane(-0.5, -0.5, 0.5, 0.5,
                       -0.5, -0.5, 0.5, 0.5);
    const Cube z_plane(-0.5, -0.5, -0.5, -0.5,
                        0.5, 0.5, 0.5, 0.5);
    const Cube x_plane_physical(-0.5, 0.5, 0.5, -0.5,
                                -0.5, 0.5, 0.5, -0.5,
                                CellSpacing(2.0, 3.0, 5.0));
    const Cube y_plane_physical(-0.5, -0.5, 0.5, 0.5,
                                -0.5, -0.5, 0.5, 0.5,
                                CellSpacing(2.0, 3.0, 5.0));
    const Cube z_plane_physical(-0.5, -0.5, -0.5, -0.5,
                                0.5, 0.5, 0.5, 0.5,
                                CellSpacing(2.0, 3.0, 5.0));
    const Cube diagonal(-1.5, -0.5, 0.5, -0.5,
                        -0.5, 0.5, 1.5, 0.5);
    // A nondegenerate ambiguous face.  Its sign complement must retain the
    // same geometric surface; exactly balanced saddle faces are intentionally
    // excluded because the strict-sign convention chooses a tie-break.
    const Cube ambiguous(-2.0, 1.0, -1.0, 3.0,
                         2.0, 4.0, 5.0, 6.0);
    const Cube ambiguous_complement(2.0, -1.0, 1.0, -3.0,
                                    -2.0, -4.0, -5.0, -6.0);

    failures += Check("no surface", ComputeArea(no_surface), 0.0);
    failures += Check("fully inside", ComputeArea(fully_inside), 0.0);
    failures += Check("unit x plane", ComputeArea(x_plane), 1.0);
    failures += Check("unit y plane", ComputeArea(y_plane), 1.0);
    failures += Check("unit z plane", ComputeArea(z_plane), 1.0);
    failures += Check("physical x plane", ComputeArea(x_plane_physical), 15.0);
    failures += Check("physical y plane", ComputeArea(y_plane_physical), 10.0);
    failures += Check("physical z plane", ComputeArea(z_plane_physical), 6.0);
    failures += Check("diagonal plane", ComputeArea(diagonal),
                      3.0*std::sqrt(3.0)/4.0);
    failures += Check("ambiguous complement", ComputeArea(ambiguous),
                      ComputeArea(ambiguous_complement));

    Kokkos::View<Real *> device_results("marching_cube_areas", 5);
    Kokkos::parallel_for("marching_cubes_device_test", 5, KOKKOS_LAMBDA(int n) {
      const CellSpacing spacing(2.0, 3.0, 5.0);
      if (n == 0) {
        device_results(n) = ComputeArea(Cube(1.0, 1.0, 1.0, 1.0,
                                             1.0, 1.0, 1.0, 1.0));
      } else if (n == 1) {
        device_results(n) = ComputeArea(Cube(-0.5, 0.5, 0.5, -0.5,
                                             -0.5, 0.5, 0.5, -0.5, spacing));
      } else if (n == 2) {
        device_results(n) = ComputeArea(Cube(-0.5, -0.5, 0.5, 0.5,
                                             -0.5, -0.5, 0.5, 0.5, spacing));
      } else if (n == 3) {
        device_results(n) = ComputeArea(Cube(-0.5, -0.5, -0.5, -0.5,
                                             0.5, 0.5, 0.5, 0.5, spacing));
      } else {
        device_results(n) = ComputeArea(Cube(-1.5, -0.5, 0.5, -0.5,
                                             -0.5, 0.5, 1.5, 0.5));
      }
    });
    const auto host_results = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                                    device_results);
    failures += Check("device no surface", host_results(0), 0.0);
    failures += Check("device physical x plane", host_results(1), 15.0);
    failures += Check("device physical y plane", host_results(2), 10.0);
    failures += Check("device physical z plane", host_results(3), 6.0);
    failures += Check("device diagonal plane", host_results(4),
                      3.0*std::sqrt(3.0)/4.0);
  }
  Kokkos::finalize();
  return failures == 0 ? 0 : 1;
}
