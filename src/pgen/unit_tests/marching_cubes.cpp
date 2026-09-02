//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file marching_cubes.cpp
//! \brief Functional regression test for temperature-isosurface area integration.
//!
//! The test deliberately follows the in-situ RadShock/TRML diagnostic convention:
//! it reconstructs log10[(gamma-1) * w0(IEN) / w0(IDN)] at each of the eight cell
//! corners, subtracts log10(Tiso), and sums marching-cubes areas.  The input has a
//! planar T=1 isosurface normal to x1, periodic transverse directions, anisotropic
//! cells, and enough x2/x3 MeshBlocks to distribute the reduction across MPI ranks.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "utils/marching_cubes.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::MarchingCubes()
//! \brief Functional temperature-isosurface test using the hydro primitive state.

void ProblemGenerator::MarchingCubes(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Marching-cubes temperature test requires a <hydro> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!pmbp->phydro->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Marching-cubes temperature test requires an ideal EOS." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  if (indcs.nx1 < 2 || indcs.nx2 < 1 || indcs.nx3 < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Marching-cubes temperature test requires a 3D mesh with nx1 >= 2."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  auto &size = pmbp->pmb->mb_size;
  auto &w0 = pmbp->phydro->w0;
  auto &u0 = pmbp->phydro->u0;

  // Set log10(T) = x1 at all active and ghost-zone cell centers.  The supplied
  // input has x1 in [-1,1], an even nx1, and Tiso=1, so no sampled corner lies
  // exactly on the isosurface.  Filling transverse ghost zones makes the caller's
  // periodic x2/x3 cube ownership explicit rather than relying on task ordering.
  par_for("pgen_marching_cubes_temperature", DevExeSpace(), 0, nmb - 1,
          ks - ng, ke + ng, js - ng, je + ng, is - ng, ie + ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real x1 = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                                size.d_view(m).x1max);
    const Real temperature = std::pow(10.0, x1);
    w0(m, IDN, k, j, i) = 1.0;
    w0(m, IM1, k, j, i) = 0.0;
    w0(m, IM2, k, j, i) = 0.0;
    w0(m, IM3, k, j, i) = 0.0;
    w0(m, IEN, k, j, i) = temperature / gm1;
  });
  Kokkos::fence();
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  // Each active (j,k) owns one transverse cell.  The x1 range ends at ie-1 so
  // all x1 corner data are active; x2/x3 high-side corners use the initialized
  // periodic ghosts.  This gives one, and only one, ownership of each x2-x3 cell.
  const int ni = nx1 - 1;
  const int nji = nx2 * ni;
  const int nkji = nx3 * nji;
  const int nmkji = nmb * nkji;
  Real local_area = 0.0;
  Kokkos::parallel_reduce(
      "pgen_marching_cubes_temperature_area", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &area) {
        const int m = idx / nkji;
        const int k = (idx - m * nkji) / nji + ks;
        const int j = (idx - m * nkji - (k - ks) * nji) / ni + js;
        const int i = idx - m * nkji - (k - ks) * nji - (j - js) * ni + is;

        const Real v0 = std::log10(gm1 * w0(m, IEN, k,     j,     i    ) /
                                    w0(m, IDN, k,     j,     i    ));
        const Real v1 = std::log10(gm1 * w0(m, IEN, k,     j,     i + 1) /
                                    w0(m, IDN, k,     j,     i + 1));
        const Real v2 = std::log10(gm1 * w0(m, IEN, k,     j + 1, i + 1) /
                                    w0(m, IDN, k,     j + 1, i + 1));
        const Real v3 = std::log10(gm1 * w0(m, IEN, k,     j + 1, i    ) /
                                    w0(m, IDN, k,     j + 1, i    ));
        const Real v4 = std::log10(gm1 * w0(m, IEN, k + 1, j,     i    ) /
                                    w0(m, IDN, k + 1, j,     i    ));
        const Real v5 = std::log10(gm1 * w0(m, IEN, k + 1, j,     i + 1) /
                                    w0(m, IDN, k + 1, j,     i + 1));
        const Real v6 = std::log10(gm1 * w0(m, IEN, k + 1, j + 1, i + 1) /
                                    w0(m, IDN, k + 1, j + 1, i + 1));
        const Real v7 = std::log10(gm1 * w0(m, IEN, k + 1, j + 1, i    ) /
                                    w0(m, IDN, k + 1, j + 1, i    ));
        const marching_cubes::CellSpacing spacing(size.d_view(m).dx1,
                                                   size.d_view(m).dx2,
                                                   size.d_view(m).dx3);
        const marching_cubes::Cube cube(v0, v1, v2, v3, v4, v5, v6, v7, spacing);
        area += marching_cubes::ComputeArea(cube);
      }, Kokkos::Sum<Real>(local_area));

  Real global_area = local_area;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_area, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  const Real expected_area = (pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min) *
                             (pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min);
  const Real error = std::abs(global_area - expected_area);
  const Real tolerance = 128.0 * std::numeric_limits<Real>::epsilon() *
                         std::max(1.0, std::abs(expected_area));
  const bool passed = std::isfinite(global_area) && error <= tolerance;

  if (global_variable::my_rank == 0) {
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1)
              << "MARCHING_CUBES_TEMPERATURE_AREA " << (passed ? "PASS" : "FAIL")
              << ": area=" << global_area << " expected=" << expected_area
              << " abs_error=" << error << " tolerance=" << tolerance
              << " ranks=" << global_variable::nranks << std::endl;
  }
  if (!passed) std::exit(EXIT_FAILURE);
}
