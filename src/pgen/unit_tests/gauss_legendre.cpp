//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gauss_legendre.cpp
//  \brief Problem generator for unit tests of Gauss-Legendre integrations

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>
#include <cctype>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "utils/spherical_harm.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::GaussLegendre()
//! \brief problem generator for unit tests of Gauss-Legendre integration
//! Tests that the cross integral of spherical harmonics are delta functions.

void ProblemGenerator::GaussLegendre(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  int ntheta = pin->GetOrAddInteger("problem", "ntheta", 16);

  GaussLegendreGrid grid(pmbp, ntheta, 1);

  // Cache every resolvable harmonic once. Randomly selecting ten modes previously
  // missed cancellation errors near l=ntheta-1, making the test intermittent.
  const int nmodes = ntheta*ntheta;
  std::vector<std::vector<Real>> ylm_r(nmodes, std::vector<Real>(grid.nangles));
  std::vector<std::vector<Real>> ylm_i(nmodes, std::vector<Real>(grid.nangles));
  for (int l = 0; l < ntheta; ++l) {
    for (int m = -l; m <= l; ++m) {
      const int index = l*l + l + m;
      for (int ip = 0; ip < grid.nangles; ++ip) {
        SWSphericalHarm(&ylm_r[index][ip], &ylm_i[index][ip], l, m, 0,
                        grid.polar_pos.h_view(ip,0), grid.polar_pos.h_view(ip,1));
      }
    }
  }

  double max_err = 0;
  auto check_pair = [&](int l1, int m1, int l2, int m2) {
    const int n1 = l1*l1 + l1 + m1;
    const int n2 = l2*l2 + l2 + m2;
    double int_r = 0;
    double int_i = 0;
    for (int ip = 0; ip < grid.nangles; ++ip) {
      const Real weight = grid.int_weights.h_view(ip);
      int_r += weight*(ylm_r[n1][ip]*ylm_r[n2][ip] + ylm_i[n1][ip]*ylm_i[n2][ip]);
      int_i += weight*(ylm_i[n1][ip]*ylm_r[n2][ip] - ylm_r[n1][ip]*ylm_i[n2][ip]);
    }
    const double expected = (n1 == n2) ? 1.0 : 0.0;
    const double err_r = std::abs(int_r - expected);
    const double err_i = std::abs(int_i);
    max_err = std::max(max_err, std::max(err_r, err_i));
    if (!std::isfinite(int_r) || !std::isfinite(int_i)
        || err_r >= 1e-10 || err_i >= 1e-10) {
      std::cout << "Gauss Legendre Integral Test Failed"<< std::endl;
      std::cout << "l1=" << l1 << '\t' << "m1=" << m1 << std::endl;
      std::cout << "l2=" << l2 << '\t' << "m2=" << m2 << std::endl;
      std::cout << "Maximum Error is " << max_err << std::endl;
      std::cout << std::endl;
      exit(EXIT_FAILURE);
    }
  };

  // Every norm, including all formerly failing high-l modes, and every same-m
  // cross-degree pair test the polar quadrature deterministically.
  for (int m = 1 - ntheta; m < ntheta; ++m) {
    for (int l1 = std::abs(m); l1 < ntheta; ++l1) {
      for (int l2 = l1; l2 < ntheta; ++l2) check_pair(l1, m, l2, m);
    }
  }
  // All different-m pairs at the highest degree cover every azimuthal frequency
  // difference up to 2*(ntheta-1), rather than relying on a random sample.
  const int lmax = ntheta - 1;
  for (int m1 = -lmax; m1 <= lmax; ++m1) {
    for (int m2 = m1 + 1; m2 <= lmax; ++m2) check_pair(lmax, m1, lmax, m2);
  }
  std::cout << "Test Passed with Maximum Error is " << max_err << std::endl;

  return;
}
