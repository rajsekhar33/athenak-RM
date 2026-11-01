//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft_unit_test.cpp
//! \brief Unit tests and benchmark for the in-house batched 1D FFT (fft/local_fft.hpp)
//!
//! Build with -D PROBLEM=fft_unit_test and run inputs/tests/fft_unit_test.athinput.  The
//! mesh is a trivial uniform hydro state; all work happens in UserProblem().  Checks:
//!  (a) forward FFT vs a direct O(N^2) DFT evaluated on the device in double,
//!  (b) single-mode inputs exp(2 pi i k0 n/N) -> N delta(k - k0) (phase check),
//!  (c) Inverse(Forward(x))/N == x,
//!  (d) ForwardR2C vs the first N/2+1 entries of the c2c result (and vs the direct DFT),
//!  (e) binary dumps for tests/fft/compare_numpy.py (numpy.fft.fft / rfft).
//! Optional benchmark (<problem>/bench = true): forward FFT throughput, ~2^27 elements.
//!
//! Dump format (little endian), one file <outdir>/fft_N<N>.bin per length:
//!   int64 N, int64 nline,
//!   complex128 x[nline][N]        (c2c input)
//!   complex128 F[nline][N]        (LocalFFT::Forward(x))
//!   float64    xr[nline][N]       (r2c input)
//!   complex128 R[nline][N/2+1]    (LocalFFT::ForwardR2C(xr))

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"
#include "fft/local_fft.hpp"

namespace {

using fft::Complex;
using fft::CplxArray1D;
using fft::CplxArray2D;
using RealArray2D = Kokkos::View<double **, Kokkos::LayoutRight, DevMemSpace>;

//----------------------------------------------------------------------------------------
//! direct DFT on the device: out(l,k) = sum_n in(l,n) W[(k n) mod N], W in long double

void DirectDFT(const CplxArray2D &in, const CplxArray2D &out) {
  const int n = static_cast<int>(in.extent(1));
  const std::int64_t nline = static_cast<std::int64_t>(in.extent(0));
  CplxArray1D w = fft::detail::ToDevice(fft::detail::HostTwiddles(n, -1), "dft_w");
  Kokkos::parallel_for("direct_dft", fft::FFTPolicy(DevExeSpace(), 0, nline*n),
  KOKKOS_LAMBDA(const std::int64_t t) {
    const std::int64_t l = t / n;
    const int k = static_cast<int>(t - l*n);
    double sr = 0.0, si = 0.0;
    std::int64_t idx = 0;
    for (int i = 0; i < n; ++i) {
      const Complex a = in(l, i)*w(idx);
      sr += a.real();
      si += a.imag();
      idx += k;
      if (idx >= n) idx -= n;
    }
    out(l, k) = Complex(sr, si);
  });
}

//! max_i |a_i - b_i| / max_i |b_i| over the whole batch (host mirrors)
template <typename HA, typename HB>
double RelErr(const HA &a, const HB &b, int nl, int nc) {
  double emax = 0.0, rmax = 0.0;
  for (int l = 0; l < nl; ++l) {
    for (int i = 0; i < nc; ++i) {
      emax = std::max(emax, Kokkos::abs(a(l, i) - b(l, i)));
      rmax = std::max(rmax, Kokkos::abs(b(l, i)));
    }
  }
  return (rmax > 0.0) ? emax/rmax : emax;
}

std::string RadixString(const fft::LocalFFT &p) {
  std::ostringstream s;
  if (p.UsesBluestein()) {
    s << "bluestein(M=" << p.ConvolutionSize() << ")";
  } else if (p.Radices().empty()) {
    s << "identity";
  } else {
    for (std::size_t i = 0; i < p.Radices().size(); ++i) {
      s << (i ? "x" : "") << p.Radices()[i];
    }
  }
  return s.str();
}

//----------------------------------------------------------------------------------------
//! run all correctness tests for one length; returns worst error

double TestLength(int n, int nline, double tol, std::mt19937_64 &rng,
                  const std::string &outdir, bool dump, int &nfail) {
  const bool root = (global_variable::my_rank == 0);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  fft::LocalFFT plan(n);
  const int nh = n/2 + 1;

  CplxArray2D x("x", nline, n), y("y", nline, n), ref("ref", nline, n);
  auto hx = Kokkos::create_mirror_view(x);
  for (int l = 0; l < nline; ++l) {
    for (int i = 0; i < n; ++i) hx(l, i) = Complex(uni(rng), uni(rng));
  }
  Kokkos::deep_copy(x, hx);

  // (a) forward vs direct DFT
  DirectDFT(x, ref);
  Kokkos::deep_copy(y, x);
  plan.Forward(y);
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto href = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref);
  const double e_fwd = RelErr(hy, href, nline, n);

  // (c) round trip
  CplxArray2D z("z", nline, n);
  Kokkos::deep_copy(z, y);
  plan.Inverse(z);
  auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
  for (int l = 0; l < nline; ++l) {
    for (int i = 0; i < n; ++i) hz(l, i) /= static_cast<double>(n);
  }
  const double e_inv = RelErr(hz, hx, nline, n);

  // (b) single modes exp(+2 pi i k0 n/N) -> N delta(k - k0); one mode per line
  std::vector<int> k0s = {0, 1, n/3, n/2, n - 1};
  const int nmode = static_cast<int>(k0s.size());
  CplxArray2D sm("single_mode", nmode, n);
  auto hsm = Kokkos::create_mirror_view(sm);
  auto wp = fft::detail::HostTwiddles(n, +1);
  for (int l = 0; l < nmode; ++l) {
    for (int i = 0; i < n; ++i) {
      hsm(l, i) = wp[(static_cast<std::int64_t>(k0s[l])*i) % n];
    }
  }
  Kokkos::deep_copy(sm, hsm);
  plan.Forward(sm);
  auto hsm_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sm);
  double e_mode = 0.0;
  for (int l = 0; l < nmode; ++l) {
    for (int k = 0; k < n; ++k) {
      Complex expect = (k == k0s[l] % n) ? Complex(n, 0.0) : Complex(0.0, 0.0);
      e_mode = std::max(e_mode, Kokkos::abs(hsm_out(l, k) - expect)/n);
    }
  }

  // (d) r2c vs c2c of promoted input and vs direct DFT
  RealArray2D xr("xr", nline, n);
  auto hxr = Kokkos::create_mirror_view(xr);
  CplxArray2D xc("xc", nline, n), refr("refr", nline, n), rout("rout", nline, nh);
  auto hxc = Kokkos::create_mirror_view(xc);
  for (int l = 0; l < nline; ++l) {
    for (int i = 0; i < n; ++i) {
      hxr(l, i) = uni(rng);
      hxc(l, i) = Complex(hxr(l, i), 0.0);
    }
  }
  Kokkos::deep_copy(xr, hxr);
  Kokkos::deep_copy(xc, hxc);
  DirectDFT(xc, refr);
  plan.Forward(xc);
  plan.ForwardR2C(xr, rout);
  auto hrout = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rout);
  auto hxcf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xc);
  auto hrefr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), refr);
  const double e_r2c_c2c = RelErr(hrout, hxcf, nline, nh);
  const double e_r2c_dft = RelErr(hrout, hrefr, nline, nh);

  const double worst = std::max({e_fwd, e_inv, e_mode, e_r2c_c2c, e_r2c_dft});
  const bool pass = (worst <= tol);
  if (!pass) ++nfail;
  if (root) {
    std::cout << (pass ? "PASS" : "FAIL") << "  N=" << std::setw(5) << n
              << "  plan=" << std::left << std::setw(18) << RadixString(plan)
              << std::right << std::scientific << std::setprecision(2)
              << "  fwd_vs_dft=" << e_fwd << "  single_mode=" << e_mode
              << "  inv(fwd)/N=" << e_inv << "  r2c_vs_c2c=" << e_r2c_c2c
              << "  r2c_vs_dft=" << e_r2c_dft << std::defaultfloat << std::endl;
  }

  // (e) dump for numpy
  if (dump && root) {
    std::string fname = outdir + "/fft_N" + std::to_string(n) + ".bin";
    FILE *fp = std::fopen(fname.c_str(), "wb");
    if (fp == nullptr) {
      std::cout << "FAIL  could not open " << fname << std::endl;
      ++nfail;
    } else {
      std::int64_t hdr[2] = {n, nline};
      std::fwrite(hdr, sizeof(std::int64_t), 2, fp);
      for (int l = 0; l < nline; ++l) {
        for (int i = 0; i < n; ++i) {
          double v[2] = {hx(l, i).real(), hx(l, i).imag()};
          std::fwrite(v, sizeof(double), 2, fp);
        }
      }
      for (int l = 0; l < nline; ++l) {
        for (int i = 0; i < n; ++i) {
          double v[2] = {hy(l, i).real(), hy(l, i).imag()};
          std::fwrite(v, sizeof(double), 2, fp);
        }
      }
      for (int l = 0; l < nline; ++l) {
        for (int i = 0; i < n; ++i) {
          double v = hxr(l, i);
          std::fwrite(&v, sizeof(double), 1, fp);
        }
      }
      for (int l = 0; l < nline; ++l) {
        for (int i = 0; i < nh; ++i) {
          double v[2] = {hrout(l, i).real(), hrout(l, i).imag()};
          std::fwrite(v, sizeof(double), 2, fp);
        }
      }
      std::fclose(fp);
    }
  }
  return worst;
}

//----------------------------------------------------------------------------------------
//! forward-FFT throughput: nline x N with nline*N ~ 2^log2_elems

void Benchmark(int log2_elems, int nrep, const std::vector<int> &lengths) {
  const bool root = (global_variable::my_rank == 0);
  const std::int64_t target = std::int64_t{1} << log2_elems;
  if (root) {
    std::cout << std::endl << "BENCHMARK forward c2c, ~2^" << log2_elems
              << " complex<double> per call, " << nrep << " reps, exec space "
              << DevExeSpace::name() << std::endl;
    std::cout << "      N     nline  plan                 ms/call  ms(min)   ns/elem  "
              << "GB/s(1R+1W)  GB/s(all passes)" << std::endl;
  }
  for (int n : lengths) {
    const std::int64_t nline = std::max<std::int64_t>(1, target / n);
    fft::LocalFFT plan(n);
    CplxArray2D d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "bench"), nline, n);
    Kokkos::parallel_for("bench_fill", fft::FFTPolicy(DevExeSpace(), 0, nline*n),
    KOKKOS_LAMBDA(const std::int64_t t) {
      d.data()[t] = Complex(1.0e-3*static_cast<double>(t % 977), 0.5);
    });
    plan.Forward(d);  // warm-up (allocates work buffers)
    Kokkos::fence();
    // mean over nrep back-to-back calls, and the fastest individually fenced call
    Kokkos::Timer timer;
    for (int r = 0; r < nrep; ++r) plan.Forward(d);
    Kokkos::fence();
    const double t = timer.seconds()/nrep;
    double tmin = 1.0e30;
    for (int r = 0; r < nrep; ++r) {
      Kokkos::Timer t1;
      plan.Forward(d);
      Kokkos::fence();
      tmin = std::min(tmin, t1.seconds());
    }
    const double nel = static_cast<double>(nline)*n;
    const double bytes1 = 2.0*sizeof(Complex)*nel;
    int npass = static_cast<int>(plan.Radices().size());
    if (!plan.UsesBluestein() && npass % 2 == 1) npass += 1;  // final copy back
    double bytes_all = bytes1*npass;
    if (plan.UsesBluestein()) {
      // pre + post pointwise kernels on N, mul on M, two sub-FFTs of length M
      const double m = plan.ConvolutionSize();
      fft::LocalFFT sub(plan.ConvolutionSize());
      int sp = static_cast<int>(sub.Radices().size());
      if (sp % 2 == 1) sp += 1;
      bytes_all = 2.0*bytes1 + 2.0*sizeof(Complex)*nline*m*(2.0*sp + 2.0);
    }
    if (root) {
      std::cout << std::setw(7) << n << std::setw(10) << nline << "  "
                << std::left << std::setw(20) << RadixString(plan) << std::right
                << std::fixed << std::setprecision(3) << std::setw(9) << t*1.0e3
                << std::setw(9) << tmin*1.0e3
                << std::setw(10) << t*1.0e9/nel << std::setprecision(1)
                << std::setw(13) << bytes1/t*1.0e-9 << std::setw(18)
                << bytes_all/t*1.0e-9 << std::defaultfloat << std::endl;
    }
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief sets a uniform hydro state, then runs the FFT tests (and optional benchmark)

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro != nullptr) {
    auto &indcs = pmy_mesh_->mb_indcs;
    int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    auto &u0 = pmbp->phydro->u0;
    const bool ideal = pmbp->phydro->peos->eos_data.is_ideal;
    const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
    par_for("fft_test_ic", DevExeSpace(), 0, (pmbp->nmb_thispack-1), ks, ke, js, je,
            is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      u0(m, IDN, k, j, i) = 1.0;
      u0(m, IM1, k, j, i) = 0.0;
      u0(m, IM2, k, j, i) = 0.0;
      u0(m, IM3, k, j, i) = 0.0;
      if (ideal) u0(m, IEN, k, j, i) = 1.0/gm1;
    });
  }

  const int nline = pin->GetOrAddInteger("problem", "nline", 5);
  const Real tol = pin->GetOrAddReal("problem", "tol", 1.0e-12);
  const bool dump = pin->GetOrAddBoolean("problem", "dump", true);
  std::string outdir = pin->GetOrAddString("problem", "outdir", "fft_test_out");
  const bool run_tests = pin->GetOrAddBoolean("problem", "run_tests", true);
  const bool bench = pin->GetOrAddBoolean("problem", "bench", false);
  const int bench_log2 = pin->GetOrAddInteger("problem", "bench_log2_elems", 27);
  const int bench_nrep = pin->GetOrAddInteger("problem", "bench_nrep", 10);
  const bool root = (global_variable::my_rank == 0);

  int nfail = 0;
  if (run_tests) {
    if (dump && root) mkdir(outdir.c_str(), 0775);
    std::vector<int> lengths = {1, 2, 3, 4, 5, 7, 8, 11, 16, 30, 64, 96, 127, 128, 243,
                                251, 256, 640, 1000, 1024, 2039, 2048, 3000, 4096, 6144,
                                8191, 8192};
    std::mt19937_64 rng(20261025);
    if (root) {
      std::cout << std::endl << "LocalFFT unit tests (exec space " << DevExeSpace::name()
                << ", nline=" << nline << ", tol=" << tol << ")" << std::endl;
    }
    double worst = 0.0;
    for (int n : lengths) {
      worst = std::max(worst, TestLength(n, nline, tol, rng, outdir, dump, nfail));
    }
    if (root) {
      std::cout << (nfail == 0 ? "ALL PASS" : "SOME FAILED") << "  failures=" << nfail
                << "  worst_rel_err=" << std::scientific << std::setprecision(3) << worst
                << std::defaultfloat << std::endl;
    }
  }
  if (bench) {
    Benchmark(bench_log2, bench_nrep, {128, 256, 512, 1000, 1024, 2039, 2048, 4096});
  }
  return;
}
