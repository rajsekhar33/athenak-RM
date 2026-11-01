#ifndef FFT_LOCAL_FFT_HPP_
#define FFT_LOCAL_FFT_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file local_fft.hpp
//! \brief Batched, rank-local 1D FFT written in plain Kokkos (no external FFT library).
//!
//! Header-only, so no build-system change is needed.  Computation is always done in
//! double precision (Kokkos::complex<double>) regardless of the AthenaK Real type.
//!
//! CONVENTIONS
//!  - Forward: unnormalised DFT with numpy's sign, F_m = sum_n x_n exp(-2 pi i m n / N).
//!  - Inverse: unnormalised, exp(+2 pi i m n / N), so Inverse(Forward(x)) = N x.
//!  - ForwardR2C: the first N/2+1 forward coefficients of a real line (numpy.fft.rfft).
//!
//! DATA LAYOUT
//!  Batches of lines are stored line-major: a LayoutRight view of shape (nline, N), i.e.
//!  element n of line l lives at flat offset l*N + n and every line is contiguous.  This
//!  matches an x-pencil whose fastest index runs along the transform direction.  The
//!  flat-pointer entry points use exactly the same layout.
//!
//! ALGORITHM
//!  - A plan (LocalFFT object) is built once per length N: N is factorised into radices
//!    {8,4,2,3,5,7} and a length-N twiddle table exp(-2 pi i k/N) is built on the host in
//!    long double and copied to the device.
//!  - Default path: global-memory multi-pass Stockham autosort.  Each pass is one Kokkos
//!    kernel over (all lines) x (N/R butterflies); passes ping-pong between the user
//!    buffer and an internal work buffer, so no bit-reversal step is needed.  Per pass:
//!      v_r = src[j + r N/R] * W_{Ns R}^{r q},  q = j % Ns, g = j / Ns,
//!      dst[g Ns R + q + s Ns] = sum_r v_r W_R^{r s}
//!    where Ns is the product of the radices of the earlier passes.  Team scratch is not
//!    used (long lines do not fit in 64 KiB LDS on MI250X).
//!  - Lengths with a prime factor > 7 use Bluestein's chirp-z algorithm over the whole
//!    line: a power-of-two zero-padded circular convolution of length M >= 2N-1, carried
//!    out with two radix-path FFTs of length M plus a precomputed transformed chirp.
//!    Cost is therefore roughly 2 x (M/N) x log(M)/log(N) times a smooth-N FFT (~4-6x).
//!  - N = 1 is the identity.
//!
//! MEMORY
//!  Work buffers are allocated on first use and grown (never shrunk) to the largest batch
//!  seen: nline*N complex for the radix path, plus 2*nline*M for Bluestein, plus nline*N
//!  for ForwardR2C.  ReleaseBuffers() frees them.  A plan is not thread-safe.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"

namespace fft {

using Complex = Kokkos::complex<double>;
//! 2D batch of lines, shape (nline, N), line-major (LayoutRight)
using CplxArray2D = Kokkos::View<Complex **, Kokkos::LayoutRight, DevMemSpace>;
//! flat device buffer
using CplxArray1D = Kokkos::View<Complex *, Kokkos::LayoutRight, DevMemSpace>;
//! unmanaged flat view wrapping a raw device pointer
using CplxUView1D = Kokkos::View<Complex *, Kokkos::LayoutRight, DevMemSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using FFTPolicy = Kokkos::RangePolicy<DevExeSpace, Kokkos::IndexType<std::int64_t>>;

namespace detail {

//----------------------------------------------------------------------------------------
//! \fn void StockhamPass()
//! \brief One radix-R Stockham pass over all lines.  src and dst hold nline*n elements.
//! tw is the length-n table exp(-2 pi i k/n).  ns is the product of earlier radices.

template <int R>
inline void StockhamPass(CplxUView1D src, CplxUView1D dst, CplxArray1D tw,
                         std::int64_t nline, int n, int ns, int rgen) {
  // R > 0: compile-time radix (2, 4, 8); R == 0: generic radix rgen (3, 5, 7)
  const int rr = (R > 0) ? R : rgen;
  const int nb = n / rr;            // butterflies per line (= input stride)
  const int mstride = n / (ns*rr);  // twiddle stride: W_{ns R}^{r q} = tw(r q mstride)
  const std::int64_t ntot = nline * static_cast<std::int64_t>(nb);
  Kokkos::parallel_for("fft_stockham", FFTPolicy(DevExeSpace(), 0, ntot),
  KOKKOS_LAMBDA(const std::int64_t t) {
    // NB: plain if on the template constant (nvcc forbids first captures inside
    // if constexpr in extended lambdas); dead branches are removed by the compiler.
    const std::int64_t l = t / nb;
    const int j = static_cast<int>(t - l*nb);
    const int q = j % ns;
    const int g = j / ns;
    const std::int64_t base = l*n;
    if (R == 2) {
      Complex v0 = src(base + j);
      Complex v1 = src(base + j + nb);
      if (q != 0) v1 *= tw(q*mstride);
      const std::int64_t o = base + static_cast<std::int64_t>(g)*ns*2 + q;
      dst(o) = v0 + v1;
      dst(o + ns) = v0 - v1;
    } else if (R == 4) {
      Complex v0 = src(base + j);
      Complex v1 = src(base + j + nb);
      Complex v2 = src(base + j + 2*nb);
      Complex v3 = src(base + j + 3*nb);
      if (q != 0) {
        v1 *= tw(q*mstride);
        v2 *= tw(2*q*mstride);
        v3 *= tw(3*q*mstride);
      }
      const Complex a0 = v0 + v2, a1 = v0 - v2, a2 = v1 + v3;
      const Complex d = v1 - v3;
      const Complex a3(d.imag(), -d.real());  // (v1 - v3) * (-i)
      const std::int64_t o = base + static_cast<std::int64_t>(g)*ns*4 + q;
      dst(o) = a0 + a2;
      dst(o + ns) = a1 + a3;
      dst(o + 2*ns) = a0 - a2;
      dst(o + 3*ns) = a1 - a3;
    } else if (R == 8) {
      Complex v[8];
      v[0] = src(base + j);
      for (int r = 1; r < 8; ++r) {
        v[r] = src(base + j + r*nb);
        if (q != 0) v[r] *= tw(r*q*mstride);
      }
      // two radix-4 DFTs on even and odd samples, then a radix-2 combine
      Complex e[4], od[4];
      {
        const Complex a0 = v[0] + v[4], a1 = v[0] - v[4], a2 = v[2] + v[6];
        const Complex d = v[2] - v[6];
        const Complex a3(d.imag(), -d.real());
        e[0] = a0 + a2; e[1] = a1 + a3; e[2] = a0 - a2; e[3] = a1 - a3;
      }
      {
        const Complex a0 = v[1] + v[5], a1 = v[1] - v[5], a2 = v[3] + v[7];
        const Complex d = v[3] - v[7];
        const Complex a3(d.imag(), -d.real());
        od[0] = a0 + a2; od[1] = a1 + a3; od[2] = a0 - a2; od[3] = a1 - a3;
      }
      constexpr double h = 0.70710678118654752440084436210485;
      // W8^1 = (1-i)/sqrt2, W8^2 = -i, W8^3 = -(1+i)/sqrt2
      const Complex o1(h*(od[1].real() + od[1].imag()), h*(od[1].imag() - od[1].real()));
      const Complex o2(od[2].imag(), -od[2].real());
      const Complex o3(h*(od[3].imag() - od[3].real()), -h*(od[3].real() + od[3].imag()));
      const std::int64_t o = base + static_cast<std::int64_t>(g)*ns*8 + q;
      dst(o) = e[0] + od[0];
      dst(o + 4*ns) = e[0] - od[0];
      dst(o + ns) = e[1] + o1;
      dst(o + 5*ns) = e[1] - o1;
      dst(o + 2*ns) = e[2] + o2;
      dst(o + 6*ns) = e[2] - o2;
      dst(o + 3*ns) = e[3] + o3;
      dst(o + 7*ns) = e[3] - o3;
    } else {
      // generic small odd radix (3, 5, 7): O(R^2) DFT with W_R^k = tw(k n/R)
      Complex v[7];
      v[0] = src(base + j);
      for (int r = 1; r < rr; ++r) {
        v[r] = src(base + j + r*nb);
        if (q != 0) v[r] *= tw(r*q*mstride);
      }
      const std::int64_t o = base + static_cast<std::int64_t>(g)*ns*rr + q;
      for (int s = 0; s < rr; ++s) {
        Complex acc = v[0];
        int k = 0;
        for (int r = 1; r < rr; ++r) {
          k += s;
          if (k >= rr) k -= rr;
          acc += v[r]*tw(k*nb);
        }
        dst(o + static_cast<std::int64_t>(s)*ns) = acc;
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn std::vector<Complex> HostTwiddles()
//! \brief exp(sign * 2 pi i k / n), k = 0..n-1, evaluated in long double

inline std::vector<Complex> HostTwiddles(int n, int sign) {
  std::vector<Complex> w(n);
  const long double twopi = 6.283185307179586476925286766559005768L;
  for (int k = 0; k < n; ++k) {
    long double a = twopi * static_cast<long double>(k) / static_cast<long double>(n);
    w[k] = Complex(static_cast<double>(std::cos(a)),
                   static_cast<double>(sign*std::sin(a)));
  }
  return w;
}

inline CplxArray1D ToDevice(const std::vector<Complex> &h, const std::string &name) {
  CplxArray1D d(name, h.size());
  auto hm = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) hm(i) = h[i];
  Kokkos::deep_copy(d, hm);
  return d;
}

inline void EnsureSize(CplxArray1D &buf, std::int64_t n, const char *name) {
  if (static_cast<std::int64_t>(buf.extent(0)) < n) {
    buf = CplxArray1D();  // free first to lower peak memory
    buf = CplxArray1D(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(name)), n);
  }
}

}  // namespace detail

//----------------------------------------------------------------------------------------
//! \class LocalFFT
//! \brief Plan + executor for batched length-N complex FFTs on the default device.

class LocalFFT {
 public:
  explicit LocalFFT(int n) : n_(n) {
    if (n < 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "LocalFFT length must be >= 1, got " << n << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // factorise: radix 8 first (fewest passes), then 4, 2, then odd radices
    int m = n;
    while (m % 8 == 0) { radices_.push_back(8); m /= 8; }
    while (m % 4 == 0) { radices_.push_back(4); m /= 4; }
    while (m % 2 == 0) { radices_.push_back(2); m /= 2; }
    for (int r : {3, 5, 7}) {
      while (m % r == 0) { radices_.push_back(r); m /= r; }
    }
    if (m != 1) {
      // Bluestein over the whole line
      bluestein_ = true;
      radices_.clear();
      std::int64_t mm = 1;
      while (mm < 2*static_cast<std::int64_t>(n) - 1) mm *= 2;
      if (mm > (std::int64_t{1} << 30)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "LocalFFT length too large: " << n << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nconv_ = static_cast<int>(mm);
      sub_ = std::make_unique<LocalFFT>(nconv_);
      BuildBluestein();
    } else {
      tw_ = detail::ToDevice(detail::HostTwiddles(n_, -1), "fft_twiddle");
    }
  }

  int Size() const { return n_; }
  bool UsesBluestein() const { return bluestein_; }
  int ConvolutionSize() const { return nconv_; }
  //! radices of the Stockham passes (empty for N = 1 or Bluestein)
  const std::vector<int> &Radices() const { return radices_; }

  //! forward FFT in place on a (nline, N) LayoutRight view
  void Forward(const CplxArray2D &data) {
    CheckExtent(data.extent(1));
    Forward(data.data(), static_cast<std::int64_t>(data.extent(0)));
  }
  //! unnormalised inverse FFT in place on a (nline, N) LayoutRight view
  void Inverse(const CplxArray2D &data) {
    CheckExtent(data.extent(1));
    Inverse(data.data(), static_cast<std::int64_t>(data.extent(0)));
  }

  //! forward FFT in place on nline contiguous lines starting at device pointer p
  void Forward(Complex *p, std::int64_t nline) {
    if (nline <= 0 || n_ == 1) return;
    CplxUView1D d(p, nline*n_);
    if (bluestein_) {
      BluesteinForward(d, nline);
    } else {
      RadixForward(d, nline);
    }
  }

  //! unnormalised inverse via conj(Forward(conj(x)))
  void Inverse(Complex *p, std::int64_t nline) {
    if (nline <= 0 || n_ == 1) return;
    CplxUView1D d(p, nline*n_);
    Conjugate(d);
    Forward(p, nline);
    Conjugate(d);
  }

  //! Real-to-complex forward: in is (nline, N) of float or double, out is
  //! (nline, N/2+1).  Implemented by promotion to complex + full c2c, then truncation:
  //! about 2x the flops and 2x+ the memory traffic of a packed real transform.
  template <typename InView>
  void ForwardR2C(const InView &in, const CplxArray2D &out) {
    static_assert(InView::rank == 2, "ForwardR2C expects a rank-2 view");
    CheckExtent(in.extent(1));
    const std::int64_t nline = static_cast<std::int64_t>(in.extent(0));
    const int nh = n_/2 + 1;
    if (static_cast<int>(out.extent(1)) != nh ||
        static_cast<std::int64_t>(out.extent(0)) != nline) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "LocalFFT::ForwardR2C output must be (nline, N/2+1)"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (nline == 0) return;
    detail::EnsureSize(r2c_buf_, nline*n_, "fft_r2c_buf");
    auto buf = r2c_buf_;
    const int n = n_;
    Kokkos::parallel_for("fft_r2c_promote", FFTPolicy(DevExeSpace(), 0, nline*n),
    KOKKOS_LAMBDA(const std::int64_t t) {
      const std::int64_t l = t / n;
      const int i = static_cast<int>(t - l*n);
      buf(t) = Complex(static_cast<double>(in(l, i)), 0.0);
    });
    Forward(buf.data(), nline);
    Kokkos::parallel_for("fft_r2c_truncate", FFTPolicy(DevExeSpace(), 0, nline*nh),
    KOKKOS_LAMBDA(const std::int64_t t) {
      const std::int64_t l = t / nh;
      const int i = static_cast<int>(t - l*nh);
      out(l, i) = buf(l*n + i);
    });
  }

  //! device bytes held by this plan (tables and work buffers, including Bluestein's)
  std::size_t Bytes() const {
    std::size_t b = (tw_.extent(0) + chirp_.extent(0) + bhat_.extent(0) +
                     work_.extent(0) + conv_a_.extent(0) + r2c_buf_.extent(0)) *
                    sizeof(Complex);
    return b + (sub_ ? sub_->Bytes() : 0);
  }

  //! free all work buffers (they are re-allocated on next use)
  void ReleaseBuffers() {
    work_ = CplxArray1D();
    conv_a_ = CplxArray1D();
    r2c_buf_ = CplxArray1D();
    if (sub_) sub_->ReleaseBuffers();
  }

  // Implementation details below.  They are public only because CUDA extended lambdas
  // may not be defined inside private or protected member functions.

  void CheckExtent(std::size_t n) const {
    if (static_cast<int>(n) != n_) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "LocalFFT plan of length " << n_
                << " applied to lines of length " << n << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  static void Conjugate(CplxUView1D d) {
    const std::int64_t nt = static_cast<std::int64_t>(d.extent(0));
    Kokkos::parallel_for("fft_conj", FFTPolicy(DevExeSpace(), 0, nt),
    KOKKOS_LAMBDA(const std::int64_t t) { d(t) = Kokkos::conj(d(t)); });
  }

  void RadixForward(CplxUView1D data, std::int64_t nline) {
    const std::int64_t ntot = nline*n_;
    detail::EnsureSize(work_, ntot, "fft_work");
    CplxUView1D a = data;
    CplxUView1D b(work_.data(), ntot);
    int ns = 1;
    for (int r : radices_) {
      switch (r) {
        case 8: detail::StockhamPass<8>(a, b, tw_, nline, n_, ns, 8); break;
        case 4: detail::StockhamPass<4>(a, b, tw_, nline, n_, ns, 4); break;
        case 2: detail::StockhamPass<2>(a, b, tw_, nline, n_, ns, 2); break;
        default: detail::StockhamPass<0>(a, b, tw_, nline, n_, ns, r); break;
      }
      ns *= r;
      CplxUView1D tmp = a; a = b; b = tmp;
    }
    if (a.data() != data.data()) Kokkos::deep_copy(DevExeSpace(), data, a);
  }

  void BuildBluestein() {
    // chirp w_k = exp(-i pi k^2 / N); reduce k^2 mod 2N exactly in integers
    std::vector<Complex> w(n_);
    const long double pi = 3.141592653589793238462643383279502884L;
    const std::int64_t twon = 2*static_cast<std::int64_t>(n_);
    for (int k = 0; k < n_; ++k) {
      std::int64_t k2 = (static_cast<std::int64_t>(k)*k) % twon;
      long double a = pi*static_cast<long double>(k2)/static_cast<long double>(n_);
      w[k] = Complex(static_cast<double>(std::cos(a)), static_cast<double>(-std::sin(a)));
    }
    chirp_ = detail::ToDevice(w, "fft_chirp");
    std::vector<Complex> b(nconv_, Complex(0.0, 0.0));
    for (int k = 0; k < n_; ++k) {
      b[k] = Kokkos::conj(w[k]);
      if (k > 0) b[nconv_ - k] = Kokkos::conj(w[k]);
    }
    bhat_ = detail::ToDevice(b, "fft_bluestein_bhat");
    sub_->Forward(bhat_.data(), 1);
  }

  void BluesteinForward(CplxUView1D data, std::int64_t nline) {
    const int n = n_, m = nconv_;
    const std::int64_t ntot = nline*m;
    detail::EnsureSize(conv_a_, ntot, "fft_bluestein_a");
    auto a = conv_a_;
    auto w = chirp_;
    auto bh = bhat_;
    // a = [x * chirp, 0...]
    Kokkos::parallel_for("fft_bluestein_pre", FFTPolicy(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const std::int64_t t) {
      const std::int64_t l = t / m;
      const int i = static_cast<int>(t - l*m);
      a(t) = (i < n) ? data(l*n + i)*w(i) : Complex(0.0, 0.0);
    });
    sub_->Forward(a.data(), nline);
    // a = conj(A * Bhat), so that a second forward FFT gives conj(unnormalised inverse)
    Kokkos::parallel_for("fft_bluestein_mul", FFTPolicy(DevExeSpace(), 0, ntot),
    KOKKOS_LAMBDA(const std::int64_t t) {
      const std::int64_t l = t / m;
      const int i = static_cast<int>(t - l*m);
      a(t) = Kokkos::conj(a(t)*bh(i));
    });
    sub_->Forward(a.data(), nline);
    const double inv_m = 1.0/static_cast<double>(m);
    Kokkos::parallel_for("fft_bluestein_post",
                         FFTPolicy(DevExeSpace(), 0, nline*static_cast<std::int64_t>(n)),
    KOKKOS_LAMBDA(const std::int64_t t) {
      const std::int64_t l = t / n;
      const int i = static_cast<int>(t - l*n);
      data(t) = Kokkos::conj(a(l*m + i))*w(i)*inv_m;
    });
  }

 private:
  int n_;
  bool bluestein_ = false;
  int nconv_ = 0;                 // Bluestein convolution length (power of two)
  std::vector<int> radices_;
  CplxArray1D tw_;                // exp(-2 pi i k/N), radix path
  CplxArray1D chirp_;             // exp(-i pi k^2/N), Bluestein
  CplxArray1D bhat_;              // FFT_M of conj(chirp), wrapped, Bluestein
  CplxArray1D work_, conv_a_, r2c_buf_;
  std::unique_ptr<LocalFFT> sub_;  // radix-path plan of length nconv_
};

}  // namespace fft

#endif  // FFT_LOCAL_FFT_HPP_
