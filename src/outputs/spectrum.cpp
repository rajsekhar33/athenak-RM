//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectrum.cpp
//  \brief on-the-fly power spectra (file_type = spec); see spectrum.hpp.
//
//  Stored per output (all float64; E, nk are sums over the full set of modes, i.e. every
//  retained kx > 0 mode also counts its conjugate partner):
//    kbins[nbin+1]  actual bin edges (Python _spectral_grid)
//    k[nbin]        representative k (midpoint, or geometric mean for log bins)
//    nk[nbin]       number of modes per bin
//    scalar:    E[nweight, nvar, nbin], P_DC[nweight, nvar], sumsq[nvar]
//    helmholtz: E_comp[nbin], E_sol[nbin], P_DC, sumsq
//  sumsq is the sum of the (windowed) squared field over the transformed cells, the
//  Parseval reference: sum(E) = sumsq for linear bins (which include k = 0) and
//  sumsq - P_DC for log bins (which exclude it), for unweighted spectra only.

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"
#include "spectrum.hpp"

namespace {

constexpr int kSpecMaxVal = 8;   // values binned per mode in one pass

void SpecFatal(const std::string &block, const std::string &msg) {
  std::cout << "### FATAL ERROR in spec output block '" << block << "'" << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//! numpy fftfreq label of index m on an axis of length n (Nyquist negative)
KOKKOS_INLINE_FUNCTION
int SpecLabel(int m, int n) {
  return (m <= (n - 1)/2) ? m : m - n;
}

//! k component exactly as Python forms it: fftfreq(n)*n/L = ((label*(1/n))*n)/L
KOKKOS_INLINE_FUNCTION
double SpecK(int label, int n, double len) {
  return ((static_cast<double>(label)*(1.0/n))*n)/len;
}

//! numpy.histogram bin of k for edges e[0..nb]: [e_i, e_i+1), last bin closed; -1 when
//! outside
KOKKOS_INLINE_FUNCTION
int SpecBin(const DvceArray1D<double> &e, int nb, double k) {
  if (!(k >= e(0))) return -1;
  const double last = e(nb);
  if (k >= last) {
    // The host edge and device norm can round differently at the Fourier corner.
    // Keep that closed endpoint inclusive within double-precision roundoff.
    return (k - last <= 8.0*DBL_EPSILON*last) ? nb - 1 : -1;
  }
  int lo = 0, hi = nb;
  while (hi - lo > 1) {
    int mid = (lo + hi)/2;
    if (e(mid) <= k) {lo = mid;} else {hi = mid;}
  }
  return lo;
}

bool SpecWriteHeader(FILE *pfile, const std::string &meta) {
  bool ok = (std::fwrite(meta.data(), 1, meta.size(), pfile) == meta.size());
  char dummy[64];
  int line_len = std::snprintf(dummy, sizeof(dummy), "header offset=%012ld\n", 0L);
  int64_t header_offset = static_cast<int64_t>(std::ftell(pfile)) + line_len;
  ok &= (std::fprintf(pfile, "header offset=%012lld\n",
                      static_cast<long long>(header_offset)) > 0);  // NOLINT
  return ok;
}

//! per-mode values of one binning pass (see SpectrumOutput::BinModes)
struct SpecModeEval {
  fft::SpectralSlab slab;
  DvceArray1D<double> edges;
  fft::CplxArray1D c0, c1, c2;
  DvceArray1D<double> pw[kSpecMaxVal];
  bool hasw[kSpecMaxVal];
  int kind, nval, nb, nx, ny, nz;
  double lx, ly, lz, ninv;

  //! bin b of mode f (-1 if outside the edges), whether it is k = 0, and its values
  KOKKOS_INLINE_FUNCTION
  bool Eval(std::int64_t f, int &b, bool &dc, double *v) const {
    int gx, gy, gz;
    slab.Global(f, gx, gy, gz);
    const double kx = SpecK(SpecLabel(gx, nx), nx, lx);
    const double ky = SpecK(SpecLabel(gy, ny), ny, ly);
    const double kz = SpecK(SpecLabel(gz, nz), nz, lz);
    const double kk = Kokkos::sqrt(kx*kx + ky*ky + kz*kz);
    b = SpecBin(edges, nb, kk);
    dc = (gx == 0 && gy == 0 && gz == 0);
    if (b < 0 && !dc) return false;
    const bool twin = !(gx == 0 || (nx % 2 == 0 && gx == nx/2));
    const double h = twin ? 2.0 : 1.0;
    if (kind == 0) {
      v[0] = h;
    } else if (kind == 1) {
      const double pq = (c0(f).real()*c0(f).real() + c0(f).imag()*c0(f).imag())*ninv;
      for (int iw = 0; iw < nval; ++iw) v[iw] = h*pq*(hasw[iw] ? pw[iw](f) : 1.0);
    } else {
      // compressive (k.F)k/|k|^2 and solenoidal F - comp, for the mode and, when it
      // stands for two, for its partner conj(F) at the negated indices
      v[0] = v[1] = 0.0;
      for (int pass = 0; pass < (twin ? 2 : 1); ++pass) {
        double qx = kx, qy = ky, qz = kz, sgn = 1.0;
        if (pass == 1) {
          qx = SpecK(SpecLabel((nx - gx) % nx, nx), nx, lx);
          qy = SpecK(SpecLabel((ny - gy) % ny, ny), ny, ly);
          qz = SpecK(SpecLabel((nz - gz) % nz, nz), nz, lz);
          sgn = -1.0;   // conjugate
        }
        const double fr[3] = {c0(f).real(), c1(f).real(), c2(f).real()};
        const double fi[3] = {sgn*c0(f).imag(), sgn*c1(f).imag(), sgn*c2(f).imag()};
        const double q2 = qx*qx + qy*qy + qz*qz;
        double sr = 0.0, si = 0.0;
        if (q2 > 0.0) {
          sr = (qx*fr[0] + qy*fr[1] + qz*fr[2])/q2;
          si = (qx*fi[0] + qy*fi[1] + qz*fi[2])/q2;
        }
        const double qv[3] = {qx, qy, qz};
        for (int c = 0; c < 3; ++c) {
          const double cr = sr*qv[c], ci = si*qv[c];
          v[0] += (cr*cr + ci*ci)*ninv;
          v[1] += ((fr[c] - cr)*(fr[c] - cr) + (fi[c] - ci)*(fi[c] - ci))*ninv;
        }
      }
    }
    return true;
  }
};

}  // namespace

//----------------------------------------------------------------------------------------
// ctor: parse, build the selection, k bins, window and the transform plan

SpectrumOutput::SpectrumOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
    BaseTypeOutput(pin, pm, op), time_setup_(0.0), time_fft_(0.0), time_bin_(0.0),
    bin_passes_(0.0), bin_global_(0.0),
    mem_device_(0.0), mem_host_(0.0) {
  for (int i = 0; i < 6; ++i) time_stage_[i] = 0.0;
  const std::string &blk = out_params.block_name;
  if (pm->multilevel) SpecFatal(blk, "spec output supports uniform grids only");

  std::string mode = pin->GetOrAddString(blk, "mode", "scalar");
  if (mode.compare("scalar") == 0) {
    mode_ = 0;
  } else if (mode.compare("helmholtz") == 0) {
    mode_ = 1;
  } else {
    SpecFatal(blk, "mode='" + mode + "' must be one of scalar, helmholtz");
  }
  var_names_ = SplitOutputList(pin->GetString(blk, "variables"));
  weight_names_ = SplitOutputList(pin->GetOrAddString(blk, "weights", "ones"));
  if (weight_names_.empty()) weight_names_.push_back("ones");
  nvar_ = static_cast<int>(var_names_.size());
  nweight_ = static_cast<int>(weight_names_.size());
  if (mode_ == 1) {
    if (nvar_ != 3) {
      SpecFatal(blk, "mode=helmholtz needs exactly three variables (x1, x2, x3 "
                     "components)");
    }
    if (nweight_ != 1 || weight_names_[0].compare("ones") != 0) {
      SpecFatal(blk, "mode=helmholtz takes no weights (as spectra.py)");
    }
  }
  if (nweight_*nvar_ > kSpecMaxVal) {
    SpecFatal(blk, "at most " + std::to_string(kSpecMaxVal) +
                   " (weight, variable) combinations per spec output");
  }
  if (out_params.contains_derived) {
    SpecFatal(blk, "derived variables are not yet supported by spec output");
  }
  for (std::size_t n = 0; n < outvars_per_name.size(); ++n) {
    if (outvars_per_name[n] != 1) {
      SpecFatal(blk, "'" + out_params.variable_list[n] + "' gives " +
                     std::to_string(outvars_per_name[n]) + " field components here; "
                     "every spec variable/weight must name exactly one");
    }
  }
  for (std::size_t n = 0; n < outvars.size(); ++n) {
    field_arr_.push_back(*(outvars[n].data_ptr));
    field_index_.push_back(outvars[n].data_index);
  }
  weight_field_.assign(nweight_, -1);
  for (int iw = 0; iw < nweight_; ++iw) {
    if (weight_names_[iw].compare("ones") == 0) continue;
    for (std::size_t n = 0; n < out_params.variable_list.size(); ++n) {
      if (out_params.variable_list[n] == weight_names_[iw]) {
        weight_field_[iw] = static_cast<int>(n);
        break;
      }
    }
  }

  // ---- selection: whole mesh, or z cropped by 'skip' at both ends (Python
  // get_refined_data index rule, int() truncation) ----
  strat_ = pin->GetOrAddBoolean(blk, "strat", false);
  skip_ = GetOutputReal(pin, blk, "skip", 0.0);
  auto &ms = pm->mesh_size;
  const int nmesh[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  const double box[3] = {ms.x1max - ms.x1min, ms.x2max - ms.x2min, ms.x3max - ms.x3min};
  for (int d = 0; d < 3; ++d) {
    lo_[d] = 0;
    hi_[d] = nmesh[d];
  }
  if (strat_) {
    const double fac = nmesh[2]/box[2];
    lo_[2] = static_cast<int>(((ms.x3min + skip_) - ms.x3min)*fac);
    hi_[2] = static_cast<int>(((ms.x3max - skip_) - ms.x3min)*fac);
    if (hi_[2] <= lo_[2]) SpecFatal(blk, "skip leaves no cells in x3");
  }
  for (int d = 0; d < 3; ++d) {
    n_[d] = hi_[d] - lo_[d];
    len_[d] = (n_[d] == nmesh[d]) ? box[d] : n_[d]*(box[d]/nmesh[d]);
  }

  // ---- k bins: Python _spectral_grid ----
  nbin_ = pin->GetOrAddInteger(blk, "nbin", mode_ == 0 ? 256 : 200);
  logscale_ = pin->GetOrAddBoolean(blk, "logscale", false);
  {
    double kabs[3], kmin_len = 1.0e300, kmin_n = 1.0e300;
    bool any_active = (n_[0] > 1 || n_[1] > 1 || n_[2] > 1);
    for (int d = 0; d < 3; ++d) {
      kabs[d] = 0.0;
      for (int m = 0; m < n_[d]; ++m) {
        kabs[d] = std::max(kabs[d], std::abs(SpecK(SpecLabel(m, n_[d]), n_[d], len_[d])));
      }
      if (n_[d] > 1 || !any_active) {
        kmin_n = std::min(kmin_n, n_[d]/len_[d]);
        kmin_len = std::min(kmin_len, 1.0/len_[d]);
      }
    }
    const double kcorner = std::sqrt(kabs[0]*kabs[0] + kabs[1]*kabs[1] + kabs[2]*kabs[2]);
    const double kmax = std::max(kmin_n, kcorner);
    if (logscale_) {
      if (nbin_ < 1) SpecFatal(blk, "nbin must be >= 1");
      const double a = std::log10(0.25*kmin_len), b = std::log10(kmax);
      kedges_.resize(nbin_ + 1);
      const double step = (b - a)/nbin_;
      for (int i = 0; i <= nbin_; ++i) kedges_[i] = std::pow(10.0, i*step + a);
      // spectra.py uses 10**log10(kmax), which can round 1 ulp below the corner |k|
      // and drop the corner modes; keep the final edge inclusive (only change from
      // Python, so every non-DC mode is binned and Parseval holds exactly)
      kedges_[nbin_] = std::max(std::pow(10.0, b), kmax);
      for (int i = 0; i < nbin_; ++i) kmid_.push_back(std::sqrt(kedges_[i]*kedges_[i+1]));
    } else {
      // np.linspace(0, kmax, nbin): nbin edges, nbin-1 bins
      if (nbin_ < 2) SpecFatal(blk, "nbin must be >= 2 for linear bins");
      kedges_.resize(nbin_);
      const double step = kmax/(nbin_ - 1);
      for (int i = 0; i < nbin_; ++i) kedges_[i] = i*step;
      kedges_[nbin_ - 1] = kmax;
      nbin_ -= 1;   // histogram bins from here on
      for (int i = 0; i < nbin_; ++i) kmid_.push_back(0.5*(kedges_[i] + kedges_[i+1]));
    }
    d_kedges_ = DvceArray1D<double>("spec_kedges", nbin_ + 1);
    auto h = Kokkos::create_mirror_view(d_kedges_);
    for (int i = 0; i <= nbin_; ++i) h(i) = kedges_[i];
    Kokkos::deep_copy(d_kedges_, h);
  }

  // ---- z window (scipy hamming, scaled to unit mean square), on q only ----
  window_ = DvceArray1D<double>("spec_window", n_[2]);
  {
    auto h = Kokkos::create_mirror_view(window_);
    const int nz = n_[2];
    double msq = 0.0;
    for (int k = 0; k < nz; ++k) {
      h(k) = (strat_ && nz > 1) ? 0.54 - 0.46*std::cos(2.0*M_PI*k/(nz - 1)) : 1.0;
      msq += h(k)*h(k);
    }
    msq /= nz;
    for (int k = 0; k < nz; ++k) h(k) /= std::sqrt(msq);
    Kokkos::deep_copy(window_, h);
  }

  std::string transport = pin->GetOrAddString(blk, "transport", "device");
  if (transport != "device" && transport != "host") {
    SpecFatal(blk, "transport='" + transport + "' must be one of device, host");
  }
  host_stage_ = (transport == "host");
  // the FFT plan and its buffers are built on first use and, unless
  // persistent_buffers = true, released after every output
  persistent_ = pin->GetOrAddBoolean(blk, "persistent_buffers", false);
#if MPI_PARALLEL_ENABLED
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
#endif

  std::string dir_name = "spec_" + out_params.file_id;
  mkdir(dir_name.c_str(), 0775);
}

SpectrumOutput::~SpectrumOutput() {
#if MPI_PARALLEL_ENABLED
  MPI_Comm_free(&comm_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn double SpectrumOutput::LoadField()
//  \brief in_ = field n on the interior of every local block, times the z window when
//  'window'; returns the local sum of squares over the selected cells

double SpectrumOutput::LoadField(Mesh *pm, int n, bool window) {
  auto &mbi = pm->mb_indcs;
  const int is = mbi.is, js = mbi.js, ks = mbi.ks;
  const int nx1 = mbi.nx1, nx2 = mbi.nx2, nx3 = mbi.nx3;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const int gid0 = pm->gids_eachrank[global_variable::my_rank];
  // global index offset of every local block (uniform grid: root-level location)
  DvceArray2D<int> off("spec_off", nmb, 3);
  {
    auto h = Kokkos::create_mirror_view(off);
    for (int m = 0; m < nmb; ++m) {
      const LogicalLocation &ll = pm->lloc_eachmb[gid0 + m];
      h(m, 0) = ll.lx1*nx1;
      h(m, 1) = ll.lx2*nx2;
      h(m, 2) = ll.lx3*nx3;
    }
    Kokkos::deep_copy(off, h);
  }
  auto arr = field_arr_[n];
  const int idx = field_index_[n];
  auto in = in_;
  auto win = window_;
  const int lo0 = lo_[0], lo1 = lo_[1], lo2 = lo_[2];
  const int hi0 = hi_[0], hi1 = hi_[1], hi2 = hi_[2];
  double sumsq = 0.0;
  Kokkos::parallel_reduce("spec_load",
  Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                       {nmb, nx3, nx2, nx1}),
  KOKKOS_LAMBDA(int m, int k, int j, int i, double &s) {
    const int gx = off(m, 0) + i, gy = off(m, 1) + j, gz = off(m, 2) + k;
    double v = static_cast<double>(arr(m, idx, k + ks, j + js, i + is));
    const bool sel = (gx >= lo0 && gx < hi0 && gy >= lo1 && gy < hi1 &&
                      gz >= lo2 && gz < hi2);
    if (sel && window) v *= win(gz - lo2);
    in(m, k, j, i) = v;
    if (sel) s += v*v;
  }, sumsq);
  return sumsq;
}

//----------------------------------------------------------------------------------------
//! \fn void SpectrumOutput::KeepCoefficients()

void SpectrumOutput::KeepCoefficients(fft::CplxArray1D *dst) {
  fft::CplxArray1D src = pfft_->Forward(in_);
  const std::int64_t n = pfft_->Active() ? pfft_->Slab().Size() : 0;
  if (static_cast<std::int64_t>(dst->extent(0)) != std::max<std::int64_t>(n, 1)) {
    *dst = fft::CplxArray1D("spec_coef", std::max<std::int64_t>(n, 1));
  }
  if (n > 0) {
    Kokkos::deep_copy(Kokkos::subview(*dst, std::make_pair(std::int64_t{0}, n)),
                      Kokkos::subview(src, std::make_pair(std::int64_t{0}, n)));
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SpectrumOutput::BinModes()
//  \brief histogram over this rank's retained modes into host_out[nval*(nbin+1)]
//  (summed over all ranks, valid on rank 0); slot nbin of every value holds the k = 0
//  mode. kind 0: mode count h; kind 1: h P_q P_w for every weight (coef_[0] holds F_q,
//  pw_[w] the weight power, nval = nweight); kind 2: Helmholtz compressive and
//  solenoidal power (coef_[0..2], nval = 2). h = 1 for kx = 0 and the even-length x
//  Nyquist, else 2 (the conjugate partner); for kind 2 the partner is evaluated with
//  its own wavevector, which differs from -k on Nyquist planes.

void SpectrumOutput::BinModes(int kind, int nval, double *host_out) {
  const int nb = nbin_, nslot = nbin_ + 1;
  const int nent = nval*nslot;
  if (static_cast<int>(work_.extent(0)) < nent) {
    work_ = DvceArray1D<double>("spec_work", nent);
  }
  Kokkos::deep_copy(work_, 0.0);
  const fft::SpectralSlab slab = pfft_->Slab();
  const std::int64_t nmode = pfft_->Active() ? slab.Size() : 0;
  if (nmode > 0) {
    SpecModeEval ev;
    ev.slab = slab;
    ev.edges = d_kedges_;
    ev.c0 = coef_[0];
    ev.c1 = coef_[1];
    ev.c2 = coef_[2];
    for (int i = 0; i < kSpecMaxVal; ++i) {
      ev.hasw[i] = (kind == 1 && i < nweight_ && weight_field_[i] >= 0);
      if (ev.hasw[i]) ev.pw[i] = pw_[i];
    }
    ev.kind = kind;
    ev.nval = nval;
    ev.nb = nb;
    ev.nx = n_[0];
    ev.ny = n_[1];
    ev.nz = n_[2];
    ev.lx = len_[0];
    ev.ly = len_[1];
    ev.lz = len_[2];
    ev.ninv = 1.0/(static_cast<double>(n_[0])*n_[1]*n_[2]);
    auto work = work_;

    // values per pass whose histograms fit in team scratch (48 KiB cap, as sf); if not
    // even one fits, accumulate with global atomics instead
    std::size_t smax = Kokkos::TeamPolicy<>(1, Kokkos::AUTO).scratch_size_max(0);
    smax = std::min<std::size_t>(smax, 48*1024);
    int vpass = 0;
    for (int n = nval; n >= 1; --n) {
      if (ScrArray1D<double>::shmem_size(n*nslot) + 64 <= smax) {vpass = n; break;}
    }
    if (vpass == 0) {
      bin_global_ = 1.0;
    } else {
      bin_passes_ = std::max(bin_passes_, static_cast<double>((nval + vpass - 1)/vpass));
    }
    if (vpass == 0) {
      Kokkos::parallel_for("spec_bin_global",
      Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, nmode),
      KOKKOS_LAMBDA(const std::int64_t f) {
        int b;
        bool dc;
        double v[kSpecMaxVal];
        if (!ev.Eval(f, b, dc, v)) return;
        for (int i = 0; i < ev.nval; ++i) {
          if (b >= 0) Kokkos::atomic_add(&work(i*nslot + b), v[i]);
          if (dc) Kokkos::atomic_add(&work(i*nslot + nb), v[i]);
        }
      });
    } else {
      constexpr int kChunk = 4096;
      const std::int64_t nteam = (nmode + kChunk - 1)/kChunk;
      for (int v0 = 0; v0 < nval; v0 += vpass) {
        const int v1 = std::min(nval, v0 + vpass);
        const int nloc = (v1 - v0)*nslot;
        Kokkos::TeamPolicy<> policy(DevExeSpace(), static_cast<int>(nteam), Kokkos::AUTO);
        policy.set_scratch_size(0, Kokkos::PerTeam(ScrArray1D<double>::shmem_size(nloc)));
        Kokkos::parallel_for("spec_bin", policy,
        KOKKOS_LAMBDA(const TeamMember_t &member) {
          ScrArray1D<double> hist(member.team_scratch(0), nloc);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nloc), [&](const int i) {
            hist(i) = 0.0;
          });
          member.team_barrier();
          const std::int64_t f0 = static_cast<std::int64_t>(member.league_rank())*kChunk;
          const std::int64_t f1 = (f0 + kChunk < nmode) ? f0 + kChunk : nmode;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, f0, f1),
          [&](const std::int64_t f) {
            int b;
            bool dc;
            double v[kSpecMaxVal];
            if (!ev.Eval(f, b, dc, v)) return;
            for (int i = v0; i < v1; ++i) {
              if (b >= 0) Kokkos::atomic_add(&hist((i - v0)*nslot + b), v[i]);
              if (dc) Kokkos::atomic_add(&hist((i - v0)*nslot + nb), v[i]);
            }
          });
          member.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, nloc), [&](const int i) {
            if (hist(i) != 0.0) Kokkos::atomic_add(&work(v0*nslot + i), hist(i));
          });
        });
      }
    }
  }
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), work_);
  for (int i = 0; i < nent; ++i) host_out[i] = h(i);
#if MPI_PARALLEL_ENABLED
  const bool root = (global_variable::my_rank == 0);
  MPI_Reduce(root ? MPI_IN_PLACE : host_out, host_out, nent, MPI_DOUBLE, MPI_SUM, 0,
             comm_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void SpectrumOutput::LoadOutputData()
//  \brief transforms and histograms; results on rank 0

void SpectrumOutput::LoadOutputData(Mesh *pm) {
  const int nslot = nbin_ + 1;
  Kokkos::Timer timer;
  time_setup_ = time_fft_ = time_bin_ = 0.0;
  bin_passes_ = bin_global_ = 0.0;
  if (pfft_ == nullptr) {   // collective: every rank builds its part of the plan
    pfft_ = std::make_unique<fft::PencilFFT>(pm, lo_, hi_, host_stage_);
    auto &mbi = pm->mb_indcs;
    in_ = DvceArray4D<double>("spec_in", pm->pmb_pack->nmb_thispack, mbi.nx3, mbi.nx2,
                              mbi.nx1);
    Kokkos::fence();
    time_setup_ = timer.seconds();
  }
  // mode counts (geometry only; cheap)
  nk_.assign(nbin_ + 1, 0.0);
  BinModes(0, 1, nk_.data());
  std::vector<double> buf;
  auto reduce_sum = [&](double x) {
#if MPI_PARALLEL_ENABLED
    double y = 0.0;
    MPI_Reduce(&x, &y, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
    return y;
#else
    return x;
#endif
  };
  if (mode_ == 0) {
    E_.assign(static_cast<std::size_t>(nweight_)*nvar_*nbin_, 0.0);
    pdc_.assign(static_cast<std::size_t>(nweight_)*nvar_, 0.0);
    sumsq_.assign(nvar_, 0.0);
    // weight powers |F_w|^2/N (weights are not windowed, as spectra.py)
    pw_.resize(nweight_);
    const double ninv = 1.0/(static_cast<double>(n_[0])*n_[1]*n_[2]);
    timer.reset();
    for (int iw = 0; iw < nweight_; ++iw) {
      if (weight_field_[iw] < 0) continue;
      LoadField(pm, weight_field_[iw], false);
      KeepCoefficients(&coef_[0]);
      const std::int64_t n = coef_[0].extent(0);
      pw_[iw] = DvceArray1D<double>("spec_pw", n);
      auto pw = pw_[iw];
      auto c = coef_[0];
      Kokkos::parallel_for("spec_pw",
      Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, n),
      KOKKOS_LAMBDA(const std::int64_t f) {
        pw(f) = (c(f).real()*c(f).real() + c(f).imag()*c(f).imag())*ninv;
      });
    }
    Kokkos::fence();
    time_fft_ += timer.seconds();
    buf.assign(static_cast<std::size_t>(nweight_)*nslot, 0.0);
    for (int iv = 0; iv < nvar_; ++iv) {
      timer.reset();
      sumsq_[iv] = reduce_sum(LoadField(pm, iv, strat_));
      KeepCoefficients(&coef_[0]);
      Kokkos::fence();
      time_fft_ += timer.seconds();
      timer.reset();
      BinModes(1, nweight_, buf.data());
      time_bin_ += timer.seconds();
      for (int iw = 0; iw < nweight_; ++iw) {
        for (int b = 0; b < nbin_; ++b) {
          E_[(static_cast<std::size_t>(iw)*nvar_ + iv)*nbin_ + b] = buf[iw*nslot + b];
        }
        pdc_[static_cast<std::size_t>(iw)*nvar_ + iv] = buf[iw*nslot + nbin_];
      }
    }
  } else {
    E_.assign(2*static_cast<std::size_t>(nbin_), 0.0);
    pdc_.assign(1, 0.0);
    sumsq_.assign(1, 0.0);
    timer.reset();
    double s = 0.0;
    for (int c = 0; c < 3; ++c) {
      s += LoadField(pm, c, strat_);
      KeepCoefficients(&coef_[c]);
    }
    sumsq_[0] = reduce_sum(s);
    Kokkos::fence();
    time_fft_ = timer.seconds();
    timer.reset();
    buf.assign(2*static_cast<std::size_t>(nslot), 0.0);
    BinModes(2, 2, buf.data());
    time_bin_ = timer.seconds();
    for (int b = 0; b < nbin_; ++b) {
      E_[b] = buf[b];
      E_[nbin_ + b] = buf[nslot + b];
    }
    pdc_[0] = buf[nbin_] + buf[nslot + nbin_];
  }

  // memory held at the peak of this output, per rank
  pencil_py_ = pfft_->Py();
  pencil_pz_ = pfft_->Pz();
  const double *ts = pfft_->Timings();
  for (int i = 0; i < 6; ++i) time_stage_[i] = ts[i];
  std::size_t dev = pfft_->DeviceBytes() + in_.size()*sizeof(double) +
                    work_.size()*sizeof(double);
  for (int c = 0; c < 3; ++c) dev += coef_[c].size()*sizeof(fft::Complex);
  for (auto &p : pw_) dev += p.size()*sizeof(double);
  mem_device_ = static_cast<double>(dev);
  mem_host_ = static_cast<double>(pfft_->HostBytes());
#if MPI_PARALLEL_ENABLED
  double loc[13] = {time_setup_, time_fft_, time_bin_, mem_device_, mem_host_,
                    ts[0], ts[1], ts[2], ts[3], ts[4], ts[5], bin_passes_, bin_global_};
  double mx[13];
  MPI_Reduce(loc, mx, 13, MPI_DOUBLE, MPI_MAX, 0, comm_);
  if (global_variable::my_rank == 0) {
    time_setup_ = mx[0];
    time_fft_ = mx[1];
    time_bin_ = mx[2];
    mem_device_ = mx[3];
    mem_host_ = mx[4];
    for (int i = 0; i < 6; ++i) time_stage_[i] = mx[5 + i];
    bin_passes_ = mx[11];
    bin_global_ = mx[12];
  }
#endif
  if (!persistent_) {
    pfft_.reset();
    in_ = DvceArray4D<double>();
    for (int c = 0; c < 3; ++c) coef_[c] = fft::CplxArray1D();
    pw_.clear();
    work_ = DvceArray1D<double>();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void SpectrumOutput::WriteOutputFile()
//  \brief rank 0 writes spec_<id>/<basename>.NNNNN.spec: ASCII key=value header, then
//  float64 arrays in the order listed under 'payload'

void SpectrumOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  if (global_variable::my_rank == 0) {
    char number[16];
    std::snprintf(number, sizeof(number), "%05d", out_params.file_number);
    std::string fname = "spec_" + out_params.file_id + "/" + out_params.file_basename +
                        "." + number + ".spec";
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "wb")) == nullptr) {
      SpecFatal(out_params.block_name, "output file '" + fname + "' could not be opened");
    }
    auto join = [](const std::vector<std::string> &v) {
      std::string s;
      for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
      return s;
    };
    auto fmt = [](double x) {
      char b[40];
      std::snprintf(b, sizeof(b), "%.17g", x);
      return std::string(b);
    };
    std::string meta;
    meta += "Athena spec format_version=1\n";
    meta += "time=" + fmt(pm->time) + "\n";
    meta += "cycle=" + std::to_string(pm->ncycle) + "\n";
    meta += "file_number=" + std::to_string(out_params.file_number) + "\n";
    meta += "mode=" + std::string(mode_ == 0 ? "scalar" : "helmholtz") + "\n";
    meta += "variables=" + join(var_names_) + "\n";
    meta += "weights=" + join(weight_names_) + "\n";
    meta += "nvar=" + std::to_string(nvar_) + "\n";
    meta += "nweight=" + std::to_string(nweight_) + "\n";
    meta += "nbin=" + std::to_string(nbin_) + "\n";
    meta += "logscale=" + std::to_string(logscale_ ? 1 : 0) + "\n";
    meta += "strat=" + std::to_string(strat_ ? 1 : 0) + "\n";
    meta += "skip=" + fmt(skip_) + "\n";
    meta += "index_lo=" + std::to_string(lo_[0]) + "," + std::to_string(lo_[1]) + "," +
            std::to_string(lo_[2]) + "\n";
    meta += "shape=" + std::to_string(n_[0]) + "," + std::to_string(n_[1]) + "," +
            std::to_string(n_[2]) + "\n";
    meta += "length=" + fmt(len_[0]) + "," + fmt(len_[1]) + "," + fmt(len_[2]) + "\n";
    meta += "mesh_nx=" + std::to_string(pm->mesh_indcs.nx1) + "," +
            std::to_string(pm->mesh_indcs.nx2) + "," +
            std::to_string(pm->mesh_indcs.nx3) + "\n";
    meta += "fft_convention=numpy_rfftn_unnormalised,P=|F|^2/N,k=fftfreq*N/L,"
            "nyquist_negative,x_half_axis_with_partner_multiplicity\n";
    meta += "pencils=" + std::to_string(pencil_py_) + "x" + std::to_string(pencil_pz_) +
            "\n";
    meta += "persistent_buffers=" + std::to_string(persistent_ ? 1 : 0) + "\n";
    meta += "transport=" + std::string(host_stage_ ? "host" : "device") + "\n";
    meta += "nranks=" + std::to_string(global_variable::nranks) + "\n";
    meta += "timing=max over ranks\n";
    meta += "time_setup=" + fmt(time_setup_) + "\n";
    meta += "time_fft=" + fmt(time_fft_) + "\n";
    meta += "time_bin=" + fmt(time_bin_) + "\n";
    meta += "bin_scratch_passes_max=" + std::to_string(static_cast<int>(bin_passes_)) +
            "\n";
    meta += "bin_global_fallback=" + std::to_string(static_cast<int>(bin_global_)) +
            "\n";
    meta += "time_last_transform=" + fmt(time_stage_[0]) + "," + fmt(time_stage_[1]) +
            "," + fmt(time_stage_[2]) + "," + fmt(time_stage_[3]) + "," +
            fmt(time_stage_[4]) + "," + fmt(time_stage_[5]) + "\n";
    meta += "time_last_transform_stages=R0,x,R1,y,R2,z\n";
    meta += "memory_device_bytes_max_rank=" + fmt(mem_device_) + "\n";
    meta += "memory_host_bytes_max_rank=" + fmt(mem_host_) + "\n";
    const uint16_t one = 1;
    const bool little = (*reinterpret_cast<const unsigned char *>(&one) == 1);
    meta += "dtype=float64\n";
    meta += std::string("byte_order=") + (little ? "little" : "big") + "\n";
    if (mode_ == 0) {
      meta += "payload=kbins[nbin+1],k[nbin],nk[nbin],E[nweight,nvar,nbin],"
              "P_DC[nweight,nvar],sumsq[nvar]\n";
    } else {
      meta += "payload=kbins[nbin+1],k[nbin],nk[nbin],E_comp[nbin],E_sol[nbin],"
              "P_DC[1],sumsq[1]\n";
    }
    // every write is counted; a short write or stream error is fatal before the output
    // counter advances (rank 0 writes; its exit aborts the MPI job)
    bool wok = SpecWriteHeader(pfile, meta);
    auto put = [&](const void *ptr, std::size_t size, std::size_t n) {
      wok &= (std::fwrite(ptr, size, n, pfile) == n);
    };
    put(kedges_.data(), sizeof(double), nbin_ + 1);
    put(kmid_.data(), sizeof(double), nbin_);
    put(nk_.data(), sizeof(double), nbin_);
    put(E_.data(), sizeof(double), E_.size());
    put(pdc_.data(), sizeof(double), pdc_.size());
    put(sumsq_.data(), sizeof(double), sumsq_.size());
    wok &= (std::ferror(pfile) == 0);
    if (std::fclose(pfile) != 0 || !wok) {
      SpecFatal(out_params.block_name, "error writing or closing '" + fname + "'");
    }
  }

  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
