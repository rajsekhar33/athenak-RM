#ifndef OUTPUTS_SPECTRUM_HPP_
#define OUTPUTS_SPECTRUM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spectrum.hpp
//  \brief on-the-fly power spectra (file_type = spec) from the distributed FFT
//  (fft/pencil_fft.hpp), matching athena_research/operations/spectra.py:
//    P(k) = |F(k)|^2 / N with numpy's unnormalised forward FFT, k in cycles per unit
//    length (numpy fftfreq labels, Nyquist negative), shell histograms with Python's
//    _spectral_grid edges, optional z crop by 'skip' with a unit-mean-square Hamming
//    window (strat = true), the legacy weighted product P_q * P_w (weight not windowed),
//    and the Helmholtz split (k.F)k/|k|^2 with the k = 0 mode solenoidal.
//  Only kx >= 0 is stored (real input); every other mode stands for itself and its
//  conjugate partner, which is evaluated explicitly (its own wavevector) where needed.
//  See docs/spectrum_output.md.

#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "outputs.hpp"
#include "fft/pencil_fft.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
//! \class SpectrumOutput
//  \brief derived BaseTypeOutput class for on-the-fly power spectra

class SpectrumOutput : public BaseTypeOutput {
 public:
  SpectrumOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  ~SpectrumOutput();

  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;

  // kernels; public only because CUDA lambdas cannot live in private methods
  //! fill in_ with field n (window applied if 'window'); returns sum of squares of
  //! the selected cells on this rank
  double LoadField(Mesh *pm, int n, bool window);
  //! copy this rank's coefficients of the last transform into dst
  void KeepCoefficients(fft::CplxArray1D *dst);
  //! histogram of nval values per retained mode (see spectrum.cpp)
  void BinModes(int kind, int nval, double *host_out);

 private:
  int mode_;                          // 0 = scalar, 1 = helmholtz
  std::vector<std::string> var_names_, weight_names_;
  std::vector<int> weight_field_;     // field index per weight, -1 = "ones"
  int nvar_, nweight_, nbin_;         // nbin_ = number of histogram bins
  bool logscale_, strat_, host_stage_;
  bool persistent_;                   // keep the FFT plan and buffers between outputs
  Real skip_;
  int lo_[3], hi_[3], n_[3];          // selected global index range, its lengths
  double len_[3];                     // transformed lengths N_i dx_i (Python L)
  std::vector<double> kedges_, kmid_;
  DvceArray1D<double> d_kedges_;
  DvceArray1D<double> window_;        // z window over the selected range (or ones)

  // fields: variables first, then non-unit weights
  std::vector<DvceArray5D<Real>> field_arr_;
  std::vector<int> field_index_;

  std::unique_ptr<fft::PencilFFT> pfft_;
  DvceArray4D<double> in_;            // (nmb, nx3, nx2, nx1) transform input
  // per-mode work: weight powers (scalar) or vector coefficients (helmholtz)
  std::vector<DvceArray1D<double>> pw_;
  fft::CplxArray1D coef_[3];
  DvceArray1D<double> work_;          // nval*nbin accumulator

  // results (rank 0 after reduction)
  std::vector<double> nk_, E_, pdc_, sumsq_;
  // timings of the last output (max over ranks): plan setup, transforms (including
  // weight transforms), binning; per-stage times of the last transform; memory held
  double time_setup_, time_fft_, time_bin_, time_stage_[6];
  // binning path of the last output (max over ranks and BinModes calls): scratch passes
  // per call, and whether the global-atomics fallback was used
  double bin_passes_, bin_global_;
  double mem_device_, mem_host_;
  int pencil_py_ = 1, pencil_pz_ = 1; // pencil grid of the last output

#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_;
#endif
};

#endif // OUTPUTS_SPECTRUM_HPP_
