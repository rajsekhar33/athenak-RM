#ifndef OUTPUTS_STRUCTURE_FUNCTION_HPP_
#define OUTPUTS_STRUCTURE_FUNCTION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file structure_function.hpp
//  \brief on-the-fly structure-function output (file_type = sf).
//
//  Estimates pair statistics of increments |w2*q2 - w1*q1|^p binned in separation |l|,
//  matching athena_research/operations/structure_functions.py conventions. Only a
//  sample of field values is communicated. Pairs are split into two disjoint populations:
//    E1: both cells in the same MeshBlock (sampled locally, lag-first, no communication)
//    E2: cells in different MeshBlocks (local anchors x an all-gathered global sample)
//  Each of R replicates is a complete, independent estimator; replicates are averaged.
//  See docs/structure_function_output.md for the estimator contract.

#include <cstdint>
#include <string>
#include <vector>

#include "athena.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

// maximum number of distinct fields (variables + non-unit weights) per sf output
constexpr int kSFMaxFields = 8;

//----------------------------------------------------------------------------------------
//! \struct SFFieldSet
//  \brief device-copyable set of (array, component) references to the fields an sf
//  output reads, so kernels read the solver arrays directly with no copies.

struct SFFieldSet {
  DvceArray5D<Real> arr[kSFMaxFields];
  int index[kSFMaxFields];
  int nfield = 0;
  KOKKOS_INLINE_FUNCTION
  Real Get(int n, int m, int k, int j, int i) const {
    return arr[n](m, index[n], k, j, i);
  }
};

//! field index of each weight (-1 for "ones"), copyable into device kernels
struct SFWeightMap {
  int field[kSFMaxFields];
};

//! fields computed from the primitives (eos = ideal, Newtonian): sf variables and
//! select<n>_variable "temperature" = (gamma-1) e/rho (Python 'temp') and "pressure"
//! = (gamma-1) e (Python 'pres')
constexpr int kSFTemperature = -1;
constexpr int kSFPressure = -2;

//! cell selection (select1_*, select2_*): a cell is selected where every condition
//! lo <= value <= hi holds (NaN fails); field = loaded-field index, or kSFTemperature /
//! kSFPressure
struct SFSelect {
  int nsel = 0;
  int field[2];
  double lo[2], hi[2];
};

//! minimum image along a periodic axis of n cells, in whole cells; a separation of
//! exactly half the box keeps its sign
KOKKOS_INLINE_FUNCTION
int64_t SFMinImage(int64_t d, bool periodic, int64_t n) {
  if (periodic && 2*(d < 0 ? -d : d) > n) {
    d -= (d > 0) ? n : -n;
  }
  return d;
}

//! physical separation from a global cell offset (di, dj, dk). E1 and E2 both call
//! this, so a pair's bin depends only on its cell offset, not on the MeshBlock layout.
struct SFGeom {
  bool per0, per1, per2;
  int64_t n0, n1, n2;    // mesh cells per axis
  double h0, h1, h2;     // mesh cell widths, domain length / cells

  KOKKOS_INLINE_FUNCTION
  double Separation(int64_t di, int64_t dj, int64_t dk, double &dx, double &dy,
                    double &dz) const {
    dx = static_cast<double>(SFMinImage(di, per0, n0))*h0;
    dy = static_cast<double>(SFMinImage(dj, per1, n1))*h1;
    dz = static_cast<double>(SFMinImage(dk, per2, n2))*h2;
    return Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
  }
};

//----------------------------------------------------------------------------------------
//! \class SFOutput
//  \brief derived BaseTypeOutput class for on-the-fly structure functions

class SFOutput : public BaseTypeOutput {
 public:
  SFOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  ~SFOutput();

  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;

  // estimator kernels; public only because CUDA lambdas cannot live in private methods
  void SampleWithinBlocks(Mesh *pm);   // E1
  void SampleAcrossBlocks(Mesh *pm);   // E2
  void BuildHistEdges(Mesh *pm);       // increment-histogram edges for this output
  void BuildSupport(Mesh *pm);         // sparse = true: cells with some weight > 0
  void ApplySelection(Mesh *pm);       // select*: fill sel_w_ for this output
  void ComputeFields(Mesh *pm);        // fill der_ (temperature/pressure variables)

 private:
  // configuration
  int mode_;   // 0 = scalar, 1 = long_trans, 2 = aniso_mhd, 3 = correlation
  int nvar_;                          // number of variables q
  int nweight_;                       // number of weights (including "ones")
  int nq_;                            // increments per pair: nvar (scalar) or 2 (l,t)
  std::vector<std::string> var_names_, weight_names_;
  std::vector<int> weight_field_;     // field index of each weight, -1 for "ones"
  int max_order_, nbin_, nrep_;
  bool logscale_, timing_;
  Real r_min_, r_max_;
  int64_t ndraw_block_, nsample_global_, nanchor_global_;
  uint64_t seed_;
  int halo_requested_, halo_effective_;
  bool periodic_[3];
  Real len_[3];                       // physical domain lengths
  SFGeom geom_;                       // cell widths and periodicity for separations
  int nactive_dim_;                   // 1, 2 or 3
  int cmax_;                          // largest Chebyshev lag radius used by E1
  int nw_pass_;                       // weights per team-kernel pass; 0 = global atomics

  // increment histograms (Python delta_pdf): per (weight, q) edges = template * sigma
  bool hist_;
  bool hist_sigma_scale_;             // true: template in units of sigma; false: absolute
  int hist_nbin_, hist_ncol_;
  double hist_min_, hist_max_;
  std::vector<int> hist_nedge_;       // edges per q (signed longitudinal row: one more)
  std::vector<double> hist_tmpl_[2];  // unsigned, signed templates
  std::vector<double> hist_sigma_;    // (nweight, nq) of the current output
  HostArray2D<double> h_hist_edges_;  // (nweight*nq, ncol-1)
  DvceArray2D<double> hist_edges_;
  DvceArray1D<double> acc_H_;         // (est, rep, weight, q, bin, col)
  HostArray1D<double> h_H_;

  SFFieldSet fields_;
  DvceArray1D<double> rbins_;         // nbin_l_+1 separation-bin edges (|l| family)
  // aniso_mhd: three bin families, accumulated side by side (nbin_ = their sum)
  int nbin_l_, nbin_prll_, nbin_perp_;
  double theta_prll_max_, theta_perp_min_;
  DvceArray1D<double> rbins_prll_, rbins_perp_;
  DvceArray1D<double> cdf_c_;         // inverse-CDF table for E1 Chebyshev radius
  double harmonic_cmax_;              // normalisation of p(c) ~ 1/c

  // E2 sampling strata: region cells per local MeshBlock (sub-box) and per rank
  HostArray2D<int> h_box_;            // (nmb, 6) region index box per local MB
  DvceArray2D<int> box_;
  DvceArray1D<int64_t> box_offset_;   // prefix sum of region cells over local MBs
  HostArray1D<int64_t> h_box_offset_;
  int64_t ncell_region_local_;
  std::vector<int64_t> ncell_region_rank_;          // N_s for every rank
  std::vector<std::vector<int64_t>> nsample_rt_;    // m_{s,t} [rank][rep]
  std::vector<std::vector<int64_t>> nanchor_rt_;    // n_{s,t} [rank][rep]

  // sparse = true (Python sparse_data): support cells = region cells where some weight
  // is > 0, rebuilt at every output. E2 anchors and samples are drawn only from them
  // (rank quotas follow the support counts); each E1 anchor comes, with probability
  // 1/2 each, from the block's support cells or from the default admissible proposal
  bool sparse_;
  DvceArray1D<int64_t> supp_;         // region-local linear index of each support cell
  DvceArray1D<int64_t> supp_off_;     // (nmb+1) first support entry of each local MB
  HostArray1D<int64_t> h_supp_off_;
  int64_t nsupp_local_;
  std::vector<int64_t> nsupp_rank_;   // support cells of every rank

  // fields_ holds the variables (slots 0..nvar-1) followed by the distinct non-unit
  // weights. Computed variables (temperature, pressure) read der_(m, n, k, j, i),
  // filled at every output from the primitives.
  std::vector<int> der_kind_;         // kSFTemperature or kSFPressure per der_ component
  DvceArray5D<Real> der_;             // (nmb, nder, n3, n2, n1), interior filled
  double gm1_;                        // gamma - 1 for computed fields
  bool eos_mhd_;                      // computed fields from the mhd primitives

  // cell selection (Python's region masks, e.g. T_lo <= T <= T_hi): when configured,
  // weight iw reads sel_w_(m, iw, k, j, i) = (selected ? weight : 0), filled at every
  // output from the loaded fields src_fields_; fields_ then holds the variables
  // followed by one sel_w_ component per weight
  SFSelect select_;
  std::vector<std::string> select_names_;
  SFFieldSet src_fields_;             // every loaded field, in variable_list order
  std::vector<int> weight_src_;       // src_fields_ index of each weight, -1 for "ones"
  DvceArray5D<Real> sel_w_;           // (nmb, nweight, n3, n2, n1), interior filled
  int64_t nsel_total_;                // selected region cells (all ranks)
  double time_select_;

  // accumulators, flattened (est, rep, weight, q, order, bin) etc.; see Index*()
  DvceArray1D<double> acc_T_, acc_W_, acc_C_;
  DvceArray1D<uint64_t> acc_n_;       // raw accepted evaluations (est,rep,w,b)
  uint64_t attempts_[2];                          // raw attempted draws/evaluations
  HostArray1D<double> h_T_, h_W_, h_C_;
  HostArray1D<uint64_t> h_n_;
  double time_e1_, time_e2_, time_reduce_, time_support_;
  int output_count_;

#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_;
#endif

  void BuildStrata(Mesh *pm);
  void SetupComputed(Mesh *pm, const std::string &what);
  bool AllocateBudgets(const std::vector<int64_t> &ncell_rank, bool allow_empty);
  void ReduceAccumulators();
};

#endif // OUTPUTS_STRUCTURE_FUNCTION_HPP_
