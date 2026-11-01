#ifndef FFT_PENCIL_FFT_HPP_
#define FFT_PENCIL_FFT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pencil_fft.hpp
//! \brief Distributed forward real-to-complex 3D FFT of a (cropped) uniform-grid field,
//! with no external FFT library (see docs/spectrum_output.md, "Distributed FFT").
//!
//! Data path, for active ranks arranged as a Py x Pz grid (py = r % Py, pz = r / Py):
//!   R0  MeshBlocks -> x-pencils (z chunk pz, y chunk py, all x): sparse point-to-point
//!       from every rank (ranks beyond Py*Pz only send)
//!   x   complex FFT of the promoted real lines; only m = 0..Nx/2 are kept (numpy rfft)
//!   R1  x-pencils -> y-pencils (z chunk pz, x chunk py of Nx/2+1, all y), row comm
//!   y   complex FFT
//!   R2  y-pencils -> z-pencils (x chunk py, y chunk pz, all z), column comm (3D only)
//!   z   complex FFT
//! The result is numpy.fft.rfftn(q.T).T in AthenaK's (x, y, z) index order: unnormalised,
//! index 0 at the first selected cell, x the half-length axis.
//!
//! Exchanges use device buffers with point-to-point Isend/Irecv (as bvals does), or
//! host-staged buffers (transport = host); every pack kernel is fenced before MPI and
//! Waitall completes before unpacking. Messages are chunked below INT_MAX elements.

#include <cstdint>
#include <memory>
#include <vector>

#include "athena.hpp"
#include "fft/local_fft.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

class Mesh;

namespace fft {

//! device-copyable description of the spectral coefficients a rank holds
struct SpectralSlab {
  int nx, ny, nz;          // global transform lengths (crop); x is stored as nx/2+1
  int x0, xl, y0, yl, z0, zl;   // this rank's global offsets and counts (x in 0..nx/2)
  bool zpencil;            // true: flat = (ix*yl + iy)*zl + iz; false: (iz*xl+ix)*yl+iy
  KOKKOS_INLINE_FUNCTION
  void Global(std::int64_t f, int &gx, int &gy, int &gz) const {
    if (zpencil) {
      gz = z0 + static_cast<int>(f % zl);
      f /= zl;
      gy = y0 + static_cast<int>(f % yl);
      gx = x0 + static_cast<int>(f/yl);
    } else {
      gy = y0 + static_cast<int>(f % yl);
      f /= yl;
      gx = x0 + static_cast<int>(f % xl);
      gz = z0 + static_cast<int>(f/xl);
    }
  }
  std::int64_t Size() const {
    return static_cast<std::int64_t>(xl)*yl*zl;
  }
};

//! one contiguous piece of an exchange: peer and element range in a flat buffer
struct ExchangePlan {
  std::vector<int> send_peer, recv_peer;               // ranks in the exchange comm
  std::vector<std::int64_t> send_off, recv_off;        // nsend+1, nrecv+1 offsets
};

class PencilFFT {
 public:
  //! lo/hi: selected global (root-grid) cell index range [lo, hi) on each axis;
  //! host_stage: stage MPI buffers on the host instead of passing device pointers
  PencilFFT(Mesh *pm, const int lo[3], const int hi[3], bool host_stage);
  ~PencilFFT();
  PencilFFT(const PencilFFT &) = delete;
  PencilFFT &operator=(const PencilFFT &) = delete;

  //! forward transform of a real field given on the interior cells of every local
  //! MeshBlock, in(m, k, j, i) with (k, j, i) relative to the block interior. Cells
  //! outside the selection are ignored. Returns this rank's coefficients (empty view on
  //! inactive ranks), laid out as described by Slab().
  CplxArray1D Forward(const DvceArray4D<double> &in);

  const SpectralSlab &Slab() const {return slab_;}
  //! device and host bytes held now (pencils, exchange buffers, plans and work buffers)
  std::size_t DeviceBytes() const;
  std::size_t HostBytes() const {
    return (hsbuf_.extent(0) + hrbuf_.extent(0))*sizeof(double);
  }
  bool Active() const {return active_;}
  int Py() const {return py_n_;}
  int Pz() const {return pz_n_;}
  //! largest MPI message in doubles (default 2^30); tests set a small value to exercise
  //! the chunking without large buffers
  void SetMaxMessage(std::int64_t n) {max_msg_ = (n > 0) ? n : max_msg_;}
  //! most chunks used for one remote message by this rank so far (tests: tag wrap)
  std::int64_t MaxChunks() const {return max_chunks_;}
  //! timings of the last Forward(): R0, x FFT, R1, y FFT, R2, z FFT (seconds)
  const double *Timings() const {return time_;}

  // Implementation below; public only because CUDA lambdas cannot live in private
  // member functions.
  void BuildR0(Mesh *pm);
  void BuildTransposes();
  //! exchange 'p' (offsets in doubles) over comm 0 = world, 1 = row, 2 = column
  void Exchange(const ExchangePlan &p, int which, int tag_base);

 private:
  int n_[3], lo_[3];               // transform lengths and crop origin (global index)
  int nh_;                         // nx/2 + 1
  int py_n_, pz_n_;                // pencil grid
  bool active_, host_stage_;
  std::int64_t max_msg_ = std::int64_t{1} << 30;   // doubles per MPI message
  std::int64_t max_chunks_ = 0;
  int py_, pz_;                    // my pencil coordinates (active ranks)
  std::vector<int> ych_off_, zch_off_, xch_off_, y2ch_off_;   // chunk offsets (P+1)
  SpectralSlab slab_;

  // R0: segments (block-interior boxes) packed by this rank / unpacked into my x-pencil
  ExchangePlan r0_;
  DvceArray2D<int> r0_send_seg_, r0_recv_seg_;        // (nseg, 7): m|pencil, lo[3], n[3]
  DvceArray1D<std::int64_t> r0_send_segoff_, r0_recv_segoff_;
  int r0_nsend_seg_, r0_nrecv_seg_;
  ExchangePlan r1_, r2_;                               // offsets in doubles
  DvceArray1D<std::int64_t> r1_soff_, r1_roff_, r2_soff_, r2_roff_;   // complex units
  DvceArray1D<int> xch_d_, ych_d_, zch_d_, y2ch_d_;

  // buffers
  CplxArray1D xpen_, ypen_, zpen_;
  DvceArray1D<double> sbuf_, rbuf_;
  HostArray1D<double> hsbuf_, hrbuf_;
  std::unique_ptr<LocalFFT> fx_, fy_, fz_;
  double time_[6];

#if MPI_PARALLEL_ENABLED
  MPI_Comm world_, row_, col_;     // dup of world; same-pz row; same-py column
#endif
};

}  // namespace fft

#endif  // FFT_PENCIL_FFT_HPP_
