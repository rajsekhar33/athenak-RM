//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pencil_fft.cpp
//! \brief distributed forward R2C FFT over pencils; see pencil_fft.hpp

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "fft/pencil_fft.hpp"

namespace fft {

namespace {

void PFFatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in PencilFFT" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//! offsets of an even split of n items over p parts (p+1 entries)
std::vector<int> Chunks(int n, int p) {
  std::vector<int> off(p + 1);
  for (int i = 0; i <= p; ++i) {
    off[i] = static_cast<int>((static_cast<std::int64_t>(n)*i)/p);
  }
  return off;
}

DvceArray1D<int> ToDevice(const std::vector<int> &v, const char *name) {
  DvceArray1D<int> d(std::string(name), static_cast<std::int64_t>(v.size()));
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

DvceArray1D<std::int64_t> ToDevice(const std::vector<std::int64_t> &v,
                                   const char *name) {
  DvceArray1D<std::int64_t> d(std::string(name), static_cast<std::int64_t>(v.size()));
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

//! s with off(s) <= idx < off(s+1), for off of length n+1
KOKKOS_INLINE_FUNCTION
int FindPiece(const DvceArray1D<std::int64_t> &off, int n, std::int64_t idx) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1)/2;
    if (off(mid) <= idx) {lo = mid;} else {hi = mid - 1;}
  }
  return lo;
}

using DUView1D = Kokkos::View<double *, Kokkos::LayoutRight, DevMemSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>;


}  // namespace

//----------------------------------------------------------------------------------------
// ctor: pencil grid, communicators, exchange plans and buffers

PencilFFT::PencilFFT(Mesh *pm, const int lo[3], const int hi[3], bool host_stage) :
    host_stage_(host_stage) {
  if (pm->multilevel) PFFatal("uniform grids only (no SMR/AMR)");
  const int nmesh[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  for (int d = 0; d < 3; ++d) {
    if (lo[d] < 0 || hi[d] > nmesh[d] || hi[d] <= lo[d]) {
      PFFatal("selected index range is empty or outside the mesh on axis " +
              std::to_string(d + 1));
    }
    lo_[d] = lo[d];
    n_[d] = hi[d] - lo[d];
  }
  nh_ = n_[0]/2 + 1;
  const int nranks = global_variable::nranks, me = global_variable::my_rank;

  // pencil grid: as many active ranks as possible, then as square as possible, subject
  // to non-empty pencils at every stage
  const int pymax = std::min(n_[1], nh_);
  const int pzmax = (n_[2] > 1) ? std::min(n_[2], n_[1]) : 1;
  py_n_ = pz_n_ = 1;
  for (int py = 1; py <= std::min(nranks, pymax); ++py) {
    int pz = std::min(nranks/py, pzmax);
    int best = py_n_*pz_n_, cand = py*pz;
    if (cand > best || (cand == best && std::abs(py - pz) < std::abs(py_n_ - pz_n_))) {
      py_n_ = py;
      pz_n_ = pz;
    }
  }
  active_ = (me < py_n_*pz_n_);
  py_ = active_ ? me % py_n_ : -1;
  pz_ = active_ ? me / py_n_ : -1;
  ych_off_ = Chunks(n_[1], py_n_);
  zch_off_ = Chunks(n_[2], pz_n_);
  xch_off_ = Chunks(nh_, py_n_);
  y2ch_off_ = Chunks(n_[1], pz_n_);
  xch_d_ = ToDevice(xch_off_, "pfft_xch");
  ych_d_ = ToDevice(ych_off_, "pfft_ych");
  zch_d_ = ToDevice(zch_off_, "pfft_zch");
  y2ch_d_ = ToDevice(y2ch_off_, "pfft_y2ch");

#if MPI_PARALLEL_ENABLED
  MPI_Comm_dup(MPI_COMM_WORLD, &world_);
  MPI_Comm_split(world_, active_ ? pz_ : MPI_UNDEFINED, py_, &row_);
  MPI_Comm_split(world_, active_ ? py_ : MPI_UNDEFINED, pz_, &col_);
#endif

  slab_.nx = n_[0];
  slab_.ny = n_[1];
  slab_.nz = n_[2];
  slab_.zpencil = (n_[2] > 1);
  if (active_) {
    slab_.x0 = xch_off_[py_];
    slab_.xl = xch_off_[py_+1] - xch_off_[py_];
    if (slab_.zpencil) {
      slab_.y0 = y2ch_off_[pz_];
      slab_.yl = y2ch_off_[pz_+1] - y2ch_off_[pz_];
      slab_.z0 = 0;
      slab_.zl = n_[2];
    } else {
      slab_.y0 = 0;
      slab_.yl = n_[1];
      slab_.z0 = 0;
      slab_.zl = 1;
    }
  } else {
    slab_.x0 = slab_.y0 = slab_.z0 = 0;
    slab_.xl = slab_.yl = slab_.zl = 0;
  }

  fx_ = std::make_unique<LocalFFT>(n_[0]);
  fy_ = std::make_unique<LocalFFT>(n_[1]);
  fz_ = std::make_unique<LocalFFT>(n_[2]);
  for (int i = 0; i < 6; ++i) time_[i] = 0.0;

  BuildR0(pm);
  BuildTransposes();

  // flat exchange buffers sized for the largest exchange (in doubles)
  std::int64_t nmax = 1;
  for (const ExchangePlan *p : {&r0_, &r1_, &r2_}) {
    if (!p->send_off.empty()) nmax = std::max(nmax, p->send_off.back());
    if (!p->recv_off.empty()) nmax = std::max(nmax, p->recv_off.back());
  }
  sbuf_ = DvceArray1D<double>("pfft_sbuf", nmax);
  rbuf_ = DvceArray1D<double>("pfft_rbuf", nmax);
  if (host_stage_) {
    hsbuf_ = HostArray1D<double>("pfft_hsbuf", nmax);
    hrbuf_ = HostArray1D<double>("pfft_hrbuf", nmax);
  }
  if (active_) {
    const int yc = ych_off_[py_+1] - ych_off_[py_], zc = zch_off_[pz_+1] - zch_off_[pz_];
    const int xc = xch_off_[py_+1] - xch_off_[py_];
    xpen_ = CplxArray1D("pfft_xpen", static_cast<std::int64_t>(zc)*yc*n_[0]);
    ypen_ = CplxArray1D("pfft_ypen", static_cast<std::int64_t>(zc)*xc*n_[1]);
    if (slab_.zpencil) {
      const int y2c = y2ch_off_[pz_+1] - y2ch_off_[pz_];
      zpen_ = CplxArray1D("pfft_zpen", static_cast<std::int64_t>(xc)*y2c*n_[2]);
    }
  }
}

std::size_t PencilFFT::DeviceBytes() const {
  std::size_t b = (xpen_.extent(0) + ypen_.extent(0) + zpen_.extent(0))*sizeof(Complex) +
                  (sbuf_.extent(0) + rbuf_.extent(0))*sizeof(double);
  b += (r0_send_seg_.size() + r0_recv_seg_.size() + xch_d_.size() + ych_d_.size() +
        zch_d_.size() + y2ch_d_.size())*sizeof(int);
  b += (r0_send_segoff_.size() + r0_recv_segoff_.size() + r1_soff_.size() +
        r1_roff_.size() + r2_soff_.size() + r2_roff_.size())*sizeof(std::int64_t);
  return b + fx_->Bytes() + fy_->Bytes() + fz_->Bytes();
}

PencilFFT::~PencilFFT() {
#if MPI_PARALLEL_ENABLED
  if (row_ != MPI_COMM_NULL) MPI_Comm_free(&row_);
  if (col_ != MPI_COMM_NULL) MPI_Comm_free(&col_);
  MPI_Comm_free(&world_);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void PencilFFT::BuildR0()
//  \brief R0 segments: each local block's intersection with every x-pencil (send) and
//  every block's intersection with my x-pencil (receive). Both sides order the elements
//  of one message by block gid, then (k, j, i) with i fastest.

void PencilFFT::BuildR0(Mesh *pm) {
  const int me = global_variable::my_rank;
  const int nb[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  const int nactive = py_n_*pz_n_;
  // crop-relative index range [a, b) of block gid on every axis, clipped to the crop
  auto block_range = [&](int gid, int d, int &a, int &b) {
    const LogicalLocation &ll = pm->lloc_eachmb[gid];
    const int l = (d == 0) ? ll.lx1 : ((d == 1) ? ll.lx2 : ll.lx3);
    const int g0 = l*nb[d] - lo_[d];
    a = std::max(g0, 0);
    b = std::min(g0 + nb[d], n_[d]);
  };

  // ---- send side ----
  std::vector<int> seg;                       // 7 ints per segment
  std::vector<std::int64_t> segoff(1, 0);
  const int gid0 = pm->gids_eachrank[me], nmb = pm->nmb_eachrank[me];
  for (int dst = 0; dst < nactive; ++dst) {
    const int py = dst % py_n_, pz = dst / py_n_;
    const int rlo[3] = {0, ych_off_[py], zch_off_[pz]};
    const int rhi[3] = {n_[0], ych_off_[py+1], zch_off_[pz+1]};
    std::int64_t start = segoff.back();
    for (int m = 0; m < nmb; ++m) {
      int a[3], b[3], g0[3];
      bool empty = false;
      for (int d = 0; d < 3; ++d) {
        int ba, bb;
        block_range(gid0 + m, d, ba, bb);
        const LogicalLocation &ll = pm->lloc_eachmb[gid0 + m];
        const int l = (d == 0) ? ll.lx1 : ((d == 1) ? ll.lx2 : ll.lx3);
        g0[d] = l*nb[d] - lo_[d];
        a[d] = std::max(ba, rlo[d]);
        b[d] = std::min(bb, rhi[d]);
        if (b[d] <= a[d]) empty = true;
      }
      if (empty) continue;
      seg.insert(seg.end(), {m, a[0] - g0[0], a[1] - g0[1], a[2] - g0[2],
                             b[0] - a[0], b[1] - a[1], b[2] - a[2]});
      segoff.push_back(segoff.back() + static_cast<std::int64_t>(b[0] - a[0])*
                                       (b[1] - a[1])*(b[2] - a[2]));
    }
    if (segoff.back() > start) {
      r0_.send_peer.push_back(dst);
      if (r0_.send_off.empty()) r0_.send_off.push_back(start);
      r0_.send_off.push_back(segoff.back());
    }
  }
  if (r0_.send_off.empty()) r0_.send_off.push_back(0);
  r0_nsend_seg_ = static_cast<int>(segoff.size()) - 1;
  r0_send_seg_ = DvceArray2D<int>("pfft_r0_sseg", std::max(r0_nsend_seg_, 1), 7);
  {
    auto h = Kokkos::create_mirror_view(r0_send_seg_);
    for (int s = 0; s < r0_nsend_seg_; ++s) {
      for (int c = 0; c < 7; ++c) h(s, c) = seg[7*s + c];
    }
    Kokkos::deep_copy(r0_send_seg_, h);
  }
  r0_send_segoff_ = ToDevice(segoff, "pfft_r0_ssegoff");

  // ---- receive side (gids are ordered by rank, so sources come out grouped) ----
  seg.clear();
  segoff.assign(1, 0);
  if (active_) {
    const int rlo[3] = {0, ych_off_[py_], zch_off_[pz_]};
    const int rhi[3] = {n_[0], ych_off_[py_+1], zch_off_[pz_+1]};
    int cur = -1;
    for (int gid = 0; gid < pm->nmb_total; ++gid) {
      int a[3], b[3];
      bool empty = false;
      for (int d = 0; d < 3; ++d) {
        int ba, bb;
        block_range(gid, d, ba, bb);
        a[d] = std::max(ba, rlo[d]);
        b[d] = std::min(bb, rhi[d]);
        if (b[d] <= a[d]) empty = true;
      }
      if (empty) continue;
      const int src = pm->rank_eachmb[gid];
      if (src != cur) {
        if (cur >= 0) r0_.recv_off.push_back(segoff.back());
        if (r0_.recv_off.empty()) r0_.recv_off.push_back(segoff.back());
        r0_.recv_peer.push_back(src);
        cur = src;
      }
      seg.insert(seg.end(), {0, a[0] - rlo[0], a[1] - rlo[1], a[2] - rlo[2],
                             b[0] - a[0], b[1] - a[1], b[2] - a[2]});
      segoff.push_back(segoff.back() + static_cast<std::int64_t>(b[0] - a[0])*
                                       (b[1] - a[1])*(b[2] - a[2]));
    }
    if (cur >= 0) r0_.recv_off.push_back(segoff.back());
    const std::int64_t want = static_cast<std::int64_t>(n_[0])*
        (rhi[1] - rlo[1])*(rhi[2] - rlo[2]);
    if (segoff.back() != want) {
      PFFatal("MeshBlocks do not tile the selected region (" +
              std::to_string(segoff.back()) + " of " + std::to_string(want) + " cells)");
    }
  }
  if (r0_.recv_off.empty()) r0_.recv_off.push_back(0);
  r0_nrecv_seg_ = static_cast<int>(segoff.size()) - 1;
  r0_recv_seg_ = DvceArray2D<int>("pfft_r0_rseg", std::max(r0_nrecv_seg_, 1), 7);
  {
    auto h = Kokkos::create_mirror_view(r0_recv_seg_);
    for (int s = 0; s < r0_nrecv_seg_; ++s) {
      for (int c = 0; c < 7; ++c) h(s, c) = seg[7*s + c];
    }
    Kokkos::deep_copy(r0_recv_seg_, h);
  }
  r0_recv_segoff_ = ToDevice(segoff, "pfft_r0_rsegoff");
}

//----------------------------------------------------------------------------------------
//! \fn void PencilFFT::BuildTransposes()
//  \brief R1 (row comm, peer = py) and R2 (column comm, peer = pz) message sizes.

void PencilFFT::BuildTransposes() {
  std::vector<std::int64_t> s1(1, 0), r1(1, 0), s2(1, 0), r2(1, 0);
  if (active_) {
    const int yc = ych_off_[py_+1] - ych_off_[py_], zc = zch_off_[pz_+1] - zch_off_[pz_];
    const int xc = xch_off_[py_+1] - xch_off_[py_];
    for (int p = 0; p < py_n_; ++p) {
      const int xcp = xch_off_[p+1] - xch_off_[p], ycp = ych_off_[p+1] - ych_off_[p];
      s1.push_back(s1.back() + static_cast<std::int64_t>(zc)*yc*xcp);
      r1.push_back(r1.back() + static_cast<std::int64_t>(zc)*ycp*xc);
      r1_.send_peer.push_back(p);
      r1_.recv_peer.push_back(p);
    }
    if (n_[2] > 1) {
      const int y2c = y2ch_off_[pz_+1] - y2ch_off_[pz_];
      for (int p = 0; p < pz_n_; ++p) {
        const int y2p = y2ch_off_[p+1] - y2ch_off_[p], zcp = zch_off_[p+1] - zch_off_[p];
        s2.push_back(s2.back() + static_cast<std::int64_t>(zc)*xc*y2p);
        r2.push_back(r2.back() + static_cast<std::int64_t>(zcp)*xc*y2c);
        r2_.send_peer.push_back(p);
        r2_.recv_peer.push_back(p);
      }
    }
  }
  r1_soff_ = ToDevice(s1, "pfft_r1_soff");
  r1_roff_ = ToDevice(r1, "pfft_r1_roff");
  r2_soff_ = ToDevice(s2, "pfft_r2_soff");
  r2_roff_ = ToDevice(r2, "pfft_r2_roff");
  auto to_doubles = [](const std::vector<std::int64_t> &v) {
    std::vector<std::int64_t> o(v);
    for (auto &x : o) x *= 2;
    return o;
  };
  r1_.send_off = to_doubles(s1);
  r1_.recv_off = to_doubles(r1);
  r2_.send_off = to_doubles(s2);
  r2_.recv_off = to_doubles(r2);
}

//----------------------------------------------------------------------------------------
//! \fn void PencilFFT::Exchange()
//  \brief point-to-point exchange of sbuf_ pieces into rbuf_ pieces (offsets in doubles)
//  over comm 0 = world, 1 = row, 2 = column; the message to oneself is a local copy.

void PencilFFT::Exchange(const ExchangePlan &p, int which, int tag_base) {
  int me = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = (which == 0) ? world_ : ((which == 1) ? row_ : col_);
  MPI_Comm_rank(comm, &me);
#endif
  const std::int64_t nsend = p.send_off.back(), nrecv = p.recv_off.back();
  double *sp = sbuf_.data(), *rp = rbuf_.data();
  if (host_stage_) {
    if (nsend > 0) {
      Kokkos::deep_copy(Kokkos::subview(hsbuf_, std::make_pair(std::int64_t{0}, nsend)),
                        Kokkos::subview(sbuf_, std::make_pair(std::int64_t{0}, nsend)));
    }
    sp = hsbuf_.data();
    rp = hrbuf_.data();
  }
  int self_s = -1, self_r = -1;
  for (std::size_t i = 0; i < p.send_peer.size(); ++i) {
    if (p.send_peer[i] == me) self_s = static_cast<int>(i);
  }
  for (std::size_t i = 0; i < p.recv_peer.size(); ++i) {
    if (p.recv_peer[i] == me) self_r = static_cast<int>(i);
  }
#if MPI_PARALLEL_ENABLED
  std::vector<MPI_Request> req;
  auto post = [&](bool send, int peer, double *base, std::int64_t len) {
    // chunks below INT_MAX elements; tags stay below 1000 per exchange (MPI's
    // non-overtaking rule keeps same-tag chunks of one peer pair in order)
    for (std::int64_t c = 0, k = 0; c < len; c += max_msg_, ++k) {
      const int cnt = static_cast<int>(std::min(max_msg_, len - c));
      const int tag = tag_base + static_cast<int>(k % 100);
      req.emplace_back();
      if (send) {
        MPI_Isend(base + c, cnt, MPI_DOUBLE, peer, tag, comm,
                  &req.back());
      } else {
        MPI_Irecv(base + c, cnt, MPI_DOUBLE, peer, tag, comm,
                  &req.back());
      }
    }
    max_chunks_ = std::max(max_chunks_, (len + max_msg_ - 1)/max_msg_);
  };
  for (std::size_t i = 0; i < p.recv_peer.size(); ++i) {
    if (static_cast<int>(i) == self_r) continue;
    post(false, p.recv_peer[i], rp + p.recv_off[i], p.recv_off[i+1] - p.recv_off[i]);
  }
  for (std::size_t i = 0; i < p.send_peer.size(); ++i) {
    if (static_cast<int>(i) == self_s) continue;
    post(true, p.send_peer[i], sp + p.send_off[i], p.send_off[i+1] - p.send_off[i]);
  }
#endif
  if (self_s >= 0 || self_r >= 0) {
    if (self_s < 0 || self_r < 0 || p.send_off[self_s+1] - p.send_off[self_s] !=
                                    p.recv_off[self_r+1] - p.recv_off[self_r]) {
      PFFatal("inconsistent self message in exchange " + std::to_string(tag_base));
    }
    const std::int64_t len = p.send_off[self_s+1] - p.send_off[self_s];
    if (host_stage_) {
      std::memcpy(rp + p.recv_off[self_r], sp + p.send_off[self_s], len*sizeof(double));
    } else {
      Kokkos::deep_copy(DUView1D(rp + p.recv_off[self_r], len),
                        DUView1D(sp + p.send_off[self_s], len));
    }
  }
#if MPI_PARALLEL_ENABLED
  if (!req.empty()) {
    MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
  }
#endif
  if (host_stage_ && nrecv > 0) {
    Kokkos::deep_copy(Kokkos::subview(rbuf_, std::make_pair(std::int64_t{0}, nrecv)),
                      Kokkos::subview(hrbuf_, std::make_pair(std::int64_t{0}, nrecv)));
  }
}

//----------------------------------------------------------------------------------------
//! \fn CplxArray1D PencilFFT::Forward()
//  \brief R0, x FFT, R1, y FFT, (R2, z FFT); see pencil_fft.hpp

CplxArray1D PencilFFT::Forward(const DvceArray4D<double> &in) {
  Kokkos::Timer timer;
  auto sbuf = sbuf_;
  auto rbuf = rbuf_;
  const int nx = n_[0], ny = n_[1], nz = n_[2];

  // ---- R0: blocks -> x-pencils ----
  {
    const auto seg = r0_send_seg_;
    const auto segoff = r0_send_segoff_;
    const int nseg = r0_nsend_seg_;
    Kokkos::parallel_for("pfft_r0_pack",
    Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, r0_.send_off.back()),
    KOKKOS_LAMBDA(const std::int64_t idx) {
      const int s = FindPiece(segoff, nseg, idx);
      const std::int64_t e = idx - segoff(s);
      const int ni = seg(s, 4), nj = seg(s, 5);
      const int i = static_cast<int>(e % ni);
      const int j = static_cast<int>((e/ni) % nj);
      const int k = static_cast<int>(e/(static_cast<std::int64_t>(ni)*nj));
      sbuf(idx) = in(seg(s, 0), seg(s, 3) + k, seg(s, 2) + j, seg(s, 1) + i);
    });
  }
  Kokkos::fence();
  Exchange(r0_, 0, 100);
  if (active_) {
    const auto seg = r0_recv_seg_;
    const auto segoff = r0_recv_segoff_;
    const int nseg = r0_nrecv_seg_;
    const int yc = ych_off_[py_+1] - ych_off_[py_];
    auto xpen = xpen_;
    Kokkos::parallel_for("pfft_r0_unpack",
    Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, r0_.recv_off.back()),
    KOKKOS_LAMBDA(const std::int64_t idx) {
      const int s = FindPiece(segoff, nseg, idx);
      const std::int64_t e = idx - segoff(s);
      const int ni = seg(s, 4), nj = seg(s, 5);
      const int i = seg(s, 1) + static_cast<int>(e % ni);
      const int j = seg(s, 2) + static_cast<int>((e/ni) % nj);
      const int k = seg(s, 3) + static_cast<int>(e/(static_cast<std::int64_t>(ni)*nj));
      xpen((static_cast<std::int64_t>(k)*yc + j)*nx + i) = Complex(rbuf(idx), 0.0);
    });
  }
  Kokkos::fence();
  time_[0] = timer.seconds();
  if (!active_) {
    for (int i = 1; i < 6; ++i) time_[i] = 0.0;
    return CplxArray1D();
  }

  const int yc = ych_off_[py_+1] - ych_off_[py_], zc = zch_off_[pz_+1] - zch_off_[pz_];
  const int xc = xch_off_[py_+1] - xch_off_[py_];
  // ---- x transform ----
  timer.reset();
  fx_->Forward(xpen_.data(), static_cast<std::int64_t>(zc)*yc);
  Kokkos::fence();
  time_[1] = timer.seconds();

  // ---- R1: x-pencils -> y-pencils (row comm); keep m = 0..nx/2 ----
  timer.reset();
  {
    const auto soff = r1_soff_;
    const auto xch = xch_d_;
    const int npeer = py_n_;
    auto xpen = xpen_;
    Kokkos::parallel_for("pfft_r1_pack",
    Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, r1_.send_off.back()/2),
    KOKKOS_LAMBDA(const std::int64_t idx) {
      const int p = FindPiece(soff, npeer, idx);
      const std::int64_t e = idx - soff(p);
      const int xcp = xch(p+1) - xch(p);
      const int x = xch(p) + static_cast<int>(e % xcp);
      const int y = static_cast<int>((e/xcp) % yc);
      const int z = static_cast<int>(e/(static_cast<std::int64_t>(xcp)*yc));
      const Complex v = xpen((static_cast<std::int64_t>(z)*yc + y)*nx + x);
      sbuf(2*idx) = v.real();
      sbuf(2*idx + 1) = v.imag();
    });
    Kokkos::fence();
    Exchange(r1_, 1, 200);
    const auto roff = r1_roff_;
    const auto ych = ych_d_;
    auto ypen = ypen_;
    Kokkos::parallel_for("pfft_r1_unpack",
    Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, r1_.recv_off.back()/2),
    KOKKOS_LAMBDA(const std::int64_t idx) {
      const int p = FindPiece(roff, npeer, idx);
      const std::int64_t e = idx - roff(p);
      const int ycp = ych(p+1) - ych(p);
      const int x = static_cast<int>(e % xc);
      const int y = ych(p) + static_cast<int>((e/xc) % ycp);
      const int z = static_cast<int>(e/(static_cast<std::int64_t>(xc)*ycp));
      ypen((static_cast<std::int64_t>(z)*xc + x)*ny + y) =
          Complex(rbuf(2*idx), rbuf(2*idx + 1));
    });
  }
  Kokkos::fence();
  time_[2] = timer.seconds();

  // ---- y transform ----
  timer.reset();
  fy_->Forward(ypen_.data(), static_cast<std::int64_t>(zc)*xc);
  Kokkos::fence();
  time_[3] = timer.seconds();
  if (nz == 1) {
    time_[4] = time_[5] = 0.0;
    return ypen_;
  }

  // ---- R2: y-pencils -> z-pencils (column comm) ----
  timer.reset();
  const int y2c = y2ch_off_[pz_+1] - y2ch_off_[pz_];
  {
    const auto soff = r2_soff_;
    const auto y2ch = y2ch_d_;
    const int npeer = pz_n_;
    auto ypen = ypen_;
    Kokkos::parallel_for("pfft_r2_pack",
    Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, r2_.send_off.back()/2),
    KOKKOS_LAMBDA(const std::int64_t idx) {
      const int p = FindPiece(soff, npeer, idx);
      const std::int64_t e = idx - soff(p);
      const int y2p = y2ch(p+1) - y2ch(p);
      const int y = y2ch(p) + static_cast<int>(e % y2p);
      const int x = static_cast<int>((e/y2p) % xc);
      const int z = static_cast<int>(e/(static_cast<std::int64_t>(y2p)*xc));
      const Complex v = ypen((static_cast<std::int64_t>(z)*xc + x)*ny + y);
      sbuf(2*idx) = v.real();
      sbuf(2*idx + 1) = v.imag();
    });
    Kokkos::fence();
    Exchange(r2_, 2, 300);
    const auto roff = r2_roff_;
    const auto zch = zch_d_;
    auto zpen = zpen_;
    Kokkos::parallel_for("pfft_r2_unpack",
    Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, r2_.recv_off.back()/2),
    KOKKOS_LAMBDA(const std::int64_t idx) {
      const int p = FindPiece(roff, npeer, idx);
      const std::int64_t e = idx - roff(p);
      const int y = static_cast<int>(e % y2c);
      const int x = static_cast<int>((e/y2c) % xc);
      const int z = zch(p) + static_cast<int>(e/(static_cast<std::int64_t>(y2c)*xc));
      zpen((static_cast<std::int64_t>(x)*y2c + y)*nz + z) =
          Complex(rbuf(2*idx), rbuf(2*idx + 1));
    });
  }
  Kokkos::fence();
  time_[4] = timer.seconds();

  // ---- z transform ----
  timer.reset();
  fz_->Forward(zpen_.data(), static_cast<std::int64_t>(xc)*y2c);
  Kokkos::fence();
  time_[5] = timer.seconds();
  return zpen_;
}

}  // namespace fft
