//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file area.cpp
//! \brief Distributed marching-cubes areas with optional Gaussian coarse graining.

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "area.hpp"
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "fft/local_fft.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "utils/marching_cubes.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

[[noreturn]] void AreaError(const std::string &message) {
  std::cerr << "### FATAL ERROR: area output: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

// Reach this guard on every rank before entering the next collective.
void AreaRequire(bool local_ok, const char *message) {
  int bad = local_ok ? 0 : 1;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (bad != 0) AreaError(message);
}

std::vector<std::string> AreaTokens(const std::string &text) {
  std::vector<std::string> result;
  std::size_t begin = 0;
  do {
    const auto end = text.find(',', begin);
    std::string token = text.substr(begin, end == std::string::npos ? end : end-begin);
    const auto first = token.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) AreaError("empty item in comma-separated list");
    const auto last = token.find_last_not_of(" \t\r\n");
    result.push_back(token.substr(first, last-first+1));
    if (end == std::string::npos) break;
    begin = end+1;
  } while (true);
  return result;
}

Real AreaReal(const std::string &token) {
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(token.c_str(), &end);
  if (errno == ERANGE || end == token.c_str() || *end != '\0' ||
      !std::isfinite(value) || !std::isfinite(static_cast<Real>(value))) {
    AreaError("expected a finite real value, got '"+token+"'");
  }
  return static_cast<Real>(value);
}

int AreaStride(const std::string &token) {
  char *end = nullptr;
  errno = 0;
  const auto value = std::strtol(token.c_str(), &end, 10);
  if (errno == ERANGE || end == token.c_str() || *end != '\0' || value < 1 ||
      value > std::numeric_limits<int>::max()) {
    AreaError("expected a positive integer stride, got '"+token+"'");
  }
  return static_cast<int>(value);
}

struct AreaRequestPlan {
  std::vector<int> send_count, recv_count, send_disp, recv_disp;
  DvceArray1D<std::int64_t> source;  // local active-cell offsets in incoming order
  DvceArray1D<std::int64_t> target;  // halo offsets in outgoing order
  int sent = 0, received = 0;
};

struct AreaLinePlan {
  int axis = 0, n = 0, segment_length = 0, sent = 0, received = 0;
  std::int64_t owned_lines = 0;
  std::vector<int> send_count, recv_count, send_disp, recv_disp;
  DvceArray2D<int> local_map;
  DvceArray2D<std::int64_t> owner_map;
};

struct AreaLattice {
  int x, y, z, nx, ny, nz;
};
struct AreaCoarsePlan {
  std::vector<AreaLattice> lattice;
  std::vector<std::uint64_t> prefix{0};
  std::uint64_t max_count = 0;
};

struct AreaRow {
  int method, stride;
  Real level, sigma, area;
};

}  // namespace

struct AreaOutputState {
  int n[3], nb[3], root_blocks[3], blocks = 0, gid_start = 0, total_blocks = 0;
  int root_level = 0, rank = 0, nranks = 1, component = 0;
  Real lower[3], upper[3], dx[3], sigma_factor = 1.0, truncate = 4.0;
  bool periodic[3], log_sample = false, ready = false;
  std::int64_t cells_per_block = 0;
  std::string label, configuration;
  std::vector<Real> levels;
  std::vector<int> strides, methods, block_gid, rank_eachmb, origins;
  DvceArray2D<int> local_origin;
  AreaRequestPlan halo_plan;
  std::array<AreaLinePlan,3> line_plan;
  std::vector<AreaCoarsePlan> coarse_plan;
  DvceArray4D<Real> field;
  std::vector<AreaRow> rows;
};

namespace {

void AreaValidateGeometry(Mesh *pm, const AreaOutputState &s) {
  const int n[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  const int nb[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  const Real lo[3] = {pm->mesh_size.x1min,pm->mesh_size.x2min,pm->mesh_size.x3min};
  const Real hi[3] = {pm->mesh_size.x1max,pm->mesh_size.x2max,pm->mesh_size.x3max};
  bool valid = pm->three_d && !pm->multilevel && !pm->adaptive &&
      pm->nmb_packs_thisrank == 1 && pm->nmb_total == s.total_blocks &&
      pm->pmb_pack->nmb_thispack == s.blocks && pm->nmb_thisrank == s.blocks &&
      pm->pmb_pack->gids == s.gid_start && pm->root_level == s.root_level;
  for (int a = 0; a < 3; ++a) {
    valid = valid && n[a] == s.n[a] && nb[a] == s.nb[a] &&
        lo[a] == s.lower[a] && hi[a] == s.upper[a] &&
        (pm->mesh_bcs[2*a] == BoundaryFlag::periodic) == s.periodic[a] &&
        (pm->mesh_bcs[2*a+1] == BoundaryFlag::periodic) == s.periodic[a] &&
        pm->mesh_bcs[2*a] != BoundaryFlag::shear_periodic &&
        pm->mesh_bcs[2*a+1] != BoundaryFlag::shear_periodic;
  }
  AreaRequire(valid, "mesh geometry changed after area output was configured");
  for (int g = 0; g < s.total_blocks; ++g) {
    const auto &loc = pm->lloc_eachmb[g];
    valid = valid && pm->rank_eachmb[g] == s.rank_eachmb[g] &&
        loc.level == s.root_level && loc.lx1*s.nb[0] == s.origins[3*g] &&
        loc.lx2*s.nb[1] == s.origins[3*g+1] && loc.lx3*s.nb[2] == s.origins[3*g+2];
  }
  AreaRequire(valid, "mesh ownership changed after area output was configured");
}

std::uint64_t AreaCellKey(const AreaOutputState &s, int x, int y, int z) {
  const int bi = ((z/s.nb[2])*s.root_blocks[1]+y/s.nb[1])*s.root_blocks[0]+x/s.nb[0];
  const int gid = s.block_gid[bi];
  return static_cast<std::uint64_t>(gid)*s.cells_per_block+
      (static_cast<std::uint64_t>(z%s.nb[2])*s.nb[1]+y%s.nb[1])*s.nb[0]+x%s.nb[0];
}

// Pack keys by owner and exchange their addresses once. Values use this same map later.
AreaRequestPlan AreaRequests(const AreaOutputState &s,
                            const std::vector<std::uint64_t> &keys,
                            std::vector<int> *packed_index) {
  AreaRequire(keys.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "area request count exceeds MPI int capacity");
  AreaRequestPlan p;
  p.send_count.assign(s.nranks, 0);
  p.recv_count.resize(s.nranks);
  for (auto key : keys) {
    const auto gid = key/s.cells_per_block;
    if (gid >= static_cast<std::uint64_t>(s.total_blocks)) AreaError("invalid cell key");
    ++p.send_count[s.rank_eachmb[gid]];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Alltoall(p.send_count.data(), 1, MPI_INT, p.recv_count.data(), 1, MPI_INT,
               MPI_COMM_WORLD);
#else
  p.recv_count = p.send_count;
#endif
  std::int64_t received = 0;
  for (int count : p.recv_count) received += count;
  AreaRequire(received <= std::numeric_limits<int>::max(),
              "area receive count exceeds MPI int capacity");
  p.sent = static_cast<int>(keys.size());
  p.received = static_cast<int>(received);
  p.send_disp.assign(s.nranks, 0);
  p.recv_disp.assign(s.nranks, 0);
  for (int r = 1; r < s.nranks; ++r) {
    p.send_disp[r] = p.send_disp[r-1]+p.send_count[r-1];
    p.recv_disp[r] = p.recv_disp[r-1]+p.recv_count[r-1];
  }
  std::vector<int> cursor = p.send_disp;
  packed_index->resize(keys.size());
  std::vector<std::uint64_t> send(std::max(1, p.sent)), recv(std::max(1, p.received));
  for (std::size_t q = 0; q < keys.size(); ++q) {
    const int owner = s.rank_eachmb[keys[q]/s.cells_per_block];
    const int dest = cursor[owner]++;
    (*packed_index)[q] = dest;
    send[dest] = keys[q];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Alltoallv(send.data(), p.send_count.data(), p.send_disp.data(), MPI_UINT64_T,
                recv.data(), p.recv_count.data(), p.recv_disp.data(), MPI_UINT64_T,
                MPI_COMM_WORLD);
#else
  recv = send;
#endif
  p.source = DvceArray1D<std::int64_t>("area_request_source", p.received);
  auto source_h = Kokkos::create_mirror_view(p.source);
  bool valid = true;
  for (int q = 0; q < p.received; ++q) {
    const auto gid = recv[q]/s.cells_per_block;
    valid = valid && gid >= static_cast<std::uint64_t>(s.gid_start) &&
        gid < static_cast<std::uint64_t>(s.gid_start+s.blocks) &&
        gid < static_cast<std::uint64_t>(s.total_blocks);
    if (valid) valid = s.rank_eachmb[gid] == s.rank;
    source_h(q) = static_cast<std::int64_t>(recv[q])-s.gid_start*s.cells_per_block;
  }
  AreaRequire(valid, "area vertex request has invalid ownership");
  Kokkos::deep_copy(p.source, source_h);
  return p;
}

DvceArray1D<Real> AreaExchange(const AreaOutputState &s, const AreaRequestPlan &p,
                             const DvceArray4D<Real> &field) {
  const int nx = s.nb[0], ny = s.nb[1];
  const auto nc = s.cells_per_block;
  const auto indices = p.source;
  DvceArray1D<Real> values("area_requested_values", std::max(1, p.received));
  Kokkos::parallel_for("area_fetch_vertices", fft::FFTPolicy(0, p.received),
  KOKKOS_LAMBDA(const std::int64_t q) {
    const std::int64_t cell = indices(q)%nc;
    values(q) = field(indices(q)/nc, cell/(static_cast<std::int64_t>(nx)*ny),
                      (cell/nx)%ny, cell%nx);
  });
  if (s.nranks == 1) return values;
  auto values_h = Kokkos::create_mirror_view(values);
  Kokkos::deep_copy(values_h, values);
  using HostValues = Kokkos::View<Real *, Kokkos::HostSpace>;
  HostValues returned_h("area_returned_host", std::max(1, p.sent));
#if MPI_PARALLEL_ENABLED
  MPI_Alltoallv(values_h.data(), p.recv_count.data(), p.recv_disp.data(), MPI_ATHENA_REAL,
                returned_h.data(), p.send_count.data(), p.send_disp.data(),
                MPI_ATHENA_REAL, MPI_COMM_WORLD);
#endif
  DvceArray1D<Real> returned("area_returned_values", std::max(1, p.sent));
  Kokkos::deep_copy(returned, returned_h);
  return returned;
}

void AreaBuildHalo(AreaOutputState *s) {
  const int nx = s->nb[0], ny = s->nb[1], nz = s->nb[2];
  std::int64_t hn = 1;
  for (int a = 0; a < 3; ++a) {
    const std::int64_t width = static_cast<std::int64_t>(s->nb[a])+1;
    AreaRequire(width <= std::numeric_limits<int>::max() &&
        hn <= std::numeric_limits<std::int64_t>::max()/width,
        "area halo dimensions exceed the supported integer range");
    hn *= width;
  }
  const std::int64_t surface = hn-s->cells_per_block;
  AreaRequire(surface <= std::numeric_limits<int>::max()/std::max(1, s->blocks),
              "area halo request count exceeds MPI int capacity");
  std::vector<std::uint64_t> keys;
  std::vector<std::int64_t> target;
  keys.reserve(surface*s->blocks);
  target.reserve(surface*s->blocks);
  for (int m = 0; m < s->blocks; ++m) {
    const int gid = s->gid_start+m;
    for (int k = 0; k <= nz; ++k) for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        if (i != nx && j != ny && k != nz) continue;
        int index[3] = {s->origins[3*gid]+i,s->origins[3*gid+1]+j,
                        s->origins[3*gid+2]+k};
        for (int a = 0; a < 3; ++a) {
          if (index[a] == s->n[a]) index[a] = s->periodic[a] ? 0 : s->n[a]-1;
        }
        keys.push_back(AreaCellKey(*s, index[0], index[1], index[2]));
        target.push_back(m*hn+(static_cast<std::int64_t>(k)*(ny+1)+j)*(nx+1)+i);
      }
    }
  }
  std::vector<int> packed;
  s->halo_plan = AreaRequests(*s, keys, &packed);
  s->halo_plan.target = DvceArray1D<std::int64_t>("area_halo_target", keys.size());
  auto target_h = Kokkos::create_mirror_view(s->halo_plan.target);
  for (std::size_t q = 0; q < target.size(); ++q) target_h(packed[q]) = target[q];
  Kokkos::deep_copy(s->halo_plan.target, target_h);
}

// A common scale is essential: adjacent cubes must use the same tolerance on
// their shared face, including the MC utility's absolute ambiguity tests.
Real AreaNormalization(const AreaOutputState &s, const DvceArray4D<Real> &field,
                       bool logarithmic) {
  const auto nc = s.cells_per_block;
  const int nx = s.nb[0], ny = s.nb[1];
  Real largest = 0.0;
  Kokkos::parallel_reduce("area_scalar_scale",fft::FFTPolicy(0,nc*s.blocks),
  KOKKOS_LAMBDA(const std::int64_t q, Real &maximum) {
    const auto c = q%nc;
    const Real value = field(q/nc,c/(static_cast<std::int64_t>(nx)*ny),(c/nx)%ny,c%nx);
    const Real magnitude = fabs(logarithmic ? log10(value) : value);
    maximum = fmax(maximum,magnitude);
  },Kokkos::Max<Real>(largest));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&largest,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  return largest;
}

Real AreaLevelScale(Real largest, Real level) {
  const Real scale = std::max(largest,std::abs(level));
  return scale == 0.0 ? 1.0 : scale;
}

KOKKOS_INLINE_FUNCTION
Real AreaCube(const Real *value, Real level, Real scale,
              const marching_cubes::CellSpacing &spacing) {
  // Normalize every cube of this extraction by the same globally reduced scale.
  // Nonzero levels use a vertex-relative tie tolerance. Zero levels use 1e-12
  // of the global scale to suppress roundoff on zero-valued plateaus (including
  // log10(T)=0 at T=1). Near-level ties belong to the outside phase.
  Real v[8];
  const Real b = level/scale;
  for (int q = 0; q < 8; ++q) {
    const Real a = value[q]/scale;
    const Real epsilon = level == 0.0 ? 1.0e-12 :
                         1.0e-12*fmax(fabs(a),fabs(b));
    v[q] = a-b;
    if (fabs(v[q]) <= epsilon) {
      v[q] = epsilon > 0.0 ? epsilon : std::numeric_limits<Real>::min();
    }
  }
  const marching_cubes::Cube cube(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],spacing);
  return marching_cubes::ComputeArea(cube);
}

std::vector<Real> AreaFine(const AreaOutputState &s, const DvceArray4D<Real> &field,
                           const std::vector<Real> &levels, bool logarithmic) {
  const int nx = s.nb[0], ny = s.nb[1], nz = s.nb[2];
  const int gx = s.n[0], gy = s.n[1], gz = s.n[2];
  const bool px = s.periodic[0], py = s.periodic[1], pz = s.periodic[2];
  const auto nc = s.cells_per_block;
  const std::int64_t hn = static_cast<std::int64_t>(nx+1)*(ny+1)*(nz+1);
  const auto origins = s.local_origin;
  DvceArray4D<Real> halo("area_scalar_halo", s.blocks, nz+1, ny+1, nx+1);
  Kokkos::parallel_for("area_halo_interior", fft::FFTPolicy(0, nc*s.blocks),
  KOKKOS_LAMBDA(const std::int64_t q) {
    const std::int64_t c = q%nc;
    const int i = c%nx, j = (c/nx)%ny, k = c/(static_cast<std::int64_t>(nx)*ny);
    halo(q/nc,k,j,i) = field(q/nc,k,j,i);
  });
  const auto values = AreaExchange(s, s.halo_plan, field);
  const auto target = s.halo_plan.target;
  Kokkos::parallel_for("area_halo_upper_layer", fft::FFTPolicy(0, s.halo_plan.sent),
  KOKKOS_LAMBDA(const std::int64_t q) {
    const auto c = target(q)%hn;
    halo(target(q)/hn,c/(static_cast<std::int64_t>(nx+1)*(ny+1)),
         (c/(nx+1))%(ny+1),c%(nx+1)) = values(q);
  });
  const marching_cubes::CellSpacing spacing(s.dx[0],s.dx[1],s.dx[2]);
  std::vector<Real> total(levels.size(), 0.0);
  const Real largest = AreaNormalization(s,field,logarithmic);
  for (std::size_t first = 0; first < levels.size(); first += 3) {
    const int count = std::min<std::size_t>(3, levels.size()-first);
    const Real l0 = logarithmic ? std::log10(levels[first]) : levels[first];
    const Real l1 = count > 1 ? (logarithmic ? std::log10(levels[first+1]) :
                                 levels[first+1]) : l0;
    const Real l2 = count > 2 ? (logarithmic ? std::log10(levels[first+2]) :
                                 levels[first+2]) : l0;
    const Real scale0 = AreaLevelScale(largest,l0), scale1 = AreaLevelScale(largest,l1);
    const Real scale2 = AreaLevelScale(largest,l2);
    Real a0 = 0.0, a1 = 0.0, a2 = 0.0;
    Kokkos::parallel_reduce("area_fine_cubes", fft::FFTPolicy(0, nc*s.blocks),
    KOKKOS_LAMBDA(const std::int64_t q, Real &v0, Real &v1, Real &v2) {
      const std::int64_t c = q%nc;
      const int m = q/nc, i = c%nx, j = (c/nx)%ny;
      const int k = c/(static_cast<std::int64_t>(nx)*ny);
      if ((!px && origins(m,0)+i == gx-1) || (!py && origins(m,1)+j == gy-1) ||
          (!pz && origins(m,2)+k == gz-1)) return;
      Real data[8] = {halo(m,k,j,i),halo(m,k,j,i+1),halo(m,k,j+1,i+1),
          halo(m,k,j+1,i),halo(m,k+1,j,i),halo(m,k+1,j,i+1),
          halo(m,k+1,j+1,i+1),halo(m,k+1,j+1,i)};
      if (logarithmic) {
        for (int v = 0; v < 8; ++v) data[v] = log10(data[v]);
      }
      v0 += AreaCube(data,l0,scale0,spacing);
      if (count > 1) v1 += AreaCube(data,l1,scale1,spacing);
      if (count > 2) v2 += AreaCube(data,l2,scale2,spacing);
    },Kokkos::Sum<Real>(a0),Kokkos::Sum<Real>(a1),Kokkos::Sum<Real>(a2));
    total[first] = a0;
    if (count > 1) total[first+1] = a1;
    if (count > 2) total[first+2] = a2;
  }
  return total;
}

// Apply the normalized, finite Gaussian along complete real lines.  Sigma is
// in cells and radius=floor(truncate*sigma+0.5).  FFTs evaluate this discrete
// convolution, including its boundary extension; they do not substitute the
// continuum Fourier multiplier exp(-k^2 sigma^2/2).
void AreaFilterLines(const DvceArray2D<Real> &lines, std::int64_t nline,
                         int n, bool periodic, Real sigma, Real truncate) {
  if (n < 1 || nline < 0 || lines.extent(0) != static_cast<std::size_t>(nline) ||
      lines.extent(1) != static_cast<std::size_t>(n) || !std::isfinite(sigma) ||
      sigma < 0.0 || !std::isfinite(truncate) || truncate <= 0.0) {
    AreaError("invalid Gaussian line-filter dimensions or width");
  }
  const double support = static_cast<double>(sigma)*truncate;
  if (!std::isfinite(support) ||
      support > static_cast<double>(std::numeric_limits<int>::max()-1)) {
    AreaError("Gaussian kernel radius exceeds supported integer range");
  }
  const int radius = static_cast<int>(std::floor(support+0.5));
  if (nline == 0 || n == 1 || radius == 0) return;

  std::int64_t length = n;
  if (!periodic) {
    length = 1;
    while (length < 2*static_cast<std::int64_t>(n)-1) length *= 2;
  }
  if (length > (std::int64_t{1} << 30)) {
    AreaError("Gaussian line FFT exceeds supported transform length");
  }
  const int nfft = static_cast<int>(length);
  std::vector<double> weight(static_cast<std::size_t>(radius)+1);
  double norm = 1.0;
  weight[0] = 1.0;
  for (int d = 1; d <= radius; ++d) {
    const double z = static_cast<double>(d)/sigma;
    weight[d] = std::exp(-0.5*z*z);
    norm += 2.0*weight[d];
  }
  for (double &value : weight) value /= norm;

  fft::CplxArray1D kernel("area_gaussian_kernel", nfft);
  auto kernel_h = Kokkos::create_mirror_view(kernel);
  for (int q = 0; q < nfft; ++q) kernel_h(q) = fft::Complex(0.0, 0.0);
  kernel_h(0) = fft::Complex(weight[0], 0.0);
  for (int d = 1; d <= radius; ++d) {
    if (periodic) {
      // Multiple wrapped images contribute when the support exceeds the box.
      const int q = d % n;
      kernel_h(q) += fft::Complex(weight[d], 0.0);
      kernel_h(q == 0 ? 0 : n-q) += fft::Complex(weight[d], 0.0);
    } else if (d < n) {
      kernel_h(d) += fft::Complex(weight[d], 0.0);
      kernel_h(nfft-d) += fft::Complex(weight[d], 0.0);
    }
  }
  Kokkos::deep_copy(kernel, kernel_h);

  // tail(i)=sum_{d=i+1}^radius weight[d].  Missing samples outside a nearest
  // boundary equal that endpoint, even when radius exceeds the line length.
  DvceArray1D<double> tail("area_gaussian_tail", n);
  auto tail_h = Kokkos::create_mirror_view(tail);
  for (int i = 0; i < n; ++i) tail_h(i) = 0.0;
  double cumulative = 0.0;
  for (int d = radius; d >= 1; --d) {
    cumulative += weight[d];
    if (d <= n) tail_h(d-1) = cumulative;
  }
  Kokkos::deep_copy(tail, tail_h);

  fft::LocalFFT plan(nfft);
  plan.Forward(kernel.data(), 1);
  // Bound the batch scratch independently of sigma and global cell count.
  // Bluestein uses a larger internal convolution, included in this estimate.
  const std::int64_t line_work = plan.UsesBluestein() ?
      nfft+2*static_cast<std::int64_t>(plan.ConvolutionSize()) : nfft;
  const std::int64_t batch = std::max<std::int64_t>(1, 262144/line_work);
  const std::int64_t capacity = std::min(nline, batch);
  fft::CplxArray1D work("area_gaussian_fft_batch", capacity*nfft);
  DvceArray2D<Real> endpoints("area_gaussian_endpoints", capacity, 2);
  for (std::int64_t first = 0; first < nline; first += batch) {
    const std::int64_t count = std::min(batch, nline-first);
    const std::int64_t size = count*nfft;
    Kokkos::parallel_for("area_gaussian_fft_pack", fft::FFTPolicy(0, size),
    KOKKOS_LAMBDA(const std::int64_t q) {
      const std::int64_t row = q/nfft;
      const int i = static_cast<int>(q % nfft);
      work(q) = fft::Complex(i < n ? lines(first+row, i) : 0.0, 0.0);
      if (i == 0) {
        endpoints(row, 0) = lines(first+row, 0);
        endpoints(row, 1) = lines(first+row, n-1);
      }
    });
    plan.Forward(work.data(), count);
    Kokkos::parallel_for("area_gaussian_fft_multiply", fft::FFTPolicy(0, size),
    KOKKOS_LAMBDA(const std::int64_t q) {
      work(q) *= kernel(q % nfft);
    });
    plan.Inverse(work.data(), count);
    Kokkos::parallel_for("area_gaussian_fft_unpack", fft::FFTPolicy(0, count*n),
    KOKKOS_LAMBDA(const std::int64_t q) {
      const std::int64_t row = q/n;
      const int i = static_cast<int>(q % n);
      double value = work(row*nfft+i).real()/nfft;
      if (!periodic) {
        value += endpoints(row, 0)*tail(i)+endpoints(row, 1)*tail(n-1-i);
      }
      lines(first+row, i) = static_cast<Real>(value);
    });
  }
  Kokkos::fence();
}

// Geometry-only line maps are reused by every filter width and every output time.
AreaLinePlan AreaBuildLines(const AreaOutputState &s, int axis) {
  AreaLinePlan p;
  p.axis = axis;
  p.n = s.n[axis];
  p.segment_length = s.nb[axis];
  const int a = (axis+1)%3, b = (axis+2)%3;
  const std::int64_t total_lines = static_cast<std::int64_t>(s.n[a])*s.n[b];
  p.owned_lines = s.rank < total_lines ? (total_lines-1-s.rank)/s.nranks+1 : 0;
  const std::int64_t segments = static_cast<std::int64_t>(s.blocks)*s.nb[a]*s.nb[b];
  const auto imax = std::numeric_limits<int>::max();
  AreaRequire(segments <= imax/2 && segments <= imax/p.segment_length &&
      p.owned_lines <= imax/p.n, "Gaussian redistribution exceeds MPI int capacity");
  std::vector<int> send_segments(s.nranks,0), recv_segments(s.nranks,0);
  for (int m = 0; m < s.blocks; ++m) {
    const int *origin = &s.origins[3*(s.gid_start+m)];
    for (int ib = 0; ib < s.nb[b]; ++ib) for (int ia = 0; ia < s.nb[a]; ++ia) {
      const std::int64_t line = origin[a]+ia+
          static_cast<std::int64_t>(s.n[a])*(origin[b]+ib);
      ++send_segments[line%s.nranks];
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Alltoall(send_segments.data(),1,MPI_INT,recv_segments.data(),1,MPI_INT,
               MPI_COMM_WORLD);
#else
  recv_segments = send_segments;
#endif
  std::int64_t receiving = 0;
  for (int count : recv_segments) receiving += count;
  AreaRequire(receiving <= imax/2 && receiving <= imax/p.segment_length &&
      receiving*p.segment_length == p.owned_lines*p.n,
      "Gaussian line ownership is incomplete or exceeds MPI count capacity");
  p.send_count.resize(s.nranks); p.recv_count.resize(s.nranks);
  p.send_disp.resize(s.nranks); p.recv_disp.resize(s.nranks);
  std::vector<int> sm_count(s.nranks), rm_count(s.nranks), sm_disp(s.nranks);
  std::vector<int> rm_disp(s.nranks), cursor(s.nranks);
  int sm_size = 0, rm_size = 0;
  for (int r = 0; r < s.nranks; ++r) {
    p.send_disp[r] = p.sent; p.recv_disp[r] = p.received;
    sm_disp[r] = sm_size; rm_disp[r] = rm_size;
    cursor[r] = sm_size/2;
    p.send_count[r] = send_segments[r]*p.segment_length;
    p.recv_count[r] = recv_segments[r]*p.segment_length;
    sm_count[r] = 2*send_segments[r]; rm_count[r] = 2*recv_segments[r];
    p.sent += p.send_count[r]; p.received += p.recv_count[r];
    sm_size += sm_count[r]; rm_size += rm_count[r];
  }
  p.local_map = DvceArray2D<int>("area_filter_local_segments",segments,4);
  auto local_h = Kokkos::create_mirror_view(p.local_map);
  std::vector<std::int64_t> send_meta(std::max(1,sm_size));
  std::vector<std::int64_t> recv_meta(std::max(1,rm_size));
  for (int m = 0; m < s.blocks; ++m) {
    const int *origin = &s.origins[3*(s.gid_start+m)];
    for (int ib = 0; ib < s.nb[b]; ++ib) for (int ia = 0; ia < s.nb[a]; ++ia) {
      const std::int64_t line = origin[a]+ia+
          static_cast<std::int64_t>(s.n[a])*(origin[b]+ib);
      const int index = cursor[line%s.nranks]++;
      int xyz[3] = {0,0,0}; xyz[a] = ia; xyz[b] = ib;
      local_h(index,0) = m;
      for (int d = 0; d < 3; ++d) local_h(index,d+1) = xyz[d];
      send_meta[2*index] = line/s.nranks;
      send_meta[2*index+1] = origin[axis];
    }
  }
  Kokkos::deep_copy(p.local_map,local_h);
#if MPI_PARALLEL_ENABLED
  MPI_Alltoallv(send_meta.data(),sm_count.data(),sm_disp.data(),MPI_INT64_T,
                recv_meta.data(),rm_count.data(),rm_disp.data(),MPI_INT64_T,
                MPI_COMM_WORLD);
#else
  recv_meta = send_meta;
#endif
  p.owner_map = DvceArray2D<std::int64_t>("area_filter_owned_segments",receiving,2);
  auto owner_h = Kokkos::create_mirror_view(p.owner_map);
  bool valid = true;
  for (std::int64_t q = 0; q < receiving; ++q) {
    const std::int64_t line = recv_meta[2*q], start = recv_meta[2*q+1];
    valid = valid && line >= 0 && line < p.owned_lines && start >= 0 &&
        start <= p.n-p.segment_length && start%p.segment_length == 0;
    owner_h(q,0) = line; owner_h(q,1) = start;
  }
  AreaRequire(valid,"invalid Gaussian block-line ownership metadata");
  Kokkos::deep_copy(p.owner_map,owner_h);
  return p;
}

DvceArray4D<Real> AreaSmooth(const AreaOutputState &s, const DvceArray4D<Real> &field,
                            Real sigma) {
  DvceArray4D<Real> result("area_filtered_field",s.blocks,s.nb[2],s.nb[1],s.nb[0]);
  Kokkos::deep_copy(result,field);
  if (sigma*s.truncate < 0.5) return result;
  for (int axis = 2; axis >= 0; --axis) {
    const auto &p = s.line_plan[axis];
    const int length = p.segment_length;
    const auto local = p.local_map;
    const auto owner = p.owner_map;
    using HostValues = Kokkos::View<Real *,Kokkos::HostSpace>;
    HostValues send_h("area_filter_send_host",std::max(1,p.sent));
    HostValues recv_h("area_filter_recv_host",std::max(1,p.received));
    DvceArray1D<Real> send("area_filter_send",std::max(1,p.sent));
    Kokkos::parallel_for("area_filter_pack",fft::FFTPolicy(0,p.sent),
    KOKKOS_LAMBDA(const std::int64_t q) {
      const auto segment = q/length;
      const int d = q%length;
      const int i = local(segment,1)+(axis == 0 ? d : 0);
      const int j = local(segment,2)+(axis == 1 ? d : 0);
      const int k = local(segment,3)+(axis == 2 ? d : 0);
      send(q) = result(local(segment,0),k,j,i);
    });
    Kokkos::deep_copy(send_h,send);
    send = DvceArray1D<Real>();
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(send_h.data(),p.send_count.data(),p.send_disp.data(),MPI_ATHENA_REAL,
                  recv_h.data(),p.recv_count.data(),p.recv_disp.data(),MPI_ATHENA_REAL,
                  MPI_COMM_WORLD);
#else
    Kokkos::deep_copy(recv_h,send_h);
#endif
    DvceArray2D<Real> owned("area_filter_lines",p.owned_lines,p.n);
    DvceArray1D<Real> recv("area_filter_recv",std::max(1,p.received));
    Kokkos::deep_copy(recv,recv_h);
    Kokkos::parallel_for("area_filter_assemble",fft::FFTPolicy(0,p.received),
    KOKKOS_LAMBDA(const std::int64_t q) {
      owned(owner(q/length,0),owner(q/length,1)+q%length) = recv(q);
    });
    Kokkos::fence();
    recv = DvceArray1D<Real>();
    AreaFilterLines(owned,p.owned_lines,p.n,s.periodic[axis],sigma,s.truncate);
    recv = DvceArray1D<Real>("area_filter_return",std::max(1,p.received));
    Kokkos::parallel_for("area_filter_disassemble",fft::FFTPolicy(0,p.received),
    KOKKOS_LAMBDA(const std::int64_t q) {
      recv(q) = owned(owner(q/length,0),owner(q/length,1)+q%length);
    });
    Kokkos::deep_copy(recv_h,recv);
    recv = DvceArray1D<Real>();
    owned = DvceArray2D<Real>();
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(recv_h.data(),p.recv_count.data(),p.recv_disp.data(),MPI_ATHENA_REAL,
                  send_h.data(),p.send_count.data(),p.send_disp.data(),MPI_ATHENA_REAL,
                  MPI_COMM_WORLD);
#else
    Kokkos::deep_copy(send_h,recv_h);
#endif
    send = DvceArray1D<Real>("area_filter_returned",std::max(1,p.sent));
    Kokkos::deep_copy(send,send_h);
    Kokkos::parallel_for("area_filter_unpack",fft::FFTPolicy(0,p.sent),
    KOKKOS_LAMBDA(const std::int64_t q) {
      const auto segment = q/length;
      const int d = q%length;
      const int i = local(segment,1)+(axis == 0 ? d : 0);
      const int j = local(segment,2)+(axis == 1 ? d : 0);
      const int k = local(segment,3)+(axis == 2 ? d : 0);
      result(local(segment,0),k,j,i) = send(q);
    });
    Kokkos::fence();
  }
  return result;
}

struct AreaInterval { int next, width; };
AreaInterval AreaAxis(int origin, int cells, int step, bool periodic) {
  const int remaining = cells-origin-(periodic ? 0 : 1);
  const int width = std::min(step,remaining);
  return {periodic && width == remaining ? 0 : origin+width,width};
}

AreaCoarsePlan AreaBuildCoarse(const AreaOutputState &s, int step) {
  AreaCoarsePlan p;
  for (int m = 0; m < s.blocks; ++m) {
    int begin[3], count[3];
    for (int a = 0; a < 3; ++a) {
      const int origin = s.origins[3*(s.gid_start+m)+a];
      const std::int64_t first = ((static_cast<std::int64_t>(origin)+step-1)/step)*step;
      const int end = std::min(origin+s.nb[a],s.n[a]-(s.periodic[a] ? 0 : 1));
      begin[a] = first < end ? static_cast<int>(first) : end;
      count[a] = first < end ? static_cast<int>((end-1-first)/step+1) : 0;
    }
    const auto count_total = static_cast<std::uint64_t>(count[0])*count[1]*count[2];
    if (count_total == 0) continue;
    p.lattice.push_back({begin[0],begin[1],begin[2],count[0],count[1],count[2]});
    p.prefix.push_back(p.prefix.back()+count_total);
  }
  p.max_count = p.prefix.back();
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&p.max_count,1,MPI_UINT64_T,MPI_MAX,MPI_COMM_WORLD);
#endif
  return p;
}

std::vector<Real> AreaCoarse(const AreaOutputState &s, const AreaCoarsePlan &plan,
                             const DvceArray4D<Real> &field, int step) {
  const int cx[8] = {0,1,1,0,0,1,1,0}, cy[8] = {0,0,1,1,0,0,1,1};
  const int cz[8] = {0,0,0,0,1,1,1,1};
  const Real dx = s.dx[0], dy = s.dx[1], dz = s.dx[2];
  const bool logarithmic = s.log_sample;
  std::vector<Real> total(s.levels.size(),0.0);
  const Real largest = AreaNormalization(s,field,logarithmic);
  constexpr int max_vertices = 1048576;
  int batch = 32768;
  for (std::uint64_t begin = 0; begin < plan.max_count;) {
    const int n = begin < plan.prefix.back() ?
        std::min<std::uint64_t>(batch,plan.prefix.back()-begin) : 0;
    std::unordered_map<std::uint64_t,int> unique;
    std::vector<std::uint64_t> keys;
    std::vector<int> corners(8*n), send_count(s.nranks,0), recv_count(s.nranks,0);
    DvceArray2D<int> widths("area_coarse_width",n,3);
    auto widths_h = Kokkos::create_mirror_view(widths);
    unique.reserve(8*n);
    for (int c = 0; c < n; ++c) {
      const auto index = begin+c;
      const auto bidx = std::upper_bound(plan.prefix.begin(),plan.prefix.end(),index)-
          plan.prefix.begin()-1;
      const auto &b = plan.lattice[bidx];
      const auto offset = index-plan.prefix[bidx];
      const int x = b.x+(offset%b.nx)*step;
      const int y = b.y+((offset/b.nx)%b.ny)*step;
      const int z = b.z+(offset/(static_cast<std::uint64_t>(b.nx)*b.ny))*step;
      const auto ix = AreaAxis(x,s.n[0],step,s.periodic[0]);
      const auto iy = AreaAxis(y,s.n[1],step,s.periodic[1]);
      const auto iz = AreaAxis(z,s.n[2],step,s.periodic[2]);
      widths_h(c,0) = ix.width; widths_h(c,1) = iy.width; widths_h(c,2) = iz.width;
      for (int v = 0; v < 8; ++v) {
        const auto key = AreaCellKey(s,cx[v] ? ix.next : x,cy[v] ? iy.next : y,
                                     cz[v] ? iz.next : z);
        auto inserted = unique.emplace(key,keys.size());
        if (inserted.second) {
          keys.push_back(key);
          ++send_count[s.rank_eachmb[key/s.cells_per_block]];
        }
        corners[8*c+v] = inserted.first->second;
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Alltoall(send_count.data(),1,MPI_INT,recv_count.data(),1,MPI_INT,MPI_COMM_WORLD);
#else
    recv_count = send_count;
#endif
    std::int64_t largest_receive = 0;
    for (int count : recv_count) largest_receive += count;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&largest_receive,1,MPI_INT64_T,MPI_MAX,MPI_COMM_WORLD);
#endif
    if (largest_receive > max_vertices) {
      if (batch == 1) AreaError("coarse-vertex receive budget exceeded");
      batch = std::max<std::int64_t>(1,
          batch/((largest_receive+max_vertices-1)/max_vertices));
      continue;
    }
    std::vector<int> packed;
    const auto request = AreaRequests(s,keys,&packed);
    const auto values = AreaExchange(s,request,field);
    DvceArray2D<int> indices("area_coarse_indices",n,8);
    auto indices_h = Kokkos::create_mirror_view(indices);
    for (int c = 0; c < n; ++c) for (int v = 0; v < 8; ++v) {
      indices_h(c,v) = packed[corners[8*c+v]];
    }
    Kokkos::deep_copy(indices,indices_h);
    Kokkos::deep_copy(widths,widths_h);
    for (std::size_t first = 0; first < s.levels.size(); first += 3) {
      const int count = std::min<std::size_t>(3,s.levels.size()-first);
      const Real l0 = logarithmic ? std::log10(s.levels[first]) : s.levels[first];
      const Real l1 = count > 1 ? (logarithmic ? std::log10(s.levels[first+1]) :
                                   s.levels[first+1]) : l0;
      const Real l2 = count > 2 ? (logarithmic ? std::log10(s.levels[first+2]) :
                                   s.levels[first+2]) : l0;
      const Real scale0 = AreaLevelScale(largest,l0), scale1 = AreaLevelScale(largest,l1);
      const Real scale2 = AreaLevelScale(largest,l2);
      Real a0 = 0.0, a1 = 0.0, a2 = 0.0;
      Kokkos::parallel_reduce("area_coarse_cubes",fft::FFTPolicy(0,n),
      KOKKOS_LAMBDA(const std::int64_t c, Real &v0, Real &v1, Real &v2) {
        Real data[8];
        for (int v = 0; v < 8; ++v) {
          const Real value = values(indices(c,v));
          data[v] = logarithmic ? log10(value) : value;
        }
        const marching_cubes::CellSpacing spacing(widths(c,0)*dx,widths(c,1)*dy,
                                                   widths(c,2)*dz);
        v0 += AreaCube(data,l0,scale0,spacing);
        if (count > 1) v1 += AreaCube(data,l1,scale1,spacing);
        if (count > 2) v2 += AreaCube(data,l2,scale2,spacing);
      },Kokkos::Sum<Real>(a0),Kokkos::Sum<Real>(a1),Kokkos::Sum<Real>(a2));
      total[first] += a0;
      if (count > 1) total[first+1] += a1;
      if (count > 2) total[first+2] += a2;
    }
    begin += batch;
  }
  return total;
}

void AreaCheckField(const AreaOutputState &s, const DvceArray4D<Real> &field,
                    bool positive) {
  const auto nc = s.cells_per_block;
  const int nx = s.nb[0], ny = s.nb[1];
  std::int64_t bad = 0;
  Kokkos::parallel_reduce("area_check_scalar",fft::FFTPolicy(0,nc*s.blocks),
  KOKKOS_LAMBDA(const std::int64_t q, std::int64_t &invalid) {
    const auto c = q%nc;
    const Real value = field(q/nc,c/(static_cast<std::int64_t>(nx)*ny),(c/nx)%ny,c%nx);
    if (!Kokkos::isfinite(value) || (positive && value <= 0.0)) ++invalid;
  },Kokkos::Sum<std::int64_t>(bad));
  AreaRequire(bad == 0, positive ?
      "log10 sampled areas require positive finite scalar data" :
      "area calculation requires finite scalar data");
}

const char *AreaMethod(int method) {
  return method == 0 ? "sampled" : (method == 1 ? "indicator" : "field");
}

std::string AreaFilename(const OutputParameters &op) {
  std::ostringstream name;
  name << "area_" << op.file_id << '/' << op.file_basename << '.' << std::setw(5)
       << std::setfill('0') << op.file_number << ".area";
  return name.str();
}

void AreaWriteExclusive(const std::string &path, const std::string &text) {
  const int fd = open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0664);
  if (fd < 0) AreaError("cannot create '"+path+"': "+std::strerror(errno));
  std::size_t written = 0;
  while (written < text.size()) {
    const auto count = write(fd,text.data()+written,text.size()-written);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      close(fd);
      AreaError("failed writing '"+path+"'");
    }
    written += count;
  }
  if (close(fd) != 0) AreaError("failed closing '"+path+"'");
}

void AreaCheckDestination(const OutputParameters &op, const AreaOutputState &s) {
  if (s.rank != 0) return;
  const std::string directory = "area_"+op.file_id;
  if (mkdir(directory.c_str(),0775) != 0 && errno != EEXIST) {
    AreaError("cannot create directory '"+directory+"'");
  }
  const std::string manifest = directory+"/configuration.txt";
  struct stat info;
  if (stat(manifest.c_str(),&info) == 0) {
    FILE *fp = std::fopen(manifest.c_str(),"rb");
    if (fp == nullptr) AreaError("cannot read '"+manifest+"'");
    std::string saved;
    char buffer[4096];
    std::size_t size;
    while ((size = std::fread(buffer,1,sizeof(buffer),fp)) != 0) {
      saved.append(buffer,size);
    }
    const bool bad = std::ferror(fp) != 0;
    std::fclose(fp);
    if (bad || saved != s.configuration) {
      AreaError("area configuration differs from '"+manifest+"'; choose a new output id");
    }
  } else if (errno == ENOENT) {
    AreaWriteExclusive(manifest,s.configuration);
  } else {
    AreaError("cannot inspect '"+manifest+"'");
  }
  const auto filename = AreaFilename(op);
  if (stat(filename.c_str(),&info) == 0) {
    AreaError("refusing to overwrite existing event '"+filename+
              "'; choose a new output id");
  }
  if (errno != ENOENT) AreaError("cannot inspect output path '"+filename+"'");
}

}  // namespace

AreaOutput::AreaOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
    BaseTypeOutput(pin,pm,op), state_(new AreaOutputState) {
  auto &s = *state_;
  const auto &block = out_params.block_name;
  auto *pack = pm->pmb_pack;
  if (!pm->three_d || pm->multilevel || pm->adaptive || pm->nmb_packs_thisrank != 1) {
    AreaError("area output requires a uniform three-dimensional mesh "
              "with one pack per rank");
  }
  if (pack->pdyngr != nullptr || (pack->pcoord != nullptr &&
      (pack->pcoord->is_general_relativistic || pack->pcoord->is_special_relativistic ||
       pack->pcoord->is_dynamical_relativistic))) {
    AreaError("area output supports Newtonian Cartesian coordinates");
  }
  if (outvars.empty()) AreaError("area output requires one scalar variable");
  const bool has_component = pin->DoesParameterExist(block,"component");
  if (outvars.size() != 1 && !has_component) {
    AreaError("variable expands to multiple fields; select a zero-based component");
  }
  s.component = pin->GetOrAddInteger(block,"component",0);
  if (s.component < 0 || static_cast<std::size_t>(s.component) >= outvars.size()) {
    AreaError("component is outside the selected variable's fields");
  }
  s.label = outvars[s.component].label;
  if (out_params.variable == "temperature") {
    if ((pack->phydro == nullptr) == (pack->pmhd == nullptr)) {
      AreaError("temperature area requires exactly one hydro or MHD fluid");
    }
    const bool ideal = pack->phydro != nullptr ? pack->phydro->peos->eos_data.is_ideal :
                                                pack->pmhd->peos->eos_data.is_ideal;
    if (!ideal) AreaError("temperature area requires an ideal-gas EOS");
  }
  for (const auto &token : AreaTokens(pin->GetString(block,"levels"))) {
    const Real value = AreaReal(token);
    if (std::find(s.levels.begin(),s.levels.end(),value) != s.levels.end()) {
      AreaError("duplicate area level");
    }
    s.levels.push_back(value);
  }
  for (const auto &token : AreaTokens(pin->GetString(block,"strides"))) {
    const int value = AreaStride(token);
    if (std::find(s.strides.begin(),s.strides.end(),value) != s.strides.end()) {
      AreaError("duplicate area stride");
    }
    s.strides.push_back(value);
  }
  for (const auto &token : AreaTokens(pin->GetOrAddString(block,"methods","sampled"))) {
    const int method = token == "sampled" ? 0 : token == "indicator" ? 1 :
                       token == "field" ? 2 : -1;
    if (method < 0) AreaError("methods must contain sampled, indicator or field");
    if (std::find(s.methods.begin(),s.methods.end(),method) != s.methods.end()) {
      AreaError("duplicate area method");
    }
    s.methods.push_back(method);
  }
  const auto transform = pin->GetOrAddString(block,"sample_transform","linear");
  if (transform != "linear" && transform != "log10") {
    AreaError("sample_transform must be linear or log10");
  }
  s.log_sample = transform == "log10";
  s.sigma_factor = pin->GetOrAddReal(block,"sigma_factor",1.0);
  s.truncate = pin->GetOrAddReal(block,"truncate",4.0);
  if (!std::isfinite(s.sigma_factor) || s.sigma_factor <= 0.0 ||
      !std::isfinite(s.truncate) || s.truncate <= 0.0) {
    AreaError("sigma_factor and truncate must be positive and finite");
  }
  const bool sampled = std::find(s.methods.begin(),s.methods.end(),0) != s.methods.end();
  if (sampled && s.log_sample) {
    for (Real level : s.levels) if (level <= 0.0) {
      AreaError("log10 sampled areas require positive levels in original scalar units");
    }
  }
  s.n[0] = pm->mesh_indcs.nx1; s.n[1] = pm->mesh_indcs.nx2; s.n[2] = pm->mesh_indcs.nx3;
  s.nb[0] = pm->mb_indcs.nx1; s.nb[1] = pm->mb_indcs.nx2; s.nb[2] = pm->mb_indcs.nx3;
  s.lower[0] = pm->mesh_size.x1min; s.lower[1] = pm->mesh_size.x2min;
  s.lower[2] = pm->mesh_size.x3min;
  s.upper[0] = pm->mesh_size.x1max; s.upper[1] = pm->mesh_size.x2max;
  s.upper[2] = pm->mesh_size.x3max;
  s.root_blocks[0] = pm->nmb_rootx1; s.root_blocks[1] = pm->nmb_rootx2;
  s.root_blocks[2] = pm->nmb_rootx3;
  s.blocks = pack->nmb_thispack; s.gid_start = pack->gids;
  s.total_blocks = pm->nmb_total; s.root_level = pm->root_level;
  s.rank = global_variable::my_rank; s.nranks = global_variable::nranks;
  std::int64_t global_cells = 1, root_blocks = 1;
  bool valid = s.nranks > 0 && s.rank >= 0 && s.rank < s.nranks && s.blocks >= 0 &&
      s.blocks == pm->nmb_thisrank && s.gid_start >= 0 &&
      s.gid_start <= s.total_blocks && s.blocks <= s.total_blocks-s.gid_start;
  int cap = std::numeric_limits<int>::max();
  bool has_periodic = false;
  for (int a = 0; a < 3; ++a) {
    const auto inner = pm->mesh_bcs[2*a], outer = pm->mesh_bcs[2*a+1];
    if (inner == BoundaryFlag::shear_periodic || outer == BoundaryFlag::shear_periodic) {
      AreaError("area output does not support shearing-periodic boundaries");
    }
    s.periodic[a] = inner == BoundaryFlag::periodic;
    if (s.periodic[a] != (outer == BoundaryFlag::periodic)) {
      AreaError("periodic boundaries must occur on both faces of an axis");
    }
    valid = valid && s.n[a] > 1 && s.nb[a] > 0 && s.root_blocks[a] > 0 &&
        s.n[a]%std::max(1,s.nb[a]) == 0 &&
        s.n[a]/std::max(1,s.nb[a]) == s.root_blocks[a] &&
        std::isfinite(s.lower[a]) && std::isfinite(s.upper[a]) &&
        s.upper[a] > s.lower[a] &&
        global_cells <= std::numeric_limits<std::int64_t>::max()/std::max(1,s.n[a]);
    if (!valid) AreaError("invalid uniform-mesh dimensions or coordinates");
    global_cells *= s.n[a]; root_blocks *= s.root_blocks[a];
    s.dx[a] = (s.upper[a]-s.lower[a])/s.n[a];
    if (!std::isfinite(s.dx[a]) || s.dx[a] <= 0.0) AreaError("invalid cell spacing");
    if (s.periodic[a]) { cap = std::min(cap,s.n[a]/2); has_periodic = true; }
  }
  if (!has_periodic) {
    for (int a = 0; a < 3; ++a) cap = std::min(cap,s.n[a]-1);
  }
  for (int step : s.strides) {
    if (step > cap) {
      AreaError("stride exceeds half the smallest periodic axis, or the shortest "
                "nonperiodic centre span when no axis is periodic");
    }
    const double support = static_cast<double>(step)*s.sigma_factor*s.truncate;
    if (!std::isfinite(support) || support > std::numeric_limits<int>::max()-1) {
      AreaError("Gaussian support exceeds the supported integer range");
    }
  }
  AreaRequire(root_blocks == s.total_blocks,"incomplete uniform block tiling");
  s.cells_per_block = static_cast<std::int64_t>(s.nb[0])*s.nb[1]*s.nb[2];
  s.block_gid.assign(s.total_blocks,-1);
  s.rank_eachmb.resize(s.total_blocks);
  s.origins.resize(static_cast<std::size_t>(3)*s.total_blocks);
  for (int gid = 0; gid < s.total_blocks; ++gid) {
    const auto &loc = pm->lloc_eachmb[gid];
    const int rank = pm->rank_eachmb[gid];
    valid = loc.level == s.root_level && loc.lx1 >= 0 && loc.lx1 < s.root_blocks[0] &&
        loc.lx2 >= 0 && loc.lx2 < s.root_blocks[1] && loc.lx3 >= 0 &&
        loc.lx3 < s.root_blocks[2] && rank >= 0 && rank < s.nranks;
    if (!valid) AreaError("invalid root block coordinates or owner");
    const int bi = (loc.lx3*s.root_blocks[1]+loc.lx2)*s.root_blocks[0]+loc.lx1;
    if (s.block_gid[bi] != -1) AreaError("duplicate uniform block coordinate");
    s.block_gid[bi] = gid; s.rank_eachmb[gid] = rank;
    s.origins[3*gid] = loc.lx1*s.nb[0];
    s.origins[3*gid+1] = loc.lx2*s.nb[1];
    s.origins[3*gid+2] = loc.lx3*s.nb[2];
  }
  for (int m = 0; m < s.blocks; ++m) {
    if (s.rank_eachmb[s.gid_start+m] != s.rank) {
      AreaError("noncontiguous local block owner");
    }
  }
  s.local_origin = DvceArray2D<int>("area_local_origin",s.blocks,3);
  auto origin_h = Kokkos::create_mirror_view(s.local_origin);
  for (int m = 0; m < s.blocks; ++m) for (int a = 0; a < 3; ++a) {
    origin_h(m,a) = s.origins[3*(s.gid_start+m)+a];
  }
  Kokkos::deep_copy(s.local_origin,origin_h);
  const bool smoothing = s.methods.size() > (sampled ? 1u : 0u);
  const bool fine_sampled = sampled &&
      std::find(s.strides.begin(),s.strides.end(),1) != s.strides.end();
  if (smoothing || fine_sampled) AreaBuildHalo(&s);
  if (smoothing) {
    for (int a = 0; a < 3; ++a) s.line_plan[a] = AreaBuildLines(s,a);
  }
  for (int step : s.strides) s.coarse_plan.push_back(AreaBuildCoarse(s,step));
  if (out_params.file_id.empty() || out_params.file_id.find('/') != std::string::npos ||
      out_params.file_id == "." || out_params.file_id == "..") {
    AreaError("area id must be a nonempty directory-safe name");
  }
  if (out_params.file_number < 0) AreaError("file_number must be nonnegative");
  std::ostringstream meta;
  meta << std::setprecision(std::numeric_limits<Real>::max_digits10);
  meta << "# AthenaK area format=1\n# variable=" << out_params.variable
       << " component=" << s.component << " label=" << s.label << '\n';
  meta << "# dimensions=" << s.n[0] << ',' << s.n[1] << ',' << s.n[2] << '\n';
  meta << "# lower=" << s.lower[0] << ',' << s.lower[1] << ',' << s.lower[2] << '\n';
  meta << "# upper=" << s.upper[0] << ',' << s.upper[1] << ',' << s.upper[2] << '\n';
  meta << "# periodic=" << s.periodic[0] << ',' << s.periodic[1] << ',' << s.periodic[2]
       << "\n# sample_transform=" << transform << "\n# sigma_factor=" << s.sigma_factor
       << " truncate=" << s.truncate << '\n';
  meta << "# levels=";
  for (std::size_t i = 0; i < s.levels.size(); ++i) meta << (i ? "," : "") << s.levels[i];
  meta << "\n# strides=";
  for (std::size_t i = 0; i < s.strides.size(); ++i) {
    meta << (i ? "," : "") << s.strides[i];
  }
  meta << "\n# methods=";
  for (std::size_t i = 0; i < s.methods.size(); ++i) {
    meta << (i ? "," : "") << AreaMethod(s.methods[i]);
  }
  meta << "\n# lattice=origin_zero terminal=actual_width nonperiodic=interior_centres\n"
       << "# gaussian=discrete_separable boundary=periodic_or_nearest "
          "indicator=scalar<=level\n"
       << "# cube_scale=max(global_max(abs(extracted_scalar)),abs(extracted_level)) "
          "zero_scale=1\n"
       << "# ties=positive_epsilon nonzero_level=vertex_relative_1e-12 "
          "zero_level=global_relative_1e-12\n";
  s.configuration = meta.str();
}

AreaOutput::~AreaOutput() = default;

void AreaOutput::LoadOutputData(Mesh *pm) {
  auto &s = *state_;
  AreaValidateGeometry(pm,s);
  AreaCheckDestination(out_params,s);
  if (out_params.contains_derived) ComputeDerivedVariable(out_params.variable,pm);
  const auto source = *outvars[s.component].data_ptr;
  const int component = outvars[s.component].data_index;
  const auto ind = pm->mb_indcs;
  const int nx = s.nb[0], ny = s.nb[1];
  const auto nc = s.cells_per_block;
  AreaRequire(source.extent(0) >= static_cast<std::size_t>(s.blocks) && component >= 0 &&
      source.extent(1) > static_cast<std::size_t>(component) &&
      source.extent(2) > static_cast<std::size_t>(ind.ke) &&
      source.extent(3) > static_cast<std::size_t>(ind.je) &&
      source.extent(4) > static_cast<std::size_t>(ind.ie),
      "selected scalar storage does not cover active cells");
  s.field = DvceArray4D<Real>("area_scalar",s.blocks,s.nb[2],s.nb[1],s.nb[0]);
  const auto field = s.field;
  Kokkos::parallel_for("area_load_scalar",fft::FFTPolicy(0,nc*s.blocks),
  KOKKOS_LAMBDA(const std::int64_t q) {
    const auto c = q%nc;
    const int i = c%nx, j = (c/nx)%ny, k = c/(static_cast<std::int64_t>(nx)*ny);
    field(q/nc,k,j,i) = source(q/nc,component,k+ind.ks,j+ind.js,i+ind.is);
  });
  const bool sampled = std::find(s.methods.begin(),s.methods.end(),0) != s.methods.end();
  AreaCheckField(s,field,sampled && s.log_sample);
  s.rows.clear();
  for (std::size_t ns = 0; ns < s.strides.size(); ++ns) {
    const int step = s.strides[ns];
    const Real sigma = s.sigma_factor*step;
    for (int method : s.methods) {
      std::vector<Real> areas;
      if (method == 0) {
        areas = step == 1 ? AreaFine(s,field,s.levels,s.log_sample) :
                            AreaCoarse(s,s.coarse_plan[ns],field,step);
      } else if (method == 2) {
        const auto smooth = AreaSmooth(s,field,sigma);
        AreaCheckField(s,smooth,false);
        areas = AreaFine(s,smooth,s.levels,false);
      } else {
        DvceArray4D<Real> indicator("area_indicator",s.blocks,s.nb[2],s.nb[1],s.nb[0]);
        for (Real level : s.levels) {
          Kokkos::parallel_for("area_make_indicator",fft::FFTPolicy(0,nc*s.blocks),
          KOKKOS_LAMBDA(const std::int64_t q) {
            const auto c = q%nc;
            const int i = c%nx, j = (c/nx)%ny, k = c/(static_cast<std::int64_t>(nx)*ny);
            indicator(q/nc,k,j,i) = field(q/nc,k,j,i) <= level ? 1.0 : 0.0;
          });
          const auto smooth = AreaSmooth(s,indicator,sigma);
          AreaCheckField(s,smooth,false);
          areas.push_back(AreaFine(s,smooth,{0.5},false)[0]);
        }
      }
#if MPI_PARALLEL_ENABLED
      AreaRequire(areas.size() <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "too many area levels for MPI reduction");
      MPI_Allreduce(MPI_IN_PLACE,areas.data(),static_cast<int>(areas.size()),
                     MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
      for (std::size_t level = 0; level < s.levels.size(); ++level) {
        if (!std::isfinite(areas[level]) || areas[level] < 0.0) {
          AreaError("marching-cubes area is nonfinite or negative");
        }
        s.rows.push_back({method,step,s.levels[level],
                          method == 0 ? 0.0 : sigma,areas[level]});
      }
    }
  }
  s.field = DvceArray4D<Real>();
  s.ready = true;
}

void AreaOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  auto &s = *state_;
  AreaRequire(s.ready,"area data were not loaded before writing");
  AreaCheckDestination(out_params,s);
  if (s.rank == 0) {
    std::ostringstream text;
    text << s.configuration << std::setprecision(std::numeric_limits<Real>::max_digits10)
         << "# time=" << pm->time << " cycle=" << pm->ncycle
         << " file_number=" << out_params.file_number << '\n'
         << "# columns: method level stride sigma_cells area\n";
    for (const auto &row : s.rows) {
      text << AreaMethod(row.method) << ' ' << row.level << ' ' << row.stride << ' '
           << row.sigma << ' ' << row.area << '\n';
    }
    AreaWriteExclusive(AreaFilename(out_params),text.str());
  }
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (out_params.file_number == std::numeric_limits<int>::max()) {
    AreaError("area output file counter exhausted");
  }
  ++out_params.file_number;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name,"file_number",out_params.file_number);
  pin->SetReal(out_params.block_name,"last_time",out_params.last_time);
  s.ready = false;
}
