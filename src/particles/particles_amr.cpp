//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles_amr.cpp
//! \brief Particle topology transfer, separate from physical timestep migration.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// All ranks take the same failure path before entering the next communication phase.
void RequireAll(bool local_ok, const char *message) {
  int ok = local_ok ? 1 : 0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
  if (!ok) {
    if (!local_ok) {
      std::cerr << "Particle AMR rank " << global_variable::my_rank << ": "
                << message << std::endl;
    }
    Kokkos::abort(message);
  }
}

// Same Cartesian edge convention as MeshBlock construction, including physical edges.
void Bounds(const Mesh *pm, const LogicalLocation &loc, Real lo[3], Real hi[3]) {
  const Real lower[3] = {pm->mesh_size.x1min, pm->mesh_size.x2min,
                         pm->mesh_size.x3min};
  const Real upper[3] = {pm->mesh_size.x1max, pm->mesh_size.x2max,
                         pm->mesh_size.x3max};
  const int root[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
  const int lx[3] = {loc.lx1, loc.lx2, loc.lx3};
  const int ndim = pm->three_d ? 3 : 2;
  for (int d=0; d<3; ++d) {
    lo[d] = lower[d];
    hi[d] = upper[d];
    if (d < ndim) {
      const int nb = root[d] << (loc.level - pm->root_level);
      if (lx[d] != 0) lo[d] = LeftEdgeX(lx[d], nb, lower[d], upper[d]);
      if (lx[d] != nb-1) hi[d] = LeftEdgeX(lx[d]+1, nb, lower[d], upper[d]);
    }
  }
}

uint64_t Mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// Dedicated AMR stream; unaffected by rank, array order, or the physical-pusher RNG.
Real ChildDraw(int64_t seed, int tag, int cycle, int level, const int64_t cell[3]) {
  uint64_t z = Mix(static_cast<uint64_t>(seed) ^ 0x414d525f4348494cULL);
  z = Mix(z ^ static_cast<uint64_t>(tag));
  z = Mix(z ^ static_cast<uint64_t>(cycle));
  z = Mix(z ^ static_cast<uint64_t>(level));
  for (int d=0; d<3; ++d) z = Mix(z ^ static_cast<uint64_t>(cell[d]));
  // Use the number of mantissa bits available in Real; never round the result to 1.
  constexpr int bits = std::numeric_limits<Real>::digits < 53 ?
                       std::numeric_limits<Real>::digits : 53;
  return std::ldexp(static_cast<Real>(z >> (64-bits)), -bits);
}
}  // namespace

namespace particles {

Particles::MeshRedistribution Particles::PrepareMeshRedistribution(
    const LogicalLocation *new_locations, const int *new_ranks,
    const int *old_to_new, int new_nmb) {
  Mesh *pm = pmy_pack->pmesh;
  const int ndim = pm->three_d ? 3 : 2;
  const int nleaf = 1 << ndim;
  Kokkos::fence();
  MeshRedistribution t;
  // Explicit allocations: create_mirror_view could alias the live arrays on CPU.
  t.real = HostArray2D<Real>("particle_amr_real", nrdata, nprtcl_thispack);
  t.integer = HostArray2D<int>("particle_amr_int", nidata, nprtcl_thispack);
  Kokkos::deep_copy(t.real, prtcl_rdata);
  Kokkos::deep_copy(t.integer, prtcl_idata);
  t.destination.resize(nprtcl_thispack);
  t.old_level.resize(nprtcl_thispack);
  t.global_count = pm->nprtcl_total;
  t.next_dt = dtnew;
  if (pusher == ParticlesPusher::ito_2 && ito2_enforce_locality_dt) {
    // NewTimeStep has already measured the old-cycle saved-flux drift. Preserve
    // its global minimum and tighten by the largest decrease in mesh spacing.
    // Do not reinterpret old saved flux arrays with new block indices. This is
    // a lagged accuracy predictor (as is the ordinary next-step flux estimate),
    // not a guarantee for arbitrary changes in the next step's numerical flux.
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &t.next_dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
    int max_increase = 0;
    for (int g=0; g<pm->nmb_total; ++g) {
      max_increase = std::max(max_increase,
          new_locations[old_to_new[g]].level-pm->lloc_eachmb[g].level);
    }
    t.next_dt = std::ldexp(t.next_dt,-max_increase);
  }
  bool valid = true;
  for (int p=0; p<nprtcl_thispack; ++p) {
    const int oldgid = t.integer(PGID,p);
    if (oldgid < pmy_pack->gids || oldgid > pmy_pack->gide) {
      valid = false;
      continue;
    }
    const auto &oldloc = pm->lloc_eachmb[oldgid];
    t.old_level[p] = oldloc.level;
    Real lo[3], hi[3], pos[3];
    Bounds(pm, oldloc, lo, hi);
    const bool active = particle_type != ParticleType::lagrangian_tracer ||
                        t.integer(PLASTMOVE,p) >= 0;
    for (int d=0; d<3; ++d) {
      pos[d] = t.real(IPX+d,p);
      if (!std::isfinite(pos[d])) valid = false;
      if (d >= ndim) continue;
      if (active) {
        if (!(pos[d] >= lo[d] && pos[d] < hi[d])) valid = false;
      } else {
        // Only an ownership key: retained exit coordinates remain untouched.
        pos[d] = std::clamp(pos[d], lo[d], std::nextafter(hi[d], lo[d]));
      }
    }
    int first = old_to_new[oldgid];
    if (first < 0 || first >= new_nmb) {
      valid = false;
      continue;
    }
    int delta = new_locations[first].level - oldloc.level;
    if (delta < -1 || delta > 1) {
      valid = false;
      continue;
    }
    int owner = -1;
    // A single event replaces an old leaf with one ancestor, the same leaf, or
    // its immediate children. Bound the search by 4/8, not the global MB count.
    const int candidates = delta > 0 ? nleaf : 1;
    for (int g=first; g<std::min(first+candidates, new_nmb); ++g) {
      Bounds(pm, new_locations[g], lo, hi);
      bool inside = true;
      for (int d=0; d<ndim; ++d) inside &= pos[d] >= lo[d] && pos[d] < hi[d];
      if (inside) {
        if (owner != -1) valid = false;
        owner = g;
      }
    }
    if (owner < 0 || new_ranks[owner] < 0 ||
        new_ranks[owner] >= global_variable::nranks) {
      valid = false;
      continue;
    }
    t.integer(PGID,p) = owner;
    t.destination[p] = new_ranks[owner];
  }
  RequireAll(valid, "Particle AMR invalid old ownership or new leaf mapping");
  return t;
}

void Particles::FinishMeshRedistribution(const MeshRedistribution &t) {
  Mesh *pm = pmy_pack->pmesh;
  const int ranks = global_variable::nranks;
  const int ni = nidata + 1;  // old owner level travels with the full persistent record
  const int limit = std::numeric_limits<int>::max();
  std::vector<int> sc(ranks,0), rc(ranks,0), sd(ranks,0), rd(ranks,0);
  for (int rank : t.destination) ++sc[rank];
#if MPI_PARALLEL_ENABLED
  // Fluid AMR communication has finished; isolate the particle transfer nevertheless.
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  MPI_Alltoall(sc.data(), 1, MPI_INT, rc.data(), 1, MPI_INT, comm);
#else
  rc = sc;
#endif
  int64_t ns=0, nr=0;
  for (int r=0; r<ranks; ++r) {
    // Do not cast overflowing displacements before the collective bounds check.
    sd[r] = ns <= limit ? static_cast<int>(ns) : 0;
    rd[r] = nr <= limit ? static_cast<int>(nr) : 0;
    ns += sc[r];
    nr += rc[r];
  }
  RequireAll(ns <= limit/std::max(nrdata,ni) && nr <= limit/std::max(nrdata,ni),
             "Particle AMR record count exceeds supported MPI count range");
  const int count = static_cast<int>(nr);
  std::vector<Real> sr(std::max<int64_t>(1,ns*nrdata)), rr(std::max<int64_t>(1,nr*nrdata));
  std::vector<int> si(std::max<int64_t>(1,ns*ni)), ri(std::max<int64_t>(1,nr*ni));
  auto next = sd;
  for (int p=0; p<static_cast<int>(t.destination.size()); ++p) {
    int slot = next[t.destination[p]]++;
    for (int v=0; v<nrdata; ++v) sr[slot*nrdata+v] = t.real(v,p);
    for (int v=0; v<nidata; ++v) si[slot*ni+v] = t.integer(v,p);
    si[slot*ni+nidata] = t.old_level[p];
  }
#if MPI_PARALLEL_ENABLED
  std::vector<int> scv(ranks), rcv(ranks), sdv(ranks), rdv(ranks);
  for (int r=0; r<ranks; ++r) {
    scv[r] = sc[r]*nrdata; rcv[r] = rc[r]*nrdata;
    sdv[r] = sd[r]*nrdata; rdv[r] = rd[r]*nrdata;
  }
  MPI_Alltoallv(sr.data(), scv.data(), sdv.data(), MPI_ATHENA_REAL,
                rr.data(), rcv.data(), rdv.data(), MPI_ATHENA_REAL, comm);
  for (int r=0; r<ranks; ++r) {
    scv[r] = sc[r]*ni; rcv[r] = rc[r]*ni;
    sdv[r] = sd[r]*ni; rdv[r] = rd[r]*ni;
  }
  MPI_Alltoallv(si.data(), scv.data(), sdv.data(), MPI_INT,
                ri.data(), rcv.data(), rdv.data(), MPI_INT, comm);
  MPI_Comm_free(&comm);
#else
  rr = sr;
  ri = si;
#endif
  HostArray2D<Real> real("particle_amr_new_real",nrdata,count);
  HostArray2D<int> integer("particle_amr_new_int",nidata,count);
  for (int p=0; p<count; ++p) {
    for (int v=0; v<nrdata; ++v) real(v,p) = rr[p*nrdata+v];
    for (int v=0; v<nidata; ++v) integer(v,p) = ri[p*ni+v];
  }

  const int ndim = pm->three_d ? 3 : 2;
  const int nx[3] = {pm->mb_indcs.nx1, pm->mb_indcs.nx2, pm->mb_indcs.nx3};
  const bool mc = pusher == ParticlesPusher::lagrangian_mc;
  bool valid = true;
  // Stage only density, not all fluid variables. Empty for continuous pushers.
  HostArray4D<Real> density;
  if (mc) {
    RequireAll(pmy_pack->phydro != nullptr || pmy_pack->pmhd != nullptr,
               "Particle AMR MC requires conservative fluid density");
    auto &u = pmy_pack->phydro != nullptr ? pmy_pack->phydro->u0 : pmy_pack->pmhd->u0;
    density = HostArray4D<Real>("particle_amr_density",u.extent(0),u.extent(2),
                                u.extent(3),u.extent(4));
    // Selecting IDN leaves a strided subview across MeshBlocks. Pack in device
    // memory first; a direct strided CUDA-to-host copy has no valid copy path.
    DvceArray4D<Real> packed_density("particle_amr_density_packed",u.extent(0),
                                    u.extent(2),u.extent(3),u.extent(4));
    Kokkos::deep_copy(packed_density,
        Kokkos::subview(u,Kokkos::ALL(),static_cast<int>(IDN),Kokkos::ALL(),
                       Kokkos::ALL(),Kokkos::ALL()));
    Kokkos::deep_copy(density, packed_density);
  }
  for (int p=0; p<count; ++p) {
    const int gid = integer(PGID,p);
    if (gid < pmy_pack->gids || gid > pmy_pack->gide ||
        pm->rank_eachmb[gid] != global_variable::my_rank) {
      valid = false;
      continue;
    }
    const auto &loc = pm->lloc_eachmb[gid];
    // Inactive positions and negative status sentinels survive unchanged.
    if (particle_type == ParticleType::lagrangian_tracer && integer(PLASTMOVE,p) < 0) {
      continue;
    }
    Real lo[3], hi[3];
    Bounds(pm,loc,lo,hi);
    const int oldlevel = ri[p*ni+nidata];
    if (mc && loc.level != oldlevel) {
      int cell[3] = {0,0,0};
      bool cell_ok = true;
      for (int d=0; d<ndim; ++d) {
        Real q = (real(IPX+d,p)-lo[d])/(hi[d]-lo[d])*nx[d];
        if (!(q >= 0 && q < nx[d])) {cell_ok = false; continue;}
        cell[d] = static_cast<int>(q);
        if (loc.level > oldlevel) {
          // MB dimensions are even under AMR: all fine children of an old cell
          // are in this new leaf. Check this before accessing child densities.
          cell[d] = (cell[d]/2)*2;
          cell_ok &= nx[d]%2 == 0 && cell[d]+1 < nx[d] &&
                     std::abs(q-(cell[d]+1)) < 1.0e-8;
        }
      }
      if (!cell_ok || std::abs(loc.level-oldlevel) != 1) {
        valid = false;
        continue;
      }
      if (loc.level > oldlevel) {
        Real weight[8] = {}, total=0;
        const int nchild = 1 << ndim;
        const int m = gid-pmy_pack->gids;
        for (int c=0; c<nchild; ++c) {
          int i = cell[0]+(c&1)+pm->mb_indcs.is;
          int j = cell[1]+((c>>1)&1)+pm->mb_indcs.js;
          int k = cell[2]+((c>>2)&1)+pm->mb_indcs.ks;
          // Child volumes are equal within this Cartesian parent cell, so the
          // common volume cancels from the mass fractions.
          weight[c] = density(m,k,j,i);
          cell_ok &= std::isfinite(weight[c]) && weight[c] >= 0;
          total += weight[c];
        }
        if (!cell_ok || !std::isfinite(total) || total <= 0) {
          valid = false;
          continue;
        }
        const int64_t parent[3] = {
          (static_cast<int64_t>(loc.lx1)*nx[0]+cell[0])/2,
          (static_cast<int64_t>(loc.lx2)*nx[1]+cell[1])/2,
          pm->three_d ? (static_cast<int64_t>(loc.lx3)*nx[2]+cell[2])/2 : 0};
        Real target = ChildDraw(random_seed,integer(PTAG,p),pm->ncycle,oldlevel,parent)*total;
        int chosen = -1;
        Real cumulative = 0;
        for (int c=0; c<nchild; ++c) {
          if (weight[c] == 0) continue;
          chosen = c;
          cumulative += weight[c];
          if (target < cumulative) break;
        }
        for (int d=0; d<ndim; ++d) cell[d] += (chosen>>d)&1;
      }
      for (int d=0; d<ndim; ++d) {
        real(IPX+d,p) = CellCenterX(cell[d],nx[d],lo[d],hi[d]);
      }
    }
    if (particle_type == ParticleType::lagrangian_tracer) {
      integer(PLASTLEVEL,p) = loc.level;
    }
    if (mc) integer(PLASTMOVE,p) = 0;  // topology transfer is not a previous flux crossing
    for (int d=0; d<ndim; ++d) {
      valid &= real(IPX+d,p) >= lo[d] && real(IPX+d,p) < hi[d];
    }
  }
  RequireAll(valid, "Particle AMR invalid received owner, MC cell or child mass");
  std::vector<int> counts(ranks);
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD);
#else
  counts[0] = count;
#endif
  int64_t total = 0;
  for (int n : counts) total += n;
  RequireAll(total == t.global_count, "Particle AMR changed global particle count");
  ReallocateParticles(count);
  Kokkos::deep_copy(prtcl_rdata,real);
  Kokkos::deep_copy(prtcl_idata,integer);
  pm->nprtcl_thisrank = count;
  pm->nprtcl_total = static_cast<int>(total);
  dtnew = t.next_dt;
  for (int r=0; r<ranks; ++r) pm->nprtcl_eachrank[r] = counts[r];
  Kokkos::fence();
}
}  // namespace particles
