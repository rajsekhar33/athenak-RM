#ifndef OUTPUTS_CHECKPOINT_METADATA_HPP_
#define OUTPUTS_CHECKPOINT_METADATA_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file checkpoint_metadata.hpp
//! \brief Shared, versioned checkpoint phase and topology identity.

#include <cstdint>
#include <cstring>

#include "mesh/mesh.hpp"

namespace checkpoint {
constexpr uint64_t fluid_magic = 0x414b43504c494645ULL;
constexpr uint64_t version = 1;
constexpr int64_t particle_magic = 44;
constexpr std::size_t particle_header_bytes = 6*sizeof(uint64_t) + 2*sizeof(double);

// Hash explicit values, not structure padding. This identifies the global mesh,
// independent of MPI decomposition, to reject mismatched fluid/particle files.
inline uint64_t TopologyHash(const Mesh *pm) {
  uint64_t hash = 14695981039346656037ULL;
  auto add = [&hash](uint64_t value) {
    for (int b=0; b<8; ++b) {
      hash ^= (value >> (8*b)) & 255;
      hash *= 1099511628211ULL;
    }
  };
  add(pm->root_level);
  add(pm->nmb_total);
  add(pm->mb_indcs.nx1); add(pm->mb_indcs.nx2); add(pm->mb_indcs.nx3);
  const double bounds[] = {pm->mesh_size.x1min, pm->mesh_size.x1max,
                            pm->mesh_size.x2min, pm->mesh_size.x2max,
                            pm->mesh_size.x3min, pm->mesh_size.x3max};
  for (double x : bounds) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    add(bits);
  }
  for (int g=0; g<pm->nmb_total; ++g) {
    const auto &loc = pm->lloc_eachmb[g];
    add(loc.lx1); add(loc.lx2); add(loc.lx3); add(loc.level);
  }
  return hash;
}
}  // namespace checkpoint
#endif  // OUTPUTS_CHECKPOINT_METADATA_HPP_
