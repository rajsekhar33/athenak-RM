//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sf_unit_test.cpp
//! \brief Problem generator for validating the on-the-fly structure-function output
//! (file_type = sf). Sets a deterministic isothermal hydro state: smooth modes plus
//! hash noise of the global cell index, so tests/sf/sf_reference.py can rebuild exactly
//! the same fields and compute exhaustive all-pairs references. Run with nlim = 0; the
//! initial outputs are the test data.
//! With <problem>/fft_test = true it also runs the distributed FFT (fft/pencil_fft.hpp)
//! on the density over the index range fft_lo1..3, fft_hi1..3 (default: whole mesh) and
//! writes every coefficient to fft_test.bin for tests/fft/compare_pencil.py.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "fft/pencil_fft.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"

namespace {
//! splitmix64 finaliser; tests/sf/sf_reference.py implements the same function
KOKKOS_INLINE_FUNCTION
uint64_t TestMix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

//! uniform noise in [-0.5, 0.5) keyed by global cell index and component
KOKKOS_INLINE_FUNCTION
Real TestNoise(int64_t gi, int64_t gj, int64_t gk, int64_t comp) {
  uint64_t h = TestMix(static_cast<uint64_t>(comp));
  h = TestMix(h ^ static_cast<uint64_t>(gi));
  h = TestMix(h ^ static_cast<uint64_t>(gj));
  h = TestMix(h ^ static_cast<uint64_t>(gk));
  return static_cast<Real>(static_cast<double>(h >> 11)*(1.0/9007199254740992.0) - 0.5);
}
//! phase-indicator mask (0 or 1) at unit cell centre (cx, cy, cz), for sparse-weight
//! tests: 1 = two disconnected spheres, one across the periodic x and z boundaries;
//! 2 = one small sphere centred on a MeshBlock corner (rare selection); 3 = empty;
//! 4 = one small sphere across the x faces of the first two MeshBlocks of a 32^3 mesh
//! with 8^3 blocks (support on a single rank).
//! tests/sf/sf_reference.py builds the same mask.
KOKKOS_INLINE_FUNCTION
Real TestMask(int type, Real cx, Real cy, Real cz) {
  // sphere centres and radii (x, y, z, r), one sphere per row
  const Real c[4][4] = {0.30, 0.30, 0.30, 0.16,
                        0.95, 0.55, 0.02, 0.12,
                        0.50, 0.50, 0.50, 0.07,
                        0.25, 0.125, 0.125, 0.07};
  int s0 = 0, s1 = 0;
  if (type == 1) {s0 = 0; s1 = 2;}
  if (type == 2) {s0 = 2; s1 = 3;}
  if (type == 4) {s0 = 3; s1 = 4;}
  for (int s=s0; s<s1; ++s) {
    Real dx = cx - c[s][0], dy = cy - c[s][1], dz = cz - c[s][2];
    dx -= Kokkos::floor(dx + 0.5);
    dy -= Kokkos::floor(dy + 0.5);
    dz -= Kokkos::floor(dz + 0.5);
    if (dx*dx + dy*dy + dz*dz < c[s][3]*c[s][3]) return 1.0;
  }
  return 0.0;
}
//! distributed FFT of the density over the selected index range; rank 0 writes
//! fft_test.bin: int32 nx, ny, nz, lo1, lo2, lo3, then per coefficient
//! (gx, gy, gz, re, im) as float64, in any order
void RunFFTTest(Mesh *pm, ParameterInput *pin) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  int lo[3], hi[3];
  const int nmesh[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  for (int d = 0; d < 3; ++d) {
    lo[d] = pin->GetOrAddInteger("problem", "fft_lo" + std::to_string(d+1), 0);
    hi[d] = pin->GetOrAddInteger("problem", "fft_hi" + std::to_string(d+1), nmesh[d]);
  }
  const bool host = pin->GetOrAddBoolean("problem", "fft_host_stage", false);
  const int nrep = pin->GetOrAddInteger("problem", "fft_nrep", 2);
  // largest MPI message in doubles (0 = default); small values exercise the chunking
  const int max_msg = pin->GetOrAddInteger("problem", "fft_max_msg", 0);

  DvceArray4D<double> in("fft_in", nmb, nx3, nx2, nx1);
  auto &u0 = (pmbp->phydro != nullptr) ? pmbp->phydro->u0 : pmbp->pmhd->u0;
  par_for("fft_in", DevExeSpace(), 0, nmb-1, 0, nx3-1, 0, nx2-1, 0, nx1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    in(m, k, j, i) = u0(m, IDN, k + ks, j + js, i + is);
  });
  fft::PencilFFT pf(pm, lo, hi, host);
  if (max_msg > 0) pf.SetMaxMessage(max_msg);
  fft::CplxArray1D out;
  for (int r = 0; r < nrep; ++r) {
    out = pf.Forward(in);
    if (global_variable::my_rank == 0) {
      const double *t = pf.Timings();
      std::printf("pencil_fft %dx%d host_stage=%d rep %d: R0 %.3e x %.3e R1 %.3e y %.3e "
                  "R2 %.3e z %.3e s\n", pf.Py(), pf.Pz(), host ? 1 : 0, r, t[0], t[1],
                  t[2], t[3], t[4], t[5]);
    }
  }
  // most chunks of one remote message on any rank (the tests check the tag wrap at 100)
  int64_t chunks = static_cast<int64_t>(pf.MaxChunks());
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &chunks, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::printf("pencil_fft_info nranks=%d pencils=%dx%d "
                "max_chunks_per_message=%" PRId64 "\n",
                global_variable::nranks, pf.Py(), pf.Pz(), chunks);
  }

  const fft::SpectralSlab slab = pf.Slab();
  const std::int64_t nloc = pf.Active() ? slab.Size() : 0;
  DvceArray1D<double> rec("fft_rec", 5*std::max<std::int64_t>(nloc, 1));
  if (nloc > 0) {
    Kokkos::parallel_for("fft_rec",
    Kokkos::RangePolicy<DevExeSpace, std::int64_t>(0, nloc),
    KOKKOS_LAMBDA(const std::int64_t f) {
      int gx, gy, gz;
      slab.Global(f, gx, gy, gz);
      rec(5*f) = gx;
      rec(5*f + 1) = gy;
      rec(5*f + 2) = gz;
      rec(5*f + 3) = out(f).real();
      rec(5*f + 4) = out(f).imag();
    });
  }
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rec);
  const int nranks = global_variable::nranks;
  std::vector<int> cnt(nranks, 0), dsp(nranks, 0);
  int mine = static_cast<int>(5*nloc);
#if MPI_PARALLEL_ENABLED
  MPI_Gather(&mine, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
#else
  cnt[0] = mine;
#endif
  int tot = 0;
  for (int r = 0; r < nranks; ++r) {dsp[r] = tot; tot += cnt[r];}
  std::vector<double> all(std::max(tot, 1));
#if MPI_PARALLEL_ENABLED
  MPI_Gatherv(h.data(), mine, MPI_DOUBLE, all.data(), cnt.data(), dsp.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
#else
  for (int n = 0; n < mine; ++n) all[n] = h(n);
#endif
  if (global_variable::my_rank == 0) {
    FILE *f = std::fopen("fft_test.bin", "wb");
    const int hdr[6] = {hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2], lo[0], lo[1], lo[2]};
    std::fwrite(hdr, sizeof(int), 6, f);
    std::fwrite(all.data(), sizeof(double), tot, f);
    std::fclose(f);
  }
}

//! problem/mask_toggle > 0 (hydro only): after every step the indicator scalar is reset
//! to the mask when (ncycle / mask_toggle) is even and to 0 when it is odd, so the sf
//! support vanishes and reappears across outputs. u0 and w0 of the scalar are set; the
//! fluid itself is untouched.
int mask_type = 0, mask_toggle = 0;
void MaskToggle(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  const int nhyd = pmbp->phydro->nhydro, nscal = pmbp->phydro->nscalars;
  const bool on = ((pm->ncycle/mask_toggle) % 2) == 0;
  const int type = mask_type;
  auto &ms = pm->mesh_size;
  const Real x1m = ms.x1min, x2m = ms.x2min, x3m = ms.x3min;
  const Real dx1 = ms.dx1, dx2 = ms.dx2, dx3 = ms.dx3;
  const int gn1 = pm->mesh_indcs.nx1, gn2 = pm->mesh_indcs.nx2, gn3 = pm->mesh_indcs.nx3;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  par_for("sf_mask_toggle", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is,
          ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    int64_t gi = static_cast<int64_t>(Kokkos::floor((x - x1m)/dx1));
    int64_t gj = static_cast<int64_t>(Kokkos::floor((y - x2m)/dx2));
    int64_t gk = static_cast<int64_t>(Kokkos::floor((z - x3m)/dx3));
    const Real s = on ? TestMask(type, (gi + 0.5)/gn1, (gj + 0.5)/gn2, (gk + 0.5)/gn3)
                      : 0.0;
    for (int n=nhyd; n<nhyd+nscal; ++n) {
      u0(m,n,k,j,i) = u0(m,IDN,k,j,i)*s;
      w0(m,n,k,j,i) = s;
    }
  });
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "sf_unit_test needs a <hydro> or <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  // time-dependent support (registered on restarts too)
  mask_type = pin->GetOrAddInteger("problem", "mask", 0);
  mask_toggle = pin->GetOrAddInteger("problem", "mask_toggle", 0);
  if (mask_toggle > 0) {
    if (pmbp->phydro == nullptr || pmbp->phydro->nscalars < 1 || mask_type < 1 ||
        !user_work_in_loop) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "problem/mask_toggle needs <hydro> nscalars >= 1, "
                << "problem/mask >= 1 and problem/user_work_in_loop = true" << std::endl;
      exit(EXIT_FAILURE);
    }
    user_work_in_loop_func = MaskToggle;
  }
  if (restart) return;
  const Real noise = pin->GetOrAddReal("problem", "noise", 1.0);
  // constant = true: uniform state, so every structure-function increment is exactly 0
  const bool constant = pin->GetOrAddBoolean("problem", "constant", false);
  // wave_amp > 0: d = 1, v = (boost + wave_amp sin(2 pi x/Lx), 0, 0), a one-component
  // wave with analytic increment scales (histogram sigma tests)
  const Real wave_amp = pin->GetOrAddReal("problem", "wave_amp", 0.0);
  const Real boost = pin->GetOrAddReal("problem", "boost", 0.0);
  // mask > 0: the first passive scalar is the indicator TestMask(mask, ...), so the
  // output variable hydro_w_s (mhd_w_s) can serve as a sparse 0/1 sf weight
  const int mask = mask_type;
  const int nhyd = (pmbp->phydro != nullptr) ? pmbp->phydro->nhydro : pmbp->pmhd->nmhd;
  const int nscal = (pmbp->phydro != nullptr) ? pmbp->phydro->nscalars :
                                                pmbp->pmhd->nscalars;
  if (mask > 0 && nscal < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "problem/mask needs <hydro> or <mhd> nscalars >= 1"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  const int gn1 = pmy_mesh_->mesh_indcs.nx1, gn2 = pmy_mesh_->mesh_indcs.nx2;
  const int gn3 = pmy_mesh_->mesh_indcs.nx3;
  auto &ms = pmy_mesh_->mesh_size;
  const Real x1m = ms.x1min, x2m = ms.x2min, x3m = ms.x3min;
  const Real lx = ms.x1max - ms.x1min, ly = ms.x2max - ms.x2min;
  const Real lz = ms.x3max - ms.x3min;
  const Real dx1 = ms.dx1, dx2 = ms.dx2, dx3 = ms.dx3;
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = (pmbp->phydro != nullptr) ? pmbp->phydro->u0 : pmbp->pmhd->u0;
  const Real twopi = 2.0*M_PI;
  // eos = ideal: analytic pressure, so T = p/rho is a known field (sf selection on
  // temperature); MHD adds the magnetic energy of the face fields below
  EOS_Data &eosd = (pmbp->phydro != nullptr) ? pmbp->phydro->peos->eos_data :
                                               pmbp->pmhd->peos->eos_data;
  const bool ideal = eosd.is_ideal;
  const Real gm1 = ideal ? eosd.gamma - 1.0 : 0.0;

  par_for("sf_unit_test", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
    int64_t gi = static_cast<int64_t>(Kokkos::floor((x - x1m)/dx1));
    int64_t gj = static_cast<int64_t>(Kokkos::floor((y - x2m)/dx2));
    int64_t gk = static_cast<int64_t>(Kokkos::floor((z - x3m)/dx3));
    Real sx = (x - x1m)/lx, sy = (y - x2m)/ly, sz = (z - x3m)/lz;
    Real d = 1.0 + 0.3*sin(twopi*sx)*cos(twopi*sy) + 0.2*noise*TestNoise(gi, gj, gk, 0);
    Real vx = sin(twopi*(sx + 2.0*sy)) + 0.5*noise*TestNoise(gi, gj, gk, 1);
    Real vy = cos(twopi*(3.0*sx - sz)) + 0.5*noise*TestNoise(gi, gj, gk, 2);
    Real vz = sin(twopi*sz)*cos(twopi*sx) + 0.5*noise*TestNoise(gi, gj, gk, 3);
    if (wave_amp > 0.0) {
      d = 1.0;
      vx = boost + wave_amp*sin(twopi*sx);
      vy = 0.0;
      vz = 0.0;
    }
    if (constant) {
      d = 1.25;
      vx = 0.3;
      vy = -0.2;
      vz = 0.1;
    }
    u0(m,IDN,k,j,i) = d;
    u0(m,IM1,k,j,i) = d*vx;
    u0(m,IM2,k,j,i) = d*vy;
    u0(m,IM3,k,j,i) = d*vz;
    for (int n=nhyd; n<nhyd+nscal; ++n) {
      u0(m,n,k,j,i) = (mask > 0) ? d*TestMask(mask, (gi + 0.5)/gn1, (gj + 0.5)/gn2,
                                              (gk + 0.5)/gn3) : 0.0;
    }
    if (ideal) {
      Real p = 0.6 + 0.25*cos(twopi*(sx - sy))*sin(twopi*sz) +
               0.1*noise*TestNoise(gi, gj, gk, 4);
      u0(m,IEN,k,j,i) = p/gm1 + 0.5*d*(vx*vx + vy*vy + vz*vz);
    }
  });

  // MHD: face fields as functions of the global face index (periodic wrap, so blocks
  // agree on shared faces); bcc = face averages, computed by AthenaK at start-up.
  // tests/sf/sf_reference.py builds the same fields.
  if (pmbp->pmhd != nullptr) {
    auto &b0 = pmbp->pmhd->b0;
    // bscale multiplies every face field: -1 tests field-reversal symmetry, 0 the
    // zero-field policy of aniso_mhd
    const Real bscale = pin->GetOrAddReal("problem", "bscale", 1.0);
    const int n1 = pmy_mesh_->mesh_indcs.nx1, n2 = pmy_mesh_->mesh_indcs.nx2;
    const int n3 = pmy_mesh_->mesh_indcs.nx3;
    par_for("sf_unit_test_b", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke+1, js, je+1,
            is, ie+1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // global indices of cell (i,j,k) of this block (its lower faces have the same)
      Real x = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real y = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real z = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      int64_t gi = static_cast<int64_t>(Kokkos::floor((x - x1m)/dx1));
      int64_t gj = static_cast<int64_t>(Kokkos::floor((y - x2m)/dx2));
      int64_t gk = static_cast<int64_t>(Kokkos::floor((z - x3m)/dx3));
      const int64_t fi = gi % n1, fj = gj % n2, fk = gk % n3;   // wrapped face index
      const Real ci = (fi + 0.5)/n1, cj = (fj + 0.5)/n2, ck = (fk + 0.5)/n3;
      const Real fx = static_cast<Real>(fi)/n1, fy = static_cast<Real>(fj)/n2;
      const Real fz = static_cast<Real>(fk)/n3;
      if (j <= je && k <= ke) {
        b0.x1f(m,k,j,i) = bscale*(0.7 + 0.3*sin(twopi*(cj + ck)) +
                                   0.2*noise*TestNoise(fi, fj, fk, 11));
      }
      if (i <= ie && k <= ke) {
        b0.x2f(m,k,j,i) = bscale*(0.2*cos(twopi*(ci + 2.0*ck)) +
                                   0.2*noise*TestNoise(fi, fj, fk, 12));
      }
      if (i <= ie && j <= je) {
        b0.x3f(m,k,j,i) = bscale*(0.1 + 0.25*sin(twopi*ci)*cos(twopi*cj) +
                                   0.2*noise*TestNoise(fi, fj, fk, 13));
      }
    });
    if (ideal) {
      par_for("sf_unit_test_eb", DevExeSpace(), 0, pmbp->nmb_thispack-1, ks, ke, js, je,
              is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
        const Real by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
        const Real bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
        u0(m,IEN,k,j,i) += 0.5*(bx*bx + by*by + bz*bz);
      });
    }
  }
  if (pin->GetOrAddBoolean("problem", "fft_test", false)) RunFFTTest(pmy_mesh_, pin);
}
