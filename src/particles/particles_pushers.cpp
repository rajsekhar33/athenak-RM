//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_pushers.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "particles.hpp"
#include "globals.hpp"

namespace {
// Deterministic hashed pseudo-random draws (splitmix64-derived), used instead of a
// stateful RNG pool so that particle randomness depends only on (tag, cycle, seed,
// stream) -- never on thread scheduling or draw order -- and is therefore exactly
// reproducible across a restart.
KOKKOS_INLINE_FUNCTION
Real StatelessUniform01(const int tag, const int ncycle, const int64_t base_seed,
                        const int stream) {
  uint64_t z = static_cast<uint64_t>(base_seed);
  z += static_cast<uint64_t>(tag) * 0x9e3779b97f4a7c15ULL;
  z += static_cast<uint64_t>(ncycle) * 0xbf58476d1ce4e5b9ULL;
  z += static_cast<uint64_t>(stream) * 0x94d049bb133111ebULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return static_cast<Real>(z >> 11) *
         static_cast<Real>(1.0 / 9007199254740992.0);
}

KOKKOS_INLINE_FUNCTION
Real LagrangianMCUniform01(const int tag, const int ncycle, const int64_t base_seed,
                           const int sub = 0) {
  int64_t det_seed = tag * 7919 + ncycle * 104729 + base_seed + sub * 1299709;
  uint64_t z = static_cast<uint64_t>(det_seed);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return static_cast<Real>(z & 0x7FFFFFFFULL) / static_cast<Real>(0x80000000ULL);
}

KOKKOS_INLINE_FUNCTION
Real Ito2DisplacementFromMoments(const Real cminus, const Real variance, const Real dx,
                                 const Real xi) {
  Real var = variance < 0.0 ? 0.0 : variance;
  return dx*(cminus + sqrt(var)*xi);
}

KOKKOS_INLINE_FUNCTION
Real Ito2Displacement(const Real pleft, const Real pright, const Real dx,
                      const Real xi) {
  Real cplus = pleft + pright;
  Real cminus = pright - pleft;
  Real variance = cplus - cminus*cminus;
  return Ito2DisplacementFromMoments(cminus, variance, dx, xi);
}
} // namespace

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn  void Particles::ParticlesPush
//  \brief wrapper with switch to access different particle pushers

TaskStatus Particles::Push(Driver *pdriver, int stage) {
  switch (pusher) {
    case ParticlesPusher::drift:
      PushDrift();
      break;

    case ParticlesPusher::lagrangian_mc:
      PushLagrangianMC();
      break;

    case ParticlesPusher::lagrangian_tracer:
      PushLagrangianTracer();
      break;

    case ParticlesPusher::ito_2:
      PushIto2();
      break;

    default:
      break;
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::PushDrift
//! \brief push particles based on stored particle internal velocity

void Particles::PushDrift() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto dt_ = pmy_pack->pmesh->dt;
  auto gids = pmy_pack->gids;
  int nmb = pmy_pack->nmb_thispack;

  par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;

    // Bounds check: ensure m is valid for this rank
    if (m < 0 || m >= nmb) {
      return;
    }

    int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
    pr(IPX,p) += 0.5*dt_*pr(IPVX,p);

    if (multi_d) {
      int jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
      pr(IPY,p) += 0.5*dt_*pr(IPVY,p);
    }

    if (three_d) {
      int kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
      pr(IPZ,p) += 0.5*dt_*pr(IPVZ,p);
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::PushLagrangianTracer
//! \brief classical velocity-field Lagrangian tracer with cloud-in-cell interpolation.

void Particles::PushLagrangianTracer() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &gids = pmy_pack->gids;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &w0_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->w0:pmy_pack->pmhd->w0;
  auto dt_ = pmy_pack->pmesh->dt;
  int nmb = pmy_pack->nmb_thispack;

  par_for("part_lagrangian_tracer_update",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    if (pi(PLASTMOVE,p) < 0) {
      return;
    }

    int m = pi(PGID,p) - gids;
    if (m < 0 || m >= nmb) {
      pi(PLASTMOVE,p) = -1;
      return;
    }

    int ie = is + indcs.nx1 - 1;
    int je = js + indcs.nx2 - 1;
    int ke = ks + indcs.nx3 - 1;

    // cloud-in-cell weights: locate the particle relative to the surrounding cell
    // centers (not cell edges), clamped to the active zone so a particle near a
    // boundary doesn't read ghost data from the wrong side
    Real xidx = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 +
                static_cast<Real>(is) - 0.5;
    int i0 = static_cast<int>(xidx);
    Real wx = xidx - static_cast<Real>(i0);
    if (i0 < is - 1) {
      i0 = is - 1;
      wx = 0.0;
    } else if (i0 > ie) {
      i0 = ie;
      wx = 0.0;
    }

    int j0 = js;
    Real wy = 0.0;
    if (multi_d) {
      Real yidx = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 +
                  static_cast<Real>(js) - 0.5;
      j0 = static_cast<int>(yidx);
      wy = yidx - static_cast<Real>(j0);
      if (j0 < js - 1) {
        j0 = js - 1;
        wy = 0.0;
      } else if (j0 > je) {
        j0 = je;
        wy = 0.0;
      }
    }

    int k0 = ks;
    Real wz = 0.0;
    if (three_d) {
      Real zidx = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 +
                  static_cast<Real>(ks) - 0.5;
      k0 = static_cast<int>(zidx);
      wz = zidx - static_cast<Real>(k0);
      if (k0 < ks - 1) {
        k0 = ks - 1;
        wz = 0.0;
      } else if (k0 > ke) {
        k0 = ke;
        wz = 0.0;
      }
    }

    Real vx = 0.0;
    Real vy = 0.0;
    Real vz = 0.0;
    int nj = multi_d ? 2 : 1;
    int nk = three_d ? 2 : 1;
    for (int kk=0; kk<nk; ++kk) {
      Real wk = (kk == 0) ? (1.0 - wz) : wz;
      int k = k0 + kk;
      for (int jj=0; jj<nj; ++jj) {
        Real wj = (jj == 0) ? (1.0 - wy) : wy;
        int j = j0 + jj;
        for (int ii=0; ii<2; ++ii) {
          Real wi = (ii == 0) ? (1.0 - wx) : wx;
          int i = i0 + ii;
          Real wght = wi*wj*wk;
          vx += wght*w0_(m,IVX,k,j,i);
          vy += wght*w0_(m,IVY,k,j,i);
          vz += wght*w0_(m,IVZ,k,j,i);
        }
      }
    }

    pr(IPX,p) += dt_*vx;
    if (multi_d) {
      pr(IPY,p) += dt_*vy;
    }
    if (three_d) {
      pr(IPZ,p) += dt_*vz;
    }
    pi(PLASTLEVEL,p) = mblev.d_view(m);
    pi(PLASTMOVE,p) = 0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::PushLagrangianMC
//! \brief push with Lagrangian Monte Carlo method (Genel+ 2013, MNRAS.435.1426G)
//         WARNING: this implementation may not work well with AMR

void Particles::PushLagrangianMC() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &gids = pmy_pack->gids;
  auto &mblev = pmy_pack->pmb->mb_lev;

  auto &u1_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->u1:pmy_pack->pmhd->u1;
  auto &uflxidn_ = (pmy_pack->phydro != nullptr)?
                    pmy_pack->phydro->uflxidnsaved:pmy_pack->pmhd->uflxidnsaved;
  auto &flx1_ = uflxidn_.x1f;
  auto &flx2_ = uflxidn_.x2f;
  auto &flx3_ = uflxidn_.x3f;

  int ncycle = pmy_pack->pmesh->ncycle;
  int64_t rseed = random_seed;  // capture to avoid implicit 'this' capture in the lambda

  int nmb = pmy_pack->nmb_thispack;

  par_for("part_update",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    if (pi(PLASTMOVE,p) >= 0) {
      // only update particles that are not frozen or marked for deletion

      int m = pi(PGID,p) - gids;

      // Bounds check: ensure m is valid for this rank
      if (m < 0 || m >= nmb) {
        pi(PLASTMOVE,p) = -1;  // Mark as frozen
        return;
      }

      int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
      int jp = js;
      int kp = ks;

      if (multi_d) {
        jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
      }

      if (three_d) {
        kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
      }

      // Minimal bounds check to prevent out-of-bounds flux array access
      // The particle boundary conditions should handle these cases, but add safety here
      int ie = is + indcs.nx1;
      int je = js + indcs.nx2;
      int ke = ks + indcs.nx3;

      // Check if flux access at ip+1, jp+1, kp+1 would be in bounds
      // Flux arrays extend one element beyond the active zone
      if (ip < is || ip >= ie || jp < js || jp >= je || kp < ks || kp >= ke) {
        // Particle is outside active zone - skip this timestep
        // Boundary conditions should handle freezing/deletion
        return;
      }

      // get normalized fluxes based on local density
      Real mass = u1_(m,IDN,kp,jp,ip);
      if (mass <= 0.0) {
        // near-vacuum or unphysical donor cell -- skip this timestep rather than
        // divide by a non-positive mass (matches the equivalent check in PushIto2)
        return;
      }

      // Raw (unnormalized) outflow mass flux through each face, clamped to
      // exclude inflow (negative) and sub-epsilon residuals. By convention
      // these are positive when there is outflow with respect to the
      // particle's current cell.
      const Real flx_eps = 1.0e-12 * mass;
      Real raw1L = fmax(-flx1_(m,kp,jp,ip), 0.0);
      Real raw1R = fmax(flx1_(m,kp,jp,ip+1), 0.0);
      Real raw2L = (multi_d) ? fmax(-flx2_(m,kp,jp,ip), 0.0) : 0.0;
      Real raw2R = (multi_d) ? fmax(flx2_(m,kp,jp+1,ip), 0.0) : 0.0;
      Real raw3L = (three_d) ? fmax(-flx3_(m,kp,jp,ip), 0.0) : 0.0;
      Real raw3R = (three_d) ? fmax(flx3_(m,kp+1,jp,ip), 0.0) : 0.0;
      raw1L = raw1L < flx_eps ? 0.0 : raw1L;
      raw1R = raw1R < flx_eps ? 0.0 : raw1R;
      raw2L = raw2L < flx_eps ? 0.0 : raw2L;
      raw2R = raw2R < flx_eps ? 0.0 : raw2R;
      raw3L = raw3L < flx_eps ? 0.0 : raw3L;
      raw3R = raw3R < flx_eps ? 0.0 : raw3R;

      // save refinement level of current zone
      pi(PLASTLEVEL,p) = mblev.d_view(m);

      // save parity of current zone stored as (i_isodd,j_isodd,k_isodd) * 8
      pi(PLASTMOVE,p) = 32 * (ip % 2) + 16 * (jp % 2) + 8 * (kp % 2);

      // Genel+2013 Sec 2.2 "Monte Carlo tracers": each face is checked in
      // sequence against a reduced mass mtilde that starts at the cell's mass
      // and has each already-checked face's outflow subtracted before the
      // NEXT face's probability is computed. p_j = raw_j / mtilde is then a
      // conditional probability (escape via face j, given the particle didn't
      // already escape via an earlier face), which is naturally bounded to
      // <=1 as long as the cumulative outflow checked so far never exceeds the
      // cell's own mass -- unlike computing all six probabilities independently
      // from the SAME initial mass and summing them (this file's original
      // approach, and also the earlier sub-cycling patch built on top of it),
      // which has no such guarantee and can see a combined probability above 1
      // whenever a near-vacuum donor cell sits next to a fast, dense neighbor.
      // If mtilde is driven to exactly zero by earlier faces (cumulative
      // outflow checked so far already accounts for the cell's entire mass),
      // any remaining face's escape probability saturates at 1, mirroring
      // Genel+2013's explicit handling of forcing the last neighbor's
      // probability to 1 to dispose of any residual mass/round-off.
      Real raws[6] = {raw1L, raw1R, raw2L, raw2R, raw3L, raw3R};
      bool axis_active[6] = {true, true, multi_d, multi_d, three_d, three_d};
      Real mtilde = mass;
      int move_dir = 0;
      for (int f = 0; f < 6; ++f) {
        if (!axis_active[f] || raws[f] <= 0.0) {
          continue;
        }
        Real rand_f = LagrangianMCUniform01(pi(PTAG,p), ncycle, rseed, f);
        Real p_f = (mtilde > 0.0) ? fmin(raws[f] / mtilde, 1.0) : 1.0;
        if (rand_f < p_f) {
          move_dir = f + 1;
          break;
        }
        mtilde = fmax(mtilde - raws[f], 0.0);
      }

      if (move_dir == 1) {
        pr(IPX,p) -= mbsize.d_view(m).dx1;
        pi(PLASTMOVE,p) += 1;
      } else if (move_dir == 2) {
        pr(IPX,p) += mbsize.d_view(m).dx1;
        pi(PLASTMOVE,p) += 2;
      } else if (move_dir == 3) {
        pr(IPY,p) -= mbsize.d_view(m).dx2;
        pi(PLASTMOVE,p) += 3;
      } else if (move_dir == 4) {
        pr(IPY,p) += mbsize.d_view(m).dx2;
        pi(PLASTMOVE,p) += 4;
      } else if (move_dir == 5) {
        pr(IPZ,p) -= mbsize.d_view(m).dx3;
        pi(PLASTMOVE,p) += 5;
      } else if (move_dir == 6) {
        pr(IPZ,p) += mbsize.d_view(m).dx3;
        pi(PLASTMOVE,p) += 6;
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Particles::PushIto2
//! \brief Continuous Ito-2 tracer using moments of the MC transition kernel.

void Particles::PushIto2() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &gids = pmy_pack->gids;
  auto &mblev = pmy_pack->pmb->mb_lev;

  auto &u1_ = (pmy_pack->phydro != nullptr)?pmy_pack->phydro->u1:pmy_pack->pmhd->u1;
  auto &uflxidn_ = (pmy_pack->phydro != nullptr)?
                    pmy_pack->phydro->uflxidnsaved:pmy_pack->pmhd->uflxidnsaved;
  auto &flx1_ = uflxidn_.x1f;
  auto &flx2_ = uflxidn_.x2f;
  auto &flx3_ = uflxidn_.x3f;

  int ncycle = pmy_pack->pmesh->ncycle;
  int64_t rseed = random_seed;
  int nmb = pmy_pack->nmb_thispack;

  par_for("part_ito2_update",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    if (pi(PLASTMOVE,p) < 0) {
      return;
    }

    int m = pi(PGID,p) - gids;
    if (m < 0 || m >= nmb) {
      pi(PLASTMOVE,p) = -1;
      return;
    }

    int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
    int jp = js;
    int kp = ks;

    if (multi_d) {
      jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
    }

    if (three_d) {
      kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
    }

    int ie = is + indcs.nx1;
    int je = js + indcs.nx2;
    int ke = ks + indcs.nx3;
    if (ip < is || ip >= ie || jp < js || jp >= je || kp < ks || kp >= ke) {
      return;
    }

    Real mass = u1_(m,IDN,kp,jp,ip);
    if (mass <= 0.0) {
      return;
    }

    Real p1_left = -flx1_(m,kp,jp,ip) / mass;
    Real p1_right = flx1_(m,kp,jp,ip+1) / mass;
    Real p2_left = (multi_d) ? -flx2_(m,kp,jp,ip) / mass : 0.0;
    Real p2_right = (multi_d) ? flx2_(m,kp,jp+1,ip) / mass : 0.0;
    Real p3_left = (three_d) ? -flx3_(m,kp,jp,ip) / mass : 0.0;
    Real p3_right = (three_d) ? flx3_(m,kp+1,jp,ip) / mass : 0.0;

    p1_left = p1_left < 0.0 ? 0.0 : p1_left;
    p1_right = p1_right < 0.0 ? 0.0 : p1_right;
    p2_left = p2_left < 0.0 ? 0.0 : p2_left;
    p2_right = p2_right < 0.0 ? 0.0 : p2_right;
    p3_left = p3_left < 0.0 ? 0.0 : p3_left;
    p3_right = p3_right < 0.0 ? 0.0 : p3_right;

    // Tracer-specific CFL, mirroring PushLagrangianMC: the hydro CFL condition
    // only bounds the net six-face flux divergence, not any single face's
    // flux/mass ratio in isolation, so a near-vacuum donor cell can leave
    // p_left/p_right well above the [0,1] range Ito2Displacement's variance
    // term assumes, producing an unbounded single-step displacement that can
    // push a particle's position to a value whose downstream cell-index
    // computation (see cell_locations.hpp CellCenterIndex) is undefined
    // behavior.
    //
    // Sub-cycle by splitting the TOTAL first/second moments (cminus, variance)
    // -- computed once from the original, unscaled probabilities -- into nsub
    // equal shares (cminus/nsub, variance/nsub), rather than rescaling
    // p_left/p_right and recomputing moments from the rescaled pair each
    // sub-step. The latter (tried first, see conversation) exactly preserves
    // the summed mean but systematically inflates the summed variance to
    // dx^2*(cplus - cminus^2/nsub) instead of the correct dx^2*(cplus -
    // cminus^2) once nsub>1 -- since Var[xi]=1 is exact by construction,
    // splitting the moments directly instead makes both
    // E[sum of nsub draws] = dx*cminus and Var[sum] = dx^2*variance hold
    // exactly, for any nsub, matching the un-split single-step formula's
    // moments regardless of how many sub-steps the crash-avoidance splitting
    // requires.
    Real cminus1 = p1_right - p1_left;
    Real cminus2 = p2_right - p2_left;
    Real cminus3 = p3_right - p3_left;
    Real cplus1 = p1_left + p1_right;
    Real cplus2 = p2_left + p2_right;
    Real cplus3 = p3_left + p3_right;
    Real variance1 = fmax(cplus1 - cminus1*cminus1, 0.0);
    Real variance2 = fmax(cplus2 - cminus2*cminus2, 0.0);
    Real variance3 = fmax(cplus3 - cminus3*cminus3, 0.0);

    Real max_cplus = fmax(cplus1, fmax(cplus2, cplus3));
    const int max_nsub = 1000;
    int nsub = 1;
    if (max_cplus > 1.0) {
      nsub = static_cast<int>(ceil(max_cplus));
      nsub = (nsub > max_nsub) ? max_nsub : nsub;
    }
    Real inv_nsub = 1.0 / static_cast<Real>(nsub);
    Real cminus1_sub = cminus1 * inv_nsub;
    Real cminus2_sub = cminus2 * inv_nsub;
    Real cminus3_sub = cminus3 * inv_nsub;
    Real variance1_sub = variance1 * inv_nsub;
    Real variance2_sub = variance2 * inv_nsub;
    Real variance3_sub = variance3 * inv_nsub;

    constexpr Real sqrt3 = 1.7320508075688772935;
    Real disp1_total = 0.0;
    Real disp2_total = 0.0;
    Real disp3_total = 0.0;
    for (int isub = 0; isub < nsub; ++isub) {
      Real xi1 = sqrt3*(2.0*StatelessUniform01(pi(PTAG,p), ncycle, rseed,
                                                1*10000 + isub) - 1.0);
      disp1_total += Ito2DisplacementFromMoments(cminus1_sub, variance1_sub,
                                                  mbsize.d_view(m).dx1, xi1);

      if (multi_d) {
        Real xi2 = sqrt3*(2.0*StatelessUniform01(pi(PTAG,p), ncycle, rseed,
                                                  2*10000 + isub) - 1.0);
        disp2_total += Ito2DisplacementFromMoments(cminus2_sub, variance2_sub,
                                                    mbsize.d_view(m).dx2, xi2);
      }

      if (three_d) {
        Real xi3 = sqrt3*(2.0*StatelessUniform01(pi(PTAG,p), ncycle, rseed,
                                                  3*10000 + isub) - 1.0);
        disp3_total += Ito2DisplacementFromMoments(cminus3_sub, variance3_sub,
                                                    mbsize.d_view(m).dx3, xi3);
      }
    }

    pr(IPX,p) += disp1_total;
    if (multi_d) {
      pr(IPY,p) += disp2_total;
    }
    if (three_d) {
      pr(IPZ,p) += disp3_total;
    }

    pi(PLASTLEVEL,p) = mblev.d_view(m);
    pi(PLASTMOVE,p) = 0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus Particles::AdjustMeshRefinement
//! \brief update locations of particles that enter meshblocks with new refinement levels

TaskStatus Particles::AdjustMeshRefinement(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  auto &pi = prtcl_idata;
  auto &pr = prtcl_rdata;
  auto &gids = pmy_pack->gids;
  auto &mblev = pmy_pack->pmb->mb_lev;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;

  auto &uflxidn_ = (pmy_pack->phydro != nullptr)?
                   pmy_pack->phydro->uflxidnsaved:pmy_pack->pmhd->uflxidnsaved;
  auto &flx1_ = uflxidn_.x1f;
  auto &flx2_ = uflxidn_.x2f;
  auto &flx3_ = uflxidn_.x3f;

  int ncycle = pmy_pack->pmesh->ncycle;
  int64_t rseed = random_seed;  // capture to avoid implicit 'this' capture in the lambda
  int nmb = pmy_pack->nmb_thispack;

  par_for("particle_meshshift",DevExeSpace(),0,(nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    if (pi(PLASTMOVE,p) >= 0) {
      // only update particles that are not frozen or marked for deletion

      int m = pi(PGID,p) - gids;

      // Bounds check: ensure m is valid for this rank
      if (m < 0 || m >= nmb) {
        pi(PLASTMOVE,p) = -1;  // Mark as frozen
        return;
      }

      int level = mblev.d_view(m);

      int lastlevel = pi(PLASTLEVEL,p);
      int lastmove = pi(PLASTMOVE,p);

      // oddness of the last cell that the particle lived in
      int i_parity = lastmove / 32;
      int j_parity = (lastmove % 32) / 16;
      int k_parity = (lastmove % 16) / 8;

      // direction of last move:
      //   1 -> "left" x1 face was chosen
      //   2 -> "right" x1 face was chosen
      //   3 -> "left" x2 face was chosen
      //   4 -> "right" x2 face was chosen
      //   5 -> "left" x3 face was chosen
      //   6 -> "right" x3 face was chosen
      lastmove = lastmove % 8;

      Real dx1 = mbsize.d_view(m).dx1;
      Real dx2 = multi_d ? mbsize.d_view(m).dx2 : 0.;
      Real dx3 = three_d ? mbsize.d_view(m).dx3 : 0.;

      if (level > lastlevel) {
        // this is a higher refinement level, i.e., the zones are smaller now

        if (lastmove == 1) {
          // came from zone to right (dx--)
          pr(IPX,p) += dx1/2;

          pr(IPY,p) -= dx2/2;
          pr(IPZ,p) -= dx3/2;
        } else if (lastmove == 2) {
          // came from zone to left (dx++)
          pr(IPX,p) -= dx1/2;

          pr(IPY,p) -= dx2/2;
          pr(IPZ,p) -= dx3/2;
        } else if (multi_d && lastmove == 3) {
          // came from zone above (dy--)
          pr(IPY,p) += dx2/2;

          pr(IPX,p) -= dx1/2;
          pr(IPZ,p) -= dx3/2;
        } else if (multi_d && lastmove == 4) {
          // came from zone below (dy++)
          pr(IPY,p) -= dx2/2;

          pr(IPX,p) -= dx1/2;
          pr(IPZ,p) -= dx3/2;
        } else if (three_d && lastmove == 5) {
          // came from zone in front (dz--)
          pr(IPZ,p) += dx3/2;

          pr(IPX,p) -= dx1/2;
          pr(IPY,p) -= dx2/2;
        } else if (three_d && lastmove == 6) {
          // came from zone behind (dz++)
          pr(IPZ,p) -= dx3/2;

          pr(IPX,p) -= dx1/2;
          pr(IPY,p) -= dx2/2;
        }

        int ip = (pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1 + is;
        int jp = js;
        int kp = ks;

        if (multi_d) {
          jp = (pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2 + js;
        }

        if (three_d) {
          kp = (pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3 + ks;
        }

        // Bounds check: unlike PushLagrangianMC/PushIto2 (which both check this
        // before indexing the flux arrays), this refinement-level-increase
        // branch below reads flx1_/flx2_/flx3_ at ip+1/jp+1/kp+1 with NO
        // preceding bounds check at all -- found while investigating (but
        // ultimately not the cause of) the turb64_iso lagrangian_mc-only
        // crash, whose actual root cause was an unconditional w0(m,IEN,...)
        // read for isothermal EOS in outputs/bin_prtcl.cpp (isothermal has no
        // energy slot in w0 at all). Still a genuine, independent gap worth
        // fixing defensively: flux arrays extend one element beyond the
        // active zone, so the valid range for the +1 access is [is, ie-1]
        // etc; skip if not in range rather than read OOB.
        int ie_arm = is + indcs.nx1 - 1;
        int je_arm = js + indcs.nx2 - 1;
        int ke_arm = ks + indcs.nx3 - 1;
        if (ip < is || ip > ie_arm || jp < js || jp > je_arm || kp < ks || kp > ke_arm) {
          return;
        }

        // get fluxes into the four zones that the particle could have ended up in
        Real flx1 = 0.;
        Real flx2 = 0.;
        Real flx3 = 0.;
        Real flx4 = 0.;

        if (lastmove == 1) {
          // came from zone to the right
          flx1 = -flx1_(m,kp,jp,ip+1);
          flx2 = (multi_d) ? -flx1_(m,kp,jp+1,ip+1) : 0.;
          flx3 = (three_d) ? -flx1_(m,kp+1,jp,ip+1) : 0.;
          flx4 = (multi_d && three_d) ? -flx1_(m,kp+1,jp+1,ip+1) : 0.;
        } else if (lastmove == 2) {
          // came from zone to the left
          flx1 = flx1_(m,kp,jp,ip);
          flx2 = (multi_d) ? flx1_(m,kp,jp+1,ip) : 0.;
          flx3 = (three_d) ? flx1_(m,kp+1,jp,ip) : 0.;
          flx4 = (multi_d && three_d) ? flx1_(m,kp+1,jp+1,ip) : 0.;
        } else if (lastmove == 3) {
          // came from zone above. is at least multi_d
          flx1 = -flx2_(m,kp,jp+1,ip);
          flx2 = -flx2_(m,kp,jp+1,ip+1);
          flx3 = (three_d) ? -flx2_(m,kp+1,jp+1,ip) : 0.;
          flx4 = (three_d) ? -flx2_(m,kp+1,jp+1,ip+1) : 0.;
        } else if (lastmove == 4) {
          // came from zone below. is at least multi_d
          flx1 = flx2_(m,kp,jp,ip);
          flx2 = flx2_(m,kp,jp,ip+1);
          flx3 = (three_d) ? flx2_(m,kp+1,jp,ip) : 0.;
          flx4 = (three_d) ? flx2_(m,kp+1,jp,ip+1) : 0.;
        } else if (lastmove == 5) {
          // came from zone in front. is three_d
          flx1 = -flx3_(m,kp+1,jp,ip);
          flx2 = -flx3_(m,kp+1,jp,ip+1);
          flx3 = -flx3_(m,kp+1,jp+1,ip);
          flx4 = -flx3_(m,kp+1,jp+1,ip+1);
        } else if (lastmove == 6) {
          // came from zone behind. is three_d
          flx1 = flx3_(m,kp,jp,ip);
          flx2 = flx3_(m,kp,jp,ip+1);
          flx3 = flx3_(m,kp,jp+1,ip);
          flx4 = flx3_(m,kp,jp+1,ip+1);
        }

        flx1 = (flx1 < 0) ? 0. : flx1;
        flx2 = (flx2 < 0) ? 0. : flx2;
        flx3 = (flx3 < 0) ? 0. : flx3;
        flx4 = (flx4 < 0) ? 0. : flx4;

        Real flx_total = flx1 + flx2 + flx3 + flx4;
        flx_total = (flx_total > 0) ? flx_total : 1.e-10;

        flx1 /= flx_total;
        flx2 /= flx_total;
        flx3 /= flx_total;
        flx4 /= flx_total;

        // Deterministic random seed: tag * prime1 + ncycle * prime2 + input_seed
        // Using large primes to avoid correlations: 7919 and 104729
        // Add 1 to differentiate from the seed used in PushLagrangianMC
        int64_t det_seed = pi(PTAG,p) * 7919 + ncycle * 104729 + rseed + 1;

        // Hash-based pseudo-random number generation (splitmix64 algorithm)
        // Fast, stateless, device-compatible alternative to Kokkos random pool
        uint64_t z = static_cast<uint64_t>(det_seed);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        Real rand = static_cast<Real>(z & 0x7FFFFFFFULL) /
                    static_cast<Real>(0x80000000ULL);

        int target_zone = 4;
        if (rand < flx1) {
          target_zone = 1;
        } else if (rand < flx1 + flx2) {
          target_zone = 2;
        } else if (rand < flx1 + flx2 + flx3) {
          target_zone = 3;
        }

        if (lastmove == 1 || lastmove == 2) {
          if (target_zone == 2) {
            pr(IPY,p) += mbsize.d_view(m).dx2;
          } else if (target_zone == 3) {
            pr(IPZ,p) += mbsize.d_view(m).dx3;
          } else if (target_zone == 4) {
            pr(IPY,p) += mbsize.d_view(m).dx2;
            pr(IPZ,p) += mbsize.d_view(m).dx3;
          }
        } else if (lastmove == 3 || lastmove == 4) {
          if (target_zone == 2) {
            pr(IPX,p) += mbsize.d_view(m).dx1;
          } else if (target_zone == 3) {
            pr(IPZ,p) += mbsize.d_view(m).dx3;
          } else if (target_zone == 4) {
            pr(IPX,p) += mbsize.d_view(m).dx1;
            pr(IPZ,p) += mbsize.d_view(m).dx3;
          }
        } else if (lastmove == 5 || lastmove == 6) {
          if (target_zone == 2) {
            pr(IPX,p) += mbsize.d_view(m).dx1;
          } else if (target_zone == 3) {
            pr(IPY,p) += mbsize.d_view(m).dx2;
          } else if (target_zone == 4) {
            pr(IPX,p) += mbsize.d_view(m).dx1;
            pr(IPY,p) += mbsize.d_view(m).dx2;
          }
        }

      } else if (level < lastlevel) {
        // this is a lower refinement level, i.e., the zones are larger now,
        // there's nothing special to do other than to move the particle to
        // the center of the new zone

        if (i_parity) {
          pr(IPX,p) -= mbsize.d_view(m).dx1/4;
        } else {
          pr(IPX,p) += mbsize.d_view(m).dx1/4;
        }
        if (multi_d) {
          if (j_parity) {
            pr(IPY,p) -= mbsize.d_view(m).dx2/4;
          } else {
            pr(IPY,p) += mbsize.d_view(m).dx2/4;
          }
        }
        if (three_d) {
          if (k_parity) {
            pr(IPZ,p) -= mbsize.d_view(m).dx3/4;
          } else {
            pr(IPZ,p) += mbsize.d_view(m).dx3/4;
          }
        }

        if (lastmove == 1) {
          // came from zone to right (dx--)
          pr(IPX,p) -= mbsize.d_view(m).dx1/2;
        } else if (lastmove == 2) {
          // came from zone to left (dx++)
          pr(IPX,p) += mbsize.d_view(m).dx1/2;
        } else if (lastmove == 3) {
          // came from zone above (dy--)
          pr(IPY,p) -= mbsize.d_view(m).dx2/2;
        } else if (lastmove == 4) {
          // came from zone below (dy++)
          pr(IPY,p) += mbsize.d_view(m).dx2/2;
        } else if (lastmove == 5) {
          // came from zone in front (dz--)
          pr(IPZ,p) -= mbsize.d_view(m).dx3/2;
        } else if (lastmove == 6) {
          // came from zone behind (dz++)
          pr(IPZ,p) += mbsize.d_view(m).dx3/2;
        }
      }
    }
  });

  return TaskStatus::complete;
}

} // namespace particles
