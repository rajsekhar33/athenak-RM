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
Real LagrangianMCUniform01(const int tag, const int ncycle, const int64_t base_seed) {
  int64_t det_seed = tag * 7919 + ncycle * 104729 + base_seed;
  uint64_t z = static_cast<uint64_t>(det_seed);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return static_cast<Real>(z & 0x7FFFFFFFULL) / static_cast<Real>(0x80000000ULL);
}

KOKKOS_INLINE_FUNCTION
Real Ito2Displacement(const Real pleft, const Real pright, const Real dx,
                      const Real xi) {
  Real cplus = pleft + pright;
  Real cminus = pright - pleft;
  Real variance = cplus - cminus*cminus;
  variance = variance < 0.0 ? 0.0 : variance;
  return dx*(cminus + sqrt(variance)*xi);
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

      // by convention, these values will be positive when there is outflow
      // with respect to the current particle's cell
      Real flx1_left = -flx1_(m,kp,jp,ip) / mass;
      Real flx1_right = flx1_(m,kp,jp,ip+1) / mass;
      Real flx2_left = (multi_d) ? -flx2_(m,kp,jp,ip) / mass : 0.;
      Real flx2_right = (multi_d) ? flx2_(m,kp,jp+1,ip) / mass : 0.;
      Real flx3_left = (three_d) ? -flx3_(m,kp,jp,ip) / mass : 0.;
      Real flx3_right = (three_d) ? flx3_(m,kp+1,jp,ip) / mass : 0.;

      // Clamp both negative values and sub-epsilon positive residuals to exactly
      // zero, as basic numerical hygiene for a Monte Carlo jump probability
      // (confirmed via instrumentation that flx2/flx3 are exactly zero for this
      // flow_dir=1-only test, so this clamp is not itself the fix for the
      // multi-rank restart divergence -- see bvals_part.cpp investigation).
      const Real flx_eps = 1.0e-12;
      flx1_left = flx1_left < flx_eps ? 0 : flx1_left;
      flx1_right = flx1_right < flx_eps ? 0 : flx1_right;
      flx2_left = flx2_left < flx_eps ? 0 : flx2_left;
      flx2_right = flx2_right < flx_eps ? 0 : flx2_right;
      flx3_left = flx3_left < flx_eps ? 0 : flx3_left;
      flx3_right = flx3_right < flx_eps ? 0 : flx3_right;

      Real rand = LagrangianMCUniform01(pi(PTAG,p), ncycle, rseed);

      // save refinement level of current zone
      pi(PLASTLEVEL,p) = mblev.d_view(m);

      // save parity of current zone stored as (i_isodd,j_isodd,k_isodd) * 8
      pi(PLASTMOVE,p) = 32 * (ip % 2) + 16 * (jp % 2) + 8 * (kp % 2);

      if (rand < flx1_left) {
        pr(IPX,p) -= mbsize.d_view(m).dx1;
        pi(PLASTMOVE,p) += 1;
      } else if (rand < flx1_left + flx1_right) {
        pr(IPX,p) += mbsize.d_view(m).dx1;
        pi(PLASTMOVE,p) += 2;
      } else if (multi_d && rand < flx1_left + flx1_right + flx2_left) {
        pr(IPY,p) -= mbsize.d_view(m).dx2;
        pi(PLASTMOVE,p) += 3;
      } else if (multi_d && rand < flx1_left + flx1_right + flx2_left + flx2_right) {
        pr(IPY,p) += mbsize.d_view(m).dx2;
        pi(PLASTMOVE,p) += 4;
      } else if (three_d && rand < flx1_left + flx1_right + flx2_left + flx2_right
                                + flx3_left) {
        pr(IPZ,p) -= mbsize.d_view(m).dx3;
        pi(PLASTMOVE,p) += 5;
      } else if (three_d && rand < flx1_left + flx1_right + flx2_left + flx2_right
                                + flx3_left + flx3_right) {
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

    constexpr Real sqrt3 = 1.7320508075688772935;
    Real xi1 = sqrt3*(2.0*StatelessUniform01(pi(PTAG,p), ncycle, rseed, 1) - 1.0);
    pr(IPX,p) += Ito2Displacement(p1_left, p1_right, mbsize.d_view(m).dx1, xi1);

    if (multi_d) {
      Real xi2 = sqrt3*(2.0*StatelessUniform01(pi(PTAG,p), ncycle, rseed, 2) - 1.0);
      pr(IPY,p) += Ito2Displacement(p2_left, p2_right, mbsize.d_view(m).dx2, xi2);
    }

    if (three_d) {
      Real xi3 = sqrt3*(2.0*StatelessUniform01(pi(PTAG,p), ncycle, rseed, 3) - 1.0);
      pr(IPZ,p) += Ito2Displacement(p3_left, p3_right, mbsize.d_view(m).dx3, xi3);
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
