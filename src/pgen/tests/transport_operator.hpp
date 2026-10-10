#ifndef PGEN_TESTS_TRANSPORT_OPERATOR_HPP_
#define PGEN_TESTS_TRANSPORT_OPERATOR_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only exercise of the REAL conduction flux kernel, with host-controlled RK2.
// A reflected period-four extension embeds the paper's logical 2x2 problem in
// an allocated 4x4 block. This does not relax AthenaK's minimum block size and
// does not claim to test the driver's RK stages or physical boundary routines.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "diffusion/conduction.hpp"

namespace transport_operator {
using State = std::array<Real,16>;

inline void Require(bool ok, const char *message) {
  if (!ok) {
    std::cerr << "transport operator FAILED: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

inline void Check(Mesh *pm, bool restart) {
  auto *pack = pm->pmb_pack;
  auto *mhd = pack->pmhd;
  auto &ind = pm->mb_indcs;
  Require(!restart && global_variable::nranks == 1 && pack->nmb_thispack == 1 &&
          ind.nx1 == 4 && ind.nx2 == 4 && ind.nx3 == 1,
          "requires a fresh single-rank single-block 4x4 mesh");
  Require(mhd->pcond != nullptr, "missing conduction object");
  auto *cond = mhd->pcond;
  auto eos = mhd->peos->eos_data;
  Require(eos.is_ideal && eos.gamma == 2.0 && cond->alpha_aniso > 0.0 &&
          cond->alpha_iso == 0.0, "requires gamma=2 and pure anisotropic diffusion");
  const int n1=mhd->w0.extent(4), n2=mhd->w0.extent(3), n3=mhd->w0.extent(2);
  // Swap view handles, not data: normal pgen initialization retains its own arrays.
  auto saved_b = mhd->b0;
  auto saved_bcc = mhd->bcc0;
  mhd->b0 = DvceFaceFld4D<Real>("operator_b",1,n3,n2,n1);
  mhd->bcc0 = DvceArray5D<Real>("operator_bcc",1,3,n3,n2,n1);
  DvceArray5D<Real> w("operator_w",1,5,n3,n2,n1);
  DvceFaceFld5D<Real> flux("operator_flux",1,5,n3,n2,n1);
  auto wh = Kokkos::create_mirror_view(w);
  auto bh = Kokkos::create_mirror_view(mhd->bcc0);
  auto size = pack->pmb->mb_size.h_view(0);
  FILE *out = std::fopen("transport.operator.dat", "w");
  Require(out != nullptr, "cannot open diagnostics");
  std::fprintf(out, "# case step min max sum lower_left oracle_error rhs_error\n");
  for (int test=0; test<4; ++test) {
    const bool checker = test >= 2;
    const int n = checker ? 4 : 2;
    const Real bx = test%2 == 0 ? 1.0/std::sqrt(2.0) : 1.0;
    const Real by = test%2 == 0 ? -1.0/std::sqrt(2.0) : 0.0;
    const Real cx = cond->alpha_aniso*bx*bx/(size.dx1*size.dx1);
    const Real cy = cond->alpha_aniso*by*by/(size.dx2*size.dx2);
    const Real dt = 0.1/(cx+cy);
    Kokkos::deep_copy(mhd->b0.x1f,bx);
    Kokkos::deep_copy(mhd->b0.x2f,by);
    Kokkos::deep_copy(mhd->b0.x3f,0.0);
    for (int k=0; k<n3; ++k) for (int j=0; j<n2; ++j) {
      for (int i=0; i<n1; ++i) {
        bh(0,0,k,j,i)=bx; bh(0,1,k,j,i)=by; bh(0,2,k,j,i)=0.0;
      }
    }
    Kokkos::deep_copy(mhd->bcc0,bh);
    State state{}, reference{};
    Real initial_sum=0.0;
    for (int j=0; j<n; ++j) for (int i=0; i<n; ++i) {
      state[j*n+i] = checker ? 1.0+0.1*((i+j)%2 == 0 ? 1.0 : -1.0) :
                              (i == 1 && j == 1 ? 10.0 : 0.1);
      initial_sum += state[j*n+i];
    }
    reference = state;
    const Real low=checker ? 0.9 : 0.1, high=checker ? 1.1 : 10.0;
    auto bounds = [&](const State &a) {
      for (int p=0; p<n*n; ++p) {
        Require(std::isfinite(a[p]) && a[p] >= low-1.e-12 && a[p] <= high+1.e-12,
                "raw stage temperature outside initial bounds");
      }
    };
    // Independent graph-Laplacian oracle. For these two special fixtures every
    // limited transverse slope vanishes, for BOTH van Leer and MC.
    auto oracle = [&](const State &a) {
      State rhs{};
      for (int j=0; j<n; ++j) for (int i=0; i<n; ++i) {
        int il=checker ? (i+n-1)%n : std::max(0,i-1);
        int ir=checker ? (i+1)%n : std::min(n-1,i+1);
        int jl=checker ? (j+n-1)%n : std::max(0,j-1);
        int jr=checker ? (j+1)%n : std::min(n-1,j+1);
        rhs[j*n+i] = cx*(a[j*n+ir]-2*a[j*n+i]+a[j*n+il]) +
                     cy*(a[jr*n+i]-2*a[j*n+i]+a[jl*n+i]);
      }
      return rhs;
    };
    Real rhs_error=0.0;
    auto actual = [&](const State &a) {
      for (int k=0; k<n3; ++k) for (int j=0; j<n2; ++j) {
        for (int i=0; i<n1; ++i) {
          int x=((i-ind.is)%4+4)%4, y=((j-ind.js)%4+4)%4;
          if (!checker) {
            x=x<2 ? x : 3-x;
            y=y<2 ? y : 3-y;
          }
          wh(0,IDN,k,j,i)=1.0;
          wh(0,IVX,k,j,i)=wh(0,IVY,k,j,i)=wh(0,IVZ,k,j,i)=0.0;
          wh(0,IEN,k,j,i)=a[y*n+x];
        }
      }
      Kokkos::deep_copy(w,wh);
      Kokkos::deep_copy(flux.x1f,0.0);
      Kokkos::deep_copy(flux.x2f,0.0);
      Kokkos::deep_copy(flux.x3f,0.0);
      cond->AddHeatFluxAniso(w,eos,flux);
      auto fx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),flux.x1f);
      auto fy=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),flux.x2f);
      State rhs{}, expected=oracle(a);
      for (int j=0; j<n; ++j) for (int i=0; i<n; ++i) {
        int x=ind.is+i, y=ind.js+j, k=ind.ks, p=j*n+i;
        rhs[p]=-(fx(0,IEN,k,y,x+1)-fx(0,IEN,k,y,x))/size.dx1 -
                (fy(0,IEN,k,y+1,x)-fy(0,IEN,k,y,x))/size.dx2;
        Require(std::isfinite(rhs[p]), "nonfinite flux divergence");
        Real error=std::abs(rhs[p]-expected[p]);
        rhs_error=std::max(rhs_error,error);
        Require(error <= 2.e-11*std::max(1.0,std::abs(expected[p])),
                "production flux differs from independent operator oracle");
      }
      return rhs;
    };
    for (int step=0; step<=32; ++step) {
      Real lo=high, hi=low, sum=0.0, error=0.0;
      for (int p=0; p<n*n; ++p) {
        lo=std::min(lo,state[p]); hi=std::max(hi,state[p]); sum+=state[p];
        error=std::max(error,std::abs(state[p]-reference[p]));
      }
      Require(error <= 2.e-11 && std::abs(sum-initial_sum) <= 2.e-11,
              "oracle evolution or conservation mismatch");
      if (!checker && test%2 == 1) {
        Require(std::abs(state[0]-0.1) <= 1.e-13, "axis cold corner changed");
      }
      if (checker) {
        Real amplitude=0.1*std::pow(0.68,step);  // RK2: 1-.4+.4^2/2.
        Require(std::abs(state[0]-1.0-amplitude) <= 2.e-11,
                "checkerboard fails exact discrete RK2 damping");
      }
      std::fprintf(out,"%d %d %.17e %.17e %.17e %.17e %.17e %.17e\n",
                   test,step,lo,hi,sum,state[0],error,rhs_error);
      if (step == 32) break;
      auto d0=actual(state), r0=oracle(reference);
      State stage{}, refstage{};
      for (int p=0; p<n*n; ++p) {
        stage[p]=state[p]+dt*d0[p]; refstage[p]=reference[p]+dt*r0[p];
      }
      bounds(stage);
      auto d1=actual(stage), r1=oracle(refstage);
      State raw{};
      for (int p=0; p<n*n; ++p) {
        raw[p]=stage[p]+dt*d1[p];
        state[p]=0.5*state[p]+0.5*raw[p];
        reference[p]=0.5*reference[p]+0.5*(refstage[p]+dt*r1[p]);
      }
      bounds(raw); bounds(state);
    }
  }
  Require(std::fclose(out) == 0, "cannot close diagnostics");
  Kokkos::fence();
  mhd->b0=saved_b;
  mhd->bcc0=saved_bcc;
  std::cout << "transport conduction operator: 4 cases PASS" << std::endl;
}
} // namespace transport_operator
#endif // PGEN_TESTS_TRANSPORT_OPERATOR_HPP_
