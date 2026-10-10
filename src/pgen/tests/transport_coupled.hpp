#ifndef PGEN_TESTS_TRANSPORT_COUPLED_HPP_
#define PGEN_TESTS_TRANSPORT_COUPLED_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only 1D dynamical MHD mode. Python supplies the coupled eigenvector;
// this fixture NEVER applies an analytic evolution or corrective source.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"

namespace transport_coupled {
inline Real re[7], im[7], amplitude, bfield[3], gm1;
inline std::string basename;
inline int first_cycle, calls, counts[2];
inline Real minrho[2], minnonkin[2];
inline bool fault;

inline void Reject(const char *why) {
  std::cerr << "Coupled test invalid configuration: " << why << std::endl;
  std::exit(EXIT_FAILURE);
}

// Each cell table exports conservative averages and staggered-face values.
// Pressure/velocity below are reconstructed diagnostics, not claimed exact
// primitive cell averages at finite amplitude.
inline void Measure(Mesh *pm, Real time, int cycle, const char *suffix) {
  auto *pack=pm->pmb_pack;
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->u0);
  auto bx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x1f);
  auto by=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x2f);
  auto bz=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x3f);
  auto ind=pm->mb_indcs;
  auto sizes=pack->pmb->mb_size.h_view;
  FILE *cells=nullptr;
  if (suffix) {
    cells=std::fopen((basename+suffix).c_str(),"w");
    if (!cells) Reject("cell output");
    std::fprintf(cells,"# x dx rho momx momy momz energy "
                       "bxlo bxhi bylo byhi bzlo bzhi\n");
  }
  Real sums[12]={}, coeff[14]={}, rhomin=1.e100, pmin=1.e100, divmax=0.0;
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto s=sizes(m);
    int k=ind.ks, j=ind.js;
    for (int i=ind.is; i<=ind.ie; ++i) {
      Real x=s.x1min+(i-ind.is+0.5)*s.dx1;
      Real f[6]={bx(m,k,j,i),bx(m,k,j,i+1),by(m,k,j,i),by(m,k,j+1,i),
                 bz(m,k,j,i),bz(m,k+1,j,i)};
      Real rho=u(m,IDN,k,j,i), e=u(m,IEN,k,j,i);
      Real kin=0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+SQR(u(m,IM3,k,j,i)))/rho;
      Real mag=0.125*(SQR(f[0]+f[1])+SQR(f[2]+f[3])+SQR(f[4]+f[5]));
      Real pressure=gm1*(e-kin-mag);
      Real div=(f[1]-f[0])/s.dx1;
      Real q[7]={rho,u(m,IM1,k,j,i)/rho,u(m,IM2,k,j,i)/rho,u(m,IM3,k,j,i)/rho,
                 pressure,0.5*(f[2]+f[3]),0.5*(f[4]+f[5])};
      if (!std::isfinite(pressure) || rho<0.5 || rho>1.5 || pressure<0.5 ||
          pressure>1.5 || !std::isfinite(div)) Reject("post-cycle fluid state");
      rhomin=std::fmin(rhomin,rho); pmin=std::fmin(pmin,pressure);
      divmax=std::fmax(divmax,std::abs(div));
      Real background[7]={1,0,0,0,1,bfield[1],bfield[2]};
      for (int v=0; v<7; ++v) {
        if (!std::isfinite(q[v])) Reject("nonfinite post-cycle primitive");
        coeff[2*v]+=2*s.dx1*(q[v]-background[v])*std::cos(2*M_PI*x);
        coeff[2*v+1]-=2*s.dx1*(q[v]-background[v])*std::sin(2*M_PI*x);
      }
      sums[0]+=s.dx1*rho; sums[1]+=s.dx1*e;
      for (int d=0; d<3; ++d) {
        sums[2+d]+=s.dx1*u(m,IM1+d,k,j,i);
        sums[5+d]+=s.dx1*0.5*(f[2*d]+f[2*d+1]);
      }
      sums[8]+=s.dx1*kin; sums[9]+=s.dx1*mag; sums[10]+=s.dx1*pressure/gm1;
      if (cells) {
        std::fprintf(cells,"%.17e %.17e",x,s.dx1);
        for (int v=0; v<5; ++v) std::fprintf(cells," %.17e",u(m,v,k,j,i));
        for (Real value : f) std::fprintf(cells," %.17e",value);
        std::fprintf(cells,"\n");
      }
    }
  }
  if (cells && std::fclose(cells)!=0) Reject("cell close");
  // Finalize exports final cells only, avoiding a duplicated history endpoint.
  if (suffix && std::string(suffix)==".coupled.dat") return;
  FILE *hist=std::fopen((basename+".coupled.history").c_str(),"a");
  if (!hist) Reject("history output");
  std::fprintf(hist,"%.17e %d %.17e %.17e %.17e",time,cycle,rhomin,pmin,divmax);
  for (int i=0; i<11; ++i) std::fprintf(hist," %.17e",sums[i]);
  for (Real c : coeff) std::fprintf(hist," %.17e",c);
  std::fprintf(hist,"\n");
  if (std::fclose(hist)!=0) Reject("history close");
}

inline void RawState(Mesh *pm, Real unused_dt) {
  // This hook is PRE-CT. Do not subtract stale magnetic energy and call it the
  // current pressure. Only conserved finiteness, density and E-kinetic are tested.
  auto *pack=pm->pmb_pack;
  auto u=pack->pmhd->u0;
  auto ind=pm->mb_indcs;
  if (fault && calls==0) {
    par_for("coupled_fault",DevExeSpace(),0,0,KOKKOS_LAMBDA(int unused) {
      u(0,IDN,ind.ks,ind.js,ind.ie)=-1.0;
    });
  }
  Real lowrho, lowenergy;
  int bad;
  Kokkos::parallel_reduce("coupled_raw_conserved",
      Kokkos::RangePolicy<DevExeSpace>(0,pack->nmb_thispack*ind.nx1),
  KOKKOS_LAMBDA(int idx, Real &rmin, Real &emin, int &invalid) {
    int m=idx/ind.nx1, i=idx%ind.nx1+ind.is, j=ind.js, k=ind.ks;
    for (int v=0; v<5; ++v) {
      Real value=u(m,v,k,j,i);
      if (!(value==value) || fabs(value)>1.e100) ++invalid;
    }
    Real rho=u(m,IDN,k,j,i);
    Real nonkin=u(m,IEN,k,j,i)-0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+
                                      SQR(u(m,IM3,k,j,i)))/rho;
    rmin=fmin(rmin,rho); emin=fmin(emin,nonkin);
  },Kokkos::Min<Real>(lowrho),Kokkos::Min<Real>(lowenergy),Kokkos::Sum<int>(bad));
  int stage=calls%2;
  ++calls; ++counts[stage];
  minrho[stage]=std::fmin(minrho[stage],lowrho);
  minnonkin[stage]=std::fmin(minnonkin[stage],lowenergy);
  if (bad || lowrho<0.5 || lowenergy<0.5) {
    std::cerr << "coupled raw conserved FAILED: stage=" << stage+1 << std::endl;
    std::_Exit(EXIT_FAILURE);
  }
}

inline void Monitor(Mesh *pm) { Measure(pm,pm->time+pm->dt,pm->ncycle+1,nullptr); }

inline void Initialize(ParameterInput *pin, Mesh *pm, bool restart,
                       bool split_fixture=false) {
  auto *pack=pm->pmb_pack;
  if (!pack->pmhd || pm->multi_d || pm->multilevel || global_variable::nranks!=1 ||
      !pack->pmhd->peos->eos_data.is_ideal ||
      pin->GetString("time","evolution")!="dynamic" ||
      pin->GetString("time","integrator")!="rk2" ||
      pin->GetString("mhd","rsolver")!="hlld" ||
      (!split_fixture && pin->GetOrAddString("time","sts_integrator","none")!="none") ||
      pm->mesh_size.x1min!=0 || pm->mesh_size.x1max!=1) {
    Reject("1D unit-box dynamical HLLD/RK2 only");
  }
  if (pin->GetString("mesh","ix1_bc")!="periodic" ||
      pin->GetString("mesh","ox1_bc")!="periodic") Reject("periodic x");
  gm1=pack->pmhd->peos->eos_data.gamma-1.0;
  amplitude=pin->GetOrAddReal("problem","amplitude",1.e-4);
  if (!std::isfinite(amplitude) || amplitude<=0 || amplitude>1.e-3 ||
      std::abs(gm1-2.0/3.0)>1.e-14) Reject("amplitude/gamma");
  Real vector_norm=0.0;
  for (int v=0; v<7; ++v) {
    re[v]=pin->GetReal("problem","vr"+std::to_string(v));
    im[v]=pin->GetReal("problem","vi"+std::to_string(v));
    if (!std::isfinite(re[v]) || !std::isfinite(im[v]) ||
        std::hypot(re[v],im[v])>1.00001) Reject("eigenvector");
    vector_norm+=SQR(re[v])+SQR(im[v]);
  }
  if (vector_norm<1.e-20) Reject("zero eigenvector");
  bfield[0]=1.0; bfield[1]=0.7; bfield[2]=0.3;
  basename=pin->GetString("job","basename");
  first_cycle=pm->ncycle; calls=counts[0]=counts[1]=0;
  minrho[0]=minrho[1]=minnonkin[0]=minnonkin[1]=1.e100;
  fault=pin->GetOrAddBoolean("problem","raw_probe_negative",false);
  FILE *history=std::fopen((basename+".coupled.history").c_str(),"w");
  if (!history) Reject("history open");
  std::fprintf(history,"# t cycle rhomin pmin divmax mass energy mx my mz bx by bz "
                       "kin mag thermal; then 7 complex primitive "
                       "Fourier coefficients\n");
  if (std::fclose(history)!=0) Reject("history close");
  if (restart) {
    Measure(pm,pm->time,pm->ncycle,".coupled.initial");
    return;
  }
  auto ind=pm->mb_indcs;
  auto sizes=pack->pmb->mb_size.h_view;
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->u0);
  auto bx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x1f);
  auto by=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x2f);
  auto bz=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x3f);
  // Eight-point Gauss-Legendre integrates rho*v and rho*v^2 plus B^2 directly.
  const Real node[4]={0.1834346424956498,0.5255324099163290,
                     0.7966664774136267,0.9602898564975363};
  const Real weight[4]={0.3626837833783620,0.3137066458778873,
                       0.2223810344533745,0.1012285362903763};
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto s=sizes(m);
    int k=ind.ks,j=ind.js;
    for (int i=ind.is; i<=ind.ie; ++i) {
      Real x=s.x1min+(i-ind.is+0.5)*s.dx1, cons[5]={}, bavg[2]={};
      for (int l=0; l<4; ++l) for (int sign : {-1,1}) {
        Real phase=2*M_PI*(x+0.5*sign*s.dx1*node[l]);
        Real q[7]={1,0,0,0,1,0.7,0.3};
        for (int v=0; v<7; ++v) {
          q[v]+=amplitude*(re[v]*std::cos(phase)-im[v]*std::sin(phase));
        }
        Real w=0.5*weight[l];
        cons[0]+=w*q[0];
        for (int v=0; v<3; ++v) cons[1+v]+=w*q[0]*q[1+v];
        cons[4]+=w*(q[4]/gm1+0.5*q[0]*(SQR(q[1])+SQR(q[2])+SQR(q[3]))+
                   0.5*(1+SQR(q[5])+SQR(q[6])));
        bavg[0]+=w*q[5]; bavg[1]+=w*q[6];
      }
      for (int v=0; v<5; ++v) u(m,v,k,j,i)=cons[v];
      bx(m,k,j,i)=bx(m,k,j,i+1)=1.0;
      by(m,k,j,i)=by(m,k,j+1,i)=bavg[0];
      bz(m,k,j,i)=bz(m,k+1,j,i)=bavg[1];
    }
  }
  Kokkos::deep_copy(pack->pmhd->u0,u);
  Kokkos::deep_copy(pack->pmhd->b0.x1f,bx);
  Kokkos::deep_copy(pack->pmhd->b0.x2f,by);
  Kokkos::deep_copy(pack->pmhd->b0.x3f,bz);
  Measure(pm,pm->time,pm->ncycle,".coupled.initial");
}

inline void Finalize(ParameterInput *pin, Mesh *pm) {
  Measure(pm,pm->time,pm->ncycle,".coupled.dat");
  int cycles=pm->ncycle-first_cycle;
  if (cycles<1 || calls!=2*cycles) Reject("stage coverage");
  FILE *f=std::fopen((basename+".coupled.summary").c_str(),"w");
  if (!f) Reject("summary output");
  std::fprintf(f,"%.17e %d\n",pm->time,cycles);
  if (std::fclose(f)!=0) Reject("summary close");
  f=std::fopen((basename+".coupled.stages").c_str(),"w");
  if (!f) Reject("stage output");
  for (int s=0; s<2; ++s) {
    std::fprintf(f,"%d %d %.17e %.17e\n",s+1,counts[s],minrho[s],minnonkin[s]);
  }
  if (std::fclose(f)!=0) Reject("stage close");
}
} // namespace transport_coupled
#endif // PGEN_TESTS_TRANSPORT_COUPLED_HPP_
