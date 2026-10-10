#ifndef PGEN_TESTS_TRANSPORT_STAGES_HPP_
#define PGEN_TESTS_TRANSPORT_STAGES_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only raw conserved-state diagnostics for static-field RK1/RK2 fixtures.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace transport_stages {
inline int stages, calls, first_cycle, counts[2];
inline Real lower, upper, minimum[2], maximum[2];
inline bool inject_fault;

inline void Reset(ParameterInput *pin, Mesh *pm, Real lo, Real hi) {
  std::string integrator=pin->GetString("time","integrator");
  if (integrator!="rk1" && integrator!="rk2") {
    std::cerr << "transport raw-stage fixture requires rk1 or rk2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  stages=integrator=="rk1" ? 1 : 2;
  calls=counts[0]=counts[1]=0;
  first_cycle=pm->ncycle;
  lower=lo; upper=hi;
  inject_fault=pin->GetOrAddBoolean("problem","stage_probe_negative",false);
  for (int s=0; s<2; ++s) {
    minimum[s]=std::numeric_limits<Real>::max(); maximum[s]=-minimum[s];
  }
}

// Called from the source hook AFTER the RK update, BEFORE boundary C2P/floors.
// The magnetic field is static in these fixtures, so bcc0 is stage independent.
inline void Check(Mesh *pm, Real unused_bdt=0.0) {
  auto *pack=pm->pmb_pack;
  auto u=pack->pmhd->u0;
  auto b=pack->pmhd->bcc0;
  auto ind=pm->mb_indcs;
  Real gm1=pack->pmhd->peos->eos_data.gamma-1.0;
  int ni=ind.nx1, nj=ind.nx2, nk=ind.nx3, k=ind.ke;
  // Fault at the last plane also verifies that the diagnostic covers all of 3D.
  if (inject_fault && calls==0) {
    par_for("transport_stage_fault",DevExeSpace(),0,0,KOKKOS_LAMBDA(int unused) {
      int i=ind.is, j=ind.js;
      Real rho=u(0,IDN,k,j,i);
      Real mag=0.5*(SQR(b(0,IBX,k,j,i))+SQR(b(0,IBY,k,j,i))+
                    SQR(b(0,IBZ,k,j,i)));
      u(0,IEN,k,j,i)=mag-rho/gm1;  // Deliberately invalid TEST input only.
    });
  }
  Real lo, hi;
  int bad;
  Kokkos::parallel_reduce("transport_raw_stage",
      Kokkos::RangePolicy<DevExeSpace>(0,pack->nmb_thispack*ni*nj*nk),
  KOKKOS_LAMBDA(int idx, Real &mn, Real &mx, int &invalid) {
    int m=idx/(ni*nj*nk), k=(idx/(ni*nj))%nk+ind.ks;
    int j=(idx/ni)%nj+ind.js, i=idx%ni+ind.is;
    Real rho=u(m,IDN,k,j,i);
    Real kin=0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+
                  SQR(u(m,IM3,k,j,i)))/rho;
    Real mag=0.5*(SQR(b(m,IBX,k,j,i))+SQR(b(m,IBY,k,j,i))+SQR(b(m,IBZ,k,j,i)));
    Real temp=gm1*(u(m,IEN,k,j,i)-kin-mag)/rho;
    if (!(rho>0.0) || !(temp==temp) || fabs(temp)>1.e100) ++invalid;
    mn=fmin(mn,temp); mx=fmax(mx,temp);
  },Kokkos::Min<Real>(lo),Kokkos::Max<Real>(hi),Kokkos::Sum<int>(bad));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&lo,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&hi,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  int s=calls%stages;
  ++calls; ++counts[s];
  minimum[s]=std::fmin(minimum[s],lo); maximum[s]=std::fmax(maximum[s],hi);
  if (bad || lo<lower-1.e-10 || hi>upper+1.e-10) {
    std::cerr << "transport raw stage FAILED: stage=" << s+1 << " min=" << lo
              << " max=" << hi << " invalid=" << bad << std::endl;
    // Fatal test rejection occurs inside a Kokkos task. Flush the diagnostic
    // above, then bypass static view teardown without normal solver finalization.
    // Successful simulations still exercise the ordinary finalization path.
    std::_Exit(EXIT_FAILURE);
  }
}

inline void Write(ParameterInput *pin, Mesh *pm) {
  int cycles=pm->ncycle-first_cycle;
  if (cycles<1 || calls!=stages*cycles) {
    std::cerr << "transport raw stage coverage FAILED" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank!=0) return;
  std::string path=pin->GetString("job","basename")+".stages";
  FILE *file=std::fopen(path.c_str(),"w");
  if (!file) std::exit(EXIT_FAILURE);
  std::fprintf(file,"# stage calls raw_Tmin raw_Tmax\n");
  for (int s=0; s<stages; ++s) {
    std::fprintf(file,"%d %d %.17e %.17e\n",s+1,counts[s],minimum[s],maximum[s]);
  }
  if (std::fclose(file)!=0) std::exit(EXIT_FAILURE);
}
} // namespace transport_stages
#endif // PGEN_TESTS_TRANSPORT_STAGES_HPP_
