#ifndef PGEN_TESTS_TRANSPORT_EQUILIBRIUM_HPP_
#define PGEN_TESTS_TRANSPORT_EQUILIBRIUM_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Fixed-background, RK2-only tests of prescribed-temperature heat budgets.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/conduction.hpp"
#include "pgen/tests/transport_stages.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace transport_equilibrium {
struct Config {
  bool sovinec;
  Real contrast, perturbation, iso;
};
inline Config config;
inline Real initial_energy, final_energy, source_heat, boundary_heat, start_time;
inline Real minimum, maximum, budget_error, last_change;
inline int phase, samples;
inline DvceArray4D<Real> previous;

KOKKOS_INLINE_FUNCTION
Real Temperature(Config c, Real x, Real y) {
  if (c.sovinec) return 1.0+cos(M_PI*x)*cos(M_PI*y);
  Real resistance = x<0.0 ? x+0.5 : 0.5+x/c.contrast;
  return 1.0+resistance/(0.5+0.5/c.contrast);
}

KOKKOS_INLINE_FUNCTION
Real Density(Config c, Real x) {
  return c.sovinec || x<0.0 ? 1.0 : c.contrast;
}

KOKKOS_INLINE_FUNCTION
Real Potential(Real x, Real y) {
  return cos(M_PI*x)*cos(M_PI*y)/M_PI;
}

KOKKOS_INLINE_FUNCTION
Real Heating(Config c, Real x, Real y) {
  return c.sovinec ? 2.0*SQR(M_PI)*c.iso*cos(M_PI*x)*cos(M_PI*y) : 0.0;
}

inline void Field(Mesh *pm, bool fresh) {
  auto *pack=pm->pmb_pack;
  auto b=pack->pmhd->b0;
  auto size=pack->pmb->mb_size;
  auto ind=pm->mb_indcs;
  auto c=config;
  int ni=ind.nx1+2*ind.ng, nj=ind.nx2+2*ind.ng, k=ind.ks;
  par_for("equilibrium_field",DevExeSpace(),0,pack->nmb_thispack-1,0,nj,0,ni,
  KOKKOS_LAMBDA(int m, int j, int i) {
    auto s=size.d_view(m);
    Real xl=LeftEdgeX(i-ind.is,ind.nx1,s.x1min,s.x1max);
    Real yl=LeftEdgeX(j-ind.js,ind.nx2,s.x2min,s.x2max);
    Real xc=xl+0.5*s.dx1, yc=yl+0.5*s.dx2;
    if (j<nj && (fresh || xl< -0.5 || xl>0.5 || yc< -0.5 || yc>0.5)) {
      b.x1f(m,k,j,i)=c.sovinec ?
          (Potential(xl,yl+s.dx2)-Potential(xl,yl))/s.dx2 : 1.0;
    }
    if (i<ni && (fresh || xc< -0.5 || xc>0.5 || yl< -0.5 || yl>0.5)) {
      b.x2f(m,k,j,i)=c.sovinec ?
          -(Potential(xl+s.dx1,yl)-Potential(xl,yl))/s.dx1 : 0.0;
    }
    if (i<ni && j<nj && (fresh || xc< -0.5 || xc>0.5 || yc< -0.5 || yc>0.5)) {
      b.x3f(m,k,j,i)=b.x3f(m,k+1,j,i)=0.0;
    }
  });
}

inline void Boundary(Mesh *pm) {
  Field(pm,false);
  auto *pack=pm->pmb_pack;
  auto u=pack->pmhd->u0;
  auto b=pack->pmhd->b0;
  auto bc=pack->pmb->mb_bcs;
  auto ind=pm->mb_indcs;
  auto c=config;
  auto size=pack->pmb->mb_size;
  int ni=ind.nx1+2*ind.ng, nj=ind.nx2+2*ind.ng, k=ind.ks;
  par_for("equilibrium_boundary",DevExeSpace(),0,pack->nmb_thispack-1,0,nj-1,0,ni-1,
  KOKKOS_LAMBDA(int m, int j, int i) {
    int ii=i, jj=j, sign=1;
    bool physical=false;
    if (i<ind.is && bc.d_view(m,0)==BoundaryFlag::user) {
      ii=2*ind.is-1-i; sign=-sign; physical=true;
    }
    if (i>ind.ie && bc.d_view(m,1)==BoundaryFlag::user) {
      ii=2*ind.ie+1-i; sign=-sign; physical=true;
    }
    if (j<ind.js && bc.d_view(m,2)==BoundaryFlag::user) {
      jj=2*ind.js-1-j; sign=-sign; physical=true;
    }
    if (j>ind.je && bc.d_view(m,3)==BoundaryFlag::user) {
      jj=2*ind.je+1-j; sign=-sign; physical=true;
    }
    if (!physical) return;
    auto s=size.d_view(m);
    Real x=CellCenterX(i-ind.is,ind.nx1,s.x1min,s.x1max);
    Real rho=Density(c,x), temp;
    Real mbx=0.5*(b.x1f(m,k,jj,ii)+b.x1f(m,k,jj,ii+1));
    Real mby=0.5*(b.x2f(m,k,jj,ii)+b.x2f(m,k,jj+1,ii));
    Real mirror=(u(m,IEN,k,jj,ii)-0.5*(mbx*mbx+mby*mby))/u(m,IDN,k,jj,ii);
    if (c.sovinec) {
      temp=1.0+sign*(mirror-1.0);
    } else {
      temp=2.0*(x< -0.5 ? 1.0 : 2.0)-mirror;
    }
    Real bx=0.5*(b.x1f(m,k,j,i)+b.x1f(m,k,j,i+1));
    Real by=0.5*(b.x2f(m,k,j,i)+b.x2f(m,k,j+1,i));
    u(m,IDN,k,j,i)=rho;
    u(m,IM1,k,j,i)=u(m,IM2,k,j,i)=u(m,IM3,k,j,i)=0.0;
    u(m,IEN,k,j,i)=rho*temp+0.5*(bx*bx+by*by);
  });
}

inline void Source(Mesh *pm, Real bdt) {
  auto *pack=pm->pmb_pack;
  auto u=pack->pmhd->u0;
  auto f=pack->pmhd->uflx;
  auto bc=pack->pmb->mb_bcs;
  auto size=pack->pmb->mb_size;
  auto ind=pm->mb_indcs;
  auto c=config;
  Real heating, outward;
  int ni=ind.nx1, nj=ind.nx2, k=ind.ks;
  Kokkos::parallel_reduce("equilibrium_source_budget",
      Kokkos::RangePolicy<DevExeSpace>(0,pack->nmb_thispack*ni*nj),
  KOKKOS_LAMBDA(int idx, Real &heat, Real &out) {
    int m=idx/(ni*nj), j=(idx/ni)%nj+ind.js, i=idx%ni+ind.is;
    auto s=size.d_view(m);
    Real x=CellCenterX(i-ind.is,ni,s.x1min,s.x1max);
    Real y=CellCenterX(j-ind.js,nj,s.x2min,s.x2max);
    Real q=Heating(c,x,y);
    u(m,IEN,k,j,i)+=bdt*q;
    heat+=q*s.dx1*s.dx2*s.dx3;
    if (i==ind.is && bc.d_view(m,0)==BoundaryFlag::user)
      out-=f.x1f(m,IEN,k,j,i)*s.dx2*s.dx3;
    if (i==ind.ie && bc.d_view(m,1)==BoundaryFlag::user)
      out+=f.x1f(m,IEN,k,j,i+1)*s.dx2*s.dx3;
    if (j==ind.js && bc.d_view(m,2)==BoundaryFlag::user)
      out-=f.x2f(m,IEN,k,j,i)*s.dx1*s.dx3;
    if (j==ind.je && bc.d_view(m,3)==BoundaryFlag::user)
      out+=f.x2f(m,IEN,k,j+1,i)*s.dx1*s.dx3;
  },Kokkos::Sum<Real>(heating),Kokkos::Sum<Real>(outward));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&heating,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&outward,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  // RK2: first-stage increment is subsequently weighted by 1/2; second bdt=dt/2.
  Real weight=phase==0 ? 0.5*bdt : bdt;
  source_heat+=weight*heating;
  boundary_heat+=weight*outward;
  transport_stages::Check(pm);
  phase=1-phase;
}

inline void Monitor(Mesh *pm) {
  auto *pack=pm->pmb_pack;
  auto u=pack->pmhd->u0;
  auto b=pack->pmhd->bcc0;
  auto size=pack->pmb->mb_size;
  auto ind=pm->mb_indcs;
  auto old=previous;
  Real lo, hi, change, energy;
  int bad, ni=ind.nx1, nj=ind.nx2, k=ind.ks;
  Kokkos::parallel_reduce("equilibrium_monitor",
      Kokkos::RangePolicy<DevExeSpace>(0,pack->nmb_thispack*ni*nj),
  KOKKOS_LAMBDA(int idx, Real &mn, Real &mx, Real &delta, Real &sum, int &invalid) {
    int m=idx/(ni*nj), j=(idx/ni)%nj+ind.js, i=idx%ni+ind.is;
    auto s=size.d_view(m);
    Real t=(u(m,IEN,k,j,i)-0.5*(SQR(b(m,IBX,k,j,i))+SQR(b(m,IBY,k,j,i))+
                               SQR(b(m,IBZ,k,j,i))))/u(m,IDN,k,j,i);
    if (!(t==t) || fabs(t)>1.e100 || !(u(m,IDN,k,j,i)>0.0)) ++invalid;
    mn=fmin(mn,t); mx=fmax(mx,t);
    delta=fmax(delta,fabs(t-old(m,k,j,i)));
    old(m,k,j,i)=t;
    sum+=u(m,IEN,k,j,i)*s.dx1*s.dx2*s.dx3;
  },Kokkos::Min<Real>(lo),Kokkos::Max<Real>(hi),Kokkos::Max<Real>(change),
    Kokkos::Sum<Real>(energy),Kokkos::Sum<int>(bad));
  bad+=pm->ecounter.neos_dfloor+pm->ecounter.neos_efloor+
       pm->ecounter.neos_tfloor+pm->ecounter.neos_fail;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&lo,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&hi,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&change,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&energy,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  minimum=std::fmin(minimum,lo); maximum=std::fmax(maximum,hi);
  final_energy=energy;
  last_change=change/pm->dt;
  Real residual=fabs(energy-initial_energy-source_heat+boundary_heat);
  budget_error=std::fmax(budget_error,residual);
  ++samples;
  if (bad || phase!=0 || lo<1.0-1.e-9 || hi>2.1 ||
      residual>1.e-8*std::fmax(1.0,initial_energy)) {
    std::cerr << "equilibrium state/budget FAILED: " << lo << " " << hi
              << " events=" << bad << " residual=" << residual << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

inline void Finalize(ParameterInput *pin, Mesh *pm) {
  transport_stages::Write(pin,pm);
  auto *pack=pm->pmb_pack;
  auto *mhd=pack->pmhd;
  // Re-evaluate the actual operator on the final, ghost-filled state for residuals.
  Kokkos::deep_copy(mhd->uflx.x1f,0.0);
  Kokkos::deep_copy(mhd->uflx.x2f,0.0);
  Kokkos::deep_copy(mhd->uflx.x3f,0.0);
  mhd->pcond->AddHeatFluxes(mhd->w0,mhd->peos->eos_data,mhd->uflx);
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mhd->u0);
  auto b=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mhd->bcc0);
  auto fx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mhd->uflx.x1f);
  auto fy=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mhd->uflx.x2f);
  auto bx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mhd->b0.x1f);
  auto by=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mhd->b0.x2f);
  auto ind=pm->mb_indcs;
  std::string base=pin->GetString("job","basename");
  std::string file=base+".eq."+std::to_string(global_variable::my_rank)+".dat";
  FILE *fp=std::fopen(file.c_str(),"w");
  if (!fp) std::exit(EXIT_FAILURE);
  std::fprintf(fp,"# x y volume rho T target energy fxlo fxhi fylo fyhi divB\n");
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto s=pack->pmb->mb_size.h_view(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        int k=ind.ks;
        Real x=CellCenterX(i-ind.is,ind.nx1,s.x1min,s.x1max);
        Real y=CellCenterX(j-ind.js,ind.nx2,s.x2min,s.x2max);
        Real t=(u(m,IEN,k,j,i)-0.5*(SQR(b(m,IBX,k,j,i))+SQR(b(m,IBY,k,j,i))+
                                   SQR(b(m,IBZ,k,j,i))))/u(m,IDN,k,j,i);
        Real div=(bx(m,k,j,i+1)-bx(m,k,j,i))/s.dx1+
                 (by(m,k,j+1,i)-by(m,k,j,i))/s.dx2;
        std::fprintf(fp,"%.17e %.17e %.17e %.17e %.17e %.17e %.17e "
                        "%.17e %.17e %.17e %.17e %.17e\n",x,y,s.dx1*s.dx2*s.dx3,
                     u(m,IDN,k,j,i),t,Temperature(config,x,y),u(m,IEN,k,j,i),
                     fx(m,IEN,k,j,i),fx(m,IEN,k,j,i+1),fy(m,IEN,k,j,i),
                     fy(m,IEN,k,j+1,i),div);
      }
    }
  }
  if (std::fclose(fp)!=0) std::exit(EXIT_FAILURE);
  if (global_variable::my_rank==0) {
    fp=std::fopen((base+".eq.summary").c_str(),"w");
    if (!fp) std::exit(EXIT_FAILURE);
    std::fprintf(fp,"# time start samples Tmin Tmax E0 E source out "
        "max_budget dTdt phase\n"
        "%.17e %.17e %d %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %d\n",
        pm->time,start_time,samples,minimum,maximum,initial_energy,final_energy,
        source_heat,boundary_heat,budget_error,last_change,phase);
    if (std::fclose(fp)!=0) std::exit(EXIT_FAILURE);
  }
  // Release the fixture's namespace-scope view before Athena finalizes Kokkos.
  previous = DvceArray4D<Real>();
}

inline void Initialize(ParameterInput *pin, Mesh *pm, bool restart) {
  auto *pack=pm->pmb_pack;
  auto *mhd=pack->pmhd;
  config.sovinec=pin->GetString("problem","profile")=="sovinec";
  config.contrast=pin->GetOrAddReal("problem","density_ratio",10.0);
  config.perturbation=pin->GetOrAddReal("problem","perturbation",0.0);
  config.iso=pin->GetReal("mhd","alpha_iso");
  if (!pm->two_d || pm->multilevel || !mhd || !mhd->pcond ||
      !mhd->peos->eos_data.is_ideal || mhd->peos->eos_data.gamma!=2.0 ||
      pin->GetString("time","evolution")!="kinematic" ||
      pin->GetString("time","integrator")!="rk2" ||
      pin->GetOrAddString("time","sts_integrator","none")!="none" ||
      !std::isfinite(config.contrast) || config.contrast<=0.0 ||
      !std::isfinite(config.perturbation) || fabs(config.perturbation)>0.05 ||
      pm->mesh_size.x1min!= -0.5 || pm->mesh_size.x1max!=0.5 ||
      pm->mesh_size.x2min!= -0.5 || pm->mesh_size.x2max!=0.5 ||
      (config.sovinec && config.iso<=0.0) ||
      (!config.sovinec && (config.iso!=0.0 || mhd->pcond->alpha_aniso<=0.0))) {
    std::cerr << "equilibrium requires 2D uniform-level ideal kinematic RK2 MHD, "
              << "gamma=2, explicit conduction and valid unit-square fixture"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int d=0; d<4; ++d) {
    auto expected=(config.sovinec || d<2) ? BoundaryFlag::user : BoundaryFlag::periodic;
    if (pm->mesh_bcs[d]!=expected) std::exit(EXIT_FAILURE);
  }
  Field(pm,!restart);
  auto u=mhd->u0;
  auto b=mhd->b0;
  auto ind=pm->mb_indcs;
  auto size=pack->pmb->mb_size;
  auto c=config;
  int ni=ind.nx1+2*ind.ng, nj=ind.nx2+2*ind.ng, k=ind.ks;
  Kokkos::realloc(previous,pack->nmb_thispack,1,nj,ni);
  auto old=previous;
  Real energy;
  Kokkos::parallel_reduce("equilibrium_initialize",
      Kokkos::RangePolicy<DevExeSpace>(0,pack->nmb_thispack*ni*nj),
  KOKKOS_LAMBDA(int idx, Real &sum) {
    int m=idx/(ni*nj), j=(idx/ni)%nj, i=idx%ni;
    auto s=size.d_view(m);
    Real x=CellCenterX(i-ind.is,ind.nx1,s.x1min,s.x1max);
    Real y=CellCenterX(j-ind.js,ind.nx2,s.x2min,s.x2max);
    Real b1=0.5*(b.x1f(m,k,j,i)+b.x1f(m,k,j,i+1));
    Real b2=0.5*(b.x2f(m,k,j,i)+b.x2f(m,k,j+1,i));
    Real mag=0.5*(b1*b1+b2*b2);
    if (!restart) {
      Real t=Temperature(c,x,y)+c.perturbation*sin(M_PI*(x+0.5));
      u(m,IDN,k,j,i)=Density(c,x);
      u(m,IM1,k,j,i)=u(m,IM2,k,j,i)=u(m,IM3,k,j,i)=0.0;
      u(m,IEN,k,j,i)=u(m,IDN,k,j,i)*t+mag;
    }
    if (i>=ind.is && i<=ind.ie && j>=ind.js && j<=ind.je) {
      sum+=u(m,IEN,k,j,i)*s.dx1*s.dx2*s.dx3;
      old(m,k,j,i)=(u(m,IEN,k,j,i)-mag)/u(m,IDN,k,j,i);
    }
  },Kokkos::Sum<Real>(energy));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&energy,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  initial_energy=final_energy=energy;
  source_heat=boundary_heat=budget_error=last_change=0.0;
  minimum=std::numeric_limits<Real>::max(); maximum=-minimum;
  phase=samples=0;
  start_time=pm->time;
  transport_stages::Reset(pin,pm,1.0,2.1);
}
} // namespace transport_equilibrium
#endif // PGEN_TESTS_TRANSPORT_EQUILIBRIUM_HPP_
