#ifndef PGEN_TESTS_TRANSPORT_GAUSSIAN_HPP_
#define PGEN_TESTS_TRANSPORT_GAUSSIAN_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only periodic cell-average Gaussian, constant field and density, explicit RK2.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/tests/transport_stages.hpp"
#include "pgen/tests/transport_front.hpp"

namespace transport_gaussian {
struct Config {
  Real sigma, amplitude, background, rho, bx, by, bz;
  bool three_d;
  bool front;
  int nx, ny, nz;
  Real front_smoothing;
};
inline Config config;

KOKKOS_INLINE_FUNCTION
Real CellFactor(Real center, Real dx, Real sigma) {
  Real total=0.0;
  // Nearest periodic images suffice for the constrained sigma <= 0.15 fixture.
  for (int image=-1; image<=1; ++image) {
    Real offset=center-0.5-image;
    total+=sigma*sqrt(M_PI/2.0)/dx*
           (erf((offset+0.5*dx)/(sqrt(2.0)*sigma))-
            erf((offset-0.5*dx)/(sqrt(2.0)*sigma)));
  }
  return total;
}

KOKKOS_INLINE_FUNCTION
Real Initial(Config c, Real x, Real y, Real z, Real dx, Real dy, Real dz) {
  if (c.front) {
    Real widths[3], phase=c.nx*x+c.ny*y+c.nz*z;
    int dim=0;
    if (c.nx) widths[dim++]=dx;
    if (c.ny) widths[dim++]=dy;
    if (c.nz) widths[dim++]=dz;
    return c.background+c.amplitude*
           transport_front::Fraction(phase,widths,dim,c.front_smoothing);
  }
  Real value=CellFactor(x,dx,c.sigma)*CellFactor(y,dy,c.sigma);
  if (c.three_d) value*=CellFactor(z,dz,c.sigma);
  return c.background+c.amplitude*value;
}

inline void Finalize(ParameterInput *pin, Mesh *pm) {
  transport_stages::Write(pin,pm);
  auto *pack=pm->pmb_pack;
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->u0);
  auto bx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x1f);
  auto by=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x2f);
  auto bz=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x3f);
  auto &ind=pm->mb_indcs;
  auto sizes=pack->pmb->mb_size.h_view;
  Real gm1=pack->pmhd->peos->eos_data.gamma-1.0;
  std::string base=pin->GetString("job","basename");
  std::string tag=config.front ? ".front" : ".gaussian";
  FILE *f=std::fopen((base+tag+".dat").c_str(),"w");
  if (!f) std::exit(EXIT_FAILURE);
  std::fprintf(f,"# x y z volume rho temperature initial_temperature energy divb\n");
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto s=sizes(m);
    for (int k=ind.ks; k<=ind.ke; ++k) {
      for (int j=ind.js; j<=ind.je; ++j) {
        for (int i=ind.is; i<=ind.ie; ++i) {
          Real x=CellCenterX(i-ind.is,ind.nx1,s.x1min,s.x1max);
          Real y=CellCenterX(j-ind.js,ind.nx2,s.x2min,s.x2max);
          Real z=CellCenterX(k-ind.ks,ind.nx3,s.x3min,s.x3max);
          Real b1=0.5*(bx(m,k,j,i)+bx(m,k,j,i+1));
          Real b2=0.5*(by(m,k,j,i)+by(m,k,j+1,i));
          Real b3=0.5*(bz(m,k,j,i)+bz(m,k+1,j,i));
          Real rho=u(m,IDN,k,j,i);
          Real kinetic=0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+
                            SQR(u(m,IM3,k,j,i)))/rho;
          Real temp=gm1*(u(m,IEN,k,j,i)-kinetic-0.5*(b1*b1+b2*b2+b3*b3))/rho;
          Real divb=(bx(m,k,j,i+1)-bx(m,k,j,i))/s.dx1+
                    (by(m,k,j+1,i)-by(m,k,j,i))/s.dx2;
          if (pm->three_d) divb+=(bz(m,k+1,j,i)-bz(m,k,j,i))/s.dx3;
          std::fprintf(f,"%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
                       x,y,z,s.dx1*s.dx2*s.dx3,rho,temp,
                       Initial(config,x,y,z,s.dx1,s.dx2,s.dx3),u(m,IEN,k,j,i),divb);
        }
      }
    }
  }
  if (std::fclose(f)!=0) std::exit(EXIT_FAILURE);
  f=std::fopen((base+tag+".summary").c_str(),"w");
  if (!f) std::exit(EXIT_FAILURE);
  std::fprintf(f,"# time cycles\n%.17e %d\n",pm->time,
               pm->ncycle-transport_stages::first_cycle);
  if (std::fclose(f)!=0) std::exit(EXIT_FAILURE);
}

inline void Initialize(ParameterInput *pin, Mesh *pm, bool restart,
                       bool separate_stage_probe=false) {
  auto *pack=pm->pmb_pack;
  auto s=pm->mesh_size;
  if (!pack->pmhd || !pm->multi_d || pm->multilevel || global_variable::nranks!=1 ||
      !pack->pmhd->peos->eos_data.is_ideal ||
      pin->GetString("time","evolution")!="kinematic" ||
      pin->GetString("time","integrator")!="rk2" ||
      (!separate_stage_probe &&
       pin->GetOrAddString("time","sts_integrator","none")!="none") ||
      s.x1min!=0.0 || s.x1max!=1.0 || s.x2min!=0.0 || s.x2max!=1.0 ||
      s.x3min!=0.0 || s.x3max!=1.0) {
    std::cerr << "Gaussian test requires one-rank uniform unit-box 2D/3D kinematic RK2"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (const char *key : {"ix1_bc","ox1_bc","ix2_bc","ox2_bc","ix3_bc","ox3_bc"}) {
    if (pin->GetString("mesh",key)!="periodic") std::exit(EXIT_FAILURE);
  }
  Config c;
  c.sigma=pin->GetOrAddReal("problem","sigma",0.1);
  c.amplitude=pin->GetOrAddReal("problem","amplitude",0.1);
  c.background=pin->GetOrAddReal("problem","temp0",1.0);
  c.rho=pin->GetOrAddReal("problem","rho0",1.0);
  c.bx=pin->GetOrAddReal("problem","bx",1.0);
  c.by=pin->GetOrAddReal("problem","by",2.0);
  c.bz=pin->GetOrAddReal("problem","bz",3.0);
  c.three_d=pm->three_d;
  c.front=pin->GetString("problem","profile")=="front";
  if (separate_stage_probe) c.front=pin->GetString("problem","multidim_shape")=="front";
  c.nx=c.ny=c.nz=0;
  c.front_smoothing=0.0;
  if (c.front) {
    c.nx=pin->GetOrAddInteger("problem","normal_x",1);
    c.ny=pin->GetOrAddInteger("problem","normal_y",0);
    c.nz=pin->GetOrAddInteger("problem","normal_z",0);
    c.front_smoothing=pin->GetOrAddReal("problem","front_smoothing",0.04);
    if (c.nx<0 || c.nx>1 || c.ny<0 || c.ny>1 || c.nz<0 || c.nz>1 ||
        c.nx+c.ny+c.nz==0 || (!pm->three_d && c.nz!=0) ||
        !std::isfinite(c.front_smoothing) ||
        (c.front_smoothing!=0.0 && (c.front_smoothing<0.03 || c.front_smoothing>0.15))) {
      std::cerr << "Front test invalid normal or smoothing" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (!std::isfinite(c.sigma) || c.sigma<0.03 || c.sigma>0.15 ||
      !std::isfinite(c.amplitude) || c.amplitude<=0.0 ||
      !std::isfinite(c.background) || c.background<=0.0 ||
      !std::isfinite(c.rho) || c.rho<=0.0 || !std::isfinite(c.bx) ||
      !std::isfinite(c.by) || !std::isfinite(c.bz)) {
    std::cerr << "Gaussian test invalid finite-positive parameters" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  config=c;
  if (!separate_stage_probe) {
    transport_stages::Reset(pin,pm,c.background,c.background+c.amplitude+1.e-12);
  }
  if (restart) return;
  auto ind=pm->mb_indcs;
  auto sizes=pack->pmb->mb_size;
  auto u=pack->pmhd->u0;
  auto b=pack->pmhd->b0;
  Real gm1=pack->pmhd->peos->eos_data.gamma-1.0;
  par_for("transport_gaussian_init",DevExeSpace(),0,pack->nmb_thispack-1,
          ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    auto size=sizes.d_view(m);
    Real x=CellCenterX(i-ind.is,ind.nx1,size.x1min,size.x1max);
    Real y=CellCenterX(j-ind.js,ind.nx2,size.x2min,size.x2max);
    Real z=CellCenterX(k-ind.ks,ind.nx3,size.x3min,size.x3max);
    u(m,IDN,k,j,i)=c.rho;
    u(m,IM1,k,j,i)=u(m,IM2,k,j,i)=u(m,IM3,k,j,i)=0.0;
    u(m,IEN,k,j,i)=c.rho*Initial(c,x,y,z,size.dx1,size.dx2,size.dx3)/gm1+
                    0.5*(c.bx*c.bx+c.by*c.by+c.bz*c.bz);
    b.x1f(m,k,j,i)=c.bx; b.x2f(m,k,j,i)=c.by; b.x3f(m,k,j,i)=c.bz;
    if (i==ind.ie) b.x1f(m,k,j,i+1)=c.bx;
    if (j==ind.je) b.x2f(m,k,j+1,i)=c.by;
    if (k==ind.ke) b.x3f(m,k+1,j,i)=c.bz;
  });
}
} // namespace transport_gaussian
#endif // PGEN_TESTS_TRANSPORT_GAUSSIAN_HPP_
