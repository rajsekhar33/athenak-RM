#ifndef PGEN_TESTS_TRANSPORT_HELICAL_HPP_
#define PGEN_TESTS_TRANSPORT_HELICAL_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only frozen-field 3D helix. No analytic evolution is applied to the solution.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/tests/transport_stages.hpp"

namespace transport_helical {
constexpr int nq=64;
inline int angular_mode, axial_mode, field_sign;
inline Real nodes[nq], weights[nq];

inline void Reject(const char *why) {
  std::cerr << "Helical test invalid configuration: " << why << std::endl;
  std::exit(EXIT_FAILURE);
}

// Host-only Gauss-Legendre initialization. Independent Python quadrature checks
// every xy cell, at both the initial and comparison time, at orders 64 and 96.
inline void Quadrature() {
  for (int j=0; j<nq; ++j) {
    Real z=std::cos(M_PI*(j+0.75)/(nq+0.5)), deriv=0.0;
    bool converged=false;
    for (int iteration=0; iteration<32; ++iteration) {
      Real p0=1.0, p1=z;
      for (int k=2; k<=nq; ++k) {
        Real pn=((2*k-1)*z*p1-(k-1)*p0)/k;
        p0=p1; p1=pn;
      }
      deriv=nq*(z*p1-p0)/(z*z-1.0);
      Real step=p1/deriv;
      z-=step;
      if (std::abs(step)<2.e-15) {
        converged=true;
        break;
      }
    }
    if (!converged) Reject("quadrature iteration");
    nodes[j]=z;
    weights[j]=2.0/((1.0-z*z)*deriv*deriv);
  }
}

// Periodic, C2-tapered potential. The transverse field is unchanged at r<.37,
// enclosing the entire thermal support (.17<r<.33), and zero at r>.45.
inline Real Az(Real x, Real y) {
  Real r=std::hypot(x-0.5,y-0.5), integral=0.5*r*r;
  if (r>0.37) {
    Real u=std::fmin((r-0.37)/0.08,1.0);
    Real u2=u*u, u4=u2*u2, u5=u4*u, u6=u5*u, u7=u6*u;
    Real j0=u-2.5*u4+3.0*u5-u6;
    Real j1=0.5*u2-2.0*u5+2.5*u6-6.0/7.0*u7;
    integral=0.5*0.37*0.37+0.37*0.08*j0+0.08*0.08*j1;
  }
  return -field_sign*2.0*M_PI*integral;
}

struct XY { Real cosine, sine; };
inline XY AverageXY(Real x, Real y, Real dx, Real dy) {
  XY out{0.0,0.0};
  for (int j=0; j<nq; ++j) {
    for (int i=0; i<nq; ++i) {
      Real X=x-0.5+0.5*dx*nodes[i], Y=y-0.5+0.5*dy*nodes[j];
      Real r=std::hypot(X,Y), s=(r-0.25)/0.08;
      if (std::abs(s)>=1.0) continue;
      Real g=std::exp(1.0-1.0/(1.0-s*s));
      Real phi=angular_mode*std::atan2(Y,X);
      Real weight=0.25*weights[i]*weights[j]*g;
      out.cosine+=weight*std::cos(phi);
      out.sine+=weight*std::sin(phi);
    }
  }
  return out;
}

inline Real Temperature(XY xy, Real z, Real dz) {
  Real k=2.0*M_PI*axial_mode;
  Real sinc=axial_mode==0 ? 1.0 : std::sin(0.5*k*dz)/(0.5*k*dz);
  return 1.0+0.1*sinc*(xy.cosine*std::cos(k*z)-xy.sine*std::sin(k*z));
}

inline void Initialize(ParameterInput *pin, Mesh *pm, bool restart) {
  auto *pack=pm->pmb_pack;
  auto s=pm->mesh_size;
  if (!pack->pmhd || !pm->three_d || pm->multilevel || global_variable::nranks!=1 ||
      !pack->pmhd->peos->eos_data.is_ideal ||
      pack->pmhd->peos->eos_data.gamma!=2.0 ||
      pin->GetString("time","evolution")!="kinematic" ||
      pin->GetString("time","integrator")!="rk2" ||
      pin->GetOrAddString("time","sts_integrator","none")!="none" ||
      s.x1min!=0.0 || s.x1max!=1.0 || s.x2min!=0.0 || s.x2max!=1.0 ||
      s.x3min!=0.0 || s.x3max!=1.0) Reject("unit-box 3D, one-rank ideal gamma=2 RK2");
  for (const char *key : {"ix1_bc","ox1_bc","ix2_bc","ox2_bc","ix3_bc","ox3_bc"}) {
    if (pin->GetString("mesh",key)!="periodic") Reject("periodic boundaries");
  }
  angular_mode=pin->GetOrAddInteger("problem","angular_mode",1);
  axial_mode=pin->GetOrAddInteger("problem","axial_mode",1);
  field_sign=pin->GetOrAddInteger("problem","field_sign",1);
  if (angular_mode<0 || angular_mode>1 || axial_mode<-1 || axial_mode>1 ||
      (angular_mode==0 && axial_mode==0) || std::abs(field_sign)!=1) Reject("mode/sign");
  Real alpha=pin->GetOrAddReal("mhd","alpha_aniso",0.0);
  if (!std::isfinite(alpha) || alpha<0.0 ||
      pin->GetOrAddReal("mhd","alpha_iso",0.0)!=0.0) Reject("anisotropic only");
  Quadrature();
  transport_stages::Reset(pin,pm,0.9,1.1);
  if (restart) return;
  auto ind=pm->mb_indcs;
  auto sizes=pack->pmb->mb_size.h_view;
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->u0);
  auto bx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x1f);
  auto by=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x2f);
  auto bz=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x3f);
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto size=sizes(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        Real x=size.x1min+(i-ind.is)*size.dx1;
        Real y=size.x2min+(j-ind.js)*size.dx2;
        Real bxl=(Az(x,y+size.dx2)-Az(x,y))/size.dx2;
        Real bxr=(Az(x+size.dx1,y+size.dx2)-Az(x+size.dx1,y))/size.dx2;
        Real byl=-(Az(x+size.dx1,y)-Az(x,y))/size.dx1;
        Real byr=-(Az(x+size.dx1,y+size.dx2)-Az(x,y+size.dx2))/size.dx1;
        XY xy=AverageXY(x+0.5*size.dx1,y+0.5*size.dx2,size.dx1,size.dx2);
        Real magnetic=0.125*(SQR(bxl+bxr)+SQR(byl+byr))+0.5;
        for (int k=ind.ks; k<=ind.ke; ++k) {
          Real z=size.x3min+(k-ind.ks+0.5)*size.dx3;
          u(m,IDN,k,j,i)=1.0;
          u(m,IM1,k,j,i)=u(m,IM2,k,j,i)=u(m,IM3,k,j,i)=0.0;
          u(m,IEN,k,j,i)=Temperature(xy,z,size.dx3)+magnetic;
          bx(m,k,j,i)=bxl; bx(m,k,j,i+1)=bxr;
          by(m,k,j,i)=byl; by(m,k,j+1,i)=byr;
          bz(m,k,j,i)=bz(m,k+1,j,i)=field_sign;
        }
      }
    }
  }
  Kokkos::deep_copy(pack->pmhd->u0,u);
  Kokkos::deep_copy(pack->pmhd->b0.x1f,bx);
  Kokkos::deep_copy(pack->pmhd->b0.x2f,by);
  Kokkos::deep_copy(pack->pmhd->b0.x3f,bz);
}

inline void Finalize(ParameterInput *pin, Mesh *pm) {
  transport_stages::Write(pin,pm);
  auto *pack=pm->pmb_pack;
  auto u=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->u0);
  auto bx=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x1f);
  auto by=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x2f);
  auto bz=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x3f);
  auto ind=pm->mb_indcs;
  auto sizes=pack->pmb->mb_size.h_view;
  std::string base=pin->GetString("job","basename")+".helical";
  FILE *file=std::fopen((base+".dat").c_str(),"w");
  if (!file) Reject("output file");
  std::fprintf(file,"# x y z volume rho temperature initial_temperature energy divb "
                    "bxlo bxhi bylo byhi bzlo bzhi\n");
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto s=sizes(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        Real x=s.x1min+(i-ind.is+0.5)*s.dx1;
        Real y=s.x2min+(j-ind.js+0.5)*s.dx2;
        XY xy=AverageXY(x,y,s.dx1,s.dx2);
        for (int k=ind.ks; k<=ind.ke; ++k) {
          Real z=s.x3min+(k-ind.ks+0.5)*s.dx3;
          Real faces[6]={bx(m,k,j,i),bx(m,k,j,i+1),by(m,k,j,i),by(m,k,j+1,i),
                         bz(m,k,j,i),bz(m,k+1,j,i)};
          Real mag=0.125*(SQR(faces[0]+faces[1])+SQR(faces[2]+faces[3])+
                           SQR(faces[4]+faces[5]));
          Real rho=u(m,IDN,k,j,i);
          Real kin=0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+
                        SQR(u(m,IM3,k,j,i)))/rho;
          Real temp=(u(m,IEN,k,j,i)-kin-mag)/rho;
          Real div=(faces[1]-faces[0])/s.dx1+(faces[3]-faces[2])/s.dx2+
                   (faces[5]-faces[4])/s.dx3;
          std::fprintf(file,"%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e",
                       x,y,z,s.dx1*s.dx2*s.dx3,rho,temp,Temperature(xy,z,s.dx3),
                       u(m,IEN,k,j,i),div);
          for (Real v : faces) std::fprintf(file," %.17e",v);
          std::fprintf(file,"\n");
        }
      }
    }
  }
  if (std::fclose(file)!=0) Reject("output close");
  file=std::fopen((base+".summary").c_str(),"w");
  if (!file) Reject("summary file");
  std::fprintf(file,"# time cycles\n%.17e %d\n",pm->time,
               pm->ncycle-transport_stages::first_cycle);
  if (std::fclose(file)!=0) Reject("summary close");
}
} // namespace transport_helical
#endif // PGEN_TESTS_TRANSPORT_HELICAL_HPP_
