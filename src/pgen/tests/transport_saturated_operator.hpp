#ifndef PGEN_TESTS_TRANSPORT_SATURATED_OPERATOR_HPP_
#define PGEN_TESTS_TRANSPORT_SATURATED_OPERATOR_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only production TOTAL conduction-flux exercise. This is not a PDE evolution.
// Host scratch states cover donor selection, mixed closures, full three-component
// vector suppression, weak gradients, extrema and an affine perpendicular null.
// Spitzer uses the harmonic full conductivity times limited temperature slopes.
// No Conduction object is copied.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "diffusion/conduction.hpp"

namespace transport_saturated_operator {

inline void Require(bool okay, const char *message) {
  if (!okay) {
    std::cerr << "saturated conduction operator FAILED: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// Independent host arithmetic: do not call production constitutive/limiter helpers.
inline long double Limit(long double a, long double b, bool mc) {
  if (a*b <= 0.0L) return 0.0L;
  if (!mc) return 2.0L*a*b/(a+b);
  const long double value = std::min(std::abs(0.5L*(a+b)),
                                    2.0L*std::min(std::abs(a),std::abs(b)));
  return a > 0.0L ? value : -value;
}

inline long double HarmonicSpitzer(long double left, long double right, long double k0) {
  if (k0 == 0.0L) return 0.0L;
  const long double kl=k0*std::pow(left,2.5L), kr=k0*std::pow(right,2.5L);
  return 2.0L*kl*kr/(kl+kr);
}

inline void Check(ParameterInput *pin, Mesh *pm, bool restart) {
  auto *pack = pm->pmb_pack;
  auto *mhd = pack->pmhd;
  auto &ind = pm->mb_indcs;
  Require(!restart && mhd != nullptr && global_variable::nranks == 1 &&
          pack->nmb_thispack == 1, "requires fresh single-rank single-block MHD");
  Require(ind.nx1 >= 4 && ind.nx1 <= 16 && ind.nx2 >= 4 && ind.nx2 <= 16 &&
          (ind.nx3 == 1 || (ind.nx3 >= 4 && ind.nx3 <= 16)),
          "requires 2D/3D block with 4 to 16 cells per active direction");
  Require(mhd->pcond != nullptr, "missing conduction object");
  auto *cond = mhd->pcond;
  const auto eos = mhd->peos->eos_data;
  Require(eos.is_ideal && eos.gamma > 1.0, "requires positive ideal-gas heat capacity");
  const bool saturated = pin->GetOrAddString("mhd","conduction_saturation","none")
                         == "cowie_mckee";
  const Real phi = saturated ? pin->GetOrAddReal("mhd","saturation_phi",0.3) : 0.0;
  const bool mc = pin->GetOrAddString("mhd","anisotropic_limiter","vanleer") == "mc";
  const Real gm1 = eos.gamma-1.0;
  const int ndim = pm->three_d ? 3 : 2;
  const int n1=mhd->w0.extent(4), n2=mhd->w0.extent(3), n3=mhd->w0.extent(2);
  const auto size = pack->pmb->mb_size.h_view(0);
  const Real dx[3] = {size.dx1,size.dx2,size.dx3};
  const Real field[3] = {pin->GetOrAddReal("problem","bx",1.0),
                         pin->GetOrAddReal("problem","by",2.0),
                         pin->GetOrAddReal("problem","bz",3.0)};
  const bool check_null=pin->GetOrAddBoolean("problem",
                                           "check_saturated_perpendicular_null",false);
  if (check_null) {
    Require(eos.gamma == 2.0 && field[0] == 1.0 && field[1] == 1.0 &&
            field[2] == 0.0 && ind.nx1 == 4 && ind.nx2 == 4 && ind.nx3 == 1 &&
            size.x1min == 0.0 && size.x2min == 0.0 && dx[0] == 0.25 && dx[1] == 0.25,
            "exact null requires gamma=2, B=(1,1,0), unit-box 4x4 cells");
    Require(cond->alpha_iso == 0.0 && cond->alpha_spitzer &&
            cond->spitzer_field_aligned && cond->spitzer_kappa0_code > 0.0 &&
            (!saturated || phi > 0.0),
            "null check requires active field-aligned Spitzer");
  }
  long double b[3] = {field[0],field[1],field[2]};
  const long double norm = std::sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
  if (norm > 0.0L) {
    for (int d=0; d<3; ++d) b[d] /= norm;
  }

  // Swap only magnetic view handles. Scratch primitive and flux arrays never
  // replace or modify the actual pgen's conserved/primitive state.
  auto saved_field = mhd->b0;
  auto saved_bcc = mhd->bcc0;
  mhd->b0 = DvceFaceFld4D<Real>("saturated_operator_b",1,n3,n2,n1);
  mhd->bcc0 = DvceArray5D<Real>("saturated_operator_bcc",1,3,n3,n2,n1);
  DvceArray5D<Real> w("saturated_operator_w",1,5,n3,n2,n1);
  DvceFaceFld5D<Real> flux("saturated_operator_flux",1,5,n3,n2,n1);
  auto wh = Kokkos::create_mirror_view(w);
  auto bh = Kokkos::create_mirror_view(mhd->bcc0);
  Kokkos::deep_copy(mhd->b0.x1f,field[0]);
  Kokkos::deep_copy(mhd->b0.x2f,field[1]);
  Kokkos::deep_copy(mhd->b0.x3f,field[2]);
  for (int k=0; k<n3; ++k) for (int j=0; j<n2; ++j) for (int i=0; i<n1; ++i) {
    for (int d=0; d<3; ++d) bh(0,d,k,j,i)=field[d];
  }
  Kokkos::deep_copy(mhd->bcc0,bh);
  const std::string filename=pin->GetString("job","basename")+".saturated_operator.dat";
  FILE *out=std::fopen(filename.c_str(),"w");
  Require(out != nullptr, "cannot open face diagnostics");
  std::fprintf(out,"# test dir k j i rhoL rhoR thetaL thetaR rhoA Pdonor qsat "
                   "class_x class_y class_z expected observed seed after error_scaled\n");
  std::fprintf(out,"# stencil=harmonic_full_conductivity_limited_temperature "
                   "states=7 affine_null_state=6 exact_null_check=%d\n",
                   check_null ? 1 : 0);
  Real maximum_error=0.0;
  int faces=0;
  for (int test=0; test<7; ++test) {
    for (int k=0; k<n3; ++k) for (int j=0; j<n2; ++j) for (int i=0; i<n1; ++i) {
      const Real x=size.x1min+(i-ind.is+0.5)*dx[0];
      const Real y=size.x2min+(j-ind.js+0.5)*dx[1];
      const Real z=ndim == 3 ? size.x3min+(k-ind.ks+0.5)*dx[2] : 0.0;
      const Real linear=0.1*x+0.2*y+0.3*z;
      Real rho=1.0, theta=2.0+linear;
      if (test == 1) { rho=x < 0.5 ? 1.0 : 4.0; theta=2.0-linear; }
      if (test == 2) theta=1.5+linear*linear;
      if (test == 3) {
        rho=1.0+0.1*std::cos(2.0*M_PI*x);
        theta=2.0+0.2*std::cos(2.0*M_PI*x)+0.15*std::sin(2.0*M_PI*y);
        if (ndim == 3) theta+=0.1*std::cos(2.0*M_PI*z);
      }
      if (test == 4) { rho=x < 0.5 ? 1.0 : 4.0; theta=2.0; }
      if (test == 5) theta=2.0+1.e-8*linear;
      if (test == 6) theta=2.0+0.25*(x-y);
      Require(rho > 0.0 && theta > 0.0, "invalid manufactured scratch state");
      wh(0,IDN,k,j,i)=rho;
      wh(0,IVX,k,j,i)=wh(0,IVY,k,j,i)=wh(0,IVZ,k,j,i)=0.0;
      wh(0,IEN,k,j,i)=rho*theta/gm1;
    }
    Kokkos::deep_copy(w,wh);
    // The null probe has zero seed so its zero test cannot hide a small flux
    // beneath the rounding of a much larger pre-existing energy flux.
    const bool null_probe=check_null && test == 6;
    Kokkos::deep_copy(flux.x1f,null_probe ? 0.0 : 0.5);
    Kokkos::deep_copy(flux.x2f,null_probe ? 0.0 : 0.625);
    Kokkos::deep_copy(flux.x3f,null_probe ? 0.0 : 0.75);
    cond->AddHeatFluxes(w,eos,flux);
    for (int dir=0; dir<ndim; ++dir) {
      auto f=dir == 0 ? flux.x1f : (dir == 1 ? flux.x2f : flux.x3f);
      auto fh=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),f);
      const int di=dir == 0, dj=dir == 1, dk=dir == 2;
      const Real seed=null_probe ? 0.0 : 0.5+0.125*dir;
      auto temp = [&](int k, int j, int i) -> long double {
        // Match the physical theta supplied by this Real-valued primitive state;
        // subsequent flux arithmetic is independent long-double host arithmetic.
        return gm1*(wh(0,IEN,k,j,i)/wh(0,IDN,k,j,i));
      };
      for (int k=ind.ks; k<=ind.ke+dk; ++k) {
        for (int j=ind.js; j<=ind.je+dj; ++j) for (int i=ind.is; i<=ind.ie+di; ++i) {
          const int il=i-di, jl=j-dj, kl=k-dk;
          const long double tl=temp(kl,jl,il), tr=temp(k,j,i);
          const long double rl=wh(0,IDN,kl,jl,il), rr=wh(0,IDN,k,j,i);
          const long double rhoa=0.5L*(rl+rr), rhoh=2.0L*rl*rr/(rl+rr);
          long double gt[3]={0.0L,0.0L,0.0L};
          const long double k0=cond->alpha_spitzer ? cond->spitzer_kappa0_code : 0.0L;
          const long double kh=HarmonicSpitzer(tl,tr,k0);
          gt[dir]=(tr-tl)/dx[dir];
          for (int d=0; d<ndim; ++d) {
            if (d == dir) continue;
            const int ti=d == 0, tj=d == 1, tk=d == 2;
            const long double rp=temp(k+tk,j+tj,i+ti), rm=temp(k-tk,j-tj,i-ti);
            const long double lp=temp(kl+tk,jl+tj,il+ti), lm=temp(kl-tk,jl-tj,il-ti);
            gt[d]=Limit(Limit((rp-tr)/dx[d],(tr-rm)/dx[d],mc),
                        Limit((lp-tl)/dx[d],(tl-lm)/dx[d],mc),mc);
          }
          long double bt=0.0L, classical[3];
          for (int d=0; d<3; ++d) bt+=b[d]*gt[d];
          for (int d=0; d<3; ++d) {
            classical[d]=-rhoa*cond->alpha_iso*gt[d]-rhoh*cond->alpha_aniso*b[d]*bt;
            classical[d]-=kh*(cond->spitzer_field_aligned ? b[d]*bt : gt[d]);
          }
          const long double magnitude=std::sqrt(classical[0]*classical[0]+
                          classical[1]*classical[1]+classical[2]*classical[2]);
          const long double donor=gm1*(classical[dir] > 0.0L ?
                                       wh(0,IEN,kl,jl,il) : wh(0,IEN,k,j,i));
          const long double qs =
              saturated ? 5.0L*phi*std::pow(donor,1.5L)/std::sqrt(rhoa) : 0.0L;
          const long double factor =
              !saturated || magnitude == 0.0L ? 1.0L : qs/(qs+magnitude);
          const Real expected=static_cast<Real>(factor*classical[dir]);
          const Real after=fh(0,IEN,k,j,i), observed=after-seed;
          const Real error=std::abs(observed-expected)/std::max(1.0,std::abs(expected));
          maximum_error=std::max(maximum_error,error);
          Require(std::isfinite(after) && error <= 2.e-11,
                  "production total face flux differs from independent oracle");
          if (null_probe) {
            Require(gt[0] == 0.25L && gt[1] == -0.25L && bt == 0.0L && kh > 0.0L,
                    "affine null oracle is not an active nonzero perpendicular gradient");
            Require(expected == 0.0 && observed == 0.0 && after == 0.0,
                    "affine perpendicular Spitzer null has nonzero raw face flux");
          }
          for (int v=0; v<5; ++v) if (v != IEN) {
            Require(fh(0,v,k,j,i) == seed, "nonenergy supplied flux was changed");
          }
          std::fprintf(out,"%d %d %d %d %d %.17e %.17e %.17e %.17e %.17e %.17e "
              "%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
              test,dir,k,j,i,static_cast<Real>(rl),static_cast<Real>(rr),
              static_cast<Real>(tl),static_cast<Real>(tr),static_cast<Real>(rhoa),
              static_cast<Real>(donor),static_cast<Real>(qs),
              static_cast<Real>(classical[0]),
              static_cast<Real>(classical[1]),static_cast<Real>(classical[2]),expected,
              observed,seed,after,error);
          ++faces;
        }
      }
    }
  }
  Require(std::fclose(out) == 0, "cannot close face diagnostics");
  Kokkos::fence();
  mhd->b0=saved_field;
  mhd->bcc0=saved_bcc;
  std::cout << "saturated conduction operator: 7 states, " << faces
            << " faces PASS; maximum scaled error=" << maximum_error << std::endl;
}
} // namespace transport_saturated_operator
#endif // PGEN_TESTS_TRANSPORT_SATURATED_OPERATOR_HPP_
