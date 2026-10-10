#ifndef PGEN_TESTS_TRANSPORT_PATCH_HPP_
#define PGEN_TESTS_TRANSPORT_PATCH_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only helpers for transport_mode.cpp.
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen/tests/transport_stages.hpp"
#include "pgen/tests/transport_ring_state.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace transport_patch {
struct PatchConfig {
  int kind;
  Real cold, hot, bx, by, sign, field_radius2;
};
inline PatchConfig patch_config;
inline Real patch_min, patch_max;
inline int patch_samples;
inline bool ring_diagnostic=false;
inline Real ring_last_time, ring_last_lap, ring_initial_heat, ring_lap_integral;
inline int ring_stride;
inline FILE *ring_history=nullptr;
inline bool ring_checkpoint=false;
inline std::string ring_fixture, ring_state_output;
inline transport_ring_restart::State ring_state;

inline void RingFailure(const std::string &message) {
  std::cerr << "ring checkpoint diagnostic: " << message << std::endl;
  std::exit(EXIT_FAILURE);
}

// Compare actual physical settings, not a caller-provided hash. Continuation
// controls (nlim, tlim, ndiag, output cadence and history stride) are excluded.
inline std::string RingFixture(ParameterInput *pin, Mesh *pm) {
  std::ostringstream key;
  key.imbue(std::locale::classic());
  key << std::hexfloat << "ring-v1|" << pm->mb_indcs.nx1 << '|' << pm->mb_indcs.nx2;
  auto c=patch_config;
  auto s=pm->mesh_size;
  for (Real v : {c.cold,c.hot,c.sign,c.field_radius2,s.x1min,s.x1max,s.x2min,
                 s.x2max,s.x3min,s.x3max,pin->GetReal("mhd","gamma"),
                 pin->GetReal("mhd","alpha_iso"),pin->GetReal("mhd","alpha_aniso"),
                 pin->GetReal("time","cfl_number"),pin->GetReal("mhd","dfloor"),
                 pin->GetReal("mhd","pfloor"),pin->GetReal("mhd","tfloor")}) {
    key << '|' << v;
  }
  for (const char *name : {"ix1_bc","ox1_bc","ix2_bc","ox2_bc"}) {
    key << '|' << pin->GetString("mesh",name);
  }
  for (const char *name : {"eos","anisotropic_limiter","reconstruct","rsolver",
                           "conductivity_integrator"}) {
    key << '|' << pin->GetString("mhd",name);
  }
  key << '|' << pin->GetString("time","integrator") << '|'
      << pin->GetString("time","sts_integrator");
  return key.str();
}

inline void WriteRingState() {
  // Final outputs have already written the fluid restart. A hard-link publish
  // is atomic on the same filesystem and refuses to overwrite an existing file.
  std::string temporary=ring_state_output+".tmp";
  FILE *file=std::fopen(temporary.c_str(),"wx");
  if (!file) RingFailure("cannot create temporary sidecar");
  auto text=ring_state.Encode();
  bool failed=std::fwrite(text.data(),1,text.size(),file)!=text.size();
  failed=std::fclose(file)!=0 || failed;
  if (failed || ::link(temporary.c_str(),ring_state_output.c_str())!=0) {
    RingFailure("sidecar write/publish failed (existing output is not overwritten)");
  }
  if (::unlink(temporary.c_str())!=0) RingFailure("cannot remove sidecar temporary link");
}

KOKKOS_INLINE_FUNCTION
Real PatchTemperature(const PatchConfig &c, Real x, Real y) {
  Real r = sqrt(x*x+y*y);
  if (c.kind == 0) {
    return (x>0.0 && x<0.5 && y>0.0 && y<0.5) ? c.hot : c.cold;
  }
  if (c.kind == 1) {
    // Sector centered on the negative x axis, angular width pi/6.
    bool hot = r>0.5 && r<0.7 && x<0.0 && fabs(y)<(-x)*tan(M_PI/12.0);
    return hot ? c.hot : c.cold;
  }
  return c.cold+(c.hot-c.cold)*exp(-SQR((r-0.6)/0.12));
}

KOKKOS_INLINE_FUNCTION
Real PatchPotential(const PatchConfig &c, Real x, Real y) {
  // Compact circular potential, r=0.9 by default. Its discrete curl is
  // solenoidal even where the cutoff intersects a cell. Radius 1 supports
  // the Sharma-Hammett late-time ring comparison without changing defaults.
  return 0.5*fmax(0.0,c.field_radius2-x*x-y*y);
}

KOKKOS_INLINE_FUNCTION
Real PatchBx(const PatchConfig &c, Real xface, Real ylo, Real yhi, Real dy) {
  return c.kind == 0 ? c.bx :
      c.sign*(PatchPotential(c,xface,yhi)-PatchPotential(c,xface,ylo))/dy;
}

KOKKOS_INLINE_FUNCTION
Real PatchBy(const PatchConfig &c, Real xlo, Real xhi, Real yface, Real dx) {
  return c.kind == 0 ? c.by :
      -c.sign*(PatchPotential(c,xhi,yface)-PatchPotential(c,xlo,yface))/dx;
}

template <class U, class B>
KOKKOS_INLINE_FUNCTION
Real RingTemperature(const U &u, const B &b, Real gm1, int k, int j, int i) {
  Real rho=u(0,IDN,k,j,i);
  Real bx=0.5*(b.x1f(0,k,j,i)+b.x1f(0,k,j,i+1));
  Real by=0.5*(b.x2f(0,k,j,i)+b.x2f(0,k,j+1,i));
  Real bz=0.5*(b.x3f(0,k,j,i)+b.x3f(0,k+1,j,i));
  Real kin=0.5*(SQR(u(0,IM1,k,j,i))+SQR(u(0,IM2,k,j,i))+
                SQR(u(0,IM3,k,j,i)))/rho;
  return gm1*(u(0,IEN,k,j,i)-kin-0.5*(bx*bx+by*by+bz*bz))/rho;
}

// Eq. (39) diagnostic: volume-integrated temperature change within the annulus
// divided by its time-integrated Laplacian. Every completed step contributes to
// trapezoidal quadrature; only the history output is decimated. Restart remains
// opt-in and requires a checked sidecar; fresh legacy tests are unchanged.
inline void SampleRing(Mesh *pm, Real time, Real tmin, Real tmax,
                       bool initial=false, bool final=false, bool restored=false) {
  if (!ring_diagnostic) return;
  auto *pack=pm->pmb_pack;
  auto w=pack->pmhd->w0;
  auto u=pack->pmhd->u0;
  auto b=pack->pmhd->b0;
  auto sizes=pack->pmb->mb_size;
  auto ind=pm->mb_indcs;
  auto c=patch_config;
  Real gm1=pack->pmhd->peos->eos_data.gamma-1.0;
  if (restored) {
    // The restart constructor restored u0/b0, but primitives and ghost cells
    // are not initialized yet. Sample only interior cells for endpoint extrema.
    int invalid;
    Kokkos::parallel_reduce("ring_restart_extrema",Kokkos::RangePolicy<DevExeSpace>(
        0,ind.nx1*ind.nx2),
    KOKKOS_LAMBDA(const int idx, Real &lo, Real &hi, int &bad) {
      int i=idx%ind.nx1+ind.is, j=idx/ind.nx1+ind.js;
      Real temperature=RingTemperature(u,b,gm1,ind.ks,j,i);
      if (!(u(0,IDN,ind.ks,j,i)>0.0) || !(temperature==temperature) ||
          fabs(temperature)>1.e100) ++bad;
      lo=fmin(lo,temperature); hi=fmax(hi,temperature);
    },Kokkos::Min<Real>(tmin),Kokkos::Max<Real>(tmax),Kokkos::Sum<int>(invalid));
    if (invalid || !std::isfinite(tmin) || !std::isfinite(tmax) ||
        tmin<c.cold-1.e-10 || tmax>c.hot+1.e-10) {
      RingFailure("restored temperature bounds failed");
    }
  }
  Real heat, lap;
  Kokkos::parallel_reduce("ring_integral",Kokkos::RangePolicy<DevExeSpace>(
      0,ind.nx1*ind.nx2),
  KOKKOS_LAMBDA(const int idx, Real &h, Real &l) {
    int i=idx%ind.nx1+ind.is, j=idx/ind.nx1+ind.js, k=ind.ks;
    auto s=sizes.d_view(0);
    Real x=CellCenterX(i-ind.is,ind.nx1,s.x1min,s.x1max);
    Real y=CellCenterX(j-ind.js,ind.nx2,s.x2min,s.x2max);
    if (x*x+y*y <= 0.25 || x*x+y*y >= 0.49) return;
    Real t[5];
    for (int q=0; q<5; ++q) {
      int di=(q==1)-(q==2), dj=(q==3)-(q==4);
      if (restored) {
        t[q]=RingTemperature(u,b,gm1,k,j+dj,i+di);
      } else {
        t[q]=initial ? PatchTemperature(c,x+di*s.dx1,y+dj*s.dx2) :
            gm1*w(0,IEN,k,j+dj,i+di)/w(0,IDN,k,j+dj,i+di);
      }
    }
    Real volume=s.dx1*s.dx2*s.dx3;
    h+=volume*t[0];
    l+=volume*((t[1]+t[2]-2*t[0])/(s.dx1*s.dx1)+
               (t[3]+t[4]-2*t[0])/(s.dx2*s.dx2));
  },Kokkos::Sum<Real>(heat),Kokkos::Sum<Real>(lap));
  if (!std::isfinite(heat) || !std::isfinite(lap) ||
      (!initial && !restored && time<ring_last_time)) {
    std::cerr << "invalid ring time-integral diagnostic" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ring_checkpoint) {
    try {
      if (initial) {
        ring_state={ring_fixture,pm->ncycle,time,heat,lap,0.0,heat,tmin,tmax};
        ring_state.Validate();
      } else if (restored) {
        if (std::abs(heat-ring_state.heat)>1.e-10 ||
            std::abs(lap-ring_state.laplacian)>1.e-10) {
          RingFailure("restored conserved endpoint does not match sidecar");
        }
        // Preserve the exact previous quadrature endpoint, not a re-reduction.
        heat=ring_state.heat; lap=ring_state.laplacian;
      } else {
        ring_state.Sample(final ? pm->ncycle : pm->ncycle+1,time,heat,lap,tmin,tmax);
      }
    } catch (const std::exception &error) { RingFailure(error.what()); }
    ring_initial_heat=ring_state.initial_heat;
    ring_lap_integral=ring_state.integral;
  } else if (initial) {
    ring_initial_heat=heat;
    ring_lap_integral=0.0;
  } else {
    ring_lap_integral+=0.5*(time-ring_last_time)*(lap+ring_last_lap);
  }
  ring_last_time=time; ring_last_lap=lap;
  if (initial || restored || final || pm->ncycle%ring_stride == 0) {
    std::fprintf(ring_history,"%.17e %.17e %.17e %.17e %.17e %.17e\n",
                 time,heat,lap,ring_lap_integral,tmin,tmax);
  }
  if (final) {
    if (std::fclose(ring_history) != 0) std::exit(EXIT_FAILURE);
    ring_history=nullptr;
    if (ring_checkpoint) WriteRingState();
  }
}

inline void MonitorPatchAt(Mesh *pm, bool final) {
  auto *pack = pm->pmb_pack;
  auto u = pack->pmhd->u0;
  auto b = pack->pmhd->bcc0;
  auto &ind = pm->mb_indcs;
  int ni=ind.nx1, nj=ind.nx2, is=ind.is, js=ind.js, ks=ind.ks;
  Real gm1 = pack->pmhd->peos->eos_data.gamma-1.0;
  Real tmin, tmax;
  int invalid;
  Kokkos::parallel_reduce("transport_patch_monitor",
      Kokkos::RangePolicy<DevExeSpace>(0,pack->nmb_thispack*ni*nj),
  KOKKOS_LAMBDA(const int idx, Real &lo, Real &hi, int &bad) {
    int m=idx/(ni*nj), j=(idx/ni)%nj+js, i=idx%ni+is;
    Real rho = u(m,IDN,ks,j,i);
    Real kin = 0.5*(SQR(u(m,IM1,ks,j,i))+SQR(u(m,IM2,ks,j,i))+
                    SQR(u(m,IM3,ks,j,i)))/rho;
    Real mag = 0.5*(SQR(b(m,IBX,ks,j,i))+SQR(b(m,IBY,ks,j,i))+
                    SQR(b(m,IBZ,ks,j,i)));
    Real t = gm1*(u(m,IEN,ks,j,i)-kin-mag)/rho;
    if (!(rho>0.0) || !(t==t) || fabs(t)>1.e100) ++bad;
    lo = fmin(lo,t);
    hi = fmax(hi,t);
  }, Kokkos::Min<Real>(tmin), Kokkos::Max<Real>(tmax), Kokkos::Sum<int>(invalid));
  invalid += pm->ecounter.neos_dfloor+pm->ecounter.neos_efloor+
             pm->ecounter.neos_tfloor+pm->ecounter.neos_fail;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&tmin,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&tmax,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&invalid,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  patch_min = std::fmin(patch_min,tmin);
  patch_max = std::fmax(patch_max,tmax);
  ++patch_samples;
  if (invalid || tmin<patch_config.cold-1.e-10 || tmax>patch_config.hot+1.e-10) {
    std::cerr << "transport patch extrema/floor check FAILED: " << tmin << " "
              << tmax << " events=" << invalid << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // WorkInLoop runs after RK but BEFORE Driver increments pm->time.
  SampleRing(pm,final ? pm->time : pm->time+pm->dt,tmin,tmax,false,final);
}

inline void MonitorPatch(Mesh *pm) {
  MonitorPatchAt(pm,false);
}

inline void FinalizePatch(ParameterInput *pin, Mesh *pm) {
  transport_stages::Write(pin,pm);
  MonitorPatchAt(pm,true);
  auto *pack = pm->pmb_pack;
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->u0);
  auto bx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x1f);
  auto by = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),pack->pmhd->b0.x2f);
  auto &ind = pm->mb_indcs;
  auto sizes = pack->pmb->mb_size.h_view;
  Real gm1 = pack->pmhd->peos->eos_data.gamma-1.0;
  std::string base = pin->GetString("job","basename")+".patch."+
                     std::to_string(global_variable::my_rank);
  FILE *f = std::fopen((base+".dat").c_str(),"w");
  if (f == nullptr) std::exit(EXIT_FAILURE);
  std::fprintf(f,"# x y volume rho temperature initial_temperature energy divb\n");
  for (int m=0; m<pack->nmb_thispack; ++m) {
    auto s = sizes(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        int k=ind.ks;
        Real x=CellCenterX(i-ind.is,ind.nx1,s.x1min,s.x1max);
        Real y=CellCenterX(j-ind.js,ind.nx2,s.x2min,s.x2max);
        Real b1=0.5*(bx(m,k,j,i)+bx(m,k,j,i+1));
        Real b2=0.5*(by(m,k,j,i)+by(m,k,j+1,i));
        Real rho=u(m,IDN,k,j,i);
        Real kin=0.5*(SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+
                       SQR(u(m,IM3,k,j,i)))/rho;
        Real temp=gm1*(u(m,IEN,k,j,i)-kin-0.5*(b1*b1+b2*b2))/rho;
        Real divb=(bx(m,k,j,i+1)-bx(m,k,j,i))/s.dx1+
                   (by(m,k,j+1,i)-by(m,k,j,i))/s.dx2;
        std::fprintf(f,"%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
                     x,y,s.dx1*s.dx2*s.dx3,rho,temp,
                     PatchTemperature(patch_config,x,y),u(m,IEN,k,j,i),divb);
      }
    }
  }
  if (std::fclose(f) != 0) std::exit(EXIT_FAILURE);
  if (global_variable::my_rank == 0) {
    f=std::fopen((pin->GetString("job","basename")+".patch.summary").c_str(),"w");
    if (f == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(f,"# time cycle_samples minimum maximum\n%.17e %d %.17e %.17e\n",
                 pm->time,patch_samples,patch_min,patch_max);
    if (std::fclose(f) != 0) std::exit(EXIT_FAILURE);
  }
}

inline void InitializePatch(ParameterInput *pin, Mesh *pm, bool restart) {
  auto *pack = pm->pmb_pack;
  std::string profile=pin->GetString("problem","profile");
  if (pack->pmhd == nullptr || !pm->two_d || pm->multilevel ||
      pin->GetString("time","evolution") != "kinematic" ||
      !pack->pmhd->peos->eos_data.is_ideal ||
      pin->GetOrAddString("time","sts_integrator","none") != "none" ||
      (profile != "square" && profile != "ring" && profile != "radial")) {
    std::cerr << "transport patch requires uniform-level 2D kinematic ideal MHD, "
              << "explicit transport, and square/ring/radial profile" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  PatchConfig c;
  c.kind=profile == "square" ? 0 : (profile == "ring" ? 1 : 2);
  c.cold=pin->GetOrAddReal("problem","cold",0.1);
  c.hot=pin->GetOrAddReal("problem","hot",10.0);
  c.bx=pin->GetOrAddReal("problem","bx",1.0);
  c.by=pin->GetOrAddReal("problem","by",-1.0);
  c.sign=pin->GetOrAddReal("problem","field_sign",1.0);
  Real field_radius=pin->GetOrAddReal("problem","field_radius",0.9);
  c.field_radius2=field_radius*field_radius;
  if (!(c.cold>0.0) || !(c.hot>c.cold) || !std::isfinite(c.hot) ||
      !std::isfinite(c.bx) || !std::isfinite(c.by) || fabs(c.sign) != 1.0 ||
      !(field_radius>0.0) || !std::isfinite(c.field_radius2)) {
    std::cerr << "transport patch invalid initial parameters" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  patch_config=c;
  ring_diagnostic=pin->GetOrAddBoolean("problem","ring_diagnostic",false);
  ring_stride=pin->GetOrAddInteger("problem","ring_history_stride",1000);
  ring_checkpoint=pin->GetOrAddBoolean("problem","ring_checkpoint",false);
  if (ring_checkpoint && !ring_diagnostic) RingFailure("checkpoint requires diagnostic");
  if (ring_diagnostic) {
    if ((restart && !ring_checkpoint) || c.kind != 1 || global_variable::nranks != 1 ||
        pack->nmb_thispack != 1 || ring_stride<1) {
      std::cerr << "ring diagnostic requires fresh single-block single-rank ring"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (ring_checkpoint) {
      // This diagnostic must not read uninitialized restart ghost primitives.
      auto ind=pm->mb_indcs;
      auto s=pm->mesh_size;
      if (sizeof(Real)!=sizeof(double) || ind.nx1<16 || ind.nx2<16 ||
          s.x1min!=-1.0 || s.x1max!=1.0 || s.x2min!=-1.0 || s.x2max!=1.0) {
        RingFailure("checkpoint requires double precision and interior annular stencil");
      }
      ring_fixture=RingFixture(pin,pm);
      ring_state_output=pin->GetString("job","basename")+".ring.state";
      if (restart) {
        std::string path=pin->GetOrAddString("problem","ring_restart_state","");
        std::ifstream input(path);
        if (!input) RingFailure("missing restart sidecar");
        std::ostringstream bytes;
        bytes << input.rdbuf();
        try {
          ring_state=transport_ring_restart::Decode(bytes.str(),ring_fixture,
                                                   pm->ncycle,pm->time);
        } catch (const std::exception &error) { RingFailure(error.what()); }
      }
    }
    std::string name=pin->GetString("job","basename")+".ring.history";
    ring_history=std::fopen(name.c_str(),"w");
    if (ring_history == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(ring_history,"# time annular_heat laplacian time_integral min max\n");
    SampleRing(pm,pm->time,c.cold,c.hot,!restart,false,restart);
  }
  patch_min=std::numeric_limits<Real>::max();
  patch_max=-std::numeric_limits<Real>::max();
  patch_samples=0;
  transport_stages::Reset(pin,pm,c.cold,c.hot);
  if (restart) return;
  auto &ind=pm->mb_indcs;
  auto sizes=pack->pmb->mb_size;
  auto u=pack->pmhd->u0;
  auto b=pack->pmhd->b0;
  int is=ind.is, js=ind.js, nx=ind.nx1, ny=ind.nx2;
  Real gm1=pack->pmhd->peos->eos_data.gamma-1.0;
  par_for("transport_patch_init",DevExeSpace(),0,pack->nmb_thispack-1,
          ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    auto s=sizes.d_view(m);
    Real xl=LeftEdgeX(i-is,nx,s.x1min,s.x1max);
    Real xr=LeftEdgeX(i-is+1,nx,s.x1min,s.x1max);
    Real yl=LeftEdgeX(j-js,ny,s.x2min,s.x2max);
    Real yr=LeftEdgeX(j-js+1,ny,s.x2min,s.x2max);
    Real bxl=PatchBx(c,xl,yl,yr,s.dx2), bxr=PatchBx(c,xr,yl,yr,s.dx2);
    Real byl=PatchBy(c,xl,xr,yl,s.dx1), byr=PatchBy(c,xl,xr,yr,s.dx1);
    b.x1f(m,k,j,i)=bxl;
    b.x2f(m,k,j,i)=byl;
    b.x3f(m,k,j,i)=b.x3f(m,k+1,j,i)=0.0;
    if (i == is+nx-1) b.x1f(m,k,j,i+1)=bxr;
    if (j == js+ny-1) b.x2f(m,k,j+1,i)=byr;
    u(m,IDN,k,j,i)=1.0;
    u(m,IM1,k,j,i)=u(m,IM2,k,j,i)=u(m,IM3,k,j,i)=0.0;
    u(m,IEN,k,j,i)=PatchTemperature(c,0.5*(xl+xr),0.5*(yl+yr))/gm1+
                    0.125*(SQR(bxl+bxr)+SQR(byl+byr));
  });
}
} // namespace transport_patch
#endif // PGEN_TESTS_TRANSPORT_PATCH_HPP_
