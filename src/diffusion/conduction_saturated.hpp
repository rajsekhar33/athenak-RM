#ifndef DIFFUSION_CONDUCTION_SATURATED_HPP_
#define DIFFUSION_CONDUCTION_SATURATED_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction_saturated.hpp
//! \brief Shared face reconstruction for explicit, whole-vector saturated conduction.

#include "athena.hpp"
#include "diffusion/conduction_closure.hpp"

namespace conduction {

struct SaturatedFaceParameters {
  Real gm1, alpha_iso, alpha_aniso, kappa0, phi;
  int ndim;
  bool spitzer_aligned, mc;
};

struct SaturatedFaceResult {
  Real normal_flux;
  Real donor_speed;
  bool valid;
};

// Harmonic Spitzer reconstructs theta gradients themselves. Fail closed if a
// nonzero gradient is not representable, rather than silently losing a flux
// that multiplication by a large face conductivity could otherwise restore.
KOKKOS_INLINE_FUNCTION
Real SpitzerTemperatureSlope(const Real tl, const Real tr, const Real dx) {
  if (!IsFiniteNonNegative(tl) || !IsFiniteNonNegative(tr) ||
      !IsFinitePositive(dx)) return InvalidClosureValue();
  const Real delta = tr-tl;
  const Real gradient = delta/dx;
  if (!IsFinite(gradient) || (delta != 0.0 && gradient == 0.0)) {
    return InvalidClosureValue();
  }
  return gradient;
}

// Robust signed product for the density-weighted projected classical vector.
KOKKOS_INLINE_FUNCTION
Real SaturatedProduct5(const Real a, const Real b, const Real c,
                       const Real d, const Real e) {
  const Real values[5] = {a,b,c,d,e};
  bool zero = false, negative = false, safe = true;
  Real product = 1.0;
  for (int n=0; n<5; ++n) {
    if (!IsFinite(values[n])) return InvalidClosureValue();
    zero = zero || values[n] == 0.0;
    negative = negative != (values[n] < 0.0);
    product *= fabs(values[n]);
    safe = safe && product >= kMinNormal && IsFinite(product);
  }
  if (zero) return 0.0;
  if (!safe) {
    Real exponent = 0.0;
    for (int n=0; n<5; ++n) exponent += log(fabs(values[n]));
    product = exp(exponent);
  }
  return negative ? -product : product;
}

// Mignone et al. (2012), Appendix A: arithmetic face density and donor
// pressure. Their RHS conduction vector has the OPPOSITE sign to our physical
// energy flux; our positive normal flux selects the LEFT pressure.
KOKKOS_INLINE_FUNCTION
Real SaturatedDonorFlux(const Real rho_face, const Real pressure, const Real phi) {
  if (!IsFinitePositive(rho_face) || !IsFiniteNonNegative(pressure) ||
      !IsFiniteNonNegative(phi)) return InvalidClosureValue();
  if (pressure == 0.0 || phi == 0.0) return 0.0;
  const Real ratio = pressure/rho_face;
  const Real prefactor = 5.0*phi;
  Real value = prefactor*pressure*sqrt(ratio);
  if (!(ratio >= kMinNormal && IsFinite(ratio) &&
        prefactor >= kMinNormal && IsFinite(prefactor) &&
        IsFinitePositive(value))) {
    value = exp(log(5.0)+log(phi)+1.5*log(pressure)-0.5*log(rho_face));
  }
  // A positive unrepresentable cap is not silently converted to no conduction.
  return IsFinitePositive(value) ? value : InvalidClosureValue();
}

// Face index names the cell on its right. Transverse theta slopes use the
// nested limiter; inactive-coordinate gradients are zero. Spitzer's harmonic
// endpoint conductivity multiplies the SAME limited theta gradient vector.
// Even isotropic saturation consumes transverse edge ghosts to form |C|.
// All three aligned vector components (including Bz in 2-D) enter that norm.
KOKKOS_INLINE_FUNCTION
SaturatedFaceResult SaturatedFace(const DvceArray5D<Real> &w,
    const DvceArray5D<Real> &bcc, const DvceArray4D<Real> &bf,
    const SaturatedFaceParameters &p, const Real dx[3], const int dir,
    const int m, const int k, const int j, const int i) {
  const SaturatedFaceResult invalid = {0.0,0.0,false};
  if (p.ndim < 1 || p.ndim > 3 || dir < 0 || dir >= p.ndim ||
      !IsFinitePositive(p.gm1) || !IsFiniteNonNegative(p.alpha_iso) ||
      !IsFiniteNonNegative(p.alpha_aniso) || !IsFiniteNonNegative(p.kappa0) ||
      !IsFiniteNonNegative(p.phi)) return invalid;
  for (int d=0; d<p.ndim; ++d) {
    if (!IsFinitePositive(dx[d])) return invalid;
  }
  const int il = i-(dir == 0), jl = j-(dir == 1), kl = k-(dir == 2);
  const Real rho_l = w(m,IDN,kl,jl,il), rho_r = w(m,IDN,k,j,i);
  const Real e_l = w(m,IEN,kl,jl,il), e_r = w(m,IEN,k,j,i);
  const Real tl = SpitzerTemperature(rho_l,e_l,p.gm1);
  const Real tr = SpitzerTemperature(rho_r,e_r,p.gm1);
  if (!IsFiniteNonNegative(tl) || !IsFiniteNonNegative(tr)) return invalid;
  const Real pl = SpitzerProduct3(p.gm1,e_l,1.0);
  const Real pr = SpitzerProduct3(p.gm1,e_r,1.0);
  if (!IsFiniteNonNegative(pl) || !IsFiniteNonNegative(pr) ||
      (e_l > 0.0 && pl == 0.0) || (e_r > 0.0 && pr == 0.0)) return invalid;
  const Real rho_a = 0.5*rho_l+0.5*rho_r;
  const Real rho_h = HarmonicMeanPositive(rho_l,rho_r);
  if (!IsFinitePositive(rho_a) || !IsFinitePositive(rho_h)) return invalid;
  const Real kappal = SpitzerConductivity(tl,p.kappa0);
  const Real kappar = SpitzerConductivity(tr,p.kappa0);
  if (!IsFiniteNonNegative(kappal) || !IsFiniteNonNegative(kappar) ||
      (p.kappa0 > 0.0 && tl > 0.0 && kappal == 0.0) ||
      (p.kappa0 > 0.0 && tr > 0.0 && kappar == 0.0)) return invalid;
  const Real kface = HarmonicMeanPositive(kappal,kappar);
  if (!IsFiniteNonNegative(kface)) return invalid;
  const bool aligned = p.alpha_aniso > 0.0 ||
                       (p.kappa0 > 0.0 && p.spitzer_aligned);
  Real b[3] = {0.0,0.0,0.0};
  if (aligned) {
    for (int d=0; d<3; ++d) {
      b[d] = d == dir ? bf(m,k,j,i) :
          0.5*bcc(m,d,k,j,i)+0.5*bcc(m,d,kl,jl,il);
      if (!IsFinite(b[d])) return invalid;
    }
    const Real scale = fmax(fabs(b[0]),fmax(fabs(b[1]),fabs(b[2])));
    if (scale > 0.0) {
      for (int d=0; d<3; ++d) b[d] /= scale;
      const Real norm = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
      for (int d=0; d<3; ++d) b[d] /= norm;
    }
  }
  Real gt[3] = {0.0,0.0,0.0};
  const bool conducting = p.alpha_iso > 0.0 || p.alpha_aniso > 0.0 || p.kappa0 > 0.0;
  if (conducting) {
    gt[dir] = p.kappa0 > 0.0 ? SpitzerTemperatureSlope(tl,tr,dx[dir]) :
                             (tr-tl)/dx[dir];
  }
  for (int d=0; d<p.ndim; ++d) {
    if (d == dir) continue;
    const int ti = d == 0, tj = d == 1, tk = d == 2;
    const Real rp = SpitzerTemperature(w(m,IDN,k+tk,j+tj,i+ti),
                                      w(m,IEN,k+tk,j+tj,i+ti),p.gm1);
    const Real rm = SpitzerTemperature(w(m,IDN,k-tk,j-tj,i-ti),
                                      w(m,IEN,k-tk,j-tj,i-ti),p.gm1);
    const Real lp = SpitzerTemperature(w(m,IDN,kl+tk,jl+tj,il+ti),
                                      w(m,IEN,kl+tk,jl+tj,il+ti),p.gm1);
    const Real lm = SpitzerTemperature(w(m,IDN,kl-tk,jl-tj,il-ti),
                                      w(m,IEN,kl-tk,jl-tj,il-ti),p.gm1);
    if (!IsFiniteNonNegative(rp) || !IsFiniteNonNegative(rm) ||
        !IsFiniteNonNegative(lp) || !IsFiniteNonNegative(lm)) return invalid;
    if (p.kappa0 > 0.0) {
      gt[d] = SpitzerLimited4(SpitzerTemperatureSlope(tr,rp,dx[d]),
          SpitzerTemperatureSlope(rm,tr,dx[d]),SpitzerTemperatureSlope(tl,lp,dx[d]),
          SpitzerTemperatureSlope(lm,tl,dx[d]),p.mc);
    } else if (conducting) {
      gt[d] = SpitzerLimited4((rp-tr)/dx[d],(tr-rm)/dx[d],
                             (lp-tl)/dx[d],(tl-lm)/dx[d],p.mc);
    }
  }
  for (int d=0; d<3; ++d) {
    if (!IsFinite(gt[d])) return invalid;
  }
  const Real st = fmax(fabs(gt[0]),fmax(fabs(gt[1]),fabs(gt[2])));
  Real dot_t = 0.0;
  for (int d=0; d<p.ndim; ++d) {
    if (st > 0.0) dot_t += b[d]*(gt[d]/st);
  }
  Real c[3];
  for (int d=0; d<3; ++d) {
    const Real iso = SpitzerProduct3(p.alpha_iso,rho_a,gt[d]);
    const Real aniso = SaturatedProduct5(p.alpha_aniso,rho_h,st,b[d],dot_t);
    const Real spitzer = p.spitzer_aligned ?
        SaturatedProduct5(kface,st,b[d],dot_t,1.0) : SpitzerProduct3(kface,gt[d],1.0);
    if (!IsFinite(iso) || !IsFinite(aniso) || !IsFinite(spitzer)) return invalid;
    const Real scale = fmax(fabs(iso),fmax(fabs(aniso),fabs(spitzer)));
    c[d] = scale == 0.0 ? 0.0 :
        -SpitzerProduct3(scale,iso/scale+aniso/scale+spitzer/scale,1.0);
    if (!IsFinite(c[d])) return invalid;
  }
  // Validate the entire consumed stencil even for phi=0 or an exactly zero B.
  const Real pressure = c[dir] > 0.0 ? pl : pr;
  const Real q = SaturatedDonorFlux(rho_a,pressure,p.phi);
  if (!IsFiniteNonNegative(q)) return invalid;
  const HeatFluxVector classical = {c[0],c[1],c[2]};
  const HeatFluxVector flux = SaturateHeatFlux(classical,q);
  const Real speed = SaturatedDonorSpeed(classical,dir,q,pressure,p.gm1);
  if (!ValidHeatFlux(flux) || !IsFiniteNonNegative(speed)) return invalid;
  const Real normal = dir == 0 ? flux.x : (dir == 1 ? flux.y : flux.z);
  return {normal,speed,true};
}

} // namespace conduction

#endif // DIFFUSION_CONDUCTION_SATURATED_HPP_
