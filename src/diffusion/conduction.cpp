//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction.cpp
//! \brief Implements functions for Conduction class. This includes isotropic thermal
//! conduction, in which heat flux is proportional to negative local temperature gradient.
//! Conduction may be added to Hydro and/or MHD independently.

#include <float.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_hyd.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "conduction.hpp"
#include "units/units.hpp"

namespace {

// Constitutive arithmetic and face reconstruction are private to this module.
namespace conduction {

// Evaluate standard-library constants on the host at compile time. Device
// functions use scalar values without calling host-only numeric_limits members.
inline constexpr Real kMaxFinite = std::numeric_limits<Real>::max();
inline constexpr Real kMinNormal = std::numeric_limits<Real>::min();
inline constexpr Real kQuietNaN = std::numeric_limits<Real>::quiet_NaN();

KOKKOS_INLINE_FUNCTION
bool IsFiniteNonNegative(const Real value) {
  return value >= 0.0 && value <= kMaxFinite;
}

KOKKOS_INLINE_FUNCTION
bool IsFinitePositive(const Real value) {
  return value > 0.0 && value <= kMaxFinite;
}

KOKKOS_INLINE_FUNCTION
bool IsFinite(const Real value) {
  return IsFiniteNonNegative(fabs(value));
}

KOKKOS_INLINE_FUNCTION
bool ValidThermalState(const Real rho, const Real theta) {
  return IsFinitePositive(rho) && IsFiniteNonNegative(theta);
}

KOKKOS_INLINE_FUNCTION
Real InvalidClosureValue() {
  return kQuietNaN;
}

// Interpolate conductivity itself, not a separately averaged diffusivity and
// density. Multiplication after division avoids overflow for a == b == max().
KOKKOS_INLINE_FUNCTION
Real HarmonicMeanPositive(const Real a, const Real b) {
  if (!IsFiniteNonNegative(a) || !IsFiniteNonNegative(b)) {
    return InvalidClosureValue();
  }
  const Real lo = fmin(a, b), hi = fmax(a, b);
  if (lo == 0.0) return 0.0;
  return lo*(2.0/(1.0 + lo/hi));
}

// Pure T^(5/2) Spitzer law. kappa0_code includes ALL unit conversions.
// A logarithmic fallback avoids intermediate overflow/underflow when the final
// coefficient remains representable. A genuinely overflowing result stays inf.
KOKKOS_INLINE_FUNCTION
Real SpitzerConductivity(const Real theta, const Real kappa0_code) {
  if (!IsFiniteNonNegative(theta) || !IsFiniteNonNegative(kappa0_code)) {
    return InvalidClosureValue();
  }
  if (theta == 0.0 || kappa0_code == 0.0) return 0.0;
  const Real power = pow(theta, 2.5);
  const Real value = kappa0_code*power;
  if (power >= kMinNormal && IsFinitePositive(value)) return value;
  return exp(log(kappa0_code) + 2.5*log(theta));
}

// Recover theta without requiring either gm1*eint or eint/rho to be
// representable. Positive energy whose theta underflows to zero is rejected;
// exactly zero energy remains a valid mathematical constitutive limit.
KOKKOS_INLINE_FUNCTION
Real SpitzerTemperature(const Real rho, const Real eint, const Real gm1) {
  if (!IsFinitePositive(rho) || !IsFiniteNonNegative(eint) ||
      !IsFinitePositive(gm1)) return InvalidClosureValue();
  if (eint == 0.0) return 0.0;
  const Real ratio = eint/rho;
  Real theta = gm1*ratio;
  if (ratio >= kMinNormal && IsFinite(ratio) && IsFinitePositive(theta)) return theta;
  theta = exp(log(gm1)+log(eint)-log(rho));
  return IsFinitePositive(theta) ? theta : InvalidClosureValue();
}

// Mathematically the existing VL/MC limiters, evaluated without product/sum
// overflow for Spitzer temperature slopes. Legacy constant-alpha arithmetic is
// deliberately not changed. Validate all arguments before sign/zero exits.
KOKKOS_INLINE_FUNCTION
Real SpitzerVL2(const Real a, const Real b) {
  if (!IsFinite(a) || !IsFinite(b)) return InvalidClosureValue();
  if (!((a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0))) return 0.0;
  const Real lo = fmin(fabs(a),fabs(b)), hi = fmax(fabs(a),fabs(b));
  const Real limited = lo*(2.0/(1.0+lo/hi));
  return a > 0.0 ? limited : -limited;
}

KOKKOS_INLINE_FUNCTION
Real SpitzerMC2(const Real a, const Real b) {
  if (!IsFinite(a) || !IsFinite(b)) return InvalidClosureValue();
  if (!((a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0))) return 0.0;
  const Real lo = fmin(fabs(a),fabs(b)), hi = fmax(fabs(a),fabs(b));
  const Real mean = hi <= 0.5*kMaxFinite ? 0.5*(lo+hi) : 0.5*lo+0.5*hi;
  // If lo>max/2, 2*lo exceeds every representable mean and cannot limit it.
  const Real limited = lo <= 0.5*kMaxFinite ? fmin(mean,2.0*lo) : mean;
  return a > 0.0 ? limited : -limited;
}

KOKKOS_INLINE_FUNCTION
Real SpitzerLimited4(const Real a, const Real b, const Real c, const Real d,
                     const bool mc) {
  return mc ? SpitzerMC2(SpitzerMC2(a,b),SpitzerMC2(c,d)) :
              SpitzerVL2(SpitzerVL2(a,b),SpitzerVL2(c,d));
}

// Signed product used to rescale the normalized gradient projection. A
// logarithmic fallback recovers representable products after intermediate
// overflow/underflow; neither it nor the projection is guaranteed correctly rounded.
KOKKOS_INLINE_FUNCTION
Real SpitzerProduct3(const Real a, const Real b, const Real c) {
  if (!IsFinite(a) || !IsFinite(b) || !IsFinite(c)) return InvalidClosureValue();
  if (a == 0.0 || b == 0.0 || c == 0.0) return 0.0;
  const Real aa = fabs(a), ab = fabs(b), ac = fabs(c);
  const Real pair = aa*ab;
  Real magnitude = pair*ac;
  if (!(pair >= kMinNormal && IsFinite(pair) && IsFinitePositive(magnitude))) {
    magnitude = exp(log(aa)+log(ab)+log(ac));
  }
  const bool negative = ((a < 0.0) != (b < 0.0)) != (c < 0.0);
  return negative ? -magnitude : magnitude;
}

// Cowie-McKee convention used by Vaidya et al. (2017), equations (2)-(4):
// q_sat = 5 phi rho c_iso^3, with c_iso^2 = theta, NOT gamma*theta.
KOKKOS_INLINE_FUNCTION
Real CowieMcKeeFlux(const Real rho, const Real theta, const Real phi) {
  if (!ValidThermalState(rho, theta) || !IsFiniteNonNegative(phi)) {
    return InvalidClosureValue();
  }
  if (theta == 0.0 || phi == 0.0) return 0.0;
  const Real prefactor = 5.0*phi*rho;
  const Real power = theta*sqrt(theta);
  const Real value = prefactor*power;
  if (prefactor >= kMinNormal && power >= kMinNormal && IsFinitePositive(value))
    return value;
  return exp(log(5.0) + log(phi) + log(rho) + 1.5*log(theta));
}

struct HeatFluxVector {
  Real x, y, z;
};

KOKKOS_INLINE_FUNCTION
bool ValidHeatFlux(const HeatFluxVector &flux) {
  return IsFiniteNonNegative(fabs(flux.x)) &&
         IsFiniteNonNegative(fabs(flux.y)) &&
         IsFiniteNonNegative(fabs(flux.z));
}

// Limit the vector magnitude once, preserving its direction. The large-flux
// branch also avoids underflow of a suppression factor followed by loss of a
// representable saturated flux (e.g. F=1e300 and q_sat=1e-300 in double precision).
KOKKOS_INLINE_FUNCTION
HeatFluxVector SaturateHeatFlux(const HeatFluxVector &flux, const Real q_sat) {
  if (!ValidHeatFlux(flux) || !IsFiniteNonNegative(q_sat)) {
    const Real invalid = InvalidClosureValue();
    return {invalid, invalid, invalid};
  }
  const Real scale = fmax(fabs(flux.x), fmax(fabs(flux.y), fabs(flux.z)));
  if (scale == 0.0 || q_sat == 0.0) return {0.0, 0.0, 0.0};
  const Real sx = flux.x/scale, sy = flux.y/scale, sz = flux.z/scale;
  const Real norm_scaled = sqrt(sx*sx + sy*sy + sz*sz);
  if (scale >= q_sat) {
    const Real denominator = norm_scaled + q_sat/scale;
    const Real amplitude = q_sat/denominator;
    const Real component[3] = {flux.x, flux.y, flux.z};
    const Real normalized[3] = {sx, sy, sz};
    Real result[3] = {sx*amplitude, sy*amplitude, sz*amplitude};
    for (int d=0; d<3; ++d) {
      // Normalized components can underflow even when the final flux does
      // not. Preserve the ordinary arithmetic unless normalization loses
      // relative precision; split exponents only for this rare fallback.
      if (component[d] != 0.0 && fabs(normalized[d]) < kMinNormal) {
        int ec, eq, es;
        const Real mc = frexp(component[d], &ec);
        const Real mq = frexp(q_sat, &eq);
        const Real ms = frexp(scale, &es);
        result[d] = ldexp((mc*mq/ms)/denominator, ec+eq-es);
      }
    }
    return {result[0], result[1], result[2]};
  }
  const Real denominator = 1.0 + (scale/q_sat)*norm_scaled;
  return {flux.x/denominator, flux.y/denominator, flux.z/denominator};
}

// Extra donor-pressure derivative at frozen density for whole-vector saturation:
// a = 1.5*gm1*(q/P)*[r/(q+r)]^2*abs(C_normal)/r.
// Retain a separate unsuppressed diffusion budget. Scaled/log evaluation avoids
// forming r, q/P, or a suppression factor whose intermediate value overflows
// or underflows even though the final speed is representable.
KOKKOS_INLINE_FUNCTION
Real SaturatedDonorSpeed(const HeatFluxVector &classical, const int dir,
                         const Real q, const Real pressure, const Real gm1) {
  if (!ValidHeatFlux(classical) || dir < 0 || dir > 2 ||
      !IsFiniteNonNegative(q) || !IsFiniteNonNegative(pressure) ||
      !IsFinitePositive(gm1)) return InvalidClosureValue();
  const Real c[3] = {classical.x,classical.y,classical.z};
  const Real scale = fmax(fabs(c[0]),fmax(fabs(c[1]),fabs(c[2])));
  if (scale == 0.0 || c[dir] == 0.0 || q == 0.0) return 0.0;
  if (pressure == 0.0) return InvalidClosureValue();
  const Real x = c[0]/scale, y = c[1]/scale, z = c[2]/scale;
  const Real norm = sqrt(x*x+y*y+z*z);
  const Real log_r = log(scale)+log(norm);
  const Real log_t = scale >= q ? -log1p((q/scale)/norm) :
      log_r-log(q)-log1p((scale/q)*norm);
  const Real exponent = log(1.5)+log(gm1)+log(q)-log(pressure)+
                        2.0*log_t+log(fabs(c[dir]))-log_r;
  const Real speed = exp(exponent);
  return IsFinitePositive(speed) ? speed : InvalidClosureValue();
}

// Host-side dimensional conversion. Each member is the physical cgs scale of
// one code unit. A units block is unnecessary when kappa0_code is supplied.
struct ConductivityUnits {
  Real pressure_cgs;
  Real velocity_cgs;
  Real length_cgs;
  Real temperature_cgs;

  bool IsValid() const {
    return IsFinitePositive(pressure_cgs) && IsFinitePositive(velocity_cgs) &&
           IsFinitePositive(length_cgs) && IsFinitePositive(temperature_cgs);
  }
};

// kappa_phys = kappa0_cgs*T_kelvin^(5/2), where kappa0_cgs has units
// erg s^-1 cm^-1 K^-7/2. Thus K_code = kappa0_code*theta^(5/2) and
// kappa0_code = kappa0_cgs*T_unit^(7/2)/(P_unit*V_unit*L_unit).
inline Real SpitzerCoefficientCode(const Real kappa0_cgs,
                                   const ConductivityUnits &units) {
  if (!IsFiniteNonNegative(kappa0_cgs) || !units.IsValid()) {
    return InvalidClosureValue();
  }
  if (kappa0_cgs == 0.0) return 0.0;
  const long double exponent = std::log(static_cast<long double>(kappa0_cgs)) +
      3.5L*std::log(static_cast<long double>(units.temperature_cgs)) -
      std::log(static_cast<long double>(units.pressure_cgs)) -
      std::log(static_cast<long double>(units.velocity_cgs)) -
      std::log(static_cast<long double>(units.length_cgs));
  return static_cast<Real>(std::exp(exponent));
}

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


parabolic::DiffusionSelection ParseConductivityIntegrator(const std::string &block,
                                                           ParameterInput *pin) {
  std::string integrator =
      pin->GetOrAddString(block, "conductivity_integrator", "explicit");
  if (integrator == "explicit") {
    return parabolic::DiffusionSelection::explicit_only;
  }
  if (integrator == "sts") {
    return parabolic::DiffusionSelection::sts_only;
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<" << block << ">/conductivity_integrator = '" << integrator
            << "' must be 'explicit' or 'sts'" << std::endl;
  std::exit(EXIT_FAILURE);
}

} // namespace

// VanLeer Limiter which takes 2 slopes
KOKKOS_INLINE_FUNCTION
Real VLL2State(const Real a, const Real b) {
  if (a*b > 0) {
    return 2.0*a*b/(a+b);
  } else {
    return 0.0;
  }
}

// VanLeer Limiter which takes 4 slopes
KOKKOS_INLINE_FUNCTION
Real VLL4State(const Real a, const Real b, const Real c, const Real d) {
  return VLL2State(VLL2State(a,b), VLL2State(c,d));
}

// Monotonized-central limiter: Sharma & Hammett (2007), equation 18.
// Keep the van-Leer default available for matched numerical-diffusion controls.
KOKKOS_INLINE_FUNCTION
Real MCL2State(const Real a, const Real b) {
  if ((a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0)) {
    Real limited = fmin(0.5*fabs(a)+0.5*fabs(b),2.0*fmin(fabs(a),fabs(b)));
    return a > 0.0 ? limited : -limited;
  }
  return 0.0;
}

//----------------------------------------------------------------------------------------
//! \brief Conduction constructor
// Note first argument passes string ("hydro" or "mhd") denoting in wihch class this
// object is being constructed, and therefore which <block> in the input file from which
// the parameters are read.
// Note that the coefficients of thermal conduction, alpha_iso, etc., correspond to
// diffusivities. The conductivity is kappa = (dens)*alpha, and the energy flux
// q = -kappa * (dT/dx) = - alpha * d * *dT/dx)

Conduction::Conduction(std::string block, MeshBlockPack *pp, ParameterInput *pin) :
    pmy_pack(pp) {
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  dt_transport = dtnew;
  // Read parameters for thermal diffusivity (if any)
  alpha_iso = pin->GetOrAddReal(block,"alpha_iso", 0.0);
  alpha_aniso = pin->GetOrAddReal(block,"alpha_aniso", 0.0);
  alpha_spitzer = pin->GetOrAddBoolean(block,"alpha_spitzer", false);
  std::string limiter = pin->GetOrAddString(block,"anisotropic_limiter","vanleer");
  if (limiter != "vanleer" && limiter != "mc") {
    std::cerr << "anisotropic_limiter must be vanleer or mc" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  use_mc_limiter = limiter == "mc";
  if (!std::isfinite(alpha_iso) || !std::isfinite(alpha_aniso) ||
      alpha_iso < 0.0 || alpha_aniso < 0.0 || (alpha_aniso > 0.0 && block != "mhd")) {
    std::cerr << "Conduction requires finite nonnegative alpha coefficients; "
              << "anisotropic conduction requires MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string saturation =
      pin->GetOrAddString(block, "conduction_saturation", "none");
  if (saturation != "none" && saturation != "cowie_mckee") {
    std::cerr << "conduction_saturation must be none or cowie_mckee" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  saturated = saturation == "cowie_mckee";
  if (!saturated && pin->DoesParameterExist(block, "saturation_phi")) {
    std::cerr << "saturation_phi requires conduction_saturation=cowie_mckee"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (saturated) {
    saturation_phi = pin->GetOrAddReal(block, "saturation_phi", 0.3);
    if (!std::isfinite(saturation_phi) || saturation_phi < 0.0) {
      std::cerr << "saturation_phi must be finite and nonnegative" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // Preserve the legacy disabled sentinel. Physical saturation is selected
  // above and depends on the local gas state, not an absolute flux ceiling.
  q_limit = pin->GetOrAddReal(block,"q_limit",
                     static_cast<Real>(std::numeric_limits<float>::max()));
  // Saturation is a separate closure, not an implicit conductivity ceiling.
  // GetOrAddReal serializes legacy default values with six significant digits.
  // Accept its rounded float-max sentinel as well, including on restart.
  if (!std::isfinite(q_limit) || q_limit < static_cast<Real>(3.40282e38)) {
    std::cerr << "q_limit must be finite and at least the legacy disabled "
              << "sentinel (3.40282e38); select physical saturation with "
              << "conduction_saturation=cowie_mckee" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  mode = ParseConductivityIntegrator(block, pin);
  const bool has_code = pin->DoesParameterExist(block, "spitzer_kappa0_code");
  const bool has_cgs = pin->DoesParameterExist(block, "spitzer_kappa0_cgs");
  const bool has_geometry = pin->DoesParameterExist(block, "spitzer_geometry");
  if (!alpha_spitzer && (has_code || has_cgs || has_geometry)) {
    std::cerr << "Spitzer parameters require alpha_spitzer=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (alpha_spitzer) {
    if (has_code == has_cgs) {
      std::cerr << "Spitzer requires exactly one of spitzer_kappa0_code and "
                << "spitzer_kappa0_cgs" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (has_code) {
      spitzer_kappa0_code = pin->GetReal(block, "spitzer_kappa0_code");
    } else {
      if (pmy_pack->punit == nullptr) {
        std::cerr << "spitzer_kappa0_cgs requires an explicit units block" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      auto *pu = pmy_pack->punit;
      conduction::ConductivityUnits units{pu->pressure_cgs(), pu->velocity_cgs(),
                                          pu->length_cgs(), pu->temperature_cgs()};
      const Real k0_cgs = pin->GetReal(block, "spitzer_kappa0_cgs");
      spitzer_kappa0_code = conduction::SpitzerCoefficientCode(k0_cgs, units);
      if (k0_cgs > 0.0 && spitzer_kappa0_code == 0.0) {
        std::cerr << "Spitzer coefficient conversion underflows code precision"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Do not add the derived code coefficient to ParameterInput: restart
      // must retain exactly the originally selected coefficient convention.
    }
    if (!std::isfinite(spitzer_kappa0_code) || spitzer_kappa0_code < 0.0) {
      std::cerr << "Spitzer coefficient must be finite and nonnegative" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (block == "mhd" && !has_geometry) {
      std::cerr << "MHD Spitzer requires explicit spitzer_geometry selection"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const std::string geometry =
        pin->GetOrAddString(block, "spitzer_geometry", "isotropic");
    if (geometry != "isotropic" && geometry != "field_aligned") {
      std::cerr << "spitzer_geometry must be isotropic or field_aligned" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    spitzer_field_aligned = geometry == "field_aligned";
    if (spitzer_field_aligned && block != "mhd") {
      std::cerr << "Field-aligned Spitzer conduction requires MHD" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (alpha_spitzer || saturated) {
    const Real cfl = pin->GetReal("time", "cfl_number");
    if (!std::isfinite(cfl) || cfl <= 0.0 || cfl > 1.0) {
      std::cerr << "Spitzer/saturation requires finite 0 < cfl_number <= 1"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const std::string rk = pin->GetOrAddString("time", "integrator", "rk2");
    if (rk != "rk1" && rk != "rk2" && rk != "rk3") {
      std::cerr << "Spitzer/saturation requires SSP rk1, rk2 or rk3 for the "
                << "non-STS part of the update" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // This energy-flux closure is Newtonian, not a relativistic heat-flux model.
  if ((alpha_aniso > 0.0 || alpha_spitzer || saturated) &&
      (pmy_pack->pcoord->is_special_relativistic ||
      pmy_pack->pcoord->is_general_relativistic ||
      pmy_pack->pcoord->is_dynamical_relativistic)) {
    std::cerr << "Anisotropic/nonlinear conduction requires Newtonian physics"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (mode == parabolic::DiffusionSelection::sts_only &&
      !(alpha_iso > 0.0 || alpha_aniso > 0.0 || spitzer_kappa0_code > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "STS requires positive thermal conduction"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \brief Conduction destructor

Conduction::~Conduction() {
}

// Coordinate failures before any rank enters the next exchange. In particular,
// a bad face on one rank must not leave its neighbors waiting for flux messages.
void Conduction::FailIfInvalid(int local_count, const char *where) const {
  int invalid = local_count;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (invalid != 0) {
    std::cerr << "### FATAL ERROR: nonlinear conduction: " << where << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }
}

// Inspect raw conserved states before EOS conversion can repair them. Include
// the consumed one-cell stencil halo, but not unused triple-corner ghost cells.
// EOS conversion is performed only on local copies, with the current CT field.
void Conduction::ValidateRawState(const DvceArray5D<Real> &u0, const EOS_Data &eos_data,
                                  const DvceFaceFld4D<Real> *b) const {
  if (!saturated && (!alpha_spitzer || spitzer_kappa0_code == 0.0)) return;
  const auto state = InspectRawState(u0, eos_data, b);
  FailIfInvalid(state.invalid,
                "raw conserved state is invalid or would require an EOS floor");
}

Conduction::ThermalState Conduction::InspectRawState(const DvceArray5D<Real> &u0,
    const EOS_Data &eos_data, const DvceFaceFld4D<Real> *b, bool active_only) const {
  FailIfInvalid(!eos_data.is_ideal || !std::isfinite(eos_data.gamma) ||
                eos_data.gamma <= 1.0, "invalid ideal-gas EOS");
  const auto &idx = pmy_pack->pmesh->mb_indcs;
  const bool multi = pmy_pack->pmesh->multi_d;
  const bool three = pmy_pack->pmesh->three_d;
  const bool edges = saturated || spitzer_field_aligned || alpha_aniso > 0.0;
  const int is = idx.is, ie = idx.ie, js = idx.js, je = idx.je;
  const int ks = idx.ks, ke = idx.ke;
  const int ni = idx.nx1+2, nj = idx.nx2+(multi ? 2 : 0);
  const int nk = idx.nx3+(three ? 2 : 0);
  const int nji = nj*ni, nkji = nk*nji;
  const bool mhd = b != nullptr;
  DvceArray4D<Real> b1, b2, b3;
  if (mhd) { b1 = b->x1f; b2 = b->x2f; b3 = b->x3f; }
  const auto eos = eos_data;
  int invalid = 0;
  Real tmin = std::numeric_limits<Real>::max(), tmax = 0.0;
  Kokkos::parallel_reduce("spitzer_raw_state", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*nkji),
  KOKKOS_LAMBDA(const int n, int &bad, Real &minimum, Real &maximum) {
    const int m = n/nkji, rem = n-m*nkji;
    const int k = rem/nji + ks-(three ? 1 : 0);
    const int j = (rem%nji)/ni + js-(multi ? 1 : 0);
    const int i = rem%ni + is-1;
    const int outside = (i < is || i > ie)+(j < js || j > je)+(k < ks || k > ke);
    if (outside > (active_only ? 0 : (edges ? 2 : 1))) return;
    HydCons1D u;
    u.d = u0(m,IDN,k,j,i);
    u.mx = u0(m,IM1,k,j,i); u.my = u0(m,IM2,k,j,i); u.mz = u0(m,IM3,k,j,i);
    u.e = u0(m,IEN,k,j,i);
    if (!conduction::IsFinitePositive(u.d) || !conduction::IsFinite(u.e) ||
        !conduction::IsFinite(u.mx) || !conduction::IsFinite(u.my) ||
        !conduction::IsFinite(u.mz)) { ++bad; return; }
    bool df = false, ef = false, tf = false;
    HydPrim1D w;
    if (mhd) {
      MHDCons1D um;
      um.d = u.d; um.mx = u.mx; um.my = u.my; um.mz = u.mz; um.e = u.e;
      const Real bx0 = b1(m,k,j,i), bx1 = b1(m,k,j,i+1);
      const Real by0 = b2(m,k,j,i), by1 = b2(m,k,j+1,i);
      const Real bz0 = b3(m,k,j,i), bz1 = b3(m,k+1,j,i);
      if (!conduction::IsFinite(bx0) || !conduction::IsFinite(bx1) ||
          !conduction::IsFinite(by0) || !conduction::IsFinite(by1) ||
          !conduction::IsFinite(bz0) || !conduction::IsFinite(bz1)) { ++bad; return; }
      um.bx = 0.5*bx0+0.5*bx1; um.by = 0.5*by0+0.5*by1; um.bz = 0.5*bz0+0.5*bz1;
      SingleC2P_IdealMHD(um, eos, w, df, ef, tf);
    } else {
      SingleC2P_IdealHyd(u, eos, w, df, ef, tf);
    }
    const Real temp = conduction::SpitzerTemperature(u.d, w.e, eos.gamma-1.0);
    if (df || ef || tf || !conduction::IsFiniteNonNegative(w.e) ||
        !conduction::IsFinite(w.vx) || !conduction::IsFinite(w.vy) ||
        !conduction::IsFinite(w.vz) || !conduction::IsFiniteNonNegative(temp)) {
      ++bad;
      return;
    }
    minimum = fmin(minimum, temp);
    maximum = fmax(maximum, temp);
  }, Kokkos::Sum<int>(invalid), Kokkos::Min<Real>(tmin), Kokkos::Max<Real>(tmax));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  return ThermalState{invalid, tmin, tmax};
}

bool Conduction::UsesNonlinearSTS() const {
  return mode == parabolic::DiffusionSelection::sts_only &&
         (saturated || (alpha_spitzer && spitzer_kappa0_code > 0.0));
}

// Evaluate the existing timestep stencil on the current raw stage, without
// repairing live U/W or changing EOS counters. A joint diffusion sweep may heat
// the gas above its initial temperature. Its raw-state/budget guard replaces
// the conduction-only maximum principle, while the saturation donor cap remains.
Real Conduction::RawStageTimeStep(const DvceArray5D<Real> &u0,
    const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
    const DvceFaceFld4D<Real> *b, bool include_ghosts) {
  const int nm = u0.extent_int(0), nk = u0.extent_int(2);
  const int nj = u0.extent_int(3), ni = u0.extent_int(4);
  if (sts_stage_w.extent_int(0) != nm || sts_stage_w.extent_int(2) != nk ||
      sts_stage_w.extent_int(3) != nj || sts_stage_w.extent_int(4) != ni) {
    sts_stage_w = DvceArray5D<Real>("conduction_stage_primitives", nm,
                                   w0.extent_int(1), nk, nj, ni);
  }
  // Before exchange, ghost U and interface B belong to different stages. Keep
  // their last valid primitive stencil; after exchange, inspect raw ghosts too.
  // Every new flux evaluation follows the refreshed-ghost check.
  if (!include_ghosts) Kokkos::deep_copy(DevExeSpace(), sts_stage_w, w0);
  const auto wstage = sts_stage_w;
  const auto eos = eos_data;
  const auto idx = pmy_pack->pmesh->mb_indcs;
  const bool mhd = b != nullptr;
  DvceArray4D<Real> b1, b2, b3;
  if (mhd) { b1 = b->x1f; b2 = b->x2f; b3 = b->x3f; }
  const int nji = nj*ni, ncell = nk*nji;
  // AMR reserves extra array slots for future MeshBlocks. They have no current
  // fluid state; keep the allocated scratch shape but inspect active blocks only.
  const int active_blocks = pmy_pack->nmb_thispack;
  int invalid = 0;
  Kokkos::parallel_reduce("conduction_raw_stage_primitives", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, active_blocks*ncell),
  KOKKOS_LAMBDA(const int n, int &bad) {
    const int m = n/ncell, cell = n-m*ncell;
    const int k = cell/nji, j = (cell%nji)/ni, i = cell%ni;
    if (!include_ghosts && (i < idx.is || i > idx.ie || j < idx.js || j > idx.je ||
                           k < idx.ks || k > idx.ke)) return;
    HydCons1D u;
    u.d = u0(m,IDN,k,j,i); u.e = u0(m,IEN,k,j,i);
    u.mx = u0(m,IM1,k,j,i); u.my = u0(m,IM2,k,j,i); u.mz = u0(m,IM3,k,j,i);
    bool df = false, ef = false, tf = false;
    HydPrim1D primitive;
    if (mhd) {
      MHDCons1D um;
      um.d = u.d; um.e = u.e; um.mx = u.mx; um.my = u.my; um.mz = u.mz;
      um.bx = 0.5*(b1(m,k,j,i)+b1(m,k,j,i+1));
      um.by = 0.5*(b2(m,k,j,i)+b2(m,k,j+1,i));
      um.bz = 0.5*(b3(m,k,j,i)+b3(m,k+1,j,i));
      SingleC2P_IdealMHD(um, eos, primitive, df, ef, tf);
    } else {
      SingleC2P_IdealHyd(u, eos, primitive, df, ef, tf);
    }
    if (df || ef || tf || !conduction::IsFinitePositive(primitive.d) ||
        !conduction::IsFinitePositive(primitive.e) ||
        !conduction::IsFinite(primitive.vx) || !conduction::IsFinite(primitive.vy) ||
        !conduction::IsFinite(primitive.vz)) {
      ++bad;
      return;
    }
    wstage(m,IDN,k,j,i) = primitive.d;
    wstage(m,IEN,k,j,i) = primitive.e;
    wstage(m,IVX,k,j,i) = primitive.vx;
    wstage(m,IVY,k,j,i) = primitive.vy;
    wstage(m,IVZ,k,j,i) = primitive.vz;
  }, Kokkos::Sum<int>(invalid));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (invalid != 0) return 0.0;
  // The unsuppressed diffusion budget is STS-accelerated. Check the separate
  // saturation donor bound from current primitives before each flux evaluation.
  return SpitzerTimeStep(wstage, eos_data, false, true);
}

void Conduction::BeginSTSTrial(const DvceArray5D<Real> &u,
    const DvceArray5D<Real> &w, DvceArray5D<Real> &snapshot) {
  // AMR may resize the owner arrays between completed cycles.
  bool resize = false;
  for (int d=0; d<5; ++d) resize |= sts_w0.extent(d) != w.extent(d);
  if (resize) Kokkos::realloc(sts_w0, w.extent(0), w.extent(1), w.extent(2),
                            w.extent(3), w.extent(4));
  Kokkos::deep_copy(DevExeSpace(), snapshot, u);
  Kokkos::deep_copy(DevExeSpace(), sts_w0, w);
}

void Conduction::RestoreSTSTrial(DvceArray5D<Real> &u, DvceArray5D<Real> &w,
    const DvceArray5D<Real> &snapshot) {
  Kokkos::deep_copy(DevExeSpace(), u, snapshot);
  Kokkos::deep_copy(DevExeSpace(), w, sts_w0);
}

bool Conduction::CheckSTSTransport(const DvceArray5D<Real> &w,
    const EOS_Data &eos_data, Real interval) const {
  if (!saturated) return true;
  Real bound = SaturationTimeStep(w, eos_data, true);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &bound, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  bound *= pmy_pack->pmesh->cfl_no;
  return std::isfinite(interval) && interval > 0.0 &&
         interval <= bound*(1.0+64.0*std::numeric_limits<Real>::epsilon());
}

// Positive harmonic face conductivities are <= K(Tmax). For scalar isotropic
// forward Euler this bounds the current-state convex-combination row sum;
// derivatives of the temperature-dependent coefficients are not needed for
// that argument. Check every SSP stage, not just the start of the RK step.
// The factor four per dimension for limited field-aligned transverse terms is
// deliberately conservative, not a general nonlinear stability theorem.
Real Conduction::SpitzerTimeStep(const DvceArray5D<Real> &w0,
                                 const EOS_Data &eos_data,
                                 bool include_saturation, bool retryable) const {
  auto reject = [&](int local_count, const char *reason) {
    if (!retryable) {
      FailIfInvalid(local_count, reason);
      return false;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &local_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    return local_count != 0;
  };
  const auto &idx = pmy_pack->pmesh->mb_indcs;
  const bool multi = pmy_pack->pmesh->multi_d, three = pmy_pack->pmesh->three_d;
  const int ndim = three ? 3 : (multi ? 2 : 1);
  const bool edges = saturated || spitzer_field_aligned || alpha_aniso > 0.0;
  const int is = idx.is, ie = idx.ie, js = idx.js, je = idx.je;
  const int ks = idx.ks, ke = idx.ke;
  const int ni = idx.nx1+2, nj = idx.nx2+(multi ? 2 : 0);
  const int nk = idx.nx3+(three ? 2 : 0);
  const int nji = nj*ni, nkji = nk*nji;
  const Real gm1 = eos_data.gamma-1.0;
  int invalid = 0;
  Real tmax = 0.0;
  Kokkos::parallel_reduce("spitzer_temperature_cap", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*nkji),
  KOKKOS_LAMBDA(const int n, Real &maximum, int &bad) {
    const int m = n/nkji, rem = n-m*nkji;
    const int k = rem/nji + ks-(three ? 1 : 0);
    const int j = (rem%nji)/ni + js-(multi ? 1 : 0);
    const int i = rem%ni + is-1;
    const int outside = (i < is || i > ie)+(j < js || j > je)+(k < ks || k > ke);
    if (outside > (edges ? 2 : 1)) return;
    const Real temp = conduction::SpitzerTemperature(
        w0(m,IDN,k,j,i), w0(m,IEN,k,j,i), gm1);
    if (!conduction::IsFiniteNonNegative(temp)) { ++bad; return; }
    maximum = fmax(maximum, temp);
  }, Kokkos::Max<Real>(tmax), Kokkos::Sum<int>(invalid));
  if (reject(invalid, "invalid thermal state in timestep stencil")) return 0.0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  const Real kcap = conduction::SpitzerConductivity(tmax, spitzer_kappa0_code);
  // Positive conductivity underflow is not a physical zero coefficient.
  // Reject it consistently with the face operator instead of dropping dt.
  if (reject(!std::isfinite(kcap) || kcap < 0.0 ||
                (spitzer_kappa0_code > 0.0 && tmax > 0.0 && kcap == 0.0),
                "unrepresentable conductivity cap")) return 0.0;
  // Whole-vector saturation couples transverse gradients even for isotropic
  // transport. Retain an unsuppressed diffusion budget, not only a cap speed.
  const Real alpha0 = saturated ? 4.0*ndim*(alpha_iso+alpha_aniso) :
                                  alpha_iso+4.0*ndim*alpha_aniso;
  const Real kfactor = (saturated || spitzer_field_aligned) ? 4.0*ndim : 1.0;
  const auto size = pmy_pack->pmb->mb_size;
  const int nx1 = idx.nx1, nx2 = idx.nx2, nx3 = idx.nx3;
  const int active_nji = nx1*nx2, active_nkji = active_nji*nx3;
  Real bound = std::numeric_limits<float>::max();
  invalid = 0;
  Kokkos::parallel_reduce("spitzer_dt", Kokkos::RangePolicy<>(
      DevExeSpace(), 0, pmy_pack->nmb_thispack*active_nkji),
  KOKKOS_LAMBDA(const int n, Real &minimum, int &bad) {
    const int m = n/active_nkji, rem = n-m*active_nkji;
    const int k = rem/active_nji+ks, j = (rem%active_nji)/nx1+js, i = rem%nx1+is;
    const Real rho = w0(m,IDN,k,j,i);
    const Real heat_capacity_inverse = gm1/rho;
    const bool active_diffusion = alpha0 > 0.0 || kcap > 0.0;
    if (active_diffusion &&
        (!conduction::IsFinitePositive(heat_capacity_inverse) ||
         heat_capacity_inverse < conduction::kMinNormal)) {
      ++bad;
      return;
    }
    const Real dx[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
    Real inverse_square = 0.0, row_sum = 0.0;
    for (int d=0; d<ndim; ++d) {
      if (!conduction::IsFinitePositive(dx[d])) { ++bad; return; }
      const Real dsq = 1.0/SQR(dx[d]);
      // A finite dx can overflow when squared. Do not let its zero reciprocal
      // silently erase a representable K/dx^2 rate for a large conductivity.
      if (active_diffusion &&
          (!conduction::IsFinitePositive(dsq) || dsq < conduction::kMinNormal)) {
        ++bad;
        return;
      }
      inverse_square += dsq;
      const int di = d == 0, dj = d == 1, dk = d == 2;
      // A lost density-weighted term can be rescued later by alpha/rho.
      // Reject that unsupported intermediate range before accumulating it.
      if (alpha0 > 0.0) {
        const Real row_term = 0.5*(w0(m,IDN,k-dk,j-dj,i-di)+2.0*rho+
                                   w0(m,IDN,k+dk,j+dj,i+di))*dsq;
        if (!conduction::IsFinitePositive(row_term) ||
            row_term < conduction::kMinNormal) { ++bad; return; }
      }
      row_sum += 0.5*(w0(m,IDN,k-dk,j-dj,i-di)+2.0*rho+
                       w0(m,IDN,k+dk,j+dj,i+di))*dsq;
    }
    // Check each active term separately: a positive contribution from the
    // other closure must not mask zero/subnormal loss before multiplication
    // by inverse heat capacity. Keep ordinary rate arithmetic unchanged.
    if (alpha0 > 0.0) {
      const Real alpha_term = alpha0*row_sum;
      if (!conduction::IsFinitePositive(alpha_term) ||
          alpha_term < conduction::kMinNormal) { ++bad; return; }
    }
    if (kcap > 0.0) {
      const Real spitzer_term = 2.0*kfactor*kcap*inverse_square;
      if (!conduction::IsFinitePositive(spitzer_term) ||
          spitzer_term < conduction::kMinNormal) { ++bad; return; }
    }
    const Real rate = heat_capacity_inverse*
        (alpha0*row_sum+2.0*kfactor*kcap*inverse_square);
    if (!conduction::IsFiniteNonNegative(rate) ||
        (active_diffusion && rate < conduction::kMinNormal)) { ++bad; return; }
    if (rate > 0.0) minimum = fmin(minimum, 1.0/rate);
  }, Kokkos::Min<Real>(bound), Kokkos::Sum<int>(invalid));
  if (reject(invalid || !std::isfinite(bound) || bound <= 0.0,
             "invalid explicit conduction timestep")) return 0.0;
  if (saturated && include_saturation) {
    // Saturation's donor-pressure response is hyperbolic. Add its rate; it
    // must never be counted as an independently STS-accelerated diffusion rate.
    const Real adv_bound = SaturationTimeStep(w0, eos_data);
    const Real rate = 1.0/bound+1.0/adv_bound;
    if (reject(!std::isfinite(rate) || rate < 0.0,
               "invalid combined diffusion/saturation rate")) return 0.0;
    bound = rate > 0.0 ? 1.0/rate : std::numeric_limits<float>::max();
  }
  return bound;
}

void Conduction::CheckExplicitStage(const DvceArray5D<Real> &w0,
                                    const EOS_Data &eos_data) {
  if (!saturated && (!alpha_spitzer || spitzer_kappa0_code == 0.0)) return;
  // In SSP RK1/2/3 every constituent forward-Euler interval is dt, including
  // RK2 stage 2 (beta/gam0 = 1). Never change the committed mesh dt mid-step.
  // Mesh::NewTimeStep already applies the requested CFL safety factor when
  // selecting dt. Here check the current forward-Euler bound itself: harmless
  // stage heating/compression may use a little of that reserved margin.
  const Real bound = SpitzerTimeStep(w0, eos_data);
  const Real dt = pmy_pack->pmesh->dt;
  const Real tolerance = 64.0*std::numeric_limits<Real>::epsilon();
  FailIfInvalid(!std::isfinite(bound) || bound <= 0.0 ||
                !std::isfinite(dt) || dt <= 0.0 || dt > bound*(1.0+tolerance),
                "current SSP stage exceeds the temperature-dependent CFL bound");
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxes()
//! \brief Wrapper function that adds heat fluxes for different types of thermal
//! conduction to face-centered fluxes of conserved variables

void Conduction::AddHeatFluxes(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  if (saturated) {
    AddHeatFluxSaturated(w0, eos, flx);
    return;
  }
  if (alpha_iso != 0) {
    AddHeatFluxIso(w0, eos, flx);
  }
  if (alpha_aniso != 0) {
    AddHeatFluxAniso(w0, eos, flx);
  }
  if (alpha_spitzer) {
    AddHeatFluxSpitzer(w0, eos, flx);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxIso()
//! \brief Adds isotropic heat flux computed using constant conductivity to face-centered
//! fluxes of conserved variables

void Conduction::AddHeatFluxIso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  Real gm1 = eos.gamma-1.0;
  Real &alpha_ = alpha_iso;

  // fluxes in x1-direction
  auto &flx1 = flx.x1f;
  par_for("conduct1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j,i  )/w0(m,IDN,k,j,i  );
    Real templ = w0(m,IEN,k,j,i-1)/w0(m,IDN,k,j,i-1);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx1;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
    flx1(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->one_d) {
    return;
  }

  // fluxes in x2-direction
  auto &flx2 = flx.x2f;
  par_for("conduct2",DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k,j  ,i)/w0(m,IDN,k,j  ,i);
    Real templ = w0(m,IEN,k,j-1,i)/w0(m,IDN,k,j-1,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx2;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
    flx2(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  if (pmy_pack->pmesh->two_d) {
    return;
  }

  // fluxes in x3-direction
  auto &flx3 = flx.x3f;
  par_for("conduct3",DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real tempr = w0(m,IEN,k  ,j,i)/w0(m,IDN,k  ,j,i);
    Real templ = w0(m,IEN,k-1,j,i)/w0(m,IDN,k-1,j,i);
    Real dtempdx = (tempr - templ) * gm1 / size.d_view(m).dx3;
    Real densf = 0.5*(w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
    flx3(m,IEN,k,j,i) -= alpha_ * densf * dtempdx;
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddHeatFluxAniso()
//! \brief Conservative constant-alpha field-aligned heat flux. Transverse slopes
//! use the limited asymmetric method of Sharma & Hammett (2007), equations 5,17.
//! alpha_aniso is ADDITIONAL parallel diffusion above alpha_iso. Exactly zero
//! face field gives zero anisotropic flux; no arbitrary weak-field threshold.

void Conduction::AddHeatFluxAniso(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const int nmb1 = pmy_pack->nmb_thispack-1;
  auto w = w0;
  auto bcc = pmy_pack->pmhd->bcc0;
  auto size = pmy_pack->pmb->mb_size;
  const Real gm1 = eos.gamma-1.0, alpha = alpha_aniso;
  const bool mc = use_mc_limiter;
  for (int dir=0; dir<ndim; ++dir) {
    auto f = dir == 0 ? flx.x1f : (dir == 1 ? flx.x2f : flx.x3f);
    auto bf = dir == 0 ? pmy_pack->pmhd->b0.x1f :
              (dir == 1 ? pmy_pack->pmhd->b0.x2f : pmy_pack->pmhd->b0.x3f);
    const int di = dir == 0, dj = dir == 1, dk = dir == 2;
    par_for("conduct_aniso", DevExeSpace(), 0,nmb1, ind.ks,ind.ke+dk,
            ind.js,ind.je+dj, ind.is,ind.ie+di,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const int il=i-di, jl=j-dj, kl=k-dk;
      Real b[3];
      for (int d=0; d<3; ++d) {
        b[d] = d == dir ? bf(m,k,j,i) :
               0.5*bcc(m,d,k,j,i)+0.5*bcc(m,d,kl,jl,il);
      }
      // Normalize all three components, including Bz in a 2-D calculation.
      // Scaling first avoids underflow/overflow when field magnitudes vary.
      Real scale = fmax(fabs(b[0]),fmax(fabs(b[1]),fabs(b[2])));
      if (scale == 0.0) return;
      for (int d=0; d<3; ++d) b[d] /= scale;
      Real norm = sqrt(SQR(b[0])+SQR(b[1])+SQR(b[2]));
      for (int d=0; d<3; ++d) b[d] /= norm;
      Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
      Real tr = gm1*w(m,IEN,k,j,i)/w(m,IDN,k,j,i);
      Real tl = gm1*w(m,IEN,kl,jl,il)/w(m,IDN,kl,jl,il);
      Real grad = b[dir]*(tr-tl)/dx[dir];
      for (int d=0; d<ndim; ++d) {
        if (d == dir) continue;
        int ti=d == 0, tj=d == 1, tk=d == 2;
        Real rp = gm1*w(m,IEN,k+tk,j+tj,i+ti)/w(m,IDN,k+tk,j+tj,i+ti);
        Real rm = gm1*w(m,IEN,k-tk,j-tj,i-ti)/w(m,IDN,k-tk,j-tj,i-ti);
        Real lp = gm1*w(m,IEN,kl+tk,jl+tj,il+ti)/w(m,IDN,kl+tk,jl+tj,il+ti);
        Real lm = gm1*w(m,IEN,kl-tk,jl-tj,il-ti)/w(m,IDN,kl-tk,jl-tj,il-ti);
        Real transverse = mc ? MCL2State(MCL2State(rp-tr,tr-rm),MCL2State(lp-tl,tl-lm)) :
                               VLL4State(rp-tr,tr-rm,lp-tl,tl-lm);
        grad += b[d]*transverse/dx[d];
      }
      // Harmonic interpolation of rho*alpha handles conductivity contrasts.
      Real lo = fmin(w(m,IDN,k,j,i),w(m,IDN,kl,jl,il));
      Real hi = fmax(w(m,IDN,k,j,i),w(m,IDN,kl,jl,il));
      Real rhof = 2.0*lo/(1.0+lo/hi);
      f(m,IEN,k,j,i) -= alpha*rhof*b[dir]*grad;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void TempDependentHeatFlux()
//! \brief Adds heat flux to face-centered fluxes of conserved variables with
//! temperature-dependent conductivity


//----------------------------------------------------------------------------------------
//! \fn void Conduction::NewTimeStep()
//! \brief Compute new time step for thermal conduction.

void Conduction::NewTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  dt_transport = static_cast<Real>(std::numeric_limits<float>::max());
  if (saturated || (alpha_spitzer && spitzer_kappa0_code > 0.0)) {
    dtnew = SpitzerTimeStep(w0, eos_data, !UsesNonlinearSTS());
    if (UsesNonlinearSTS() && saturated) {
      dt_transport = SaturationTimeStep(w0, eos_data, true);
    }
    return;
  }
  dtnew = static_cast<Real>(std::numeric_limits<float>::max());

  // The active constant-alpha operator uses code units only. Do not dereference
  // punit here: a units block is optional and these conversions were unused.

  // capture variables for kernel
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &w0_ = w0;
  auto &multi_d = pmy_pack->pmesh->multi_d;
  auto &three_d = pmy_pack->pmesh->three_d;
  auto &size = pmy_pack->pmb->mb_size;
  Real gm1 = eos_data.gamma-1.0;
  const int ndim = three_d ? 3 : (multi_d ? 2 : 1);
  // Deliberately conservative explicit bound for the limited cross derivatives.
  // |b_i b_j| <= 1 and 1/(dx_i dx_j) <= (1/dx_i^2+1/dx_j^2)/2;
  // allow a factor four per dimension for the nested limited slopes. Arithmetic
  // face densities below bound the harmonic density in the anisotropic flux.
  // This bound is not a proof of nonlinear STS or refined-mesh acceptance.
  Real alpha0 = alpha_iso + 4.0*ndim*alpha_aniso;

  // find smallest timestep for thermal conduction in each cell
  // Note loop over all cells needed even for constant conductivity
  Kokkos::parallel_reduce("cond_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real row_sum = 0.5*(w0_(m,IDN,k,j,i-1) + 2.0*w0_(m,IDN,k,j,i) +
                        w0_(m,IDN,k,j,i+1))/SQR(size.d_view(m).dx1);
    if (multi_d) {
      row_sum += 0.5*(w0_(m,IDN,k,j-1,i) + 2.0*w0_(m,IDN,k,j,i) +
                     w0_(m,IDN,k,j+1,i))/SQR(size.d_view(m).dx2);
    }
    if (three_d) {
      row_sum += 0.5*(w0_(m,IDN,k-1,j,i) + 2.0*w0_(m,IDN,k,j,i) +
                     w0_(m,IDN,k+1,j,i))/SQR(size.d_view(m).dx3);
    }

    Real rate = alpha0*gm1*row_sum/w0_(m,IDN,k,j,i);
    if (rate > 0.0) {
      min_dt = fmin(min_dt, 1.0/rate);
    }
  }, Kokkos::Min<Real>(dtnew));

  return;
}

//----------------------------------------------------------------------------------------
// Temperature-dependent and saturated conductive fluxes.

// Interpolate K(theta)=K0*theta^(5/2) harmonically at each face. The SAME face
// coefficient multiplies the normal and limited transverse theta gradients;
// no direction-dependent nonlinear thermal-potential differences are used.
// A truly zero endpoint K gives zero Spitzer flux at that face. This is the
// harmonic closure's cold-endpoint limit, not a numerical temperature floor.
void Conduction::AddHeatFluxSpitzer(const DvceArray5D<Real> &w0, const EOS_Data &eos,
                                  DvceFaceFld5D<Real> &flx) {
  const Real k0 = spitzer_kappa0_code;
  if (!alpha_spitzer || k0 == 0.0) return;
  const Real gm1 = eos.gamma-1.0;
  FailIfInvalid(!eos.is_ideal || !conduction::IsFinitePositive(gm1) ||
                !conduction::IsFinitePositive(k0),
                "Spitzer equation of state/coefficient");
  const bool aligned = spitzer_field_aligned;
  FailIfInvalid(aligned && pmy_pack->pmhd == nullptr, "Spitzer magnetic geometry");
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int nmb = pmy_pack->nmb_thispack;
  const bool mc = use_mc_limiter;
  auto w = w0;
  auto size = pmy_pack->pmb->mb_size;
  DvceArray5D<Real> bcc;
  if (aligned) bcc = pmy_pack->pmhd->bcc0;
  for (int dir=0; dir<ndim; ++dir) {
    const int di = dir == 0, dj = dir == 1, dk = dir == 2;
    const int ni = ind.nx1+di, nj = ind.nx2+dj, nk = ind.nx3+dk;
    const int nji = ni*nj, nface = nji*nk;
    auto f = dir == 0 ? flx.x1f : (dir == 1 ? flx.x2f : flx.x3f);
    DvceArray4D<Real> bf;
    if (aligned) {
      bf = dir == 0 ? pmy_pack->pmhd->b0.x1f :
           (dir == 1 ? pmy_pack->pmhd->b0.x2f : pmy_pack->pmhd->b0.x3f);
    }
    int invalid_faces = 0;
    Kokkos::parallel_reduce("conduct_spitzer_harmonic",
      Kokkos::RangePolicy<>(DevExeSpace(),0,nmb*nface),
    KOKKOS_LAMBDA(const int idx, int &invalid) {
      const int m = idx/nface, cell = idx-m*nface;
      const int kk = cell/nji, jj = (cell-kk*nji)/ni, ii = cell-kk*nji-jj*ni;
      const int k = ks+kk, j = js+jj, i = is+ii;
      const int il = i-di, jl = j-dj, kl = k-dk;
      const Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
      Real b[3] = {0.0,0.0,0.0};
      if (aligned) {
        for (int d=0; d<3; ++d) {
          b[d] = d == dir ? bf(m,k,j,i) :
                 0.5*bcc(m,d,k,j,i)+0.5*bcc(m,d,kl,jl,il);
          if (!conduction::IsFinite(b[d])) { ++invalid; return; }
        }
        const Real scale = fmax(fabs(b[0]),fmax(fabs(b[1]),fabs(b[2])));
        // A zero field still consumes/validates the thermal stencil below.
        if (scale > 0.0) {
          for (int d=0; d<3; ++d) b[d] /= scale;
          const Real norm = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
          for (int d=0; d<3; ++d) b[d] /= norm;
        }
      }
      const Real tr = conduction::SpitzerTemperature(w(m,IDN,k,j,i),w(m,IEN,k,j,i),gm1);
      const Real tl = conduction::SpitzerTemperature(w(m,IDN,kl,jl,il),
                                                     w(m,IEN,kl,jl,il),gm1);
      if (!conduction::IsFiniteNonNegative(tl) ||
          !conduction::IsFiniteNonNegative(tr)) { ++invalid; return; }
      const Real kappal = conduction::SpitzerConductivity(tl,k0);
      const Real kappar = conduction::SpitzerConductivity(tr,k0);
      if (!conduction::IsFiniteNonNegative(kappal) ||
          !conduction::IsFiniteNonNegative(kappar) ||
          (tl > 0.0 && kappal == 0.0) || (tr > 0.0 && kappar == 0.0)) {
        ++invalid;
        return;
      }
      const Real kface = conduction::HarmonicMeanPositive(kappal,kappar);
      if (!conduction::IsFiniteNonNegative(kface) ||
          !conduction::IsFinitePositive(dx[dir])) { ++invalid; return; }
      Real g[3] = {0.0,0.0,0.0};
      g[dir] = conduction::SpitzerTemperatureSlope(tl,tr,dx[dir]);
      if (!conduction::IsFinite(g[dir])) { ++invalid; return; }
      if (aligned) {
        for (int d=0; d<ndim; ++d) {
          if (d == dir) continue;
          const int ti = d == 0, tj = d == 1, tk = d == 2;
          const Real rp = conduction::SpitzerTemperature(w(m,IDN,k+tk,j+tj,i+ti),
                                                       w(m,IEN,k+tk,j+tj,i+ti),gm1);
          const Real rm = conduction::SpitzerTemperature(w(m,IDN,k-tk,j-tj,i-ti),
                                                       w(m,IEN,k-tk,j-tj,i-ti),gm1);
          const Real lp = conduction::SpitzerTemperature(w(m,IDN,kl+tk,jl+tj,il+ti),
                                                       w(m,IEN,kl+tk,jl+tj,il+ti),gm1);
          const Real lm = conduction::SpitzerTemperature(w(m,IDN,kl-tk,jl-tj,il-ti),
                                                       w(m,IEN,kl-tk,jl-tj,il-ti),gm1);
          if (!conduction::IsFiniteNonNegative(rp) ||
              !conduction::IsFiniteNonNegative(rm) ||
              !conduction::IsFiniteNonNegative(lp) ||
              !conduction::IsFiniteNonNegative(lm) ||
              !conduction::IsFinitePositive(dx[d])) { ++invalid; return; }
          // These are temperature slopes, not potential slopes. Positive
          // homogeneity permits limiting gradients before the common K_face.
          g[d] = conduction::SpitzerLimited4(
              conduction::SpitzerTemperatureSlope(tr,rp,dx[d]),
              conduction::SpitzerTemperatureSlope(rm,tr,dx[d]),
              conduction::SpitzerTemperatureSlope(tl,lp,dx[d]),
              conduction::SpitzerTemperatureSlope(lm,tl,dx[d]),mc);
          if (!conduction::IsFinite(g[d])) { ++invalid; return; }
        }
        // Exactly perpendicular or zero face field: no flux, but no invalid
        // thermal input was hidden by this exit. Retain all three B components.
        if (b[dir] == 0.0) return;
      }
      Real flux = -conduction::SpitzerProduct3(kface,g[dir],1.0);
      if (aligned) {
        const Real scale = fmax(fabs(g[0]),fmax(fabs(g[1]),fabs(g[2])));
        Real projected = 0.0;
        if (scale > 0.0) {
          for (int d=0; d<ndim; ++d) projected += b[d]*(g[d]/scale);
        }
        // Reuse the robust generic product without forming K_face*scale first.
        flux = -conduction::SaturatedProduct5(kface,scale,b[dir],projected,1.0);
      }
      const Real updated = f(m,IEN,k,j,i)+flux;
      if (!conduction::IsFinite(flux) || !conduction::IsFinite(updated)) {
        ++invalid;
        return;
      }
      f(m,IEN,k,j,i) = updated;
    }, Kokkos::Sum<int>(invalid_faces));
    // Every rank calls the coordinated check in every direction, including
    // local zero-error/zero-block cases. This is rejection, not rollback/retry.
    FailIfInvalid(invalid_faces,"Spitzer face thermal state, geometry, or flux");
  }
}

void Conduction::AddHeatFluxSaturated(const DvceArray5D<Real> &w0,
    const EOS_Data &eos, DvceFaceFld5D<Real> &flx) {
  const Real gm1 = eos.gamma-1.0;
  const Real k0 = alpha_spitzer ? spitzer_kappa0_code : 0.0;
  const bool aligned = alpha_aniso > 0.0 || (k0 > 0.0 && spitzer_field_aligned);
  FailIfInvalid(!eos.is_ideal || !conduction::IsFinitePositive(gm1) ||
                (aligned && pmy_pack->pmhd == nullptr), "saturated conduction setup");
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const conduction::SaturatedFaceParameters p = {gm1,alpha_iso,alpha_aniso,k0,
      saturation_phi,ndim,spitzer_field_aligned,use_mc_limiter};
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int nmb = pmy_pack->nmb_thispack;
  auto w = w0;
  auto size = pmy_pack->pmb->mb_size;
  DvceArray5D<Real> bcc;
  if (aligned) bcc = pmy_pack->pmhd->bcc0;
  for (int dir=0; dir<ndim; ++dir) {
    const int ni = ind.nx1+(dir == 0), nj = ind.nx2+(dir == 1);
    const int nk = ind.nx3+(dir == 2), nji = ni*nj, nface = nji*nk;
    auto f = dir == 0 ? flx.x1f : (dir == 1 ? flx.x2f : flx.x3f);
    DvceArray4D<Real> bf;
    if (aligned) {
      bf = dir == 0 ? pmy_pack->pmhd->b0.x1f :
          (dir == 1 ? pmy_pack->pmhd->b0.x2f : pmy_pack->pmhd->b0.x3f);
    }
    int invalid_faces = 0;
    Kokkos::parallel_reduce("conduct_saturated", Kokkos::RangePolicy<>(
        DevExeSpace(),0,nmb*nface),
    KOKKOS_LAMBDA(const int idx, int &invalid) {
      const int m = idx/nface, cell = idx-m*nface;
      const int k = ks+cell/nji, j = js+(cell%nji)/ni, i = is+cell%ni;
      const Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
      const auto face = conduction::SaturatedFace(w,bcc,bf,p,dx,dir,m,k,j,i);
      if (!face.valid) { ++invalid; return; }
      const Real updated = f(m,IEN,k,j,i)+face.normal_flux;
      if (!conduction::IsFinite(updated)) { ++invalid; return; }
      // Only conduction is suppressed; existing hyperbolic flux is untouched.
      f(m,IEN,k,j,i) = updated;
    }, Kokkos::Sum<int>(invalid_faces));
    FailIfInvalid(invalid_faces,"saturated face thermal state, geometry, or flux");
  }
}

// The larger of the donor-pressure derivative rate and outgoing-energy drain
// rate, summed over actual adjacent faces of each cell. The caller combines
// this inverse timestep with the UNSUPPRESSED diffusion row bound. The drain
// check bounds conduction alone; it does not promise coupled-MHD positivity.
// With donor_only, omit the forward-Euler drain restriction: this returns only
// the nonaccelerated donor-pressure rate needed by guarded nonlinear STS.
Real Conduction::SaturationTimeStep(const DvceArray5D<Real> &w0,
                                    const EOS_Data &eos, bool donor_only) const {
  const Real gm1 = eos.gamma-1.0;
  const Real k0 = alpha_spitzer ? spitzer_kappa0_code : 0.0;
  const bool aligned = alpha_aniso > 0.0 || (k0 > 0.0 && spitzer_field_aligned);
  FailIfInvalid(!eos.is_ideal || !conduction::IsFinitePositive(gm1) ||
                (aligned && pmy_pack->pmhd == nullptr), "saturated timestep setup");
  auto &ind = pmy_pack->pmesh->mb_indcs;
  const int ndim = pmy_pack->pmesh->three_d ? 3 : (pmy_pack->pmesh->multi_d ? 2 : 1);
  const conduction::SaturatedFaceParameters p = {gm1,alpha_iso,alpha_aniso,k0,
      saturation_phi,ndim,spitzer_field_aligned,use_mc_limiter};
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int nmb = pmy_pack->nmb_thispack;
  auto w = w0;
  auto size = pmy_pack->pmb->mb_size;
  DvceArray5D<Real> bcc;
  if (aligned) bcc = pmy_pack->pmhd->bcc0;
  DvceArray4D<Real> b1, b2, b3;
  if (aligned) {
    b1 = pmy_pack->pmhd->b0.x1f;
    b2 = pmy_pack->pmhd->b0.x2f;
    b3 = pmy_pack->pmhd->b0.x3f;
  }
  const int ni = ind.nx1, nj = ind.nx2, nji = ni*nj, ncell = nji*ind.nx3;
  Real rate = 0.0;
  int invalid_cells = 0;
  Kokkos::parallel_reduce("saturated_donor_dt", Kokkos::RangePolicy<>(
      DevExeSpace(),0,nmb*ncell),
  KOKKOS_LAMBDA(const int idx, Real &max_rate, int &invalid) {
    const int m = idx/ncell, cell = idx-m*ncell;
    const int k = ks+cell/nji, j = js+(cell%nji)/ni, i = is+cell%ni;
    const Real dx[3] = {size.d_view(m).dx1,size.d_view(m).dx2,size.d_view(m).dx3};
    const Real eint = w(m,IEN,k,j,i);
    if (!conduction::IsFiniteNonNegative(eint)) { ++invalid; return; }
    Real advective = 0.0, drain = 0.0;
    for (int dir=0; dir<ndim; ++dir) {
      auto bf = dir == 0 ? b1 : (dir == 1 ? b2 : b3);
      const auto left = conduction::SaturatedFace(w,bcc,bf,p,dx,dir,m,k,j,i);
      const auto right = conduction::SaturatedFace(w,bcc,bf,p,dx,dir,m,
          k+(dir == 2),j+(dir == 1),i+(dir == 0));
      if (!left.valid || !right.valid) { ++invalid; return; }
      // Evaluate divisions separately to avoid overflowing a sum of speeds.
      const Real al = left.donor_speed/dx[dir];
      const Real ar = right.donor_speed/dx[dir];
      if (!conduction::IsFiniteNonNegative(al) ||
          !conduction::IsFiniteNonNegative(ar) ||
          (left.donor_speed > 0.0 && al == 0.0) ||
          (right.donor_speed > 0.0 && ar == 0.0)) { ++invalid; return; }
      advective += al+ar;
      if (!donor_only) {
        const Real outward[2] = {fmax(-left.normal_flux,0.0),
                                 fmax(right.normal_flux,0.0)};
        for (int side=0; side<2; ++side) {
          if (outward[side] == 0.0) continue;
          if (eint == 0.0) { ++invalid; return; }
          // Log scaling avoids either dx*eint or outward/dx overflowing first.
          const Real contribution = exp(log(outward[side])-log(dx[dir])-log(eint));
          if (!conduction::IsFinitePositive(contribution)) { ++invalid; return; }
          drain += contribution;
        }
      }
    }
    const Real local_rate = donor_only ? advective : fmax(advective,drain);
    if (!conduction::IsFiniteNonNegative(local_rate)) { ++invalid; return; }
    max_rate = fmax(max_rate,local_rate);
  }, Kokkos::Max<Real>(rate), Kokkos::Sum<int>(invalid_cells));
  if (nmb == 0) rate = 0.0;
  // One combined kernel visits every direction; every rank joins this check,
  // including zero-error and empty-pack ranks.
  FailIfInvalid(invalid_cells || !conduction::IsFiniteNonNegative(rate),
                "saturated donor-pressure or outgoing-energy timestep rate");
  const Real bound = rate == 0.0 ?
      static_cast<Real>(std::numeric_limits<float>::max()) : 1.0/rate;
  FailIfInvalid(!conduction::IsFinitePositive(bound),
                "unrepresentable saturated donor-pressure timestep");
  return bound;
}
