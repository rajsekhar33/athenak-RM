#ifndef DIFFUSION_CONDUCTION_CLOSURE_HPP_
#define DIFFUSION_CONDUCTION_CLOSURE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction_closure.hpp
//! \brief Constitutive functions for Newtonian ideal-gas thermal conduction.
//
// theta = p/rho is the code temperature. Conductivity K is defined by
// F_class = -K grad(theta), so its temperature diffusivity is (gamma-1)*K/rho.
// These helpers neither apply EOS floors nor repair invalid states. Invalid
// arguments return NaN; callers must check raw states and reject invalid or
// unrepresentable results before updating conserved energy. Zero temperature
// is a valid mathematical limit; the caller may enforce stricter EOS floors.

#include <cmath>
#include <limits>

#include "athena.hpp"

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

// Reference-only Kirchhoff gradient; not the production Spitzer face law.
// [Psi(tr)-Psi(tl)]/dx, Psi(T)=K0*T^(7/2)/(7/2), without forming Psi.
// For r=Tlo/Thi, the exact secant conductivity is K(Thi)*Q(r), where
// Q=[1+r+r^2+r^3/(1+sqrt(r))]/3.5 lies in [2/7,1]. This factorization avoids
// cancellation between nearly equal powers. Unrepresentable FINAL gradients
// may return inf or underflow to zero; no temperature floor or cap is applied.
KOKKOS_INLINE_FUNCTION
Real SpitzerPotentialGradient(const Real tl, const Real tr,
                              const Real dx, const Real kappa0_code) {
  if (!IsFiniteNonNegative(tl) || !IsFiniteNonNegative(tr) ||
      !IsFinitePositive(dx) || !IsFiniteNonNegative(kappa0_code)) {
    return InvalidClosureValue();
  }
  if (kappa0_code == 0.0 || tl == tr) return 0.0;
  const Real hi = fmax(tl,tr), lo = fmin(tl,tr);
  const Real delta = hi-lo;
  const Real r = lo/hi;
  const Real q = (1.0+r*(1.0+r*(1.0+r/(1.0+sqrt(r)))))/3.5;
  const Real secant = SpitzerConductivity(hi,kappa0_code)*q;
  const Real slope = delta/dx;
  Real magnitude = secant*slope;
  if (!(secant >= kMinNormal && IsFinite(secant) &&
        slope >= kMinNormal && IsFinite(slope) && IsFinitePositive(magnitude))) {
    // Absolute delta, rather than delta/hi, preserves small absolute scales.
    magnitude = exp(log(kappa0_code)+2.5*log(hi)+log(delta)-log(dx)+log(q));
  }
  return tr > tl ? magnitude : -magnitude;
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

// Optional absolute code-flux ceiling. Call this only when an absolute ceiling
// is explicitly enabled; disabled legacy q_limit sentinels are parser policy.
KOKKOS_INLINE_FUNCTION
Real LimitSaturatedFlux(const Real q_sat, const Real constant_cap) {
  if (!IsFiniteNonNegative(q_sat) || !IsFiniteNonNegative(constant_cap)) {
    return InvalidClosureValue();
  }
  return fmin(q_sat, constant_cap);
}

// q_sat/(q_sat+|F_class|), evaluated without overflowing the denominator.
// At zero classical flux the identity factor is returned, including q_sat=0.
KOKKOS_INLINE_FUNCTION
Real SaturationSuppression(const Real flux_norm, const Real q_sat) {
  if (!IsFiniteNonNegative(flux_norm) || !IsFiniteNonNegative(q_sat)) {
    return InvalidClosureValue();
  }
  if (flux_norm == 0.0) return 1.0;
  if (q_sat == 0.0) return 0.0;
  if (flux_norm > q_sat) {
    const Real ratio = q_sat/flux_norm;
    return ratio/(1.0 + ratio);
  }
  return 1.0/(1.0 + flux_norm/q_sat);
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

// A true norm larger than max(Real) returns inf. Use SaturateHeatFlux directly
// when the classical vector may be that large: it never needs this full norm.
KOKKOS_INLINE_FUNCTION
Real FluxMagnitude(const Real x, const Real y, const Real z) {
  if (!ValidHeatFlux({x, y, z})) return InvalidClosureValue();
  const Real scale = fmax(fabs(x), fmax(fabs(y), fabs(z)));
  if (scale == 0.0) return 0.0;
  const Real sx = x/scale, sy = y/scale, sz = z/scale;
  return scale*sqrt(sx*sx + sy*sy + sz*sz);
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

} // namespace conduction

#endif // DIFFUSION_CONDUCTION_CLOSURE_HPP_
