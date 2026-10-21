#ifndef UTILS_SPHERICAL_HARM_HPP_
#define UTILS_SPHERICAL_HARM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file spherical_harm.hpp
//! \brief spin-weighted spherical harmonics

#include <cmath>
#include <iostream>
#include <list>
#include <algorithm>

#include "athena.hpp"


KOKKOS_INLINE_FUNCTION
double fac(int i) {
  double result = 1;
  if (i>0) {
    while (i>0) {
      result*=i;
      i-=1;
    }
  }
  return(result);
}

KOKKOS_INLINE_FUNCTION
void SphericalHarm(Real *ylmR, Real *ylmI, int l, int m, Real theta, Real phi);

//Calculate spin-weighted spherical harmonics using Wigner-d matrix notation
// see e.g. Eq II.7, II.8 in 0709.0093
KOKKOS_INLINE_FUNCTION
void SWSphericalHarm(Real * ylmR, Real * ylmI, int l, int m, int s,
                     Real theta, Real phi) {
  if (s == 0) {
    SphericalHarm(ylmR, ylmI, l, m, theta, phi);
    return;
  }
  Real wignerd = 0;
  int k1,k2,k;
  k1 = Kokkos::max(0, m-s);
  k2 = Kokkos::min(l+m,l-s);

  for (k = k1; k<=k2; ++k) {
    wignerd += Kokkos::pow((-1),k)*Kokkos::pow(Kokkos::cos(theta/2.0),2*l+m-s-2*k)
                *Kokkos::pow(Kokkos::sin(theta/2.0),2*k+s-m)
                /(fac(l+m-k)*fac(l-s-k)*fac(k)*fac(k+s-m));
  }
  wignerd *= Kokkos::pow((-1),s)*Kokkos::sqrt((2*l+1)/(4*M_PI))*Kokkos::sqrt(fac(l+m))
              *Kokkos::sqrt(fac(l-m))
              *Kokkos::sqrt(fac(l+s))*Kokkos::sqrt(fac(l-s));
  *ylmR = wignerd*Kokkos::cos(m*phi);
  *ylmI = wignerd*Kokkos::sin(m*phi);
}

// Normalized associated-Legendre recurrence, avoiding cancellation in the Wigner sum.
// See DLMF 14.10.3 (recurrence), 14.30.1 (normalization), 14.30.6 (negative m).
KOKKOS_INLINE_FUNCTION
void SphericalHarm(Real *ylmR, Real *ylmI, int l, int m, Real theta, Real phi) {
  const int am = (m < 0) ? -m : m;
  if (am > l) {
    *ylmR = 0.0;
    *ylmI = 0.0;
    return;
  }
  const Real ct = Kokkos::cos(theta);
  const Real st = Kokkos::sin(theta);
  Real plm = 1.0/Kokkos::sqrt(4.0*M_PI);
  // The signed sine preserves the old angular continuation outside [0, pi].
  for (int k = 1; k <= am; ++k) {
    plm *= -st*Kokkos::sqrt((2.0*k + 1.0)/(2.0*k));
  }
  if (l > am) {
    Real prev = plm;
    plm *= ct*Kokkos::sqrt(2.0*am + 3.0);
    for (int k = am + 2; k <= l; ++k) {
      const Real kd = k;
      const Real md = am;
      const Real denom = kd*kd - md*md;
      const Real a = Kokkos::sqrt((4.0*kd*kd - 1.0)/denom);
      const Real b = Kokkos::sqrt((2.0*kd + 1.0)*((kd - 1.0)*(kd - 1.0) - md*md)
                                 /((2.0*kd - 3.0)*denom));
      const Real next = a*ct*plm - b*prev;
      prev = plm;
      plm = next;
    }
  }
  if (m < 0 && am%2 != 0) plm = -plm;
  *ylmR = plm*Kokkos::cos(m*phi);
  *ylmI = plm*Kokkos::sin(m*phi);
}

// Calculate first derivatives of spherical harmonics using Wigner-d matrix notation
KOKKOS_INLINE_FUNCTION
void SphericalHarmFirstDerivs(Real *YlmR, Real *YlmI, Real *YlmRdth, Real *YlmIdth,
                         Real *YlmRdphi, Real *YlmIdphi, int l, int m, Real theta,
                         Real phi) {
  SphericalHarm(YlmR, YlmI, l, m, theta, phi);

  // Phi first derivative
  *YlmRdphi = -m * (*YlmI);
  *YlmIdphi =  m * (*YlmR);

  // Theta first derivative
  // Initialization is important, since for l=m modes
  // the (m < l) block is skipped and *YlmRdth/*YlmIdth subtract
  // from garbage values
  *YlmRdth = 0.0;
  *YlmIdth = 0.0;
  Real Y_upR, Y_upI, Y_downR, Y_downI;

  if (m < l) {
      SphericalHarm(&Y_upR, &Y_upI, l, m + 1, theta, phi);
      Real coeff = 0.5 * Kokkos::sqrt((l - m)*(l + m + 1));
      *YlmRdth = coeff * (Y_upR * Kokkos::cos(phi) + Y_upI * Kokkos::sin(phi));
      *YlmIdth = coeff * (-Y_upR * Kokkos::sin(phi) + Y_upI * Kokkos::cos(phi));
  }
  if (m > -l) {
      SphericalHarm(&Y_downR, &Y_downI, l, m-1, theta, phi);
      Real coeff = 0.5 * Kokkos::sqrt((l + m)*(l - m + 1));
      *YlmRdth -= coeff * (Y_downR * Kokkos::cos(phi) - Y_downI * Kokkos::sin(phi));
      *YlmIdth -= coeff * (Y_downR * Kokkos::sin(phi) + Y_downI * Kokkos::cos(phi));
  }
}

// Calculate second derivatives of spherical harmonics using Wigner-d matrix notation
KOKKOS_INLINE_FUNCTION
void SphericalHarmSecondDerivs(Real *YlmR, Real *YlmI, Real *YlmRdth, Real *YlmIdth,
                               Real *YlmRdphi, Real *YlmIdphi, Real *YlmRdth2,
                               Real *YlmIdth2, Real *YlmRdphi2, Real *YlmIdphi2,
                               Real *YlmRdthdphi, Real *YlmIdthdphi, int l, int m,
                               Real theta, Real phi) {
  SphericalHarmFirstDerivs(YlmR, YlmI, YlmRdth, YlmIdth, YlmRdphi,
                           YlmIdphi, l, m, theta, phi);

  // Phi second derivative
  *YlmRdphi2 = -Kokkos::pow(m, 2) * (*YlmR);
  *YlmIdphi2 = -Kokkos::pow(m, 2) * (*YlmI);

  // Mixed second derivative
  *YlmRdthdphi = -m * (*YlmIdth);
  *YlmIdthdphi = m * (*YlmRdth);

  // Theta second derivative
  Real sin_theta = Kokkos::sin(theta);
  Real cot_theta = Kokkos::cos(theta) / sin_theta;

  *YlmRdth2 = -l * (l+1) * (*YlmR) - cot_theta * (*YlmRdth) +
                Kokkos::pow(m, 2)/Kokkos::pow(sin_theta, 2) * (*YlmR);
  *YlmIdth2 = -l * (l+1) * (*YlmI) - cot_theta * (*YlmIdth) +
                Kokkos::pow(m, 2)/Kokkos::pow(sin_theta, 2) * (*YlmI);
}

//----------------------------------------------------------------------------------------
//! \fn int lmindex(const int l, const int m, const lm)
//! \brief Multipolar single index (l,m) -> index
KOKKOS_INLINE_FUNCTION
int lmindex(const int l, const int m, const int lm) {
  return l * (lm + 1) + m;
}

#endif // UTILS_SPHERICAL_HARM_HPP_
