#ifndef PGEN_TESTS_TRANSPORT_FRONT_HPP_
#define PGEN_TESTS_TRANSPORT_FRONT_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Test-only periodic hot slab: exact sharp cell averages or a smoothed Fourier series.
#include <cmath>
#include "athena.hpp"

namespace transport_front {
KOKKOS_INLINE_FUNCTION
Real CDF(Real t, const Real *widths, int dim) {
  Real half=0.0, norm=1.0;
  for (int a=0; a<dim; ++a) {
    half+=0.5*widths[a];
    norm*=widths[a]*(a+1);
  }
  if (t<=-half) return 0.0;
  if (t>=half) return 1.0;
  Real sum=0.0;
  for (int mask=0; mask<(1<<dim); ++mask) {
    Real v=t+half, sign=1.0;
    for (int a=0; a<dim; ++a) {
      if (mask & (1<<a)) {
        v-=widths[a];
        sign=-sign;
      }
    }
    Real power=1.0;
    for (int a=0; a<dim; ++a) power*=fmax(v,0.0);
    sum+=sign*power;
  }
  return sum/norm;
}

KOKKOS_INLINE_FUNCTION
Real Fraction(Real phase, const Real *widths, int dim, Real smoothing) {
  phase-=floor(phase);
  if (smoothing==0.0) {
    Real value=0.0;
    // Active normal entries are 0 or 1, with at least four cells per direction.
    for (int image=-1; image<=2; ++image) {
      value+=CDF(image+0.25-phase,widths,dim)-CDF(image-0.25-phase,widths,dim);
    }
    return value;
  }
  Real value=0.5;
  for (int harmonic=1; harmonic<=63; harmonic+=2) {
    Real h=static_cast<Real>(harmonic);
    Real coefficient=((harmonic%4)==1 ? 1.0 : -1.0)*2.0/(M_PI*h)*
                     exp(-2.0*M_PI*M_PI*smoothing*smoothing*h*h);
    for (int a=0; a<dim; ++a) {
      Real arg=M_PI*h*widths[a];
      coefficient*=sin(arg)/arg;
    }
    value+=coefficient*cos(2.0*M_PI*h*phase);
  }
  return value;
}
} // namespace transport_front
#endif // PGEN_TESTS_TRANSPORT_FRONT_HPP_
