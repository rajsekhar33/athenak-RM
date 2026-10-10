"""Native constitutive checks; these do not establish PDE or STS acceptance.

Run with pytest --basetemp pointing into the scratch filesystem. The tiny C++
driver includes the real header through the same type-stub approach as the
existing RKL2 controller unit tests; no AthenaK build or simulation is needed.
"""

from decimal import Decimal, localcontext
from itertools import permutations
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    prepared = os.environ.get("ATHENAK_CONDUCTION_CLOSURE_EXE")
    if prepared:
        executable = Path(prepared).resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            pytest.fail(f"Prepared closure helper is not executable: {executable}")
        return executable
    build_dir = tmp_path_factory.mktemp("conduction_closure")
    (build_dir / "athena.hpp").write_text(
        "using Real = double;\n#define KOKKOS_INLINE_FUNCTION inline\n",
        encoding="ascii",
    )
    source = build_dir / "closure.cpp"
    source.write_text(
        r"""
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "diffusion/conduction_closure.hpp"

// Frozen pre-range-fix arithmetic: compare on the same execution backend.
KOKKOS_INLINE_FUNCTION
conduction::HeatFluxVector LegacyVector(const conduction::HeatFluxVector &flux,
                                        const Real q_sat) {
  const Real scale = fmax(fabs(flux.x), fmax(fabs(flux.y), fabs(flux.z)));
  if (scale == 0.0 || q_sat == 0.0) return {0.0, 0.0, 0.0};
  const Real sx = flux.x/scale, sy = flux.y/scale, sz = flux.z/scale;
  const Real norm_scaled = sqrt(sx*sx + sy*sy + sz*sz);
  if (scale >= q_sat) {
    const Real amplitude = q_sat/(norm_scaled + q_sat/scale);
    return {sx*amplitude, sy*amplitude, sz*amplitude};
  }
  const Real denominator = 1.0 + (scale/q_sat)*norm_scaled;
  return {flux.x/denominator, flux.y/denominator, flux.z/denominator};
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const std::string op(argv[1]);
  const auto arg = [&](int i) { return std::strtod(argv[i], nullptr); };
  std::cout << std::setprecision(17);
  if (op == "harmonic" && argc == 4) {
    std::cout << conduction::HarmonicMeanPositive(arg(2), arg(3));
  } else if (op == "spitzer" && argc == 4) {
    std::cout << conduction::SpitzerConductivity(arg(2), arg(3));
  } else if (op == "spitzer_face" && argc == 5) {
    std::cout << conduction::HarmonicMeanPositive(
        conduction::SpitzerConductivity(arg(2), arg(4)),
        conduction::SpitzerConductivity(arg(3), arg(4)));
  } else if (op == "cowie" && argc == 5) {
    std::cout << conduction::CowieMcKeeFlux(arg(2), arg(3), arg(4));
  } else if (op == "cap" && argc == 4) {
    std::cout << conduction::LimitSaturatedFlux(arg(2), arg(3));
  } else if (op == "factor" && argc == 4) {
    std::cout << conduction::SaturationSuppression(arg(2), arg(3));
  } else if (op == "norm" && argc == 5) {
    std::cout << conduction::FluxMagnitude(arg(2), arg(3), arg(4));
  } else if ((op == "vector" || op == "vector_legacy") && argc == 6) {
    const conduction::HeatFluxVector flux{arg(2), arg(3), arg(4)};
    const auto result = op == "vector" ? conduction::SaturateHeatFlux(flux, arg(5)) :
                                         LegacyVector(flux, arg(5));
    std::cout << result.x << " " << result.y << " " << result.z;
  } else if (op == "units" && argc == 7) {
    const conduction::ConductivityUnits units{arg(3), arg(4), arg(5), arg(6)};
    std::cout << conduction::SpitzerCoefficientCode(arg(2), units);
  } else if (op == "valid" && argc == 4) {
    std::cout << conduction::ValidThermalState(arg(2), arg(3));
  } else {
    return 2;
  }
  return 0;
}
""",
        encoding="ascii",
    )
    executable = build_dir / "closure"
    subprocess.run(
        ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         f"-I{build_dir}", f"-I{REPO_ROOT / 'src'}", str(source),
         "-o", str(executable)],
        check=True,
    )
    return executable


def _run(closure, *args):
    result = subprocess.run([str(closure), *map(str, args)], check=True,
                            capture_output=True, text=True)
    return [float(value) for value in result.stdout.split()]


def _scalar(closure, *args):
    return _run(closure, *args)[0]


@pytest.mark.parametrize("a,b,expected", [
    (1., 1., 1.), (2., 6., 3.), (0., 0., 0.), (0., 4., 0.),
    (1.e-300, 1.e300, 2.e-300),
    (sys.float_info.max, sys.float_info.max, sys.float_info.max),
])
def test_harmonic_conductivity_is_symmetric_and_bounded(closure, a, b, expected):
    value = _scalar(closure, "harmonic", a, b)
    assert value == pytest.approx(expected, rel=3.e-15, abs=0.)
    assert value == _scalar(closure, "harmonic", b, a)
    assert min(a, b) <= value <= max(a, b)


def test_spitzer_law_has_no_hidden_temperature_switch(closure):
    for temperature in (0., .25, 1., 4., 6.49e4, 6.5e4, 6.51e4, 1.e6):
        value = _scalar(closure, "spitzer", temperature, .125)
        assert value == pytest.approx(.125*temperature**2.5, rel=3.e-15, abs=0.)
    assert _scalar(closure, "spitzer", 4., 0.) == 0.


@pytest.mark.parametrize("left,right,k0", [
    (1., 1., .125), (.25, 4., .3), (1.e-4, 1.e4, .1),
    (0., 3., .2), (0., 0., .2), (1., 2., 0.),
    (1.e-100, 1.e100, 1.),
])
def test_spitzer_face_harmonic_law_and_cold_limit(closure, left, right, k0):
    kl, kr = k0*left**2.5, k0*right**2.5
    expected = 0. if min(kl, kr) == 0. else 2./(1./kl+1./kr)
    actual = _scalar(closure, "spitzer_face", left, right, k0)
    assert actual == pytest.approx(expected, rel=5.e-14, abs=0.)
    assert actual == _scalar(closure, "spitzer_face", right, left, k0)
    assert min(kl, kr) <= actual <= max(kl, kr)


def test_representable_spitzer_result_survives_intermediate_extremes(closure):
    assert _scalar(closure, "spitzer", 1.e200, 1.e-300) == pytest.approx(
        1.e200, rel=3.e-13)
    assert _scalar(closure, "spitzer", 1.e-200, 1.e300) == pytest.approx(
        1.e-200, rel=3.e-13, abs=0.)
    # An intermediate subnormal must not amplify its rounding error into a
    # representable normal result when multiplied by a large prefactor.
    assert _scalar(closure, "spitzer", 1.e-129, 1.e300) == pytest.approx(
        math.sqrt(10.)*1.e-23, rel=3.e-13, abs=0.)


def test_cowie_mckee_uses_isothermal_sound_speed_and_scales_linearly(closure):
    assert _scalar(closure, "cowie", 2., 4., .3) == pytest.approx(24.)
    assert _scalar(closure, "cowie", 6., 4., .6) == pytest.approx(144.)
    assert _scalar(closure, "cowie", 1., 0., .3) == 0.
    assert _scalar(closure, "cowie", 1., 2., 0.) == 0.
    assert _scalar(closure, "cap", 24., 5.) == 5.
    assert _scalar(closure, "cap", 24., 50.) == 24.


def test_representable_cowie_mckee_result_survives_intermediate_extremes(closure):
    assert _scalar(closure, "cowie", 1.e-20, 1.e200, 1.e-300) == pytest.approx(
        5.e-20, rel=3.e-13, abs=0.)
    assert _scalar(closure, "cowie", 1.e300, 1.e-200, .3) == pytest.approx(
        1.5, rel=3.e-13, abs=0.)


@pytest.mark.parametrize("flux,q_sat,expected", [
    (0., 0., 1.), (0., 1., 1.), (1., 0., 0.), (4., 4., .5),
    (1.e-12, 1., 1./(1.+1.e-12)), (1.e12, 1., 1./(1.+1.e12)),
    (sys.float_info.max, sys.float_info.max, .5),
])
def test_harmonic_suppression_has_classical_and_saturated_limits(
        closure, flux, q_sat, expected):
    value = _scalar(closure, "factor", flux, q_sat)
    assert value == pytest.approx(expected, rel=3.e-15, abs=0.)
    assert 0. <= value <= 1.


@pytest.mark.parametrize("vector", [
    (3., 4., 0.), (0., -3., 4.), (-4., 0., -3.),
    (5./math.sqrt(3.), 5./math.sqrt(3.), 5./math.sqrt(3.)),
])
def test_vector_saturation_is_rotation_invariant_and_preserves_direction(closure, vector):
    assert _scalar(closure, "norm", *vector) == pytest.approx(5., rel=3.e-15)
    limited = _run(closure, "vector", *vector, 2.)
    assert limited == pytest.approx([component*2./7. for component in vector],
                                    rel=3.e-15, abs=1.e-15)
    assert math.hypot(*limited) == pytest.approx(10./7., rel=3.e-15)


def test_vector_saturation_handles_overflowing_norm_and_underflowing_factor(
        closure):
    classical = (1.7e308, -1.7e308, 1.7e308)
    limited = _run(closure, "vector", *classical, 1.7e308)
    amplitude = 1.7e308/(math.sqrt(3.)+1.)
    assert limited == pytest.approx([amplitude, -amplitude, amplitude], rel=3.e-15)
    assert all(math.isfinite(value) for value in limited)
    assert _run(closure, "vector", 1.e300, 0., 0., 1.e-300) == pytest.approx(
        [1.e-300, 0., 0.], rel=3.e-15, abs=0.)
    assert _run(closure, "vector", 0., 0., 0., 0.) == [0., 0., 0.]
    assert _run(closure, "vector", 3., 4., 0., 0.) == [0., 0., 0.]


@pytest.mark.parametrize("permutation", list(permutations(range(3))))
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("vector,q_sat", [
    ((1.e300, -1.e-100, 3.e-20), 1.e300),
    ((1.e300, -1.e-15, 2.e-25), 1.e300),
    ((1.e308, -1.e-10, 2.e-100), 1.e300),
    ((1.e308, -1.e-5, 1.e-10), .01),
    ((1.e300, -1.e-100, 1.e-150), 1.e-300),
    ((1.e308, -1.e-100, 1.e-300), 1.e-15),
])
def test_vector_saturation_retains_representable_small_components(
        closure, vector, q_sat, sign, permutation):
    vector = tuple(sign*vector[d] for d in permutation)
    with localcontext() as context:
        context.prec = 1800
        values = tuple(map(Decimal.from_float, vector))
        cap = Decimal.from_float(q_sat)
        norm = sum(value*value for value in values).sqrt()
        expected = [float(value*cap/(cap+norm)) for value in values]
    actual = _run(closure, "vector", *vector, q_sat)
    for value, reference in zip(actual, expected):
        assert math.isfinite(value)
        if reference == 0.0:
            # Genuine final underflow is permitted; losing a representable
            # component during normalization is not.
            assert value == 0.0
        else:
            assert value != 0.0
            assert abs(value-reference) <= max(
                8.*math.ulp(reference), 2.e-14*abs(reference))


@pytest.mark.parametrize("vector,q_sat", [
    ((0., 0., 0.), 0.), ((0., 0., 0.), 1.), ((1., 0., 0.), 0.),
    ((3., -4., 0.), 2.), ((1.e300, 0., 0.), 1.e-300),
    ((1.e-300, 0., 0.), 1.e300), ((1.e308, -1.e308, 1.e308), 1.e308),
    ((1.e-300, -2.e-300, 3.e-300), 1.e-300),
    ((1., -.75, .25), math.ulp(0.)), ((1., -.75, .25), 1.e-320),
    ((1.e300, 1.e-100, 3.e-20), 1.e301),
])
def test_vector_saturation_preserves_ordinary_arithmetic(closure, vector, q_sat):
    # Compiler contraction may differ across CPU and CUDA; this comparison
    # retains an exact ordinary-path gate on each actual execution backend.
    expected = _run(closure, "vector_legacy", *vector, q_sat)
    assert _run(closure, "vector", *vector, q_sat) == expected


def test_cgs_conversion_reproduces_physical_heat_flux(closure):
    kappa0 = 4.6e-7
    pressure, velocity, length, temperature = 3.e-12, 2.e7, 5.e20, 4.e6
    code_coeff = _scalar(
        closure, "units", kappa0, pressure, velocity, length, temperature)
    theta, grad_theta = .7, .02
    flux_code = _scalar(closure, "spitzer", theta, code_coeff)*grad_theta
    flux_cgs = kappa0*(temperature*theta)**2.5*(temperature*grad_theta/length)
    assert flux_code*pressure*velocity == pytest.approx(flux_cgs, rel=2.e-14)
    # Changing T_unit by a factor four changes kappa0_code by 4^(7/2).
    alternate = _scalar(
        closure, "units", kappa0, pressure, velocity, length, 4*temperature)
    assert alternate/code_coeff == pytest.approx(128., rel=2.e-14)


def test_analytic_spitzer_steady_profile_has_constant_flux(closure):
    # K(theta)*dtheta/dx is constant when theta^(7/2) is affine in x.
    left, right, kappa0 = 2., 3., .2
    slope = right**3.5-left**3.5
    expected = kappa0*slope/3.5
    for x in (0., .1, .4, .8, 1.):
        theta = (left**3.5+slope*x)**(1./3.5)
        derivative = slope/(3.5*theta**2.5)
        flux = _scalar(closure, "spitzer", theta, kappa0)*derivative
        assert flux == pytest.approx(expected, rel=5.e-15)

    # Harmonic face coefficients are second-order consistent on this smooth
    # profile; they are not exactly steady at finite mesh spacing.
    errors = []
    for cells in (16, 32, 64):
        dx = 1./cells
        theta_l = (left**3.5+slope*(.5-.5*dx))**(1./3.5)
        theta_r = (left**3.5+slope*(.5+.5*dx))**(1./3.5)
        k_l = _scalar(closure, "spitzer", theta_l, kappa0)
        k_r = _scalar(closure, "spitzer", theta_r, kappa0)
        face = _scalar(closure, "harmonic", k_l, k_r)
        errors.append(abs(face*(theta_r-theta_l)/dx-expected))
    assert errors[1]/errors[0] < .27
    assert errors[2]/errors[1] < .27


@pytest.mark.parametrize("rho,theta,expected", [
    (1., 0., 1.), (1., 2., 1.), (0., 1., 0.), (-1., 1., 0.),
    (1., -1., 0.), (math.nan, 1., 0.), (1., math.inf, 0.),
])
def test_temperature_validity_is_explicit(closure, rho, theta, expected):
    assert _scalar(closure, "valid", rho, theta) == expected


@pytest.mark.parametrize("operation,args", [
    ("harmonic", (-1., 2.)), ("harmonic", (math.nan, 2.)),
    ("spitzer", (-1., 1.)), ("spitzer", (1., -1.)),
    ("cowie", (0., 1., .3)), ("cowie", (1., -1., .3)),
    ("cowie", (1., 1., -1.)), ("cap", (1., -1.)),
    ("factor", (math.inf, 1.)), ("factor", (1., -1.)),
    ("norm", (math.nan, 1., 0.)), ("vector", (1., 0., math.inf, 1.)),
    ("units", (1., 1., 1., 0., 1.)), ("units", (1., 1., 1., 1., math.inf)),
])
def test_invalid_arguments_are_not_silently_floored(closure, operation, args):
    assert all(math.isnan(value) for value in _run(closure, operation, *args))
