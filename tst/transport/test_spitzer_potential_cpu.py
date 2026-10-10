"""Pure Spitzer-potential helper checks, not PDE/STS/AMR acceptance.

Use ATHENAK_SPITZER_POTENTIAL_EXE to test an already compiled wrapper. Otherwise
the fixture compiles a tiny native driver, with pytest --basetemp on scratch.
The --prepare-wrapper DIR entry point only writes that driver and its type stub;
it does not compile or run it. This supports serialized parent-owned builds.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from test_conduction_closure_cpu import closure_source


REPO_ROOT = Path(__file__).resolve().parents[2]
ATHENA_STUB = "using Real = double;\n#define KOKKOS_INLINE_FUNCTION inline\n"
WRAPPER_SOURCE = r"""
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

// CONDUCTION_TEST_CLOSURE

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const std::string op(argv[1]);
  const auto arg = [&](int i) { return std::strtod(argv[i], nullptr); };
  std::cout << std::setprecision(17);
  if (op == "temperature" && argc == 5) {
    std::cout << conduction::SpitzerTemperature(arg(2),arg(3),arg(4));
  } else if (op == "gradient" && argc == 6) {
    std::cout << conduction::SpitzerPotentialGradient(arg(2),arg(3),arg(4),arg(5));
  } else if (op == "vl" && argc == 4) {
    std::cout << conduction::SpitzerVL2(arg(2),arg(3));
  } else if (op == "mc" && argc == 4) {
    std::cout << conduction::SpitzerMC2(arg(2),arg(3));
  } else if (op == "limited4" && argc == 7) {
    std::cout << conduction::SpitzerLimited4(arg(2),arg(3),arg(4),arg(5),arg(6)!=0.0);
  } else if (op == "product" && argc == 5) {
    std::cout << conduction::SpitzerProduct3(arg(2),arg(3),arg(4));
  } else {
    return 2;
  }
  return 0;
}
"""


def prepare_wrapper(build_dir):
    """Write source/stub only; the caller controls whether/where compilation runs."""
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "athena.hpp").write_text(ATHENA_STUB, encoding="ascii")
    source = build_dir / "spitzer_potential.cpp"
    source.write_text(
        WRAPPER_SOURCE.replace('// CONDUCTION_TEST_CLOSURE', closure_source()),
        encoding="ascii",
    )
    executable = build_dir / "spitzer_potential"
    command = ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               f"-I{build_dir}", f"-I{REPO_ROOT / 'src'}", str(source),
               "-o", str(executable)]
    return executable, command


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    prepared = os.environ.get("ATHENAK_SPITZER_POTENTIAL_EXE")
    if prepared:
        executable = Path(prepared).resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            pytest.fail(f"Prepared Spitzer helper is not executable: {executable}")
        return executable
    executable, command = prepare_wrapper(tmp_path_factory.mktemp("spitzer_potential"))
    subprocess.run(command, check=True)
    return executable


def _scalar(potential, *args):
    result = subprocess.run([str(potential), *map(str, args)], check=True,
                            capture_output=True, text=True)
    return float(result.stdout)


def _gradient_reference(tl, tr, dx, k0):
    """Independent direct-potential oracle with 120 decimal digits."""
    with localcontext() as context:
        context.prec = 120
        left, right, spacing, coefficient = map(Decimal.from_float, (tl, tr, dx, k0))
        potential_difference = right**3*right.sqrt()-left**3*left.sqrt()
        return float(coefficient*potential_difference/(Decimal("3.5")*spacing))


@pytest.mark.parametrize("tl,tr,dx,k0", [
    (0., 1., 1., 1.), (.25, 1., .125, 2.), (3., 2., .75, .1),
    (1., math.nextafter(1., math.inf), 1., 1.),
    # Psi itself overflows, while this close-state gradient is representable.
    (1.e100, math.nextafter(1.e100, math.inf), 1.e100, 1.),
    # Psi underflows, while division by the small spacing recovers a normal gradient.
    (1.e-100, 2.e-100, 1.e-300, 1.),
    # Even K(Thi) underflows/overflows; the final gradient still exists.
    (1.e-10, 2.e-10, 1.e-300, 1.e-300),
    (2., math.nextafter(2., math.inf), 1., sys.float_info.max),
    (1.e200, 2.e200, 1.e300, 1.e-300),
    (1.e-200, 2.e-200, 1.e-100, 1.e300),
])
def test_gradient_matches_direct_high_precision_potential(potential, tl, tr, dx, k0):
    expected = _gradient_reference(tl, tr, dx, k0)
    value = _scalar(potential, "gradient", tl, tr, dx, k0)
    assert math.isfinite(value)
    assert value == pytest.approx(expected, rel=6.e-13, abs=0.)
    assert value == -_scalar(potential, "gradient", tr, tl, dx, k0)


@pytest.mark.parametrize("tl,tr,dx,k0", [
    (0., 0., 1., 1.), (1., 1., 1., 1.),
    (sys.float_info.max, sys.float_info.max, 1., 1.),
    (0., sys.float_info.max, 1., 0.),
])
def test_gradient_equal_state_and_zero_coefficient_are_exact(potential, tl, tr, dx, k0):
    assert _scalar(potential, "gradient", tl, tr, dx, k0) == 0.


def test_gradient_reports_final_overflow_and_allows_true_underflow(potential):
    assert _scalar(potential, "gradient", 0., 1.e200, 1., 1.) == math.inf
    assert _scalar(potential, "gradient", 1.e200, 0., 1., 1.) == -math.inf
    assert _scalar(potential, "gradient", 0., 1.e-200, 1., 1.e-300) == 0.


@pytest.mark.parametrize("ratio", [0., 1.e-20, .01, .1, .5, .9, 1.-1.e-12])
def test_secant_factor_is_bounded_and_cold_face_does_not_lock(potential, ratio):
    lo, hi, dx, k0 = 2.*ratio, 2., .3, .2
    gradient = _scalar(potential, "gradient", lo, hi, dx, k0)
    secant = gradient*dx/(hi-lo)
    assert k0*lo**2.5 <= secant <= k0*hi**2.5*(1.+2.e-15)
    q = secant/(k0*hi**2.5)
    assert 2./7.*(1.-2.e-15) <= q <= 1.+2.e-15
    if ratio == 0.:
        assert gradient > 0.
        assert q == pytest.approx(2./7., rel=2.e-15)


def test_steady_potential_profile_has_constant_discrete_face_flux(potential):
    left, right, k0 = 2., 3., .2
    slope = right**3.5-left**3.5
    expected = k0*slope/3.5
    for cells in (8, 32, 128):
        dx = 1./cells
        for face in (0, cells//2, cells-1):
            tl = (left**3.5+slope*face*dx)**(1./3.5)
            tr = (left**3.5+slope*(face+1)*dx)**(1./3.5)
            assert _scalar(potential, "gradient", tl, tr, dx, k0) == pytest.approx(
                expected, rel=6.e-13, abs=0.)


def test_gradient_endpoint_derivatives_match_local_conductivity(potential):
    tl, tr, dx, k0, h = .25, 1., .7, .3, 1.e-5
    dl = (_scalar(potential, "gradient", tl+h, tr, dx, k0)
          - _scalar(potential, "gradient", tl-h, tr, dx, k0))/(2.*h)
    dr = (_scalar(potential, "gradient", tl, tr+h, dx, k0)
          - _scalar(potential, "gradient", tl, tr-h, dx, k0))/(2.*h)
    assert dl == pytest.approx(-k0*tl**2.5/dx, rel=2.e-8)
    assert dr == pytest.approx(k0*tr**2.5/dx, rel=2.e-8)


@pytest.mark.parametrize("rho,eint,gm1,expected", [
    (2., 3., .4, .6), (1., 0., .4, 0.),
    (1.e-300, 1.e300, 1.e-300, 1.e300),
    (1.e300, 1.e-300, 1.e300, 1.e-300),
])
def test_temperature_avoids_intermediate_range_loss(potential, rho, eint, gm1, expected):
    assert _scalar(potential, "temperature", rho, eint, gm1) == pytest.approx(
        expected, rel=6.e-13, abs=0.)


def _limiter_reference(a, b, mc):
    if not ((a > 0. and b > 0.) or (a < 0. and b < 0.)):
        return 0.
    # Exact rationals preserve ties at subnormal midpoints; finite decimal
    # precision can perturb a tie and round the oracle to the wrong float.
    da, db = Fraction.from_float(abs(a)), Fraction.from_float(abs(b))
    value = min((da+db)/2, 2*min(da, db)) if mc else 2*da*db/(da+db)
    return math.copysign(float(value), a)


@pytest.mark.parametrize("op", ["vl", "mc"])
@pytest.mark.parametrize("a,b", [
    (1., 3.), (-2., -7.), (0., 4.), (-1., 2.),
    (sys.float_info.max, sys.float_info.max), (1.e308, 1.7e308),
    (1.e-300, 1.e300),
    (math.ulp(0.), math.ulp(0.)), (math.ulp(0.), 2.*math.ulp(0.)),
])
def test_limiters_are_finite_and_preserve_subnormal_slopes(potential, op, a, b):
    expected = _limiter_reference(a, b, op == "mc")
    value = _scalar(potential, op, a, b)
    assert value == pytest.approx(expected, rel=3.e-15, abs=0.)
    assert value == _scalar(potential, op, b, a)


@pytest.mark.parametrize("mc", [False, True])
def test_nested_limiter_matches_existing_mathematical_order(potential, mc):
    slopes = (2., 7., 3., 8.)
    expected = _limiter_reference(_limiter_reference(*slopes[:2], mc),
                                  _limiter_reference(*slopes[2:], mc), mc)
    assert _scalar(potential, "limited4", *slopes, int(mc)) == pytest.approx(
        expected, rel=3.e-15)
    assert _scalar(potential, "limited4", 1., -1., 3., 8., int(mc)) == 0.


@pytest.mark.parametrize("a,b,c,expected", [
    (2., 3., 4., 24.), (-2., 3., 4., -24.), (-2., -3., -4., -24.),
    (1.e300, 1.e300, 1.e-300, 1.e300),
    (1.e-300, 1.e-300, 1.e300, 1.e-300),
    (0., 1.e300, -1.e300, 0.),
])
def test_projected_product_recovers_representable_extremes(potential, a, b, c, expected):
    assert _scalar(potential, "product", a, b, c) == pytest.approx(
        expected, rel=6.e-13, abs=0.)


@pytest.mark.parametrize("operation,args", [
    ("temperature", (0., 0., 1.)), ("temperature", (1., -1., 1.)),
    ("temperature", (1., 0., math.nan)), ("temperature", (math.inf, 0., 1.)),
    ("temperature", (1., sys.float_info.max, 2.)),
    ("temperature", (sys.float_info.max, math.ulp(0.), 1.)),
    ("gradient", (-1., 1., 1., 0.)), ("gradient", (0., math.nan, 1., 0.)),
    ("gradient", (1., 1., 0., 1.)), ("gradient", (1., 1., math.inf, 1.)),
    ("gradient", (1., 1., 1., math.inf)), ("gradient", (0., 1., 1., -1.)),
    ("vl", (0., math.inf)), ("mc", (0., math.nan)),
    ("limited4", (1., -1., math.nan, 0., 0.)),
    ("limited4", (1., -1., math.nan, 0., 1.)),
    ("product", (0., 1., math.inf)), ("product", (0., math.nan, 1.)),
])
def test_invalid_inputs_are_not_hidden_by_zero_limits(potential, operation, args):
    assert math.isnan(_scalar(potential, operation, *args))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare native helper source; never compile")
    parser.add_argument("--prepare-wrapper", type=Path, required=True)
    arguments = parser.parse_args()
    executable_path, compile_command = prepare_wrapper(arguments.prepare_wrapper)
    print(shlex.join(compile_command))
    print(f"ATHENAK_SPITZER_POTENTIAL_EXE={executable_path}")
