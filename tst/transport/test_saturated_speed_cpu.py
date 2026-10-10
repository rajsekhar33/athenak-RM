"""Pure saturation donor-speed checks, not PDE/STS/AMR acceptance.

Set ATHENAK_SATURATED_SPEED_EXE to a precompiled wrapper to avoid building while
testing. The --prepare-wrapper DIR entry point writes sources and prints a
compile command, but never compiles or executes the helper. This permits the
parent to compile on the compile host and run pytest only on della-vis1.
"""

from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import shlex
import subprocess

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
  if (op == "speed" && argc == 9) {
    std::cout << conduction::SaturatedDonorSpeed(
        {arg(2),arg(3),arg(4)},std::atoi(argv[5]),arg(6),arg(7),arg(8));
  } else if (op == "donor_flux" && argc == 10) {
    const Real pressure = arg(6);
    const Real rho_face = 0.5*arg(7)+0.5*arg(8);
    const Real q = conduction::CowieMcKeeFlux(rho_face,pressure/rho_face,arg(9));
    const auto flux = conduction::SaturateHeatFlux({arg(2),arg(3),arg(4)},q);
    const int dir = std::atoi(argv[5]);
    if (dir < 0 || dir > 2) return 2;
    std::cout << (dir == 0 ? flux.x : (dir == 1 ? flux.y : flux.z));
  } else {
    return 2;
  }
  return 0;
}
"""


def prepare_wrapper(build_dir):
    """Write source/stub only; refuse to overwrite earlier build evidence."""
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "saturated_speed.cpp"
    stub = build_dir / "athena.hpp"
    executable = build_dir / "saturated_speed"
    if any(path.exists() for path in (source, stub, executable)):
        raise FileExistsError("Saturated-speed wrapper directory must be new")
    stub.write_text(ATHENA_STUB, encoding="ascii")
    source.write_text(
        WRAPPER_SOURCE.replace('// CONDUCTION_TEST_CLOSURE', closure_source()),
        encoding="ascii",
    )
    command = ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               f"-I{build_dir}", f"-I{REPO_ROOT / 'src'}", str(source),
               "-o", str(executable)]
    return executable, command


@pytest.fixture(scope="module")
def speed(tmp_path_factory):
    prepared = os.environ.get("ATHENAK_SATURATED_SPEED_EXE")
    if prepared:
        executable = Path(prepared).resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            pytest.fail(
                f"Prepared saturated-speed helper is not executable: {executable}")
        return executable
    executable, command = prepare_wrapper(tmp_path_factory.mktemp("saturated_speed"))
    subprocess.run(command, check=True)
    return executable


def _scalar(executable, *args):
    result = subprocess.run([str(executable), *map(str, args)], check=True,
                            capture_output=True, text=True)
    return float(result.stdout)


def _reference(vector, direction, q, pressure, gm1):
    """Independent 120-digit evaluation of the unsimplified donor derivative."""
    with localcontext() as context:
        context.prec = 120
        components = tuple(Decimal.from_float(value) for value in vector)
        cap, p, gamma_minus_one = map(Decimal.from_float, (q, pressure, gm1))
        norm = sum(value*value for value in components).sqrt()
        if cap == 0 or norm == 0 or components[direction] == 0:
            return 0.
        value = (Decimal("1.5")*gamma_minus_one*cap/p
                 * (norm/(cap+norm))**2 * abs(components[direction])/norm)
        return float(value)


@pytest.mark.parametrize("vector,direction,q,pressure,gm1", [
    ((3., 4., 0.), 0, 2., 1., 2./3.),
    ((3., 4., 0.), 1, 2., 1., 2./3.),
    ((-3., -4., 0.), 0, 2., 1., 2./3.),
    ((1., -2., 3.), 2, .5, 4., .4),
    ((1., 0., 0.), 0, 1.e-12, .3, 1.),
    ((1., 0., 0.), 0, 1.e12, .3, 1.),
    # q/P overflows and [r/(q+r)]^2 underflows, but the speed is O(1).
    ((1., 0., 0.), 0, 1.e300, 1.e-300, 2./3.),
    # Every vector component is finite but its true Euclidean norm overflows.
    ((1.7e308, -1.7e308, 1.7e308), 1, 1.e308, 1.e308, 2./3.),
    ((1.e300, -1.e300, 1.e300), 2, 1.e-300, 1.e-300, .4),
    ((1.e-300, 2.e-300, -3.e-300), 2, 1.e300, 1.e-300, 1.e300),
    ((1.e-300, 1.e300, 0.), 0, 1.e300, 1.e-300, .4),
])
def test_weighted_speed_matches_high_precision_reference(
        speed, vector, direction, q, pressure, gm1):
    expected = _reference(vector, direction, q, pressure, gm1)
    value = _scalar(speed, "speed", *vector, direction, q, pressure, gm1)
    assert math.isfinite(expected) and expected > 0.
    assert math.isfinite(value)
    assert value == pytest.approx(expected, rel=8.e-13, abs=0.)


@pytest.mark.parametrize("vector,direction,q", [
    ((1., 2., 3.), 0, 0.),
    ((0., 0., 0.), 0, 0.),
    ((0., 0., 0.), 2, 1.e300),
    ((0., 3., 4.), 0, 2.),
])
def test_valid_zero_speed_limits_are_exact(speed, vector, direction, q):
    assert _scalar(speed, "speed", *vector, direction, q, 1., 2./3.) == 0.


def test_zero_pressure_cowie_mckee_limit_and_inconsistent_cap(speed):
    # Cowie-McKee q is proportional to P^(3/2): both q and its pressure
    # derivative vanish at zero pressure. A nonzero cap at P=0 is inconsistent.
    assert _scalar(speed, "speed", 1., 2., 3., 0, 0., 0., 2./3.) == 0.
    assert math.isnan(_scalar(speed, "speed", 1., 2., 3., 0, 1., 0., 2./3.))


@pytest.mark.parametrize("q", [1.e-8, .1, 1., 10., 1.e8])
def test_vector_projection_and_reflection(speed, q):
    vector = (3., -4., 12.)
    values = [_scalar(speed, "speed", *vector, d, q, 2., .4) for d in range(3)]
    assert values[1]/values[0] == pytest.approx(4./3., rel=8.e-13)
    assert values[2]/values[0] == pytest.approx(4., rel=8.e-13)
    for d in range(3):
        reflected = _scalar(speed, "speed", *(-x for x in vector), d, q, 2., .4)
        assert reflected == values[d]


def test_classical_limit_does_not_create_a_large_phi_timestep_penalty(speed):
    low = _scalar(speed, "speed", 1., 0., 0., 0, 1.e8, 1., 2./3.)
    high = _scalar(speed, "speed", 1., 0., 0., 0, 1.e12, 1., 2./3.)
    assert high < low
    assert high == pytest.approx(1.e-12, rel=3.e-12, abs=0.)
    assert high/low == pytest.approx(1.e-4, rel=3.e-8, abs=0.)


@pytest.mark.parametrize("vector,direction,pressure,rho_l,rho_r,phi,gamma", [
    ((3., 4., 0.), 0, 2., 1., 9., .3, 5./3.),
    ((-4., 2., 1.), 0, .8, .2, 5., .1, 1.4),
    ((1., -2., 3.), 2, 4., 100., 1.e4, 2., 2.),
    ((.3, .4, .5), 1, 10., .01, .09, .3, 5./3.),
])
def test_native_flux_pressure_derivative_at_fixed_face_density(
        speed, vector, direction, pressure, rho_l, rho_r, phi, gamma):
    # Only pressure changes. The classical vector and both face densities are
    # fixed, so this isolates exactly the extra donor response, not diffusion.
    h = 2.e-5*pressure
    plus = _scalar(speed, "donor_flux", *vector, direction,
                   pressure+h, rho_l, rho_r, phi)
    minus = _scalar(speed, "donor_flux", *vector, direction,
                    pressure-h, rho_l, rho_r, phi)
    derivative = abs((gamma-1.)*(plus-minus)/(2.*h))
    rho_face = .5*rho_l+.5*rho_r
    q = 5.*phi*pressure**1.5/math.sqrt(rho_face)
    value = _scalar(speed, "speed", *vector, direction, q, pressure, gamma-1.)
    assert value == pytest.approx(derivative, rel=3.e-7, abs=0.)


@pytest.mark.parametrize("gm1", [.1, .4, 2./3., 1., 3.])
def test_gamma_dependence_is_linear(speed, gm1):
    reference = _scalar(speed, "speed", 2., -3., 4., 1, .7, 1.3, 1.)
    assert _scalar(speed, "speed", 2., -3., 4., 1, .7, 1.3, gm1) == pytest.approx(
        gm1*reference, rel=8.e-13, abs=0.)


@pytest.mark.parametrize("vector,direction,q,pressure,gm1", [
    ((math.nan, 0., 0.), 0, 0., 1., 1.),
    ((0., math.inf, 0.), 0, 0., 1., 1.),
    ((0., 0., 0.), -1, 0., 1., 1.),
    ((0., 0., 0.), 3, 0., 1., 1.),
    ((0., 0., 0.), 0, -1., 1., 1.),
    ((0., 0., 0.), 0, math.nan, 1., 1.),
    ((0., 0., 0.), 0, math.inf, 1., 1.),
    ((0., 0., 0.), 0, 0., -1., 1.),
    ((0., 0., 0.), 0, 0., math.nan, 1.),
    ((0., 0., 0.), 0, 0., math.inf, 1.),
    ((0., 0., 0.), 0, 0., 1., 0.),
    ((0., 0., 0.), 0, 0., 1., -1.),
    ((0., 0., 0.), 0, 0., 1., math.nan),
    ((0., 0., 0.), 0, 0., 1., math.inf),
])
def test_invalid_inputs_are_not_masked_by_zero_flux(
        speed, vector, direction, q, pressure, gm1):
    assert math.isnan(_scalar(speed, "speed", *vector, direction, q, pressure, gm1))


@pytest.mark.parametrize("q,pressure,component", [
    (1.e300, 1.e-300, 1.e300),  # true positive speed overflows
    (1.e-300, 1.e300, 1.),  # true positive speed underflows
])
def test_unrepresentable_nonzero_speed_fails_closed(speed, q, pressure, component):
    assert math.isnan(_scalar(speed, "speed", component, 0., 0., 0, q, pressure, 1.))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare helper source; never compile or execute")
    parser.add_argument("--prepare-wrapper", type=Path, required=True)
    arguments = parser.parse_args()
    executable_path, compile_command = prepare_wrapper(arguments.prepare_wrapper)
    print(shlex.join(compile_command))
    print(f"ATHENAK_SATURATED_SPEED_EXE={executable_path}")
