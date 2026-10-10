"""Focused unit tests for the host-side RKL2 controller."""

import math
from pathlib import Path
import subprocess

import pytest
from numpy.polynomial.legendre import legval


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def controller(tmp_path_factory):
    """Build the controller with a minimal athena.hpp type stub."""
    build_dir = tmp_path_factory.mktemp("sts_rkl2_controller")
    (build_dir / "athena.hpp").write_text("using Real = double;\n", encoding="ascii")
    source = build_dir / "controller.cpp"
    source.write_text(
        r"""
#include <iomanip>
#include <iostream>
#include <string>

#include "diffusion/sts_rkl2.hpp"

int main(int argc, char **argv) {
  std::string operation(argv[1]);
  std::cout << std::setprecision(17);
  if (operation == "stages") {
    std::cout << parabolic::ComputeRKL2StageCount(std::stod(argv[2]),
                                                  std::stod(argv[3]));
    return 0;
  }
  auto coeffs = parabolic::ComputeRKL2Coefficients(std::stoi(argv[2]),
                                                   std::stoi(argv[3]));
  std::cout << coeffs.muj << " " << coeffs.nuj << " "
            << coeffs.muj_tilde << " " << coeffs.gammaj_tilde;
  return 0;
}
""",
        encoding="ascii",
    )
    executable = build_dir / "controller"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            f"-I{build_dir}",
            f"-I{REPO_ROOT / 'src'}",
            str(source),
            str(REPO_ROOT / "src/diffusion/sts_rkl2.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def _run(controller, *args):
    result = subprocess.run(
        [str(controller), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize("nstages", [3, 5, 7, 101, 46341])
def test_exact_odd_threshold_selects_smallest_stage_count(controller, nstages):
    ratio = (nstages*nstages + nstages - 2)/4
    assert int(_run(controller, "stages", ratio, 1.0)) == nstages
    assert int(_run(controller, "stages", math.nextafter(ratio, math.inf), 1.0)) == (
        nstages + 2
    )


def test_zero_ratio_retains_three_stage_minimum(controller):
    assert int(_run(controller, "stages", 0.0, 1.0)) == 3


def test_large_stage_coefficients_do_not_overflow_integer_products(controller):
    stage = 1_000_000_001
    values = [float(value) for value in _run(controller, "coeff", stage, stage).split()]
    jr = float(stage)
    bj = (jr*jr + jr - 2.0)/(2.0*jr*(jr + 1.0))
    jm1 = jr - 1.0
    bj_m1 = (jm1*jm1 + jm1 - 2.0)/(2.0*jm1*(jm1 + 1.0))
    jm2 = jr - 2.0
    bj_m2 = (jm2*jm2 + jm2 - 2.0)/(2.0*jm2*(jm2 + 1.0))
    muj = ((2.0*jr - 1.0)/jr)*bj/bj_m1
    nuj = -((jr - 1.0)/jr)*bj/bj_m2
    denom = jr*jr + jr - 2.0
    muj_tilde = muj*4.0/denom
    gammaj_tilde = -(1.0 - bj_m1)*muj_tilde

    assert all(math.isfinite(value) for value in values)
    assert values == pytest.approx(
        [muj, nuj, muj_tilde, gammaj_tilde], rel=1.0e-14
    )


def test_unrepresentable_stage_count_fails_cleanly(controller):
    result = subprocess.run(
        [str(controller), "stages", "1e300", "1.0"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "stage count exceeds the supported integer range" in result.stdout


def _native_coefficients(controller, stages):
    return [[float(v) for v in _run(controller, 'coeff', j, stages).split()]
            for j in range(1, stages+1)]


def _recurrence(coefficients, z):
    """Apply exported native coefficients to y'=lambda*y, y(0)=1."""
    older = previous = 1.0
    values = []
    for mu, nu, mt, gt in coefficients:
        current = mu*previous + nu*older + (1-mu-nu) + mt*z*previous + gt*z
        values.append(current)
        older, previous = previous, current
    return values


@pytest.mark.parametrize('stages', [3, 5, 7, 11, 23, 33])
def test_every_native_stage_matches_independent_legendre_polynomial(controller, stages):
    # Polynomial evaluation uses NumPy's Clenshaw algorithm, not the native
    # coefficient recurrence. This is a scalar linear check, NOT positivity
    # acceptance for a nonlinear limited spatial operator.
    coefficients = _native_coefficients(controller, stages)
    denominator = stages*(stages+1)-2
    for fraction in (0, .001, .1, .5, .9, 1):
        z = -.5*denominator*fraction
        actual = _recurrence(coefficients, z)
        for j, value in enumerate(actual, 1):
            bj = 1/3 if j < 3 else (j*j+j-2)/(2*j*(j+1))
            expected = 1-bj+bj*legval(1+4*z/denominator, [0]*j+[1])
            assert value == pytest.approx(expected, rel=0, abs=3e-12)
            assert abs(value) <= 1+3e-12


@pytest.mark.parametrize('stages', [3, 5, 11])
def test_native_recurrence_has_second_order_time_convergence(controller, stages):
    coefficients = _native_coefficients(controller, stages)
    errors = [abs(_recurrence(coefficients, -1/n)[-1]**n-math.exp(-1))
              for n in (8, 16, 32)]
    assert errors[1]/errors[0] < .28
    assert errors[2]/errors[1] < .28


@pytest.mark.parametrize('cycle_ratio,stages', [(1, 3), (4, 3), (16, 7), (64, 11), (256, 23)])
def test_native_controller_uses_half_sweep_ratio(controller, cycle_ratio, stages):
    assert int(_run(controller, 'stages', cycle_ratio/2, 1)) == stages


@pytest.mark.parametrize('dt,dt_fe', [
    ('nan', '1'), ('inf', '1'), ('1', 'nan'), ('1', 'inf'),
    ('0', 'inf'), ('-1', '1'), ('1', '0'), ('1', '-1')])
def test_invalid_controller_times_rejected(controller, dt, dt_fe):
    result = subprocess.run([str(controller), 'stages', dt, dt_fe],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert 'FATAL ERROR' in result.stdout
