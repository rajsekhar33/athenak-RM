"""Shared portable entry point for exact-executable transport functional tests."""

import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
CASES = (
    "disabled", "isotropic_no_units", "isotropic_n64_restart",
    "isotropic_sts_restart", "anisotropic_parallel_restart",
    "anisotropic_perpendicular", "anisotropic_out_of_plane",
    "anisotropic_reversed", "anisotropic_zero_field",
    "combined_iso_aniso", "combined_iso_aniso_sts_restart",
    "anisotropic_z", "oblique_3d",
)


def run_transport(tmp_path, limiter, distributed=False, guards=False):
    """Never compile, submit jobs, silently skip a binary, or change physics gates."""
    binary = Path(os.environ.get("ATHENAK_TEST_BINARY",
                                 REPOSITORY / "tst/build/src/athena")).resolve()
    assert binary.is_file(), (
        f"Set ATHENAK_TEST_BINARY to an existing executable: {binary}")
    launcher = (os.environ.get("ATHENAK_TEST_LAUNCHER", "mpirun -np 2")
                if distributed else "")
    ranks = os.environ.get("ATHENAK_TEST_RANKS", "2") if distributed else "1"
    assert not distributed or launcher.strip(), (
        "Distributed tests need an explicit launcher")
    assert not distributed or int(ranks) > 1, "Distributed tests need multiple ranks"
    command = [sys.executable, str(REPOSITORY / "tst/transport/run_pilot.py"),
               "--binary", str(binary), "--root", str(tmp_path / "campaign"),
               "--limiter", limiter, "--wall-minutes", "9", "--launcher", launcher,
               "--expected-ranks", ranks]
    command += ["--guards-only"] if guards else ["--only", *CASES]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # The runner limits each Athena invocation. Do not put a shorter outer
    # timeout around it, which could orphan the Athena/MPI process tree.
    result = subprocess.run(command, capture_output=True, text=True,
                            env=environment, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "campaign/results.json").read_text())
    assert report["status"] == "PASS", report
    assert report["cases"] and all(case["status"] == "PASS"
                                   for case in report["cases"].values()), report


def run_sts(tmp_path, limiter, distributed=False):
    """Bounded anisotropic RKL2 controls and real checkpoint/restart tests."""
    binary = Path(os.environ.get("ATHENAK_TEST_BINARY",
                                 REPOSITORY / "tst/build/src/athena")).resolve()
    assert binary.is_file(), (
        f"Set ATHENAK_TEST_BINARY to an existing executable: {binary}")
    launcher = (os.environ.get("ATHENAK_TEST_LAUNCHER", "mpirun -np 2")
                if distributed else "")
    ranks = os.environ.get("ATHENAK_TEST_RANKS", "2") if distributed else "1"
    assert not distributed or launcher.strip(), (
        "Distributed tests need an explicit launcher")
    assert not distributed or int(ranks) > 1, "Distributed tests need multiple ranks"
    command = [sys.executable, str(REPOSITORY / "tst/transport/run_sts.py"),
               "--binary", str(binary), "--root", str(tmp_path / "sts"),
               "--limiter", limiter, "--launcher", launcher, "--expected-ranks", ranks]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, capture_output=True, text=True,
                            env=environment, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "sts/results.json").read_text())
    assert report["status"] == "PASS", report
    assert len(report["cases"]) == 18 and len(report["comparisons"]) == 12, report
    assert all(case["status"] == "PASS" for case in report["cases"].values()), report
    assert all(case["status"] == "PASS"
               for case in report["comparisons"].values()), report


def run_auxiliary(tmp_path, matrix):
    """Run bounded localized-pulse or boundary-budget coverage, one process only."""
    script, option = {"gaussian": ("run_gaussian.py", "--smoke"),
                      "boundary": ("run_equilibrium.py", "--boundary-only")}[matrix]
    binary = Path(os.environ.get("ATHENAK_TEST_BINARY",
                                 REPOSITORY / "tst/build/src/athena")).resolve()
    assert binary.is_file(), (
        f"Set ATHENAK_TEST_BINARY to an existing executable: {binary}")
    command = [sys.executable, str(REPOSITORY / "tst/transport" / script),
               "--binary", str(binary), "--root", str(tmp_path / "campaign"), option]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "campaign/results.json").read_text())
    assert report["status"] == "PASS", report
    assert report["cases"] and all(case["status"] == "PASS"
                                   for case in report["cases"].values()), report
