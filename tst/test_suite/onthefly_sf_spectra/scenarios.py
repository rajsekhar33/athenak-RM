"""Analytic entropy-wave, diagnostic restart and input-contract checks.

Uses the normal prebuilt test binary and only NumPy; the feature-branch campaign
and external post-processing repository are not dependencies. ATHENAK_SF_LAUNCHER
can supply an explicit launcher, e.g. 'mpirun -np 2' for two GPUs on one node.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from test_suite.imported_features.feature_utils import ATHENA, INPUTS, _launcher

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vis" / "python"))
from sf_spectra_read import read_sf, read_spec  # noqa: E402


RANKS = {"cpu": 1, "mpicpu": 4, "gpu": 1, "mpigpu": 2}


def _run(directory, backend, overrides=(), restart=None):
    directory.mkdir(parents=True, exist_ok=True)
    if "ATHENAK_SF_LAUNCHER" in os.environ:
        launcher = shlex.split(os.environ["ATHENAK_SF_LAUNCHER"])
        devices = ["--kokkos-map-device-id-by=mpi_rank"] if "gpu" in backend else []
    else:
        launcher, devices = _launcher(backend)
    command = [*launcher, str(ATHENA), *devices]
    if restart is not None:
        command += ["-r", str(restart)]
    command += ["-i", str(INPUTS / "onthefly_sf_spectra.athinput"), *overrides]
    result = subprocess.run(
        command, cwd=directory, capture_output=True, text=True, timeout=180,
        check=False,
    )
    stem = "restart" if restart is not None else "athena"
    (directory / f"{stem}.stdout.log").write_text(result.stdout)
    (directory / f"{stem}.stderr.log").write_text(result.stderr)
    result.check_returncode()
    banners = [line.strip() for line in result.stdout.splitlines()
               if "Number of parallel ranks" in line]
    assert banners == [f"Number of parallel ranks = {RANKS[backend]}"]


def _read(path):
    """Exercise the public reader and require finite scalar diagnostics."""
    metadata, arrays = (read_sf if path.suffix == ".sf" else read_spec)(path)
    assert metadata["mode"] == "scalar"
    for name, array in arrays.items():
        assert np.isfinite(array).all(), (path, name)
    return metadata, arrays


def _initial(directory, extension):
    path = next(directory.glob(f"**/Diagnostics.00000.{extension}"))
    metadata, arrays = _read(path)
    assert int(metadata["cycle"]) == 0
    return metadata, arrays


def _sf_reference(edges):
    """Enumerate unordered cell pairs independently on the 8-cubed lattice."""
    cells = np.indices((8, 8, 8)).reshape(3, -1).T
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * (cells[:, 2] + 0.5) / 8.0)
    left, right = np.triu_indices(len(cells), 1)
    displacement = np.abs(cells[right] - cells[left])
    displacement = np.minimum(displacement, 8 - displacement) / 8.0
    radius = np.sqrt(np.sum(displacement**2, axis=1))
    delta = np.abs(rho[right] - rho[left])
    count = np.histogram(radius, bins=edges)[0]
    moments = np.array([np.histogram(radius, bins=edges, weights=delta**p)[0]
                        for p in (1, 2, 3)])
    return count, moments


def _sampled_sum(actual, replicates, expected):
    """Eight-SE guard with a separate precision requirement and exact-zero checks."""
    error = np.std(replicates, axis=0, ddof=1) / np.sqrt(len(replicates))
    tolerance = 8.0 * error + 1.0e-10 * np.maximum(np.abs(expected), 1.0)
    assert np.all(np.abs(actual - expected) <= tolerance)
    populated = expected > 0.0
    assert np.all(error[populated] <= 0.05 * expected[populated])
    np.testing.assert_allclose(actual[~populated], 0.0, atol=1.0e-12)


def analytic(root: Path, backend: str) -> None:
    """Check exact spectral power and sampled p=1..3 moments at initialization."""
    directory = root / "analytic"
    _run(directory, backend, ["time/nlim=0"])
    metadata, sf = _initial(directory, "sf")
    assert int(metadata["nranks"]) == RANKS[backend]
    count, moment = _sf_reference(sf["rbins"])
    _sampled_sum(sf["C"][0], sf["C_er"].sum(0)[:, 0], count)
    _sampled_sum(sf["T"][0, 0], sf["T_er"].sum(0)[:, 0, 0], moment)
    np.testing.assert_allclose(sf["W"], 2.0 * sf["C"], rtol=1.0e-12)
    np.testing.assert_array_equal(sf["T"][0, 1], 0.0)  # vy is exactly zero
    np.testing.assert_allclose(sf["T_er"].sum(0).mean(0), sf["T"], rtol=1.0e-12)

    metadata, spec = _initial(directory, "spec")
    assert int(metadata["nranks"]) == RANKS[backend]
    assert metadata["shape"] == "8,8,8"
    edges = spec["kbins"]
    expected = np.histogram([0.0, 1.0], bins=edges, weights=[512., 512.*0.2**2/2])[0]
    k = np.fft.fftfreq(8) * 8
    kz, ky, kx = np.meshgrid(k, k, k, indexing="ij")
    mode_count = np.histogram(np.sqrt(kx*kx + ky*ky + kz*kz), bins=edges)[0]
    np.testing.assert_array_equal(spec["nk"], mode_count)
    np.testing.assert_allclose(spec["E"][0, 0], expected, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(spec["P_DC"], 512.0, rtol=1.0e-12)
    np.testing.assert_allclose(spec["sumsq"], 512.0 * (1.0 + 0.2**2/2), rtol=1.0e-12)
    np.testing.assert_allclose(spec["E"].sum(), spec["sumsq"].sum(), rtol=1.0e-12)
    assert list((directory / "bin").glob("*.bin"))


def restart(root: Path, backend: str) -> None:
    """Preserve sampled output numbers and spectral arrays across a split run."""
    continuous, split = root / "continuous", root / "split"
    _run(continuous, backend)
    _run(split, backend, ["time/nlim=2"])
    checkpoint = split / "rst" / "Diagnostics.00001.rst"
    assert checkpoint.is_file()
    _run(split, backend, restart=checkpoint)
    for suffix in ("sf", "spec"):
        files = sorted(continuous.glob(f"**/*.{suffix}"))
        assert len(files) == 6  # cycles 0..4 and the final duplicate
        assert len(list(split.glob(f"**/*.{suffix}"))) == 6
        for path in files:
            original_meta, original = _read(path)
            resumed_meta, resumed = _read(split / path.relative_to(continuous))
            for key in ("cycle", "file_number", "time", "nranks"):
                assert original_meta[key] == resumed_meta[key]
            for key in original:
                np.testing.assert_allclose(
                    resumed[key], original[key], rtol=1.0e-10, atol=1.0e-12,
                    err_msg=f"{path.name}: {key}",
                )


def guards(root: Path, backend: str) -> None:
    """Unsupported spectrum restrictions must fail before producing diagnostics."""
    directory = root / "invalid_region"
    with pytest.raises(subprocess.CalledProcessError):
        _run(directory, backend, ["output2/x1_max=0.5", "time/nlim=0"])
    text = (directory / "athena.stdout.log").read_text()
    assert "x1_max is not supported; spec transforms the whole mesh" in text
    assert not list(directory.glob("**/*.spec"))


def spectral_edges(root: Path, backend: str) -> None:
    """Retain every mode at a non-cubic Fourier corner on either backend."""
    for logarithmic in (False, True):
        directory = root / ("edge_log" if logarithmic else "edge_linear")
        _run(directory, backend, [
            "time/nlim=0", "mesh/x1max=0.7", "mesh/x2max=7.0", "mesh/x3max=2.0",
            "output1/dcycle=0", "output2/x1_max=0.7",
            "output2/logscale=" + str(logarithmic).lower(),
        ])
        metadata, spec = _initial(directory, "spec")
        assert metadata["shape"] == "8,8,8"
        assert spec["nk"].sum() == 512 - int(logarithmic)
        reference = spec["sumsq"].sum()
        if logarithmic:
            reference -= spec["P_DC"].sum()
        np.testing.assert_allclose(spec["E"].sum(), reference, rtol=1.0e-12)


CASES = [("analytic", analytic), ("restart", restart), ("guards", guards),
         ("spectral_edges", spectral_edges)]
