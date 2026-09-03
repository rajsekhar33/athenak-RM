"""Shared execution and file-format helpers for public-main feature regressions."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import subprocess
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ATHENA = Path(
    os.environ.get(
        "ATHENAK_FEATURE_TEST_BINARY",
        REPO_ROOT / "tst" / "build" / "src" / "athena",
    )
)
INPUTS = REPO_ROOT / "tst" / "inputs"


def _launcher(backend: str) -> tuple[list[str], list[str]]:
    """Return the Slurm-aware launcher and Athena device arguments for one backend."""
    use_srun = os.environ.get("ATHENAK_FEATURE_LAUNCHER") == "srun"
    if use_srun:
        launchers = {
            "cpu": ["srun", "-N", "1", "-n", "1", "--kill-on-bad-exit=1"],
            "mpicpu": [
                "srun", "-N", "2", "-n", "4", "--ntasks-per-node=2",
                "--kill-on-bad-exit=1",
            ],
            "gpu": ["srun", "-N", "1", "-n", "1", "--kill-on-bad-exit=1"],
            "mpigpu": [
                "srun", "-N", "2", "-n", "2", "--ntasks-per-node=1",
                "--kill-on-bad-exit=1",
            ],
        }
        if backend not in launchers:
            raise RuntimeError(f"Unknown public-feature backend: {backend}")
        device_args = (
            ["--kokkos-map-device-id-by=mpi_rank"]
            if backend in {"gpu", "mpigpu"}
            else []
        )
        return launchers[backend], device_args
    if backend == "mpicpu":
        return ["mpirun", "-np", "4"], []
    if backend == "mpigpu":
        raise RuntimeError("mpigpu requires ATHENAK_FEATURE_LAUNCHER=srun")
    if backend == "gpu":
        return [], []
    if backend == "cpu":
        return [], []
    raise RuntimeError(f"Unknown public-feature backend: {backend}")


def run_athena(
    run_dir: Path,
    input_name: str,
    backend: str,
    overrides: Iterable[str] = (),
    restart: Path | None = None,
    particle_restart: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one test in an isolated directory using the test-suite build."""
    if not ATHENA.is_file():
        raise RuntimeError(f"Athena test binary is missing: {ATHENA}")

    run_dir.mkdir(parents=True, exist_ok=True)
    launcher, device_args = _launcher(backend)
    command = [*launcher, str(ATHENA), *device_args]
    if restart is not None:
        command.extend(["-r", str(restart)])
    if particle_restart is not None:
        command.extend(["-p", str(particle_restart)])
    command.extend(["-i", str(INPUTS / input_name), *overrides])
    completed = subprocess.run(
        command,
        cwd=run_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    (run_dir / "athena.stdout.log").write_text(completed.stdout)
    (run_dir / "athena.stderr.log").write_text(completed.stderr)
    completed.check_returncode()
    return completed


def latest(path: Path, pattern: str) -> Path:
    """Return the last lexicographic dump, failing clearly when none was produced."""
    matches = sorted(path.glob(pattern))
    if not matches:
        raise AssertionError(f"No files matching {pattern!r} below {path}")
    return matches[-1]


def initial(path: Path, pattern: str) -> Path:
    """Return the initial dump, whose numeric suffix sorts first."""
    matches = sorted(path.glob(pattern))
    if not matches:
        raise AssertionError(f"No files matching {pattern!r} below {path}")
    return matches[0]


def read_ascii_binary(path: Path) -> tuple[dict[str, str], bytes]:
    """Read Athena's ASCII-metadata/raw-native-Real profile or PDF format."""
    metadata: dict[str, str] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise AssertionError(f"{path} ended before its header offset")
            key, separator, value = line.decode("ascii").strip().partition("=")
            if not separator:
                raise AssertionError(f"Malformed metadata line in {path}: {line!r}")
            metadata[key] = value
            if key == "header offset":
                stream.seek(int(value))
                return metadata, stream.read()


def read_real_payload(
    path: Path, real_size: int | None = None
) -> tuple[dict[str, str], np.ndarray]:
    """Read a raw profile/PDF payload using the precision declared by its bins file."""
    metadata, payload = read_ascii_binary(path)
    if real_size is None:
        real_size = int(metadata.get("size of Real", "8"))
    dtype = np.dtype("=f8" if real_size == 8 else "=f4")
    if len(payload) % dtype.itemsize != 0:
        raise AssertionError(f"Payload size in {path} is not a whole number of Reals")
    return metadata, np.frombuffer(payload, dtype=dtype).copy()


def read_particle_binary(path: Path) -> dict[str, np.ndarray | float | int | list[str]]:
    """Read the documented pbin layout without assuming rank-local ordering."""
    raw = path.read_bytes()
    if len(raw) < 64:
        raise AssertionError(f"Particle binary file is too short: {path}")
    magic, count, nrdata, nidata, ngrid = struct.unpack_from("=5q", raw, 0)
    if magic != 43:
        raise AssertionError(f"Unexpected particle-binary magic {magic} in {path}")
    time, dt, ncycle = struct.unpack_from("=3d", raw, 40)
    offset = 64
    names: list[str] = []
    for _ in range(ngrid):
        names.append(raw[offset:offset + 16].split(b"\0", 1)[0].decode("ascii"))
        offset += 16

    real_count = nrdata * count
    real_bytes = real_count * np.dtype("=f8").itemsize
    real_data = np.frombuffer(raw, dtype="=f8", count=real_count, offset=offset).copy()
    offset += real_bytes
    int_count = nidata * count
    int_bytes = int_count * np.dtype("=i4").itemsize
    int_data = np.frombuffer(raw, dtype="=i4", count=int_count, offset=offset).copy()
    offset += int_bytes
    grid_count = ngrid * count
    grid_data = np.frombuffer(raw, dtype="=f8", count=grid_count, offset=offset).copy()
    offset += grid_count * np.dtype("=f8").itemsize
    if offset != len(raw):
        raise AssertionError(f"Unexpected trailing bytes in particle binary file {path}")
    return {
        "count": count,
        "nrdata": nrdata,
        "nidata": nidata,
        "ngrid": ngrid,
        "time": time,
        "dt": dt,
        "ncycle": int(round(ncycle)),
        "names": names,
        "real": real_data.reshape(nrdata, count),
        "integer": int_data.reshape(nidata, count),
        "grid": grid_data.reshape(ngrid, count),
    }


def sorted_particles(
    data: dict[str, np.ndarray | float | int | list[str]]
) -> dict[str, np.ndarray]:
    """Sort pbin arrays by persistent tag (integer row 1)."""
    integer = data["integer"]
    real = data["real"]
    grid = data["grid"]
    assert isinstance(integer, np.ndarray)
    assert isinstance(real, np.ndarray)
    assert isinstance(grid, np.ndarray)
    order = np.argsort(integer[1], kind="stable")
    return {"real": real[:, order], "integer": integer[:, order], "grid": grid[:, order]}


def periodic_displacement(
    current: np.ndarray, initial: np.ndarray, length: float
) -> np.ndarray:
    """Return the minimum signed displacement on a periodic interval."""
    return (current - initial + 0.5 * length) % length - 0.5 * length


def history_rows(path: Path) -> np.ndarray:
    """Load numeric rows from an Athena history file, ignoring metadata comments."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.size == 0:
        raise AssertionError(f"History file has no numeric rows: {path}")
    return rows
