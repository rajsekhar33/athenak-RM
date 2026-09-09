"""Backend-independent assertions for features imported into public main."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from test_suite.imported_features.feature_utils import (
    history_rows,
    initial,
    latest,
    periodic_displacement,
    read_ascii_binary,
    read_particle_binary,
    read_real_payload,
    run_athena,
    sorted_particles,
)


RTOL = 1.0e-6
ATOL = 1.0e-10
HOOK_RANKS = {"cpu": 1, "mpicpu": 4, "gpu": 1, "mpigpu": 2}


def _profile(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata, payload = read_real_payload(path)
    nbin = int(metadata["nbin"])
    return metadata, payload.reshape(3, nbin + 2)


def _pdf(path: Path, real_size: int) -> tuple[dict[str, str], np.ndarray]:
    metadata, payload = read_real_payload(path, real_size)
    nbin = int(metadata["nbin"])
    nbin2 = int(metadata.get("nbin2", "0"))
    rows = nbin2 + 2 if nbin2 else 1
    return metadata, payload.reshape(rows, nbin + 2)


def verify_hooks_and_geometry(run_dir: Path, backend: str) -> None:
    """Check hook and geometry artifacts produced by an external Athena run."""
    history = history_rows(run_dir / "PublicHookGeometry.user.hst")
    assert backend in HOOK_RANKS
    # UserTimeStep provides the raw cap (0.01); Mesh::NewTimeStep applies the
    # configured CFL number (0.4), so the observable integration step is 0.004.
    # The final history row is the driver's terminal duplicate at nlim = 4.
    expected_dt = 0.4 * 0.01
    np.testing.assert_allclose(
        history[:, 0], [0.0, expected_dt, 2.0 * expected_dt,
                        3.0 * expected_dt, 4.0 * expected_dt, 4.0 * expected_dt],
    )
    np.testing.assert_allclose(history[:, 1], expected_dt)
    np.testing.assert_allclose(
        history[:, 2], np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.0]) * HOOK_RANKS[backend],
    )

    _, x1_profile = _profile(initial(run_dir, "prof_dens_x1/*.prof"))
    np.testing.assert_allclose(x1_profile[0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(x1_profile[1], [0.0, 2.0, 2.0, 2.0, 2.0, 0.0])
    np.testing.assert_allclose(x1_profile[2], [0.0, 2.0, 2.0, 2.0, 2.0, 0.0])

    _, radial_profile = _profile(initial(run_dir, "prof_dens_cylr/*.prof"))
    # Eight-by-eight cell centers sample the three cylindrical-radius bins in
    # 12/40/12 proportions. The profile's geometry normalization yields these
    # exact discrete weights for the valid four-MeshBlock test domain.
    np.testing.assert_allclose(radial_profile[0], [0.0, 0.75, 2.5, 0.75, 0.0])
    np.testing.assert_allclose(radial_profile[2], [0.0, 2.0, 2.0, 2.0, 0.0])

    bins_meta, _ = read_ascii_binary(
        run_dir / "pdf_dens_slab" / "PublicHookGeometry.bins.pdf"
    )
    _, slab_pdf = _pdf(
        initial(run_dir, "pdf_dens_slab/*.pdf"), int(bins_meta["size of Real"])
    )
    assert bins_meta["pdf_dimension"] == "1"
    np.testing.assert_allclose(slab_pdf, [[0.0, 0.0, 4.0, 0.0]])


def check_hooks_and_geometry(root: Path, backend: str) -> None:
    """Launch and verify the hook/geometry regression for pytest compatibility."""
    run_dir = root / "hooks_geometry"
    run_athena(run_dir, "public_main_hook_geometry.athinput", backend)
    verify_hooks_and_geometry(run_dir, backend)


def verify_profile_and_pdf_exactness(run_dir: Path) -> None:
    """Check profile/PDF artifacts produced by an external Athena run."""
    _, profile = _profile(initial(run_dir, "prof_temp_vs_x1/*.prof"))
    np.testing.assert_allclose(profile[0], [0.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(profile[2], [0.0, 1.0, 0.8, 0.0], rtol=RTOL, atol=ATOL)

    bins_meta, _ = read_ascii_binary(run_dir / "pdf_pdf_temp" / "ProfTempTest.bins.pdf")
    _, temperature_pdf = _pdf(
        initial(run_dir, "pdf_pdf_temp/*.pdf"), int(bins_meta["size of Real"])
    )
    np.testing.assert_allclose(temperature_pdf, [[0.0, 0.5, 0.5, 0.0]])

    bins_meta, _ = read_ascii_binary(
        run_dir / "pdf_pdf2d_dens_temp_temperature" / "ProfTempTest.bins.pdf"
    )
    _, joint_pdf = _pdf(
        initial(run_dir, "pdf_pdf2d_dens_temp_temperature/*.pdf"),
        int(bins_meta["size of Real"]),
    )
    expected = np.zeros((4, 4))
    expected[1, 1] = 0.5
    expected[2, 2] = 0.5
    np.testing.assert_allclose(joint_pdf, expected)


def check_profile_and_pdf_exactness(root: Path, backend: str) -> None:
    """Launch and verify the profile/PDF regression for pytest compatibility."""
    run_dir = root / "profile_pdf"
    run_athena(run_dir, "profile_temperature_test.athinput", backend)
    verify_profile_and_pdf_exactness(run_dir)


def verify_dt_wall(run_dir: Path) -> None:
    """Check that a wall-clock-only output emitted during evolution."""
    # One file comes from Driver::Initialize(). A second proves the wall-clock-only
    # trigger was retained in the output list and fired during time evolution.
    assert len(list((run_dir / "tab").glob("*.tab"))) >= 2


def check_dt_wall(root: Path, backend: str) -> None:
    """Launch and verify the wall-clock output regression for pytest compatibility."""
    run_dir = root / "dt_wall"
    run_athena(run_dir, "public_main_dt_wall.athinput", backend)
    verify_dt_wall(run_dir)


def verify_turbulence_restart(reference_dir: Path, split_dir: Path) -> None:
    """Compare completed uninterrupted and split turbulence histories."""

    reference = history_rows(reference_dir / "PublicTurbulence.hydro.hst")
    restarted = history_rows(split_dir / "PublicTurbulence.hydro.hst")
    assert reference[-1, 0] == restarted[-1, 0]
    np.testing.assert_allclose(restarted[-1], reference[-1], rtol=RTOL, atol=ATOL)


def check_turbulence_restart(root: Path, backend: str) -> None:
    """Launch and verify the turbulence restart regression for pytest compatibility."""
    reference_dir = root / "turbulence_reference"
    split_dir = root / "turbulence_restart"
    run_athena(reference_dir, "public_main_turbulence.athinput", backend)
    run_athena(split_dir, "public_main_turbulence.athinput", backend, ["time/nlim=2"])
    fluid_restart = latest(split_dir, "rst/*.rst")
    run_athena(
        split_dir,
        "public_main_turbulence.athinput",
        backend,
        restart=fluid_restart,
    )
    verify_turbulence_restart(reference_dir, split_dir)


def _particle_output(run_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Read initial and terminal particle outputs from one completed run directory."""
    start = read_particle_binary(initial(run_dir, "pbin/*.prtclbin"))
    finish = read_particle_binary(latest(run_dir, "pbin/*.prtclbin"))
    return start, finish


def _particle_run(
    root: Path, backend: str, pusher: str, nlim: int = 8
) -> tuple[Path, dict[str, object], dict[str, object]]:
    run_dir = root / f"particles_{pusher}_{nlim}"
    run_athena(
        run_dir,
        "public_main_particles.athinput",
        backend,
        [f"particles/pusher={pusher}", f"time/nlim={nlim}"],
    )
    start, finish = _particle_output(run_dir)
    return run_dir, start, finish


def _check_particle_transport(
    start: dict[str, object], finish: dict[str, object], pusher: str
) -> None:
    start_sorted = sorted_particles(start)  # type: ignore[arg-type]
    finish_sorted = sorted_particles(finish)  # type: ignore[arg-type]
    assert start_sorted["real"].shape == finish_sorted["real"].shape
    assert np.array_equal(start_sorted["integer"][1], finish_sorted["integer"][1])
    assert np.all(np.isfinite(finish_sorted["real"]))
    assert np.all((finish_sorted["real"][0] >= 0.0) & (finish_sorted["real"][0] < 1.0))
    assert np.all((finish_sorted["real"][1] >= 0.0) & (finish_sorted["real"][1] < 1.0))

    displacement = periodic_displacement(
        finish_sorted["real"][0], start_sorted["real"][0], 1.0
    )
    expected_mean = 0.1 * float(finish["time"])
    if pusher == "classical":
        np.testing.assert_allclose(displacement, expected_mean, rtol=RTOL, atol=ATOL)
    else:
        assert abs(float(np.mean(displacement)) - expected_mean) < 0.025
    if pusher == "lagrangian_mc":
        cell_phase = np.mod(finish_sorted["real"][0] / 0.125 - 0.5, 1.0)
        np.testing.assert_allclose(cell_phase, 0.0, atol=1.0e-12)


def _verify_particle_restart(
    split_dir: Path, reference: dict[str, object]
) -> None:
    """Compare a completed particle restart run with its uninterrupted reference."""
    restarted = sorted_particles(
        read_particle_binary(latest(split_dir, "pbin/*.prtclbin"))
    )
    reference_sorted = sorted_particles(reference)  # type: ignore[arg-type]
    assert np.array_equal(restarted["integer"], reference_sorted["integer"])
    assert np.array_equal(restarted["real"], reference_sorted["real"])
    assert np.array_equal(restarted["grid"], reference_sorted["grid"])


def verify_particles(root: Path) -> None:
    """Check particle transport/restart artifacts produced by external Athena runs."""
    for pusher in ("classical", "lagrangian_mc", "ito_2"):
        fresh_dir = root / f"particles_{pusher}_8"
        start, finish = _particle_output(fresh_dir)
        _check_particle_transport(start, finish, pusher)
        if pusher != "classical":
            _verify_particle_restart(root / f"particles_restart_{pusher}", finish)

def check_particles(root: Path, backend: str) -> None:
    """Launch and verify particle regressions for pytest compatibility."""
    for pusher in ("classical", "lagrangian_mc", "ito_2"):
        _, start, finish = _particle_run(root, backend, pusher)
        _check_particle_transport(start, finish, pusher)
        if pusher != "classical":
            split_dir = root / f"particles_restart_{pusher}"
            run_athena(
                split_dir,
                "public_main_particles.athinput",
                backend,
                [f"particles/pusher={pusher}", "time/nlim=4"],
            )
            fluid_restart = latest(split_dir, "rst/*.rst")
            particle_restart = latest(split_dir, "prst/*.prtclrst")
            run_athena(
                split_dir,
                "public_main_particles.athinput",
                backend,
                [f"particles/pusher={pusher}", "time/nlim=8"],
                restart=fluid_restart,
                particle_restart=particle_restart,
            )
            _verify_particle_restart(split_dir, finish)

def verify_mhd_rk4(root: Path) -> None:
    """Check linear-wave convergence artifacts produced by external Athena runs."""
    errors: list[float] = []
    for nx1, nx2, nx3 in ((16, 8, 8), (32, 16, 16)):
        run_dir = root / f"mhd_rk4_{nx1}"
        values = np.loadtxt(latest(run_dir, "PublicMhdRk4-errs.dat"), ndmin=2)
        assert np.all(np.isfinite(values))
        errors.append(float(values[-1, 4]))
    assert errors[1] < 0.6 * errors[0]


def check_mhd_rk4(root: Path, backend: str) -> None:
    """Launch and verify linear-wave convergence for pytest compatibility."""
    for nx1, nx2, nx3 in ((16, 8, 8), (32, 16, 16)):
        run_dir = root / f"mhd_rk4_{nx1}"
        run_athena(
            run_dir,
            "public_main_mhd_rk4.athinput",
            backend,
            [f"mesh/nx1={nx1}", f"mesh/nx2={nx2}", f"mesh/nx3={nx3}"],
        )
    verify_mhd_rk4(root)


def run_feature_regressions(root: Path, backend: str) -> None:
    """Run every imported-feature contract for one CPU, MPI-CPU, or GPU backend."""
    for _, scenario in FEATURE_CASES:
        scenario(root, backend)


# Keep scenarios independently runnable: pytest can report all failed contracts in one
# allocation, while run_feature_regressions remains available for the bash diagnostic.
FEATURE_CASES = (
    ("hooks_geometry", check_hooks_and_geometry),
    ("profile_pdf", check_profile_and_pdf_exactness),
    ("dt_wall", check_dt_wall),
    ("turbulence_restart", check_turbulence_restart),
    ("particles", check_particles),
    ("mhd_rk4", check_mhd_rk4),
)
