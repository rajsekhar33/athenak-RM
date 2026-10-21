#!/usr/bin/env python3
"""Reduce the archived square-pulse and turbulent tracer snapshots.

This optional extraction step requires the original September 2026 revalidation
directory, NumPy, h5py and this repository's vis/python/bin_convert.py. Plotting
the bundled reductions only requires NumPy and Matplotlib. No simulation runs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "vis" / "python"))
from bin_convert import read_binary

PUSHERS = ("classical", "ito_2", "lagrangian_mc")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(grid, name):
    result = np.full((grid["Nx3"], grid["Nx2"], grid["Nx1"]), np.nan)
    for block, logical in zip(grid["mb_data"][name], grid["mb_logical"]):
        nz, ny, nx = block.shape
        i, j, k = np.asarray(logical[:3], int) * (nx, ny, nz)
        result[k:k+nz, j:j+ny, i:i+nx] = block
    assert np.isfinite(result).all()
    return result


def positions(path):
    with path.open("rb") as stream:
        magic, number, nr, ni, ng = np.fromfile(stream, np.int64, 5)
        assert magic == 43 and nr >= 3
        time, dt, cycle = np.fromfile(stream, np.float64, 3)
        stream.seek(int(ng) * 16, 1)
        xyz = np.fromfile(stream, np.float64, int(3 * number)).reshape(3, -1)
    assert np.isfinite(xyz).all()
    return xyz, {"count": int(number), "time": float(time), "cycle": int(cycle)}


def deposit(xyz, grid):
    """Periodic cell-centered CIC, and containing-cell (NGP) integer counts."""
    sizes = np.array([grid[f"Nx{a}"] for a in (1, 2, 3)])
    lower = np.array([grid[f"x{a}min"] for a in (1, 2, 3)])
    length = np.array([grid[f"x{a}max"] for a in (1, 2, 3)]) - lower
    shape = tuple(sizes[::-1])
    cic = np.zeros(shape)
    ngp = np.zeros(shape, dtype=np.int64)
    for start in range(0, xyz.shape[1], 250000):
        x = (xyz[:, start:start+250000] - lower[:, None]) % length[:, None]
        cell = x * (sizes / length)[:, None]
        nearest = np.floor(cell).astype(int) % sizes[:, None]
        index = (nearest[2] * sizes[1] + nearest[1]) * sizes[0] + nearest[0]
        ngp.ravel()[:] += np.bincount(index, minlength=ngp.size)
        coordinate = cell - 0.5
        left = np.floor(coordinate).astype(int)
        frac = coordinate - left
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    shift = np.array([dx, dy, dz])[:, None]
                    index3 = (left + shift) % sizes[:, None]
                    weight = np.prod(np.where(shift, frac, 1-frac), axis=0)
                    index = (index3[2] * sizes[1] + index3[1]) * sizes[0] + index3[0]
                    cic.ravel()[:] += np.bincount(index, weights=weight, minlength=cic.size)
    assert ngp.sum() == xyz.shape[1]
    np.testing.assert_allclose(cic.sum(), xyz.shape[1], rtol=1e-14)
    return cic, ngp


def spectrum(array, shell, number):
    delta = array / array.mean() - 1
    power = np.abs(np.fft.fftn(delta) / delta.size) ** 2
    return np.bincount(shell.ravel(), weights=power.ravel()) / number


def extract(root, output):
    arrays = {}
    meta = {
        "historical_campaign": "revalidation_rk4_20260901",
        "source_base": "13fa232e6d7177e9e11b6c89529f56b6541ba8df",
        "source_state": "Base plus uncommitted CIC-moment, RK-flux/donor and timestep repairs; execution logs pin each executable. These snapshots precede the later AMR/boundary implementation.",
        "analysis": "Periodic cell-centered CIC for density maps, joint PDFs and spectra; containing-cell counts for sampling the gas-density PDF. All cell/column densities are normalized by their own mean.",
        "families": {},
    }
    for family, array_indices in (("square_hi", (0, 1, 2)), ("turb64_iso", (3, 4, 5))):
        record = {"cases": {}}
        gas_reference = None
        for pusher, array_index in zip(PUSHERS, array_indices):
            suffix = "ito" if pusher == "ito_2" else pusher
            run = (root / "runs" / f"{family}_{suffix}_revalidation").resolve()
            fluid = sorted((run / "bin").glob("*.bin"))[-1]
            particle = sorted((run / "pbin").glob("*.prtclbin"))[-1]
            initial = sorted((run / "bin").glob("*.bin"))[0]
            log = root / "runs" / f"slurm.ito-paper-cpu.13302345.{array_index}.out"
            log_text = log.read_text()
            assert f"PASS: {family}_{suffix} completed" in log_text
            executable_sha = next(s.split("=", 1)[1] for s in log_text.splitlines()
                                  if s.startswith("sha256="))
            grid = read_binary(str(fluid))
            gas = field(grid, "dens")
            xyz, snapshot = positions(particle)
            np.testing.assert_allclose(grid["time"], snapshot["time"], atol=1e-13)
            cic, ngp = deposit(xyz, grid)
            if gas_reference is None:
                gas_reference = gas
                record.update(time=grid["time"], shape=list(gas.shape),
                              mean_density=float(gas.mean()),
                              volume_rms_mach=float(np.sqrt(sum(np.mean(field(grid, v)**2)
                                                               for v in ("velx", "vely", "velz")))))
            else:
                np.testing.assert_array_equal(gas, gas_reference)
            case = {**snapshot, "run": run.name, "binary_sha256": executable_sha,
                    "files": {str(path.relative_to(root)): sha256(path)
                              for path in (fluid, particle, initial, run / "athinput", log)}}
            input_name = f"{family}_{pusher}.athinput"
            (output / input_name).write_text((run / "athinput").read_text())
            case["input"] = input_name
            prefix = f"{family}_{pusher}"
            if family == "square_hi":
                profile = gas.mean(axis=(0, 1))
                arrays["square_x"] = grid["x1min"] + (np.arange(grid["Nx1"])+0.5) / grid["Nx1"]
                arrays["square_gas"] = profile
                arrays["square_initial"] = field(read_binary(str(initial)), "dens").mean(axis=(0, 1))
                for method, count in (("cic", cic), ("ngp", ngp)):
                    tracer = count.mean(axis=(0, 1)) / count.mean() * gas.mean()
                    arrays[f"{prefix}_{method}"] = tracer
                    residual = tracer / profile - 1
                    case[f"{method}_relative_rms"] = float(np.sqrt(np.mean(residual**2)))
                    case[f"{method}_relative_max"] = float(np.max(np.abs(residual)))
            else:
                norm_gas = gas / gas.mean()
                norm_tracer = cic / cic.mean()
                arrays["turb_gas_column"] = norm_gas.mean(axis=0)
                arrays[f"{prefix}_column"] = norm_tracer.mean(axis=0)
                column_ratio = np.log10(norm_tracer.mean(axis=0) / norm_gas.mean(axis=0))
                case["column_log10_ratio_mean"] = float(column_ratio.mean())
                case["column_log10_ratio_std"] = float(column_ratio.std())
                ngp_column = ngp.mean(axis=0) / ngp.mean()
                case["ngp_column_log10_ratio_std"] = float(np.log10(ngp_column / norm_gas.mean(axis=0)).std())
                positive = norm_tracer > 0
                case["empty_cic_cell_fraction"] = float(1-positive.mean())
                ratio = np.log10(norm_tracer[positive] / norm_gas[positive])
                ratio_edges = np.linspace(-16, 4, 401)
                assert ratio.min() >= ratio_edges[0] and ratio.max() <= ratio_edges[-1]
                arrays["turb_ratio_edges"] = ratio_edges
                arrays[f"{prefix}_ratio_pdf"] = np.histogram(ratio, ratio_edges, density=True)[0]
                case["cell_log10_ratio_mean_positive"] = float(ratio.mean())
                case["cell_log10_ratio_std_positive"] = float(ratio.std())
                gas_edges = np.linspace(-4, 3, 113)
                tracer_edges = np.linspace(-16, 4, 321)
                assert np.log10(norm_gas).min() >= gas_edges[0]
                assert np.log10(norm_gas).max() <= gas_edges[-1]
                assert np.log10(norm_tracer[positive]).min() >= tracer_edges[0]
                assert np.log10(norm_tracer[positive]).max() <= tracer_edges[-1]
                arrays["turb_gas_log_edges"] = gas_edges
                arrays["turb_tracer_log_edges"] = tracer_edges
                arrays[f"{prefix}_joint"] = np.histogram2d(np.log10(norm_gas[positive]),
                    np.log10(norm_tracer[positive]), bins=(gas_edges, tracer_edges))[0]
                width = np.diff(gas_edges)
                arrays["turb_volume_pdf"] = np.histogram(np.log10(norm_gas), gas_edges)[0] / gas.size / width
                arrays["turb_mass_pdf"] = np.histogram(np.log10(norm_gas), gas_edges,
                    weights=gas)[0] / gas.sum() / width
                arrays[f"{prefix}_sampled_gas_pdf"] = np.histogram(np.log10(norm_gas), gas_edges,
                    weights=ngp)[0] / ngp.sum() / width
                case["sampled_gas_pdf_total_variation"] = float(0.5*np.sum(width*np.abs(
                    arrays[f"{prefix}_sampled_gas_pdf"]-arrays["turb_mass_pdf"])))
                n = grid["Nx1"]
                freq = np.fft.fftfreq(n) * n
                kz, ky, kx = np.meshgrid(freq, freq, freq, indexing="ij")
                shell = np.floor(np.sqrt(kx*kx+ky*ky+kz*kz)+0.5).astype(int)
                number = np.bincount(shell.ravel())
                arrays["turb_shell_k"] = np.arange(len(number))
                arrays["turb_shell_modes"] = number
                arrays["turb_gas_power"] = spectrum(gas, shell, number)
                arrays[f"{prefix}_power"] = spectrum(cic, shell, number)
            record["cases"][pusher] = case
            print(f"Reduced {family} {pusher}: {snapshot['count']} tracers", flush=True)
        meta["families"][family] = record
    convergence = {"resolution": 128, "seed": 42, "initial_strain_rate": 1,
                   "amplitude": 0.2, "time": 0.3,
                   "cmake_problem": "tests/converging_flow",
                   "binary_sha256": "40442070ebe261aaa373ce24c23129bf75d5f8de4852760c3fd388ea555e32fe",
                   "execution_log": "runs/slurm.ito-cflow-seeds.13302360.out",
                   "execution_log_sha256": sha256(root / "runs/slurm.ito-cflow-seeds.13302360.out"),
                   "reference": "Archived gas follows pressureless self-advection: x=x0*(1-a*t), v=-a*x/(1-a*t), rho=rho0(x/(1-a*t))/(1-a*t). The archived input omits problem/user_work_in_loop, whose default is false; the velocity-reset callback therefore remains disabled.",
                   "cases": {}}
    for pusher in PUSHERS:
        run = root / "runs" / "converging-resolution-seed-13302360" / f"cflow_n128_{pusher}_s42"
        fluid = sorted((run / "bin").glob("*.bin"))[-1]
        particle = sorted((run / "pbin").glob("*.prtclbin"))[-1]
        grid = read_binary(str(fluid))
        gas = field(grid, "dens").mean(axis=(0, 1))
        xyz, snapshot = positions(particle)
        x = -0.5 + (np.arange(128)+0.5)/128
        valid = np.abs(x) <= 0.85*0.5*np.exp(-0.3)
        profile = np.histogram(xyz[0], np.linspace(-0.5, 0.5, 129))[0]
        arrays["converging_x"] = x
        arrays["converging_valid"] = valid
        gas_normalized = gas / gas[valid].mean()
        if "converging_gas" in arrays:
            np.testing.assert_array_equal(arrays["converging_gas"], gas_normalized)
        arrays["converging_gas"] = gas_normalized
        arrays[f"converging_{pusher}"] = profile / profile[valid].mean()
        analytic = (1+0.2*np.cos(2*np.pi*x/(1-0.3)))/(1-0.3)
        arrays["converging_analytic"] = analytic / analytic[valid].mean()
        velocity = field(grid, "velx").mean(axis=(0, 1))
        arrays["converging_velocity"] = velocity
        convergence["cases"][pusher] = {**snapshot, "run": run.name,
            "velocity_slope_valid": float(np.polyfit(x[valid], velocity[valid], 1)[0]),
            "gas_analytic_mean_abs_relative_valid": float(np.abs(gas/analytic-1)[valid].mean()),
            "files": {str(path.relative_to(root)): sha256(path)
                      for path in (fluid, particle, run / "athinput")}}
        if pusher == "ito_2":
            (output / "converging_n128_ito_2_s42.athinput").write_text((run / "athinput").read_text())
    meta["converging_profile"] = convergence
    np.savez_compressed(output / "paper_comparisons.npz", **arrays)
    meta["data_sha256"] = sha256(output / "paper_comparisons.npz")
    (output / "paper_comparisons.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original revalidation_rk4_20260901 directory")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract(args.archive, args.output_dir)
