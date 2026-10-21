#!/usr/bin/env python3
"""Extract archived conduction fields; never runs Athena.

Usage: python export_physical_profiles.py CAMPAIGN
Requires NumPy and SciPy. Campaign paths in the manifest are relative to CAMPAIGN.
Plotting the bundled export requires no access to the campaign.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    root = args.campaign.resolve()
    arrays, sources, studies = {}, {}, {}

    def source(path, expected=None):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None:
            assert digest == expected, str(path)
        sources[str(path.relative_to(root))] = digest
        return path

    def report(relative):
        return json.loads(source(root / relative).read_text())

    def field(path, coordinate_columns, value_columns, prefix, expected=None):
        table = np.loadtxt(source(path, expected))
        coords = [np.unique(table[:, c]) for c in coordinate_columns]
        assert table.shape[0] == np.prod([len(x) for x in coords])
        order = np.lexsort(tuple(table[:, c] for c in reversed(coordinate_columns)))
        shape = tuple(len(x) for x in coords)
        for label, col in value_columns.items():
            arrays[prefix + "_" + label] = table[order, col].reshape(shape)
        for label, coord in zip("xyz", coords):
            arrays[prefix + "_" + label] = coord
        return table[order], shape

    ring = report("figures/full-ring-20260922-g/analysis.json")
    bundled = json.loads((HERE / "results.json").read_text())
    for limiter in ("vanleer", "mc"):
        row = next(r for r in bundled["ring"]
                   if r["configuration"]["n"] == 400
                   and r["configuration"]["limiter"] == limiter)
        path = root / "runs/gpu-ring400-segments-20260922" / limiter / "combined/patch.patch.0.dat"
        field(path, (0, 1), {"temperature": 4, "initial": 5}, "ring_" + limiter,
              row["field_sha256"])
        x, y = np.meshgrid(arrays["ring_" + limiter + "_x"],
                           arrays["ring_" + limiter + "_y"], indexing="ij")
        radius = np.hypot(x, y)
        reference = 10 + ((radius > .5) & (radius < .7)) / 6
        difference = arrays["ring_" + limiter + "_temperature"] - reference
        measured = (np.abs(difference).mean(), np.sqrt((difference**2).mean()),
                    np.abs(difference).max())
        for key, value in zip(("paper_l1", "paper_l2", "paper_linf"), measured):
            assert abs(value - row["metrics"][key]) < 1.e-12
        studies["ring_" + limiter] = dict(n=400, time=200, gamma=2, rho=1,
            alpha_iso=0, alpha_aniso=.01, binary_sha256=row["binary_sha256"],
            metrics=row["metrics"])

    oracle_path = root / "runs/gaussian-cpu-matrix-20260922-a/checker/gaussian_oracle.py"
    oracle = load_module("archived_gaussian_oracle", source(oracle_path))
    for dimension, campaign, case in (
            (2, "gaussian-cpu-refinement-20260922-a", "mc_2d_ratio100_n32_n64"),
            (3, "gaussian-cpu-matrix-20260922-a", "mc_3d_ratio100_n32")):
        base = "runs/" + campaign
        data = report(base + "/results.json")
        row = data["cases"][case]
        conf = row["configuration"]
        prefix = f"gaussian{dimension}"
        _, shape = field(root / base / case / "gaussian.gaussian.dat",
                         tuple(range(dimension)), {"temperature": 5, "initial": 6}, prefix)
        tensor = oracle.tensor(dimension, conf["gamma"], conf["iso"], conf["aniso"],
                               conf["field"])
        exact = oracle.periodic_gaussian(shape, conf["stop"], tensor)
        arrays[prefix + "_reference"] = exact
        metrics = oracle.metrics(arrays[prefix + "_temperature"], exact,
                                 arrays[prefix + "_initial"])
        for key in ("l1", "l2", "linf_over_initial_amplitude"):
            assert abs(metrics[key] - row["metrics"][key]) < 1.e-12
        studies[prefix] = dict(configuration=conf, metrics=row["metrics"],
                              binary_sha256=data["binary_sha256"],
                              diffusion_tensor=tensor.tolist())

    sovinec = report("figures/sovinec100-20260922-b/analysis.json")
    for limiter, folder in (("vanleer", "cpu-paper-matrix-TkwHx9"),
                            ("mc", "cpu-paper-matrix-ns6yx8")):
        case = "sovinec_r100_n16_" + limiter
        row = sovinec["cases"][case]
        base = root / "runs" / folder / "paper"
        field(base / case / "equilibrium.eq.0.dat", (0, 1),
              {"temperature": 4, "reference": 5}, "sovinec_" + limiter,
              row["field_sha256"])
        error = np.abs(arrays["sovinec_" + limiter + "_temperature"] -
                       arrays["sovinec_" + limiter + "_reference"]).mean()
        assert abs(error - row["independent_l1"]) < 1.e-12
        studies["sovinec_" + limiter] = {k: row[k] for k in
            ("metrics", "independent_l1", "fraction_of_parallel", "binary_sha256")}
    field(base / "sovinec_r1_n16_mc/equilibrium.eq.0.dat", (0, 1),
          {"temperature": 4}, "sovinec_isotropic")

    reader = load_module("athena_binary_reader", HERE.parents[2] / "vis/python/bin_convert.py")
    pulse_base = root / "runs/harmonic-cpu-rk-20261008-ZO4dy1/matrix"
    pulse = report(str((pulse_base / "report.json").relative_to(root)))
    case = "rk3_cfl0p025"
    for label, number in (("initial", 0), ("final", 4)):
        path = pulse_base / case / "bin" / f"{case}.cons.{number:05}.bin"
        raw = reader.read_binary_as_athdf(str(source(path)))
        rho = raw["dens"]
        internal = raw["ener"] - sum(raw[m]**2 for m in ("mom1", "mom2", "mom3")) / (2*rho)
        arrays["pulse_" + label] = ((2/3)*internal/rho).ravel()
        arrays["pulse_x"] = raw["x1v"]
    studies["pulse"] = dict(n=8, time=.2, gamma=5/3, rho=1, kappa0=.1, phi=.03,
                            diagnostic_precision="double", field_precision="float32",
                            binary_sha256=pulse["binary_sha256"])

    coupled_base = root / "runs/nonlinear-coupled-long-cpu-20261009-MwPazl/matrix"
    coupled = report(str((coupled_base / "report.json").relative_to(root)))
    for label, case in (("explicit", "long_explicit_cfl0p05"),
                        ("rkl", "long_rkl_cfl0p05")):
        filename = case + ".refinement_cells.rank0.dat"
        row = coupled["cases"][case]
        table, shape = field(coupled_base / case / filename, (5, 6),
                             {"density": 11}, "coupled_" + label,
                             row["artifact_sha256"][filename])
        rho = table[:, 11]
        kinetic = np.sum(table[:, 12:15]**2, axis=1)/(2*rho)
        bcc = (table[:, 16:22:2] + table[:, 17:22:2]) / 2
        temperature = (2/3)*(table[:, 15] - kinetic - .5*np.sum(bcc**2, axis=1))/rho
        arrays["coupled_" + label + "_temperature"] = temperature.reshape(shape)
        studies["coupled_" + label] = dict(n=16, time=.2, gamma=5/3, kappa0=.1,
                                          phi=.03, binary_sha256=coupled["binary_sha256"])
    case = "long_rkl_cfl0p05"
    path = coupled_base / case / "bin" / (case + ".cons.00000.bin")
    raw = reader.read_binary_as_athdf(str(source(path)))
    rho = raw["dens"]
    kinetic = sum(raw[m]**2 for m in ("mom1", "mom2", "mom3"))/(2*rho)
    # The initial field is uniform, so face and cell magnetic energies coincide.
    arrays["coupled_initial"] = ((2/3)*(raw["ener"]-kinetic-.5*(1+.5**2+.25**2))/rho)[0].T

    assert all(np.all(np.isfinite(value)) for value in arrays.values())
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "physical_profiles.npz", **arrays)
    metadata = dict(schema=1, axes="x,y[,z]", temperature="p/rho",
                    archive_sources=sources, studies=studies)
    (args.output / "physical_profiles.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {len(arrays)} arrays from {len(sources)} hashed archive files.")


if __name__ == "__main__":
    main()
