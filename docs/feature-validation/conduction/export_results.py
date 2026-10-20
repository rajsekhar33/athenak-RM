#!/usr/bin/env python3
"""Export a compact public evidence bundle from the archived validation campaign.

This optional provenance tool reads existing results; it never runs Athena.
Figure reproduction requires only plot_results.py and the bundled JSON files.
Usage: python export_results.py /path/to/transport-validation /path/to/consolidation
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = "588d737fac0251b5538c287cf50c3ccdd4cad706"
TIP = "c3a8fccca928eef8b22d233458c4a4aabd9cb171"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("consolidation", type=Path)
    args = parser.parse_args()
    evidence = {}

    def read(label, relative):
        path = args.campaign / relative
        evidence[label] = {"sha256": sha(path)}
        return json.loads(path.read_text())

    ring = read("ring_comparison", "figures/full-ring-20260922-g/analysis.json")
    results = {"schema": 1, "source_baseline": BASE, "source_tip": TIP,
               "scope": "Conduction operators, closures, time integration and refinement"}
    results["ring"] = [{k: case[k] for k in (
        "name", "configuration", "metrics", "reference",
        "error_relative_difference_percent", "binary_sha256", "field_sha256")}
        for case in ring["cases"]]
    rk = read("explicit_temporal", "runs/harmonic-cpu-rk-20261008-ZO4dy1/matrix/report.json")
    results["explicit_temporal"] = {
        "binary_sha256": rk["binary_sha256"], "checks": rk["checks"],
        "gates": rk["gates"], "reference_kind": rk["reference_kind"]}
    coupled = read("coupled_temporal", "runs/nonlinear-coupled-long-cpu-20261009-MwPazl/matrix/report.json")
    results["coupled_temporal"] = {
        "binary_sha256": coupled["binary_sha256"],
        "convergence": coupled["checks"]["empirical_adaptive_time_convergence"],
        "explicit_halfstep": coupled["checks"]["explicit_halfstep"],
        "counts": coupled["counts"]}
    sov = read("coarse_sovinec", "figures/sovinec100-20260922-b/analysis.json")
    results["coarse_sovinec"] = {
        "physics_status": sov["physics_status"],
        "cases": {name: {k: case[k] for k in (
            "status", "failures", "independent_l1", "fraction_of_parallel",
            "stationary", "accurate", "binary_sha256", "field_sha256")}
            for name, case in sov["cases"].items()}}
    retained = read("retained_3d_refinement", "runs/refinement-mpicpu3d-gpu3d-20261008-differences.json")
    results["retained_3d_refinement"] = {
        k: retained[k] for k in ("status", "tolerance", "source_manifest_checks", "cases")}
    results["retained_3d_refinement"].update({
        "initial_temperature_amplitude": 0.1,
        "qualification": "The six failures are static N32 refinement cases, not failed dynamic regridding"})
    assert sum(c["status"] == "FAIL" for c in retained["cases"].values()) == 6
    smr = read("coupled_smr_density", "runs/nonlinear-coupled-smr-density-postfix-cpu-20261009-UUAtuB/matrix/report.json")
    results["coupled_smr_density"] = {
        "status": smr["status"], "counts": smr["counts"], "binary_sha256": smr["binary_sha256"],
        "comparisons": {name: {"status": check["status"], "metrics": check["metrics"]}
                        for name, check in smr["checks"].items()
                        if name.startswith("density_")}}
    results["coupled_amr"] = {}
    for lane in ("gpu", "mpicpu"):
        d = read("coupled_amr_" + lane, "runs/nonlinear-coupled-amr-cpu-" + lane + "-20261009-audit.json")
        results["coupled_amr"][lane] = {k: d[k] for k in (
            "status", "gates", "maxima")}
        results["coupled_amr"][lane]["comparisons"] = {
            name: {"status": case["status"], "final_raw": case["final_raw"],
                   "checkpoint_pairs": len(case["checkpoints"])}
            for name, case in d["case_comparisons"].items()}
    saw_root = args.campaign / "runs/saturated-sawtooth-cpu-20261009-a/matrix"
    saw = read("saturated_sawtooth", "runs/saturated-sawtooth-cpu-20261009-a/matrix/comparison.json")
    results["sawtooth"] = {k: saw[k] for k in (
        "status", "gates", "checks", "spatial_sensitivity", "case_status")}
    profiles = {}
    for name in saw["case_status"]:
        report_path = saw_root / name / "report.json"
        report = json.loads(report_path.read_text())
        assert report["status"] == "PASS"
        evidence[name] = {"sha256": sha(report_path)}
        profiles[name] = {}
        for phase in ("initial", "final"):
            path = saw_root / name / (name + ".sawtooth." + phase + ".dat")
            assert sha(path) == report["artifact_sha256"][path.name]
            a = np.loadtxt(path)
            a = a[np.argsort(a[:, 5])]
            assert a.shape[1] == 22 and np.isfinite(a).all()
            b = .5 * (a[:, 16:22:2] + a[:, 17:22:2])
            temperature = (a[:, 15] - .5 * np.sum(a[:, 12:15]**2, axis=1) / a[:, 11]
                           - .5 * np.sum(b*b, axis=1)) / a[:, 11]
            profiles[name][phase] = {"x": a[:, 5].tolist(), "temperature": temperature.tolist(),
                                     "raw_sha256": sha(path)}
    results["sawtooth"]["profiles"] = profiles

    results["consolidation"] = {}
    for lane in ("cpu", "gpu"):
        path = args.consolidation / "runs" / (lane + "-01") / "results.json"
        evidence["consolidation_" + lane] = {"sha256": sha(path)}
        d = json.loads(path.read_text())
        results["consolidation"][lane] = {"status": d.get("status"),
            "binary_sha256": d["binary_sha256"],
            "cases": {name: {"status": case["status"], "metrics": case["metrics"]}
                      for name, case in d["cases"].items()},
            "scope": "Three native smoke cases; all fields at 149 checkpoints bitwise match the earlier consolidated implementation"}
    results["evidence"] = evidence
    encoded = json.dumps(results, indent=2, allow_nan=False) + "\n"
    assert not re.search(r'"(?:/|[A-Za-z]:\\)', encoded), "absolute path in public data"
    (HERE / "results.json").write_text(encoded)

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(REPO), *argv], text=True)

    paths = git("diff", "--name-only", BASE, TIP).splitlines()
    assert len(paths) == 77
    inventory = []
    for path in paths:
        diff = git("diff", "--unified=0", BASE, TIP, "--", path)
        content = subprocess.check_output(["git", "-C", str(REPO), "show", TIP + ":" + path])
        inventory.append({"path": path, "sha256": hashlib.sha256(content).hexdigest(),
            "hunks": [line for line in diff.splitlines() if line.startswith("@@")]})
    (HERE / "source_manifest.json").write_text(json.dumps({"baseline": BASE, "tip": TIP,
        "changed_path_count": len(paths), "files": inventory}, indent=2) + "\n")
    print("Exported source inventory and conduction-specific numerical datasets.")


if __name__ == "__main__":
    main()
