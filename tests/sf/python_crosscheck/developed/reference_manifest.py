#!/usr/bin/env python3
"""Bind sampled reference files to their input state, producer and Python sources."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MANIFEST = "reference_manifest.json"
SEEDS = tuple(range(16))
MODES = ("scalar", "longtrans")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_file(run, directory, suffix):
    paths = sorted((Path(run) / directory).glob("*" + suffix))
    if len(paths) != 1:
        raise ValueError(f"expected one {directory}/*{suffix}, found {len(paths)}")
    return paths[0].resolve()


def fingerprint(run, pyrepo, ns1=1024, nseed=16):
    if nseed != len(SEEDS) or ns1 != 1024:
        raise ValueError("developed reference requires 16 seeds and NS1=1024")
    run, pyrepo = Path(run).resolve(), Path(pyrepo).resolve()
    sources = subprocess.check_output(
        ["git", "-C", str(pyrepo), "ls-files", "-z", "--", "*.py"])
    names = [name.decode() for name in sources.split(b"\0") if name]
    if not names:
        raise ValueError("Python reference checkout contains no tracked Python sources")
    inputs = [state_file(run, "bin", ".bin"), state_file(run, "rst", ".rst")]
    for sub in ("sf_scalar", "sf_longtrans"):
        inputs.append(state_file(run, sub, ".sf"))
    producer = [HERE.parent / "python_sf.py", ROOT / "tests/rst_compare.py",
                HERE / "run_python.sh", Path(__file__).resolve()]
    return {
        "version": 1,
        "configuration": {"seeds": list(SEEDS), "modes": list(MODES),
                          "ns1": ns1, "nbins": 100, "max_order": 3,
                          "scalar_variables": ["velx", "vely", "velz", "dens"],
                          "scalar_weights": ["ones", "dens"],
                          "longtrans_weights": ["ones"]},
        "run": str(run), "pyrepo": str(pyrepo),
        "inputs": {str(path): sha256(path) for path in inputs},
        "producer": {str(path): sha256(path) for path in producer},
        "reference_python": {name: sha256(pyrepo / name) for name in names},
        "reference_commit": subprocess.check_output(
            ["git", "-C", str(pyrepo), "rev-parse", "HEAD"], text=True).strip(),
        "python_version": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy")},
    }


def expected_files():
    return {f"{mode}_sampled_s{seed}.npz" for mode in MODES for seed in SEEDS}


def output_hashes(out):
    out = Path(out)
    names = {p.name for p in out.glob("*.npz")}
    want = expected_files()
    if names != want:
        raise ValueError(f"reference inventory: missing={sorted(want - names)}, "
                         f"extra={sorted(names - want)}")
    return {name: sha256(out / name) for name in sorted(want)}


def write_manifest(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def verify(run, out):
    path = Path(out) / MANIFEST
    data = json.loads(path.read_text())
    if data.get("status") != "complete":
        raise ValueError("reference manifest is incomplete; use a fresh output directory")
    context = data["context"]
    current = fingerprint(run, context["pyrepo"])
    if context != current:
        raise ValueError("reference input state, producer or Python sources changed")
    if data["outputs"] != output_hashes(out):
        raise ValueError("sampled reference content differs from its recorded hashes")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "finish", "verify"))
    parser.add_argument("run")
    parser.add_argument("out")
    parser.add_argument("--pyrepo")
    parser.add_argument("--ns1", type=int, default=1024)
    parser.add_argument("--nseed", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.out)
    path = out / MANIFEST
    if args.action == "verify":
        verify(args.run, out)
    elif args.action == "prepare":
        if not args.pyrepo:
            raise ValueError("prepare requires --pyrepo")
        context = fingerprint(args.run, args.pyrepo, args.ns1, args.nseed)
        if path.exists():
            existing = verify(args.run, out)
            if existing["context"] != context:
                raise ValueError(
                    "existing cache uses a different requested configuration")
        else:
            if any(out.glob("*.npz")):
                raise ValueError("reference files lack provenance; use a fresh directory")
            out.mkdir(parents=True, exist_ok=True)
            write_manifest(path, {"status": "pending", "context": context})
    else:
        data = json.loads(path.read_text())
        current = fingerprint(args.run, data["context"]["pyrepo"],
                              args.ns1, args.nseed)
        if data["context"] != current:
            raise ValueError("input state or sources changed while producing references")
        if data.get("status") == "complete":
            verify(args.run, out)
        else:
            data.update(status="complete", outputs=output_hashes(out))
            write_manifest(path, data)
    print(f"REFERENCE MANIFEST {args.action}: PASS")


if __name__ == "__main__":
    main()
