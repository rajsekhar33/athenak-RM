"""Exact planar interfaces, output lifecycle, and diagnostic state preservation.

All fields are initialized by the ordinary shock-tube problem generator.  The
reference reduces a planar interface to independent one-dimensional Gaussian
convolution and counts crossings; it does not reuse the native triangulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import time

import numpy as np


REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Plane:
    axis: int = 0
    mhd: bool = False
    density: bool = True
    shape: tuple[int, int, int] = (24, 16, 16)
    blocks: tuple[int, int, int] = (8, 8, 8)
    periodic: tuple[bool, bool, bool] = (False, True, True)
    lengths: tuple[float, float, float] = (1.5, 1.0, 1.0)
    levels: tuple[float, ...] = (0.0, 1.5, 2.5, 3.5, 5.0)
    strides: tuple[int, ...] = (1, 3, 8)
    methods: tuple[str, ...] = ("sampled", "indicator", "field")
    transform: str = "linear"
    sigma_factor: float = 1.0
    truncate: float = 4.0
    shock_fraction: float = 0.43
    scalar_scale: float = 1.0
    scalar_left: float = 1.0
    scalar_right: float = 4.0


def input_text(p: Plane, *, area=True, cycles=0, restart_output=False,
               output_id="surface", area_extra="", extra="") -> str:
    """Use a stationary contact for density; temperature uses equal densities."""
    fluid = "mhd" if p.mhd else "hydro"
    variable = f"{fluid}_w_d" if p.density else "temperature"
    lines = ["<job>", "basename = Area", "<mesh>", "nghost = 2"]
    for a, (n, length, periodic) in enumerate(zip(p.shape, p.lengths, p.periodic), 1):
        bc = "periodic" if periodic else "outflow"
        lines += [f"nx{a} = {n}", f"x{a}min = 0", f"x{a}max = {length}",
                  f"ix{a}_bc = {bc}", f"ox{a}_bc = {bc}"]
    lines += ["<meshblock>"] + [f"nx{a} = {n}" for a, n in enumerate(p.blocks, 1)]
    lines += ["<time>", "evolution = dynamic", "integrator = rk2", "cfl_number = 0.2",
              f"nlim = {cycles}", "tlim = 10", "ndiag = 1", f"<{fluid}>",
              "eos = ideal", "gamma = 1.6666666666666667", "reconstruct = plm",
              "rsolver = llf", "<problem>", "pgen_name = shock_tube",
              f"shock_dir = {p.axis + 1}",
              f"xshock = {p.lengths[p.axis] * p.shock_fraction:.17g}",
              f"dl = {p.scalar_left*p.scalar_scale if p.density else 1:.17g}",
              f"dr = {p.scalar_right*p.scalar_scale if p.density else 1:.17g}",
              f"pl = {1 if p.density else p.scalar_left*p.scalar_scale:.17g}",
              f"pr = {1 if p.density else p.scalar_right*p.scalar_scale:.17g}",
              "ul = 0", "vl = 0", "wl = 0", "ur = 0", "vr = 0", "wr = 0"]
    if p.mhd:
        lines += [f"b{axis}{side} = {value}" for side in ("l", "r")
                  for axis, value in zip(("x", "y", "z"), (0.1, 0.2, 0.3))]
    if area:
        lines += ["<output1>", "file_type = area", f"id = {output_id}",
                  f"variable = {variable}", "dcycle = 1",
                  "levels = " + ",".join(map(str, p.levels)),
                  "strides = " + ",".join(map(str, p.strides)),
                  "methods = " + ",".join(p.methods),
                  f"sample_transform = {p.transform}",
                  f"sigma_factor = {p.sigma_factor}", f"truncate = {p.truncate}",
                  area_extra]
    if restart_output:
        lines += ["<output2>", "file_type = rst", "dt = -1", "dcycle = 2"]
    return "\n".join(lines + [extra, ""]).strip() + "\n"


def gaussian_1d(field, sigma, truncate, periodic):
    """Explicit finite-kernel convolution, including repeated periodic images."""
    radius = int(np.floor(truncate * sigma + 0.5))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma)**2)
    weights /= weights.sum()
    indices = np.arange(len(field))[:, None] + offsets[None, :]
    indices = indices % len(field) if periodic else indices.clip(0, len(field) - 1)
    return np.sum(np.asarray(field)[indices] * weights, axis=1)


def plane_reference(p: Plane):
    """Area = crossing count times the represented transverse physical area."""
    n = p.shape[p.axis]
    scalar = p.scalar_scale * np.where(
        (np.arange(n) + 0.5) / n < p.shock_fraction, p.scalar_left, p.scalar_right)
    transverse = np.prod([length * (1.0 if periodic else (size - 1) / size)
                          for a, (length, size, periodic) in enumerate(
                              zip(p.lengths, p.shape, p.periodic)) if a != p.axis])
    expected = {}
    for method in p.methods:
        for level in p.levels:
            for stride in p.strides:
                sigma = p.sigma_factor * stride if method != "sampled" else 0.0
                threshold = level
                if method == "sampled":
                    values = np.log10(scalar) if p.transform == "log10" else scalar
                    global_scale = float(np.max(np.abs(values))) or 1.0
                    threshold = np.log10(level) if p.transform == "log10" else level
                    nodes = list(range(0, n, stride))
                    if p.periodic[p.axis]:
                        nodes.append(0)
                    elif nodes[-1] != n - 1:
                        nodes.append(n - 1)
                    values = values[nodes]
                else:
                    field = ((scalar <= level).astype(float)
                             if method == "indicator" else scalar)
                    values = gaussian_1d(field, sigma, p.truncate, p.periodic[p.axis])
                    global_scale = float(np.max(np.abs(values))) or 1.0
                    if p.periodic[p.axis]:
                        values = np.append(values, values[0])
                    threshold = 0.5 if method == "indicator" else level
                # Zero thresholds use the global transformed-field scale; other
                # thresholds use each vertex/level scale to preserve weak contours.
                scale = (np.full(len(values), global_scale) if threshold == 0.0
                         else np.maximum(np.abs(values), abs(threshold)))
                scale[scale == 0] = 1.0
                residual = values / scale - threshold / scale
                inside = residual < -1.0e-12
                crossings = np.count_nonzero(inside[:-1] != inside[1:])
                expected[(method, float(level), stride)] = (
                    sigma, float(crossings * transverse))
    return expected


def read_area(path):
    metadata, rows = {}, {}
    columns = None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            content = line[1:].strip()
            if content.startswith("columns:"):
                columns = content.split(":", 1)[1].split()
            for key, value in re.findall(
                    r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^\s]+)", content):
                metadata[key] = value.rstrip(",")
            continue
        parts = line.split()
        assert len(parts) == 5, (path, parts)
        method, level, stride, sigma, area = parts
        key = method, float(level), int(stride)
        assert key not in rows, (path, "duplicate row", key)
        values = float(sigma), float(area)
        assert np.isfinite([key[1], *values]).all(), (path, key, values)
        assert values[0] >= 0 and values[1] >= 0, (path, key, values)
        rows[key] = values
    assert columns == ["method", "level", "stride", "sigma_cells", "area"], (
        path, columns)
    assert rows, path
    assert "cycle" in metadata and "time" in metadata, (path, metadata)
    return metadata, rows


def check_plane(directory, p):
    paths = sorted(directory.glob("area_surface/Area.*.area"))
    assert len(paths) == 2, (directory, paths)
    assert [path.name for path in paths] == ["Area.00000.area", "Area.00001.area"]
    expected = plane_reference(p)
    configuration = (directory / "area_surface" / "configuration.txt").read_text()
    assert configuration.startswith("# AthenaK area format=1\n")
    maximum = 0.0
    for path in paths:
        assert path.read_text().startswith(configuration), path
        metadata, actual = read_area(path)
        assert int(metadata["cycle"]) == 0 and float(metadata["time"]) == 0.0
        assert tuple(map(int, metadata["dimensions"].split(","))) == p.shape
        assert tuple(map(int, metadata["periodic"].split(","))) == tuple(
            map(int, p.periodic))
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for key, reference in expected.items():
            np.testing.assert_allclose(actual[key][0], reference[0],
                                       rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(actual[key][1], reference[1],
                                       rtol=2e-10, atol=2e-10)
            maximum = max(maximum, abs(actual[key][1] - reference[1]))
    return {"events": len(paths), "rows_per_event": len(expected),
            "max_absolute_area_error": maximum}


def materialize_overrides(text, overrides):
    """Write requested values into athinput, including keys without defaults yet."""
    lines = text.splitlines()
    for override in overrides:
        target, separator, value = override.partition("=")
        block, slash, parameter = target.partition("/")
        assert separator and slash and block and parameter, override
        section = None
        start, stop, match = None, len(lines), None
        for index, line in enumerate(lines):
            content = line.split("#", 1)[0].strip()
            if content.startswith("<") and content.endswith(">"):
                if section == block:
                    stop = index
                    break
                section = content[1:-1].strip()
                if section == block:
                    start = index
            elif section == block and content.partition("=")[0].strip() == parameter:
                assert match is None, (block, parameter, "duplicate fixture parameter")
                match = index
        replacement = f"{parameter} = {value}"
        if match is not None:
            lines[match] = replacement
        elif start is not None:
            lines.insert(stop, replacement)
        else:
            lines.extend([f"<{block}>", replacement])
    return "\n".join(lines) + "\n"


class Runner:
    def __init__(self, root, binary, backend="cpu", ranks=1, launcher=(),
                 execute=False, timeout=180, device_args=()):
        self.root = Path(root).resolve()
        self.binary = Path(binary).resolve()
        self.backend, self.ranks = backend, ranks
        self.launcher, self.device_args = list(launcher), list(device_args)
        self.execute, self.timeout = execute, timeout
        self.records = []
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, name, text, *, overrides=(), restart=None, reject=None,
            directory=None):
        directory = Path(directory) if directory else self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / f"{name}.athinput"
        assert not source.exists(), f"Refusing to reuse test input {source}"
        text = materialize_overrides(text, overrides)
        source.write_text(text)
        command = [*self.launcher, str(self.binary), *self.device_args]
        if restart:
            command += ["-r", str(restart)]
        command += ["-i", str(source)]
        record = {"case": name, "directory": str(directory), "command": command,
                  "status": "PREPARED", "expected_rejection": reject,
                  "materialized_overrides": list(overrides),
                  "input_sha256": hashlib.sha256(text.encode()).hexdigest()}
        self.records.append(record)
        self.save()
        if not self.execute:
            return directory
        start = time.monotonic()
        result = subprocess.run(command, cwd=directory, text=True, capture_output=True,
                                timeout=self.timeout, check=False)
        (directory / f"{name}.stdout.log").write_text(result.stdout)
        (directory / f"{name}.stderr.log").write_text(result.stderr)
        record.update(returncode=result.returncode,
                      elapsed_seconds=time.monotonic() - start,
                      status="EXECUTED")
        self.save()
        combined = result.stdout + result.stderr
        if reject:
            assert result.returncode != 0, (name, "invalid input was accepted")
            missing = re.search(
                r"(?:Parameter|Block name).*not found|Parameter.*does not exist",
                combined, re.IGNORECASE)
            assert missing is None, (name, "fixture/parser failure", combined[-4000:])
            assert re.search(reject, combined, re.IGNORECASE), (name, combined[-4000:])
        else:
            assert result.returncode == 0, (name, combined[-4000:])
            ranks = re.findall(r"Number of parallel ranks\s*=\s*(\d+)", combined)
            if ranks:
                assert ranks == [str(self.ranks)], (name, ranks, self.ranks)
            elif self.ranks > 1:
                raise AssertionError(f"{name}: MPI rank count was not reported")
        return directory

    def passed(self, evidence=None):
        if self.execute:
            self.records[-1].update(status="PASS", evidence=evidence or {})
            self.save()

    def save(self):
        data = {"backend": self.backend, "ranks": self.ranks,
                "binary": str(self.binary), "execute": self.execute,
                "records": self.records}
        (self.root / "results.json").write_text(json.dumps(data, indent=2) + "\n")


def planes(runner, *, pilot=False):
    cases = [("plane_hydro_x", Plane())]
    if not pilot:
        for axis in range(3):
            periodic = tuple(a != axis for a in range(3))
            for mhd, density in ((False, False), (True, True), (True, False)):
                name = (f"plane_{'mhd' if mhd else 'hydro'}"
                        f"_{'rho' if density else 'T'}_{axis}")
                cases.append((name, Plane(axis=axis, mhd=mhd, density=density,
                                          periodic=periodic)))
        cases += [
            ("odd_nondividing", Plane(shape=(15, 15, 15), blocks=(5, 5, 5),
                                      strides=(1, 4, 7))),
            ("periodic_wide_kernel", Plane(periodic=(True, True, True),
                                           shock_fraction=0.24)),
            ("all_outflow", Plane(periodic=(False, False, False))),
            ("log_scalar", Plane(levels=(1.5, 2.5, 3.5), transform="log10")),
            ("level_ties", Plane(levels=(1.0, 4.0))),
        ]
        for scale in (1.0e-20, 1.0e20):
            for density in (False, True):
                name = f"scale_{scale:g}_{'rho' if density else 'T'}"
                cases.append((name, Plane(
                    mhd=density, density=density,
                    scalar_scale=scale, levels=tuple(scale*x for x in (1.5, 2.5, 3.5)))))
        for scale in (1.0e-6, 1.0, 1.0e6):
            cases.append((f"high_dynamic_range_{scale:g}", Plane(
                scalar_scale=scale, scalar_left=1.0-1.0e-8, scalar_right=1.0e8,
                levels=(scale,), strides=(1, 5, 8), methods=("sampled",),
                shape=(16, 16, 16), blocks=(4, 16, 16), shock_fraction=0.5)))
    for name, plane in cases:
        directory = runner.run(name, input_text(plane))
        reference = {"|".join(map(str, key)): list(value)
                     for key, value in plane_reference(plane).items()}
        (directory / "expected.json").write_text(json.dumps(reference, indent=2) + "\n")
        if runner.execute:
            runner.passed(check_plane(directory, plane))
    if not pilot:
        plane = Plane()
        directory = runner.run(
            "group_component", input_text(plane),
            overrides=("output1/variable=hydro_w", "output1/component=0"))
        if runner.execute:
            runner.passed(check_plane(directory, plane))
        diagonal(runner)


def diagonal(runner):
    """A periodic entropy wave has two oblique planes of total area 2 sqrt(3)."""
    p = Plane(shape=(16, 16, 16), blocks=(8, 8, 8), lengths=(1.0, 1.0, 1.0),
              periodic=(True, True, True), levels=(1.0,), strides=(1, 2))
    text = input_text(p)
    before, remainder = text.split("<problem>\n")
    _, outputs = remainder.split("<output1>\n")
    problem = ("<problem>\npgen_name = linear_wave\nwave_flag = 1\namp = 0.2\n"
               "dens = 1.0\npgas = 0.6\nvx0 = 0.1\n")
    directory = runner.run("diagonal_entropy_planes",
                           before + problem + "<output1>\n" + outputs)
    expected = 2.0 * np.sqrt(3.0)
    (directory / "expected.json").write_text(json.dumps({
        "density": "1 + 0.2 sin[2 pi (x+y+z)]", "level": 1.0,
        "total_area": expected, "normal": [1, 1, 1]}, indent=2) + "\n")
    if runner.execute:
        paths = sorted(directory.glob("area_surface/*.area"))
        assert len(paths) == 2
        maximum = 0.0
        for path in paths:
            metadata, rows = read_area(path)
            assert int(metadata["cycle"]) == 0
            assert len(rows) == len(p.methods)*len(p.strides)
            for key, (_, value) in rows.items():
                np.testing.assert_allclose(value, expected, rtol=2e-10, atol=2e-10,
                                           err_msg=str(key))
                maximum = max(maximum, abs(value - expected))
        runner.passed({"analytic_area": expected, "max_absolute_area_error": maximum})


def read_restart(path, mhd=False):
    """Read active fields from these uniform Newtonian double-precision fixtures.

    Reject unsupported checkpoint layouts instead of guessing payload offsets.
    The binary header and lifecycle extension follow src/outputs/restart.cpp.
    """
    raw = Path(path).read_bytes()
    marker = b"<par_end>\n"
    offset = raw.index(marker) + len(marker)
    header = raw[:offset].decode("ascii")
    assert "checkpoint_lifecycle_version" in header
    assert not any(f"<{name}>" in header for name in ("turb_driving", "z4c", "radiation"))

    def consume(fmt):
        nonlocal offset
        values = struct.unpack_from("=" + fmt, raw, offset)
        offset += struct.calcsize("=" + fmt)
        return values

    nblocks, _ = consume("2i")
    consume("9d")
    consume("19i")
    block = consume("19i")
    moment, dt = consume("2d")
    cycle, = consume("i")
    locations = np.array(consume(f"{4*nblocks}i")).reshape(nblocks, 4)
    consume(f"{nblocks}f")
    magic, version = consume("2Q")
    assert magic == 0x414B43504C494645 and version == 1, (magic, version)
    consume("2d")
    _, ages = consume("2i")
    assert ages == 0, "This reader only supports uniform fixtures"
    payload_bytes, = consume("Q")
    ng, nx, ny, nz = block[:4]
    shape = nz + 2*ng, ny + 2*ng, nx + 2*ng
    shapes = [(5, *shape)]
    if mhd:
        shapes += [(shape[0], shape[1], shape[2]+1),
                   (shape[0], shape[1]+1, shape[2]),
                   (shape[0]+1, shape[1], shape[2])]
    assert payload_bytes == sum(np.prod(s) for s in shapes) * 8
    assert len(raw) == offset + nblocks * payload_bytes
    fields = []
    for b in range(nblocks):
        field_offset = offset + b * payload_bytes
        arrays = []
        for index, dims in enumerate(shapes):
            count = int(np.prod(dims))
            field = np.frombuffer(raw, dtype="=f8", count=count,
                                  offset=field_offset).reshape(dims)
            field_offset += count*8
            if index == 0:
                active = field[:, ng:ng+nz, ng:ng+ny, ng:ng+nx]
            else:
                active = field[ng:ng+nz+(index == 3),
                               ng:ng+ny+(index == 2), ng:ng+nx+(index == 1)]
            assert np.isfinite(active).all()
            arrays.append(active.copy())
        fields.append(arrays)
    return {"time": moment, "dt": dt, "cycle": cycle,
            "locations": locations, "fields": fields}


def final_restart(directory):
    paths = sorted(directory.glob("rst/*.rst"))
    assert paths, (directory, "no restart files")
    return paths[-1]


def equal_state(left, right):
    assert left["cycle"] == right["cycle"]
    assert left["time"] == right["time"]
    np.testing.assert_array_equal(left["locations"], right["locations"])
    maximum = 0.0
    for block_l, block_r in zip(left["fields"], right["fields"]):
        for field_l, field_r in zip(block_l, block_r):
            maximum = max(maximum, float(np.max(np.abs(field_l - field_r))))
            np.testing.assert_array_equal(field_l, field_r)
    return maximum


def lifecycle(runner):
    p = Plane(mhd=True, levels=(2.5,), strides=(1, 3))
    disabled = runner.run("no_area", input_text(
        p, area=False, cycles=4, restart_output=True))
    enabled = runner.run("with_area", input_text(p, cycles=4, restart_output=True))
    if runner.execute:
        error = equal_state(read_restart(final_restart(disabled), True),
                            read_restart(final_restart(enabled), True))
        runner.records[-2].update(status="PASS", evidence={"comparison": "with_area"})
        runner.passed({"maximum_active_field_difference": error})
    split = runner.run("split", input_text(p, cycles=2, restart_output=True))
    if runner.execute:
        checkpoint = final_restart(split)
        preserved = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in split.glob("area_surface/*.area")}
        # Read saved counters by restarting without an input file that resets them.
        restart_input = "<time>\nnlim = 4\ntlim = 10\n"
    else:
        checkpoint = split / "rst" / "Area.00002.rst"
        restart_input = "<time>\nnlim = 4\ntlim = 10\n"
    runner.passed()
    runner.run("resume", restart_input, restart=checkpoint, directory=split)
    if runner.execute:
        for filename, digest in preserved.items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
        error = equal_state(read_restart(final_restart(enabled), True),
                            read_restart(final_restart(split), True))
        continuous = [read_area(path) for path in sorted(
            enabled.glob("area_surface/*.area"))]
        resumed = [read_area(path) for path in sorted(split.glob("area_surface/*.area"))]
        reference = {int(meta["cycle"]): rows for meta, rows in continuous}
        for metadata, rows in resumed:
            assert rows.keys() == reference[int(metadata["cycle"])].keys()
            for key in rows:
                np.testing.assert_allclose(rows[key],
                                           reference[int(metadata["cycle"])][key],
                                           rtol=2e-10, atol=2e-10)
        assert {int(meta["cycle"]) for meta, _ in resumed} == set(range(5))
        runner.passed({"maximum_active_field_difference": error,
                       "preserved_event_files": len(preserved), "cycles": list(range(5))})
    collision = runner.root / "collision"
    event = collision / "area_surface" / "Area.00000.area"
    event.parent.mkdir(parents=True, exist_ok=True)
    assert not event.exists()
    event.write_text("preserve-existing-event\n")
    runner.run("collision", input_text(p), reject="exist|overwrite|collision")
    if runner.execute:
        assert event.read_text() == "preserve-existing-event\n"
        runner.passed({"existing_file_preserved": True})


def invalid_inputs(runner):
    p = Plane(levels=(2.5,), strides=(1,), methods=("sampled",))
    selection_error = "slice_x1/2/3, gid and ghost_zones are not supported"
    mesh_error = "requires a uniform three-dimensional mesh"
    cases = [
        ("bad_stride_zero", ("output1/strides=0",), "expected a positive integer stride"),
        ("bad_stride_negative", ("output1/strides=-1",),
         "expected a positive integer stride"),
        ("bad_stride_fraction", ("output1/strides=1.5",),
         "expected a positive integer stride"),
        ("bad_stride_cap", ("output1/strides=9",), "stride exceeds half"),
        ("bad_level_nan", ("output1/levels=nan",), "expected a finite real value"),
        ("bad_method", ("output1/methods=unknown",), "methods must contain"),
        ("bad_sigma", ("output1/sigma_factor=0",), "must be positive and finite"),
        ("bad_truncate", ("output1/truncate=-1",), "must be positive and finite"),
        ("duplicate_stride", ("output1/strides=1,1",), "duplicate.*stride"),
        ("duplicate_level", ("output1/levels=2.5,2.5",), "duplicate.*level"),
        ("duplicate_method", ("output1/methods=sampled,sampled",), "duplicate.*method"),
        ("bad_log_level", ("output1/sample_transform=log10", "output1/levels=0"),
         "log10 sampled areas require positive levels"),
        ("bad_log_field", ("output1/sample_transform=log10",
                           "output1/variable=hydro_w_vx"),
         "log10 sampled areas require positive finite scalar data"),
        ("bad_group", ("output1/variable=hydro_w",),
         "variable expands to multiple fields"),
        ("bad_component", ("output1/variable=hydro_w", "output1/component=99"),
         "component is outside"),
        ("bad_slice", ("output1/slice_x1=0.2",), selection_error),
        ("bad_ghosts", ("output1/ghost_zones=true",), selection_error),
        ("bad_gid", ("output1/gid=0",), selection_error),
        ("bad_crop", ("output1/x1_min=0.2",), "area requires the full lattice"),
        ("bad_2d", ("mesh/nx3=1", "meshblock/nx3=1"), mesh_error),
        ("bad_amr", ("mesh_refinement/refinement=adaptive",
                     "mesh_refinement/max_nmb_per_rank=64",
                     "amr_criterion1/method=location", "amr_criterion1/location_rad=0"),
         mesh_error),
        ("bad_smr", ("mesh_refinement/refinement=static",), mesh_error),
        ("bad_periodic_pair", ("mesh/ix1_bc=periodic",),
         "Both inner and outer x1 bcs must be periodic"),
    ]
    for name, overrides, reason in cases:
        runner.run(name, input_text(p), overrides=overrides, reject=reason)
        runner.passed()
    duplicate = ("<output3>\nfile_type = area\nid = surface\nvariable = hydro_w_d\n"
                 "levels = 2.5\nstrides = 1\ndcycle = 1\n")
    runner.run("duplicate_id", input_text(p, extra=duplicate),
               reject="give each area output a distinct id")
    runner.passed()
    directory = runner.run("configuration_initial", input_text(p))
    runner.passed()
    runner.run("configuration_changed", input_text(p, output_id="surface"),
               directory=directory,
               overrides=("output1/levels=2.0", "output1/file_number=9"),
               reject="area configuration differs")
    runner.passed()


def pytest_runner(root, backend):
    from test_suite.imported_features.feature_utils import ATHENA, _launcher
    ranks = int(os.environ.get("ATHENAK_AREA_RANKS", {
        "cpu": 1, "gpu": 1, "mpicpu": 4, "mpigpu": 2}[backend]))
    if "ATHENAK_AREA_LAUNCHER" in os.environ:
        launcher = shlex.split(os.environ["ATHENAK_AREA_LAUNCHER"])
        devices = ["--kokkos-map-device-id-by=mpi_rank"] if "gpu" in backend else []
    else:
        launcher, devices = _launcher(backend)
    return Runner(root, ATHENA, backend, ranks, launcher,
                  execute=True, device_args=devices)


def analytic(root, backend):
    planes(pytest_runner(root, backend))


def restart_and_state(root, backend):
    lifecycle(pytest_runner(root, backend))


def rejection(root, backend):
    invalid_inputs(pytest_runner(root, backend))


CASES = [("analytic_planes", analytic), ("restart_and_state", restart_and_state),
         ("invalid_inputs", rejection)]
