"""Standalone helpers for the turbulence-driver property checks.

Deliberately kept separate from imported_features/feature_utils.py: these tests are
not part of the shared public-main feature suite everyone runs (see
test_turbulence_driver_properties_cpu.py's module docstring), so they carry their own
small, self-contained reader instead of adding a dependency edge onto that module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_mesh_binary(path: Path) -> dict[str, np.ndarray]:
    """Read an Athena mesh 'bin' dump and return each variable stacked across all cells
    of all MeshBlocks, in file order (no coordinate reconstruction, single-file only)."""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        filesize = stream.tell()
        stream.seek(0, 0)
        code_header = stream.readline().split()
        if not code_header or code_header[0] != b"Athena":
            raise AssertionError(f"Unexpected bin header in {path}")
        pheader_count = int(stream.readline().split(b"=")[-1])
        pheader: dict[str, str] = {}
        for _ in range(pheader_count - 1):
            key, value = stream.readline().decode("ascii").strip().split("=")
            pheader[key.strip()] = value.strip()
        locsizebytes = int(pheader["size of location"])
        varsizebytes = int(pheader["size of variable"])
        nvars = int(stream.readline().split(b"=")[-1])
        var_list = [name.decode("ascii") for name in stream.readline().split()[1:]]
        header_size = int(stream.readline().split(b"=")[-1])
        stream.read(header_size)  # parameter-file text block, unused here

        vardtype = np.dtype("=f8" if varsizebytes == 8 else "=f4")
        blocks: list[np.ndarray] = []
        while stream.tell() < filesize:
            mb_index = np.frombuffer(stream.read(24), dtype=np.int32)
            nx1_out = int(mb_index[1] - mb_index[0]) + 1
            nx2_out = int(mb_index[3] - mb_index[2]) + 1
            nx3_out = int(mb_index[5] - mb_index[4]) + 1
            stream.read(16)  # logical location, unused here
            stream.read(6 * locsizebytes)  # geometry, unused here
            count = nx1_out * nx2_out * nx3_out * nvars
            data = np.fromfile(stream, dtype=vardtype, count=count)
            blocks.append(data.reshape(nvars, -1))
    stacked = np.concatenate(blocks, axis=1)
    return {name: stacked[i] for i, name in enumerate(var_list)}
