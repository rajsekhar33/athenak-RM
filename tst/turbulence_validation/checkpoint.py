"""Read double-precision, single-file Hydro/MHD turbulence checkpoints.

The reader retains each MeshBlock's geometry instead of interpolating AMR data.
It checks the recorded layout before exposing active cells to test oracles.
"""

from pathlib import Path
import hashlib
import math
import struct

import numpy as np


HEADER = struct.Struct("<ii9d19i19iddi")
LIFECYCLE = struct.Struct("<QQddii")
RNG_SIZE = struct.calcsize("<35qi4xd")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parameters(text):
    """Parse the parameter dump embedded in a checkpoint."""
    result = {}
    block = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            result.setdefault(block, {})
        elif block is not None and "=" in line:
            key, value = line.split("=", 1)
            result[block][key.strip()] = value.strip()
    return result


def _boolean(value):
    require(value.lower() in ("true", "false", "1", "0"), "Invalid boolean")
    return value.lower() in ("true", "1")


def read_checkpoint(path):
    """Return active block arrays, mesh geometry and restart metadata.

    Arrays have shape (block, variable, z, y, x). ``force`` and ``raw`` are
    None when restart_forcing=false. Raw-editor offsets are in bytes and refer
    to the unchanged full (ghost-inclusive) block payload.
    """
    path = Path(path)
    data = path.read_bytes()
    end = data.find(b"<par_end>\n")
    require(0 < end < 1048576, "Missing or excessive parameter header")
    params = parameters(data[:end].decode("ascii"))
    offset = end + len(b"<par_end>\n")
    owners = [name for name in ("hydro", "mhd") if name in params]
    require(len(owners) == 1, "Reader requires one Hydro or MHD fluid")
    require(not any(name in params
                    for name in ("radiation", "z4c", "adm", "ion-neutral")),
            "Unsupported additional checkpoint physics")
    owner = owners[0]
    eos = params[owner].get("eos", "")
    require(eos in ("ideal", "isothermal"), "Unsupported fluid EOS")
    nvar = (5 if eos == "ideal" else 4) + int(params[owner].get("nscalars", "0"))
    stored = ("turb_driving" in params and
              _boolean(params["turb_driving"].get("restart_forcing", "true")))
    h = HEADER.unpack_from(data, offset)
    offset += HEADER.size
    nmb, root_level = h[:2]
    require(0 < nmb < 1000000, "Invalid MeshBlock count")
    bounds = np.array(h[2:8], dtype=float).reshape(2, 3)
    mesh_indices = tuple(h[11:30])
    block_indices = tuple(h[30:49])
    ng = block_indices[0]
    mesh_xyz = tuple(mesh_indices[1:4])
    block_xyz = tuple(block_indices[1:4])
    require(0 <= ng <= 16 and min(mesh_xyz) > 0 and min(block_xyz) > 0,
            "Invalid mesh dimensions or precision/layout mismatch")
    require(all(m % b == 0 for m, b in zip(mesh_xyz, block_xyz)),
            "Root mesh must contain complete MeshBlocks")
    now, dt, cycle = h[49:52]
    require(math.isfinite(now) and math.isfinite(dt) and cycle >= 0,
            "Invalid checkpoint time/cycle")
    locations = np.frombuffer(data, "<i4", count=4*nmb,
                              offset=offset).reshape(nmb, 4).copy()
    offset += 16*nmb
    costs = np.frombuffer(data, "<f4", count=nmb, offset=offset).copy()
    offset += 4*nmb
    require(len(set(map(tuple, locations))) == nmb, "Repeated logical MeshBlock")
    require(np.all(locations[:, 3] >= root_level), "Block level below root")
    require(params.get("mesh", {}).get("checkpoint_lifecycle_version") == "1",
            "Reader requires lifecycle version 1")
    magic, version, dtold, last_dt, topology, age_count = LIFECYCLE.unpack_from(
        data, offset)
    offset += LIFECYCLE.size
    require(magic == 0x414B43504C494645 and version == 1, "Wrong lifecycle marker")
    require(age_count in (0, nmb), "Invalid refinement-age count")
    ages = np.frombuffer(data, "<i4", count=age_count, offset=offset).copy()
    offset += age_count*4
    require(np.all(ages >= 0), "Negative refinement age")
    rng = None
    updates = None
    if stored:
        rng = data[offset:offset+RNG_SIZE]
        require(len(rng) == RNG_SIZE, "Truncated RNG state")
        offset += RNG_SIZE
        updates = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        require(updates >= 0, "Negative OU update count")
    stride = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    full_xyz = tuple(b+2*ng if d == 0 or b > 1 else 1
                     for d, b in enumerate(block_xyz))
    full_shape = full_xyz[::-1]
    count = math.prod(full_shape)
    active = tuple(slice(block_indices[4+2*d], block_indices[5+2*d]+1)
                   for d in (2, 1, 0))
    require(tuple(s.stop-s.start for s in active) == block_xyz[::-1],
            "Active-index dimensions mismatch")
    cursor = nvar*count*8
    face_shapes = []
    face_offsets = []
    if owner == "mhd":
        for d in range(3):
            shape = list(full_xyz)
            shape[d] += 1
            face_shapes.append(tuple(shape[::-1]))
            face_offsets.append(cursor)
            cursor += math.prod(shape)*8
    force_offset = cursor if stored else None
    raw_offset = cursor+3*count*8 if stored else None
    if stored:
        cursor += 6*count*8
    require(stride == cursor, "Unsupported checkpoint block payload or precision")
    require(len(data) == offset+nmb*stride,
            "Checkpoint size mismatch (per-rank files are unsupported)")

    def block_array(relative, shape):
        result = np.empty((nmb,)+tuple(shape), dtype=float)
        length = math.prod(shape)
        for m in range(nmb):
            result[m] = np.frombuffer(data, "<f8", count=length,
                                      offset=offset+m*stride+relative).reshape(shape)
        return result

    all_u = block_array(0, (nvar,)+full_shape)
    u = all_u[(slice(None), slice(None))+active].copy()
    force = raw = None
    if stored:
        all_force = block_array(force_offset, (6,)+full_shape)
        force = all_force[(slice(None), slice(0, 3))+active].copy()
        raw = all_force[(slice(None), slice(3, 6))+active].copy()
    bcc = None
    faces = []
    active_faces = []
    if owner == "mhd":
        bcc = np.empty((nmb, 3)+block_xyz[::-1])
        for d, (relative, shape) in enumerate(zip(face_offsets, face_shapes)):
            field = block_array(relative, shape)
            low, high = list(active), list(active)
            axis = 2-d
            high[axis] = slice(active[axis].start+1, active[axis].stop+1)
            bcc[:, d] = .5*(field[(slice(None),)+tuple(low)] +
                            field[(slice(None),)+tuple(high)])
            faces.append(field)
            face_slice = list(active)
            face_slice[axis] = slice(active[axis].start, active[axis].stop+1)
            active_faces.append(field[(slice(None),)+tuple(face_slice)].copy())
    centers = np.empty((nmb, 3)+block_xyz[::-1])
    cell_volume = np.empty(nmb)
    widths = np.empty((nmb, 3))
    for m, loc in enumerate(locations):
        scale = 2.0**(int(loc[3])-root_level)
        axes = []
        for d in range(3):
            factor = scale if mesh_xyz[d] > 1 else 1.0
            width = (bounds[1, d]-bounds[0, d])/(mesh_xyz[d]*factor)
            require(width > 0, "Invalid physical mesh bounds")
            widths[m, d] = width
            require(0 <= loc[d] < (mesh_xyz[d]//block_xyz[d])*factor,
                    "Logical block outside mesh")
            axes.append(bounds[0, d]
                        + (loc[d]*block_xyz[d]+np.arange(block_xyz[d])+.5)*width)
        zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        centers[m] = (xx, yy, zz)
        cell_volume[m] = math.prod(widths[m])
    domain_volume = float(np.prod(bounds[1]-bounds[0]))
    require(abs(float(np.sum(cell_volume))*math.prod(block_xyz)-domain_volume)
            <= 1e-12*domain_volume, "AMR leaf volumes do not cover domain")
    for name, array in (("fluid", u), ("magnetic", bcc), ("force", force), ("raw", raw)):
        require(array is None or np.isfinite(array).all(), "Nonfinite active "+name)
    require(all(np.isfinite(field).all() for field in active_faces),
            "Nonfinite active face-centered magnetic field")
    require(np.all(u[:, 0] > 0), "Nonpositive active density")
    state = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), params=params,
                 owner=owner, eos=eos, nvar=nvar, nmb=nmb, root_level=root_level,
                 mesh_xyz=mesh_xyz, shape_xyz=block_xyz, bounds=bounds,
                 locations=locations,
                 costs=costs, ages=ages, topology=topology, time=now, dt=dt, cycle=cycle,
                 dtold=dtold, last_dt=last_dt, rng=rng, updates=updates,
                 centers=centers, cell_volume=cell_volume, widths=widths,
                 domain_volume=domain_volume, u=u, bcc=bcc, faces=faces,
                 active_faces=active_faces, force=force, raw=raw,
                 payload_offset=offset, block_stride=stride, full_shape=full_shape,
                 active_slices=active, u_offset=0, force_offset=force_offset,
                 raw_offset=raw_offset, face_offsets=face_offsets,
                 face_shapes=face_shapes)
    if eos == "ideal":
        internal = u[:, 4]-.5*np.sum(u[:, 1:4]**2, axis=1)/u[:, 0]
        if bcc is not None:
            internal -= .5*np.sum(bcc**2, axis=1)
        require(np.all(internal > 0), "Nonpositive active internal energy")
    return state


parse = read_checkpoint


def integrals(state):
    """Volume integrals of conserved quantities and energy components."""
    u = state["u"]
    weights = state["cell_volume"][:, None, None, None]

    def total(a):
        return float(np.sum(a*weights))

    kinetic = total(.5*np.sum(u[:, 1:4]**2, axis=1)/u[:, 0])
    magnetic = 0.0 if state["bcc"] is None else total(.5*np.sum(state["bcc"]**2, axis=1))
    energy = total(u[:, 4]) if state["eos"] == "ideal" else None
    return dict(mass=total(u[:, 0]), momentum=[total(u[:, d]) for d in (1, 2, 3)],
                total_energy=energy, kinetic=kinetic, magnetic=magnetic,
                internal=None if energy is None else energy-kinetic-magnetic)


def uniform_field(state, key="raw"):
    """Assemble a uniform root-level vector field for Fourier interpolation."""
    require(np.all(state["locations"][:, 3] == state["root_level"]),
            "Fourier reference must be a uniform root grid")
    values = state[key]
    require(values is not None, "Reference field not stored")
    field = np.full((values.shape[1],)+state["mesh_xyz"][::-1], np.nan)
    hits = np.zeros(state["mesh_xyz"][::-1], dtype=np.int8)
    for m, loc in enumerate(state["locations"]):
        starts = [int(loc[d])*state["shape_xyz"][d] for d in (2, 1, 0)]
        dest = tuple(slice(start, start+width)
                     for start, width in zip(starts, state["shape_xyz"][::-1]))
        field[(slice(None),)+dest] = values[m]
        hits[dest] += 1
    require(np.all(hits == 1) and np.isfinite(field).all(),
            "Incomplete uniform reference")
    return field


def compare_states(left, right, tolerance=2e-12):
    """Compare physical checkpoint state after sorting by logical block location."""
    require(left["nmb"] == right["nmb"], "Restart block count differs")
    require(left["cycle"] == right["cycle"], "Restart cycle differs")
    order_left = np.lexsort(left["locations"].T)
    order_right = np.lexsort(right["locations"].T)
    require(np.array_equal(left["locations"][order_left],
                           right["locations"][order_right]),
            "Restart topology differs")
    require(abs(left["time"]-right["time"]) <= tolerance*max(1., abs(left["time"])),
            "Restart physical time differs")
    for name in ("dt", "dtold", "last_dt"):
        require(abs(left[name]-right[name]) <= tolerance*max(1., abs(left[name])),
                "Restart timestep metadata differs: "+name)
    differences = {}
    for name in ("u", "bcc", "force", "raw"):
        a, b = left[name], right[name]
        require((a is None) == (b is None), "Restart field presence differs: "+name)
        if a is None:
            continue
        a, b = a[order_left], b[order_right]
        relative = float(np.max(np.abs(a-b))/max(1., float(np.max(np.abs(a)))))
        differences[name] = relative
        require(relative <= tolerance, "Restart field mismatch: "+name)
    for d, (a, b) in enumerate(zip(left["active_faces"], right["active_faces"])):
        a, b = a[order_left], b[order_right]
        relative = float(np.max(np.abs(a-b))/max(1., float(np.max(np.abs(a)))))
        differences[f"bface{d+1}"] = relative
        require(relative <= tolerance, "Restart face-centered field mismatch")
    require(left["updates"] == right["updates"], "Restart OU update count differs")
    require(left["rng"] == right["rng"], "Restart RNG state differs")
    require(left["topology"] == right["topology"], "Restart topology sequence differs")
    if left["ages"].size:
        require(np.array_equal(left["ages"][order_left], right["ages"][order_right]),
                "Restart refinement ages differ")
    return differences
