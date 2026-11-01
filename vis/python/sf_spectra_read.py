"""Read AthenaK structure-function (.sf) and power-spectrum (.spec) outputs.

``read_sf(path)`` and ``read_spec(path)`` return a metadata dictionary (string
values) and a dictionary of NumPy arrays. Arrays retain the file's byte order
and are read-only views of its payload; use ``array.copy()`` to modify them.
"""

from math import prod

import numpy as np


def _read_header(stream, kind):
    """Read a version-one ASCII header and verify the payload offset."""
    first = stream.readline()
    if first != f"Athena {kind} format_version=1\n".encode("ascii"):
        raise ValueError(f"expected Athena {kind} format_version=1")
    meta = {f"Athena {kind} format_version": "1"}
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("file ends inside the header")
        try:
            line = raw.decode("ascii").rstrip("\n")
        except UnicodeDecodeError as exc:
            raise ValueError("header contains non-ASCII bytes") from exc
        key, separator, value = line.partition("=")
        if not separator or key in meta:
            raise ValueError(f"invalid or repeated header key: {key!r}")
        meta[key] = value
        if key == "header offset":
            try:
                offset = int(value)
            except ValueError as exc:
                raise ValueError(f"invalid header offset: {value!r}") from exc
            if offset != stream.tell():
                raise ValueError("header offset differs from the header length")
            break
    if meta.get("byte_order") not in ("little", "big"):
        raise ValueError("byte_order must be 'little' or 'big'")
    return meta


def _dimension(meta, key):
    """Read a required positive integer dimension."""
    try:
        value = int(meta[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"missing or invalid dimension {key!r}") from exc
    if value < 1:
        raise ValueError(f"dimension {key!r} must be positive")
    return value


def _arrays(payload, layout, meta):
    """Check the complete payload size before constructing array views."""
    expected = sum(8 * prod(shape) for _, shape, _ in layout)
    if len(payload) != expected:
        raise ValueError(f"payload is {len(payload)} bytes; expected {expected}")
    order = "<" if meta["byte_order"] == "little" else ">"
    arrays, offset = {}, 0
    for name, shape, dtype in layout:
        count = prod(shape)
        arrays[name] = np.frombuffer(
            payload, dtype=order + dtype, count=count, offset=offset).reshape(shape)
        offset += 8 * count
    return arrays


def read_sf(path):
    """Return ``(metadata, arrays)`` for any version-one SF mode or histogram."""
    with open(path, "rb") as stream:
        meta = _read_header(stream, "sf")
        payload = stream.read()
    mode = meta.get("mode")
    if mode not in ("scalar", "long_trans", "aniso_mhd", "correlation"):
        raise ValueError(f"unknown structure-function mode: {mode!r}")
    nw, nq, order, nb, nr = (_dimension(meta, key) for key in
                             ("nweight", "nq", "max_order", "nbin", "nreplicate"))
    if mode == "aniso_mhd":
        bins = [_dimension(meta, key) for key in
                ("nbin_l", "nbin_prll", "nbin_perp")]
        if sum(bins) != nb:
            raise ValueError("anisotropic bin families do not sum to nbin")
        layout = [(name, (n + 1,), "f8") for name, n in
                  zip(("rbins", "rbins_prll", "rbins_perp"), bins)]
    else:
        layout = [("rbins", (nb + 1,), "f8")]
    layout += [("T", (nw, nq, order, nb), "f8"),
               ("W", (nw, nb), "f8"), ("C", (nw, nb), "f8"),
               ("T_er", (2, nr, nw, nq, order, nb), "f8"),
               ("W_er", (2, nr, nw, nb), "f8"),
               ("C_er", (2, nr, nw, nb), "f8"),
               ("n_acc_er", (2, nr, nw, nb), "u8")]
    histogram = meta.get("histogram", "0")
    if histogram not in ("0", "1"):
        raise ValueError("histogram must be 0 or 1")
    if histogram == "1":
        nc = _dimension(meta, "hist_ncol")
        try:
            edges = [int(value) for value in meta["hist_nedge"].split(",")]
        except (KeyError, ValueError) as exc:
            raise ValueError("missing or invalid hist_nedge") from exc
        if mode not in ("scalar", "long_trans") or len(edges) != nq or any(
                n < 1 or n >= nc for n in edges):
            raise ValueError("invalid histogram mode or edge counts")
        layout += [("hist_edges", (nw, nq, nc - 1), "f8"),
                   ("hist_sigma", (nw, nq), "f8"),
                   ("H", (nw, nq, nb, nc), "f8"),
                   ("H_er", (2, nr, nw, nq, nb, nc), "f8")]
    return meta, _arrays(payload, layout, meta)


def read_spec(path):
    """Return ``(metadata, arrays)`` for a version-one scalar or Helmholtz spectrum."""
    with open(path, "rb") as stream:
        meta = _read_header(stream, "spec")
        payload = stream.read()
    nb, nv, nw = (_dimension(meta, key) for key in ("nbin", "nvar", "nweight"))
    layout = [("kbins", (nb + 1,), "f8"), ("k", (nb,), "f8"), ("nk", (nb,), "f8")]
    mode = meta.get("mode")
    if mode == "scalar":
        layout += [("E", (nw, nv, nb), "f8"), ("P_DC", (nw, nv), "f8"),
                   ("sumsq", (nv,), "f8")]
    elif mode == "helmholtz":
        if nv != 3 or nw != 1:
            raise ValueError("Helmholtz spectra require nvar=3 and nweight=1")
        layout += [("E_comp", (nb,), "f8"), ("E_sol", (nb,), "f8"),
                   ("P_DC", (1,), "f8"), ("sumsq", (1,), "f8")]
    else:
        raise ValueError(f"unknown spectrum mode: {mode!r}")
    return meta, _arrays(payload, layout, meta)
