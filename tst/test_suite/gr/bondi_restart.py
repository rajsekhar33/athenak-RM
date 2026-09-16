"""Regression for rebuilding Bondi boundary parameters after fluid restart."""

import importlib.util

import numpy as np

from test_suite.imported_features.feature_utils import REPO_ROOT, latest, run_athena


def check_bondi_restart(root, backend):
    """Require eight-cycle continuation to agree after restarting at cycle three."""
    spec = importlib.util.spec_from_file_location(
        "bondi_binary_reader", REPO_ROOT / "vis/python/bin_convert.py"
    )
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    reference, split, resumed = (
        root / name for name in ("reference", "split", "resumed")
    )
    input_name = "gr_bondi_restart.athinput"
    run_athena(reference, input_name, backend, ["time/nlim=8"])
    run_athena(split, input_name, backend, ["time/nlim=3"])
    saved = reader.read_binary(latest(split, "bin/*.bin"))
    assert saved["cycle"] == 3
    run_athena(resumed, input_name, backend, ["time/nlim=8"],
               restart=latest(split, "rst/*.rst"))
    left = reader.read_binary(latest(reference, "bin/*.bin"))
    right = reader.read_binary(latest(resumed, "bin/*.bin"))
    assert left["cycle"] == right["cycle"] == 8
    assert right["time"] > saved["time"]
    for key in ("time", "mb_logical", "mb_geometry"):
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    fields = {"dens", "mom1", "mom2", "mom3", "ener"}
    assert set(left["mb_data"]) == set(right["mb_data"]) == fields
    for field in sorted(fields):
        x, y = np.asarray(left["mb_data"][field]), np.asarray(right["mb_data"][field])
        assert np.isfinite(x).all() and np.isfinite(y).all(), field
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12, err_msg=field)
