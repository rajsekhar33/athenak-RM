"""
Test face-centered div(B) preservation and conservation under moving AMR on GPU.

Single-rank counterpart of test_nr_divb_amr_mpicpu.py. All blocks share one rank,
so face data for newly refined blocks is exchanged without MPI.
"""

import pytest
import test_suite.testutils as testutils
from test_suite.nr.divb_amr_checks import check_histories, remove_histories


_CASES = [
    pytest.param(
        "2D",
        "inputs/divb_amr_2d.athinput",
        "DivBAMR2DGPU",
        [],
        48 * 48,
        20,
        id="2d",
    ),
    pytest.param(
        "3D-L1",
        "inputs/divb_amr_3d.athinput",
        "DivBAMR3DL1GPU",
        ["mesh_refinement/num_levels=2", "problem/refine_levels=1", "time/nlim=12"],
        32 * 32 * 32,
        12,
        id="3d-l1",
    ),
]


@pytest.mark.parametrize("label,input_file,basename,extra,root_ncell,min_rows", _CASES)
def test_run(label, input_file, basename, extra, root_ncell, min_rows):
    """Run one moving-AMR case and check its div(B) and conservation histories."""
    try:
        remove_histories(basename)
        results = testutils.run(
            input_file,
            [
                f"job/basename={basename}",
                "output1/user_hist_only=false",
                "mesh_refinement/max_nmb_per_rank=2048",
            ]
            + extra,
        )
        assert results, f"{label} AMR div(B) test run failed."
        check_histories(label, basename, root_ncell, min_rows, True)
    finally:
        remove_histories(basename)
        testutils.cleanup()
