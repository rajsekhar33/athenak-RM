"""
Test face-centered div(B) preservation and conservation under moving AMR with MPI.

The divb_amr problem initializes B from a discrete vector potential and forces a
moving AMR refinement pattern. The 2D and deep-3D cases exercise repeated
refinement and derefinement across ranks. The 2D and two-level 3D cases also
refine blocks next to existing same-level blocks; mass, momentum and total energy
must then be conserved to roundoff.
"""

import pytest
import test_suite.testutils as testutils
from test_suite.nr.divb_amr_checks import check_histories, remove_histories


_CASES = [
    pytest.param(
        "2D",
        "inputs/divb_amr_2d.athinput",
        "DivBAMR2D",
        [],
        48 * 48,
        20,
        True,
        id="2d",
    ),
    pytest.param(
        "3D-L1",
        "inputs/divb_amr_3d.athinput",
        "DivBAMR3DL1",
        ["mesh_refinement/num_levels=2", "problem/refine_levels=1", "time/nlim=12"],
        32 * 32 * 32,
        12,
        True,
        id="3d-l1",
    ),
    # Two physical refinement levels are the minimum 3D case that exposes the
    # deep-AMR div(B) bug.
    pytest.param(
        "3D-L2",
        "inputs/divb_amr_3d.athinput",
        "DivBAMR3DL2",
        [],
        32 * 32 * 32,
        8,
        False,
        id="3d-l2",
    ),
]


@pytest.mark.parametrize(
    "label,input_file,basename,extra,root_ncell,min_rows,conservation", _CASES
)
def test_run(label, input_file, basename, extra, root_ncell, min_rows, conservation):
    """Run one moving-AMR case and check its div(B) and conservation histories."""
    try:
        remove_histories(basename)
        results = testutils.mpi_run(
            input_file,
            [f"job/basename={basename}", "output1/user_hist_only=false"] + extra,
            threads=8,
        )
        assert results, f"{label} AMR div(B) test run failed."
        check_histories(label, basename, root_ncell, min_rows, conservation)
    finally:
        remove_histories(basename)
        testutils.cleanup()
