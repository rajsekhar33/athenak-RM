"""GPU smoke test for non-relativistic Hydro first-order flux correction (FOFC)."""

import test_suite.testutils as testutils


def test_hydro_fofc_multiple_meshblocks():
    """Exercise Hydro FOFC storage over multiple three-dimensional MeshBlocks."""
    try:
        testutils.run(
            "inputs/lwave_hydro_fofc.athinput",
            [
                "job/basename=hydro_fofc",
            ],
        )
    finally:
        testutils.cleanup()
