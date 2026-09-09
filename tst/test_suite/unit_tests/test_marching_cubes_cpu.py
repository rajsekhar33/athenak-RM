"""Functional test for the temperature-isosurface marching-cubes regression pgen."""

import test_suite.testutils as testutils


def test_marching_cubes_temperature_area():
    testutils.run("inputs/ut_marching_cubes.athinput")
