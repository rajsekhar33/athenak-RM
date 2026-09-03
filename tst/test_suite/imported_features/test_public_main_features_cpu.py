"""CPU regression coverage for functionality imported into the public fork."""

import pytest

from test_suite.imported_features.scenarios import FEATURE_CASES


@pytest.mark.parametrize(
    "scenario", [scenario for _, scenario in FEATURE_CASES],
    ids=[name for name, _ in FEATURE_CASES],
)
def test_public_main_features_cpu(tmp_path, scenario):
    scenario(tmp_path, "cpu")
