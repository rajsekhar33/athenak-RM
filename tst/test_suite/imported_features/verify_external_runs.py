"""Verify completed public-main feature artifacts without launching Athena."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


TST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TST_ROOT))

from test_suite.imported_features.scenarios import (  # noqa: E402
    verify_dt_wall,
    verify_hooks_and_geometry,
    verify_mhd_rk4,
    verify_particles,
    verify_profile_and_pdf_exactness,
    verify_turbulence_restart,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate completed public-main feature runs; never launches Athena."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Run root is missing: {root}")

    verify_hooks_and_geometry(root / "hooks_geometry", args.backend)
    verify_profile_and_pdf_exactness(root / "profile_pdf")
    verify_dt_wall(root / "dt_wall")
    verify_turbulence_restart(
        root / "turbulence_reference", root / "turbulence_restart"
    )
    verify_particles(root)
    verify_mhd_rk4(root)
    print(f"PUBLIC_MAIN_FEATURE_VERIFY_PASS backend={args.backend} run_root={root}")


if __name__ == "__main__":
    main()
