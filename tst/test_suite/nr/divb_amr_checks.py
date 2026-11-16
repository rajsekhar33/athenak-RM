"""
Checks shared by the moving-AMR div(B) tests.

The divb_amr problem initializes B from a discrete vector potential in a periodic
box and forces a moving refinement pattern. Face-centered div(B) must stay at
roundoff. Mass, momentum and total energy must also stay at roundoff: blocks that
share a face must hold identical copies of the normal field on that face, or
their ghost zones and fluxes disagree and the conserved totals drift.
"""

import math
import os

import pytest
import athena_read

MAX_NDIV_TOL = 2.0e-11
L1_NDIV_TOL = 2.0e-12
L2_NDIV_TOL = 5.0e-12
# Relative to the initial mass, sqrt(mass*energy) and energy, respectively.
CONSERVATION_TOL = 1.0e-12


def history_files(basename):
    return f"{basename}.user.hst", f"{basename}.mhd.hst"


def remove_histories(basename):
    for name in history_files(basename):
        if os.path.exists(name):
            os.remove(name)


def check_histories(label, basename, root_ncell, min_rows, conservation):
    """Check the div(B) history and, if requested, conservation of the totals."""
    user_file, mhd_file = history_files(basename)
    data = athena_read.hst(user_file)
    max_ndiv = max(data["max_ndiv"])
    l1_ndiv = max(s / v for s, v in zip(data["sum_ndiv"], data["vol"]))
    l2_ndiv = max(math.sqrt(s / v) for s, v in zip(data["sum_n2"], data["vol"]))
    max_ncell = max(data["ncell"])

    if len(data["time"]) < min_rows:
        pytest.fail(
            f"{label} AMR div(B) history is too short: "
            f"{len(data['time'])} rows, expected {min_rows}"
        )
    if max_ncell <= root_ncell:
        pytest.fail(
            f"{label} AMR div(B) test did not refine: "
            f"max_ncell={max_ncell:g}, root_ncell={root_ncell:g}"
        )
    if max_ndiv > MAX_NDIV_TOL:
        pytest.fail(
            f"{label} max normalized div(B) too large: "
            f"{max_ndiv:g} threshold {MAX_NDIV_TOL:g}"
        )
    if l1_ndiv > L1_NDIV_TOL:
        pytest.fail(
            f"{label} L1 normalized div(B) too large: "
            f"{l1_ndiv:g} threshold {L1_NDIV_TOL:g}"
        )
    if l2_ndiv > L2_NDIV_TOL:
        pytest.fail(
            f"{label} L2 normalized div(B) too large: "
            f"{l2_ndiv:g} threshold {L2_NDIV_TOL:g}"
        )
    if not conservation:
        return

    totals = athena_read.hst(mhd_file)
    if len(totals["time"]) != len(data["time"]):
        pytest.fail(f"{label} conserved-total history has a different length")
    mass0 = totals["mass"][0]
    energy0 = totals["tot-E"][0]
    scales = {
        "mass": mass0,
        "1-mom": math.sqrt(mass0 * energy0),
        "2-mom": math.sqrt(mass0 * energy0),
        "3-mom": math.sqrt(mass0 * energy0),
        "tot-E": energy0,
    }
    for name, scale in scales.items():
        drift = max(abs(v - totals[name][0]) for v in totals[name]) / scale
        if not drift <= CONSERVATION_TOL:
            pytest.fail(
                f"{label} {name} not conserved through AMR: relative drift "
                f"{drift:g} threshold {CONSERVATION_TOL:g}"
            )
