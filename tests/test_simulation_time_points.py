from datetime import date

import numpy as np

from compartment.helpers import (
    compute_parent_admin_total,
    create_jax_intervention_results,
    get_simulation_time_points,
)


def test_one_year_grid_includes_both_calendar_endpoints():
    time_points = get_simulation_time_points(365)

    assert len(time_points) == 366
    np.testing.assert_array_equal(time_points, np.arange(366, dtype=float))


def test_downsampled_grid_includes_divisible_endpoint():
    time_points = get_simulation_time_points(366)

    assert time_points[-1] == 366
    assert len(time_points) == 184
    assert time_points[-2] == 364


def test_downsampled_grid_appends_non_divisible_endpoint():
    time_points = get_simulation_time_points(367)

    assert time_points[-2:].tolist() == [366, 367]
    assert len(time_points) == 185


def test_deterministic_intervention_can_end_on_simulation_end_date():
    start_date = date(2026, 8, 20)
    duration = 365
    population_matrix = np.ones((duration + 1, 1, 1))

    events = create_jax_intervention_results(
        population_matrix=population_matrix,
        intervention_dict={
            "mask_wearing": {
                "start_date_ordinal": start_date.toordinal(),
                "end_date_ordinal": start_date.toordinal() + duration,
                "start_threshold": None,
                "end_threshold": None,
            }
        },
        compartment_list=["I"],
        start_date=start_date,
        disease_type="TEST",
        n_timesteps=duration,
    )

    assert events == [
        {
            "id": "mask_wearing",
            "trigger_date": "2026-08-20",
            "trigger_type": "DATE",
            "active": True,
        },
        {
            "id": "mask_wearing",
            "trigger_date": "2027-08-20",
            "trigger_type": "DATE",
            "active": False,
        },
    ]


def test_parent_admin_total_reuses_child_zone_dates():
    """The parent series must not re-derive dates from a fixed step.

    On a downsampled run the last sample sits off the regular grid, so
    ``t * step`` would run past the requested end date.
    """
    # A 6-day step whose final interval is short: ..., 12, 18, 20.
    zone_dates = ["2026-01-01", "2026-01-07", "2026-01-13", "2026-01-19", "2026-01-21"]
    zones = [
        {
            "time_series": [
                {"date": d, "I": {"age_all": 1.0}} for d in zone_dates
            ]
        },
        {
            "time_series": [
                {"date": d, "I": {"age_all": 2.0}} for d in zone_dates
            ]
        },
    ]

    parent = compute_parent_admin_total(
        zones,
        payload={"AdminUnit": {"id": "parent-id"}, "owner": "owner-id"},
        unique_id="result-id",
        parent_unique_id="parent-result-id",
    )

    assert [entry["date"] for entry in parent["time_series"]] == zone_dates
    assert parent["time_series"][-1]["I"]["age_all"] == 3.0
