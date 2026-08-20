from datetime import date

import numpy as np

from compartment.helpers import (
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
        step=1,
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
