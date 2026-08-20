from datetime import date

import numpy as np
import pytest

from compartment.cloud_helpers.gql import _add_v2_payloads
from compartment.helpers import (
    compute_multi_run_compartment_deltas,
    format_uncertainty_output,
)


def test_compartment_deltas_use_median_and_95_percentile_interval():
    # Shape: (runs, timesteps, compartments, regions). The deliberately skewed
    # final values make the median (1.5) visibly different from the mean (25.75).
    population_matrix = np.array(
        [
            [[[0.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[0.0]], [[2.0]]],
            [[[0.0]], [[100.0]]],
        ]
    )

    result = compute_multi_run_compartment_deltas(
        population_matrix,
        disease_type="TEST",
        n_regions=1,
        compartment_list=["I"],
    )

    assert set(result["I"]) == {"median", "lower", "upper"}
    assert result["I"]["median"] == pytest.approx(1.5)
    assert result["I"]["lower"] == pytest.approx(np.percentile([0, 1, 2, 100], 2.5))
    assert result["I"]["upper"] == pytest.approx(np.percentile([0, 1, 2, 100], 97.5))


def test_v2_deltas_preserve_median_schema_and_legacy_field_gets_central_value():
    results = {
        "compartment_deltas": {
            "I": {"median": 12.0, "lower": 3.0, "upper": 20.0}
        }
    }

    _add_v2_payloads(results)

    assert results["compartment_deltas"] == {"I": 12.0}
    assert (
        results["compartment_deltas_v2"]
        == '{"I": {"median": 12.0, "lower": 3.0, "upper": 20.0}}'
    )


def test_multi_run_output_reports_date_interventions_once_and_skips_thresholds():
    shape_child = (7, 1, 1)
    shape_parent = (7, 1)
    zeros_child = np.zeros(shape_child)
    zeros_parent = np.zeros(shape_parent)
    jan_1 = date(2026, 1, 1)

    result = format_uncertainty_output(
        zeros_child,
        zeros_child,
        zeros_child,
        zeros_parent,
        zeros_parent,
        zeros_parent,
        payload={
            "id": "job-id",
            "simulation_type": "COMPARTMENTAL",
            "owner": "owner-id",
            "start_date": "2026-01-01",
            "end_date": "2026-01-07",
            "time_steps": 6,
            "case_file": {"admin_zones": [{"id": "zone-id"}]},
            "AdminUnit": {"id": "parent-id"},
        },
        compartment_list=["I"],
        admin_units=["zone-id"],
        start_date="2026-01-01",
        n_timesteps=6,
        compartment_deltas={"I": {"median": 0, "lower": 0, "upper": 0}},
        intervention_dict={
            "mask_wearing": {
                "start_date_ordinal": jan_1.toordinal() + 1,
                "end_date_ordinal": jan_1.toordinal() + 6,
            },
            "lock_down": {
                "start_date_ordinal": None,
                "end_date_ordinal": None,
                "start_threshold": 0.05,
                "end_threshold": 0.01,
            },
        },
    )

    assert result["intervention_results"] == [
        {
            "id": "mask_wearing",
            "trigger_date": "2026-01-02",
            "trigger_type": "DATE",
            "active": True,
        },
        {
            "id": "mask_wearing",
            "trigger_date": "2026-01-07",
            "trigger_type": "DATE",
            "active": False,
        },
    ]
    assert result["parent_admin_total"]["time_series"][-1]["date"] == "2026-01-07"
