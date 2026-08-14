import numpy as np
import pytest

from compartment.cloud_helpers.gql import _add_v2_payloads
from compartment.helpers import compute_multi_run_compartment_deltas


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
