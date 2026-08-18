import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from tools.view_results import (
    parse_compartment_delta_stats,
    parse_compartment_deltas,
    plot_deltas_table,
)


def test_delta_parser_prefers_v2_ranges_over_flattened_legacy_values():
    run = {
        "compartment_deltas": {"I": 12.0},
        "compartment_deltas_v2": (
            '{"I": {"median": 12, "lower": 3, "upper": 20}}'
        ),
    }

    assert parse_compartment_delta_stats(run) == {
        "I": {"value": 12.0, "lower": 3.0, "upper": 20.0}
    }
    assert parse_compartment_deltas(run) == {"I": 12.0}


def test_delta_parser_supports_historical_mean_stats_in_legacy_field():
    run = {
        "compartment_deltas": {
            "I": {"mean": "12", "lower": "3", "upper": "20"},
            "bad": None,
        }
    }

    assert parse_compartment_delta_stats(run) == {
        "I": {"value": 12.0, "lower": 3.0, "upper": 20.0}
    }


def test_delta_table_places_muted_percentile_range_beside_central_value():
    run = {
        "compartment_deltas_v2": {
            "I": {"median": 1234, "lower": 1000, "upper": 1500}
        }
    }
    fig, ax = plt.subplots()

    assert plot_deltas_table(
        ax,
        [("With interventions", run)],
        ["I"],
        {"I": ("Infected", "#A8228E")},
    )

    table = ax.tables[0]
    assert table[0, 2].get_text().get_text() == "2.5% → 97.5%"
    assert table[1, 1].get_text().get_text() == "1,234"
    assert table[1, 2].get_text().get_text() == "1,000 → 1,500"
    assert table[1, 2].get_text().get_color() == "0.45"
    plt.close(fig)
