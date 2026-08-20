import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from tools.view_results import (
    extract_model_artifact,
    order_compartments,
    parse_compartment_delta_stats,
    parse_compartment_deltas,
    plot_deltas_table,
    plot_panel,
    resolve_labels,
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


def test_artifact_labels_and_display_order_drive_local_viewer():
    artifact = {
        "compartments": [
            {"id": "S", "label": "Susceptible"},
            {"id": "F", "label": "Dead, awaiting burial"},
        ],
        "compartment_display_order": ["S", "F"],
    }
    runs = [{"model_artifact": artifact}]

    parsed_artifact = extract_model_artifact(runs)

    assert order_compartments(["F", "S"], parsed_artifact) == ["S", "F"]
    assert resolve_labels(["F"], parsed_artifact)["F"][0] == (
        "Dead, awaiting burial"
    )

    run = {
        "parent_admin_total": {
            "time_series": [
                {"date": "2026-01-01", "F": {"age_all": 1}},
                {"date": "2026-01-02", "F": {"age_all": 2}},
            ]
        }
    }
    fig, ax = plt.subplots()
    plot_panel(
        ax,
        run,
        resolve_labels(["F"], parsed_artifact),
        ["F"],
        log_scale=False,
        draw_markers=False,
    )
    assert ax.get_legend_handles_labels()[1] == ["Dead, awaiting burial"]
    plt.close(fig)


def test_old_results_keep_legacy_label_fallbacks():
    assert extract_model_artifact([{}]) == {}
    assert resolve_labels(["I", "unknown"])["I"][0] == "Infected"
    assert resolve_labels(["I", "unknown"])["unknown"][0] == "unknown"
