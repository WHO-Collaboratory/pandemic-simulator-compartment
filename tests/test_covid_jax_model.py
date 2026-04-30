"""COVID model tests.

Covers compartment structure validation, disease dynamics, all variant schemas,
end-to-end runs for every COVID variant, and uncertainty (LHS) runs.

Run:
    python3 -m pytest tests/test_covid_jax_model.py -v -m integration
    python3 -m pytest tests/test_covid_jax_model.py -v  # unit tests only
"""

import json
import pathlib
import tempfile
import pytest
from helpers import _run_model, MODELS_DIR


# ---------------------------------------------------------------------------
# Compartment structure tests
# ---------------------------------------------------------------------------


class TestFlexibleCompartments:
    """Test that each COVID variant uses its own fixed compartment set."""

    def _make_covid_config(self) -> dict:
        """Build a minimal COVID SEIHDR config."""
        return {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "admin_zones": [
                {
                    "name": "Zone A",
                    "center_lat": 47.0,
                    "center_lon": 8.0,
                    "population": 100000,
                    "infected_population": 1.0,
                }
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {"transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"}, "value": 0.3},
                    {"transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"}, "value": 0.1},
                ]
            },
            "Interventions": {"items": []},
        }

    @pytest.mark.integration
    def test_sir_only_no_extra_compartments(self):
        """CovidSIRModel should produce only S, I, R — no E, H, D."""
        from compartment.models.covid_jax_model.variants import CovidSIRModel

        config = {
            "Disease": {"disease_type": "COVID_SIR"},
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0,
                 "population": 100000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 0},
            "TransmissionEdges": {
                "items": [
                    {"transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"}, "value": 0.3},
                    {"transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"}, "value": 7.0},
                ]
            },
            "Interventions": {"items": []},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidSIRModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        assert "S" in compartments
        assert "I" in compartments
        assert "R" in compartments
        assert "E" not in compartments, "E should not appear in SIR run"
        assert "H" not in compartments, "H should not appear in SIR run"
        assert "D" not in compartments, "D should not appear in SIR run"

    @pytest.mark.integration
    def test_seir_has_exposed_but_no_hospitalized(self):
        """CovidSEIRModel should have E but not H or D."""
        from compartment.models.covid_jax_model.variants import CovidSEIRModel

        config = {
            "Disease": {"disease_type": "COVID_SEIR"},
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0,
                 "population": 100000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 0},
            "TransmissionEdges": {
                "items": [
                    {"transmission_edge": {"source": "susceptible", "target": "exposed", "value_type": "RATE"}, "value": 0.3},
                    {"transmission_edge": {"source": "exposed", "target": "infected", "value_type": "DAYS"}, "value": 5.0},
                    {"transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"}, "value": 7.0},
                ]
            },
            "Interventions": {"items": []},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidSEIRModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        assert "E" in compartments
        assert "H" not in compartments
        assert "D" not in compartments

    @pytest.mark.integration
    def test_covid_sir_population_conserved(self):
        """In SIR (no deaths), total population at start should equal end."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = self._make_covid_config()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = [k for k in ts[0].keys() if k != "date"]

        def _sum_pop(entry):
            total = 0
            for c in compartments:
                val = entry[c]
                if isinstance(val, dict):
                    total += val.get("age_all", 0)
            return total

        pop_start = _sum_pop(ts[0])
        pop_end = _sum_pop(ts[-1])

        assert pop_start > 0, "Starting population should be non-zero"
        assert abs(pop_start - pop_end) / pop_start < 0.001, (
            f"Population not conserved: start={pop_start:.1f}, end={pop_end:.1f}, "
            f"drift={abs(pop_start - pop_end):.1f} ({abs(pop_start - pop_end) / pop_start * 100:.4f}%)"
        )

    @pytest.mark.integration
    def test_dengue_uses_fixed_compartments(self):
        """Dengue should always use its fixed strain-specific compartments."""
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        results = _run_model(
            DengueJaxModel,
            MODELS_DIR / "dengue_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        assert "S0" in compartments or "SV" in compartments, (
            f"Expected dengue-specific compartments, got {compartments}"
        )


# ---------------------------------------------------------------------------
# COVID disease dynamics tests
# ---------------------------------------------------------------------------


class TestCovidDisease:
    """Tests specific to COVID model dynamics."""

    @pytest.mark.integration
    def test_full_seihdr_run(self):
        """Full SEIHDR run with all 6 compartments should succeed."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        results = _run_model(
            CovidJaxModel,
            MODELS_DIR / "covid_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = {k for k in ts[0].keys() if k != "date"}
        for c in ["S", "E", "I", "H", "D", "R"]:
            assert c in compartments, f"Missing compartment {c} in full SEIHDR run"

    @pytest.mark.integration
    def test_intervention_changes_output(self):
        """With-interventions run should differ from control at epidemic peak."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        results = _run_model(
            CovidJaxModel,
            MODELS_DIR / "covid_jax_model" / "example-config.json",
        )
        with_run = next(r for r in results if not r["control_run"])
        ctrl_run = next(r for r in results if r["control_run"])

        with_ts = with_run["parent_admin_total"]["time_series"]
        ctrl_ts = ctrl_run["parent_admin_total"]["time_series"]

        mid = len(with_ts) // 2
        i_with = with_ts[mid]["I"]["age_all"]
        i_ctrl = ctrl_ts[mid]["I"]["age_all"]
        assert i_with != i_ctrl, (
            f"Intervention run and control are identical at midpoint (I={i_with}). "
            "Interventions may not be applied."
        )

    @pytest.mark.integration
    def test_multi_zone_travel(self):
        """With 2+ zones and travel, infection should spread between zones."""
        from compartment.models.covid_jax_model.variants import CovidSIRModel

        config = {
            "Disease": {"disease_type": "COVID_SIR"},
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 500000, "infected_population": 2.0},
                {"name": "Zone B", "center_lat": 48.0, "center_lon": 9.0, "population": 500000, "infected_population": 0.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {"transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"}, "value": 0.3},
                    {"transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"}, "value": 7.0},
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidSIRModel, pathlib.Path(config_path))

        zones = results[0]["admin_zones"]
        zone_b_ts = zones[1]["time_series"]
        zone_b_r_end = zone_b_ts[-1]["R"]["age_all"]
        assert zone_b_r_end > 0, (
            "Zone B started with 0 infected but never saw infections — travel may be broken"
        )

    @pytest.mark.integration
    def test_age_groups_sum_to_all(self):
        """age_0_17 + age_18_55 + age_56_plus should equal age_all."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        results = _run_model(
            CovidJaxModel,
            MODELS_DIR / "covid_jax_model" / "example-config.json",
        )
        ts = results[0]["parent_admin_total"]["time_series"]
        for entry in ts:
            for key, val in entry.items():
                if key == "date":
                    continue
                if isinstance(val, dict) and "age_all" in val:
                    parts = (
                        val.get("age_0_17", 0)
                        + val.get("age_18_55", 0)
                        + val.get("age_56_plus", 0)
                    )
                    assert abs(parts - val["age_all"]) < 1.0, (
                        f"Age groups don't sum to age_all for {key} on {entry['date']}: "
                        f"{parts:.2f} != {val['age_all']:.2f}"
                    )

    @pytest.mark.integration
    def test_demographic_rate_overrides_change_outcome(self):
        """High fatality rate for elderly should produce more deaths in that group.

        Setup: two equal-sized groups (young / elderly), same initial conditions,
        same beta/gamma — but elderly delta is 500x higher. After 150 days,
        elderly deaths must exceed young deaths by a large margin.
        """
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "compartment_list": ["S", "E", "I", "D", "R"],
            },
            "start_date": "2025-01-01",
            "end_date": "2025-06-01",
            "admin_zones": [
                {
                    "name": "TestZone",
                    "center_lat": 0.0,
                    "center_lon": 0.0,
                    "population": 1000000,
                    "infected_population": 5.0,
                }
            ],
            "demographics": {"young": 50.0, "elderly": 50.0},
            "travel_volume": {"leaving": 0},
            "contact_matrix_overrides": {
                "young":   {"young": 1.0, "elderly": 0.0},
                "elderly": {"young": 0.0, "elderly": 1.0},
            },
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"value_type": "RATE", "source": "susceptible", "target": "exposed"},
                        "value": 0.35,
                    },
                    {
                        "transmission_edge": {"value_type": "DAYS", "source": "exposed", "target": "infected"},
                        "value": 5.0,
                    },
                    {
                        "transmission_edge": {"value_type": "RATE", "source": "infected", "target": "deceased"},
                        "value": 0.0001,
                    },
                    {
                        "transmission_edge": {"value_type": "DAYS", "source": "infected", "target": "recovered"},
                        "value": 7.0,
                    },
                ]
            },
            "Interventions": {"items": []},
            "demographic_rate_overrides": {
                "delta": {"young": 0.0001, "elderly": 0.05},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))
        ts = results[0]["parent_admin_total"]["time_series"]
        final = ts[-1]

        d_entry = final.get("D")
        assert isinstance(d_entry, dict), (
            f"Expected D compartment to be a dict with per-group values, got {type(d_entry)}"
        )
        d_young = d_entry.get("young", 0)
        d_elderly = d_entry.get("elderly", 0)

        assert d_elderly > 0, "Elderly group recorded zero deaths — rate override may not be applied"
        assert d_young >= 0, "Young group deaths should be non-negative"
        assert d_elderly > 10 * d_young, (
            f"Elderly deaths ({d_elderly:.0f}) should be at least 10x young deaths ({d_young:.0f}). "
            "demographic_rate_overrides for delta may not be flowing through the derivative."
        )


# ---------------------------------------------------------------------------
# COVID variant schema tests (unit — no model execution)
# ---------------------------------------------------------------------------


class TestCovidVariantSchemas:
    """Unit tests verifying each COVID variant has the correct compartment structure."""

    _REGISTRY = None

    @classmethod
    def _get_registry(cls):
        if cls._REGISTRY is None:
            from compartment.models.covid_jax_model.model import CovidJaxModel
            from compartment.models.covid_jax_model.variants import (
                CovidSEIRModel, CovidSIHRModel, CovidSIDRModel,
                CovidSEIHRModel, CovidSEIDRModel, CovidSIHDRModel, CovidSIRModel,
            )
            cls._REGISTRY = {
                "COVID_SEIHDR": CovidJaxModel,
                "COVID_SIR":    CovidSIRModel,
                "COVID_SEIR":   CovidSEIRModel,
                "COVID_SIHR":   CovidSIHRModel,
                "COVID_SIDR":   CovidSIDRModel,
                "COVID_SEIHR":  CovidSEIHRModel,
                "COVID_SEIDR":  CovidSEIDRModel,
                "COVID_SIHDR":  CovidSIHDRModel,
            }
        return cls._REGISTRY

    @pytest.mark.parametrize("disease_type,expected,excluded", [
        ("COVID_SIR",   ["S", "I", "R"],          ["E", "H", "D"]),
        ("COVID_SEIR",  ["S", "E", "I", "R"],     ["H", "D"]),
        ("COVID_SIHR",  ["S", "I", "H", "R"],     ["E", "D"]),
        ("COVID_SIDR",  ["S", "I", "D", "R"],     ["E", "H"]),
        ("COVID_SEIHR", ["S", "E", "I", "H", "R"],  ["D"]),
        ("COVID_SEIDR", ["S", "E", "I", "D", "R"],  ["H"]),
        ("COVID_SIHDR", ["S", "I", "H", "D", "R"],  ["E"]),
        ("COVID_SEIHDR",["S", "E", "I", "H", "D", "R"], []),
    ])
    def test_compartment_list(self, disease_type, expected, excluded):
        model_class = self._get_registry()[disease_type]
        compartment_list = model_class.COMPARTMENT_LIST
        for c in expected:
            assert c in compartment_list, f"{disease_type} should have {c} in COMPARTMENT_LIST"
        for c in excluded:
            assert c not in compartment_list, f"{disease_type} should not have {c} in COMPARTMENT_LIST"

    def test_all_variants_have_sir_base(self):
        """Every variant must include the mandatory S, I, R compartments."""
        for disease_type, model_class in self._get_registry().items():
            for c in ["S", "I", "R"]:
                assert c in model_class.COMPARTMENT_LIST, (
                    f"{disease_type} is missing mandatory compartment {c}"
                )

    def test_disease_type_attributes(self):
        """Each variant class must expose the correct DISEASE_TYPE string."""
        registry = self._get_registry()
        for disease_type, model_class in registry.items():
            assert model_class.DISEASE_TYPE == disease_type, (
                f"{model_class.__name__}.DISEASE_TYPE should be '{disease_type}', "
                f"got '{model_class.DISEASE_TYPE}'"
            )

    @pytest.mark.parametrize("disease_type", [
        "COVID_SIR", "COVID_SIHR", "COVID_SIDR", "COVID_SIHDR",
    ])
    def test_no_e_variant_has_s_to_i_beta(self, disease_type):
        """Variants without E must re-declare beta as S→I."""
        model_class = self._get_registry()[disease_type]
        schema = model_class._build_parameter_schema()
        edges = [(e.source_id, e.target_id) for e in schema.transmission_edges]
        assert ("S", "I") in edges, f"{disease_type} should have a S→I beta edge"
        assert ("S", "E") not in edges, f"{disease_type} should not have a S→E edge (no E compartment)"

    @pytest.mark.parametrize("disease_type", [
        "COVID_SEIR", "COVID_SEIHR", "COVID_SEIDR", "COVID_SEIHDR",
    ])
    def test_e_variant_has_s_to_e_beta(self, disease_type):
        """Variants with E must have the S→E beta edge."""
        model_class = self._get_registry()[disease_type]
        schema = model_class._build_parameter_schema()
        edges = [(e.source_id, e.target_id) for e in schema.transmission_edges]
        assert ("S", "E") in edges, f"{disease_type} should have a S→E beta edge"


# ---------------------------------------------------------------------------
# COVID variant integration runs
# ---------------------------------------------------------------------------


class TestCovidVariantRuns:
    """Integration tests: each COVID variant runs end-to-end with expected output compartments."""

    @staticmethod
    def _build_config(disease_type: str, edges: list) -> dict:
        return {
            "Disease": {"disease_type": disease_type},
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0,
                 "population": 100000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 0},
            "TransmissionEdges": {"items": edges},
            "Interventions": {"items": []},
        }

    @staticmethod
    def _edge(src, tgt, vtype, value):
        return {"transmission_edge": {"source": src, "target": tgt, "value_type": vtype}, "value": value}

    _S, _E, _I, _H, _D, _R = "susceptible", "exposed", "infected", "hospitalized", "deceased", "recovered"

    @pytest.mark.integration
    def test_covid_sir(self):
        from compartment.models.covid_jax_model.variants import CovidSIRModel
        config = self._build_config("COVID_SIR", [
            self._edge(self._S, self._I, "RATE", 0.3),
            self._edge(self._I, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSIRModel, pathlib.Path(f.name))
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "I", "R"]:
            assert c in comps, f"SIR missing {c}"
        for c in ["E", "H", "D"]:
            assert c not in comps, f"SIR should not have {c}"

    @pytest.mark.integration
    def test_covid_seir(self):
        from compartment.models.covid_jax_model.variants import CovidSEIRModel
        config = self._build_config("COVID_SEIR", [
            self._edge(self._S, self._E, "RATE", 0.3),
            self._edge(self._E, self._I, "DAYS", 5.0),
            self._edge(self._I, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSEIRModel, pathlib.Path(f.name))
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "E", "I", "R"]:
            assert c in comps, f"SEIR missing {c}"
        for c in ["H", "D"]:
            assert c not in comps, f"SEIR should not have {c}"

    @pytest.mark.integration
    def test_covid_sihr(self):
        from compartment.models.covid_jax_model.variants import CovidSIHRModel
        config = self._build_config("COVID_SIHR", [
            self._edge(self._S, self._I, "RATE", 0.3),
            self._edge(self._I, self._H, "PERCENTAGE", 0.001),
            self._edge(self._I, self._R, "DAYS", 7.0),
            self._edge(self._H, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSIHRModel, pathlib.Path(f.name))
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "I", "H", "R"]:
            assert c in comps, f"SIHR missing {c}"
        for c in ["E", "D"]:
            assert c not in comps, f"SIHR should not have {c}"

    @pytest.mark.integration
    def test_covid_sidr(self):
        from compartment.models.covid_jax_model.variants import CovidSIDRModel
        config = self._build_config("COVID_SIDR", [
            self._edge(self._S, self._I, "RATE", 0.3),
            self._edge(self._I, self._D, "RATE", 0.0001),
            self._edge(self._I, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSIDRModel, pathlib.Path(f.name))
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "I", "D", "R"]:
            assert c in comps, f"SIDR missing {c}"
        for c in ["E", "H"]:
            assert c not in comps, f"SIDR should not have {c}"

    @pytest.mark.integration
    def test_covid_seihr(self):
        from compartment.models.covid_jax_model.variants import CovidSEIHRModel
        config = self._build_config("COVID_SEIHR", [
            self._edge(self._S, self._E, "RATE", 0.3),
            self._edge(self._E, self._I, "DAYS", 5.0),
            self._edge(self._I, self._H, "PERCENTAGE", 0.001),
            self._edge(self._I, self._R, "DAYS", 7.0),
            self._edge(self._H, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSEIHRModel, pathlib.Path(f.name))
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "E", "I", "H", "R"]:
            assert c in comps, f"SEIHR missing {c}"
        assert "D" not in comps, "SEIHR should not have D"

    @pytest.mark.integration
    def test_covid_seidr(self):
        from compartment.models.covid_jax_model.variants import CovidSEIDRModel
        config = self._build_config("COVID_SEIDR", [
            self._edge(self._S, self._E, "RATE", 0.3),
            self._edge(self._E, self._I, "DAYS", 5.0),
            self._edge(self._I, self._D, "RATE", 0.0001),
            self._edge(self._I, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSEIDRModel, pathlib.Path(f.name))
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "E", "I", "D", "R"]:
            assert c in comps, f"SEIDR missing {c}"
        assert "H" not in comps, "SEIDR should not have H"

    @pytest.mark.integration
    def test_covid_sihdr(self):
        from compartment.models.covid_jax_model.variants import CovidSIHDRModel
        config = self._build_config("COVID_SIHDR", [
            self._edge(self._S, self._I, "RATE", 0.3),
            self._edge(self._I, self._H, "PERCENTAGE", 0.001),
            self._edge(self._I, self._D, "RATE", 0.0001),
            self._edge(self._I, self._R, "DAYS", 7.0),
            self._edge(self._H, self._D, "RATE", 0.001),
            self._edge(self._H, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSIHDRModel, pathlib.Path(f.name))
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "I", "H", "D", "R"]:
            assert c in comps, f"SIHDR missing {c}"
        assert "E" not in comps, "SIHDR should not have E"

    @pytest.mark.integration
    def test_covid_seihdr(self):
        from compartment.models.covid_jax_model.model import CovidJaxModel
        results = _run_model(
            CovidJaxModel,
            MODELS_DIR / "covid_jax_model" / "example-config.json",
        )
        comps = {k for k in results[0]["parent_admin_total"]["time_series"][0] if k != "date"}
        for c in ["S", "E", "I", "H", "D", "R"]:
            assert c in comps, f"SEIHDR missing {c}"

    @pytest.mark.integration
    def test_population_conserved_sir(self):
        """SIR has no death compartment — total population must be conserved."""
        from compartment.models.covid_jax_model.variants import CovidSIRModel
        config = self._build_config("COVID_SIR", [
            self._edge(self._S, self._I, "RATE", 0.3),
            self._edge(self._I, self._R, "DAYS", 7.0),
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
        results = _run_model(CovidSIRModel, pathlib.Path(f.name))
        ts = results[0]["parent_admin_total"]["time_series"]
        compartments = [k for k in ts[0] if k != "date"]

        def _pop(entry):
            total = 0
            for c in compartments:
                val = entry[c]
                if isinstance(val, dict):
                    total += val.get("age_all", 0)
            return total

        pop_start = _pop(ts[0])
        pop_end = _pop(ts[-1])
        assert pop_start > 0
        assert abs(pop_start - pop_end) / pop_start < 0.001, (
            f"SIR population drift: start={pop_start:.1f}, end={pop_end:.1f}"
        )


# ---------------------------------------------------------------------------
# Uncertainty (LHS) runs
# ---------------------------------------------------------------------------


class TestCovidUncertainty:
    """Test COVID uncertainty (LHS) runs produce valid confidence intervals."""

    @pytest.mark.integration
    def test_uncertainty_varying_params(self):
        """Uncertainty run with variance on beta should produce CI bands."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-03-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 500000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{
                            "field_key": "value",
                            "has_variance": True,
                            "distribution_type": "UNIFORM",
                            "disease_param": "BETA",
                            "min": 0.15,
                            "max": 0.45,
                        }]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        assert len(results) == 2
        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            assert len(ts) > 0

            mid_point = ts[len(ts) // 2]
            for comp in ["S", "I", "R"]:
                assert comp in mid_point
                val = mid_point[comp]
                assert "mean" in val, f"Expected uncertainty format (mean/lower/upper) for {comp}"
                assert "lower" in val
                assert "upper" in val

            for entry in ts:
                for comp in ["S", "I", "R"]:
                    val = entry[comp]
                    for stat in ["mean", "lower", "upper"]:
                        assert not math.isnan(val[stat]), (
                            f"NaN in {comp}.{stat} on {entry['date']}"
                        )

    @pytest.mark.integration
    def test_uncertainty_no_nan(self):
        """Uncertainty run should produce no NaN values."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-02-01",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 100000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.25,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": True, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0.1, "max": 0.4}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict):
                        assert "mean" in val and "lower" in val and "upper" in val, (
                            f"Expected uncertainty format for {key}, got keys: {val.keys()}"
                        )
                        for stat in ["mean", "lower", "upper"]:
                            assert not math.isnan(val[stat]), (
                                f"NaN in {key}.{stat} on {entry['date']}"
                            )

    @pytest.mark.integration
    def test_uncertainty_lower_le_mean_le_upper(self):
        """In uncertainty output, lower <= mean <= upper should hold."""
        from compartment.models.covid_jax_model.model import CovidJaxModel

        config = {
            "Disease": {
                "disease_type": "COVID_SEIHDR",
                "immunity_period": 14,
            },
            "start_date": "2025-01-01",
            "end_date": "2025-02-15",
            "run_mode": "UNCERTAINTY",
            "admin_zones": [
                {"name": "Zone A", "center_lat": 47.0, "center_lon": 8.0, "population": 200000, "infected_population": 1.0},
            ],
            "demographics": {"age_0_17": 25, "age_18_55": 50, "age_56_plus": 25},
            "travel_volume": {"leaving": 20},
            "TransmissionEdges": {
                "items": [
                    {
                        "transmission_edge": {"source": "susceptible", "target": "infected", "value_type": "RATE"},
                        "value": 0.3,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": True, "distribution_type": "UNIFORM", "disease_param": "BETA", "min": 0.2, "max": 0.5}]},
                    },
                    {
                        "transmission_edge": {"source": "infected", "target": "recovered", "value_type": "DAYS"},
                        "value": 0.1,
                        "FieldConfigs": {"items": [{"field_key": "value", "has_variance": False, "distribution_type": "UNIFORM", "disease_param": "GAMMA", "min": 0, "max": 0}]},
                    },
                ]
            },
            "Interventions": {"items": []},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        results = _run_model(CovidJaxModel, pathlib.Path(config_path))

        for run in results:
            ts = run["parent_admin_total"]["time_series"]
            for entry in ts:
                for key, val in entry.items():
                    if key == "date":
                        continue
                    if isinstance(val, dict) and "mean" in val:
                        assert val["lower"] <= val["mean"] + 0.01, (
                            f"lower ({val['lower']}) > mean ({val['mean']}) "
                            f"for {key} on {entry['date']}"
                        )
                        assert val["mean"] <= val["upper"] + 0.01, (
                            f"mean ({val['mean']}) > upper ({val['upper']}) "
                            f"for {key} on {entry['date']}"
                        )
