"""
CLI tool for scaffolding a new disease model directory.

Usage:
    python -m compartment.new_model my_disease
    python -m compartment.new_model my_disease --label "My Disease" \\
        --disease-type MY_DISEASE --description "An SIR model for my disease"

Creates compartment/models/my_disease_model/ with:
    __init__.py, model.py, main.py, example-config.json

The generated model is a minimal SIR with frequency-dependent transmission.
Edit model.py to add compartments, parameters, and ODE logic specific to
your disease; see docs/DEVELOPING_MODELS.md for the full authoring guide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


MODELS_DIR = Path(__file__).parent / "models"


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------


def _normalize_base_name(raw: str) -> str:
    """Return the base snake_case name (strips a trailing _model/_jax_model suffix).

    Both the current ``_model`` suffix and the legacy ``_jax_model`` suffix are
    stripped so a name copied from either convention resolves to the same base.
    """
    name = raw.strip().lower()
    for suffix in ("_jax_model", "_jax", "_model"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _validate_name(name: str) -> None:
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        print(
            f"Error: '{name}' is not a valid model name.\n"
            "Use lowercase letters, digits, and underscores only, starting with a letter.\n"
            "Example:  python -m compartment.new_model my_disease",
            file=sys.stderr,
        )
        sys.exit(1)


def _to_pascal_case(snake: str) -> str:
    return "".join(word.capitalize() for word in snake.split("_"))


# ---------------------------------------------------------------------------
# File templates
#
# Placeholders (replaced by scaffold()):
#   CLASS_NAME    – e.g. MyDiseaseModel
#   DISEASE_TYPE  – e.g. MY_DISEASE
#   LABEL         – e.g. My Disease
#   DESCRIPTION   – e.g. A simple SIR model for My Disease
#   DIR_NAME      – e.g. my_disease_model
# ---------------------------------------------------------------------------

_MODEL_PY = '''\
import jax.numpy as jnp
import numpy as np
import logging
from compartment.model import Model, ValueType

logger = logging.getLogger(__name__)


class CLASS_NAME(Model):
    """A simple SIR compartmental model for LABEL."""

    @classmethod
    def define_parameters(cls, schema):
        schema.set_model_info(
            disease_type="DISEASE_TYPE",
            label="LABEL",
            description="DESCRIPTION",
        )

        # --- Compartments ---
        # Mark infective=True on compartments that contribute to force of infection.
        schema.add_compartment("S", "Susceptible", "Population susceptible to infection")
        schema.add_compartment("I", "Infected", "Currently infectious population", infective=True)
        schema.add_compartment("R", "Recovered", "Recovered and immune")

        # --- Transmission edges ---
        # The framework auto-generates I_total, R_total cumulative compartments
        # for the targets of these edges — do not declare them by hand.
        schema.add_transmission_edge(
            source="susceptible",
            target="infected",
            variable_name="beta",
            frequency_dependent=True,
            label="Transmission Rate (S->I)",
            description="Rate at which susceptibles become infected through contact",
            default=0.3,
            default_min=0.1,
            default_max=0.5,
            min_value=0.01,
            max_value=2.0,
            unit="per day",
        )
        schema.add_transmission_edge(
            source="infected",
            target="recovered",
            variable_name="gamma",
            label="Recovery Period (I->R)",
            description="Average number of days to recover",
            default=10.0,
            default_min=5.0,
            default_max=20.0,
            min_value=1.0,
            max_value=100.0,
            value_type=ValueType.DAYS,
            unit="days",
        )

        # --- Optional: spatial travel support ---
        # Declare your mobility parameters as custom fields, then define how
        # they build the matrix in build_travel_matrix() below. Without this,
        # the base class supplies an identity matrix (no inter-zone travel).
        # schema.add_parameter(
        #     name="travel_sigma",
        #     label="Travel Rate (σ)",
        #     description="Percentage of each zone's population away from home per day.",
        #     value_type=ValueType.PERCENTAGE,
        #     default=20.0,
        #     min_value=0.0,
        #     max_value=100.0,
        #     unit="%",
        # )

        # --- Optional: interventions ---
        # schema.add_intervention(
        #     id="my_intervention",
        #     label="My Intervention",
        #     description="Reduces transmission while active",
        #     target_rates=["beta"],
        #     adherence=50.0,
        #     transmission_reduction=50.0,
        # )

        # --- Optional: age-stratified demographics + contact matrix ---
        # schema.add_demographic_group("age_0_17",  "Children", default_weight=33.3, age_range=(0, 17))
        # schema.add_demographic_group("age_18_55", "Adults",   default_weight=44.4, age_range=(18, 55))
        # schema.add_demographic_group("age_56_plus","Elderly", default_weight=22.3, age_range=(56, 120))

    def __init__(self, config):
        super().__init__(config)
        # Add any model-specific initialisation here (e.g. temperature).

    # --- Optional: your own data ---
    # To use a data file, declare it in a datasets.yaml next to this model and
    # read it with self.dataset(name). The same call works locally and in the
    # cloud — never build the path by hand.
    #
    #   # datasets.yaml
    #   datasets:
    #     - name: my-contact-matrix
    #       version: "1"
    #       file: data/contacts.csv
    #
    #   import pandas as pd
    #   contacts = pd.read_csv(self.dataset("my-contact-matrix"))
    #
    # Upload it once with `python -m compartment.datasets push`, and see
    # docs/guides/adding-datasets.md. Limit: 500 MB per dataset.

    # --- Optional: spatial travel support ---
    # The framework calls this before prepare_initial_state() and stores the
    # result on self.travel_matrix. The default returns the identity matrix,
    # so only override it if your model has inter-zone mobility.
    #
    # def build_travel_matrix(self, admin_zones):
    #     # PERCENTAGE params arrive as 20.0, not 0.2 — convert first.
    #     sigma = self._to_rate(self.travel_sigma, ValueType.PERCENTAGE)
    #     return get_gravity_model_travel_matrix(admin_zones, sigma)

    def prepare_initial_state(self):
        return self.population_matrix

    def equation(self, y, t, p):
        C = self.COMPARTMENTS
        params = self._unpack_params(p)

        states = {c: y[i] for i, c in enumerate(self.compartment_list)}

        I = states[C.I]  # noqa: E741
        non_total = [c for c in C if not c.endswith("_total")]
        N_total = sum(states[c] for c in non_total)
        prop_infective = I.sum() / (N_total.sum() + 1e-10)

        # _apply_interventions scales target_rates and returns the updated travel
        # matrix. It is a no-op when no interventions are configured.
        rates, self.travel_matrix = self._apply_interventions(
            t, {"beta": params["beta"]}, prop_infective
        )
        rates["gamma"] = params["gamma"]

        # _compute_equations handles mass-action / frequency-dependent FOI,
        # _total accumulation, and skips compartments not active in this variant.
        derivs = self._compute_equations(states, rates)
        return jnp.stack([derivs[c] for c in self.compartment_list])
'''

_MAIN_PY = '''\
import logging
import argparse
from compartment.driver import drive_simulation
from compartment.models.DIR_NAME.model import CLASS_NAME

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def lambda_handler(event, context):
    drive_simulation(
        model_class=CLASS_NAME,
        args={"mode": "cloud", "simulation_job_id": event["simulation_job_id"]},
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LABEL simulation with specified config file"
    )
    parser.add_argument("--mode", choices=["local", "cloud"], default="local")
    parser.add_argument("--config_file", help="Path to the JSON config file for the simulation")
    parser.add_argument(
        "--output_file",
        nargs="?",
        default=None,
        help="Output file path. Use * to generate a default filename with timestamp.",
    )
    parser.add_argument(
        "--simulation_job_id",
        nargs="?",
        default=None,
        help="Existing simulation job id in a graphql backend.",
    )
    args = parser.parse_args()
    drive_simulation(model_class=CLASS_NAME, args=vars(args))
'''

# The TransmissionEdges.items top-level key is the correct wiring format.
# The generator-emitted Disease.transmission_edges format silently drops params.
_EXAMPLE_CONFIG = {
    "Disease": {
        "disease_type": "DISEASE_TYPE"
    },
    "start_date": "2026-01-01",
    "end_date": "2026-12-31",
    "TransmissionEdges": {
        "items": [
            {
                "transmission_edge": {
                    "source": "susceptible",
                    "target": "infected",
                    "value_type": "RATE"
                },
                "value": 0.3,
                "FieldConfigs": {
                    "items": [
                        {
                            "field_key": "value",
                            "has_variance": False,
                            "distribution_type": "UNIFORM",
                            "disease_param": "BETA",
                            "min": 0,
                            "max": 0
                        }
                    ]
                }
            },
            {
                "transmission_edge": {
                    "source": "infected",
                    "target": "recovered",
                    "value_type": "DAYS"
                },
                "value": 10.0,
                "FieldConfigs": {
                    "items": [
                        {
                            "field_key": "value",
                            "has_variance": False,
                            "distribution_type": "UNIFORM",
                            "disease_param": "GAMMA",
                            "min": 0,
                            "max": 0
                        }
                    ]
                }
            }
        ]
    },
    "admin_zones": [
        {
            "name": "Example Zone",
            "center_lat": 40.7128,
            "center_lon": -74.0060,
            "population": 1000000,
            "infected_population": 0.01
        }
    ]
}


# ---------------------------------------------------------------------------
# Scaffold logic
# ---------------------------------------------------------------------------


def _fill_template(
    template: str,
    class_name: str,
    disease_type: str,
    label: str,
    dir_name: str,
    description: str,
) -> str:
    # DESCRIPTION is substituted last so that user-supplied prose is never
    # re-scanned for the other placeholders.
    return (
        template
        .replace("CLASS_NAME", class_name)
        .replace("DISEASE_TYPE", disease_type)
        .replace("LABEL", label)
        .replace("DIR_NAME", dir_name)
        .replace("DESCRIPTION", description)
    )


def scaffold(
    base_name: str,
    *,
    label: str | None = None,
    disease_type: str | None = None,
    description: str | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Create a new model directory under compartment/models/.

    Args:
        base_name:    Snake-case model name without the _model suffix
                      (e.g. ``"my_disease"``).  The suffix is appended automatically.
        label:        Human-readable display name (default: title-cased base_name).
        disease_type: ALL_CAPS identifier (default: uppercased base_name).
        description:  Model description shown in set_model_info() / the UI
                      (default: "A simple SIR model for <label>").
        dry_run:      Print what would be created without writing any files.

    Returns:
        Path to the new model directory.
    """
    _validate_name(base_name)

    dir_name = f"{base_name}_model"
    class_name = f"{_to_pascal_case(base_name)}Model"
    disease_type = disease_type or base_name.upper()
    label = label or base_name.replace("_", " ").title()
    description = description or f"A simple SIR model for {label}"
    # Guard against quotes/backslashes breaking the generated string literal.
    safe_description = description.replace("\\", "\\\\").replace('"', '\\"')

    dest = MODELS_DIR / dir_name

    if dest.exists() and not dry_run:
        print(f"Error: '{dest}' already exists.", file=sys.stderr)
        sys.exit(1)

    files = {
        "__init__.py": "",
        "model.py": _fill_template(_MODEL_PY, class_name, disease_type, label, dir_name, safe_description),
        "main.py": _fill_template(_MAIN_PY, class_name, disease_type, label, dir_name, safe_description),
        "example-config.json": json.dumps(
            _patch_config(disease_type), indent=4
        ) + "\n",
    }

    if dry_run:
        print(f"Would create: {dest}/")
        for name in files:
            print(f"  {name}")
        return dest

    dest.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (dest / name).write_text(content)

    return dest


def _patch_config(disease_type: str) -> dict:
    """Return a copy of _EXAMPLE_CONFIG with the correct disease_type."""
    import copy
    cfg = copy.deepcopy(_EXAMPLE_CONFIG)
    cfg["Disease"]["disease_type"] = disease_type
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scaffold a new disease model directory under compartment/models/. "
            "Creates __init__.py, model.py, main.py, and example-config.json "
            "from a minimal SIR template."
        )
    )
    parser.add_argument(
        "name",
        help=(
            "Snake-case model name, e.g. 'my_disease'. "
            "The _model suffix is appended automatically to the directory name "
            "(e.g. 'my_disease' -> compartment/models/my_disease_model/)."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable display name (default: title-cased name, e.g. 'My Disease').",
    )
    parser.add_argument(
        "--disease-type",
        default=None,
        dest="disease_type",
        help="ALL_CAPS disease type identifier used in configs (default: uppercased name).",
    )
    parser.add_argument(
        "--description",
        default=None,
        help=(
            "Model description shown in set_model_info() and the UI "
            "(default: 'A simple SIR model for <label>')."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing any files.",
    )
    args = parser.parse_args()

    base_name = _normalize_base_name(args.name)
    dest = scaffold(
        base_name,
        label=args.label,
        disease_type=args.disease_type,
        description=args.description,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return

    dir_name = dest.name
    class_name = f"{_to_pascal_case(base_name)}Model"
    disease_type = (args.disease_type or base_name.upper())

    print(f"Created: {dest}/")
    print()
    print("Next steps:")
    print(f"  1. Edit {dest}/model.py")
    print(f"       — add/remove compartments, transmission edges, interventions")
    print(f"       — flesh out equation() with your disease's ODE")
    print(f"  2. Regenerate example-config.json from the schema (optional):")
    print(f"       python -m compartment.generate_artifact {disease_type} \\")
    print(f"           --example-config \\")
    print(f"           --config-output {dest}/example-config.json")
    print(f"  3. Run the model locally:")
    print(f"       python -m compartment.models.{dir_name}.main \\")
    print(f"           --mode local \\")
    print(f"           --config_file {dest}/example-config.json \\")
    print(f"           --output_file results/{dir_name}-test.json")
    print(f"  4. Run the smoke test:")
    print(f"       python -m pytest tests/test_smoke.py -v -m integration -k '{dir_name}'")
    print()
    print("See docs/DEVELOPING_MODELS.md for the full authoring guide.")


if __name__ == "__main__":
    main()
