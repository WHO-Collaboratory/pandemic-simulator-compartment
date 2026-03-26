"""
CLI tool for generating model artifact JSON and example config files.

Usage:
    # Generate artifact JSON (stdout)
    python -m compartment.generate_artifact MONKEYPOX

    # Write artifact to a file
    python -m compartment.generate_artifact MONKEYPOX --output artifact.json

    # Generate an example simulation config instead
    python -m compartment.generate_artifact MONKEYPOX --example-config

    # Generate both artifact and example config to files
    python -m compartment.generate_artifact MONKEYPOX --output artifact.json --example-config --config-output example.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Model registry (mirrors the one in validation but avoids circular imports)
# ---------------------------------------------------------------------------


def _discover_model_from_dir(model_dir: str):
    """Dynamically load a Model subclass from a model directory path.

    Accepts paths like:
      compartment/models/mpox_jax_model
      compartment/models/mpox_jax_model/

    Returns the Model subclass found in that directory's model.py.
    """
    from compartment.model import Model

    folder_name = Path(model_dir.rstrip("/")).name
    module_name = f"compartment.models.{folder_name}.model"

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error: Could not import '{module_name}': {e}", file=sys.stderr)
        sys.exit(1)

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Model) and obj is not Model and obj.__module__ == module_name
    ]

    if not candidates:
        print(f"Error: No Model subclass found in '{module_name}'", file=sys.stderr)
        sys.exit(1)

    return candidates[0]


def _get_model_class(disease_type: str):
    """Lazy-import model class by disease type."""
    registry = {}

    # Only import what's needed to keep startup fast
    if disease_type == "MONKEYPOX":
        from compartment.models.mpox_jax_model.model import MpoxJaxModel

        registry["MONKEYPOX"] = MpoxJaxModel
    elif disease_type == "RESPIRATORY":
        from compartment.models.covid_jax_model.model import CovidJaxModel

        registry["RESPIRATORY"] = CovidJaxModel
    elif disease_type == "VECTOR_BORNE":
        from compartment.models.dengue_jax_model.model import DengueJaxModel

        registry["VECTOR_BORNE"] = DengueJaxModel
    elif disease_type == "VECTOR_BORNE_2STRAIN":
        from compartment.models.dengue_2strain.model import Dengue2StrainModel

        registry["VECTOR_BORNE_2STRAIN"] = Dengue2StrainModel

    if disease_type not in registry:
        print(f"Error: Unknown disease type '{disease_type}'", file=sys.stderr)
        print(
            "Available types: MONKEYPOX, RESPIRATORY, VECTOR_BORNE, VECTOR_BORNE_2STRAIN",
            file=sys.stderr,
        )
        sys.exit(1)

    return registry[disease_type]


def _list_available() -> list[str]:
    """Return disease types that have implemented define_parameters()."""
    from compartment.schema_generator import has_parameter_schema

    types_to_check = [
        "MONKEYPOX",
        "RESPIRATORY",
        "VECTOR_BORNE",
        "VECTOR_BORNE_2STRAIN",
    ]
    available = []
    for dt in types_to_check:
        try:
            model_class = _get_model_class(dt)
            if has_parameter_schema(model_class):
                available.append(dt)
        except Exception:
            pass
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Generate model artifact JSON or example configs from declarative parameter definitions.",
    )
    parser.add_argument(
        "disease_type",
        nargs="?",
        help="Disease type identifier (e.g. MONKEYPOX). Omit to list available types.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write artifact JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--example-config",
        action="store_true",
        help="Generate an example simulation config instead of (or in addition to) the artifact.",
    )
    parser.add_argument(
        "--config-output",
        help="Write example config to this file (only with --example-config).",
    )
    parser.add_argument(
        "--model-dir",
        help=(
            "Model folder path (e.g. compartment/models/mpox_jax_model). "
            "Discovers the model class and disease_type dynamically via importlib, "
            "eliminating the need for a hardcoded mapping table."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_types",
        help="List disease types that support artifact generation.",
    )

    args = parser.parse_args()

    # --list mode
    if args.list_types or (args.disease_type is None and args.model_dir is None):
        available = _list_available()
        if available:
            print("Disease types with declarative parameter definitions:")
            for dt in available:
                print(f"  - {dt}")
        else:
            print("No disease types have implemented define_parameters() yet.")
        return

    # Resolve model class — prefer --model-dir over positional disease_type
    if args.model_dir:
        model_class = _discover_model_from_dir(args.model_dir)
    else:
        model_class = _get_model_class(args.disease_type)

    try:
        schema = model_class._build_parameter_schema()
    except NotImplementedError:
        print(
            f"Error: {
                model_class.__name__} has not implemented define_parameters().",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Artifact JSON ---
    artifact = schema.to_artifact_dict()
    artifact_json = json.dumps(artifact, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(artifact_json + "\n")
        print(f"Artifact written to {args.output}", file=sys.stderr)
    elif not args.example_config:
        # Print artifact to stdout (default behavior)
        print(artifact_json)

    # --- Example config ---
    if args.example_config:
        example = schema.to_example_config()
        example_json = json.dumps(example, indent=4)

        if args.config_output:
            with open(args.config_output, "w") as f:
                f.write(example_json + "\n")
            print(f"Example config written to {
                  args.config_output}", file=sys.stderr)
        else:
            print(example_json)


if __name__ == "__main__":
    main()
