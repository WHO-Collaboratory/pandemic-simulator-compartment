"""
CLI tool for generating model artifact JSON and example config files.

Usage:
    # Generate artifact JSON (stdout)
    python -m compartment.generate_artifact MPOX

    # Write artifact to a file
    python -m compartment.generate_artifact MPOX --output artifact.json

    # Generate an example simulation config instead
    python -m compartment.generate_artifact MPOX --example-config

    # Generate both artifact and example config to files
    python -m compartment.generate_artifact MPOX --output artifact.json --example-config --config-output example.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

from compartment.datasets import artifact_refs_for_model_dir, model_dir_for_class


# ---------------------------------------------------------------------------
# Dataset references
# ---------------------------------------------------------------------------


def _attach_datasets(artifact: dict, model_dir: Path | None) -> dict:
    """Merge the model's datasets.yaml references into its artifact JSON.

    Emits a top-level ``datasets`` list of ``{name, version, filename}`` so
    downstream consumers can resolve which data the model was built against.
    Models without a datasets.yaml are left untouched — no empty key.
    """
    if model_dir is None:
        return artifact
    refs = artifact_refs_for_model_dir(model_dir)
    if refs:
        artifact["datasets"] = refs
    return artifact


# ---------------------------------------------------------------------------
# Model registry (mirrors the one in validation but avoids circular imports)
# ---------------------------------------------------------------------------


def _discover_models_from_dir(model_dir: str) -> list:
    """Discover all Model subclasses from a model directory.

    Scans model.py and variants.py (if present). Returns one class per
    artifact to generate — base model plus any fixed variants.

    Accepts paths like:
      compartment/models/mpox_jax_model
      compartment/models/mpox_jax_model/
    """
    from compartment.model import Model

    folder_name = Path(model_dir.rstrip("/")).name
    all_classes = []

    for module_suffix in ("model", "variants"):
        module_name = f"compartment.models.{folder_name}.{module_suffix}"
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue  # variants.py is optional

        candidates = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, Model) and obj is not Model and obj.__module__ == module_name
        ]
        all_classes.extend(candidates)

    if not all_classes:
        print(f"Error: No Model subclass found in '{model_dir}'", file=sys.stderr)
        sys.exit(1)

    return all_classes


def _get_model_class(disease_type: str):
    """Return the model class for a disease type via the central registry."""
    from compartment.registry import MODEL_REGISTRY, resolve
    model_class = resolve(disease_type)
    if model_class is None:
        available = sorted(MODEL_REGISTRY.keys())
        print(f"Error: Unknown disease type '{disease_type}'", file=sys.stderr)
        print(f"Available types: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)
    return model_class


def _list_available() -> list[str]:
    """Return disease types that have implemented define_parameters()."""
    from compartment.registry import DISEASE_TYPE_REGISTRY, MODEL_REGISTRY
    from compartment.schema_generator import has_parameter_schema

    available = []
    for model_key, model_class in MODEL_REGISTRY.items():
        try:
            if has_parameter_schema(model_class):
                disease_type = model_class.DISEASE_TYPE
                available.append(
                    disease_type
                    if DISEASE_TYPE_REGISTRY.get(disease_type) is model_class
                    else model_key
                )
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
        help=(
            "Disease type identifier (e.g. MPOX), or the generated model_key "
            "when multiple models share a disease type. Omit to list available types."
        ),
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
        "--output-dir",
        help=(
            "Write one artifact per discovered model class into this directory, "
            "named {disease_type}.json. Use with --model-dir to generate all variants."
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

    # --model-dir + --output-dir: generate one artifact per discovered class
    if args.model_dir and args.output_dir:
        model_classes = _discover_models_from_dir(args.model_dir)
        disease_type_counts: dict[str, int] = {}
        for model_class in model_classes:
            disease_type = model_class.DISEASE_TYPE
            disease_type_counts[disease_type] = (
                disease_type_counts.get(disease_type, 0) + 1
            )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Every variant in a model dir shares that dir's datasets.yaml.
        source_dir = Path(args.model_dir.rstrip("/"))
        for model_class in model_classes:
            try:
                schema = model_class._build_parameter_schema()
            except NotImplementedError:
                print(f"Skipping {model_class.__name__}: define_parameters() not implemented.", file=sys.stderr)
                continue
            artifact_name = (
                schema.model_key
                if disease_type_counts[schema.disease_type] > 1
                else schema.disease_type
            )
            output_path = output_dir / f"{artifact_name}.json"
            artifact = _attach_datasets(schema.to_artifact_dict(), source_dir)
            with open(output_path, "w") as f:
                f.write(json.dumps(artifact, indent=2) + "\n")
            print(f"Artifact written to {output_path}", file=sys.stderr)
        return

    # Single-class mode: --model-dir (first class) or explicit disease_type
    if args.model_dir:
        model_class = _discover_models_from_dir(args.model_dir)[0]
        model_dir = Path(args.model_dir.rstrip("/"))
    else:
        model_class = _get_model_class(args.disease_type)
        model_dir = model_dir_for_class(model_class)

    try:
        schema = model_class._build_parameter_schema()
    except NotImplementedError:
        print(
            f"Error: {model_class.__name__} has not implemented define_parameters().",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Artifact JSON ---
    artifact = _attach_datasets(schema.to_artifact_dict(), model_dir)
    artifact_json = json.dumps(artifact, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(artifact_json + "\n")
        print(f"Artifact written to {args.output}", file=sys.stderr)
    elif not args.example_config:
        print(artifact_json)

    # --- Example config ---
    if args.example_config:
        example = schema.to_example_config()
        example_json = json.dumps(example, indent=4)

        if args.config_output:
            with open(args.config_output, "w") as f:
                f.write(example_json + "\n")
            print(f"Example config written to {args.config_output}", file=sys.stderr)
        else:
            print(example_json)


if __name__ == "__main__":
    main()
