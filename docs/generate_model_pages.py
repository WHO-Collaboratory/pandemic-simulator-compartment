"""
MkDocs hook: auto-generate Disease Model doc pages and nav from the registry.

Runs at build time.  For each model class discovered by the registry, this
hook:

1. Writes a ``docs/models/<slug>.md`` file with a title, description, and
   a ``mkdocstrings`` directive pointing at the model class.
2. Injects a *Disease Models* nav section into ``mkdocs.yml``'s nav list,
   replacing any existing one.

Modelers never need to touch ``mkdocs.yml`` or create markdown files —
adding a new model directory with ``DISEASE_TYPE`` is enough.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("mkdocs.hooks.generate_model_pages")

# Mapping from DISEASE_TYPE to a short URL slug for the doc page.
# Models not listed here derive a slug automatically from the disease_type.
_SLUG_OVERRIDES: dict[str, str] = {
    "COVID_SEIHDR": "covid",
    "VECTOR_BORNE": "dengue",
    "VECTOR_BORNE_2STRAIN": "dengue-2strain",
    "HANTAVIRUS_HUMAN_TRANSMISSION": "hantavirus-human",
}


def _slug_for(disease_type: str) -> str:
    """Derive a URL-friendly filename slug from a DISEASE_TYPE."""
    if disease_type in _SLUG_OVERRIDES:
        return _SLUG_OVERRIDES[disease_type]
    return disease_type.lower().replace("_", "-")


def _nav_label(disease_type: str, disease_label: str, is_test: bool) -> str:
    """Build the human-readable nav label for a model."""
    if is_test:
        return f"Test: {disease_label}"
    return disease_label


def on_config(config):
    """Hook entry point — called by MkDocs before the build starts."""
    # Ensure the repo root is importable so we can reach the registry.
    repo_root = str(Path(config["config_file_path"]).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from compartment.registry import MODEL_REGISTRY

    docs_dir = Path(config["docs_dir"])
    models_dir = docs_dir / "models"
    models_dir.mkdir(exist_ok=True)

    nav_entries: list[dict[str, str]] = []

    # Sort: non-test models first (alphabetically), then test models.
    def _sort_key(item):
        dtype, cls = item
        is_test = cls.__module__.split(".")[-2].startswith("test_")
        return (is_test, _nav_label(dtype, "", False), dtype)

    for disease_type, model_cls in sorted(MODEL_REGISTRY.items(), key=_sort_key):
        schema = model_cls._cached_schema
        if schema is None:
            logger.warning(
                "Skipping %s — no cached schema (non-migrated model?)", disease_type
            )
            continue

        module_parts = model_cls.__module__.split(".")
        model_dir_name = module_parts[-2]  # e.g. "covid_jax_model"
        is_test = model_dir_name.startswith("test_")

        label = schema.disease_label or disease_type
        description = schema.description or ""
        class_path = f"{model_cls.__module__}.{model_cls.__name__}"
        slug = _slug_for(disease_type)
        md_path = models_dir / f"{slug}.md"

        page_content = f"# {label}\n\n"
        if description:
            page_content += f"{description}\n\n"
        page_content += (
            f"::: {class_path}\n"
            f"    options:\n"
            f"      show_root_heading: true\n"
            f"      members_order: source\n"
            f"      show_source: true\n"
        )

        md_path.write_text(page_content)
        logger.info("Generated %s", md_path.relative_to(docs_dir))

        nav_label = _nav_label(disease_type, label, is_test)
        nav_entries.append({nav_label: f"models/{slug}.md"})

    # Remove stale model pages that the hook didn't generate this run.
    _cleanup_stale(models_dir, MODEL_REGISTRY)

    # Replace the "Disease Models" section in the nav.
    if nav_entries:
        nav = config.get("nav") or []
        new_nav = []
        for item in nav:
            if isinstance(item, dict) and "Disease Models" in item:
                continue  # drop the old static section
            new_nav.append(item)
        new_nav.append({"Disease Models": nav_entries})
        config["nav"] = new_nav

    return config


def _cleanup_stale(models_dir: Path, registry: dict) -> None:
    """Remove .md files in models_dir that weren't generated from the registry."""
    generated_files = {f"{_slug_for(dt)}.md" for dt in registry}
    for old_file in models_dir.glob("*.md"):
        if old_file.name not in generated_files:
            old_file.unlink()
            logger.info("Removed stale %s", old_file.name)


def on_post_build(config):
    """Final cleanup of stale model pages after build completes."""
    import sys

    repo_root = str(Path(config["config_file_path"]).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from compartment.registry import MODEL_REGISTRY

    models_dir = Path(config["docs_dir"]) / "models"
    _cleanup_stale(models_dir, MODEL_REGISTRY)
