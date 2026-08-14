"""datasets.yaml — the manifest that tags a model's datasets.

A ``datasets.yaml`` sits alongside a model (``compartment/models/<name>/``) and
is the single source of truth for which datasets that model depends on, at
which version:

.. code-block:: yaml

    datasets:
      - name: kenya-contact-matrix
        version: "1.0.0"
        file: data/kenya-contact-matrix.csv

      - name: kenya-admin-zones
        version: "2"
        file: data/kenya-admin-zones.json

``file`` is resolved relative to the directory holding the ``datasets.yaml``.
The uploaded object's filename is its basename, so the manifest entry above
lands at ``datasets/kenya-contact-matrix/1.0.0/kenya-contact-matrix.csv`` in the
confirmed-safe bucket.

Datasets are immutable — pushing a ``name``/``version`` pair that already exists
is rejected. Bump ``version`` to publish new data.

The same three fields (``name``, ``version``, ``filename``) are copied into the
generated model artifact JSON under a top-level ``datasets`` key, so consumers
of the artifact can resolve exactly which data a model was built against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

MANIFEST_FILENAME = "datasets.yaml"

# Mirrors the validation the dataset API applies — these become S3 key path
# segments, so no slashes and no traversal.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")

# Largest single dataset we accept, mirroring MAX_UPLOAD_SIZE_BYTES on the
# dataset API and the scan-complete Lambda. The binding constraint is not the
# malware scanner (GuardDuty handles 5 GiB) but the model container image:
# every dataset a model declares is baked into its image by the `datasets
# stage` step, and Lambda caps an image at 10 GB.
#
# Checked client-side too, so an oversized file is caught before it is uploaded
# rather than after the scan rejects it. Keep this exactly in step with
# dataset_max_upload_size_bytes in infra/tofu/shared-services — a client limit
# looser than the server's produces a confusing late rejection, and a tighter
# one silently blocks uploads the platform would accept.
MAX_DATASET_BYTES = 500 * 1024 * 1024  # 524288000


def human_bytes(size: int) -> str:
    """Format a byte count for a modeler-facing message."""
    megabytes = size / (1024 * 1024)
    if megabytes >= 10:
        return f"{megabytes:,.0f} MB"
    return f"{megabytes:,.1f} MB"


class ManifestError(Exception):
    """Raised when a datasets.yaml is missing, malformed, or invalid."""


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    version: str
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name

    def artifact_ref(self) -> dict:
        """The shape embedded in the model artifact JSON."""
        return {"name": self.name, "version": self.version, "filename": self.filename}


def load_manifest(manifest_path: Path) -> list[DatasetEntry]:
    """Parse and validate a datasets.yaml.

    Raises ManifestError with an actionable message rather than letting a
    KeyError or TypeError surface — this runs behind a modeler-facing CLI.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ManifestError(f"No manifest at {manifest_path}.")

    try:
        document = yaml.safe_load(manifest_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"{manifest_path} is not valid YAML: {exc}") from exc

    if not isinstance(document, dict) or "datasets" not in document:
        raise ManifestError(f"{manifest_path} must contain a top-level `datasets:` list.")

    raw_entries = document["datasets"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ManifestError(f"{manifest_path}: `datasets` must be a non-empty list.")

    base_dir = manifest_path.parent
    entries = [_parse_entry(raw, index, base_dir, manifest_path)
               for index, raw in enumerate(raw_entries, start=1)]

    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.name, entry.version)
        if key in seen:
            raise ManifestError(
                f"{manifest_path}: {entry.name} version {entry.version} is listed twice."
            )
        seen.add(key)

    return entries


def find_manifest(model_dir: Path) -> Path | None:
    """Return the datasets.yaml for a model directory, or None if it has none.

    A model without datasets is the common case, so absence is not an error.
    """
    candidate = Path(model_dir) / MANIFEST_FILENAME
    return candidate if candidate.is_file() else None


def artifact_refs_for_model_dir(model_dir: Path) -> list[dict]:
    """Dataset references to embed in a model's artifact JSON.

    Returns an empty list when the model has no datasets.yaml.
    """
    manifest_path = find_manifest(model_dir)
    if manifest_path is None:
        return []
    return [entry.artifact_ref() for entry in load_manifest(manifest_path)]


def _parse_entry(raw, index: int, base_dir: Path, manifest_path: Path) -> DatasetEntry:
    where = f"{manifest_path}: datasets[{index}]"

    if not isinstance(raw, dict):
        raise ManifestError(f"{where} must be a mapping with name, version, and file.")

    name = _require_string(raw, "name", _NAME_RE, where)
    # Quoting matters in YAML: an unquoted 1.0 parses as a float and an
    # unquoted 2 as an int, so accept both and stringify rather than making
    # the modeler debug a type error.
    version = _require_string(raw, "version", _NAME_RE, where, coerce=True)

    file_value = raw.get("file")
    if not isinstance(file_value, str) or not file_value.strip():
        raise ManifestError(f"{where} is missing a `file:` path.")

    path = (base_dir / file_value).resolve()
    if not _FILENAME_RE.match(path.name):
        raise ManifestError(
            f"{where}: filename {path.name!r} must match {_FILENAME_RE.pattern}."
        )

    return DatasetEntry(name=name, version=version, path=path)


def _require_string(raw: dict, field: str, pattern: re.Pattern, where: str,
                    coerce: bool = False) -> str:
    value = raw.get(field)
    if coerce and isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str) or not pattern.match(value):
        raise ManifestError(
            f"{where}: `{field}` is required and must match {pattern.pattern}."
        )
    return value
