"""Stage a model's declared datasets into its source tree, for image builds.

The release pipeline runs this immediately before ``docker build``, so every
file listed in a model's ``datasets.yaml`` is baked into the model image at the
path :func:`compartment.datasets.resolve.dataset_path` will resolve at runtime.

This talks to S3 with **boto3**, not the dataset API Function URL, because it
runs in CI: the API authenticates with a pasted Cognito access token, while CI
holds an AWS role. Same objects, different door.
"""

from __future__ import annotations

import os
from pathlib import Path

from compartment.datasets.manifest import (
    DatasetEntry,
    ManifestError,
    find_manifest,
    human_bytes,
    load_manifest,
)

DEFAULT_BUCKET = "collaboratory-datasets"
# Mirrors local.dataset_object_prefix in infra/tofu/shared-services/datasets.tf.
DATASET_PREFIX = "datasets/"

# Total staged bytes we allow into one image. The per-dataset cap is
# MAX_DATASET_BYTES (500 MB), but a model declaring ten of them still bloats
# its image, and Lambda caps an image at 10 GB with ~1 GB already spent on the
# jax/python base. Fail the build rather than discover it at deploy time.
DEFAULT_MAX_STAGED_BYTES = 2 * 1024 * 1024 * 1024


def object_key(entry: DatasetEntry) -> str:
    """S3 key for a manifest entry in the confirmed-safe bucket."""
    return f"{DATASET_PREFIX}{entry.name}/{entry.version}/{entry.filename}"


def stage_model_dir(model_dir: Path, bucket: str, max_total_bytes: int) -> list[str]:
    """Download every dataset the model declares to its manifest ``file:`` path.

    Returns human-readable lines describing what happened, for the build log.
    Raises ManifestError on anything that would produce an image with missing
    or oversized data.
    """
    manifest_path = find_manifest(model_dir)
    if manifest_path is None:
        # The common case — most models ship no datasets. Not an error, or the
        # CI step would fail for every model in the repo.
        return [f"{model_dir}: no datasets.yaml, nothing to stage."]

    entries = load_manifest(manifest_path)

    # Imported lazily so that importing this module (and therefore the CLI)
    # does not require boto3 to be installed or credentials to be present.
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")

    lines: list[str] = []
    total = 0
    for entry in entries:
        key = object_key(entry)
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            raise ManifestError(
                f"{entry.name}@{entry.version}: s3://{bucket}/{key} is not "
                f"available ({exc.response.get('Error', {}).get('Code', 'error')}). "
                f"Push it with `python -m compartment.datasets push` and confirm "
                f"`check-status` reports PROMOTED before tagging a release."
            ) from exc

        size = head["ContentLength"]
        total += size
        if total > max_total_bytes:
            raise ManifestError(
                f"{model_dir}: staged datasets total {human_bytes(total)}, over "
                f"the {human_bytes(max_total_bytes)} limit for one model image. "
                f"Drop a dataset, shrink it, or raise PANSIM_MAX_STAGED_BYTES if "
                f"the image can afford it."
            )

        entry.path.parent.mkdir(parents=True, exist_ok=True)
        if entry.path.is_file() and entry.path.stat().st_size == size:
            lines.append(f"{entry.name}@{entry.version}  {human_bytes(size)}  (already present)")
            continue

        s3.download_file(bucket, key, str(entry.path))
        lines.append(f"{entry.name}@{entry.version}  {human_bytes(size)}  -> {entry.path}")

    lines.append(f"{len(entries)} dataset(s) staged, {human_bytes(total)} total.")
    return lines


def resolve_bucket() -> str:
    return os.environ.get("PANSIM_DATASETS_BUCKET") or DEFAULT_BUCKET


def resolve_max_total_bytes() -> int:
    raw = os.environ.get("PANSIM_MAX_STAGED_BYTES")
    if not raw:
        return DEFAULT_MAX_STAGED_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise ManifestError(
            f"PANSIM_MAX_STAGED_BYTES must be an integer number of bytes, got {raw!r}."
        ) from None
    if value <= 0:
        raise ManifestError("PANSIM_MAX_STAGED_BYTES must be positive.")
    return value
