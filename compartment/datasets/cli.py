"""Modeler dataset CLI.

    python -m compartment.datasets push [NAME ...]     upload datasets declared
                                                        in ./datasets.yaml
    python -m compartment.datasets check-status ID     report on one upload
    python -m compartment.datasets list                list available datasets
    python -m compartment.datasets pull NAME[@VERSION] download one dataset
    python -m compartment.datasets stage --model-dir D download a model's
                                                        declared datasets (CI)

``push`` returns as soon as the bytes are in the quarantine bucket. The malware
scan runs asynchronously, so nothing blocks on it — poll ``check-status`` with
the upload-id it prints.

``stage`` is the release pipeline's step, not a modeler command: it reads a
model's datasets.yaml and downloads each entry with AWS credentials so the
files land in the model's container image.

Environment (all optional — the defaults are what modelers should use):
    PANSIM_DATASET_API   Dataset API Function URL. Defaults to the UAT
                         deployment; override only to point at a rebuild.
    PANSIM_WEBAPP_URL    Web app to open for the session token
                         (default https://uat.pandemic-simulator.com)
    PANSIM_HOME          Token cache directory (default ~/.pansim)
    PANSIM_DATASETS_BUCKET   `stage` only — confirmed-safe dataset bucket
                         (default collaboratory-datasets)
    PANSIM_MAX_STAGED_BYTES  `stage` only — total bytes allowed into one
                         model image (default 2 GB)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compartment.datasets.api import ApiError, DatasetApi
from compartment.datasets.auth import AuthError, get_token
from compartment.datasets.manifest import (
    MANIFEST_FILENAME,
    MAX_DATASET_BYTES,
    ManifestError,
    human_bytes,
    load_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # `stage` runs in CI against AWS credentials, so it must not touch the
        # dataset API or the interactive token prompt — that would block a
        # build waiting on a paste that never comes.
        if not getattr(args, "needs_auth", True):
            return args.func(args)

        api = DatasetApi()
        token = get_token(api)
        return args.func(args, api, token)
    except (ApiError, AuthError, ManifestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_push(args, api: DatasetApi, token: str) -> int:
    entries = load_manifest(Path(args.manifest))

    if args.names:
        requested = set(args.names)
        entries = [e for e in entries if e.name in requested]
        missing = requested - {e.name for e in entries}
        if missing:
            raise ManifestError(
                f"{args.manifest} does not declare: {', '.join(sorted(missing))}"
            )

    # Validate every file up front so a typo in the last entry doesn't surface
    # only after the earlier ones have already been uploaded. Size is checked
    # here too: the API rejects an oversized upload anyway, but only after the
    # bytes have crossed the wire.
    for entry in entries:
        if not entry.path.is_file():
            raise ManifestError(f"{entry.name}: no such file {entry.path}")
        size = entry.path.stat().st_size
        if size > MAX_DATASET_BYTES:
            raise ManifestError(
                f"{entry.name}: {entry.path.name} is {human_bytes(size)}, over the "
                f"{human_bytes(MAX_DATASET_BYTES)} per-dataset limit. Every dataset a "
                f"model declares is baked into its container image, so the cap is "
                f"about image size, not the malware scanner. Shrink the file "
                f"(compress it, drop unused columns, coarsen the resolution) or "
                f"split it into several smaller datasets."
            )

    exit_code = 0
    for entry in entries:
        size = entry.path.stat().st_size
        try:
            reservation = api.push(token, entry.name, entry.version, entry.filename, size)
        except ApiError as exc:
            print(f"{entry.name}@{entry.version}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        api.upload(reservation["upload_url"], entry.path)
        print(f"{entry.name}@{entry.version}  upload-id {reservation['upload_id']}")

    print(
        "\nUploads are being scanned for malware. Check progress with:\n"
        "  python -m compartment.datasets check-status <upload-id>",
        file=sys.stderr,
    )
    return exit_code


def _cmd_check_status(args, api: DatasetApi, token: str) -> int:
    record = api.status(token, args.upload_id)
    dataset = record.get("dataset", {})

    print(f"upload-id : {record['upload_id']}")
    print(f"dataset   : {dataset.get('name')}@{dataset.get('version')} "
          f"({dataset.get('filename')})")
    print(f"status    : {record['status']}")
    print(f"detail    : {record.get('detail', '')}")
    if record.get("s3_uri"):
        print(f"location  : {record['s3_uri']}")

    # Non-zero for a terminal failure so this is usable in a script; SCANNING
    # is not a failure, just not finished.
    return 1 if record["status"] in ("REJECTED", "FAILED") else 0


def _cmd_list(args, api: DatasetApi, token: str) -> int:
    datasets = api.list_datasets(token)
    if not datasets:
        print("No datasets available.")
        return 0

    width = max(len(d["name"]) for d in datasets)
    for dataset in datasets:
        print(f"{dataset['name']:<{width}}  {dataset['version']:<12}  {dataset['filename']}")
    return 0


def _cmd_stage(args) -> int:
    """Download a model's declared datasets with AWS credentials (CI step)."""
    from compartment.datasets.stage import (
        resolve_bucket,
        resolve_max_total_bytes,
        stage_model_dir,
    )

    for line in stage_model_dir(
        Path(args.model_dir), resolve_bucket(), resolve_max_total_bytes()
    ):
        print(line)
    return 0


def _cmd_pull(args, api: DatasetApi, token: str) -> int:
    name, _, version = args.dataset.partition("@")
    result = api.pull(token, name, version or None)

    destination = Path(args.dest) / result["filename"]
    api.download(result["download_url"], destination)
    print(f"{result['name']}@{result['version']} -> {destination}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m compartment.datasets",
        description="Upload and retrieve datasets for the Pandemic Simulator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    push = subparsers.add_parser(
        "push",
        help="Upload datasets declared in a datasets.yaml. Returns immediately; "
             "the malware scan runs asynchronously.",
    )
    push.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="Dataset names to push. Omit to push every entry in the manifest.",
    )
    push.add_argument(
        "--manifest",
        default=MANIFEST_FILENAME,
        help=f"Path to the datasets.yaml (default: ./{MANIFEST_FILENAME}).",
    )
    push.set_defaults(func=_cmd_push)

    check_status = subparsers.add_parser(
        "check-status",
        help="Report whether an upload was scanned and promoted, or deleted.",
    )
    check_status.add_argument("upload_id", metavar="UPLOAD_ID")
    check_status.set_defaults(func=_cmd_check_status)

    list_cmd = subparsers.add_parser("list", help="List available datasets.")
    list_cmd.set_defaults(func=_cmd_list)

    pull = subparsers.add_parser("pull", help="Download a dataset version.")
    pull.add_argument(
        "dataset",
        metavar="NAME[@VERSION]",
        help="Dataset to download. Without a version, the most recently "
             "promoted one is used.",
    )
    pull.add_argument("--dest", default=".", help="Destination directory (default: .).")
    pull.set_defaults(func=_cmd_pull)

    stage = subparsers.add_parser(
        "stage",
        help="Download a model's declared datasets to their datasets.yaml paths "
             "using AWS credentials. Used by the release pipeline before "
             "docker build; modelers want `pull`.",
    )
    stage.add_argument(
        "--model-dir",
        required=True,
        help="Model folder holding the datasets.yaml "
             "(e.g. compartment/models/my_model).",
    )
    # No auth: this path uses boto3 with the caller's AWS role, so it must skip
    # the browser/paste token flow that every other subcommand needs.
    stage.set_defaults(func=_cmd_stage, needs_auth=False)

    return parser
