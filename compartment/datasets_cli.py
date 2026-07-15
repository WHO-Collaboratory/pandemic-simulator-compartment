"""Local <-> deploy parity CLI for Bring-Your-Own-Dataset.

    python -m compartment.datasets push --file mobility.csv \
        --slug mobility/kenya --version 2026-06-01 --visibility private
    python -m compartment.datasets pull mobility/kenya --version 2026-06-01
    python -m compartment.datasets list --mine

Mirrors ``scripts/seed_model_artifacts.py`` (GraphQL client + sha256 + upsert).
A push computes the file's ``content_hash`` + ``row_count`` + ``column_schema``
with pandas (via the same safe-reader allowlist the SDK uses — never
``read_pickle``), upserts a ``Dataset``, uploads the bytes to the **quarantine**
bucket, and creates a ``DatasetVersion`` with ``status = PENDING_SCAN``. The
scan gate (GuardDuty + promote Lambda) is what flips it to ``PUBLISHED``; the
CLI never publishes directly. ``pull`` downloads a ``PUBLISHED`` version into
the same local cache the SDK reads, so local runs work offline afterwards.

Credentials/config via environment (``.env`` supported):
    GQL_API_URL, GQL_API_KEY        GraphQL endpoint + api key
    DATASETS_ENV                    deployment env suffix (default "dev")
    WHO_DATASET_CACHE               local cache root (shared with the SDK)
S3 uploads/downloads use the ambient boto3 credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from compartment.datasets.loader import ALLOWED_SUFFIXES, DEFAULT_CACHE_DIR, read_frame

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Bucket naming (mirrors compartmental-results-<env>)
# ---------------------------------------------------------------------------
def datasets_bucket(env: str) -> str:
    return f"collaboratory-datasets-{env}"


def quarantine_bucket(env: str) -> str:
    return f"collaboratory-datasets-quarantine-{env}"


# ---------------------------------------------------------------------------
# Pure helpers (no network / no S3 — unit-tested offline)
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def dataset_id_for(slug: str) -> str:
    """Deterministic Dataset id so re-pushing the same slug is idempotent."""
    return "dataset-" + slug.strip("/").replace("/", "-").lower()


def version_id_for(slug: str, version: str) -> str:
    """Deterministic DatasetVersion id for ``(slug, version)``."""
    safe_v = str(version).replace("/", "-").replace(" ", "_")
    return f"{dataset_id_for(slug)}-{safe_v}"


def object_key(slug: str, version: str, filename: str) -> str:
    """S3 key ``<namespace>/<slug>/<version>/<filename>`` (same in both buckets)."""
    return f"{slug.strip('/')}/{version}/{Path(filename).name}"


def validate_extension(path: str | Path) -> str:
    """Return the lowercase suffix, or raise if it is not an allowed format."""
    suffix = Path(path).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(s.lstrip(".") for s in ALLOWED_SUFFIXES))
        raise ValueError(
            f"Refusing to push a '{suffix}' file. Allowed formats: {allowed}. "
            "Code-deserializing formats (e.g. pickle) are never handled."
        )
    return suffix


def compute_metadata(path: str | Path) -> dict:
    """Compute 1 MB-safe registry metadata for a dataset file.

    Reads the file through the SDK's safe-reader allowlist (never pickle) and
    returns scalar metadata only — no file bytes ever enter DynamoDB.
    ``column_schema`` is column names + dtypes, never data.
    """
    path = Path(path)
    validate_extension(path)
    df = read_frame(path)
    column_schema = {
        "columns": [
            {"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns
        ]
    }
    return {
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "row_count": int(len(df)),
        "column_schema": json.dumps(column_schema),
    }


# CSV/spreadsheet formula-injection guard (CWE-1236). The authoritative check
# runs in the scan-gate Lambda; this is a client-side pre-flight warning.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def is_formula_injection(value) -> bool:
    return isinstance(value, str) and value[:1] in _FORMULA_PREFIXES


def formula_injection_cells(path: str | Path, max_report: int = 20) -> list[str]:
    """Return ``col[row]`` locations of cells that look like formula injection."""
    df = read_frame(Path(path))
    hits: list[str] = []
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        for row, value in series.items():
            if is_formula_injection(value):
                hits.append(f"{col}[{row}]")
                if len(hits) >= max_report:
                    return hits
    return hits


# ---------------------------------------------------------------------------
# GraphQL client (mirrors scripts/seed_model_artifacts.py)
# ---------------------------------------------------------------------------
class GraphQLClient:
    def __init__(self, url: str, api_key: str):
        import requests

        self._requests = requests
        self.url = url
        self.headers = {"Content-Type": "application/json", "x-api-key": api_key}

    def execute(self, query: str, variables: dict | None = None) -> dict:
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self._requests.post(
            self.url, json=payload, headers=self.headers, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errors"):
            raise RuntimeError(f"GraphQL error: {data['errors']}")
        return data.get("data", {})


UPSERT_DATASET = """
mutation UpsertDataset($input: CreateDatasetInput!) {
  createDataset(input: $input) { id slug }
}
"""

CREATE_VERSION = """
mutation CreateDatasetVersion($input: CreateDatasetVersionInput!) {
  createDatasetVersion(input: $input) { id version status }
}
"""

GET_DATASET_BY_SLUG = """
query DatasetBySlug($slug: String!) {
  datasetBySlug(slug: $slug) {
    items { id slug name format visibility status latest_version owner }
  }
}
"""

LIST_VERSIONS = """
query Versions($dataset_id: ID!) {
  datasetVersionsByDatasetId(dataset_id: $dataset_id) {
    items { id version status bucket key file_name content_hash }
  }
}
"""

LIST_DATASETS = """
query ListDatasets($limit: Int, $nextToken: String) {
  listDatasets(limit: $limit, nextToken: $nextToken) {
    items { id slug name format visibility status latest_version owner }
    nextToken
  }
}
"""


def _client(args) -> GraphQLClient:
    url = args.url or os.getenv("GQL_API_URL", "")
    key = args.api_key or os.getenv("GQL_API_KEY", "")
    if not url or not key:
        print(
            "ERROR: set GQL_API_URL and GQL_API_KEY (or pass --url/--api-key).",
            file=sys.stderr,
        )
        sys.exit(1)
    return GraphQLClient(url, key)


def _s3():
    import boto3

    return boto3.client("s3", region_name="us-east-1")


def _cache_root() -> Path:
    return Path(os.getenv("WHO_DATASET_CACHE", DEFAULT_CACHE_DIR)).expanduser()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_push(args) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return 1

    namespace = args.slug.split("/", 1)[0] if "/" in args.slug else ""
    content_hash = sha256_file(path)
    meta = compute_metadata(path)

    hits = formula_injection_cells(path)
    if hits:
        print(
            f"WARNING: {len(hits)} cell(s) look like spreadsheet formula "
            f"injection (e.g. {hits[0]}). The scan gate will sanitize/reject; "
            "review before publishing.",
            file=sys.stderr,
        )

    env = args.env
    q_bucket = quarantine_bucket(env)
    key = object_key(args.slug, args.version, meta["file_name"])

    print(f"Pushing {args.slug}@{args.version}")
    print(f"  sha256: {content_hash[:16]}...  rows: {meta['row_count']}")
    print(f"  -> s3://{q_bucket}/{key}  (quarantine; awaits scan)")

    if args.dry_run:
        print("  [DRY RUN] no GraphQL/S3 changes made")
        return 0

    client = _client(args)

    client.execute(
        UPSERT_DATASET,
        {
            "input": {
                "id": dataset_id_for(args.slug),
                "slug": args.slug,
                "namespace": namespace,
                "name": args.name or args.slug,
                "description": args.description or "",
                "format": meta["file_name"].rsplit(".", 1)[-1],
                "visibility": args.visibility.upper(),
                "status": "PENDING_SCAN",
            }
        },
    )

    _s3().upload_file(str(path), q_bucket, key)

    client.execute(
        CREATE_VERSION,
        {
            "input": {
                "id": version_id_for(args.slug, args.version),
                "dataset_id": dataset_id_for(args.slug),
                "version": str(args.version),
                "content_hash": content_hash,
                "bucket": q_bucket,
                "key": key,
                "file_name": meta["file_name"],
                "file_size": meta["file_size"],
                "row_count": meta["row_count"],
                "column_schema": meta["column_schema"],
                "source": "MANUAL",
                "status": "PENDING_SCAN",
            }
        },
    )
    print("  Created DatasetVersion (PENDING_SCAN). It becomes loadable once "
          "the scan gate promotes it to PUBLISHED.")
    return 0


def cmd_pull(args) -> int:
    client = _client(args)
    ds_items = client.execute(GET_DATASET_BY_SLUG, {"slug": args.slug}).get(
        "datasetBySlug", {}
    ).get("items", [])
    if not ds_items:
        print(f"ERROR: no dataset '{args.slug}'", file=sys.stderr)
        return 1
    dataset = ds_items[0]

    versions = client.execute(
        LIST_VERSIONS, {"dataset_id": dataset["id"]}
    ).get("datasetVersionsByDatasetId", {}).get("items", [])
    published = [v for v in versions if v.get("status") == "PUBLISHED"]
    if not published:
        print(f"ERROR: no PUBLISHED version of '{args.slug}'", file=sys.stderr)
        return 1

    target = args.version or dataset.get("latest_version")
    version = next((v for v in published if v["version"] == target), None) if target else None
    if version is None:
        version = sorted(published, key=lambda v: v["version"])[-1]

    dest_dir = _cache_root() / args.slug / version["version"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / version["file_name"]
    print(f"Pulling {args.slug}@{version['version']} -> {dest}")
    _s3().download_file(version["bucket"], version["key"], str(dest))

    expected = (version.get("content_hash") or "").split(":", 1)[-1]
    if expected:
        actual = sha256_file(dest)
        if actual != expected:
            print(
                f"ERROR: content hash mismatch (expected {expected}, got {actual})",
                file=sys.stderr,
            )
            return 1
    print("  OK (hash verified)" if expected else "  OK")
    return 0


def cmd_list(args) -> int:
    client = _client(args)
    items, next_token = [], None
    while True:
        page = client.execute(
            LIST_DATASETS, {"limit": 100, "nextToken": next_token}
        ).get("listDatasets", {})
        items.extend(page.get("items", []))
        next_token = page.get("nextToken")
        if not next_token:
            break

    if args.mine or args.public:
        want = "PUBLIC" if args.public else None
        if want:
            items = [d for d in items if d.get("visibility") == want]

    if not items:
        print("(no datasets)")
        return 0
    print(f"{'SLUG':<32} {'VERSION':<14} {'VIS':<8} {'STATUS':<12} NAME")
    for d in sorted(items, key=lambda x: x.get("slug", "")):
        print(
            f"{d.get('slug', ''):<32} {str(d.get('latest_version') or '-'):<14} "
            f"{str(d.get('visibility') or '-'):<8} {str(d.get('status') or '-'):<12} "
            f"{d.get('name') or ''}"
        )
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m compartment.datasets",
        description="Push/pull/list Bring-Your-Own-Dataset files.",
    )
    p.add_argument("--url", default=None, help="GraphQL endpoint (or GQL_API_URL)")
    p.add_argument("--api-key", default=None, help="GraphQL api key (or GQL_API_KEY)")
    p.add_argument(
        "--env",
        default=os.getenv("DATASETS_ENV", "dev"),
        help="Deployment env suffix for bucket names (default: dev).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="Upload a new immutable dataset version.")
    push.add_argument("--file", required=True)
    push.add_argument("--slug", required=True, help="Logical id '<namespace>/<slug>'.")
    push.add_argument("--version", required=True)
    push.add_argument(
        "--visibility", choices=["private", "public"], default="private"
    )
    push.add_argument("--name", default=None)
    push.add_argument("--description", default=None)
    push.add_argument("--dry-run", action="store_true")
    push.set_defaults(func=cmd_push)

    pull = sub.add_parser("pull", help="Download a PUBLISHED version to the cache.")
    pull.add_argument("slug")
    pull.add_argument("--version", default=None)
    pull.set_defaults(func=cmd_pull)

    lst = sub.add_parser("list", help="List datasets (scalar metadata only).")
    grp = lst.add_mutually_exclusive_group()
    grp.add_argument("--mine", action="store_true")
    grp.add_argument("--org", action="store_true")
    grp.add_argument("--public", action="store_true")
    lst.set_defaults(func=cmd_list)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
