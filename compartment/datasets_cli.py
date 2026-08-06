"""Bring-Your-Own-Dataset CLI — push, pull, list, and check datasets.

    python -m compartment.datasets login
    python -m compartment.datasets push --file mobility.csv \
        --slug mobility/kenya --version 2026-06-01 --wait
    python -m compartment.datasets pull mobility/kenya --version 2026-06-01
    python -m compartment.datasets list
    python -m compartment.datasets status mobility/kenya

Requires only a platform login and outbound HTTPS — **no AWS credentials and no
GraphQL api key**. The CLI authenticates to Cognito (WHO SSO in a browser, or
email/password) and talks exclusively to the platform's ``/api/datasets/*``
endpoints, which mint short-lived presigned S3 URLs.

A push lands in the **quarantine** bucket with ``status = PENDING_SCAN``. Only the
scan gate (GuardDuty malware scanning plus content validation) can promote it to
``PUBLISHED``; the CLI cannot publish, by design. Use ``push --wait`` or
``status`` to see the verdict.

Every published dataset is readable by every authenticated modeler. Only the owner
of a slug may push new versions of it.

**There is one dataset catalog, shared by every environment.** The CLI always
targets it — no endpoint or environment to configure, and therefore no way to
publish to the wrong place because a shell variable was stale. Other environments
read the same datasets cross-account, so a dataset published once is available to
every simulation everywhere.

Environment (``.env`` supported):
    WHO_DATASET_CACHE       local dataset cache root (shared with the SDK)

For local development against a ``next dev`` server, ``WHO_API_URL``,
``COGNITO_WEB_CLIENT_ID`` and ``DATASETS_ENV`` override the built-in target; they
are not needed in normal use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

from compartment.datasets import auth
from compartment.datasets.api_client import ApiClient, ApiError
from compartment.datasets.loader import ALLOWED_SUFFIXES, DEFAULT_CACHE_DIR, read_frame

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Pure helpers (no network — unit-tested offline)
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def dataset_id_for(slug: str) -> str:
    """Deterministic Dataset id so re-pushing the same slug is idempotent.

    Mirrored in the presign route (``datasetKeys.ts``) and the scan gate
    (``dataset-scan/src/handler.ts``); all three must agree or a push uploads to
    one key while the registry row points at another.
    """
    return "dataset-" + slug.strip("/").replace("/", "-").lower()


def version_id_for(slug: str, version: str) -> str:
    """Deterministic DatasetVersion id for ``(slug, version)``."""
    safe_v = str(version).replace("/", "-").replace(" ", "_")
    return f"{dataset_id_for(slug)}-{safe_v}"


def object_key(slug: str, version: str, filename: str) -> str:
    """S3 key ``<slug>/<version>/<filename>``, where slug is ``<namespace>/<name>``.

    Advisory on this side: the server derives the authoritative key from the same
    components and ignores any client-supplied key, so traversal is impossible
    rather than merely filtered.
    """
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

    Reads through the SDK's safe-reader allowlist (never pickle) and returns
    scalar metadata only — no file bytes ever enter DynamoDB. ``column_schema`` is
    column names + dtypes, never data.
    """
    path = Path(path)
    validate_extension(path)
    df = read_frame(path)
    column_schema = {
        "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
    }
    return {
        "file_name": path.name,
        "file_size": path.stat().st_size,
        "row_count": int(len(df)),
        "column_schema": json.dumps(column_schema),
    }


# CSV/spreadsheet formula-injection guard (CWE-1236). The authoritative check runs
# in the scan gate; this is a client-side pre-flight warning so a modeler finds out
# before uploading rather than after a rejection.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def is_formula_injection(value) -> bool:
    """Whether a cell would be treated as a formula by a spreadsheet.

    Mirrors ``validate.ts::isFormulaInjectionCell``, including its carve-out for
    values that are merely negative or explicitly-positive numbers ("-5", "+3.2").
    Without that carve-out this warns on ordinary numeric data held in an object
    column (e.g. one containing NAs), which the scan gate accepts — so the
    pre-flight would contradict the gate it is meant to predict.
    """
    if not isinstance(value, str) or value[:1] not in _FORMULA_PREFIXES:
        return False
    try:
        float(value)
    except ValueError:
        return True
    return False  # a real number, not a formula


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


def _cache_root() -> Path:
    return Path(os.getenv("WHO_DATASET_CACHE", DEFAULT_CACHE_DIR)).expanduser()


# Terminal states for the scan gate.
TERMINAL_STATUSES = ("PUBLISHED", "REJECTED", "ARCHIVED")

# Exit codes, so `push --wait` and `status` work as CI gates.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REJECTED = 2
EXIT_PENDING = 3


def exit_code_for(status: str | None) -> int:
    if status == "PUBLISHED":
        return EXIT_OK
    if status == "REJECTED":
        return EXIT_REJECTED
    if status in ("PENDING_SCAN", "QUARANTINED"):
        return EXIT_PENDING
    return EXIT_ERROR


def format_verdict(verdict: str | None) -> str:
    """Render a ``category:detail`` scan verdict readably."""
    if not verdict:
        return ""
    category, _, detail = verdict.partition(":")
    return f"{category}: {detail.strip()}" if detail else category


# ---------------------------------------------------------------------------
# Auth commands
# ---------------------------------------------------------------------------
def cmd_login(args) -> int:
    try:
        if args.native:
            profile = auth.login_native(email=args.email)
        else:
            profile = auth.login_browser(port=args.port, no_browser=args.no_browser)
    except auth.AuthError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    groups = ", ".join(profile.get("groups") or []) or "(none)"
    print(f"Signed in as {profile.get('email') or profile.get('sub')}")
    print(f"  groups: {groups}")
    if not auth.can_publish():
        print(
            "  NOTE: you are not in a dataset-publishing group, so `push` will be "
            "refused. Ask an admin to add you to DISEASE_MODELER.",
            file=sys.stderr,
        )
    return EXIT_OK


def cmd_logout(_args) -> int:
    print("Signed out." if auth.logout() else "Not signed in.")
    return EXIT_OK


def cmd_whoami(_args) -> int:
    who = auth.whoami()
    if not who:
        print("Not signed in. Run: python -m compartment.datasets login")
        return EXIT_ERROR
    print(f"env:     {who['env']}")
    print(f"email:   {who['email']}")
    print(f"sub:     {who['sub']}")
    print(f"groups:  {', '.join(who['groups']) or '(none)'}")
    print(
        "expires: "
        + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(who["expires_at"]))
        + (" (expired — refreshes on next command)" if who["expired"] else "")
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def cmd_push(args) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return EXIT_ERROR

    try:
        meta = compute_metadata(path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    content_hash = sha256_file(path)
    key = object_key(args.slug, args.version, meta["file_name"])

    print(f"Pushing {args.slug}@{args.version}")
    print(f"  sha256: {content_hash[:16]}...  rows: {meta['row_count']}")
    print(f"  -> quarantine/{key}  (awaits scan)")
    print("  Note: published datasets are readable by all modelers.")

    hits = formula_injection_cells(path)
    if hits:
        print(
            f"WARNING: {len(hits)} cell(s) look like spreadsheet formula injection "
            f"(e.g. {hits[0]}). The scan gate will REJECT this file — fix it before "
            "pushing.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("  [DRY RUN] no upload or registry changes made")
        return EXIT_OK

    client = ApiClient()
    try:
        presigned = client.presign_upload(
            {
                "slug": args.slug,
                "version": str(args.version),
                "file_name": meta["file_name"],
                "content_hash": content_hash,
                "file_size": meta["file_size"],
                "row_count": meta["row_count"],
                "column_schema": meta["column_schema"],
                "name": args.name,
                "description": args.description,
            }
        )
    except (ApiError, auth.AuthError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # Plain HTTPS PUT to the presigned URL — no AWS SDK, no credentials. The server
    # signed an exact Content-Length, so these headers must be sent verbatim.
    with open(path, "rb") as fh:
        resp = requests.put(
            presigned["upload_url"],
            data=fh,
            headers=presigned.get("headers") or {},
            timeout=900,
        )
    if resp.status_code >= 400:
        print(
            f"ERROR: upload failed ({resp.status_code}): {resp.text[:500]}",
            file=sys.stderr,
        )
        return EXIT_ERROR

    print(f"  uploaded ({meta['file_size']:,} bytes)")

    if not args.wait:
        print(
            "  PENDING_SCAN. It becomes loadable once the scan gate promotes it.\n"
            f"  Check: python -m compartment.datasets status {args.slug} "
            f"--version {args.version}"
        )
        return EXIT_OK

    return _wait_for_scan(client, args.slug, str(args.version), args.wait_timeout)


def _wait_for_scan(client: ApiClient, slug: str, version: str, timeout: int) -> int:
    """Poll until the scan gate reaches a terminal state.

    Polls tightly at first (small text files usually clear in seconds) then backs
    off, so a large object doesn't generate hundreds of requests.
    """
    started = time.time()
    interval = 3
    last_status: str | None = None
    is_tty = sys.stderr.isatty()

    print("  waiting for the scan gate ...")
    while True:
        elapsed = int(time.time() - started)
        if elapsed > timeout:
            print(
                f"  TIMEOUT after {elapsed}s, still {last_status or 'PENDING_SCAN'}. "
                f"Check later: python -m compartment.datasets status {slug} "
                f"--version {version}",
                file=sys.stderr,
            )
            return EXIT_PENDING

        try:
            report = client.status(slug, version)
        except (ApiError, auth.AuthError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return EXIT_ERROR

        versions = report.get("versions") or []
        current = versions[0] if versions else {}
        status = current.get("status")

        if status != last_status:
            print(f"  {status} ({elapsed}s)")
            last_status = status
        elif is_tty:
            # Only overwrite in place on a terminal; piped output gets one line
            # per state change instead of a stream of carriage returns.
            print(f"  {status} ({elapsed}s)", end="\r", file=sys.stderr)

        if status in TERMINAL_STATUSES:
            if is_tty:
                print("", file=sys.stderr)
            verdict = format_verdict(current.get("scan_verdict"))
            if status == "PUBLISHED":
                print(
                    f'  PUBLISHED after {elapsed}s. Load with: datasets.load("{slug}")'
                )
            else:
                print(f"  {status} after {elapsed}s.", file=sys.stderr)
                if verdict:
                    print(f"  verdict: {verdict}", file=sys.stderr)
                print("  Fix the file and push the same version again.", file=sys.stderr)
            return exit_code_for(status)

        time.sleep(interval)
        if time.time() - started > 30:
            interval = 10


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------
def cmd_pull(args) -> int:
    client = ApiClient()
    try:
        presigned = client.presign_download(args.slug, args.version)
    except (ApiError, auth.AuthError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    version = presigned["version"]
    file_name = presigned.get("file_name") or "data.csv"
    dest_dir = _cache_root() / args.slug / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name

    print(f"Pulling {args.slug}@{version} -> {dest}")
    with requests.get(presigned["download_url"], stream=True, timeout=900) as resp:
        if resp.status_code >= 400:
            print(
                f"ERROR: download failed ({resp.status_code}): {resp.text[:500]}",
                file=sys.stderr,
            )
            return EXIT_ERROR
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)

    expected = (presigned.get("content_hash") or "").split(":", 1)[-1]
    if expected:
        actual = sha256_file(dest)
        if actual != expected:
            # Remove it: a corrupt file left in the cache would be silently read
            # by the next local run.
            dest.unlink(missing_ok=True)
            print(
                f"ERROR: content hash mismatch (expected {expected}, got {actual}). "
                "Removed the download rather than caching the wrong bytes.",
                file=sys.stderr,
            )
            return EXIT_ERROR
        print("  OK (hash verified)")
    else:
        print("  OK")
    return EXIT_OK


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def cmd_list(args) -> int:
    client = ApiClient()
    scope = "mine" if args.mine else "all"
    try:
        items = client.list_datasets(scope).get("items", [])
    except (ApiError, auth.AuthError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not items:
        print("(you own no datasets)" if scope == "mine" else "(no datasets)")
        return EXIT_OK

    print(f"{'SLUG':<32} {'VERSION':<14} {'STATUS':<14} {'OWNER':<6} NAME")
    for d in items:
        print(
            f"{d.get('slug', ''):<32} "
            f"{str(d.get('latest_version') or '-'):<14} "
            f"{str(d.get('status') or '-'):<14} "
            f"{('me' if d.get('is_mine') else '-'):<6} "
            f"{d.get('name') or ''}"
        )
    return EXIT_OK


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def cmd_status(args) -> int:
    client = ApiClient()
    try:
        report = client.status(args.slug, args.version)
    except (ApiError, auth.AuthError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    versions = report.get("versions") or []

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if not versions:
            print(f"(no versions of {args.slug})")
            return EXIT_ERROR
        print(f"{'SLUG':<28} {'VERSION':<14} {'STATUS':<14} VERDICT")
        for v in versions:
            print(
                f"{report.get('slug', ''):<28} "
                f"{v.get('version', ''):<14} "
                f"{str(v.get('status') or '-'):<14} "
                f"{format_verdict(v.get('scan_verdict'))}"
            )

    # With --version there is exactly one row; otherwise the newest is first.
    return exit_code_for(versions[0].get("status") if versions else None)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m compartment.datasets",
        description=(
            "Push/pull/list Bring-Your-Own-Dataset files. Run `login` first; no AWS "
            "credentials required."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    login = sub.add_parser(
        "login", help="Sign in to the platform (WHO SSO or email/password)."
    )
    login.add_argument(
        "--native",
        action="store_true",
        help="Use email/password instead of the browser SSO flow.",
    )
    login.add_argument("--email", default=None, help="Email for --native login.")
    login.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the sign-in URL and prompt for the redirect (for SSH sessions).",
    )
    login.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "Loopback port for the redirect (one of "
            f"{', '.join(map(str, auth.LOOPBACK_PORTS))})."
        ),
    )
    login.set_defaults(func=cmd_login)

    sub.add_parser("logout", help="Forget cached credentials.").set_defaults(
        func=cmd_logout
    )
    sub.add_parser("whoami", help="Show the signed-in identity.").set_defaults(
        func=cmd_whoami
    )

    push = sub.add_parser("push", help="Upload a new immutable dataset version.")
    push.add_argument("--file", required=True)
    push.add_argument("--slug", required=True, help="Logical id '<namespace>/<name>'.")
    push.add_argument("--version", required=True)
    push.add_argument("--name", default=None)
    push.add_argument("--description", default=None)
    push.add_argument("--dry-run", action="store_true")
    push.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the scan gate reaches PUBLISHED or REJECTED.",
    )
    push.add_argument(
        "--wait-timeout",
        type=int,
        default=600,
        help="Seconds to wait with --wait (default 600).",
    )
    push.set_defaults(func=cmd_push)

    pull = sub.add_parser("pull", help="Download a PUBLISHED version to the cache.")
    pull.add_argument("slug")
    pull.add_argument("--version", default=None)
    pull.set_defaults(func=cmd_pull)

    lst = sub.add_parser("list", help="List datasets (scalar metadata only).")
    lst.add_argument("--mine", action="store_true", help="Only datasets you own.")
    lst.set_defaults(func=cmd_list)

    st = sub.add_parser("status", help="Show scan status / verdict for a dataset.")
    st.add_argument("slug")
    st.add_argument("--version", default=None)
    st.add_argument("--json", action="store_true", help="Emit raw JSON.")
    st.set_defaults(func=cmd_status)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
