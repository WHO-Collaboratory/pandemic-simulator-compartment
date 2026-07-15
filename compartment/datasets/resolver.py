"""Resolve a logical dataset name to a concrete, locally-readable file.

Given a dataset name and the active configuration (mode, version pins from the
simulation config, the model's ``datasets.yaml`` manifest, and a cache root),
produce the exact on-disk path the loader should read.

Version precedence (highest first):
    1. explicit ``version`` argument to ``datasets.load()``
    2. a pin in the simulation config (``dataset_pins``) — the reproducibility path
    3. the version declared in ``datasets.yaml``
    4. the registry's latest published version (cloud only; deferred in M1) /
       the newest version present in the local cache (local ``latest``)

The same concrete ``(slug, version)`` resolves identically in local and cloud
mode — only the backing store differs: a local cache directory, or an S3
object downloaded to ``/tmp`` and verified against the pin's ``content_hash``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compartment.datasets import registry_client
from compartment.datasets.manifest import DatasetDep


@dataclass(frozen=True)
class Resolved:
    """A dataset resolved to a concrete version and a ready-to-read local path."""

    slug: str
    version: str
    path: Path


def _pin_for(pins, name):
    """Return the config pin whose ``slug`` matches ``name`` (or ``None``)."""
    for pin in pins or []:
        if isinstance(pin, dict) and pin.get("slug") == name:
            return pin
    return None


class Resolver:
    """Resolves dataset names against pins, a manifest, and a cache root."""

    def __init__(
        self,
        *,
        mode: str,
        environment: str | None,
        pins: list | None,
        manifest: dict[str, DatasetDep] | None,
        cache_root: Path,
        allowed_suffixes,
    ):
        self.mode = mode
        self.environment = environment
        self.pins = pins or []
        self.manifest: dict[str, DatasetDep] = manifest or {}
        self.cache_root = Path(cache_root)
        self.allowed_suffixes = set(allowed_suffixes)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def resolve(self, name: str, explicit_version: str | None = None) -> Resolved:
        pin = _pin_for(self.pins, name)
        version = self._resolve_version(name, explicit_version, pin)

        if self.mode == "cloud":
            path = self._cloud_path(name, version, pin)
        else:
            path = self._local_path(name, version)

        suffix = path.suffix.lower()
        if suffix not in self.allowed_suffixes:
            raise ValueError(
                f"Dataset '{name}' resolves to a '{suffix}' file, which the "
                "datasets SDK will not read. Allowed formats: "
                f"{self._allowed_str()}. Code-deserializing formats "
                "(e.g. pickle) are never read."
            )

        # Reproducibility integrity: if the pin carries a content_hash, the bytes
        # behind the pinned version must match — in local and cloud alike — so a
        # run fails loudly rather than silently using wrong data.
        if pin and pin.get("content_hash"):
            self._verify_hash(path, pin["content_hash"])

        return Resolved(slug=name, version=version, path=path)

    # ------------------------------------------------------------------
    # Version resolution
    # ------------------------------------------------------------------
    def _resolve_version(self, name, explicit_version, pin):
        if explicit_version:
            return explicit_version
        if pin and pin.get("version"):
            return pin["version"]
        dep = self.manifest.get(name)
        if dep and dep.version and dep.version != "latest":
            return dep.version
        # Undeclared or "latest".
        if self.mode == "cloud":
            return registry_client.get_latest_published_version(
                name, environment=self.environment
            )
        return self._newest_local_version(name)

    def _newest_local_version(self, name):
        slug_dir = self.cache_root / name
        if slug_dir.is_dir():
            versions = sorted(
                (d.name for d in slug_dir.iterdir() if d.is_dir()), reverse=True
            )
            if versions:
                return versions[0]
        raise FileNotFoundError(self._missing_msg(name, "latest"))

    # ------------------------------------------------------------------
    # File location — local cache
    # ------------------------------------------------------------------
    def _version_dir(self, name, version) -> Path:
        return self.cache_root / name / version

    def _local_path(self, name, version) -> Path:
        version_dir = self._version_dir(name, version)
        if not version_dir.is_dir():
            raise FileNotFoundError(self._missing_msg(name, version))
        return self._locate(name, version, version_dir)

    def _locate(self, name, version, version_dir) -> Path:
        """Pick the single data file in a version directory.

        Ignores dotfiles and sidecars (e.g. ``data.csv.sha256``). A directory
        whose only content is a disallowed format (e.g. a renamed ``.pkl``)
        raises a clear "unsupported format" error — the file is never opened.
        """
        files = [
            f
            for f in version_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        allowed = [f for f in files if f.suffix.lower() in self.allowed_suffixes]
        if len(allowed) == 1:
            return allowed[0]
        if len(allowed) > 1:
            raise ValueError(
                f"Dataset '{name}' version '{version}' has multiple data files "
                f"({', '.join(f.name for f in allowed)}); expected exactly one."
            )
        if files:
            raise ValueError(
                f"Dataset '{name}' version '{version}' contains no supported "
                f"data file (found: {', '.join(f.name for f in files)}). "
                f"Allowed formats: {self._allowed_str()}. Code-deserializing "
                "formats (e.g. pickle) are never read."
            )
        raise FileNotFoundError(self._missing_msg(name, version))

    # ------------------------------------------------------------------
    # File location — cloud (S3 download to /tmp, hash-verified)
    # ------------------------------------------------------------------
    def _cloud_path(self, name, version, pin) -> Path:
        if not pin or not pin.get("key"):
            raise FileNotFoundError(
                f"No dataset pin with an S3 key for '{name}' (version "
                f"'{version}'). Cloud runs resolve datasets from the "
                "dataset_pins frozen on the SimulationJob."
            )
        version_dir = self._version_dir(name, version)
        version_dir.mkdir(parents=True, exist_ok=True)
        dest = version_dir / Path(pin["key"]).name
        if not dest.exists():
            self._download(pin, dest)
        # Hash verification happens centrally in resolve() for both modes.
        return dest

    def _download(self, pin, dest):
        import boto3  # lazy — never imported on the offline local path

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.download_file(pin["bucket"], pin["key"], str(dest))

    def _verify_hash(self, path, content_hash):
        if not content_hash:
            return
        import hashlib

        expected = content_hash.split(":", 1)[-1]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Content hash mismatch for {path}: expected {expected}, got "
                f"{actual}. The pinned dataset bytes changed — refusing to run "
                "on the wrong data."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _allowed_str(self):
        return ", ".join(sorted(s.lstrip(".") for s in self.allowed_suffixes))

    def _missing_msg(self, name, version):
        return (
            f"Dataset '{name}' version '{version}' not found in the local cache "
            f"({self.cache_root}). Run:\n"
            f"    python -m compartment.datasets pull {name} --version {version}"
        )
