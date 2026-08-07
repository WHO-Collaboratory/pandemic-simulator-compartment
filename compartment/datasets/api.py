"""HTTP client for the dataset API Lambda Function URL.

The API never carries file bytes — it hands back presigned S3 URLs and the CLI
transfers directly to and from S3.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_TIMEOUT_SECONDS = 30
# Uploads and downloads go straight to S3 and can be large, so they get their
# own, much longer budget than the JSON API calls.
TRANSFER_TIMEOUT_SECONDS = 900


class ApiError(Exception):
    """Raised when the dataset API returns a non-2xx response."""


class DatasetApi:
    """Thin wrapper over the dataset API's five routes."""

    def __init__(self, base_url: str | None = None):
        base_url = base_url or os.environ.get("PANSIM_DATASET_API", "")
        if not base_url:
            raise ApiError(
                "Set PANSIM_DATASET_API to the dataset API Function URL "
                "(tofu output dataset_api_url in infra/tofu/shared-services)."
            )
        self.base_url = base_url.rstrip("/")

    # -- routes ------------------------------------------------------------

    def whoami(self, token: str) -> dict:
        return self._request("GET", "/whoami", token)

    def push(self, token: str, name: str, version: str, filename: str, size: int) -> dict:
        return self._request("POST", "/push", token, json={
            "name": name,
            "version": version,
            "filename": filename,
            "size": size,
        })

    def status(self, token: str, upload_id: str) -> dict:
        return self._request("GET", f"/status/{upload_id}", token)

    def list_datasets(self, token: str) -> list[dict]:
        return self._request("GET", "/datasets", token)["datasets"]

    def pull(self, token: str, name: str, version: str | None) -> dict:
        params = {"name": name}
        if version:
            params["version"] = version
        return self._request("GET", "/pull", token, params=params)

    # -- S3 transfers ------------------------------------------------------

    @staticmethod
    def upload(presigned_url: str, path: Path) -> None:
        """PUT a file to a presigned URL, streaming rather than buffering."""
        with open(path, "rb") as handle:
            response = requests.put(
                presigned_url, data=handle, timeout=TRANSFER_TIMEOUT_SECONDS
            )
        if not response.ok:
            raise ApiError(f"Upload failed ({response.status_code}): {response.text[:500]}")

    @staticmethod
    def download(presigned_url: str, destination: Path) -> None:
        """GET a presigned URL to disk, streaming rather than buffering."""
        with requests.get(
            presigned_url, stream=True, timeout=TRANSFER_TIMEOUT_SECONDS
        ) as response:
            if not response.ok:
                raise ApiError(
                    f"Download failed ({response.status_code}): {response.text[:500]}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)

    # -- internals ---------------------------------------------------------

    def _request(self, method: str, path: str, token: str, **kwargs) -> dict:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
            **kwargs,
        )

        try:
            body = response.json()
        except ValueError:
            raise ApiError(f"{method} {path} returned {response.status_code}: {response.text[:500]}")

        if not response.ok:
            raise ApiError(body.get("error", f"{method} {path} failed ({response.status_code})."))
        return body
