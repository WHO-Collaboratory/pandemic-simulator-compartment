"""HTTP client for the platform's dataset endpoints.

The CLI talks **only** to ``/api/datasets/*`` on the web app — never directly to
AppSync or S3. Consequences worth keeping:

* No GraphQL api key ever lands on a modeler's laptop, and no AWS credentials are
  needed at all.
* Visibility and the publish gate are enforced server-side, so they cannot be
  bypassed by editing local arguments.

Every request carries a Cognito ID token from :mod:`compartment.datasets.auth`,
refreshed transparently.
"""

from __future__ import annotations

import os

import requests

from compartment.datasets import auth

DEFAULT_TIMEOUT = 60


class ApiError(RuntimeError):
    """The platform returned an error response."""

    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


def base_url() -> str:
    url = os.getenv("WHO_API_URL")
    if not url:
        raise ApiError(
            0,
            "Set WHO_API_URL to the platform base URL "
            "(e.g. https://dev.pandemic-simulator.com).",
        )
    return url.rstrip("/")


class ApiClient:
    """Thin JSON client with one automatic retry after a forced token refresh."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {auth.id_token()}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{base_url()}{path}"
        resp = requests.request(
            method, url, headers=self._headers(), timeout=self.timeout, **kwargs
        )

        # A 401 here means the token was rejected despite looking unexpired
        # (clock skew, a rotated pool, a revoked session). Force one refresh and
        # retry before surfacing it.
        if resp.status_code == 401:
            auth.refresh(force=True)
            resp = requests.request(
                method, url, headers=self._headers(), timeout=self.timeout, **kwargs
            )

        if resp.status_code >= 400:
            try:
                message = resp.json().get("error") or resp.text
            except ValueError:
                message = resp.text
            raise ApiError(resp.status_code, message)

        if not resp.content:
            return {}
        return resp.json()

    def get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    def post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, json=body)

    # -- endpoints ----------------------------------------------------------
    def presign_upload(self, payload: dict) -> dict:
        return self.post("/api/datasets/presign-upload", payload)

    def presign_download(self, slug: str, version: str | None = None) -> dict:
        body: dict = {"slug": slug}
        if version:
            body["version"] = version
        return self.post("/api/datasets/presign-download", body)

    def list_datasets(self, scope: str = "all") -> dict:
        return self.get("/api/datasets", {"scope": scope})

    def status(self, slug: str, version: str | None = None) -> dict:
        params: dict = {"slug": slug}
        if version:
            params["version"] = version
        return self.get("/api/datasets/status", params)
