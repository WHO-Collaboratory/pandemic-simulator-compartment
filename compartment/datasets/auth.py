"""Cognito authentication for the ``datasets`` CLI.

A modeler authenticates to the **platform**, not to AWS: no IAM user, no access
keys, no ``~/.aws``. Two login flows, both against the existing
``WhoWebClient`` app client (public, no secret):

* **Browser (default)** — WHO SSO via the Cognito hosted UI federated to
  ``WhoEntraID``, using an authorization-code + PKCE flow with a loopback
  redirect. This is the RFC 8252 pattern for native apps.
* **Native (``--native``)** — email + password straight to the Cognito IDP
  endpoint, including the TOTP MFA challenge (MFA is optional-but-enabled on the
  pool, so the challenge path is not optional here).

Tokens are cached per environment under ``~/.config/who-collaboratory/`` with
0600 permissions, and refreshed automatically. ID/access tokens last an hour and
the refresh token 30 days, so an interactive login is roughly monthly.

Deliberately depends only on ``requests`` (already a dependency):
``InitiateAuth``/``RespondToAuthChallenge`` are unauthenticated APIs, so no SRP
library is needed, and boto3 would try to SigV4-sign and fail on a laptop with no
credentials — which is precisely the state this module exists to support.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import http.server
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Datasets have exactly ONE home. There is no per-environment dataset catalog:
# one environment owns the buckets, the scan gate, and the registry, and every
# other environment reads from it cross-account. So the CLI's target is a
# constant, not something the caller configures.
#
# This is deliberate ergonomics as much as architecture. When the target came
# from ambient environment variables, a stale WHO_API_URL in a shell meant
# publishing to the wrong place with no indication -- and since every published
# dataset is readable by every modeler, "the wrong place" is a visible mistake.
# Hardcoding removes that failure mode entirely.
#
# The overrides below exist for local development against `next dev` and for the
# eventual case of a second deployment; they are not part of normal use.
DATASETS_ENVIRONMENT = "uat"
DEFAULT_API_URL = "https://uat.pandemic-simulator.com"
DEFAULT_COGNITO_CLIENT_ID = "2m0jiv8l45dio9v6gftp1r83g2"  # uat-who-web-client
DEFAULT_HOSTED_UI_DOMAIN = (
    "https://uat-pandemic-simulator.auth.us-east-1.amazoncognito.com"
)

# Fixed loopback ports, matching the callback URLs registered on WhoWebClient in
# app/backend/auth/packages/src/functions/cognito/Cognito.ts. Cognito requires an
# exact pre-registered redirect_uri and only permits http:// for the literal host
# "localhost", so these cannot be OS-assigned. Chosen above the registered-port
# range and below the ephemeral range start (49152) so the OS never hands them out.
LOOPBACK_PORTS = (34981, 34982, 34983)
CALLBACK_PATH = "/oauth/callback"

# Refresh this far before nominal expiry, so a long-running command doesn't have
# a token expire mid-flight.
REFRESH_SKEW_SECONDS = 120

DEFAULT_REGION = "us-east-1"
_TIMEOUT = 30


class AuthError(RuntimeError):
    """Login/refresh failed in a way the user must act on."""


# ---------------------------------------------------------------------------
# Environment resolution
# ---------------------------------------------------------------------------
def _env_name() -> str:
    """Profile key for the token cache.

    Constant in normal use — it only varies when someone points the CLI at a
    non-default deployment, and then their tokens shouldn't collide with the real
    ones.
    """
    return os.getenv("DATASETS_ENV", DATASETS_ENVIRONMENT)


def _region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("WHO_REGION") or DEFAULT_REGION


def _client_id() -> str:
    """Cognito app-client id for the CLI (the public ``WhoWebClient``)."""
    return (
        os.getenv("COGNITO_WEB_CLIENT_ID")
        or os.getenv("WHO_COGNITO_CLIENT_ID")
        or DEFAULT_COGNITO_CLIENT_ID
    )


def _hosted_ui_domain() -> str:
    """Base URL of the Cognito hosted UI for the dataset-owning environment."""
    explicit = os.getenv("COGNITO_HOSTED_UI_DOMAIN")
    if explicit:
        return explicit.rstrip("/")
    if _env_name() == DATASETS_ENVIRONMENT:
        return DEFAULT_HOSTED_UI_DOMAIN
    # Overridden environment: mirror the UserPoolDomain prefix in Cognito.ts.
    return (
        f"https://{_env_name()}-pandemic-simulator.auth.{_region()}.amazoncognito.com"
    )


def _idp_endpoint() -> str:
    return f"https://cognito-idp.{_region()}.amazonaws.com/"


# ---------------------------------------------------------------------------
# Token cache
# ---------------------------------------------------------------------------
def _config_dir() -> Path:
    root = os.getenv("XDG_CONFIG_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".config"
    return base / "who-collaboratory"


def _cache_path() -> Path:
    return _config_dir() / "credentials.json"


def _read_cache() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except (OSError, ValueError):
        # A corrupt cache should prompt a re-login, not crash every command.
        return {}


def _write_cache(data: dict) -> None:
    """Persist the cache 0600, via an atomic replace.

    Written to a temp file in the same directory and renamed, so a crash or a
    concurrent reader never observes a partially-written — or briefly
    world-readable — credentials file.
    """
    directory = _config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)

    tmp = directory / f".credentials.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, _cache_path())


def _get_profile(env: str | None = None) -> dict:
    return (_read_cache().get("profiles") or {}).get(env or _env_name()) or {}


def _put_profile(profile: dict, env: str | None = None) -> None:
    cache = _read_cache()
    cache.setdefault("version", 1)
    cache.setdefault("profiles", {})
    cache["profiles"][env or _env_name()] = profile
    _write_cache(cache)


def _clear_profile(env: str | None = None) -> bool:
    cache = _read_cache()
    profiles = cache.get("profiles") or {}
    if (env or _env_name()) not in profiles:
        return False
    del profiles[env or _env_name()]
    cache["profiles"] = profiles
    _write_cache(cache)
    return True


def _decode_claims(id_token: str) -> dict:
    """Decode an ID token payload **without verifying** it.

    For display and local pre-flight messages only — never for an authorization
    decision. The server re-verifies every token signature (see
    app/frontend/.../api/_helpers/datasetAuth.ts).
    """
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError):
        return {}


def _store_tokens(tokens: dict, *, env: str | None = None) -> dict:
    """Persist an AuthenticationResult / token-endpoint response."""
    id_token = tokens.get("IdToken") or tokens.get("id_token") or ""
    access = tokens.get("AccessToken") or tokens.get("access_token") or ""
    refresh = tokens.get("RefreshToken") or tokens.get("refresh_token") or ""
    expires_in = int(tokens.get("ExpiresIn") or tokens.get("expires_in") or 3600)

    claims = _decode_claims(id_token)
    existing = _get_profile(env)
    profile = {
        "id_token": id_token,
        "access_token": access,
        # A refresh-token grant returns no new refresh token — keep the old one.
        "refresh_token": refresh or existing.get("refresh_token", ""),
        "expires_at": int(time.time()) + expires_in,
        "sub": claims.get("sub", ""),
        "email": claims.get("email", ""),
        "groups": claims.get("cognito:groups") or [],
        "client_id": _client_id(),
        "region": _region(),
    }
    _put_profile(profile, env)
    return profile


# ---------------------------------------------------------------------------
# Cognito IDP (unauthenticated APIs — no SigV4)
# ---------------------------------------------------------------------------
def _idp(action: str, body: dict) -> dict:
    resp = requests.post(
        _idp_endpoint(),
        json=body,
        timeout=_TIMEOUT,
        headers={
            "Content-Type": "application/x-amz-json-1.1",
            "X-Amz-Target": f"AWSCognitoIdentityProviderService.{action}",
        },
    )
    if resp.status_code >= 400:
        try:
            err = resp.json()
            message = err.get("message") or err.get("__type") or resp.text
        except ValueError:
            message = resp.text
        raise AuthError(f"Cognito {action} failed: {message}")
    return resp.json()


def _token_endpoint(form: dict) -> dict:
    resp = requests.post(
        f"{_hosted_ui_domain()}/oauth2/token",
        data=form,
        timeout=_TIMEOUT,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code >= 400:
        raise AuthError(f"Token exchange failed ({resp.status_code}): {resp.text}")
    return resp.json()


# ---------------------------------------------------------------------------
# Browser login: authorization code + PKCE over a loopback redirect
# ---------------------------------------------------------------------------
def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(64)).rstrip(b"=").decode()
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Single-shot handler that captures ``?code=`` from the redirect."""

    result: dict = {}

    def do_GET(self):  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        type(self).result = {
            "code": (params.get("code") or [""])[0],
            "state": (params.get("state") or [""])[0],
            "error": (params.get("error_description") or params.get("error") or [""])[0],
        }
        body = (
            b"<html><body style='font-family:sans-serif;padding:2rem'>"
            b"<h3>Signed in.</h3><p>You can close this window and return to your "
            b"terminal.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass  # keep the CLI output clean


class _DualStackServer(http.server.ThreadingHTTPServer):
    """Serve on both ::1 and 127.0.0.1.

    On macOS ``localhost`` frequently resolves to ``::1`` first, so a v4-only
    bind makes the browser fail with ECONNREFUSED even though the server is up.
    """

    address_family = socket.AF_INET6
    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def _bind_loopback(preferred: int | None = None):
    """Bind the first available registered loopback port."""
    ports = (preferred,) if preferred else LOOPBACK_PORTS
    if preferred and preferred not in LOOPBACK_PORTS:
        raise AuthError(
            f"--port must be one of {', '.join(map(str, LOOPBACK_PORTS))} "
            "(these are the redirect URIs registered with Cognito)."
        )

    last: OSError | None = None
    for port in ports:
        for server_cls, addr in ((_DualStackServer, ("::", port)),
                                 (http.server.ThreadingHTTPServer, ("127.0.0.1", port))):
            try:
                return server_cls(addr, _CallbackHandler), port
            except OSError as exc:
                last = exc
    raise AuthError(
        f"Could not bind any of ports {', '.join(map(str, LOOPBACK_PORTS))} "
        f"({last}). Free one, or use --no-browser."
    )


def _authorize_url(
    redirect_uri: str, challenge: str, state: str, *, native: bool = False
) -> str:
    """Build the hosted-UI authorize URL.

    ``native=True`` omits ``identity_provider`` so the hosted UI shows its own
    provider chooser instead of jumping straight to WHO SSO.
    """
    params = {
        "client_id": _client_id(),
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": redirect_uri,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if not native:
        # Jump straight to WHO SSO, as the web app does in authentication-page.tsx.
        params["identity_provider"] = "WhoEntraID"
    return f"{_hosted_ui_domain()}/oauth2/authorize?" + urllib.parse.urlencode(params)


def login_browser(
    *, port: int | None = None, no_browser: bool = False, native_chooser: bool = False
) -> dict:
    """Run the loopback PKCE flow and cache the resulting tokens."""
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(24)

    if no_browser:
        # SSH / headless: the redirect can't reach us, so the user pastes it back.
        # Same process, so the verifier is still in memory.
        redirect_uri = f"http://localhost:{LOOPBACK_PORTS[0]}{CALLBACK_PATH}"
        url = _authorize_url(redirect_uri, challenge, state, native=native_chooser)
        print("Open this URL in a browser:\n")
        print(f"  {url}\n")
        pasted = input("Paste the full URL you were redirected to: ").strip()
        query = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query)
        code = (query.get("code") or [""])[0]
        returned_state = (query.get("state") or [""])[0]
    else:
        server, bound_port = _bind_loopback(port)
        redirect_uri = f"http://localhost:{bound_port}{CALLBACK_PATH}"
        url = _authorize_url(redirect_uri, challenge, state, native=native_chooser)

        _CallbackHandler.result = {}
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"Opening your browser to sign in (listening on {redirect_uri}) ...")
        print(f"If it doesn't open, visit:\n  {url}\n")
        webbrowser.open(url)

        deadline = time.time() + 300
        while not _CallbackHandler.result and time.time() < deadline:
            time.sleep(0.2)
        server.shutdown()
        server.server_close()

        if not _CallbackHandler.result:
            raise AuthError("Timed out after 5 minutes waiting for the browser redirect.")
        if _CallbackHandler.result.get("error"):
            raise AuthError(f"Sign-in failed: {_CallbackHandler.result['error']}")
        code = _CallbackHandler.result.get("code", "")
        returned_state = _CallbackHandler.result.get("state", "")

    if not code:
        raise AuthError("No authorization code was returned.")
    # Guards against a co-resident process feeding us a code from another flow.
    if returned_state != state:
        raise AuthError("State mismatch on the redirect — aborting.")

    tokens = _token_endpoint(
        {
            "grant_type": "authorization_code",
            "client_id": _client_id(),
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        }
    )
    return _store_tokens(tokens)


# ---------------------------------------------------------------------------
# Native login: email + password (+ TOTP)
# ---------------------------------------------------------------------------
def login_native(email: str | None = None) -> dict:
    """Email/password login, prompting for a TOTP code when challenged."""
    email = email or input("Email: ").strip()
    # Read interactively only: a --password flag would land in shell history.
    password = getpass.getpass("Password: ")

    res = _idp(
        "InitiateAuth",
        {
            "AuthFlow": "USER_PASSWORD_AUTH",
            "ClientId": _client_id(),
            "AuthParameters": {"USERNAME": email, "PASSWORD": password},
        },
    )

    while "ChallengeName" in res:
        challenge = res["ChallengeName"]
        if challenge == "SOFTWARE_TOKEN_MFA":
            code = input("Authenticator code: ").strip()
            res = _idp(
                "RespondToAuthChallenge",
                {
                    "ClientId": _client_id(),
                    "ChallengeName": challenge,
                    "Session": res["Session"],
                    "ChallengeResponses": {
                        "USERNAME": email,
                        "SOFTWARE_TOKEN_MFA_CODE": code,
                    },
                },
            )
        elif challenge in ("NEW_PASSWORD_REQUIRED", "MFA_SETUP"):
            # Deliberately punted: the invite email sends users to the web app,
            # and re-implementing TOTP enrolment in a CLI isn't worth it.
            raise AuthError(
                f"Your account needs first-time setup ({challenge}). Finish signing "
                "in on the web app, then re-run `datasets login`."
            )
        else:
            raise AuthError(f"Unsupported authentication challenge: {challenge}")

    result = res.get("AuthenticationResult")
    if not result:
        raise AuthError("Cognito returned no tokens.")
    return _store_tokens(result)


# ---------------------------------------------------------------------------
# Refresh + accessor
# ---------------------------------------------------------------------------
def refresh(*, force: bool = False) -> dict:
    """Refresh the cached tokens; return the updated profile."""
    profile = _get_profile()
    token = profile.get("refresh_token")
    if not token:
        raise AuthError("Not signed in. Run: python -m compartment.datasets login")

    if not force and profile.get("expires_at", 0) - REFRESH_SKEW_SECONDS > time.time():
        return profile

    try:
        tokens = _token_endpoint(
            {
                "grant_type": "refresh_token",
                "client_id": _client_id(),
                "refresh_token": token,
            }
        )
    except AuthError:
        # Fall back to the IDP flow: the hosted-UI token endpoint can reject a
        # refresh token that was minted natively via InitiateAuth.
        try:
            res = _idp(
                "InitiateAuth",
                {
                    "AuthFlow": "REFRESH_TOKEN_AUTH",
                    "ClientId": _client_id(),
                    "AuthParameters": {"REFRESH_TOKEN": token},
                },
            )
            tokens = res.get("AuthenticationResult") or {}
            if not tokens:
                raise AuthError("no tokens")
        except AuthError:
            # Revoked, older than 30 days, or the password changed. Never fall
            # back to an api key -- failing closed is the point.
            _clear_profile()
            raise AuthError(
                "Session expired. Run: python -m compartment.datasets login"
            ) from None

    return _store_tokens(tokens)


def id_token() -> str:
    """Return a valid ID token, refreshing if it is close to expiry.

    Cognito re-evaluates group membership on refresh, so a newly-granted
    DISEASE_MODELER takes effect within the hour without a fresh login.
    """
    profile = _get_profile()
    if not profile.get("id_token"):
        raise AuthError("Not signed in. Run: python -m compartment.datasets login")
    if profile.get("expires_at", 0) - REFRESH_SKEW_SECONDS <= time.time():
        profile = refresh()
    return profile["id_token"]


def logout() -> bool:
    """Forget the cached tokens for the current environment."""
    return _clear_profile()


def whoami() -> dict | None:
    """Return the cached identity, or None when not signed in."""
    profile = _get_profile()
    if not profile.get("id_token"):
        return None
    return {
        "env": _env_name(),
        "email": profile.get("email", ""),
        "sub": profile.get("sub", ""),
        "groups": profile.get("groups") or [],
        "expires_at": profile.get("expires_at", 0),
        "expired": profile.get("expires_at", 0) <= time.time(),
    }


def can_publish() -> bool:
    """Local pre-flight only — the server is authoritative."""
    groups = set((_get_profile().get("groups") or []))
    allowed = {
        g.strip()
        for g in os.getenv(
            "DATASET_PUBLISH_GROUPS", "DISEASE_MODELER,SUPER_ADMIN"
        ).split(",")
    }
    return bool(groups & allowed)
