"""Offline unit tests for the datasets CLI auth module (no network).

Covers the properties that matter for security rather than the happy-path flow:
the token cache must not be readable by other users, PKCE must be correctly
derived, and the CLI must never silently fall back to unauthenticated access.

The interactive flows (browser SSO, password + TOTP) need a real Cognito pool and
are exercised in the dev round-trip, not here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import stat
import time

import pytest

from compartment.datasets import auth


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect the token cache into tmp so tests never touch a real login."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("DATASETS_ENV", "test")
    monkeypatch.setenv("COGNITO_WEB_CLIENT_ID", "test-client-id")
    yield


def _fake_id_token(claims: dict) -> str:
    """Build an unsigned JWT-shaped token; only the payload is ever decoded."""
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"header.{payload}.signature"


# ---------------------------------------------------------------------------
# Token cache
# ---------------------------------------------------------------------------
def test_cache_file_is_not_group_or_world_readable():
    """Credentials on disk must be 0600, and the directory 0700.

    A refresh token is a 30-day credential; on a shared machine a default umask
    would otherwise leave it readable by every other local account.
    """
    auth._store_tokens(
        {
            "IdToken": _fake_id_token({"sub": "s", "email": "a@b.c"}),
            "AccessToken": "a",
            "RefreshToken": "r",
            "ExpiresIn": 3600,
        }
    )

    cache = auth._cache_path()
    assert cache.exists()
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    assert stat.S_IMODE(cache.parent.stat().st_mode) == 0o700


def test_no_temp_file_is_left_behind():
    """The atomic write must not leave a readable partial file around."""
    auth._store_tokens({"IdToken": _fake_id_token({"sub": "s"}), "ExpiresIn": 3600})
    leftovers = list(auth._config_dir().glob(".credentials.*.tmp"))
    assert leftovers == []


def test_profiles_are_keyed_by_environment(monkeypatch):
    """dev/uat/prod tokens must coexist rather than clobbering each other."""
    monkeypatch.setenv("DATASETS_ENV", "dev")
    auth._store_tokens({"IdToken": _fake_id_token({"sub": "dev-user"}), "ExpiresIn": 3600})
    monkeypatch.setenv("DATASETS_ENV", "uat")
    auth._store_tokens({"IdToken": _fake_id_token({"sub": "uat-user"}), "ExpiresIn": 3600})

    monkeypatch.setenv("DATASETS_ENV", "dev")
    assert auth.whoami()["sub"] == "dev-user"
    monkeypatch.setenv("DATASETS_ENV", "uat")
    assert auth.whoami()["sub"] == "uat-user"


def test_refresh_token_is_preserved_across_refreshes():
    """A refresh_token grant returns no new refresh token — don't drop the old one."""
    auth._store_tokens(
        {
            "IdToken": _fake_id_token({"sub": "s"}),
            "RefreshToken": "original-refresh",
            "ExpiresIn": 3600,
        }
    )
    # Simulate a refresh response, which omits refresh_token.
    auth._store_tokens({"id_token": _fake_id_token({"sub": "s"}), "expires_in": 3600})
    assert auth._get_profile()["refresh_token"] == "original-refresh"


def test_logout_clears_only_the_current_environment(monkeypatch):
    monkeypatch.setenv("DATASETS_ENV", "dev")
    auth._store_tokens({"IdToken": _fake_id_token({"sub": "d"}), "ExpiresIn": 3600})
    monkeypatch.setenv("DATASETS_ENV", "uat")
    auth._store_tokens({"IdToken": _fake_id_token({"sub": "u"}), "ExpiresIn": 3600})

    assert auth.logout() is True
    assert auth.whoami() is None
    monkeypatch.setenv("DATASETS_ENV", "dev")
    assert auth.whoami() is not None


def test_corrupt_cache_does_not_crash():
    """A truncated credentials file should prompt a re-login, not a traceback."""
    auth._config_dir().mkdir(parents=True, exist_ok=True)
    auth._cache_path().write_text("{not json")
    assert auth.whoami() is None


# ---------------------------------------------------------------------------
# Claims decoding
# ---------------------------------------------------------------------------
def test_claims_are_decoded_for_display_only():
    token = _fake_id_token(
        {"sub": "abc", "email": "m@who.int", "cognito:groups": ["DISEASE_MODELER"]}
    )
    auth._store_tokens({"IdToken": token, "ExpiresIn": 3600})
    who = auth.whoami()
    assert who["email"] == "m@who.int"
    assert who["groups"] == ["DISEASE_MODELER"]


def test_malformed_token_yields_empty_claims():
    assert auth._decode_claims("not-a-jwt") == {}
    assert auth._decode_claims("") == {}


def test_can_publish_reflects_group_membership(monkeypatch):
    auth._store_tokens(
        {
            "IdToken": _fake_id_token({"sub": "s", "cognito:groups": ["USER"]}),
            "ExpiresIn": 3600,
        }
    )
    assert auth.can_publish() is False

    auth._store_tokens(
        {
            "IdToken": _fake_id_token({"sub": "s", "cognito:groups": ["DISEASE_MODELER"]}),
            "ExpiresIn": 3600,
        }
    )
    assert auth.can_publish() is True


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------
def test_pkce_challenge_is_s256_of_verifier():
    verifier, challenge = auth._pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected
    # Unpadded base64url, per RFC 7636.
    assert "=" not in verifier and "=" not in challenge
    assert len(verifier) >= 43


def test_pkce_pairs_are_unique():
    assert auth._pkce_pair()[0] != auth._pkce_pair()[0]


def test_authorize_url_targets_who_sso_by_default():
    url = auth._authorize_url("http://localhost:34981/oauth/callback", "chal", "st")
    assert "identity_provider=WhoEntraID" in url
    assert "code_challenge_method=S256" in url
    assert "response_type=code" in url
    # A public client must never carry a secret in the URL.
    assert "client_secret" not in url


def test_authorize_url_omits_idp_for_native_chooser():
    url = auth._authorize_url(
        "http://localhost:34981/oauth/callback", "chal", "st", native=True
    )
    assert "identity_provider" not in url


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
def test_id_token_without_login_raises_actionable_error():
    with pytest.raises(auth.AuthError) as ei:
        auth.id_token()
    assert "login" in str(ei.value)


def test_refresh_without_credentials_raises():
    with pytest.raises(auth.AuthError):
        auth.refresh()


def test_expired_token_triggers_refresh(monkeypatch):
    """An expired cached token must refresh rather than being sent as-is."""
    auth._store_tokens(
        {
            "IdToken": _fake_id_token({"sub": "s"}),
            "RefreshToken": "r",
            "ExpiresIn": 3600,
        }
    )
    profile = auth._get_profile()
    profile["expires_at"] = int(time.time()) - 10  # backdate
    auth._put_profile(profile)

    called = {"refreshed": False}

    def fake_refresh(**_kwargs):
        called["refreshed"] = True
        return auth._store_tokens(
            {"IdToken": _fake_id_token({"sub": "s"}), "ExpiresIn": 3600}
        )

    monkeypatch.setattr(auth, "refresh", fake_refresh)
    auth.id_token()
    assert called["refreshed"] is True


def test_only_registered_loopback_ports_are_accepted():
    """Cognito matches redirect URIs exactly, so unregistered ports cannot work."""
    with pytest.raises(auth.AuthError) as ei:
        auth._bind_loopback(8080)
    assert "must be one of" in str(ei.value)
    assert auth.LOOPBACK_PORTS == (34981, 34982, 34983)


# ---------------------------------------------------------------------------
# Single dataset catalog: the CLI target is fixed, not configured
# ---------------------------------------------------------------------------
class TestFixedTarget:
    """There is one dataset catalog, so the CLI must not depend on ambient config.

    Before this, the endpoint and client id came from environment variables. A
    stale value in a shell published to the wrong environment silently -- and
    since every published dataset is readable by every modeler, that is a visible
    mistake, not a private one. These tests pin the target down.
    """

    @pytest.fixture(autouse=True)
    def _no_overrides(self, monkeypatch):
        for var in (
            "COGNITO_WEB_CLIENT_ID",
            "WHO_COGNITO_CLIENT_ID",
            "COGNITO_HOSTED_UI_DOMAIN",
            "WHO_API_URL",
            "DATASETS_ENV",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_client_id_defaults_without_configuration(self):
        assert auth._client_id() == auth.DEFAULT_COGNITO_CLIENT_ID

    def test_hosted_ui_defaults_to_the_owning_environment(self):
        assert auth._hosted_ui_domain() == auth.DEFAULT_HOSTED_UI_DOMAIN
        assert auth.DATASETS_ENVIRONMENT in auth.DEFAULT_HOSTED_UI_DOMAIN

    def test_api_base_url_defaults_without_configuration(self):
        from compartment.datasets import api_client

        assert api_client.base_url() == auth.DEFAULT_API_URL
        assert auth.DATASETS_ENVIRONMENT in api_client.base_url()

    def test_no_missing_configuration_error_is_possible(self):
        """Neither accessor may raise for want of configuration.

        A modeler with a fresh checkout and no env file must be able to run
        `login` immediately.
        """
        auth._client_id()
        auth._hosted_ui_domain()

    def test_token_cache_profile_is_the_owning_environment(self):
        assert auth._env_name() == auth.DATASETS_ENVIRONMENT

    def test_overrides_still_work_for_local_development(self, monkeypatch):
        """Pointing at a local `next dev` must remain possible."""
        from compartment.datasets import api_client

        monkeypatch.setenv("WHO_API_URL", "http://localhost:3000/")
        monkeypatch.setenv("COGNITO_WEB_CLIENT_ID", "local-client")
        assert api_client.base_url() == "http://localhost:3000"  # trailing / trimmed
        assert auth._client_id() == "local-client"
