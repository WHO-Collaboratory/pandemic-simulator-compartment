"""Session-token handling for the dataset CLI.

There is no device-code flow here on purpose: the modeler copies a session
token out of the web app (Profile → Copy Session Token) and pastes it into the
CLI once. The token is cached under ``~/.pansim`` until it expires, so the
remaining subcommands run without prompting.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

try:
    import termios
except ImportError:  # Windows has no tty line discipline to reconfigure.
    termios = None

CACHE_DIR = Path(os.environ.get("PANSIM_HOME", Path.home() / ".pansim"))
CACHE_PATH = CACHE_DIR / "dataset-session.json"

WEBAPP_URL = os.environ.get("PANSIM_WEBAPP_URL", "https://uat.pandemic-simulator.com")

# Treat a token expiring within this window as already expired, so a long
# upload doesn't start on a token that dies mid-request.
_EXPIRY_SKEW_SECONDS = 120


class AuthError(Exception):
    """Raised when no usable session token can be obtained."""


def get_token(api) -> str:
    """Return a valid session token, prompting for one only when needed.

    ``api`` is a DatasetApi; it is used to validate a freshly pasted token
    against Cognito before the token is cached.
    """
    cached = _read_cache()
    if cached is not None:
        return cached

    token = _prompt_for_token()
    identity = api.whoami(token)
    _write_cache(token)
    print(f"Authenticated as {identity.get('email', 'unknown')}.", file=sys.stderr)
    return token


def clear_cache() -> None:
    CACHE_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------

def _prompt_for_token() -> str:
    profile_url = f"{WEBAPP_URL.rstrip('/')}/profile"
    print(
        "\nA session token is required.\n"
        f"  1. Sign in at {profile_url}\n"
        "  2. Click 'Copy Session Token' on your profile\n"
        "  3. Paste it below\n",
        file=sys.stderr,
    )

    # A headless or SSH session has no browser; failing to open one is not an
    # error, the URL is already printed above.
    try:
        webbrowser.open(profile_url)
    except Exception:
        pass

    print("Session token: ", end="", file=sys.stderr, flush=True)
    try:
        token = _read_pasted_line().strip()
    except (EOFError, KeyboardInterrupt):
        raise AuthError("No session token provided.")

    if not token:
        raise AuthError("No session token provided.")
    return token


def _read_pasted_line() -> str:
    """Read one pasted line of any length from the terminal.

    A session token runs past 1100 characters, and macOS caps a canonical-mode
    input line at MAX_CANON (1024 bytes): once that buffer fills the tty driver
    drops everything that follows, the newline included, so Enter does nothing
    until the user deletes a character. Reading with ICANON off has no cap.
    Piped or redirected stdin never went through the line discipline, so it
    takes the ordinary read.
    """
    if termios is None or not sys.stdin.isatty():
        return sys.stdin.readline()
    try:
        return _read_noncanonical(sys.stdin.fileno())
    except termios.error:
        # Some pseudo-terminals refuse tcsetattr; a capped read beats none.
        return sys.stdin.readline()


def _read_noncanonical(fd: int) -> str:
    """Collect characters until Enter with the tty's line discipline disabled.

    ECHO stays on so a paste is still visible, matching the old prompt. Losing
    ICANON means the driver no longer handles erase itself, hence the backspace
    branch below; it is enough for a fat-fingered paste, not a line editor.
    """
    saved = termios.tcgetattr(fd)
    mode = termios.tcgetattr(fd)
    mode[3] &= ~termios.ICANON  # lflag
    mode[6] = list(mode[6])  # cc
    mode[6][termios.VMIN] = 1  # block until at least one byte arrives
    mode[6][termios.VTIME] = 0

    chars: list[str] = []
    try:
        termios.tcsetattr(fd, termios.TCSANOW, mode)
        while True:
            char = sys.stdin.read(1)
            if char == "" or char in "\r\n":  # EOF or Enter
                break
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x04":  # Ctrl-D
                break
            if char in ("\x7f", "\b"):
                if chars:
                    chars.pop()
                continue
            chars.append(char)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        # Enter echoed a bare carriage return, leaving the cursor mid-line.
        print(file=sys.stderr)

    line = "".join(chars)
    # Terminals with bracketed paste enabled wrap the payload in markers.
    return line.replace("\x1b[200~", "").replace("\x1b[201~", "")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _read_cache() -> str | None:
    """Return the cached token if it is still comfortably in date."""
    try:
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    token = cached.get("token")
    expires_at = cached.get("expires_at", 0)
    if not token or expires_at - _EXPIRY_SKEW_SECONDS <= time.time():
        return None
    return token


def _write_cache(token: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "token": token,
        "expires_at": token_expiry(token),
    }), encoding="utf-8")
    # The token is a live credential — keep it off other users' radar.
    CACHE_PATH.chmod(0o600)


def token_expiry(token: str) -> int:
    """Epoch seconds at which the token expires, read from its `exp` claim.

    The signature is not checked here — the API validates the token against
    Cognito on every request. This only decides how long to keep reusing it.
    """
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        return int(json.loads(base64.urlsafe_b64decode(padded))["exp"])
    except Exception:
        raise AuthError(
            "That does not look like a session token. Copy it from the "
            "'Copy Session Token' button on your profile page."
        )
