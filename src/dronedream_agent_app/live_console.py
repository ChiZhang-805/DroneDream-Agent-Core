"""Start the installed DroneDream Core and open the account-bound console."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Event
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx

from .console import _doctor
from .console import main as console_main
from .console_client import ConsoleSession, ProductConsoleClient
from .console_ui import GREEN, TerminalUI

SUPABASE_URL = "https://yggabfynndpzymlqvnim.supabase.co"
PUBLISHABLE_KEY = "sb_publishable_wtY-RPU-YQasRjHIK30Zmw_U1bTMYqW"
AUTH_URL = f"{SUPABASE_URL}/auth/v1/oauth/authorize"
TOKEN_URL = f"{SUPABASE_URL}/auth/v1/oauth/token"
PASSWORD_TOKEN_URL = f"{SUPABASE_URL}/auth/v1/token?grant_type=password"
SIGNUP_URL = f"{SUPABASE_URL}/auth/v1/signup"
EXPECTED_ISSUER = f"{SUPABASE_URL}/auth/v1"

EDITION_AUTH = {
    "universal": ("24cf4e9d-527e-42eb-abd9-5395d579b8fc", 49210),
    "sim": ("0c2ad943-a0cb-4a2f-9eda-eba44b7f58df", 49211),
    "lab": ("0b9e7a8d-2c90-4b76-8842-511363f555bd", 49212),
    "field": ("3140bbe2-5f0e-4699-8a9b-295d4030f853", 49213),
    "autonomy": ("b5a93026-52f4-4bb7-8428-51c7bc30ebd1", 49214),
}
EDITION_DIRECTORIES = {
    "DroneDream-Universal": "universal",
    "DroneDream-Sim": "sim",
    "DroneDream-Lab": "lab",
    "DroneDream-Field": "field",
    "DroneDream-Agent": "autonomy",
    "DroneDream-AGENT": "autonomy",
}


class LiveConsoleError(RuntimeError):
    """A short, user-facing startup error."""


class LiveConsoleExit(RuntimeError):
    """The operator intentionally left the account gate."""


@dataclass(frozen=True)
class InstalledProduct:
    root: Path
    core: Path
    isolator: Path
    resources: Path
    edition: str


@dataclass(frozen=True)
class AccountProfile:
    token: str
    owner_key: str
    display_name: str
    email: str


def _registered_install_roots() -> list[Path]:
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []

    roots: list[Path] = []
    uninstall = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    for product_name in EDITION_DIRECTORIES:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, f"{uninstall}\\{product_name}") as key:
                value, _ = winreg.QueryValueEx(key, "InstallLocation")
        except OSError:
            continue
        location = str(value).strip().strip('"')
        if location:
            roots.append(Path(location))
    return roots


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for variable in ("DRONEDREAM_INSTALL_ROOT", "DRONEDREAM_AGENT_INSTALL_ROOT"):
        override = os.environ.get(variable, "").strip()
        if override:
            roots.append(Path(override))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.extend(_registered_install_roots())
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        roots.extend(Path(local_app_data) / name for name in EDITION_DIRECTORIES)
    return list(dict.fromkeys(roots))


def _edition_for_root(root: Path) -> str:
    edition = EDITION_DIRECTORIES.get(root.name)
    if edition:
        return edition
    requested = os.environ.get("DRONEDREAM_CONSOLE_EDITION", "").strip().lower()
    return requested if requested in EDITION_AUTH else "autonomy"


def _installed_product(*, timeout_seconds: float = 20.0) -> InstalledProduct:
    """Resolve one complete install, waiting through an in-progress installer swap."""

    deadline = time.monotonic() + timeout_seconds
    roots = _candidate_roots()
    while True:
        for root in roots:
            core = root / "dronedream-autonomy-core.exe"
            resource_candidates = (root / "agent-core", root)
            isolator_candidates = (
                root / "agent-core" / "dronedream-plugin-isolator.exe",
                root / "dronedream-plugin-isolator.exe",
            )
            resources = next(
                (
                    item
                    for item in resource_candidates
                    if (item / "default-assets" / "index.json").is_file()
                    and (item / "official-plugins" / "index.json").is_file()
                    and (item / "runtime" / "runtime-manifest.json").is_file()
                ),
                None,
            )
            isolator = next((item for item in isolator_candidates if item.is_file()), None)
            if core.is_file() and resources is not None and isolator is not None:
                return InstalledProduct(root, core, isolator, resources, _edition_for_root(root))
        if time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    searched = ", ".join(str(path) for path in roots) or "the current user profile"
    raise LiveConsoleError(
        "No complete DroneDream installation was found. "
        f"Repair the installed edition, then try again. Checked: {searched}"
    )


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _decode_claims(token: str) -> dict:
    segments = token.split(".")
    if len(segments) != 3:
        raise LiveConsoleError("The account service returned an invalid sign-in session.")
    payload = segments[1] + "=" * (-len(segments[1]) % 4)
    try:
        value = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise LiveConsoleError(
            "The account service returned an invalid sign-in session."
        ) from error
    if not isinstance(value, dict):
        raise LiveConsoleError("The account service returned an invalid sign-in session.")
    return value


def _validate_tokens(payload: dict, nonce: str, client_id: str) -> str:
    access_token = str(payload.get("access_token", ""))
    id_token = str(payload.get("id_token", ""))
    access = _decode_claims(access_token)
    identity = _decode_claims(id_token)
    audience = identity.get("aud")
    audience_matches = audience == client_id or (
        isinstance(audience, list) and client_id in audience
    )
    now = int(time.time())
    if (
        not access.get("sub")
        or access.get("sub") != identity.get("sub")
        or access.get("client_id") != client_id
        or access.get("iss") != EXPECTED_ISSUER
        or identity.get("iss") != EXPECTED_ISSUER
        or identity.get("nonce") != nonce
        or not audience_matches
        or int(access.get("exp", 0)) <= now
        or int(identity.get("exp", 0)) <= now
    ):
        raise LiveConsoleError("The browser sign-in did not match this DroneDream edition.")
    return access_token


def _profile_from_token(token: str) -> AccountProfile:
    """Derive display data and an opaque local storage key from one account token."""
    try:
        claims = _decode_claims(token)
    except LiveConsoleError:
        claims = {}
    subject = str(claims.get("sub", "")).strip()
    email = str(claims.get("email", "")).strip()
    metadata = claims.get("user_metadata")
    candidates: list[object] = []
    if isinstance(metadata, dict):
        candidates.extend(
            metadata.get(key) for key in ("display_name", "full_name", "name", "user_name")
        )
    candidates.extend((claims.get("preferred_username"), email.partition("@")[0]))
    display_name = next(
        (
            str(value).strip()
            for value in candidates
            if isinstance(value, str)
            and value.strip()
            and not any(ord(char) < 32 for char in value)
        ),
        "Pilot",
    )[:32]
    owner_material = subject or token
    owner_key = hashlib.sha256(owner_material.encode("utf-8")).hexdigest()[:24]
    return AccountProfile(token, owner_key, display_name, email)


def _validate_password_token(payload: dict) -> str:
    token = str(payload.get("access_token", ""))
    claims = _decode_claims(token)
    now = int(time.time())
    if (
        not claims.get("sub")
        or claims.get("iss") != EXPECTED_ISSUER
        or int(claims.get("exp", 0)) <= now
    ):
        raise LiveConsoleError("The account service returned an invalid sign-in session.")
    return token


def _auth_request(url: str, body: dict) -> dict:
    try:
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            response = client.post(
                url,
                headers={"apikey": PUBLISHABLE_KEY},
                json=body,
                timeout=20,
            )
        if not response.is_success:
            raise LiveConsoleError(
                "The email or password was not accepted. Check the details and try again."
            )
        payload = response.json()
    except LiveConsoleError:
        raise
    except (httpx.HTTPError, ValueError) as error:
        raise LiveConsoleError("DroneDream could not reach the account service.") from error
    if not isinstance(payload, dict):
        raise LiveConsoleError("The account service returned an invalid response.")
    return payload


def _read_masked_password(label: str, *, get_character, stream) -> str:
    """Read one secret while showing only a star for each entered character."""
    print(label, end="", flush=True, file=stream)
    characters: list[str] = []
    while True:
        character = get_character()
        if character in {"\r", "\n"}:
            print(file=stream, flush=True)
            return "".join(characters)
        if character == "\x03":
            print(file=stream, flush=True)
            raise KeyboardInterrupt
        if character in {"\b", "\x7f"}:
            if characters:
                characters.pop()
                print("\b \b", end="", flush=True, file=stream)
            continue
        if character in {"\x00", "\xe0"}:
            get_character()
            continue
        if character.isprintable():
            characters.append(character)
            print("*", end="", flush=True, file=stream)


def _read_password(label: str, *, stream=None) -> str:
    output = stream or sys.stdout
    if os.name == "nt" and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        import msvcrt

        return _read_masked_password(label, get_character=msvcrt.getwch, stream=output)
    return getpass.getpass(label, stream=output)


def _email_password_sign_in(*, stdin, ui: TerminalUI) -> str:
    ui.prompt("Email")
    email = stdin.readline().strip()
    password = _read_password("  › Password  ", stream=ui.stream)
    if not email or len(password) < 8:
        raise LiveConsoleError(
            "Enter a valid email address and a password of at least 8 characters."
        )
    payload = _auth_request(PASSWORD_TOKEN_URL, {"email": email, "password": password})
    return _validate_password_token(payload)


def _email_password_register(*, stdin, ui: TerminalUI) -> str | None:
    ui.prompt("Display name")
    display_name = stdin.readline().strip()
    ui.prompt("Email")
    email = stdin.readline().strip()
    password = _read_password("  › Password  ", stream=ui.stream)
    confirmation = _read_password("  › Confirm password  ", stream=ui.stream)
    if not display_name or not email or len(password) < 8:
        raise LiveConsoleError(
            "Enter a display name, a valid email address, and a password of at least 8 characters."
        )
    if password != confirmation:
        raise LiveConsoleError("The two passwords do not match.")
    payload = _auth_request(
        SIGNUP_URL,
        {"email": email, "password": password, "data": {"display_name": display_name[:80]}},
    )
    token = str(payload.get("access_token", ""))
    if token:
        return _validate_password_token(payload)
    ui.panel(
        "CHECK YOUR EMAIL",
        [
            "Your DroneDream account is ready for confirmation.",
            "Open the confirmation link, then choose Sign in here.",
        ],
        accent=GREEN,
    )
    return None


def _choose_account(edition: str, *, stdin, stdout, no_color: bool) -> AccountProfile:
    injected = os.environ.get("DRONEDREAM_CONSOLE_IDENTITY_TOKEN", "").strip()
    if injected:
        return _profile_from_token(injected)

    ui = TerminalUI(stream=stdout, color=False if no_color else None)
    ui.banner()
    while True:
        ui.rule("ACCOUNT")
        print(file=stdout)
        ui.menu(
            "WELCOME TO DRONEDREAM",
            [
                ("L", "Sign in", "email and password"),
                ("R", "Create account", "register a new pilot"),
                ("B", "Browser sign-in", "use a connected provider"),
                ("Q", "Quit", "return to the system terminal"),
            ],
        )
        ui.prompt("Choose", "L / R / B / Q")
        choice = stdin.readline().strip().casefold()
        try:
            if choice in {"l", "login", "sign-in", "signin", "1"}:
                token = _email_password_sign_in(stdin=stdin, ui=ui)
            elif choice in {"r", "register", "sign-up", "signup", "2"}:
                token = _email_password_register(stdin=stdin, ui=ui)
                if token is None:
                    print(file=stdout)
                    continue
            elif choice in {"b", "browser", "3"}:
                token = _browser_identity_token(edition)
            elif choice in {"q", "quit", "exit", "4"}:
                raise LiveConsoleExit()
            else:
                ui.error("ACCOUNT_CHOICE_INVALID", "Choose L, R, B, or Q.")
                print(file=stdout)
                continue
        except LiveConsoleError as error:
            ui.error("ACCOUNT_SIGN_IN_FAILED", str(error))
            print(file=stdout)
            continue
        profile = _profile_from_token(token)
        ui.status("Account", "ready", profile.display_name)
        print(file=stdout)
        return profile


def _browser_identity_token(edition: str) -> str:
    existing = os.environ.get("DRONEDREAM_CONSOLE_IDENTITY_TOKEN", "").strip()
    if existing:
        return existing

    client_id, callback_port = EDITION_AUTH[edition]
    callback_path = f"/desktop-auth/{edition}/callback"
    callback_url = f"http://127.0.0.1:{callback_port}{callback_path}"
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(48)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": callback_url,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": "openid email profile",
            "nonce": nonce,
        }
    )
    received: dict[str, str] = {}
    finished = Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            fields = parse_qs(parsed.query, keep_blank_values=True)
            if parsed.path != callback_path:
                self.send_error(404)
                return
            if fields.get("state", [""])[0] != state:
                received["error"] = "The browser sign-in state did not match."
            elif fields.get("error"):
                received["error"] = "Browser sign-in was cancelled."
            else:
                received["code"] = fields.get("code", [""])[0]
            body = (
                b"<!doctype html><meta charset=utf-8><title>DroneDream</title>"
                b"<style>body{font:18px system-ui;display:grid;place-items:center;min-height:90vh;"
                b"background:#fff7fc;color:#171526}b{color:#d92faa}</style>"
                b"<p><b>DroneDream</b> sign-in complete. You may return to the terminal.</p>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            finished.set()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    try:
        server = HTTPServer(("127.0.0.1", callback_port), CallbackHandler)
    except OSError as error:
        raise LiveConsoleError("Another DroneDream sign-in is already open.") from error
    server.timeout = 0.25
    print("  Opening your browser to sign in to DroneDream…", flush=True)
    webbrowser.open(f"{AUTH_URL}?{query}")
    deadline = time.monotonic() + 600
    try:
        while not finished.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if not finished.is_set():
        raise LiveConsoleError("Browser sign-in timed out. Run dronedream-live to try again.")
    if received.get("error"):
        raise LiveConsoleError(received["error"])
    code = received.get("code", "")
    if not code:
        raise LiveConsoleError("The browser did not return a sign-in code.")
    try:
        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": client_id,
                "redirect_uri": callback_url,
                "code_verifier": verifier,
            },
            timeout=20,
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise LiveConsoleError("DroneDream could not complete browser sign-in.") from error
    if not isinstance(payload, dict):
        raise LiveConsoleError("The account service returned an invalid sign-in session.")
    return _validate_tokens(payload, nonce, client_id)


def _wait_for_core(
    process: subprocess.Popen[bytes], base_url: str, timeout_seconds: float = 90
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LiveConsoleError(
                "DroneDream Core stopped during startup. Repair AGENT and try again."
            )
        try:
            response = httpx.get(f"{base_url}/health", timeout=1, follow_redirects=False)
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise LiveConsoleError("DroneDream Core did not become ready within 90 seconds.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-only", action="store_true")
    return parser


def _client(session: dict[str, str]) -> ProductConsoleClient:
    return ProductConsoleClient(
        ConsoleSession(
            core_url=session["DRONEDREAM_CONSOLE_CORE_URL"],
            supabase_url=session["DRONEDREAM_CONSOLE_SUPABASE_URL"],
            local_token=session["DRONEDREAM_CONSOLE_LOCAL_TOKEN"],
            identity_token=session["DRONEDREAM_CONSOLE_IDENTITY_TOKEN"],
            publishable_key=session["DRONEDREAM_CONSOLE_PUBLISHABLE_KEY"],
            source_edition=session["DRONEDREAM_CONSOLE_EDITION"],
        )
    )


def _ensure_runtime(session: dict[str, str]) -> dict:
    client = _client(session)
    try:
        status = client.runtime()
        ready = all(
            bool(status.get(field))
            for field in ("runtime_available", "resources_ready", "provisioned")
        )
        if not ready:
            print("  Preparing the DroneDream Runtime for its first flight…", flush=True)
            client.provision_runtime()
        result = _doctor(client)
    finally:
        client.close()
    if not result.get("ready"):
        blocked = "; ".join(
            f"{item.get('name')}: {item.get('detail')}"
            for item in result.get("checks", [])
            if item.get("status") != "ready"
        )
        raise LiveConsoleError(f"DroneDream setup did not complete: {blocked}")
    return result


def _stop_core(process: subprocess.Popen[bytes], base_url: str, local_token: str) -> None:
    if process.poll() is not None:
        return
    try:
        httpx.post(
            f"{base_url}/shutdown",
            headers={"Authorization": f"Bearer {local_token}"},
            timeout=10,
            follow_redirects=False,
        )
        process.wait(timeout=10)
        return
    except (httpx.HTTPError, subprocess.TimeoutExpired):
        pass
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    """Open a self-contained live session against the installed AGENT product."""

    args, console_arguments = _parser().parse_known_args(argv)
    process: subprocess.Popen[bytes] | None = None
    log_handle = None
    base_url = ""
    local_token = ""
    previous = {
        name: os.environ.get(name)
        for name in (
            "DRONEDREAM_CONSOLE_CORE_URL",
            "DRONEDREAM_CONSOLE_SUPABASE_URL",
            "DRONEDREAM_CONSOLE_LOCAL_TOKEN",
            "DRONEDREAM_CONSOLE_IDENTITY_TOKEN",
            "DRONEDREAM_CONSOLE_PUBLISHABLE_KEY",
            "DRONEDREAM_CONSOLE_EDITION",
            "DRONEDREAM_CONSOLE_ACCOUNT_NAME",
            "DRONEDREAM_CONSOLE_ACCOUNT_EMAIL",
            "DRONEDREAM_CONSOLE_ACCOUNT_KEY",
            "DRONEDREAM_CONSOLE_BANNER_SHOWN",
            "FORCE_COLOR",
            "NO_COLOR",
        )
    }
    try:
        product = _installed_product()
        if args.check_only:
            print(f"READY ({product.edition})")
            return 0
        no_color = "--no-color" in console_arguments
        account = _choose_account(
            product.edition,
            stdin=sys.stdin,
            stdout=sys.stdout,
            no_color=no_color,
        )
        identity_token = account.token
        port = _reserve_port()
        base_url = f"http://127.0.0.1:{port}"
        local_token = secrets.token_hex(32)
        local_app_data = Path(os.environ.get("LOCALAPPDATA", product.root))
        data_root = (
            local_app_data / "DroneDream-CLI" / product.edition / "accounts" / account.owner_key
        )
        data_root.mkdir(parents=True, exist_ok=True)
        log_handle = (data_root / "core.log").open("ab")
        command = [
            str(product.core),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--token",
            local_token,
            "--data-root",
            str(data_root / "core-data"),
            "--resource-root",
            str(product.resources),
            "--plugin-isolator",
            str(product.isolator),
            "--account-memory-db",
            str(data_root / "account-memory.db"),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        _wait_for_core(process, base_url)
        session = {
            "DRONEDREAM_CONSOLE_CORE_URL": base_url,
            "DRONEDREAM_CONSOLE_SUPABASE_URL": SUPABASE_URL,
            "DRONEDREAM_CONSOLE_LOCAL_TOKEN": local_token,
            "DRONEDREAM_CONSOLE_IDENTITY_TOKEN": identity_token,
            "DRONEDREAM_CONSOLE_PUBLISHABLE_KEY": PUBLISHABLE_KEY,
            "DRONEDREAM_CONSOLE_EDITION": product.edition,
            "DRONEDREAM_CONSOLE_ACCOUNT_NAME": account.display_name,
            "DRONEDREAM_CONSOLE_ACCOUNT_EMAIL": account.email,
            "DRONEDREAM_CONSOLE_ACCOUNT_KEY": account.owner_key,
            "DRONEDREAM_CONSOLE_BANNER_SHOWN": "1",
        }
        os.environ.update(session)
        if "--no-color" not in console_arguments:
            os.environ.pop("NO_COLOR", None)
            os.environ["FORCE_COLOR"] = "1"
        _ensure_runtime(session)
        if not console_arguments:
            console_arguments = ["shell"]
        return console_main(console_arguments)
    except LiveConsoleExit:
        return 0
    except LiveConsoleError as error:
        print(f"\033[38;2;255;82;111mDroneDream could not start:\033[0m {error}", file=sys.stderr)
        return 1
    finally:
        if process is not None and base_url and local_token:
            _stop_core(process, base_url, local_token)
        if log_handle is not None:
            log_handle.close()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
