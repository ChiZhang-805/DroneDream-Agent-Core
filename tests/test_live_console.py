import base64
import hashlib
import io
import json
import threading
import time

import pytest

from dronedream_agent_app import live_console


def _file(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _installation(root):
    _file(root / "dronedream-autonomy-core.exe")
    _file(root / "agent-core" / "dronedream-plugin-isolator.exe")
    _file(root / "agent-core" / "default-assets" / "index.json")
    _file(root / "agent-core" / "official-plugins" / "index.json")
    _file(root / "agent-core" / "runtime" / "runtime-manifest.json")


def test_installed_product_prefers_complete_nested_distribution(tmp_path, monkeypatch):
    root = tmp_path / "DroneDream-Agent"
    _installation(root)
    monkeypatch.setattr(live_console, "_candidate_roots", lambda: [root])

    product = live_console._installed_product(timeout_seconds=0)

    assert product.root == root
    assert product.resources == root / "agent-core"
    assert product.isolator == root / "agent-core" / "dronedream-plugin-isolator.exe"
    assert product.edition == "autonomy"


@pytest.mark.parametrize(
    ("directory", "edition"),
    [
        ("DroneDream-Universal", "universal"),
        ("DroneDream-Sim", "sim"),
        ("DroneDream-Lab", "lab"),
        ("DroneDream-Field", "field"),
        ("DroneDream-Agent", "autonomy"),
    ],
)
def test_installed_product_recognizes_every_edition(tmp_path, monkeypatch, directory, edition):
    root = tmp_path / directory
    _installation(root)
    monkeypatch.setattr(live_console, "_candidate_roots", lambda: [root])

    assert live_console._installed_product(timeout_seconds=0).edition == edition


def test_installed_product_skips_an_incomplete_launcher_directory(tmp_path, monkeypatch):
    launcher = tmp_path / "Python" / "Scripts"
    launcher.mkdir(parents=True)
    product = tmp_path / "DroneDream-Sim"
    _installation(product)
    monkeypatch.setattr(live_console, "_candidate_roots", lambda: [launcher, product])

    assert live_console._installed_product(timeout_seconds=0).root == product


def test_installed_product_waits_through_atomic_installer_swap(tmp_path, monkeypatch):
    root = tmp_path / "DroneDream-Agent"
    monkeypatch.setattr(live_console, "_candidate_roots", lambda: [root])

    worker = threading.Thread(target=lambda: (time.sleep(0.1), _installation(root)))
    worker.start()
    try:
        product = live_console._installed_product(timeout_seconds=1)
    finally:
        worker.join()

    assert product.core.is_file()
    assert product.isolator.is_file()


def test_installed_product_reports_repair_without_a_python_traceback(tmp_path, monkeypatch):
    root = tmp_path / "DroneDream-Agent"
    monkeypatch.setattr(live_console, "_candidate_roots", lambda: [root])

    with pytest.raises(live_console.LiveConsoleError, match="Repair the installed edition"):
        live_console._installed_product(timeout_seconds=0)


def _jwt(payload):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"e30.{encoded}.signature"


def test_browser_tokens_are_bound_to_the_selected_client(monkeypatch):
    monkeypatch.setattr(live_console.time, "time", lambda: 1_000)
    client_id = live_console.EDITION_AUTH["sim"][0]
    common = {"sub": "user-1", "iss": live_console.EXPECTED_ISSUER, "exp": 2_000}
    payload = {
        "access_token": _jwt({**common, "client_id": client_id}),
        "id_token": _jwt(
            {
                **common,
                "aud": client_id,
                "nonce": "nonce-1",
            }
        ),
    }

    assert live_console._validate_tokens(payload, "nonce-1", client_id) == payload["access_token"]


def test_first_launch_provisions_runtime_before_opening_console(monkeypatch):
    calls = []

    class Client:
        def runtime(self):
            return {"runtime_available": True, "resources_ready": True, "provisioned": False}

        def provision_runtime(self):
            calls.append("provision")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(live_console, "_client", lambda _session: Client())
    monkeypatch.setattr(live_console, "_doctor", lambda _client: {"ready": True, "checks": []})

    assert live_console._ensure_runtime({})["ready"] is True
    assert calls == ["provision", "close"]


def test_core_shutdown_uses_authenticated_product_endpoint(monkeypatch):
    calls = []

    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            calls.append(("wait", timeout))

    def post(url, **options):
        calls.append((url, options))

    monkeypatch.setattr(live_console.httpx, "post", post)
    live_console._stop_core(Process(), "http://127.0.0.1:43123", "local-token")

    assert calls[0] == (
        "http://127.0.0.1:43123/shutdown",
        {
            "headers": {"Authorization": "Bearer local-token"},
            "timeout": 10,
            "follow_redirects": False,
        },
    )
    assert calls[1] == ("wait", 10)


def test_bare_live_command_opens_interactive_shell(tmp_path, monkeypatch):
    product = live_console.InstalledProduct(
        root=tmp_path,
        core=tmp_path / "dronedream-autonomy-core.exe",
        isolator=tmp_path / "dronedream-plugin-isolator.exe",
        resources=tmp_path,
        edition="lab",
    )
    received = []

    class Process:
        pass

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(live_console, "_installed_product", lambda: product)
    account = live_console.AccountProfile(
        token="identity-token",
        owner_key="account-key",
        display_name="Test Pilot",
        email="pilot@example.com",
    )
    monkeypatch.setattr(live_console, "_choose_account", lambda *args, **kwargs: account)
    monkeypatch.setattr(live_console, "_reserve_port", lambda: 43123)
    monkeypatch.setattr(live_console.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(live_console, "_wait_for_core", lambda *args, **kwargs: None)
    monkeypatch.setattr(live_console, "_ensure_runtime", lambda _session: {"ready": True})
    monkeypatch.setattr(live_console, "_stop_core", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        live_console,
        "console_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert live_console.main([]) == 0
    assert received == [["shell"]]


def test_account_gate_accepts_letter_login_and_derives_account_profile(monkeypatch):
    token = _jwt(
        {
            "sub": "account-123",
            "email": "pilot@example.com",
            "user_metadata": {"display_name": "Sky Pilot"},
        }
    )
    monkeypatch.delenv("DRONEDREAM_CONSOLE_IDENTITY_TOKEN", raising=False)
    monkeypatch.setattr(
        live_console,
        "_email_password_sign_in",
        lambda **_kwargs: token,
    )
    output = io.StringIO()

    account = live_console._choose_account(
        "autonomy",
        stdin=io.StringIO("L\n"),
        stdout=output,
        no_color=True,
    )

    assert account.display_name == "Sky Pilot"
    assert account.email == "pilot@example.com"
    assert account.owner_key == hashlib.sha256(b"account-123").hexdigest()[:24]
    assert "Sign in" in output.getvalue()
    assert "Create account" in output.getvalue()


def test_account_gate_quit_is_a_clean_operator_exit(monkeypatch):
    monkeypatch.delenv("DRONEDREAM_CONSOLE_IDENTITY_TOKEN", raising=False)
    with pytest.raises(live_console.LiveConsoleExit):
        live_console._choose_account(
            "sim",
            stdin=io.StringIO("Q\n"),
            stdout=io.StringIO(),
            no_color=True,
        )


def test_password_registration_sends_display_name_without_echoing_password(monkeypatch):
    token = _jwt(
        {
            "sub": "new-account",
            "iss": live_console.EXPECTED_ISSUER,
            "exp": int(live_console.time.time()) + 3600,
        }
    )
    observed = {}
    passwords = iter(["long-password", "long-password"])
    monkeypatch.setattr(
        live_console,
        "_read_password",
        lambda _label, **_kwargs: next(passwords),
    )
    monkeypatch.setattr(
        live_console,
        "_auth_request",
        lambda url, body: observed.update(url=url, body=body) or {"access_token": token},
    )
    ui = live_console.TerminalUI(stream=io.StringIO(), color=False, width=90)

    result = live_console._email_password_register(
        stdin=io.StringIO("Avery\npilot@example.com\n"),
        ui=ui,
    )

    assert result == token
    assert observed["url"] == live_console.SIGNUP_URL
    assert observed["body"] == {
        "email": "pilot@example.com",
        "password": "long-password",
        "data": {"display_name": "Avery"},
    }


def test_masked_password_shows_stars_and_erases_one_star_on_backspace():
    keys = iter(["s", "e", "x", "\b", "c", "r", "e", "t", "\r"])
    output = io.StringIO()

    password = live_console._read_masked_password(
        "Password: ", get_character=lambda: next(keys), stream=output
    )

    assert password == "secret"
    assert output.getvalue() == "Password: ***\b \b****\n"
    assert password not in output.getvalue()


def test_check_only_validates_install_without_sign_in(tmp_path, monkeypatch, capsys):
    product = live_console.InstalledProduct(
        root=tmp_path,
        core=tmp_path / "dronedream-autonomy-core.exe",
        isolator=tmp_path / "dronedream-plugin-isolator.exe",
        resources=tmp_path,
        edition="sim",
    )
    monkeypatch.setattr(live_console, "_installed_product", lambda: product)
    monkeypatch.setattr(
        live_console,
        "_browser_identity_token",
        lambda _edition: pytest.fail("health check must not open browser sign-in"),
    )
    monkeypatch.setattr(
        live_console.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("health check must not start Core"),
    )

    assert live_console.main(["--check-only"]) == 0
    assert capsys.readouterr().out.strip() == "READY (sim)"
