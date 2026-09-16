"""Console contract/fault tests, not paid-model or physical-flight qualification.

The API integration cases use the real desktop router/store. Only identity,
cloud transport and the slow planner are explicit test doubles.
"""

from __future__ import annotations

import io
import json
import tomllib
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from dronedream_agent_app import console
from dronedream_agent_app.console_client import (
    ConsoleError,
    ConsoleSession,
    ProductConsoleClient,
)
from dronedream_agent_app.models import ThreadCreate

THREAD = "thread-" + "1" * 32
PLAN = "plan-" + "2" * 32
CLOUD = "https://" + "a" * 20 + ".supabase.co"
GATEWAY = CLOUD + "/functions/v1/model-gateway"
GRANT = "ddg_" + "g" * 40
SESSION = {
    "core_url": "http://127.0.0.1:9017",
    "supabase_url": CLOUD,
    "local_token": "local-" + "x" * 40,
    "identity_token": "identity-" + "x" * 40,
    "publishable_key": "public-" + "x" * 40,
    "source_edition": "autonomy",
}
SELECTION = {
    "message": "从办公室出发，到取件处悬停十秒，再返回。",
    "map_id": "selected-map",
    "map_content_sha256": "a" * 64,
    "vehicle_id": "selected-vehicle",
    "vehicle_content_sha256": "b" * 64,
}


# 功能：
#   创建显式测试传输和内存会话；不连接账户、供应商或飞控。
# 输入：
#   handler：接收 HTTP 请求并提供夹具响应的函数。
# 输出：
#   client：使用测试传输的产品客户端。
def make_client(handler) -> ProductConsoleClient:
    client = ProductConsoleClient(
        ConsoleSession.model_validate(SESSION),
        transport=httpx.MockTransport(handler),
    )
    return client


# 功能：
#   提供合法计划与模型目录夹具，并保存请求以核对端点和授权边界。
# 输入：
#   request：被测试客户端发出的 HTTP 请求。
#   calls：测试请求列表。
# 输出：
#   response：没有真实模型和飞行行为的明确夹具响应。
def contract_response(request: httpx.Request, calls: list) -> httpx.Response:
    calls.append(request)
    path = request.url.path
    if path == "/v1/threads/" + THREAD:
        data = {
            "thread_id": THREAD,
            "selected_model": "kimi-k2.6",
            "locale": "zh-CN",
            "state": "awaiting_confirmation",
            "messages": [
                {"kind": "plan", "role": "assistant", "metadata": {"plan_revision_id": PLAN}},
            ],
        }
    elif path == "/v1/models":
        data = {
            "models": [
                {"id": "kimi-k2.6", "provider": "kimi", "model": "kimi-k2.6", "source": "default"}
            ]
        }
    elif path.endswith("/grants"):
        data = {"data": {"grant": GRANT, "gateway_base_url": GATEWAY}}
    elif path.endswith("/prepare"):
        data = {"plan_revision_id": PLAN, "status": "prepared", "model_calls": 3}
    elif path.endswith("/execute"):
        data = {"status": "starting", "execution_id": "fixture-execution"}
    elif path.endswith("/usage"):
        data = {"data": {"usage": {"request_count": 3, "remaining_ai_credits": 21}}}
    else:
        data = {"status": "fixture-response"}
    return httpx.Response(200, json=data)


@pytest.mark.parametrize(
    "core_url",
    [
        "https://127.0.0.1:9017",
        "http://localhost:9017",
        "http://127.0.0.1:9017/evil",
        "http://127.0.0.1:9017?x=1",
        "http://127.0.0.1:9017#x",
        "http://127.0.0.1",
        "http://user@127.0.0.1:9017",
        "http://127.0.0.1:0",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:09017",
        "http://example.org:9017",
    ],
)
def test_local_credentials_cannot_be_routed_outside_fixed_origin(core_url):
    with pytest.raises(ValidationError):
        ConsoleSession.model_validate({**SESSION, "core_url": core_url})


@pytest.mark.parametrize(
    "value",
    [
        "http://" + "a" * 20 + ".supabase.co",
        CLOUD + "/evil",
        CLOUD + "?x=1",
        CLOUD + ".example.org",
        "https://example.org",
    ],
)
def test_cloud_origin_is_explicit_and_not_redirectable(value):
    with pytest.raises(ValidationError):
        ConsoleSession.model_validate({**SESSION, "supabase_url": value})


@pytest.mark.parametrize("field", ["local_token", "identity_token", "publishable_key"])
def test_secret_fields_reject_header_injection_and_hide_repr(field):
    with pytest.raises(ValidationError):
        ConsoleSession.model_validate({**SESSION, field: "secret\r\nInjected: header"})
    session = ConsoleSession.model_validate(SESSION)
    assert SESSION[field] not in repr(session)


def test_prepare_routes_through_cloud_grant_and_desktop_api_without_execution():
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    try:
        result = client.prepare(THREAD, SELECTION)
        assert result["plan_revision_id"] == PLAN
        assert [req.url.path for req in calls] == [
            "/v1/threads/" + THREAD,
            "/v1/models",
            "/functions/v1/model-gateway/grants",
            "/v1/threads/" + THREAD + "/prepare",
        ]
        cloud, local = calls[-2:]
        assert cloud.headers["authorization"] == "Bearer " + SESSION["identity_token"]
        assert "x-dronedream-identity-token" not in cloud.headers
        assert SESSION["local_token"] not in str(cloud.headers)
        assert json.loads(cloud.content) == {
            "scope": "assistant",
            "scope_reference": THREAD,
            "provider": "kimi",
            "model": "kimi-k2.6",
        }
        assert local.headers["authorization"] == "Bearer " + SESSION["local_token"]
        assert local.headers["x-dronedream-identity-token"] == SESSION["identity_token"]
        payload = json.loads(local.content)
        assert payload["message"] == SELECTION["message"]
        assert payload["map_content_sha256"] == "a" * 64
        assert payload["model_grant"] == GRANT
        assert payload["source_edition"] == "autonomy"
        assert payload["input_metadata"] == {}
    finally:
        client.close()


@pytest.mark.parametrize(
    "patch",
    [
        {"message": ""},
        {"map_content_sha256": None},
        {"vehicle_content_sha256": "wrong"},
        {"input_metadata": {"simulation_teacher_control": True}},
        {"model_grant": GRANT},
    ],
)
def test_bad_planning_inputs_fail_before_grant_or_model_side_effects(patch):
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    try:
        with pytest.raises(ValidationError):
            client.prepare(THREAD, {**SELECTION, **patch})
        assert calls == []
    finally:
        client.close()


@pytest.mark.parametrize("bad_id", ["../threads/x", THREAD + "?other=1", "", "plan-" + "1" * 32])
def test_thread_identifiers_do_not_allow_path_injection(bad_id):
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    try:
        with pytest.raises(ConsoleError, match="THREAD_ID"):
            client.status(bad_id)
        assert calls == []
    finally:
        client.close()


def test_execution_requires_exact_reviewed_plan_and_does_not_retry():
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    try:
        with pytest.raises(ConsoleError, match="PLAN_CHANGED"):
            client.execute(THREAD, "plan-" + "3" * 32)
        assert len(calls) == 1
        assert client.execute(THREAD, PLAN)["status"] == "starting"
        executions = [req for req in calls if req.url.path.endswith("/execute")]
        assert len(executions) == 1
        assert json.loads(executions[0].content)["plan_revision_id"] == PLAN
    finally:
        client.close()


def test_usage_is_the_cloud_snapshot_not_a_local_calculation():
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    try:
        assert client.usage() == {"usage": {"request_count": 3, "remaining_ai_credits": 21}}
        assert len(calls) == 1
        assert calls[0].url.host == "a" * 20 + ".supabase.co"
    finally:
        client.close()


def test_server_cannot_redirect_any_credentials():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(307, headers={"Location": "https://example.org/steal"})

    client = make_client(handler)
    try:
        with pytest.raises(ConsoleError, match="REDIRECT_REJECTED"):
            client.usage()
        assert len(calls) == 1
    finally:
        client.close()


def test_network_failure_does_not_repeat_chargeable_write_or_claim_it_failed():
    calls = []

    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("sensitive remote exception")

    client = make_client(handler)
    try:
        with pytest.raises(ConsoleError, match="WRITE_OUTCOME_UNKNOWN"):
            client.create(ThreadCreate(selected_model="kimi-k2.6"))
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("body", [b"null", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}', b"not JSON"])
def test_invalid_response_does_not_become_success(body):
    client = make_client(lambda req: httpx.Response(200, content=body))
    try:
        with pytest.raises(ConsoleError, match="RESPONSE_INVALID"):
            client.bootstrap()
    finally:
        client.close()


def test_untrusted_error_text_is_not_printed():
    client = make_client(
        lambda req: httpx.Response(409, json={"detail": SESSION["identity_token"]})
    )
    try:
        with pytest.raises(ConsoleError, match="CONSOLE_HTTP_409") as caught:
            client.bootstrap()
        assert SESSION["identity_token"] not in str(caught.value)
    finally:
        client.close()


def test_render_hides_grants_and_authorities_without_rewriting_backend_evidence():
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    try:
        client.prepare(THREAD, SELECTION)
        evidence = {
            "execution_authority": {"permission": "opaque"},
            "model_grant": GRANT,
            "nested": [{"text": "oops " + SESSION["identity_token"] + " " + GRANT}],
        }
        rendered = client.render(evidence)
        assert GRANT not in rendered
        assert SESSION["identity_token"] not in rendered
        assert json.loads(rendered)["execution_authority"] == "[redacted]"
        assert evidence["execution_authority"] == {"permission": "opaque"}
    finally:
        client.close()


def test_console_parse_and_secret_input_failure_are_not_tracebacks():
    errors = io.StringIO()
    output = io.StringIO()
    code = console.main(
        ["--session-stdin", "catalog"],
        stdin=io.StringIO(json.dumps({**SESSION, "local_token": "sensitive\ninvalid"}) + "\n"),
        stdout=output,
        stderr=errors,
    )
    assert code == 2
    assert "sensitive" not in errors.getvalue()
    assert output.getvalue() == ""
    with pytest.raises(SystemExit):
        console.build_parser().parse_args(["execute", THREAD])


def test_console_entrypoint_uses_product_module_not_development_runner():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text("utf-8"))
    assert (
        project["project"]["scripts"]["dronedream-console"] == "dronedream_agent_app.console:main"
    )


def test_console_main_uses_secret_stdin_and_closes_client(monkeypatch):
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    monkeypatch.setattr(console, "ProductConsoleClient", lambda session: client)
    output, errors = io.StringIO(), io.StringIO()
    assert (
        console.main(
            ["--session-stdin", "usage"],
            stdin=io.StringIO(json.dumps(SESSION) + "\n"),
            stdout=output,
            stderr=errors,
        )
        == 0
    )
    assert json.loads(output.getvalue())["usage"]["request_count"] == 3
    assert errors.getvalue() == ""
    assert client._http.is_closed
    assert all(
        SESSION[name] not in output.getvalue()
        for name in ("local_token", "identity_token", "publishable_key")
    )


def test_console_main_reports_unknown_post_outcome_without_replay(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("response lost after request")

    client = make_client(handler)
    monkeypatch.setattr(console, "ProductConsoleClient", lambda session: client)
    output, errors = io.StringIO(), io.StringIO()
    assert (
        console.main(
            ["--session-stdin", "create", "--model", "kimi-k2.6"],
            stdin=io.StringIO(json.dumps(SESSION) + "\n"),
            stdout=output,
            stderr=errors,
        )
        == 1
    )
    assert json.loads(errors.getvalue()) == {
        "error": "CONSOLE_WRITE_OUTCOME_UNKNOWN",
        "automatic_retry": False,
    }
    assert len(calls) == 1
    assert output.getvalue() == ""
    assert client._http.is_closed


def test_response_budget_and_compression_are_checked_before_json(monkeypatch):
    from dronedream_agent_app import console_client

    monkeypatch.setattr(console_client, "MAX_RESPONSE_BYTES", 32)
    client = make_client(lambda req: httpx.Response(200, content=b" " * 33))
    try:
        with pytest.raises(ConsoleError, match="RESPONSE_TOO_LARGE"):
            client.bootstrap()
    finally:
        client.close()
    client = make_client(lambda req: httpx.Response(200, headers={"Content-Encoding": "br"}))
    try:
        with pytest.raises(ConsoleError, match="COMPRESSED_RESPONSE_REJECTED"):
            client.bootstrap()
    finally:
        client.close()


def test_grant_from_different_gateway_never_reaches_core():
    calls = []

    def handler(req):
        if req.url.path.endswith("/grants"):
            calls.append(req)
            return httpx.Response(
                200,
                json={
                    "data": {
                        "grant": GRANT,
                        "gateway_base_url": "https://example.org/model-gateway",
                    }
                },
            )
        return contract_response(req, calls)

    client = make_client(handler)
    try:
        with pytest.raises(ConsoleError, match="GATEWAY_ORIGIN_MISMATCH"):
            client.prepare(THREAD, SELECTION)
        assert not any(req.url.path.endswith("/prepare") for req in calls)
    finally:
        client.close()


def test_proxy_environment_is_not_inherited(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    client = ProductConsoleClient(ConsoleSession.model_validate(SESSION))
    try:
        assert client._http.trust_env is False
        assert client._http.follow_redirects is False
    finally:
        client.close()


def test_lightweight_model_directory_matches_desktop_bootstrap_and_requires_session(tmp_path):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.server import create_app
    from dronedream_agent_app.storage import AppStore

    app = create_app(store=AppStore(tmp_path), token=SESSION["local_token"])
    with TestClient(app) as desktop:
        assert desktop.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer " + SESSION["local_token"]}
        models = desktop.get("/v1/models", headers=headers).json()
        bootstrap = desktop.get("/v1/bootstrap", headers=headers).json()
        assert models["models"] == bootstrap["models"]
        assert set(models) == {"models"}


def test_real_desktop_router_records_cli_task_and_plan_but_rejects_unbound_execution(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.identity import VerifiedIdentity
    from dronedream_agent_app.mission_service import MissionService
    from dronedream_agent_app.server import create_app
    from dronedream_agent_app.storage import AppStore

    calls = []
    planner_calls = []
    store = AppStore(tmp_path)
    identity = VerifiedIdentity("account-test", None, None, CLOUD + "/auth/v1", 2_000_000_000)

    def verify(token):
        if token != SESSION["identity_token"]:
            raise ValueError("IDENTITY_TOKEN_INVALID")
        return identity

    def planner(self, **kwargs):
        planner_calls.append(kwargs)
        return {"goal": kwargs["message"], "plan_revision_id": PLAN, "status": "prepared"}

    monkeypatch.setattr(MissionService, "prepare", planner)
    app = create_app(store=store, token=SESSION["local_token"], identity_verifier=verify)
    with TestClient(app) as desktop:

        def handler(req):
            calls.append(req)
            if req.url.host != "127.0.0.1":
                return contract_response(req, [])
            response = desktop.request(
                req.method, req.url.path, headers=dict(req.headers), content=req.content
            )
            return httpx.Response(
                response.status_code,
                content=response.content,
                headers={"Content-Type": "application/json"},
            )

        client = make_client(handler)
        try:
            created = client.create(ThreadCreate(selected_model="kimi-k2.6"))
            task_id = created["thread_id"]
            plan = client.prepare(task_id, SELECTION)
            assert plan["plan_revision_id"] == PLAN
            current = client.status(task_id)
            assert current["state"] == "awaiting_confirmation"
            assert current["messages"][0]["content"] == SELECTION["message"]
            assert len(planner_calls) == 1
            assert planner_calls[0]["owner_account_id"] == "account-test"
            assert planner_calls[0]["connection"].api_key == GRANT
            # 夹具计划没有真实准备材料和所有权绑定，真实执行路由必须拒绝。
            with pytest.raises(ConsoleError, match="EXECUTION_THREAD_NOT_BOUND"):
                client.execute(task_id, PLAN)
        finally:
            client.close()


def test_real_desktop_router_rejects_invalid_account_before_planner(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.mission_service import MissionService
    from dronedream_agent_app.server import create_app
    from dronedream_agent_app.storage import AppStore

    planner_calls = []

    def verify(_token):
        raise ValueError("IDENTITY_TOKEN_INVALID")

    monkeypatch.setattr(MissionService, "prepare", lambda *a, **k: planner_calls.append(k))
    app = create_app(
        store=AppStore(tmp_path), token=SESSION["local_token"], identity_verifier=verify
    )
    with TestClient(app) as desktop:

        def handler(req):
            if req.url.host != "127.0.0.1":
                return contract_response(req, [])
            response = desktop.request(
                req.method, req.url.path, headers=dict(req.headers), content=req.content
            )
            return httpx.Response(response.status_code, content=response.content)

        client = make_client(handler)
        try:
            task_id = client.create(ThreadCreate(selected_model="kimi-k2.6"))["thread_id"]
            with pytest.raises(ConsoleError, match="CONSOLE_HTTP_401"):
                client.prepare(task_id, SELECTION)
            assert planner_calls == []
        finally:
            client.close()
