"""Console contract/fault tests, not paid-model or physical-flight qualification.

The API integration cases use the real desktop router/store. Identity, cloud
transport and the slow planner are explicit test doubles; non-Windows tests
also select an in-memory credential fixture instead of bypassing DPAPI safety.
"""

from __future__ import annotations

import io
import json
import struct
import tomllib
from argparse import Namespace
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from dronedream_agent_app import console
from dronedream_agent_app.console_client import (
    ConsoleClarificationRequired,
    ConsoleError,
    ConsoleSession,
    ProductConsoleClient,
)
from dronedream_agent_app.console_ui import TerminalUI
from dronedream_agent_app.models import ThreadCreate

pytestmark = pytest.mark.usefixtures("isolated_server_credentials")

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
#   生成独立 PNG 协议夹具，明确不来自真实相机或飞行。
# 输入：
#   无。
# 输出：
#   png：两像素宽的合法测试 PNG。
def overview_png():
    stream = io.BytesIO()
    Image.new("RGB", (2, 1), (0, 40, 80)).save(stream, format="PNG")
    png = stream.getvalue()
    return png


def test_watch_preserves_live_telemetry_and_ready_simulation_sources(monkeypatch):
    class ActiveMissionClient:
        def __init__(self):
            self.status_calls = 0

        def status(self, _thread_id):
            self.status_calls += 1
            return {
                "thread_id": THREAD,
                "state": "executing" if self.status_calls == 1 else "completed",
            }

        def telemetry(self, _thread_id):
            return {"phase": "cruise", "x_m": 12.5, "y_m": 4.0}

        def live_sources(self, _thread_id):
            return {
                "mode": "simulation",
                "sources": [
                    {
                        "id": "gazebo-render",
                        "label": "Gazebo",
                        "mode": "simulation",
                        "ready": True,
                    }
                ],
            }

    output = io.StringIO()
    client = ActiveMissionClient()
    monkeypatch.setattr(console.time, "sleep", lambda _seconds: None)

    result = console._wait_for_terminal(
        client,
        THREAD,
        poll_seconds=0.2,
        timeout_seconds=1,
        ui=TerminalUI(stream=output, color=False, width=100),
    )

    assert result["state"] == "completed"
    assert result["watch"]["telemetry_samples"] == 1
    assert result["watch"]["last_live_telemetry"]["phase"] == "cruise"
    assert result["watch"]["live_sources"]["sources"][0]["id"] == "gazebo-render"
    assert f"dronedream mission frame {THREAD}" in output.getvalue()


# 功能：
#   核对画面命令只读共享相机端点，保留原始字节，拒绝覆盖操作者已有文件。
# 输入：
#   tmp_path：独立测试输出目录。
# 输出：
#   None：端点、身份、摘要、写入或覆盖保护有偏差时测试失败。
def test_overview_command_saves_authenticated_original_png_without_overwrite(tmp_path):
    import hashlib

    calls, png = [], overview_png()

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=png, headers={"Content-Type": "image/png"})

    client = make_client(handler)
    output = tmp_path / "overview.png"
    args = console.build_parser().parse_args(["frame", THREAD, "--output", str(output)])
    try:
        receipt = console.dispatch(client, args)
        assert output.read_bytes() == png
        assert receipt["sha256"] == hashlib.sha256(png).hexdigest()
        assert receipt["qualification_granted"] is False
        assert len(calls) == 1
        assert calls[0].method == "GET"
        assert calls[0].url.path == "/v1/threads/" + THREAD + "/live-frame"
        assert calls[0].headers["authorization"] == "Bearer " + SESSION["local_token"]
        assert calls[0].headers["x-dronedream-identity-token"] == SESSION["identity_token"]
        assert calls[0].headers["accept"] == "image/png"
        with pytest.raises(ConsoleError, match="OUTPUT_EXISTS"):
            console.dispatch(client, args)
        assert output.read_bytes() == png
    finally:
        client.close()


# 功能：
#   拒绝不可信媒体、越界尺寸、损坏、压缩及重定向，不能将错误页保存为验收画面。
# 输入：
#   fault：模拟响应故障类型。
#   tmp_path：独立测试输出目录。
#   monkeypatch：临时字节预算替换器。
# 输出：
#   None：错误响应落盘或未产生对应错误码时测试失败。
@pytest.mark.parametrize(
    "fault,code",
    [
        ("type", "TYPE_INVALID"),
        ("header", "HEADER_INVALID"),
        ("dimensions", "DIMENSIONS_INVALID"),
        ("corrupt", "CONTENT_INVALID"),
        ("redirect", "REDIRECT_REJECTED"),
        ("compressed", "COMPRESSED_RESPONSE_REJECTED"),
        ("large", "RESPONSE_TOO_LARGE"),
        ("missing", "HTTP_404"),
    ],
)
def test_overview_rejects_invalid_response_before_writing(fault, code, tmp_path, monkeypatch):
    from dronedream_agent_app import console_client

    png = overview_png()
    content, status, headers = png, 200, {"Content-Type": "image/png"}
    if fault == "type":
        headers["Content-Type"] = "text/html"
    elif fault == "header":
        content = b"not a PNG"
    elif fault == "dimensions":
        content = png[:16] + struct.pack(">II", 5000, 1) + png[24:]
    elif fault == "corrupt":
        # 完整 IHDR 之后的损坏仍需要解码器验证，而不是仅检查文件魔数。
        content = png[:33] + b"broken chunk data"
    elif fault == "redirect":
        status, headers["Location"] = 307, "https://example.org/steal"
    elif fault == "compressed":
        headers["Content-Encoding"] = "unknown"
    elif fault == "large":
        monkeypatch.setattr(console_client, "MAX_RESPONSE_BYTES", len(png) - 1)
    elif fault == "missing":
        status = 404
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=content, headers=headers)

    client, output = make_client(handler), tmp_path / "refused.png"
    try:
        with pytest.raises(ConsoleError, match=code):
            console.save_overview_frame(client, THREAD, output)
        assert not output.exists()
        assert len(calls) == 1
    finally:
        client.close()


# 功能：
#   验证目标目录不存在时返回明确写入错误，不打印路径内容或成功回执。
# 输入：
#   tmp_path：没有子目录的独立测试目录。
# 输出：
#   None：错误被吞掉或误报成功时测试失败。
def test_overview_write_failure_does_not_claim_saved(tmp_path):
    client = make_client(
        lambda request: httpx.Response(
            200, content=overview_png(), headers={"Content-Type": "image/png"}
        )
    )
    try:
        with pytest.raises(ConsoleError, match="WRITE_FAILED"):
            console.save_overview_frame(client, THREAD, tmp_path / "missing" / "frame.png")
    finally:
        client.close()


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
            "scope": "job",
            "scope_reference": THREAD,
            "provider": "kimi",
            "model": "kimi-k2.6",
        }
        assert local.headers["authorization"] == "Bearer " + SESSION["local_token"]
        assert local.headers["x-dronedream-identity-token"] == SESSION["identity_token"]
        assert local.extensions["timeout"]["read"] == 1200
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


@pytest.mark.parametrize(
    "detail,expected",
    [
        ("PLAN_REVIEW_EXHAUSTED", "PLAN_REVIEW_EXHAUSTED"),
        ("TASK_GRAPH_ACTION_EVIDENCE_MISSING:pickup_payload", "TASK_GRAPH_ACTION_EVIDENCE_MISSING"),
        ({"code": "MODEL_INVOCATION_FAILED", "error_id": "opaque"}, "MODEL_INVOCATION_FAILED"),
    ],
)
def test_bounded_service_error_code_is_visible_without_exposing_context(detail, expected):
    client = make_client(lambda req: httpx.Response(409, json={"detail": detail}))
    try:
        with pytest.raises(ConsoleError, match=f"^{expected}$"):
            client.bootstrap()
    finally:
        client.close()


# 功能：
#   核对 CLI 观测命令仅访问账户绑定的读取接口，不调用模型或确认执行。
# 输入：
#   command、suffix：终端命令及唯一允许访问的产品端点。
# 输出：
#   None：请求路径、授权头与原始遥测均与产品接口一致。
@pytest.mark.parametrize(
    "command,suffix", [("live-sources", "live-sources"), ("telemetry", "live-telemetry")]
)
def test_console_observation_commands_use_only_authenticated_read(command, suffix):
    calls = []
    client = make_client(lambda req: contract_response(req, calls))
    try:
        args = console.build_parser().parse_args([command, THREAD])
        assert console.dispatch(client, args) == {"status": "fixture-response"}
        assert len(calls) == 1
        request = calls[0]
        assert request.method == "GET"
        assert request.url.path == "/v1/threads/" + THREAD + "/" + suffix
        assert request.headers["authorization"] == "Bearer " + SESSION["local_token"]
        assert request.headers["x-dronedream-identity-token"] == SESSION["identity_token"]
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


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("models", "/v1/models"),
        ("maps", "/v1/map-resource-catalog"),
        ("aircraft", "/v1/vehicle-resource-catalog"),
        ("asset_pairs", "/v1/asset-pair-catalog"),
    ],
)
def test_product_discovery_uses_shared_desktop_catalog_routes(method, path):
    calls = []
    client = make_client(lambda request: contract_response(request, calls))
    try:
        getattr(client, method)()
        assert [request.url.path for request in calls] == [path]
    finally:
        client.close()


def test_model_clarification_is_structured_and_never_treated_as_a_plan():
    fields = ["Which building should receive the package?", "Should the aircraft return?"]
    client = make_client(
        lambda request: httpx.Response(
            409,
            json={"detail": {"code": "MISSION_CLARIFICATION_REQUIRED", "fields": fields}},
        )
    )
    try:
        with pytest.raises(ConsoleClarificationRequired) as caught:
            client.bootstrap()
        assert caught.value.fields == fields
    finally:
        client.close()


def test_server_thread_directory_only_includes_archive_when_requested(tmp_path):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.server import create_app
    from dronedream_agent_app.storage import AppStore

    store = AppStore(tmp_path)
    active = store.create_thread("Active", "gpt-5.4", "en-US")
    archived = store.create_thread("Archived", "gpt-5.4", "en-US")
    store.patch_thread(archived["thread_id"], {"archived": True})
    app = create_app(store=store, token=SESSION["local_token"])
    headers = {"Authorization": "Bearer " + SESSION["local_token"]}
    with TestClient(app) as desktop:
        current = desktop.get("/v1/threads", headers=headers).json()
        complete = desktop.get("/v1/threads?include_archived=true", headers=headers).json()
    assert [item["thread_id"] for item in current["threads"]] == [active["thread_id"]]
    assert {item["thread_id"] for item in complete["threads"]} == {
        active["thread_id"],
        archived["thread_id"],
    }


class _FlyFixture:
    def __init__(self, *, qualified=True):
        self.calls = []
        self.qualified = qualified

    def models(self):
        return {"models": [{"id": "gpt-5.4", "source": "default"}]}

    def maps(self):
        return {"resources": [{"resource_id": "campus", "display_name": {"en-US": "Campus"}}]}

    def aircraft(self):
        return {"resources": [{"resource_id": "x500", "display_name": {"en-US": "PX4 x500 depth"}}]}

    def asset_pairs(self):
        return {
            "counts": {
                "qualified_builtin": 1 if self.qualified else 0,
                "compatible_requires_qualification": 0 if self.qualified else 1,
            },
            "pairs": [
                {
                    "map_resource_id": "campus",
                    "vehicle_resource_id": "x500",
                    "status": (
                        "qualified_builtin"
                        if self.qualified
                        else "compatible_requires_qualification"
                    ),
                    "qualification_id": "qualification-real",
                    "qualified_map_asset_id": "map.asset",
                    "qualified_vehicle_asset_id": "aircraft.asset",
                }
            ],
        }

    def bootstrap(self):
        return {
            "asset_versions": [
                {
                    "asset_id": "map.asset",
                    "content_sha256": "a" * 64,
                    "maturity": "qualified",
                },
                {
                    "asset_id": "aircraft.asset",
                    "content_sha256": "b" * 64,
                    "maturity": "qualified",
                },
            ]
        }

    def create(self, request):
        self.calls.append(("create", request.selected_model))
        return {"thread_id": THREAD}

    def prepare(self, thread_id, selection):
        self.calls.append(("prepare", thread_id, selection))
        return {
            "plan_revision_id": PLAN,
            "goal": selection["message"],
            "model_calls": 4,
            "successful_model_responses": 4,
            "model_id": "gpt-5.4",
            "model_source": "default",
            "model_tool_proof": {
                "tool_requests": [{"call_id": "call-1", "tool_id": "plan_route"}],
                "tool_receipts": [
                    {"call_id": "call-1", "tool_id": "plan_route", "outcome": "accepted"}
                ],
                "task_graph": {"nodes": [{"node_id": "takeoff", "action": "takeoff"}]},
            },
        }

    def execute(self, thread_id, plan_id):
        self.calls.append(("execute", thread_id, plan_id))
        return {"status": "starting", "execution_id": "execution-real"}

    def status(self, thread_id):
        self.calls.append(("status", thread_id))
        return {"thread_id": thread_id, "state": "completed"}

    def evidence(self, thread_id):
        self.calls.append(("evidence", thread_id))
        return {
            "execution_id": "execution-real",
            "state": "completed",
            "result": {"status": "verified", "gates": {"landed": True}},
        }

    def patch_thread(self, thread_id, changes):
        self.calls.append(("patch", thread_id, changes))
        return {"thread_id": thread_id, **changes}


def _fly_args(**changes):
    values = {
        "model": "gpt-5.4",
        "map_id": "campus",
        "vehicle_id": "x500",
        "text": "Bring the lab package back after delivery.",
        "title": "Lab delivery",
        "locale": "en-US",
        "start": "__auto__",
        "plan_only": False,
        "no_archive": False,
        "poll_seconds": 0.2,
        "timeout_seconds": 5.0,
    }
    values.update(changes)
    return Namespace(**values)


def test_fly_refuses_unqualified_asset_pair_before_task_or_model_call():
    client = _FlyFixture(qualified=False)
    with pytest.raises(ConsoleError, match="REQUIRES_QUALIFICATION"):
        console._fly(client, _fly_args(), stdin=io.StringIO(), ui=None)
    assert client.calls == []


def test_fly_refuses_plan_without_successful_model_and_tool_receipts():
    client = _FlyFixture()
    client.prepare = lambda _thread_id, _selection: {
        "plan_revision_id": PLAN,
        "model_calls": 0,
        "successful_model_responses": 0,
    }

    with pytest.raises(ConsoleError, match="MODEL_PLAN_PROOF_MISSING"):
        console._fly(client, _fly_args(plan_only=True), stdin=io.StringIO(), ui=None)

    assert not any(call[0] == "execute" for call in client.calls)


def test_fly_requires_short_explicit_confirmation_then_archives_verified_evidence():
    client = _FlyFixture()
    output = io.StringIO()
    ui = TerminalUI(stream=output, color=True, width=100)
    result = console._fly(client, _fly_args(), stdin=io.StringIO("fly\n"), ui=ui)
    assert result["status"] == "verified"
    assert ("execute", THREAD, PLAN) in client.calls
    assert ("patch", THREAD, {"archived": True}) in client.calls
    prepared = next(item for item in client.calls if item[0] == "prepare")
    assert prepared[2]["message"] == "Bring the lab package back after delivery."
    assert prepared[2]["map_content_sha256"] == "a" * 64
    assert prepared[2]["vehicle_content_sha256"] == "b" * 64
    assert set(prepared[2]) == {
        "message",
        "map_id",
        "map_content_sha256",
        "vehicle_id",
        "vehicle_content_sha256",
        "start_entity",
    }
    rendered = output.getvalue()
    assert "\x1b[" in rendered
    assert PLAN in rendered
    assert "Planner" in rendered
    assert "model responses" in rendered
    assert "tool receipts" in rendered


def test_doctor_uses_runtime_manager_readiness_fields():
    client = _FlyFixture()
    client.runtime = lambda: {
        "runtime_available": True,
        "resources_ready": True,
        "provisioned": True,
        "issue": None,
    }
    report = console._doctor(client)
    runtime = next(item for item in report["checks"] if item["name"] == "Runtime")
    assert report["ready"] is True
    assert runtime == {"name": "Runtime", "status": "ready", "detail": "ready"}


def test_doctor_rejects_partially_installed_runtime():
    client = _FlyFixture()
    client.runtime = lambda: {
        "runtime_available": True,
        "resources_ready": False,
        "provisioned": False,
        "issue": "RUNTIME_RESOURCES_MISSING",
    }
    report = console._doctor(client)
    runtime = next(item for item in report["checks"] if item["name"] == "Runtime")
    assert report["ready"] is False
    assert runtime == {
        "name": "Runtime",
        "status": "blocked",
        "detail": "RUNTIME_RESOURCES_MISSING",
    }


@pytest.mark.parametrize(
    ("empty_method", "check_name"), [("maps", "Map catalog"), ("aircraft", "Aircraft catalog")]
)
def test_doctor_rejects_empty_asset_catalogs(empty_method, check_name):
    client = _FlyFixture()
    client.runtime = lambda: {
        "runtime_available": True,
        "resources_ready": True,
        "provisioned": True,
        "issue": None,
    }
    setattr(client, empty_method, lambda: {"resources": []})

    report = console._doctor(client)

    check = next(item for item in report["checks"] if item["name"] == check_name)
    assert report["ready"] is False
    assert check == {"name": check_name, "status": "blocked", "detail": "0 available"}


def test_pair_filter_recomputes_counts_for_the_visible_selection():
    catalog = {
        "counts": {"qualified_builtin": 2, "compatible_requires_qualification": 1},
        "pairs": [
            {
                "map_resource_id": "one",
                "vehicle_resource_id": "x500",
                "status": "qualified_builtin",
            },
            {
                "map_resource_id": "two",
                "vehicle_resource_id": "x500",
                "status": "qualified_builtin",
            },
            {
                "map_resource_id": "two",
                "vehicle_resource_id": "iris",
                "status": "compatible_requires_qualification",
            },
        ],
    }

    filtered = console._filter_pairs(catalog, "two", None)

    assert len(filtered["pairs"]) == 2
    assert filtered["counts"] == {
        "qualified_builtin": 1,
        "compatible_requires_qualification": 1,
    }


def test_terminal_plan_and_evidence_views_show_real_model_tools_actions_and_gates():
    output = io.StringIO()
    ui = TerminalUI(stream=output, color=False, width=110)
    ui.render(
        {
            "plan_revision_id": PLAN,
            "goal": "Deliver instruments and return.",
            "model_id": "gpt-4.1",
            "model_source": "default",
            "model_calls": 3,
            "planning_attempts": 1,
            "mission_plan": {
                "task_graph": {
                    "nodes": [
                        {
                            "label": "Pick up payload",
                            "executor": "payload_controller",
                            "completion_evidence": ["payload attached"],
                        }
                    ]
                },
                "metrics": {"route_length_m": 42.5, "estimated_duration_s": 18.0},
            },
            "model_tool_proof": {
                "tool_receipts": [
                    {"tool_id": "plan_route", "outcome": "accepted", "call_id": "call-real"}
                ]
            },
        },
        "plan",
    )
    ui.render(
        {
            "state": "completed",
            "execution_id": "execution-real",
            "result": {
                "status": "verified",
                "contract_id": "contract-real",
                "gates": {"landing_confirmed": True, "executor_completed": True},
                "runtime_action_receipts": [
                    {"step_id": "pickup", "action": "pickup", "status": "accepted", "attempts": 1}
                ],
                "completion_model_call": {
                    "role": "completion",
                    "provider": "openai",
                    "model": "gpt-4.1",
                    "response_id": "response-real",
                },
                "checkpoint_model_calls": [],
            },
        },
        "evidence",
    )

    rendered = output.getvalue()
    for expected in (
        "Deliver instruments and return.",
        "plan_route",
        "Pick up payload",
        "Landing Confirmed",
        "pickup",
        "response-real",
    ):
        assert expected in rendered


def test_terminal_workflow_uses_accessible_symbols_and_structured_prompts():
    output = io.StringIO()
    ui = TerminalUI(stream=output, color=False, width=92)

    ui.banner()
    ui.choice("model", ["GPT-5.4 (gpt-5.4)", "Kimi K2.6 (kimi-k2.6)"])
    ui.confirm_plan("plan-visible")
    ui.status("Runtime", "ready", "PX4/Gazebo connected")
    ui.status("Planning", "executing", "route tool active")
    ui.error("CONSOLE_DEMO_ERROR", "No mission state was changed.")

    rendered = output.getvalue()
    for expected in (
        "DroneDream",
        "SELECT MODEL",
        "01",
        "LAUNCH CONFIRMATION",
        "plan-visible",
        "✓ Runtime",
        "◆ Planning",
        "ERROR · CONSOLE_DEMO_ERROR",
    ):
        assert expected in rendered
    assert "FLIGHT CONSOLE" not in rendered
    assert "◇" in rendered


def test_detailed_terminal_brand_matches_the_wide_console_boundary():
    output = io.StringIO()
    TerminalUI(stream=output, color=False, width=108).banner()

    lines = output.getvalue().splitlines()
    assert len(lines) == 11
    assert max(map(len, lines)) <= 108
    assert all(not line or line.startswith("   ") for line in lines)
    assert "⣿⣿⣿⣿⣿⣷⣄" in output.getvalue()
    assert "FLIGHT CONSOLE" not in output.getvalue()


def test_interactive_shell_has_product_prompt_aliases_and_clean_exit(monkeypatch):
    class TerminalBuffer(io.StringIO):
        def isatty(self):
            return True

    calls = []

    def child_main(argv, **kwargs):
        calls.append((argv, kwargs.get("_shell_child")))
        return 0

    monkeypatch.setattr(console, "main", child_main)
    output, errors = TerminalBuffer(), io.StringIO()
    code = console._shell_loop(
        stdin=io.StringIO("maps\nexit\n"),
        stdout=output,
        stderr=errors,
        no_color=True,
    )

    assert code == 0
    assert calls == [(["--no-color", "assets", "maps"], True)]
    assert "SYSTEM CHECK" in output.getvalue()
    assert "Console" in output.getvalue()
    assert "Commands" in output.getvalue()
    assert "Display" in output.getvalue()
    assert "mission | assets | runtime | history" in output.getvalue()
    assert "◆ DroneDream ›" in output.getvalue()
    assert "Flight console closed." in output.getvalue()
    assert "(base)" not in output.getvalue()
    assert "PS Q:" not in output.getvalue()


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
