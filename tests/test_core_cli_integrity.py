"""Offline CLI failure-path tests; no provider, account or flight is invoked."""

import json
from argparse import Namespace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from test_map_semantic_boundaries import _semantic, _write

from dronedream_agent_core import cli
from dronedream_agent_core.assets import load_map_catalog
from dronedream_agent_core.contracts import IntentArtifact, IntentCritique, ModelCallRecord
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   准备明确属于测试的请求、语义与候选，不使用产品账户或真实地图。
# 输入：
#   tmp_path：测试专用目录。
# 输出：
#   args：满足命令输入要求的路径及隔离测试 profile。
def _inputs(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"conversation_id": "cli-fixture", "message": "go to target and return"}),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        load_map_catalog(_write(tmp_path, _semantic())).model_dump_json(), encoding="utf-8"
    )
    intent = tmp_path / "intent.json"
    artifact = IntentArtifact(
        goal="go to target and return",
        start_entity="start",
        target_entity="target",
        return_entity="start",
        payload_action="none",
    )
    # 包装散列故意不可信，输出必须绑定候选本身。
    intent.write_text(
        json.dumps(
            {
                "artifact": artifact.model_dump(mode="json"),
                "model_call": {"output_sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        profile="test",
        request=request,
        map_catalog=catalog,
        intent=intent,
        provider="offline-fixture",
        max_attempts=1,
        output=tmp_path / "output.json",
    )
    return args


class _OfflinePort:
    # 功能：
    #   保存测试期望，不创建网络客户端。
    # 输入：
    #   fail：是否在调用阶段模拟供应商异常。
    # 输出：
    #   None：初始化计数和关闭状态。
    def __init__(self, fail):
        self.fail = fail
        self.closed = False
        self.calls = 0

    # 功能：
    #   模拟端口成功和失败，仅用于确认 CLI 清理行为，不充当飞行能力证明。
    # 输入：
    #   kwargs：命令传入的角色、输入和上下文。
    # 输出：
    #   result：带明确测试供应商的结构化结果，或抛出模拟错误。
    def call(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("OFFLINE_PROVIDER_FAILURE")
        artifact = (
            IntentCritique(accepted=True)
            if kwargs["role"] == "intent_critic"
            else IntentArtifact(
                goal="go to target and return",
                start_entity="start",
                target_entity="target",
                return_entity="start",
                payload_action="none",
            )
        )
        record = ModelCallRecord(
            call_id="model-" + "1" * 24,
            role=kwargs["role"],
            attempt=1,
            input_sha256=sha256_json(kwargs["input_artifact"]),
            output_sha256=sha256_json(artifact),
            output_schema=type(artifact).__name__,
            provider="offline-fixture",
            model="test-only",
            latency_ms=0,
            created_at=datetime.now(UTC),
        )
        result = SimpleNamespace(artifact=artifact, record=record)
        return result

    # 功能：
    #   记录 CLI 是否无论成功失败都回收端口。
    # 输入：
    #   无。
    # 输出：
    #   None：关闭状态设为真。
    def close(self):
        self.closed = True


# 功能：
#   覆盖两个模型命令的成功和失败退出，避免残留端口或未关闭的请求状态。
# 输入：
#   tmp_path：独立测试目录。
#   monkeypatch：注入完全离线的端口。
#   handler：意图提取或评审命令。
#   fail：是否模拟调用错误。
# 输出：
#   无：每种路径均应关闭端口，失败不写成功产物。
@pytest.mark.parametrize("handler", [cli._model_probe, cli._intent_critic])
@pytest.mark.parametrize("fail", [False, True])
def test_model_commands_close_their_port(tmp_path, monkeypatch, handler, fail):
    args = _inputs(tmp_path)
    port = _OfflinePort(fail)
    monkeypatch.setattr(cli, "StructuredModelPort", lambda *args, **kwargs: port)
    if fail:
        with pytest.raises(RuntimeError, match="OFFLINE_PROVIDER_FAILURE"):
            handler(args)
        assert not args.output.exists()
    else:
        assert handler(args) == 0
        output = cli._read_json(args.output)
        if handler is cli._intent_critic:
            artifact = IntentArtifact.model_validate(cli._read_json(args.intent)["artifact"])
            assert output["candidate_intent_sha256"] == sha256_json(artifact)
    assert port.calls == 1
    assert port.closed


# 功能：
#   输出已存在时在调用模型之前终止，避免消耗额度后覆盖历史产物。
# 输入：
#   tmp_path：测试目录。
#   monkeypatch：捕获模型端口的任何创建尝试。
# 输出：
#   无：已有字节不变且没有创建端口。
def test_existing_output_is_rejected_before_model_allocation(tmp_path, monkeypatch):
    args = _inputs(tmp_path)
    args.output.write_bytes(b"existing evidence")
    allocations = []
    monkeypatch.setattr(
        cli, "StructuredModelPort", lambda *args, **kwargs: allocations.append(args)
    )
    with pytest.raises(FileExistsError):
        cli._model_probe(args)
    assert allocations == []
    assert args.output.read_bytes() == b"existing evidence"


# 功能：
#   核对标准 JSON 读取边界，歧义键和非有限数不能进入模型输入。
# 输入：
#   tmp_path：测试目录。
#   raw：故意无效的文件字节。
# 输出：
#   无：输入在模型调用前抛出错误。
@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":1e400}', b"[]"])
def test_cli_reader_rejects_ambiguous_or_nonstandard_json(tmp_path, raw):
    path = tmp_path / "invalid.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        cli._read_json(path)


# 功能：
#   核对新的输出写入函数不覆盖旧文件、无效载荷也不会创建伪完成文件。
# 输入：
#   tmp_path：独立临时目录。
# 输出：
#   无：首次写入可读，重复写入被拒绝。
def test_cli_outputs_are_exclusive_and_validated_before_creation(tmp_path):
    path = tmp_path / "artifact.json"
    cli._write_output(path, {"ok": True})
    with pytest.raises(FileExistsError):
        cli._write_output(path, {"ok": False})
    assert cli._read_json(path) == {"ok": True}
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        cli._write_output(invalid, {"x": float("nan")})
    assert not invalid.exists()


# 功能：
#   确认旧学校地图生成入口已经停用，不能用历史轨迹拼出新的已核验地图。
# 输入：
#   tmp_path：新图的候选输出位置。
# 输出：
#   无：命令返回明确停用异常，且不创建任何地图文件。
def test_retired_map_generator_cannot_emit_a_graph(tmp_path):
    output = tmp_path / "graph.json"
    with pytest.raises(cli.CliProfileError, match="RETIRED_MAP_GRAPH_EXPORT"):
        cli._export_navigation_graph(Namespace(output=output))
    assert not output.exists()
