"""Persisted execution contracts retain strict JSON semantics, including dates."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from dronedream_agent_app.runtime_control_records import read_control_record
from dronedream_agent_core.contract_json import decode_contract_json
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness.graph import HarnessStageReceipt


class ReceiptEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    harness_stage_receipts: list[HarnessStageReceipt]
    notes: str = ""
    ready: bool = False


# 功能：
#   构造包含真实严格时间字段的阶段回执，覆盖桌面执行入口曾遗漏的 JSON 往返。
# 输入：
#   无。
# 输出：
#   envelope：没有模拟器或网络依赖的严格合同。
def receipt_envelope():
    envelope = ReceiptEnvelope(harness_stage_receipts=[HarnessStageReceipt(
        topology_id="test.topology", node_id="test.stage", node_kind="core",
        input_sha256="a" * 64, output_sha256="b" * 64, completed_at=datetime.now(UTC),
    )])
    return envelope


# 功能：
#   验证时间字段和超过插件消息尺寸的落盘合同仍可严格读取，摘要不发生变化。
# 输入：
#   size：追加的合法文字长度；tmp_path：隔离文件目录。
# 输出：
#   None：往返值和摘要不一致时测试失败。
@pytest.mark.parametrize("size", [0, 2 * 1024 * 1024 + 1])
def test_persisted_strict_receipt_roundtrip(tmp_path, size):
    original = receipt_envelope()
    original.notes = "x" * size
    raw = original.model_dump_json()
    with pytest.raises(ValidationError, match="datetime"):
        ReceiptEnvelope.model_validate(json.loads(raw))
    restored = decode_contract_json(raw, ReceiptEnvelope, limit=16 * 1024 * 1024, node_limit=2_000_000)
    assert restored == original
    assert sha256_json(restored) == sha256_json(original)
    if size == 0:
        path = tmp_path / "control.json"
        path.write_text(raw, encoding="utf-8")
        assert read_control_record(path, ReceiptEnvelope) == original


# 功能：
#   确认修复没有放宽重复键、非有限数、严格布尔、未知字段或损坏时间的约束。
# 输入：
#   change：针对合法合同的一种破坏方式。
# 输出：
#   None：任一坏合同被接受时测试失败。
@pytest.mark.parametrize("change", ["duplicate", "nonfinite", "bool", "extra", "date"])
def test_contract_json_still_rejects_invalid_content(change):
    raw = receipt_envelope().model_dump_json()
    if change == "duplicate":
        raw = '{"ready":true,' + raw[1:]
    elif change == "nonfinite":
        raw = raw.replace('"ready":false', '"ready":NaN')
    else:
        value = json.loads(raw)
        if change == "bool":
            value["ready"] = "false"
        elif change == "extra":
            value["unexpected"] = 1
        else:
            value["harness_stage_receipts"][0]["completed_at"] = "invalid"
        raw = json.dumps(value)
    with pytest.raises(ValueError):
        decode_contract_json(raw, ReceiptEnvelope, limit=4096, node_limit=1000)


# 功能：
#   检查字节、节点与嵌套深度预算仍先于模型校验生效，不变成无限量 JSON 入口。
# 输入：
#   raw：超限的 JSON；limit、nodes：对应预算。
# 输出：
#   None：输入越过预算时测试失败。
@pytest.mark.parametrize("raw,limit,nodes", [('"large"', 2, 100), ('[1,2]', 100, 1), ('[' * 70 + '0' + ']' * 70, 1000, 1000)])
def test_contract_json_preserves_resource_limits(raw, limit, nodes):
    with pytest.raises(ValueError):
        decode_contract_json(raw, ReceiptEnvelope, limit=limit, node_limit=nodes)
