"""Boundary regressions for causal decision inputs and corpus admission."""

import copy
import json

import pytest

from dronedream_agent_core.decision_dataset import audit_records, evidence_path, read_records
from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.decision_state_adapter import DecisionStateV2, model_state
from scripts.prepare_decision_contract_smoke import smoke_rows, synthetic_state


# 功能：校验未知、源时效与数据剥离；输入：无；输出：断言原始身份不泄漏进模型。
def test_unknown_and_optional_media():
    value = synthetic_state(1, "follow_route")
    first = DecisionStateV2.model_validate_json(json.dumps(value))
    value["frame"]["optional_media_available"] = False
    value["mission_id"] = "different-mission"
    second = DecisionStateV2.model_validate_json(json.dumps(value))
    assert model_state(first) == model_state(second)
    assert "evidence_sha256" not in model_state(first)
    assert "position_world_enu_m" not in model_state(first)
    assert first.frame.pose_source.fresh(10350)
    assert not first.frame.pose_source.fresh(9999)
    assert not first.frame.pose_source.fresh(10351)


# 功能：拒绝未来/缺源/布尔假数值等状态；输入：字段突变；输出：校验必须失败。
@pytest.mark.parametrize("mutation", ["future", "no-source", "nan", "bool-number", "extra"])
def test_invalid_state(mutation):
    state = synthetic_state(1, "follow_route")
    if mutation == "future":
        state["frame"]["pose_source"]["observed_at_ms"] = 10001
    elif mutation == "no-source":
        state["frame"]["geometry_source"] = None
    elif mutation == "nan":
        state["frame"]["clearance_up_m"] = float("nan")
    elif mutation == "bool-number":
        state["frame"]["clearance_up_m"] = True
    else:
        state["future_outcome"] = "success"
    with pytest.raises(ValueError):
        DecisionStateV2.model_validate_json(json.dumps(state))


# 功能：历史必须早于当前且有序；输入：无；输出：未来帧/重复时间拒绝。
def test_history_causality():
    state = synthetic_state(1, "wait")
    state["history"] = [copy.deepcopy(state["frame"])]
    with pytest.raises(ValueError, match="HISTORY"):
        DecisionStateV2.model_validate_json(json.dumps(state))
    prior = state["history"][0]
    prior["observed_at_ms"] = 9800
    for key in ("pose_source", "geometry_source", "route_source"):
        prior[key]["observed_at_ms"] = 9800
    assert len(DecisionStateV2.model_validate_json(json.dumps(state)).history) == 1
    state["history"].append(copy.deepcopy(prior))
    with pytest.raises(ValueError, match="HISTORY"):
        DecisionStateV2.model_validate_json(json.dumps(state))


# 功能：合成数据完整也不能变成正式资格；输入：临时目录；输出：准入为 false。
def test_smoke_never_qualifies(tmp_path):
    rows = smoke_rows()
    report = audit_records(rows, tmp_path, smoke=True)
    assert report["formal_windows"] == 0
    assert report["rows"] == 50
    assert not report["gpu_data_ready"]
    with pytest.raises(ValueError, match="FORMAL_LABEL"):
        audit_records(rows, tmp_path)


# 功能：阻止换组名的相同输入与同组异集合泄漏；输入：临时目录；输出：明确失败。
@pytest.mark.parametrize("mode", ["group", "input", "duplicate", "label"])
def test_corpus_boundaries(tmp_path, mode):
    rows = smoke_rows()
    if mode == "group":
        rows[10]["parent_group"] = rows[0]["parent_group"]
    elif mode == "input":
        rows[10]["state"] = copy.deepcopy(rows[0]["state"])
    elif mode == "duplicate":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    else:
        rows[0]["acceptable_actions"] = ["invented-action"]
    with pytest.raises(ValueError):
        audit_records(rows, tmp_path, smoke=True)


# 功能：跨平台证据路径不可逃逸；输入：非法路径；输出：拒绝。
@pytest.mark.parametrize("path", ["../x", "/x", "C:/x", "a\\b", "a/./b", "a//../b"])
def test_path_escape(tmp_path, path):
    with pytest.raises(ValueError):
        evidence_path(tmp_path, path)


# 功能：拒绝重复 JSON 键与非有限常量；输入：临时路径；输出：读取失败。
@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '[]'])
def test_json_integrity(tmp_path, text):
    source = tmp_path / "rows.jsonl"
    source.write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(source)


# 功能：正式标签不能只有哈希字符串；输入：伪资格；输出：缺少证据时失败。
def test_fake_formal_eligibility(tmp_path):
    rows = smoke_rows()
    rows[0].update(label_source="verified-simulation-outcome", formal_training_eligible=True)
    with pytest.raises(ValueError, match="EVIDENCE_REQUIRED"):
        audit_records(rows, tmp_path)


# 功能：集合损失与指标不抹除难例；输入：CPU 张量；输出：单标签等价且缺类不报宏平均。
def test_set_loss_and_calibration():
    torch = pytest.importorskip("torch")
    from dronedream_agent_core.laya_uav_training import (
        acceptable_mask,
        decision_metrics,
        fit_temperature,
        set_cross_entropy,
    )
    logits = torch.tensor([[2., 0., 0., 0., 0.], [0., 1., 2., 0., 0.]], requires_grad=True)
    accepted = torch.tensor([[True, False, False, False, False],
                             [False, True, True, False, False]])
    loss = set_cross_entropy(logits, accepted)
    assert float(loss[0].detach()) == pytest.approx(float(torch.nn.functional.cross_entropy(
        logits[:1].detach(), torch.tensor([0]))))
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()
    result = decision_metrics(logits.detach(), accepted)
    assert result["acceptable_accuracy"] == 1
    assert result["macro_f1"] is None
    assert result["ambiguous_rows"] == 1
    assert fit_temperature(logits.detach(), accepted)["unique_rows"] == 1
    order = list(reversed(ACTIONS))
    assert acceptable_mask(["wait"], order) == [a == "wait" for a in order]
    with pytest.raises(ValueError):
        set_cross_entropy(logits.detach(), torch.zeros_like(accepted))
