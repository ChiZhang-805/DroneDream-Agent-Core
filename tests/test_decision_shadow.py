"""Stage-decision contracts: unknown facts, probability checks, and no flight authority."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from dronedream_agent_core.decision_shadow import (
    ACTIONS, StageState, group_split, laya_request, numeric_features, rule_decision,
    shadow_suggestion, validate_probabilities,
)


# 功能：构造明确、可解释的接口测试状态，不作为实际飞行样本。
# 输入：待替换字段。
# 输出：正常路线的合成状态。
def state(**changes):
    return replace(StageState(True, True, True, False, False, False), **changes)


@pytest.mark.parametrize("field", ["pose_fresh", "geometry_fresh", "route_verified",
                                  "dynamic_crossing", "persistent_blockage"])
def test_missing_critical_information_is_unknown(field):
    """功能：关键未知不变成继续；输入：缺失字段；输出：补充观测建议。"""
    assert rule_decision(state(**{field: None})) == "request_observation"


@pytest.mark.parametrize("available", [False, None, True])
def test_optional_media_does_not_stop_progress(available):
    """功能：可选媒体不影响继续；输入：三态媒体；输出：同一建议。"""
    assert rule_decision(state(optional_media_available=available)) == "follow_route"


@pytest.mark.parametrize("change,expected", [({"dynamic_crossing": True}, "wait"),
    ({"persistent_blockage": True}, "replan"), ({"caution_required": True}, "slow_down"),
    ({"caution_required": None}, "slow_down")])
def test_stage_choices_preserve_mission(change, expected):
    """功能：检查等待/重规划语义；输入：明确事件；输出：阶段而非任务中止。"""
    assert rule_decision(state(**change)) == expected


@pytest.mark.parametrize("value", ["false", 0, 1, [], {}])
def test_boolean_coercion_rejected(value):
    """功能：拒绝隐式布尔转换；输入：非布尔值；输出：类型错误。"""
    with pytest.raises(ValueError):
        state(pose_fresh=value)


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "1"])
def test_numeric_corruption_rejected(value):
    """功能：拒绝损坏数值；输入：非法速度；输出：类型错误。"""
    with pytest.raises(ValueError):
        state(speed_mps=value)


def test_features_retain_unknown_mask():
    """功能：区分零和未知；输入：两种状态；输出：不同数值特征。"""
    assert numeric_features(state(speed_mps=None)) != numeric_features(state(speed_mps=0))


def test_probabilities_do_not_create_authority():
    """功能：高分不创造飞行许可；输入：不可执行最高分；输出：原分数与拒绝原因。"""
    probabilities = dict.fromkeys(ACTIONS, .01)
    probabilities["follow_route"] = .96
    suggestion = shadow_suggestion(probabilities, ("wait",))
    assert suggestion["reason"] == "top-choice-inadmissible"
    assert suggestion["probabilities"]["wait"] == .01
    assert suggestion["execution_authority"] is False
    assert shadow_suggestion(probabilities)["reason"] == "independent-admissibility-unavailable"


@pytest.mark.parametrize("values", [dict.fromkeys(ACTIONS, .1), {"follow_route": 1.},
    dict.fromkeys(ACTIONS, float("nan")), dict.fromkeys(ACTIONS, True),
    {**dict.fromkeys(ACTIONS, .2), "unexpected": 0.}])
def test_invalid_distributions_are_rejected(values):
    """功能：坏概率不降级成规则成功；输入：非法分布；输出：错误。"""
    with pytest.raises(ValueError):
        validate_probabilities(values)


def test_request_uses_upstream_question_contract():
    """功能：固定 Laya 问题接口；输入：合法状态；输出：短状态与完整选项。"""
    text, questions = laya_request(state())
    assert "Optional live-view video" in text and len(text) < 1800
    assert "Suggest one" not in text
    assert "Persistent blockage requires" in questions["stage"]["instructions"]
    assert questions["stage"]["type"] == "choice"
    assert set(questions["stage"]["criteria"]) == set(ACTIONS)
    assert questions["stage"]["instructions"]


def test_group_split_deterministic():
    """功能：同任务所有帧同组；输入：组身份；输出：稳定划分。"""
    assert group_split("route-a") == group_split("route-a")
    with pytest.raises(ValueError):
        group_split("")


# 功能：装载只读离线脚本供测试，不执行命令行入口。
# 输入：无。
# 输出：模块实例。
def script():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_stage_decisions.py"
    spec = importlib.util.spec_from_file_location("decision_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_cases_are_not_formal_training_or_cross_split_pairs():
    """功能：防止合成规则测试与训练实证混淆；输入：全案例；输出：明确隔离。"""
    cases = script().contract_cases()
    assert len(cases) == 1458
    assert all(not r["formal_training_eligible"] for r in cases)
    groups = {}
    for case in cases:
        assert groups.setdefault(case["group"], case["split"]) == case["split"]


def test_unlabeled_replay_never_gets_accuracy():
    """功能：无标签只报覆盖/延迟；输入：真实未标注回放；输出：准确率为空。"""
    assert script().metrics([{"latency_ms": 1., "choice": "wait", "reference": None}])["contract_accuracy"] is None


def test_replay_does_not_treat_empty_obstacles_as_clear_route():
    """功能：缺少检测证据不等同于无障碍；输入：空列表；输出：未知危险与路线。"""
    item = {"observation": {"observed_at_unix_ms": 1000, "dynamic_obstacles": [],
                             "stream_healthy": False},
            "command": {"generated_at_unix_ms": 1100}}
    parsed = script().replay_state(item)
    assert parsed.pose_fresh is None and parsed.route_verified is None
    assert parsed.dynamic_crossing is None


def test_existing_evidence_not_overwritten(tmp_path):
    """功能：保存报告禁止覆盖；输入：同一路径两次；输出：旧记录保留。"""
    module = script()
    path = tmp_path / "receipt.json"
    module.save(path, {"first": True})
    with pytest.raises(FileExistsError):
        module.save(path, {})


def test_absent_command_is_retained_as_unknown_not_deleted():
    """功能：无命令周期仍可回放；输入：真实空命令形态；输出：保留未知。"""
    result = script().replay_state({"command": None, "observation": None,
        "realtime_feature_snapshot": None, "recorded_at_unix_ms": 1000})
    assert result.pose_fresh is None and result.geometry_fresh is None
    assert result.speed_mps is None
