"""Live uncertainty must survive reroute, speed and coverage changes."""

import hashlib
import json
from datetime import timedelta

import pytest
from test_localization_evidence import identity
from test_runtime_replan import _replan_inputs

from dronedream_agent_core.collision import (
    assess_tracking_corridor_budget,
    localization_required_clearance_m,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_replan import (
    RuntimeReplanError,
    _hold_localization_variance,
    build_runtime_replacement,
    replacement_localization_budget_accepted,
)


# 功能：用多数量级方差验证搜索余量能覆盖执行预算，而不是重复使用三厘米默认值。
# 输入：variance：明确的测试方差，单位平方米。
# 输出：无。
@pytest.mark.parametrize("variance", [1e-9, 0.0001, 0.001, 0.01, 0.0537, 1.0, 10_000.0])
def test_planner_clearance_funds_same_live_budget(variance):
    clearance = localization_required_clearance_m(variance)
    assert assess_tracking_corridor_budget(clearance, variance)["funded"]


# 功能：不把缺失或零方差解释成完美定位，搜索和执行共享相同的证据数值边界。
# 输入：variance：无效的外部方差。
# 输出：无。
@pytest.mark.parametrize(
    "variance", [None, True, 0, -1, float("nan"), float("inf"), 10_001, "0.01", 10**400]
)
def test_planner_rejects_invalid_localization_variance(variance):
    with pytest.raises(ValueError, match="LOCALIZATION_VARIANCE_BOUND_INVALID"):
        localization_required_clearance_m(variance)


# 功能：未知、损坏、过期和未来定位不能因稳定悬停而被授权；旧回执仍能保留和读取。
# 输入：tmp_path：独立测试地图；patch：模拟旧数据或错误证据。
# 输出：无。
@pytest.mark.parametrize(
    "patch",
    [
        {"localization_variance_m2": None},
        {"localization_variance_m2": 0.0},
        {"localization_variance_m2": float("nan")},
        {"localization_variance_m2": True},
        {"localization_observed_at_unix_ms": None},
        {"stable_shift_ms": 251},
        {"stable_shift_ms": -1},
    ],
)
def test_replan_refuses_unqualified_localization(tmp_path, patch):
    inputs = _replan_inputs(tmp_path)
    ack = inputs["acknowledgement"]
    update = dict(patch)
    if "stable_shift_ms" in update:
        update["stable_at"] = ack.stable_at + timedelta(milliseconds=update.pop("stable_shift_ms"))
    ack = ack.model_copy(update=update)
    with pytest.raises(RuntimeReplanError, match="LOCALIZATION_EVIDENCE"):
        _hold_localization_variance(ack)


# 功能：相同地图和同一路线不得因是重规划就忽略增大的定位误差。
# 输入：tmp_path：独立测试地图，现有路线净空不足以容纳大方差。
# 输出：无。
def test_actual_replan_rejects_route_beyond_live_localization(tmp_path):
    inputs = _replan_inputs(tmp_path)
    semantic = inputs["semantic_path"]
    geometry = json.loads(semantic.read_text(encoding="utf-8"))
    geometry["collision_primitives"][0].update(center_x=2.5, center_y=1.2, size_x=20, size_z=10)
    semantic.write_text(json.dumps(geometry), encoding="utf-8")
    inputs["expected_semantic_sha256"] = hashlib.sha256(semantic.read_bytes()).hexdigest()
    assert build_runtime_replacement(**inputs).deterministic_gates[
        "replacement_tracking_corridor_budget_accepted"
    ]
    ack = inputs["acknowledgement"].model_copy(update={"localization_variance_m2": 0.0537})
    inputs["acknowledgement"] = ack
    inputs["decision"] = inputs["decision"].model_copy(update={"hold_ack_sha256": sha256_json(ack)})
    with pytest.raises(RuntimeReplanError, match="tracking_corridor_budget"):
        build_runtime_replacement(**inputs)


# 功能：规划完成到采用路线之间必须再次核验最新定位，缺失、过期或方差增长时不得沿用旧授权。
# 输入：pytest 临时目录。
# 输出：无。
def test_route_adoption_rechecks_fresh_covariance(tmp_path):
    replacement = build_runtime_replacement(**_replan_inputs(tmp_path))
    dynamics = identity()["dynamics"]
    covariance = [0.0] * 21
    covariance[0] = covariance[6] = covariance[11] = 0.0001
    dynamics["sources"]["odometry"]["pose_covariance_upper_m2"] = covariance
    assert replacement_localization_budget_accepted(replacement, dynamics, now_unix_ms=1000)
    assert not replacement_localization_budget_accepted(replacement, {}, now_unix_ms=1000)
    assert not replacement_localization_budget_accepted(replacement, dynamics, now_unix_ms=1251)
    covariance[0] = 10_000.0
    assert not replacement_localization_budget_accepted(replacement, dynamics, now_unix_ms=1000)
