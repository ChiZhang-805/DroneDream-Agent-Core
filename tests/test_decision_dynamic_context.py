"""Synthetic crossing geometry checks; not collected decision labels."""

import copy

import pytest
from test_decision_runtime_adapter import snapshot

from dronedream_agent_core.decision_dynamic_context import dynamic_context
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：冻结合成导航快照；输入：已修改的字典；输出：带内容摘要的快照。
def seal(value):
    value.pop("snapshot_sha256", None)
    value["snapshot_sha256"] = decision_digest(value)
    return value


# 功能：构造横穿前方路线的两帧真实格式夹具，不计入正式采集。
# 输入：无；输出：相隔 100 ms 的世界 ENU 跟踪记录。
def pair():
    old = snapshot()
    old.update(
        current_position_m=dict(x=0.0, y=0.0, z=1.5),
        goal_position_m=dict(x=3.0, y=0.0, z=1.5),
        current_velocity_mps=dict(x=0.2, y=0.0, z=0.0),
        dynamic_obstacles=[
            dict(
                obstacle_id="person-1",
                position_m=dict(x=1.0, y=1.0, z=1.5),
                velocity_mps=dict(x=0.0, y=-1.0, z=0.0),
                radius_m=0.2,
                height_m=1.7,
                observation_age_seconds=0.0,
                confidence=0.9,
            )
        ],
    )
    new = copy.deepcopy(old)
    new["control_reference_observed_at_unix_ms"] += 100
    new["dynamic_obstacles"][0]["position_m"]["y"] = 0.9
    return seal(old), seal(new)


# 功能：调用只读阶段上下文计算；输入：快照对；输出：无控制权限的上下文。
def derive(old, new):
    return dynamic_context(
        new, old, [1.0, 0.0, 0.0, 0.0], body_radius_m=0.25, body_height_m=0.3, uncertainty_m=0.02
    )


# 功能：横穿判定、FRD 符号和机器人自运动扣除均正确；输入：两帧；输出：原始时间保留。
def test_crossing_and_relative_velocity():
    old, new = pair()
    original = copy.deepcopy(new)
    result = derive(old, new)
    assert result["crossing_obstacle"] is True
    track = result["tracks"][0]
    assert track["relative_velocity_body_frd_mps"] == dict(x=-0.2, y=1.0, z=-0.0)
    assert track["relative_position_body_frd_m"] == dict(x=1.0, y=-0.9, z=-0.0)
    assert track["source"]["observed_at_ms"] == 10100
    assert new == original


# 功能：消失、单帧、过期、重复原采样、低置信、速度不一致不能变成确定空场。
# 输入：不同缺口；输出：unknown，不捏造恢复或速度。
@pytest.mark.parametrize(
    "fault",
    ["empty", "single", "stale", "same-sample", "confidence", "velocity", "gap", "goal", "route"],
)
def test_unknown_tracks_do_not_become_clear(fault):
    old, new = pair()
    if fault == "empty":
        new["dynamic_obstacles"] = []
    elif fault == "single":
        old = None
    elif fault == "stale":
        new["dynamic_obstacles"][0]["observation_age_seconds"] = 0.3
    elif fault == "same-sample":
        new["dynamic_obstacles"][0]["observation_age_seconds"] = 0.1
    elif fault == "confidence":
        new["dynamic_obstacles"][0]["confidence"] = 0.1
    elif fault == "velocity":
        new["dynamic_obstacles"][0]["velocity_mps"]["y"] = 4.0
    elif fault == "gap":
        new["control_reference_observed_at_unix_ms"] += 1000
    elif fault == "goal":
        new["strategic_context"]["task"]["navigation_goal_id"] = "different-goal"
    else:
        new["known_static_map"]["qualified_route_sha256"] = "f" * 64
    result = derive(old, seal(new))
    assert result["crossing_obstacle"] is None
    assert result["issues"]


# 功能：远处、高处和静止目标不会仅因为存在就成为“横穿”。
# 输入：两帧一致修改；输出：已观测目标不横穿，但不授予自由空间结论。
@pytest.mark.parametrize("kind", ["far", "high", "stationary", "away"])
def test_non_crossing_targets(kind):
    old, new = pair()
    for value in (old, new):
        row = value["dynamic_obstacles"][0]
        if kind == "far":
            row["position_m"]["x"] = 8.0
        elif kind == "high":
            row["position_m"]["z"] = 5.0
        elif kind == "stationary":
            row["position_m"]["y"] = 1.0
            row["velocity_mps"]["y"] = 0.0
        else:
            row["velocity_mps"]["y"] = 1.0
            row["position_m"]["y"] = 2.0 if value is old else 2.1
        seal(value)
    result = derive(old, new)
    assert result["crossing_obstacle"] is False
    assert result["scope"] == "observed-tracks-only-not-free-space"


# 功能：拒绝篡改、真值注入和重号目标；输入：无效来源；输出：显式失败。
@pytest.mark.parametrize("fault", ["hash", "truth", "duplicate"])
def test_source_integrity(fault):
    old, new = pair()
    if fault == "hash":
        new["dynamic_obstacles"][0]["confidence"] = 0.5
    elif fault == "truth":
        new["source_of_truth"] = "simulation-ground-truth"
        seal(new)
    else:
        new["dynamic_obstacles"].append(copy.deepcopy(new["dynamic_obstacles"][0]))
        seal(new)
    with pytest.raises(ValueError):
        derive(old, new)
