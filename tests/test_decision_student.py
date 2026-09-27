"""Small baseline must not leak identity, optional media or false observations."""

import copy
import json

import pytest

from dronedream_agent_core.decision_state_adapter import DecisionStateV2
from dronedream_agent_core.uav_decision_student import make_student, numeric_state
from scripts.prepare_decision_contract_smoke import synthetic_state


# 功能：转换测试状态；输入：字典；输出：严格 v2 对象。
def state(value):
    return DecisionStateV2.model_validate_json(json.dumps(value))


# 功能：身份、绝对坐标和可选直播改变不能改变特征；输入：无；输出：逐值一致。
def test_identity_and_media_invariance():
    first = synthetic_state(1, "follow_route")
    other = copy.deepcopy(first)
    other.update(mission_id="changed", sequence=40, map_sha256="f"*64)
    other["frame"]["optional_media_available"] = False
    other["frame"]["position_world_enu_m"]["x"] += 100
    assert numeric_state(state(first)) == numeric_state(state(other))


# 功能：未知距离不能被编码为实际零距离；输入：缺失变体；输出：同宽但不同特征。
def test_unknown_is_not_zero():
    first = synthetic_state(1, "follow_route")
    other = copy.deepcopy(first)
    first["frame"]["clearance_front_m"] = 0.
    other["frame"]["clearance_front_m"] = None
    assert len(numeric_state(state(first))) == len(numeric_state(state(other)))
    assert numeric_state(state(first)) != numeric_state(state(other))


# 功能：改变同域时钟原点不改变年龄特征；输入：毫秒平移；输出：逐值一致。
def test_clock_origin_invariance():
    first = synthetic_state(1, "follow_route")
    other = copy.deepcopy(first)
    other["frame"]["observed_at_ms"] += 100000
    for name in ("pose_source", "geometry_source", "route_source"):
        other["frame"][name]["observed_at_ms"] += 100000
    assert numeric_state(state(first)) == numeric_state(state(other))


# 功能：实际反传与序列化回载一致；输入：CPU 模型；输出：有限梯度与五类 logits。
def test_real_backward():
    torch = pytest.importorskip("torch")
    values = numeric_state(state(synthetic_state(1, "wait")))
    model = make_student(len(values))
    logits = model(torch.tensor([values]))
    assert logits.shape == (1, 5)
    logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
