import math

import pytest
from test_dynamic_safety import _request

import dronedream_agent_core.dynamic_safety as safety


# 功能：
#   验证公共安全决策入口重新验证被外部修改的合同，不把字符串健康标志等视作有效证据。
# 输入：
#   change：非法配置或观测字段覆盖。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change",
    [
        {"perception_stream_healthy": "false"},
        {"max_speed_mps": True},
        {"required_clearance_m": -1.0},
        {"prediction_horizon_seconds": 30.0},
    ],
)
def test_public_safety_entry_revalidates_changed_contract(change):
    request = _request().model_copy(update=change)
    with pytest.raises(ValueError):
        safety.predictive_safety_decision(request, [])
    with pytest.raises(ValueError):
        safety.predictive_braking_decision(request, [], issue_codes=["TEST_VETO"])


# 功能：
#   验证空字典不被当成空地图，过量几何也不能进入无界候选搜索。
# 输入：
#   primitives：非法地图容器或超量列表。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("primitives", [{}, (), pytest.param([{}] * 100_001, id="over-capacity")])
def test_public_safety_rejects_non_list_geometry(primitives):
    with pytest.raises(ValueError):
        safety.predictive_safety_decision(_request(), primitives)


# 功能：
#   验证评估期间原请求被修改时，控制输出仍使用已经验证的请求快照。
# 输入：
#   monkeypatch：在预测阶段修改原请求的测试工具。
# 输出：
#   None：不返回业务数据。
def test_control_result_is_bound_to_owned_request(monkeypatch):
    request = _request().model_copy(update={"requested_yaw_rate_dps": 15.0})
    original = safety._predict

    # 功能：
    #   在预测已经开始后模拟调用方修改自己的请求。
    # 输入：
    #   admitted、velocity、geometry：评估器当前所用的输入。
    # 输出：
    #   prediction：真实预测函数的结果。
    def change_request(admitted, velocity, geometry):
        request.requested_yaw_rate_dps = 90.0
        prediction = original(admitted, velocity, geometry)
        return prediction

    monkeypatch.setattr(safety, "_predict", change_request)
    decision = safety.predictive_safety_decision(request, [])
    assert decision.selected_yaw_rate_dps == 15.0


# 功能：
#   验证评估中原地图发生变化时，不会改写本次已经接纳的几何。
# 输入：
#   monkeypatch：在计算开始时修改原地图。
# 输出：
#   None：不返回业务数据。
def test_prediction_owns_its_geometry(monkeypatch):
    wall = {
        "center_x": 100.0,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 1.0,
        "size_y": 3.0,
        "size_z": 3.0,
    }
    geometry = [wall]
    baseline = safety.predictive_safety_decision(_request(), geometry)
    original = safety._predict

    # 功能：
    #   注入原始地图更新但不触碰评估器私有参数。
    # 输入：
    #   request、velocity、admitted：当前计算上下文。
    # 输出：
    #   prediction：真实预测结果。
    def change_geometry(request, velocity, admitted):
        wall["center_x"] = 0.0
        prediction = original(request, velocity, admitted)
        return prediction

    monkeypatch.setattr(safety, "_predict", change_geometry)
    assert safety.predictive_safety_decision(_request(), geometry) == baseline


# 功能：
#   验证大但有限的速度向量限幅保留方向，不因平方溢出变成零向量。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_vector_limiting_preserves_large_finite_direction():
    assert safety._limit_magnitude((1.0e200, 1.0e200, 0.0), 1.0) == pytest.approx(
        (1 / math.sqrt(2), 1 / math.sqrt(2), 0.0)
    )


# 功能：
#   验证搜索包围球覆盖实际用于净空计算的倾斜外包箱，而不只覆盖未膨胀箱体。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_culling_radius_covers_tilted_collision_envelope():
    primitive = {
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size_x": 2.0,
        "size_y": 2.0,
        "size_z": 2.0,
        "pitch_rad": math.pi / 4,
    }
    assert safety._primitive_enclosing_radius(primitive) == pytest.approx(math.sqrt(5))
