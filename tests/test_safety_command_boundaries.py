"""Input and ownership boundaries of actual short-lived safety evaluation."""

import pytest
from test_local_safety_channel import pair
from test_runtime_local_safety import _vehicle

from dronedream_agent_core import runtime_local_safety as safety
from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation


# 功能：
#   用真实安全评估器处理合成机载观测，只替换当前用例明确传入的选项。
# 输入：
#   options：待覆盖的安全评估参数。
# 输出：
#   command：真实评估器产生的指令，不代表飞机已执行。
def evaluate(**options):
    arguments = dict(
        observation=pair()[0],
        vehicle=_vehicle(),
        static_primitives=[],
        required_clearance_m=0.35,
        generated_at_unix_ms=1020,
    )
    arguments.update(options)
    command = safety.evaluate_runtime_local_safety(**arguments)
    return command


# 功能：
#   开关和物理量必须各用正确类型，不能让布尔值、整数开关或浮点租期进入控制计算。
# 输入：
#   option：当前错误类型的评估参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "option",
    [
        {"command_horizon_seconds": True},
        {"validity_milliseconds": 500.0},
        {"maximum_speed_mps": True},
        {"route_yaw_rate_dps": True},
        {"required_clearance_m": True},
        {"tracking_recovery_active": "false"},
        {"model_navigation_authorized": 0},
    ],
)
def test_control_options_preserve_declared_types(option):
    with pytest.raises(ValueError):
        evaluate(**option)


# 功能：
#   查询几何的速度上界和净空必须是真实物理量，拒绝布尔值以免查询范围计算失真。
# 输入：
#   option：错误的几何查询选项。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "option",
    [
        {"maximum_speed_mps": True},
        {"required_clearance_m": True},
    ],
)
def test_query_radius_rejects_boolean_physical_values(option):
    arguments = dict(
        observation=pair()[0], vehicle=_vehicle(), required_clearance_m=0.35, maximum_speed_mps=1.0
    )
    arguments.update(option)
    with pytest.raises(ValueError):
        safety.runtime_safety_query_radius_m(**arguments)


# 功能：
#   指令保存的目标和姿态必须属于本次评估的私有观测，外部改动不能使其与已绑定摘要分离。
# 输入：
#   无：使用一条合成机载观测。
# 输出：
#   None：不返回业务数据。
def test_command_does_not_share_mutable_source_position():
    observation = pair()[0]
    command = evaluate(observation=observation)
    expected = command.evaluated_target_position_m.model_dump()
    observation.target_position_m.x = 999.0
    assert command.evaluated_target_position_m.model_dump() == expected


# 功能：
#   绕过模型赋值校验后注入的错误健康状态不能到达碰撞规划器。
# 输入：
#   monkeypatch：提供禁止调用规划器的测试替身。
# 输出：
#   None：不返回业务数据。
def test_invalid_nested_observation_rejected_before_planning(monkeypatch):
    observation = pair()[0].model_copy(update={"stream_healthy": "true"})

    # 功能：
    #   若错误输入进入规划器立即令测试失败，区分前置拒绝与事后输出校验。
    # 输入：
    #   args：不应到达此处的规划参数。
    # 输出：
    #   None：不返回业务数据。
    def unexpected(*args):
        pytest.fail("invalid observation reached the planner")

    monkeypatch.setattr(safety, "predictive_safety_decision", unexpected)
    with pytest.raises(ValueError):
        evaluate(observation=observation)


# 功能：
#   观测发生在评估时刻之后属于时钟矛盾，即使返回制动也不能把未来观测标为当前输入。
# 输入：
#   无：一条晚于生成时刻的合成观测。
# 输出：
#   None：不返回业务数据。
def test_future_source_cannot_issue_current_command():
    payload = pair()[0].model_dump()
    payload["observed_at_unix_ms"] = 2000
    payload["stream_healthy"] = False
    with pytest.raises(ValueError):
        evaluate(observation=RuntimeLocalSafetyObservation.model_validate(payload))
