import copy

import pytest
from test_offline_flight_learning import UnitEnvironment

from dronedream_agent_core.training.flight_environment import (
    FlightReward,
    PilotAction,
    RewardConfig,
)


# 功能：
#   生成可独立修改的合成训练步，不启动模拟器或使用真实飞行证据。
# 输入：
#   无。
# 输出：
#   step：带合法奖励见证的一个保持动作结果。
def training_step():
    env = UnitEnvironment()
    env.reset(seed=1)
    step = env.step(PilotAction(mode="hold", axes=[0.] * 4))
    return step


# 功能：
#   验证奖励器独立持有构造配置，不随调用方后续修改而改变奖励规则。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reward_owns_initial_configuration():
    config = RewardConfig(time_per_second=0.1)
    reward = FlightReward(config)
    config.time_per_second = 10.
    result = reward.score(training_step())
    assert result["time"] == pytest.approx(-0.005)


# 功能：
#   验证错误配置不会因假值回退或未重新验证而被接纳。
# 输入：
#   config：非法配置对象或绕过赋值验证的模型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("config", [{}, False,
                                   RewardConfig().model_copy(update={"failure": float("nan")})])
def test_reward_rejects_invalid_constructor_configuration(config):
    with pytest.raises(ValueError):
        FlightReward(config)


# 功能：
#   验证被绕过验证篡改的奖励见证在修改账本前被拒绝，失败不留下半更新状态。
# 输入：
#   field：要篡改的见证字段。
#   value：字段的非法数值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("goal_distance_m", -1.),
                                         ("elapsed_seconds", float("nan")),
                                         ("power_joules", float("inf"))])
def test_invalid_step_cannot_mutate_reward_ledger(field, value):
    reward, step = FlightReward(), training_step()
    initial = copy.deepcopy(reward.__dict__)
    invalid = step.model_copy(update={
        "evidence": step.evidence.model_copy(update={field: value}),
    })
    with pytest.raises(ValueError):
        reward.score(invalid)
    assert reward.__dict__ == initial
    assert reward.score(step)["progress"] == 0.


# 功能：
#   验证有限输入乘法溢出和多项总和溢出都不产生可训练的无穷奖励。
# 输入：
#   aggregate：是否构造每项有限但总和溢出的情形。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("aggregate", [False, True])
def test_reward_overflow_is_rejected_before_ledger_commit(aggregate):
    config = RewardConfig(time_per_second=1e308, energy_per_joule=1e308,
                          action_change=1e308)
    reward, step = FlightReward(config), training_step()
    step.evidence.elapsed_seconds = 1. if aggregate else 60.
    step.evidence.power_joules = 1.
    step.evidence.commanded_change_squared = 1.
    initial = copy.deepcopy(reward.__dict__)
    with pytest.raises(ValueError, match="REWARD_NONFINITE"):
        reward.score(step)
    assert reward.__dict__ == initial


# 功能：
#   验证绕过模型验证制造的失败且成功状态矛盾，不能污染奖励终止账本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reward_rechecks_terminal_evidence_consistency():
    reward, step = FlightReward(), training_step()
    step = step.model_copy(update={"terminated": True, "evidence": step.evidence.model_copy(
        update={"collision": True, "verified_mission_complete": True},
    )})
    with pytest.raises(ValueError, match="FAILED_FLIGHT_CANNOT_BE_SUCCESS"):
        reward.score(step)
    assert not reward.ended and not reward.mission_awarded
