"""Independent reward facts must be finite, immutable and receipt-bound."""

import pytest
from test_training_runtime_evidence import WALL, pose

from dronedream_agent_core.contracts import DynamicObstacleObservation, Vector3
from dronedream_agent_core.training.outcome_verifier import (
    OutcomeEnvelope,
    SimulationOutcomeVerifier,
    swept_clearance_bound,
)


# 功能：
#   构造固定地图和两帧真实结构的合成评分窗口，不赋予任务完成资格。
# 输入：
#   无。
# 输出：
#   verifier：独立几何评分器。
#   arguments：与两个见证时刻一致的评估参数。
def evaluation_fixture():
    verifier = SimulationOutcomeVerifier([dict(WALL)],
                                         OutcomeEnvelope(.1, .2, (-3, -3, 0), (3, 3, 3)))
    arguments = dict(observations=[pose(1, -1), pose(2, 1)], start_ms=1050, end_ms=1100,
                     stage_id="office", goal_revision="a", goal_position_m=(2, 0, 1),
                     safety_intervened=False, commanded_change_squared=0., power_joules=None)
    return verifier, arguments


# 功能：
#   验证冻结机体包络不会继续借用调用方可变围栏边界列表。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_envelope_owns_immutable_bounds():
    lower = [-3., -3., 0.]
    envelope = OutcomeEnvelope(.1, .2, lower, [3., 3., 3.])
    lower[0] = -300.
    assert envelope.minimum_enu_m == (-3., -3., 0.)
    assert envelope.maximum_enu_m == (3., 3., 3.)


# 功能：
#   验证布尔值不能因为 Python 将其视为整数而充当机体尺度或围栏坐标。
# 输入：
#   field：要替换成布尔值的包络字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["body_radius_m", "body_height_m", "minimum_enu_m"])
def test_envelope_rejects_boolean_geometry(field):
    values = dict(body_radius_m=.1, body_height_m=.2,
                  minimum_enu_m=(-3., -3., 0.), maximum_enu_m=(3., 3., 3.))
    values[field] = (False, -3., 0.) if field == "minimum_enu_m" else True
    with pytest.raises(ValueError):
        OutcomeEnvelope(**values)


# 功能：
#   验证评分器不接纳修改后非法或过旧的独立观测。
# 输入：
#   change：绕过赋值校验注入的损坏字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", [{"sequence": -1}, {"stream_healthy": "yes"},
                                    {"stream_age_seconds": .2}])
def test_outcome_revalidates_witnesses(change):
    verifier, arguments = evaluation_fixture()
    arguments["observations"][0] = arguments["observations"][0].model_copy(update=change)
    with pytest.raises(ValueError):
        verifier.evaluate(**arguments)


# 功能：
#   验证时间仍在允许间隔内时，缺少中间序号也不能生成连续窗口奖励。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_outcome_rejects_missing_sequence():
    verifier, arguments = evaluation_fixture()
    arguments["observations"][1].sequence = 3
    with pytest.raises(ValueError):
        verifier.evaluate(**arguments)


# 功能：
#   验证相同动态目标 ID 的两条见证不会在字典转换时悄悄丢失其中一条。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_duplicate_dynamic_identity_is_not_silently_collapsed():
    verifier, arguments = evaluation_fixture()
    obstacle = DynamicObstacleObservation(obstacle_id="person", position_m=Vector3(x=0, y=2, z=1),
                                           velocity_mps=Vector3(x=0, y=0, z=0), radius_m=.3,
                                           height_m=1.7, confidence=1., age_seconds=0.)
    for row in arguments["observations"]:
        row.dynamic_obstacles = [obstacle, obstacle.model_copy(deep=True)]
    with pytest.raises(ValueError):
        verifier.evaluate(**arguments)


# 功能：
#   验证影响奖励的干预、提前落地和动作变化成本都被独立回执摘要绑定。
# 输入：
#   field：改变的奖励事实。
#   value：替代事实值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("safety_intervened", True),
                                       ("premature_landing", True),
                                       ("commanded_change_squared", 0.5)])
def test_every_reward_fact_is_bound_by_receipt(field, value):
    verifier, arguments = evaluation_fixture()
    first, _ = verifier.evaluate(**arguments)
    second, _ = verifier.evaluate(**{**arguments, field: value})
    assert first.verifier_receipt_sha256 != second.verifier_receipt_sha256


# 功能：
#   验证评分器保留的地图源与已计算摘要不受调用方原地清空影响。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_verifier_does_not_borrow_geometry_list():
    primitives = [dict(WALL)]
    verifier = SimulationOutcomeVerifier(primitives,
                                         OutcomeEnvelope(.1, .2, (-3, -3, 0), (3, 3, 3)))
    primitives.clear()
    assert verifier.primitives == [WALL]


# 功能：
#   验证标量扫掠检查拒绝错误机体尺度，不能靠负半径产生更大安全距离。
# 输入：
#   radius：非法半径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("radius", [-.1, True, float("nan")])
def test_scalar_clearance_rejects_invalid_envelope(radius):
    with pytest.raises(ValueError):
        swept_clearance_bound((-1, 0, 1), (1, 0, 1), [WALL], radius_m=radius, half_height_m=.1)
