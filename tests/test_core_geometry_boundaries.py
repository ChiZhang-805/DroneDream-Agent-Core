"""核心几何计算的数值、命中与实时工作预算反例；不替代仿真飞行验收。"""

import pytest

from dronedream_agent_core.contracts import DynamicObstacleObservation, RangeRayObservation, Vector3
from dronedream_agent_core.dynamic_clearance import (
    fresh_free_voxels,
    reachable_track_box_observed_free,
)
from dronedream_agent_core.motion_envelope import (
    interval_clearance_lower_bound,
    prediction_query_radius_m,
)


# 功能：
#   构造沿 X 轴的离线距离射线，方便独立检验命中排除规则。
# 输入：
#   start：射线起点的 X 坐标，单位米。
#   end：射线终点的 X 坐标，单位米。
#   hit：终点是否检测到障碍物。
# 输出：
#   ray：具有固定时间和可信度的测试观测。
def _ray(start=0.0, end=4.0, hit=False):
    ray = RangeRayObservation(
        origin_m=Vector3(x=start, y=0, z=0),
        endpoint_m=Vector3(x=end, y=0, z=0),
        hit=hit,
        confidence=0.95,
        observed_at_monotonic_seconds=1.0,
    )
    return ray


# 功能：
#   验证位于传感器起点的命中也阻止其他射线把该体素标为空闲。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_zero_length_hit_still_blocks_other_free_rays():
    free = fresh_free_voxels(
        [_ray(), _ray(1.5, 1.5, True)], resolution_m=0.25, now_monotonic_seconds=1.0
    )
    assert free and (6, 0, 0) not in free


# 功能：
#   验证零长度射线也消耗遍历预算，不能靠不产生投影样本绕过工作上限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_empty_rays_cannot_bypass_realtime_work_budget():
    visited = []

    # 功能：
    #   提供计数射线流，使测试能检验停止读取的位置。
    # 输入：
    #   无。
    # 输出：
    #   ray：当前零长度射线。
    def rays():
        for index in range(20):
            visited.append(index)
            ray = _ray(0.0, 0.0)
            yield ray

    free = fresh_free_voxels(
        rays(), resolution_m=0.25, now_monotonic_seconds=1.0, maximum_samples=4
    )
    assert not free
    assert len(visited) <= 5


# 功能：
#   验证绕过模型验证的非有限置信度不能作为可靠空闲观测使用。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_nan_confidence_cannot_create_free_space():
    ray = _ray().model_copy(update={"confidence": float("nan")})
    assert not fresh_free_voxels([ray], resolution_m=0.25, now_monotonic_seconds=1.0)


# 功能：
#   验证无效半径不会生成空范围并被 all 空集误判为整个目标已清空。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_negative_track_volume_cannot_be_cleared_vacuously():
    track = DynamicObstacleObservation(
        obstacle_id="target",
        position_m=Vector3(x=1, y=0, z=0),
        velocity_mps=Vector3(x=0, y=0, z=0),
        radius_m=0.1,
        height_m=0.2,
        confidence=0.9,
        age_seconds=0,
    ).model_copy(update={"radius_m": -100.0, "height_m": -200.0})
    assert not reachable_track_box_observed_free(
        track,
        age_seconds=0.1,
        free_voxels=set(),
        resolution_m=0.25,
        localization_variance_m2=0.000001,
        maximum_acceleration_mps2=1.0,
    )


# 功能：
#   验证净空计算拒绝布尔和不可表示数值，不产生数值错误类型泄漏。
# 输入：
#   values：无效的端点距离或相对行程。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("values", [(True, 1.0, 1.0), (1.0, 1.0, False), (10**400, 1.0, 1.0)])
def test_interval_rejects_nonphysical_numbers(values):
    with pytest.raises(ValueError, match="INTERVAL_INVALID"):
        interval_clearance_lower_bound(*values)


# 功能：
#   验证粗筛半径的每个物理量和速度维数都在计算前接受类型检查。
# 输入：
#   update：替换为无效输入的参数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "update",
    [
        {"maximum_speed_mps": True},
        {"horizon_seconds": True},
        {"vehicle_radius_m": True},
        {"vehicle_height_m": True},
        {"required_clearance_m": False},
        {"acceleration_speed_margin_mps": False},
        {"current_velocity_mps": (False, 0, 0)},
        {"current_velocity_mps": None},
        {"maximum_speed_mps": 10**400},
    ],
)
def test_query_rejects_ambiguous_or_unrepresentable_numbers(update):
    arguments = dict(
        current_velocity_mps=(0.0, 0.0, 0.0),
        maximum_speed_mps=1.0,
        horizon_seconds=3.0,
        vehicle_radius_m=0.2,
        vehicle_height_m=0.3,
        required_clearance_m=0.3,
        acceleration_speed_margin_mps=0.0,
    )
    arguments.update(update)
    with pytest.raises(ValueError, match="QUERY_ENVELOPE_INVALID"):
        prediction_query_radius_m(**arguments)
