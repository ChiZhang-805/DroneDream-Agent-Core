import math

import pytest

from dronedream_agent_core.contracts import (
    OnboardPerceptionFrame,
    RangeRayObservation,
    Vector3,
)
from dronedream_agent_core.depth_obstacle_tracker import DepthMotionTracker


# 功能：
#   构造低定位噪声、三个不重复端点的深度簇，时间由序号确定；仅用作合成测试。
# 输入：
#   sequence：从一开始的扫描序号。
#   center_x：簇中心的世界 X 坐标，单位米。
# 输出：
#   frame：与序号绑定的深度帧。
def _frame(sequence: int, center_x: float) -> OnboardPerceptionFrame:
    offsets = (-0.12, 0.0, 0.12)
    frame = OnboardPerceptionFrame(
        sensor_id="oakd-lite-depth",
        sequence=sequence,
        observed_at_unix_ms=sequence * 100,
        localization_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        localization_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.0001,  # Synthetic low-noise motion fixture.
        range_rays=[
            RangeRayObservation(
                origin_m=Vector3(x=0.0, y=0.0, z=1.0),
                endpoint_m=Vector3(x=center_x, y=offset, z=1.0 + offset),
                hit=True,
                confidence=0.95,
                observed_at_monotonic_seconds=sequence * 0.1,
            )
            for offset in offsets
        ],
    )
    return frame


# 功能：
#   验证第一帧不凭空宣布运动，重复观测位移后才产生预测轨迹。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_repeated_moving_depth_cluster_becomes_predictive_track() -> None:
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)

    assert tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1) == []
    tracks = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)

    assert len(tracks) == 1
    assert tracks[0].obstacle_id == "depth-track-1"
    assert tracks[0].velocity_mps.x > 0.1
    assert tracks[0].confidence >= 0.55


# 功能：
#   验证重复出现但没有位移的簇不被报告为移动目标。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_static_depth_cluster_is_not_misreported_as_moving() -> None:
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)

    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    tracks = tracker.update(_frame(2, 1.0), observed_at_monotonic_seconds=0.2)

    assert tracks == []


# 功能：天花板可见斑块的横向质心变化不足以证明物体移动；输入：平面合成深度；输出：无伪速度。
def test_planar_tangential_visibility_shift_does_not_confirm_motion():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)
    for sequence in range(1, 8):
        frame = _frame(sequence, 3.0)
        for ray, offset in zip(frame.range_rays, (-0.12, 0.0, 0.12), strict=True):
            ray.endpoint_m = Vector3(x=3.0 + offset, y=sequence * 0.1 + offset, z=3.618)
        original = frame.model_dump(mode="json")
        assert tracker.update(frame, observed_at_monotonic_seconds=sequence * 0.1) == []
        assert frame.model_dump(mode="json") == original


# 功能：同一平面若沿法向接近仍能确认运动；输入：真实法向位移的合成簇；输出：动态假设。
def test_planar_normal_approach_remains_observable():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)
    for sequence in (1, 2):
        frame = _frame(sequence, 3.0)
        for ray, offset in zip(frame.range_rays, (-0.12, 0.0, 0.12), strict=True):
            ray.endpoint_m = Vector3(x=3.0 + offset, y=offset, z=3.8 - sequence * 0.2)
        result = tracker.update(frame, observed_at_monotonic_seconds=sequence * 0.1)
    assert len(result) == 1 and result[0].velocity_mps.z < 0


# 功能：
#   验证已移动目标停下后身份连续、年龄刷新且速度衰减，而非留下继续运动的旧影。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_observed_person_stopping_does_not_become_a_lost_moving_ghost():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)
    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    moving = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)[0]
    for sequence in range(3, 30):
        observed = tracker.update(
            _frame(sequence, 1.2), observed_at_monotonic_seconds=sequence * 0.1
        )
        assert len(observed) == 1
        assert observed[0].obstacle_id == moving.obstacle_id
        assert observed[0].age_seconds == 0
        assert observed[0].position_m == moving.position_m
    assert observed[0].velocity_mps.x < 0.001


# 功能：
#   验证定位误差范围内的抖动不产生可靠运动声明，同时保留全部原始射线。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_localization_noise_is_not_differentiated_into_a_confident_moving_object():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)
    for sequence, position in enumerate((1, 1.05, 0.98, 1.06, 1.02, 0.97, 1.04), start=1):
        frame = _frame(sequence, position).model_copy(update={"localization_covariance_m2": 0.05})
        original_rays = list(frame.range_rays)
        assert tracker.update(frame, observed_at_monotonic_seconds=sequence * 0.1) == []
        assert frame.range_rays == original_rays  # Immediate geometric safety is unchanged.


# 功能：
#   验证速度虽未超限但加速度跳变不合理时，保留旧观测并增加年龄。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_physically_impossible_centroid_acceleration_retains_original_sample():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1, maximum_dynamic_acceleration_mps2=8)
    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    measured = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)[0]
    # Raw speed is under the 5 m/s limit, but the change is physically invalid.
    rejected = tracker.update(_frame(3, 0.8), observed_at_monotonic_seconds=0.3)[0]
    assert rejected.position_m == measured.position_m
    assert rejected.velocity_mps == measured.velocity_mps
    assert rejected.age_seconds == pytest.approx(0.1)


# 功能：
#   验证当前静态地图能解释的墙面端点不会进入动态关联。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_qualified_static_geometry_is_removed_before_motion_association() -> None:
    tracker = DepthMotionTracker(
        known_static_primitives=[
            {
                "name": "wall",
                "shape": "box",
                "center_x": 1.1,
                "center_y": 0.0,
                "center_z": 1.0,
                "size_x": 1.0,
                "size_y": 1.0,
                "size_z": 1.0,
                "yaw_rad": 0.0,
            }
        ],
        minimum_dynamic_speed_mps=0.1,
    )

    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    tracks = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)

    assert tracks == []


# 功能：
#   验证定位余量覆盖墙边投影误差，但不清除原射线或屏蔽确实远离墙体的运动簇。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_uncertain_wall_projection_is_not_a_moving_person_and_keeps_raw_hits():
    wall = {
        "center_x": 2.1,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 0.1,
        "size_y": 8.0,
        "size_z": 3.0,
        "yaw_rad": 0.0,
    }
    tracker = DepthMotionTracker(known_static_primitives=[wall])
    for sequence, position in enumerate((1.7, 1.9, 1.75, 1.95), start=1):
        frame = _frame(sequence, position).model_copy(update={"localization_covariance_m2": 0.06})
        original = frame.model_dump()
        assert tracker.update(frame, observed_at_monotonic_seconds=sequence * 0.1) == []
        assert frame.model_dump() == original  # Exclusion never clears geometric obstacles.
    # A distinctly separate cluster still produces measured predictive motion.
    separate = DepthMotionTracker(known_static_primitives=[wall])
    assert separate.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1) == []
    assert len(separate.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)) == 1


# 功能：
#   验证暂时看不到目标时仍保留原始观测时间，再次观测后沿用身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_missing_depth_detection_retains_original_observation_and_ages():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)
    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    fresh = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)[0]
    empty = _frame(3, 1.3)
    for ray in empty.range_rays:
        ray.hit = False  # A real scan without target hits, not an invalid empty packet.
    retained = tracker.update(empty, observed_at_monotonic_seconds=0.3)[0]
    assert retained.position_m == fresh.position_m
    assert retained.velocity_mps == fresh.velocity_mps
    assert retained.age_seconds == pytest.approx(0.1)
    returned = tracker.update(_frame(4, 1.4), observed_at_monotonic_seconds=0.4)[0]
    assert returned.obstacle_id == fresh.obstacle_id
    assert returned.age_seconds == 0


# 功能：
#   验证速度超限的错误关联不会删除已有障碍，也不会伪造新鲜位置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rejected_velocity_does_not_erase_existing_track():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1)
    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    fresh = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)[0]
    invalid = tracker.update(_frame(3, 2.2), observed_at_monotonic_seconds=0.3)[0]
    assert invalid.position_m == fresh.position_m
    assert invalid.age_seconds == pytest.approx(0.1)


# 功能：
#   验证目标移出旧位置的关联半径后，可依据上一速度的预测位置正确匹配。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_prediction_associates_target_outside_last_center_gate():
    tracker = DepthMotionTracker(minimum_dynamic_speed_mps=0.1, association_distance_m=0.3)
    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    first = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)[0]
    resumed = tracker.update(_frame(6, 1.65), observed_at_monotonic_seconds=0.6)[0]
    assert resumed.obstacle_id == first.obstacle_id
    assert resumed.position_m.x == pytest.approx(1.65)


# 功能：
#   验证重复、倒退或非有限的时钟不能推进跟踪状态。
# 输入：
#   bad_time：本次注入的非法时钟。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("bad_time", [0.1, 0.05, math.nan, math.inf])
def test_tracker_rejects_replayed_or_invalid_clock(bad_time):
    tracker = DepthMotionTracker()
    tracker.update(_frame(1, 1), observed_at_monotonic_seconds=0.1)
    with pytest.raises(ValueError, match="TIMESTAMP_NOT_MONOTONIC"):
        tracker.update(_frame(2, 1.1), observed_at_monotonic_seconds=bad_time)


# 功能：
#   验证非有限或负数物理配置在创建跟踪器时被拒绝。
# 输入：
#   kwargs：待覆盖的无效配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"association_distance_m": math.nan},
        {"maximum_track_age_seconds": math.inf},
        {"static_exclusion_m": -0.1},
        {"voxel_size_m": math.inf},
    ],
)
def test_tracker_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DepthMotionTracker(**kwargs)
