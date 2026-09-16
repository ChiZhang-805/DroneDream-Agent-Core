import math
from dataclasses import replace

import pytest
from test_depth_obstacle_tracker import _frame

from dronedream_agent_core.depth_obstacle_tracker import DepthMotionTracker, _StaticPrimitiveIndex


# 功能：
#   验证静态索引保留俯仰后跨越原竖直包围盒的障碍，不把墙面当成移动目标。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tilted_static_structure_is_indexed_at_its_rotated_extent():
    wall = {
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 1.0,
        "size_x": 1.0,
        "size_y": 1.0,
        "size_z": 12.0,
        "pitch_rad": math.pi / 2,
    }
    tracker = DepthMotionTracker(known_static_primitives=[wall])
    assert tracker.update(_frame(1, 4.0), observed_at_monotonic_seconds=0.1) == []
    assert tracker.update(_frame(2, 4.2), observed_at_monotonic_seconds=0.2) == []
    assert tracker._tracks == {}


# 功能：
#   验证数量、物理参数和静态地图入口不接受布尔、分数计数及错误容器。
# 输入：
#   kwargs：一个非法配置覆盖。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_points": True},
        {"minimum_points": 1.5},
        {"minimum_points": math.nan},
        {"voxel_size_m": True},
        {"maximum_dynamic_speed_mps": True},
        {"static_exclusion_m": False},
        {"known_static_primitives": {}},
        {"known_static_primitives": [{"size_x": 1.0, "size_y": 1.0, "size_z": 1.0}]},
    ],
)
def test_tracker_configuration_rejects_coercion_and_incomplete_geometry(kwargs):
    with pytest.raises(ValueError):
        DepthMotionTracker(**kwargs)


# 功能：
#   验证复制数据类并保留所有者标记不能伪造可提交的跟踪状态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_cloned_proposal_cannot_commit_forged_history():
    tracker = DepthMotionTracker()
    prepared = tracker.prepare(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    with pytest.raises(ValueError, match="SUPERSEDED"):
        tracker.commit(replace(prepared, sequence=900))
    tracker.commit(prepared)
    assert tracker._sequence == 1


# 功能：
#   验证同一历史上的较新候选生成后，较旧候选不能再抢先提交。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_new_successful_proposal_supersedes_previous_uncommitted_proposal():
    tracker = DepthMotionTracker()
    earlier = tracker.prepare(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    latest = tracker.prepare(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)
    with pytest.raises(ValueError, match="SUPERSEDED"):
        tracker.commit(earlier)
    tracker.commit(latest)
    assert tracker._sequence == 2
    assert tracker._tracks[1].seen_count == 1


# 功能：
#   验证对外部融合候选的修改不能被当作原始跟踪结果提交。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_candidate_observation_rejected_without_advancing_history():
    tracker = DepthMotionTracker()
    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    prepared = tracker.prepare(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)
    prepared.observations[0].position_m.x = 800.0
    with pytest.raises(ValueError, match="MODIFIED"):
        tracker.commit(prepared)
    assert tracker._sequence == 1
    assert tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)[0].position_m.x == 1.2


# 功能：
#   验证篡改后的帧在跟踪入口再次严格检查，而非把字符串序号转换成整数。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_frame_requires_exact_contract_types():
    tracker = DepthMotionTracker()
    frame = _frame(1, 1.0).model_copy(update={"sequence": "1"})
    with pytest.raises(ValueError):
        tracker.prepare(frame, observed_at_monotonic_seconds=0.1)
    assert tracker._sequence == 0


# 功能：
#   验证布尔值不能冒充单调时钟，即使数值恰好等于射线时钟。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_boolean_clock_is_not_a_measured_timestamp():
    with pytest.raises(ValueError, match="TIMESTAMP"):
        DepthMotionTracker().prepare(_frame(10, 1.0), observed_at_monotonic_seconds=True)


# 功能：
#   验证静态查询的负数与布尔不确定度不会缩小障碍保护范围。
# 输入：
#   uncertainty：非法膨胀距离。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("uncertainty", [-1.0, True])
def test_static_index_rejects_invalid_uncertainty(uncertainty):
    primitive = {"center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "radius_m": 1.0}
    index = _StaticPrimitiveIndex([primitive], exclusion_m=0.1)
    with pytest.raises(ValueError):
        index.contains_known_static((0.0, 0.0, 0.0), uncertainty_m=uncertainty)


# 功能：
#   验证大量小基元触及总索引预算后仍参与查询，不靠丢弃地图控制内存。
# 输入：
#   monkeypatch：把总索引预算缩小以验证边界的测试工具。
# 输出：
#   None：不返回业务数据。
def test_total_index_budget_keeps_unindexed_primitives(monkeypatch):
    monkeypatch.setattr(_StaticPrimitiveIndex, "_MAX_BIN_ENTRIES", 32)
    primitives = [
        {"center_x": float(i * 4), "center_y": 0.0, "center_z": 0.0, "radius_m": 0.4}
        for i in range(20)
    ]
    index = _StaticPrimitiveIndex(primitives, exclusion_m=0.1)
    assert sum(len(indices) for indices in index._bins.values()) <= 32
    assert index._large_primitives
    assert all(index.contains_known_static((float(i * 4), 0.0, 0.0)) for i in range(20))
    primitives[-1]["radius_m"] = 1.0e6
    assert not index.contains_known_static((1000.0, 0.0, 0.0))


# 功能：
#   验证巨大查询半径转为有限几何扫描，不枚举数十亿空网格。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_wide_uncertainty_query_uses_complete_direct_scan():
    primitive = {"center_x": 5000.0, "center_y": 0.0, "center_z": 0.0, "radius_m": 1.0}
    index = _StaticPrimitiveIndex([primitive], exclusion_m=0.1)
    assert index.contains_known_static((0.0, 0.0, 0.0), uncertainty_m=10_000.0)


# 功能：
#   验证失败的新帧不破坏上一份有效候选，成功提交后不能重复提交同一候选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_invalid_prepare_leaves_valid_candidate_committable_once():
    tracker = DepthMotionTracker()
    prepared = tracker.prepare(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    with pytest.raises(ValueError, match="CLOCK_MISMATCH"):
        tracker.prepare(_frame(2, 1.2), observed_at_monotonic_seconds=0.5)
    tracker.commit(prepared)
    with pytest.raises(ValueError, match="SUPERSEDED"):
        tracker.commit(prepared)


# 功能：
#   验证极小但有限的体素尺度不能在格索引计算中产生无穷值，失败不推进历史。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_overflowing_voxel_projection_fails_before_history_commit():
    tracker = DepthMotionTracker(voxel_size_m=1.0e-320)
    with pytest.raises(ValueError, match="VOXEL_SCALE"):
        tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    assert tracker._sequence == 0


# 功能：
#   验证超龄后不会恢复旧速度或旧身份，重新出现的目标需要新的重复观测。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_expired_track_is_not_reused_as_fresh_velocity():
    tracker = DepthMotionTracker()
    tracker.update(_frame(1, 1.0), observed_at_monotonic_seconds=0.1)
    earlier = tracker.update(_frame(2, 1.2), observed_at_monotonic_seconds=0.2)[0]
    assert tracker.update(_frame(20, 1.4), observed_at_monotonic_seconds=2.0) == []
    fresh = tracker.update(_frame(21, 1.6), observed_at_monotonic_seconds=2.1)[0]
    assert fresh.obstacle_id != earlier.obstacle_id
    assert fresh.age_seconds == 0


# 功能：
#   验证空间索引包含最终净空判定的一毫米探针余量，网格交界处不漏掉候选。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_bin_boundary_includes_clearance_probe_extent():
    primitive = {"center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "radius_m": 1.9995}
    index = _StaticPrimitiveIndex([primitive], exclusion_m=0.0)
    assert index.contains_known_static((2.0002, 0.0, 0.0))
