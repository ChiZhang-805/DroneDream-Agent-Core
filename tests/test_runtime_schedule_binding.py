from __future__ import annotations

from dataclasses import replace

import pytest
from test_runtime_follow import _modules

from dronedream_agent_core.contracts import RuntimeCheckpoint, RuntimeCheckpointContract


# 功能：
#   构造离开起点、原地再停一次、最后返回起点的真实调度，避免用手写采样掩盖编译问题。
# 输入：
#   无。
# 输出：
#   case：基础模块、检查点执行器、原始航点和编译计划组成的元组。
def _schedule_case():
    base, executor = _modules()
    points = [base.TrackPoint(x, 0, 1) for x in (0, 1, 1, 0)]
    plan = base.build_setpoint_schedule_plan(
        points, base.ControllerParams(1, 2), 20,
        stop_at_waypoints=True, waypoint_hold_seconds=0.4,
    )
    case = base, executor, points, plan
    return case


# 功能：
#   创建绑定指定航点的动作检查点，标签只区分任务身份，不用于推断位置。
# 输入：
#   point_index：原轨迹中的航点下标。
#   number：检查点和任务的独立序号。
# 输出：
#   checkpoint：用于验证调度绑定的类型化检查点。
def _checkpoint(point_index: int, number: int = 1) -> RuntimeCheckpoint:
    checkpoint = RuntimeCheckpoint(
        checkpoint_id=f"checkpoint-{number:03d}", segment_id=f"segment-{number:03d}",
        task_id=f"task-{number}", track_point_index=point_index, target_node=f"place-{number}",
    )
    return checkpoint


# 功能：
#   同坐标不同访问必须绑定指定访问的到达采样，不能在去程或先前航点保持时提前执行。
# 输入：
#   point_index：待检验的原地第二次访问或返回起点访问。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("point_index", [2, 3])
def test_checkpoint_binds_visit_not_first_matching_position(point_index: int) -> None:
    base, executor, points, plan = _schedule_case()
    checkpoint = _checkpoint(point_index)
    contract = RuntimeCheckpointContract(contract_id="mission-test", checkpoints=[checkpoint])
    mapped = executor._schedule_checkpoint_indices(
        base, plan.schedule, points, contract, plan.track_start_index,
        waypoint_arrival_indices=plan.waypoint_arrival_indices,
    )
    assert mapped == {plan.waypoint_arrival_indices[point_index - 1]: checkpoint}


# 功能：
#   同一航点上重复定义检查点必须拒绝，不能先转字典再静默丢失其中的动作身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_rejects_duplicate_point_bindings() -> None:
    base, executor, points, plan = _schedule_case()
    contract = RuntimeCheckpointContract(
        contract_id="mission-test", checkpoints=[_checkpoint(1, 1), _checkpoint(1, 2)],
    )
    with pytest.raises(ValueError, match="duplicate"):
        executor._schedule_checkpoint_indices(
            base, plan.schedule, points, contract, plan.track_start_index,
            waypoint_arrival_indices=plan.waypoint_arrival_indices,
        )


# 功能：
#   零距离航点也要占用到达及保持预算；不足时拒绝，恰好够时不额外消耗采样。
# 输入：
#   budget：本次编译可用采样上限。
#   hold_seconds：到达之后保持的时长。
#   expected_count：合法结果的采样数，None 表示必须拒绝。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget,hold_seconds,expected_count", [
    (0, 0, None), (1, 0, 1), (4, 0.2, None), (5, 0.2, 5),
])
def test_stationary_arrival_obeys_sample_budget(budget, hold_seconds, expected_count) -> None:
    base, _ = _modules()
    points = [base.TrackPoint(0, 0, 1), base.TrackPoint(0, 0, 1)]
    params = base.ControllerParams(1, 2)
    if expected_count is None:
        with pytest.raises(ValueError, match="sample limit"):
            base.build_stopped_waypoint_schedule(
                points, params, 20, hold_seconds, initial_yaw_deg=37, max_samples=budget,
            )
    else:
        schedule, arrivals = base.build_stopped_waypoint_schedule(
            points, params, 20, hold_seconds, initial_yaw_deg=37, max_samples=budget,
        )
        assert len(schedule) == expected_count
        assert arrivals == (0,)
        assert all(sample.yaw_deg == 37 for sample in schedule)


# 功能：
#   飞控调度接受业务轨迹允许的三十秒等待上界，不再在合法合同与执行器之间发生不一致。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_schedule_supports_contract_hold_duration_upper_bound() -> None:
    base, _ = _modules()
    points = [base.TrackPoint(0, 0, 1), base.TrackPoint(0, 0, 1)]
    plan = base.build_setpoint_schedule_plan(
        points, base.ControllerParams(1, 2), 20,
        stop_at_waypoints=True, waypoint_hold_seconds=30,
    )
    arrival = plan.waypoint_arrival_indices[0]
    assert plan.schedule[arrival + 1:arrival + 601] == [plan.schedule[arrival]] * 600


# 功能：
#   航向对齐插入转向采样后，检查点必须跟随重映射下标，而不是依旧使用对齐前的位置。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_uses_yaw_remapped_arrivals() -> None:
    base, executor, points, plan = _schedule_case()
    aligned = base.align_setpoint_schedule_to_route_tangent(
        plan, measured_px4_heading_deg=25, measured_body_heading_ned_deg=0,
        rate_hz=20, maximum_yaw_rate_deg_s=20,
    )
    assert aligned.waypoint_arrival_indices[-1] > plan.waypoint_arrival_indices[-1]
    checkpoint = _checkpoint(3)
    contract = RuntimeCheckpointContract(contract_id="mission-test", checkpoints=[checkpoint])
    mapped = executor._schedule_checkpoint_indices(
        base, aligned.schedule, points, contract, aligned.track_start_index,
        waypoint_arrival_indices=aligned.waypoint_arrival_indices,
    )
    assert mapped == {aligned.waypoint_arrival_indices[-1]: checkpoint}
    assert mapped[aligned.waypoint_arrival_indices[-1]] is not checkpoint


# 功能：
#   到达记录乱序、越界、重叠或坐标损坏时拒绝绑定，不以最近位置兜底执行任务。
# 输入：
#   fault：本次破坏的到达记录或坐标字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["order", "outside", "duplicate", "nan", "position"])
def test_checkpoint_rejects_corrupt_arrival_mapping(fault: str) -> None:
    base, executor, points, plan = _schedule_case()
    arrivals = plan.waypoint_arrival_indices
    if fault == "order":
        arrivals = tuple(reversed(arrivals))
    elif fault == "outside":
        arrivals = (*arrivals[:-1], len(plan.schedule))
    elif fault == "duplicate":
        arrivals = (arrivals[0], arrivals[0], arrivals[-1])
    else:
        plan.schedule[arrivals[-1]] = replace(
            plan.schedule[arrivals[-1]], north_m=float("nan") if fault == "nan" else 40,
        )
    contract = RuntimeCheckpointContract(contract_id="mission-test", checkpoints=[_checkpoint(3)])
    with pytest.raises(ValueError):
        executor._schedule_checkpoint_indices(
            base, plan.schedule, points, contract, plan.track_start_index,
            waypoint_arrival_indices=arrivals,
        )
