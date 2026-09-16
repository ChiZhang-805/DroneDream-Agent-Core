from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_commands import _load_executor
from test_runtime_follow import _modules
from test_runtime_replan_boundaries import _mode_inputs

from dronedream_agent_core.contracts import Px4TrackPoint, RuntimeTrackProgress, WorldTrackPoint
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_progress import bound_resume_point_index, next_track_point_index
from dronedream_agent_core.runtime_replan import RuntimeReplanError


# 功能：
#   构造有直角和重访起点的轨迹，以区分真实下一航点与空间上最近的航点。
# 输入：
#   tmp_path：独立地图目录。
#   next_index：执行器声明仍须到达的下一航点。
#   position：当前局部东、北位置。
# 输出：
#   case：限速构建器和绑定输入。
def _speed_case(tmp_path: Path, next_index: int = 1, position=(4.0, 0.0)):
    builder, inputs = _mode_inputs(tmp_path, "speed")
    positions = [(0, 0), (5, 0), (5, 5), (0, 0)]
    track = inputs["prior_track"].model_copy(update={
        "points": [Px4TrackPoint(
            x=north, y=east, z=0.8, phase="land" if index == 3 else "transit", speed_limit_mps=1,
        ) for index, (east, north) in enumerate(positions)],
        "source_world_points": [
            WorldTrackPoint(east_m=east, north_m=north, up_m=1) for east, north in positions
        ],
    })
    track_hash = sha256_json(track)
    progress = RuntimeTrackProgress(track_sha256=track_hash, next_track_point_index=next_index)
    ack = inputs["acknowledgement"]
    ack = ack.model_copy(update={
        "track_progress": progress,
        "observed_position_ned_m": ack.observed_position_ned_m.model_copy(update={
            "x": position[1], "y": position[0],
        }),
    })
    inputs.update(prior_track=track, prior_track_sha256=track_hash, acknowledgement=ack)
    inputs["decision"] = inputs["decision"].model_copy(update={"hold_ack_sha256": sha256_json(ack)})
    case = builder, inputs
    return case


# 功能：
#   无人机接近拐角但未到拐角时，限速不能从最近航点后切片并抄近路跳过拐角。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_speed_replan_keeps_unreached_corner(tmp_path: Path) -> None:
    builder, inputs = _speed_case(tmp_path)
    replacement = builder(**inputs)
    assert replacement.route.positions_m[1].model_dump() == {"x": 5.0, "y": 0.0, "z": 1.0}


# 功能：
#   回程再次靠近起点时，按当前调度进度完成返程，不能重复整条已飞过的去程。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_speed_replan_does_not_restart_loop_near_origin(tmp_path: Path) -> None:
    builder, inputs = _speed_case(tmp_path, next_index=3, position=(0.1, 0.1))
    replacement = builder(**inputs)
    assert len(replacement.route.positions_m) == 2
    assert replacement.route.positions_m[-1].model_dump() == {"x": 0.0, "y": 0.0, "z": 1.0}


# 功能：
#   进度回执缺失或属于另一条轨迹时拒绝截取剩余路段，不以位置接近作为替代证据。
# 输入：
#   tmp_path：独立测试目录。
#   fault：进度回执缺失或摘要错配。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["missing", "stale"])
def test_speed_replan_requires_bound_executor_progress(tmp_path: Path, fault: str) -> None:
    builder, inputs = _speed_case(tmp_path)
    ack = inputs["acknowledgement"]
    progress = None if fault == "missing" else ack.track_progress.model_copy(
        update={"track_sha256": "0" * 64}
    )
    ack = ack.model_copy(update={"track_progress": progress})
    inputs["acknowledgement"] = ack
    inputs["decision"] = inputs["decision"].model_copy(update={"hold_ack_sha256": sha256_json(ack)})
    with pytest.raises(RuntimeReplanError, match="TRACK_PROGRESS"):
        builder(**inputs)


# 功能：
#   到达采样仍保留当前航点，下一采样才前移；最终保持阶段不产生越界航点。
# 输入：
#   sample：当前采样下标。
#   expected：预期下一航点。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("sample,expected", [(0, 1), (10, 1), (11, 2), (20, 2), (21, 3), (99, 3)])
def test_schedule_progress_uses_order_not_position(sample: int, expected: int) -> None:
    assert next_track_point_index((10, 20, 30), sample, 4) == expected


# 功能：
#   拒绝数量错配、乱序、布尔采样或布尔点数，不能把损坏调度悄悄夹到某个有效下标。
# 输入：
#   arrivals：测试到达下标序列。
#   sample：当前采样下标。
#   count：原轨迹点数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("arrivals,sample,count", [
    ((10,), 0, 3), ((20, 10), 0, 3), ((10, 20), True, 3), ((10,), 0, True),
    ((True, 20), 0, 3), ((10, 20), -1, 3),
    ((10, 10), 0, 3),
])
def test_schedule_progress_rejects_invalid_mapping(arrivals, sample, count) -> None:
    with pytest.raises(ValueError, match="SCHEDULE_INVALID"):
        next_track_point_index(arrivals, sample, count)


# 功能：
#   校验边界函数拒绝超出实际轨迹长度的进度，即使进度结构允许更大的通用上界。
# 输入：
#   tmp_path：独立测试目录。
# 输出：
#   None：不返回业务数据。
def test_bound_progress_rejects_out_of_range_point(tmp_path: Path) -> None:
    _, inputs = _speed_case(tmp_path, next_index=9)
    with pytest.raises(ValueError, match="BINDING_INVALID"):
        bound_resume_point_index(inputs["acknowledgement"], inputs["prior_track"])


# 功能：
#   运行真实悬停回执协程，以确定性遥测和时钟替身检验稳定阶段及进度的落盘绑定。
# 输入：
#   tmp_path：仅本测试使用的回执目录。
#   monkeypatch：局部替换遥测与时间，不启动飞控或仿真。
# 输出：
#   None：不返回业务数据。
def test_executor_hold_publishes_waypoint_phase_and_progress(tmp_path, monkeypatch) -> None:
    executor = _load_executor()
    _, inputs = _speed_case(tmp_path)
    message, progress = inputs["message"], inputs["acknowledgement"].track_progress
    interruption = executor.RuntimeInterruptDetected(
        message, tmp_path / "claimed.json", message.submitted_at
    )
    ticks = itertools.count(step=0.6)
    monkeypatch.setattr(executor, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    # 功能：
    #   提供始终静止的遥测样本，只检验回执链路而非模拟动力学性能。
    # 输入：
    #   _：协程传入的刷新上下文。
    # 输出：
    #   sample：稳定位置和零速度观测。
    async def sample_tick(**_):
        sample = SimpleNamespace(
            north_m=0.0, east_m=4.0, down_m=-0.8,
            north_m_s=0.0, east_m_s=0.0, down_m_s=0.0,
        )
        return sample

    monkeypatch.setattr(executor, "_runtime_hold_tick", sample_tick)
    acknowledgement, _ = asyncio.run(executor._stabilize_runtime_hold(
        base=SimpleNamespace(), client=SimpleNamespace(),
        frozen_setpoint=SimpleNamespace(north_m=0, east_m=4, down_m=-0.8, yaw_deg=0),
        interruption=interruption, control_dir=tmp_path, phase="WAYPOINT_SETTLE",
        schedule_index=10, abort_file=tmp_path / "abort.json", rate_hz=50,
        timeout_seconds=10, track_progress=progress,
    ))
    assert acknowledgement.interrupted_phase == "WAYPOINT_SETTLE"
    assert acknowledgement.track_progress == progress
    assert acknowledgement.track_progress is not progress
    assert (tmp_path / "acks" / f"{message.message_id}.json").is_file()


# 功能：
#   验证中断入口在第一次悬停刷新前复制执行器进度，后续上下文修改不能污染该次回执。
# 输入：
#   tmp_path：测试目录。
#   monkeypatch：以捕获协程代替后续飞控处理。
# 输出：
#   None：不返回业务数据。
def test_interruption_captures_executor_progress_before_hold(tmp_path, monkeypatch) -> None:
    executor = _load_executor()
    _, inputs = _speed_case(tmp_path)
    progress = inputs["acknowledgement"].track_progress
    context = SimpleNamespace(_runtime_track_progress=progress)

    # 功能：
    #   截获悬停入口的已复制进度，原上下文改变后仍应维持本消息的原始绑定。
    # 输入：
    #   kwargs：中断入口传入的悬停参数。
    # 输出：
    #   None：不返回业务数据。
    async def capture(**kwargs):
        captured = kwargs["track_progress"]
        assert captured is not progress
        progress.next_track_point_index = 3
        assert captured.next_track_point_index == 1
        raise RuntimeError("CAPTURE_COMPLETE")

    monkeypatch.setattr(executor, "_stabilize_runtime_hold", capture)
    with pytest.raises(RuntimeError, match="CAPTURE_COMPLETE"):
        asyncio.run(executor._handle_runtime_interruption(
            base=SimpleNamespace(), client=SimpleNamespace(), frozen_setpoint=None,
            interruption=None, control_dir=tmp_path, phase="TRACK", schedule_index=1,
            abort_file=tmp_path / "abort.json", rate_hz=20, hold_timeout_seconds=2,
            decision_timeout_seconds=2, replan_hold_seconds=2,
            active_track_sha256=inputs["prior_track_sha256"], params=None,
            telemetry_args=context,
        ))


# 功能：
#   相邻同坐标航点仍各有独立到达采样，不能因运动距离为零丢掉进度或同地点的动作。
# 输入：
#   hold_seconds：每个航点的静止等待时间，覆盖零等待及有等待两种情况。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("hold_seconds", [0.0, 0.4])
def test_stationary_waypoints_keep_distinct_schedule_arrivals(hold_seconds: float) -> None:
    base, _ = _modules()
    points = [base.TrackPoint(x, 0, 1) for x in (0, 0, 1, 1, 0)]
    plan = base.build_setpoint_schedule_plan(
        points, base.ControllerParams(1, 2), 20,
        stop_at_waypoints=True, waypoint_hold_seconds=hold_seconds,
    )
    arrivals = plan.waypoint_arrival_indices
    assert len(arrivals) == len(points) - 1
    assert all(a < b for a, b in zip(arrivals, arrivals[1:], strict=False))
    for point_index, arrival_index in enumerate(arrivals, 1):
        assert next_track_point_index(arrivals, arrival_index, len(points)) == point_index
        assert plan.schedule[arrival_index].north_m == points[point_index].x


# 功能：
#   真实重规划生成的当前位置与接入锚点重合时，替换调度仍保持航点和到达记录一一对应。
# 输入：
#   tmp_path：实际重规划工具使用的独立地图目录。
# 输出：
#   None：不返回业务数据。
def test_replacement_schedule_preserves_stationary_anchor(tmp_path: Path) -> None:
    base, executor = _modules()
    builder, inputs = _mode_inputs(tmp_path, "destination")
    replacement = builder(**inputs)
    first, second = replacement.track.points[:2]
    assert (first.x, first.y, first.z) == (second.x, second.y, second.z)
    schedule, arrivals = executor._compile_replacement_schedule(
        base=base, replacement=replacement, params=base.ControllerParams(1, 2), rate_hz=20,
    )
    assert len(arrivals) == len(replacement.track.points) - 1
    assert all(a < b for a, b in zip(arrivals, arrivals[1:], strict=False))
    for point_index, arrival_index in enumerate(arrivals, 1):
        assert next_track_point_index(arrivals, arrival_index, len(arrivals) + 1) == point_index
        assert schedule[arrival_index].north_m == replacement.track.points[point_index].x
