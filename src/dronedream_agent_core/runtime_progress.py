"""Bind route continuation to executor progress rather than nearest coordinates."""

from __future__ import annotations

from bisect import bisect_left

from .contracts import Px4Track, RuntimeHoldAcknowledgement
from .hashing import sha256_json


# 功能：
#   将调度采样下标映射到下一原始航点，保留同坐标多次经过的不同顺序，不按位置猜测。
# 输入：
#   waypoint_arrival_indices：每个后续航点对应的到达采样下标，按执行顺序排列。
#   schedule_index：当前仍待完成的调度采样下标。
#   point_count：原轨迹航点总数。
# 输出：
#   next_index：当前仍待到达的原轨迹航点下标。
def next_track_point_index(
    waypoint_arrival_indices: tuple[int, ...], schedule_index: int, point_count: int,
) -> int:
    if (
        type(point_count) is not int or not 2 <= point_count <= 10_000
        or type(schedule_index) is not int or schedule_index < 0
        or not isinstance(waypoint_arrival_indices, tuple)
        or len(waypoint_arrival_indices) != point_count - 1
        or any(type(index) is not int or index < 0 for index in waypoint_arrival_indices)
        or any(a >= b for a, b in zip(
            waypoint_arrival_indices, waypoint_arrival_indices[1:], strict=False
        ))
    ):
        raise ValueError("RUNTIME_TRACK_PROGRESS_SCHEDULE_INVALID")
    # 到达采样本身尚须稳定和执行动作，因此相等时仍保留该航点，而不是提前跳到下一段。
    next_index = min(bisect_left(waypoint_arrival_indices, schedule_index) + 1, point_count - 1)
    return next_index


# 功能：
#   核对悬停回执中的执行器进度与活动轨迹摘要一致，拒绝缺失、越界或旧轨迹进度。
# 输入：
#   acknowledgement：已经完成消息和悬停身份检查的回执。
#   track：当前活动轨迹。
# 输出：
#   next_index：经绑定验证、仍须经过的下一航点下标。
def bound_resume_point_index(acknowledgement: RuntimeHoldAcknowledgement, track: Px4Track) -> int:
    progress = acknowledgement.track_progress
    if progress is None:
        raise ValueError("RUNTIME_TRACK_PROGRESS_REQUIRED")
    next_index = progress.next_track_point_index
    if (
        progress.track_sha256 != sha256_json(track)
        or type(next_index) is not int
        or not 1 <= next_index < len(track.points)
        or len(track.points) != len(track.source_world_points)
    ):
        raise ValueError("RUNTIME_TRACK_PROGRESS_BINDING_INVALID")
    return next_index
