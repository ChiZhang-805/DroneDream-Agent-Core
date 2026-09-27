"""Causal crossing evidence from tracked deployment observations, never scene scripts.

This is stage-decision context, not collision clearance or permission to move.
No detections means unknown; two distinct fresh samples are required for velocity.
"""

import math

from .decision_runtime_adapter import enu_to_frd
from .decision_state_adapter import decision_digest


# 功能：核对真实导航快照的内容身份；输入：原始快照；输出：UTC 参考毫秒。
def _clock(snapshot):
    stamp = snapshot.get("control_reference_observed_at_unix_ms")
    if (
        snapshot.get("source_of_truth")
        != "metric-range-and-localization-evidence-not-rendered-image"
        or snapshot.get("schema_version") != "dronedream.text-navigation-snapshot.v1"
        or type(stamp) is not int
        or stamp < 0
        or snapshot.get("snapshot_sha256")
        != decision_digest({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})
    ):
        raise ValueError("DECISION_DYNAMIC_SNAPSHOT_INVALID")
    return stamp


# 功能：解析带原始年龄的动态检测，不以新快照时刻刷新旧目标。
# 输入：观测行及 UTC 毫秒；输出：目标原始采样时刻、ENU 位置/速度和米制包络。
def _track(row, now):
    vector = tuple(row[key][axis] for key in ("position_m", "velocity_mps") for axis in "xyz")
    scalars = (
        *vector,
        row["observation_age_seconds"],
        row["confidence"],
        row["radius_m"],
        row["height_m"],
    )
    if (
        any(type(v) not in (int, float) or not math.isfinite(v) for v in scalars)
        or not 0 <= row["observation_age_seconds"] <= 0.25
        or not 0.35 <= row["confidence"] <= 1
        or not 0 < row["radius_m"] <= 20
        or not 0 < row["height_m"] <= 50
        or type(row["obstacle_id"]) is not str
        or not 1 <= len(row["obstacle_id"]) <= 160
    ):
        raise ValueError("DECISION_DYNAMIC_TRACK_UNUSABLE")
    return now - math.ceil(row["observation_age_seconds"] * 1000), vector[:3], vector[3:]


# 功能：在当前目标方向的有限前视带内判断已观测目标是否横穿，保留未知而非推断空场。
# 输入：相邻因果快照、world_from_body_FLU 四元数、机体包络/定位余量（米）。
# 输出：FRD 相对跟踪、横穿状态及来源说明；不输出任何飞控指令或无障碍证明。
def dynamic_context(snapshot, previous, quaternion, *, body_radius_m, body_height_m, uncertainty_m):
    now = _clock(snapshot)
    rows = snapshot.get("dynamic_obstacles")
    if (
        type(rows) is not list
        or len(rows) > 512
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for v in (body_radius_m, body_height_m, uncertainty_m)
        )
    ):
        raise ValueError("DECISION_DYNAMIC_CONTEXT_INVALID")
    result = {
        "tracks": [],
        "crossing_obstacle": None,
        "issues": [],
        "scope": "observed-tracks-only-not-free-space",
        "source_snapshots": [snapshot["snapshot_sha256"]],
    }
    if previous is None or not rows:
        result["issues"].append("DYNAMIC_TWO_FRAME_TRACK_REQUIRED")
        return result
    earlier = _clock(previous)
    if not 0 < now - earlier <= 1000:
        result["issues"].append("DYNAMIC_HISTORY_GAP")
        return result
    for key in ("source_sha256", "qualified_route_sha256"):
        if previous["known_static_map"][key] != snapshot["known_static_map"][key]:
            result["issues"].append("DYNAMIC_CONTEXT_CHANGED")
            return result
    if (
        previous["strategic_context"]["task"]["navigation_goal_id"]
        != snapshot["strategic_context"]["task"]["navigation_goal_id"]
    ):
        result["issues"].append("DYNAMIC_CONTEXT_CHANGED")
        return result
    old_rows = previous.get("dynamic_obstacles")
    if type(old_rows) is not list or len(old_rows) > 512:
        raise ValueError("DECISION_DYNAMIC_HISTORY_INVALID")
    if len({r["obstacle_id"] for r in rows}) != len(rows) or len(
        {r["obstacle_id"] for r in old_rows}
    ) != len(old_rows):
        raise ValueError("DECISION_DYNAMIC_DUPLICATE_TRACK")
    old = {r["obstacle_id"]: r for r in old_rows}
    result["source_snapshots"].insert(0, previous["snapshot_sha256"])
    origin = tuple(snapshot["current_position_m"][k] for k in "xyz")
    own_velocity = tuple(snapshot["current_velocity_mps"][k] for k in "xyz")
    goal = tuple(snapshot["goal_position_m"][k] for k in "xyz")
    dx, dy = goal[0] - origin[0], goal[1] - origin[1]
    length = math.hypot(dx, dy)
    if length < 0.05:
        result["issues"].append("DYNAMIC_HORIZONTAL_ROUTE_UNDEFINED")
        return result
    forward, right = (dx / length, dy / length), (dy / length, -dx / length)
    known, crossing = 0, False
    for row in rows:
        try:
            stamp, position, velocity = _track(row, now)
            old_stamp, old_position, _ = _track(old[row["obstacle_id"]], earlier)
            dt = (stamp - old_stamp) / 1000
            if not 0.05 <= dt <= 1:
                raise ValueError("DYNAMIC_TRACK_TIME_UNRESOLVED")
            measured = tuple((a - b) / dt for a, b in zip(position, old_position, strict=True))
            if math.dist(measured, velocity) > 0.35 + 0.25 * math.hypot(*velocity):
                raise ValueError("DYNAMIC_TRACK_VELOCITY_INCONSISTENT")
            age = (now - stamp) / 1000
            relative = tuple(
                p + v * age - o for p, v, o in zip(position, velocity, origin, strict=True)
            )
            relative_velocity = tuple(v - o for v, o in zip(velocity, own_velocity, strict=True))
            body_position = enu_to_frd(dict(zip("xyz", relative, strict=True)), quaternion)
            body_velocity = enu_to_frd(dict(zip("xyz", relative_velocity, strict=True)), quaternion)
            result["tracks"].append(
                {
                    "track_id": row["obstacle_id"],
                    "relative_position_body_frd_m": body_position,
                    "relative_velocity_body_frd_mps": body_velocity,
                    "confidence": row["confidence"],
                    "source": {
                        "source_id": "causal-two-frame-dynamic-track",
                        "evidence_sha256": decision_digest(
                            {"sources": result["source_snapshots"], "id": row["obstacle_id"]}
                        ),
                        "observed_at_ms": stamp,
                        "maximum_age_ms": 250,
                    },
                }
            )
            lateral = relative[0] * right[0] + relative[1] * right[1]
            lateral_speed = velocity[0] * right[0] + velocity[1] * right[1]
            # 以世界中的横向移动判断横穿，不能把无人机自身侧移当作行人运动。
            if abs(lateral_speed) >= 0.1:
                t = max(0.0, min(3.0, -lateral / lateral_speed))
                future = tuple(p + v * t for p, v in zip(relative, velocity, strict=True))
                along = future[0] * forward[0] + future[1] * forward[1]
                side = abs(future[0] * right[0] + future[1] * right[1])
                margin = body_radius_m + row["radius_m"] + 3 * uncertainty_m + 0.15
                crossing |= (
                    -margin <= along <= min(length, 2.0) + margin
                    and side <= margin
                    and abs(future[2])
                    <= (body_height_m + row["height_m"]) / 2 + 3 * uncertainty_m + 0.15
                )
            known += 1
        except (KeyError, TypeError, ValueError) as error:
            result["issues"].append(type(error).__name__ + ":" + str(error)[:100])
    # 有一个明确横穿即可提示等待；未知目标存在时不能给出“没有横穿”的确定结论。
    result["crossing_obstacle"] = True if crossing else (False if known == len(rows) else None)
    if len(result["tracks"]) > 12:
        result["issues"].append("DYNAMIC_TRACK_CAPACITY_EXCEEDED")
        result["tracks"] = []
        result["crossing_obstacle"] = None
    return result
