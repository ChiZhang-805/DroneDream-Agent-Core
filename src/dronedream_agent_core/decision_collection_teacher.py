"""Simulation collection teacher using deployment observations, never simulator truth.

This selects proposals only. PX4's existing independent safety outlet still decides
what is executable. A selected behavior is not a positive training label.
"""

import json
import math

from .decision_dynamic_context import dynamic_context
from .decision_runtime_adapter import state_from_navigation_snapshot
from .decision_state_adapter import DecisionStateV2, decision_digest
from .training.flight_environment import PilotAction
from .training.swept_geometry import SweptMapGeometry


class CollectionTeacher:
    # 功能：固定可供部署读取的静态地图和机型；独立结果真值不传入本教师。
    # 输入：地图摘要、静态基元、机体高度、名义速度和操作净空（米制）；输出：只提案教师。
    def __init__(
        self,
        *,
        map_sha256,
        primitives,
        body_radius_m,
        body_height_m,
        nominal_speed_mps,
        clearance_m,
    ):
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (body_radius_m, body_height_m, nominal_speed_mps, clearance_m)
        ):
            raise ValueError("DECISION_TEACHER_PHYSICAL_PARAMETERS_INVALID")
        self.map_sha256 = map_sha256
        self.geometry_sha256 = decision_digest(primitives)
        self.geometry = SweptMapGeometry(
            primitives, radius_m=body_radius_m, half_height_m=body_height_m / 2
        )
        self.nominal_speed_mps = min(nominal_speed_mps, 0.4)
        self.clearance_m = clearance_m
        self.previous_blocked = None
        self.previous_observed_ms = None
        self.previous_snapshot = None
        self.body_radius_m = body_radius_m
        self.body_height_m = body_height_m

    # 功能：从真实导航输入复核局部地图前缀，将可证明与未知字段分开保存。
    # 输入：原快照、固定采集身份参数及过去帧；输出：因果状态、原始地图核验依据、覆盖缺口。
    # 时钟UNIX毫秒；世界ENU米，机体FRD米。静态前缀通过不代表动态空间已全部观测。
    def observe(self, snapshot, **identity):
        state, issues = state_from_navigation_snapshot(snapshot, **identity)
        if state.map_sha256 != self.map_sha256:
            raise ValueError("DECISION_TEACHER_MAP_CHANGED")
        frame = state.frame.model_dump(mode="json")
        # 两个孤立的有效观测不能证明中间一直堵塞；过长缺口、回退或重复时钟重置累计。
        # 250毫秒是采集连续性上限，不延长任何传感器自身的有效期。
        if self.previous_observed_ms is not None and not (
            0 < frame["observed_at_ms"] - self.previous_observed_ms <= 250
        ):
            self.previous_blocked = None
        self.previous_observed_ms = frame["observed_at_ms"]
        p = tuple(snapshot["current_position_m"][k] for k in "xyz")
        goal = tuple(snapshot["goal_position_m"][k] for k in "xyz")
        delta = tuple(g - v for g, v in zip(goal, p, strict=True))
        length = math.hypot(*delta)
        prefix = tuple(
            v + d * min(1.0, 0.75 / max(length, 1e-9)) for v, d in zip(p, delta, strict=True)
        )
        margin = state.frame.position_uncertainty_m
        fresh_pose = state.frame.pose_source and state.frame.pose_source.fresh(
            frame["observed_at_ms"]
        )
        result = {
            "schema_version": "dronedream.decision-map-prefix-check.v1",
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "map_sha256": self.map_sha256,
            "geometry_sha256": self.geometry_sha256,
            "observed_at_ms": frame["observed_at_ms"],
            "position_world_enu_m": list(p),
            "goal_world_enu_m": list(goal),
            "prefix_world_enu_m": list(prefix),
            "scope": "static-map-prefix-only; dynamic authority remains independent",
            "localization_margin_m": None if margin is None else 3 * margin,
            "minimum_clearance_m": None,
            "clearance_at_position_m": None,
        }
        if fresh_pose and margin is not None:
            clearance = self.geometry.clearance([p, prefix])
            at_position = self.geometry.clearance([p, p])
            result.update(minimum_clearance_m=clearance, clearance_at_position_m=at_position)
            verified = clearance >= self.clearance_m + 3 * margin
            frame["local_route_verified"] = verified
            frame["can_hold_position"] = at_position >= self.clearance_m + 3 * margin
            braking = self.nominal_speed_mps**2 / (2 * state.braking_acceleration_mps2)
            frame["caution_required"] = clearance < self.clearance_m + 3 * margin + braking + 0.15
            # 仅有静态地图不能确认实际制动能力或不存在动态目标，保留未知。
            if verified:
                self.previous_blocked = None
            elif self.previous_blocked is None or self.previous_blocked[0] != state.goal_id:
                self.previous_blocked = (state.goal_id, frame["observed_at_ms"])
            frame["persistent_blockage"] = (
                not verified
                and self.previous_blocked is not None
                and frame["observed_at_ms"] - self.previous_blocked[1] >= 2000
            )
            frame["route_source"] = {
                "source_id": "known-map-local-prefix",
                "evidence_sha256": decision_digest(result),
                "observed_at_ms": state.frame.pose_source.observed_at_ms,
                "maximum_age_ms": state.frame.pose_source.maximum_age_ms,
            }
            issues.remove("LOCAL_ROUTE_STATUS_NOT_WITNESSED")
        else:
            # 缺定位期间不能累计“持续堵塞”的已观测时长。
            self.previous_blocked = None
        # 动态判定只使用截至当前时刻的原始跟踪，不能读取场景脚本或独立真值。
        flight = next(
            (
                row
                for row in (snapshot.get("realtime_feature_snapshot") or {}).get("encodings", [])
                if row.get("encoder_role") == "flight-state-encoder"
            ),
            None,
        )
        if (
            fresh_pose
            and margin is not None
            and state.frame.goal_relative_body_frd_m is not None
            and flight is not None
            and "dynamic_obstacles" in snapshot
        ):
            dynamic = dynamic_context(
                snapshot,
                self.previous_snapshot,
                flight["features"][:4],
                body_radius_m=self.body_radius_m,
                body_height_m=self.body_height_m,
                uncertainty_m=margin,
            )
            frame["tracks"] = dynamic["tracks"]
            frame["crossing_obstacle"] = dynamic["crossing_obstacle"]
            issues.extend(dynamic["issues"])
        # 快照由调用方冻结；保留前一帧而不是未来结果或物理真值。
        self.previous_snapshot = snapshot
        document = state.model_dump(mode="json")
        document["frame"] = frame
        return DecisionStateV2.model_validate_json(json.dumps(document)), result, issues

    # 功能：选择行为建议并转换为归一化四轴提案；安全执行器仍逐条复核，不获得直通权限。
    # 输入：当前因果状态、原快照、控制速度尺度；输出：行为ID、提案及选择依据。
    # 不根据脚本预定标签强迫等待/绕行；不能证明原因时明确补观测而非编造原因。
    def propose(self, state, snapshot, limits):
        frame = state.frame
        missing = [
            name
            for name in ("pose_source", "geometry_source", "route_source")
            if getattr(frame, name) is None or not getattr(frame, name).fresh(frame.observed_at_ms)
        ]
        if missing or frame.goal_relative_body_frd_m is None:
            return (
                "request_observation",
                PilotAction(mode="request-new-scan", axes=[0.0] * 4),
                {"reason": "missing-current-observation", "sources": missing},
            )
        if frame.crossing_obstacle is True and frame.can_hold_position is True:
            return "wait", PilotAction(mode="hold", axes=[0.0] * 4), {"reason": "observed-crossing"}
        if frame.local_route_verified is not True:
            # 此分支只记录重规划需求，绝不虚构规划器已产出替代路径或任务已经成功。
            action = "replan" if frame.persistent_blockage is True else "request_observation"
            return (
                action,
                PilotAction(mode="hold", axes=[0.0] * 4),
                {"reason": "static-prefix-blocked", "alternative_path_available": False},
            )
        slow = frame.caution_required is True
        speed = min(self.nominal_speed_mps / 2, 0.15) if slow else self.nominal_speed_mps
        # 目标位置仅用于引导，不能当作当前定位。3D速度统一限幅后再做FRD/上轴转换。
        relative = frame.goal_relative_body_frd_m
        vector = [relative.x, relative.y, relative.z]
        distance = math.hypot(*vector)
        desired = [v * min(0.7, speed / max(distance, 1e-9)) for v in vector]
        axes = [
            desired[0] / limits.horizontal_speed_mps,
            desired[1] / limits.horizontal_speed_mps,
            -desired[2] / limits.vertical_speed_mps,
            0.0,
        ]
        # 机型/负载上限可能低于任务名义速度。统一缩放保持三维方向，
        # 不逐轴截断而改变轨迹，也不要求控制器接受越界归一化值。
        scale = max(1.0, *(abs(v) for v in axes))
        axes = [v / scale for v in axes]
        return (
            "slow_down" if slow else "follow_route",
            PilotAction(mode="pilot-control", axes=axes),
            {
                "reason": "narrow-prefix" if slow else "verified-static-prefix",
                "requested_speed_limit_mps": speed,
                "nominal_speed_limit_mps": self.nominal_speed_mps,
            },
        )
