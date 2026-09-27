"""Versioned route-guided collection teacher; never replace pose with a point on a route."""

import json
import math

from .contracts import GraphRoute
from .decision_collection_teacher import CollectionTeacher
from .decision_runtime_adapter import enu_to_frd
from .decision_state_adapter import DecisionStateV2, decision_digest


class RouteCollectionTeacher(CollectionTeacher):
    # 功能：固定已提供的路线；不是把路线投影当实测定位，旧v1教师与原数据验证保持不变。
    # 输入：GraphRoute字典及父类静态几何/机体参数；输出：只提案的新版本教师。
    def __init__(self, *, qualified_route, **parameters):
        super().__init__(**parameters)
        route = GraphRoute.model_validate(qualified_route)
        self.route_sha256 = decision_digest(route.model_dump(mode="json"))
        self.points = [tuple(getattr(p, k) for k in "xyz") for p in route.positions_m]
        if not 2 <= len(self.points) <= 256 or any(
            math.dist(a, b) < 1e-6 for a, b in zip(self.points, self.points[1:], strict=False)
        ):
            raise ValueError("DECISION_TEACHER_ROUTE_INVALID")
        self.route_blocked = None
        self.route_clock = None

    # 功能：计算到路线的几何投影和局部参考点；当前坐标仍取实测值，沿折线前视而非直指终点。
    # 输入：实测ENU位置米、当前任务目标ENU米；输出：参考点、偏离距离、剩余路线长度米。
    def route_target(self, position, goal, *, waypoint_index=None):
        matches = [i for i, point in enumerate(self.points) if math.dist(point, goal) <= 0.01]
        if waypoint_index is not None:
            if (
                type(waypoint_index) is not int
                or waypoint_index <= 0
                or waypoint_index not in matches
            ):
                raise ValueError("DECISION_TEACHER_WAYPOINT_BINDING_INVALID")
            # 往返同坐标不是同一任务段；按执行器目标索引绑定当前边，不凭最近坐标猜方向。
            points = self.points[waypoint_index - 1 : waypoint_index + 1]
        else:
            if len(matches) != 1 or matches[0] == 0:
                raise ValueError("DECISION_TEACHER_GOAL_NOT_ON_BOUND_ROUTE")
            points = self.points[: matches[0] + 1]
        candidates = []
        arc = 0.0
        total = sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))
        for index, (a, b) in enumerate(zip(points, points[1:], strict=False)):
            length = math.dist(a, b)
            unit = tuple((v - u) / length for u, v in zip(a, b, strict=True))
            along = max(
                0.0,
                min(length, sum((v - u) * d for v, u, d in zip(position, a, unit, strict=True))),
            )
            projection = tuple(u + d * along for u, d in zip(a, unit, strict=True))
            # 到达顶点后优先下一段，避免零向量停在转角；没有把投影替换为实测位置。
            candidates.append(
                (math.dist(position, projection), -index, a, b, unit, along, arc, length)
            )
            arc += length
        distance, negative_index, a, b, unit, along, arc, length = min(
            candidates, key=lambda c: (c[0], c[1])
        )
        ahead = along + 0.75
        index = -negative_index
        while ahead > length and index + 2 < len(points):
            ahead -= length
            index += 1
            a, b = points[index : index + 2]
            length = math.dist(a, b)
            unit = tuple((v - u) / length for u, v in zip(a, b, strict=True))
        target = tuple(u + d * min(length, ahead) for u, d in zip(a, unit, strict=True))
        return target, distance, total - arc - along

    # 功能：在原始传感器状态上只替换路线引导相关字段；静态扫掠与动态感知仍独立存在。
    # 输入：未改写的导航快照及历史；输出：可从同一原始前缀重建的v2状态和路线检查依据。
    def observe(self, snapshot, **identity):
        state, check, issues = super().observe(snapshot, **identity)
        if state.route_sha256 != self.route_sha256:
            raise ValueError("DECISION_TEACHER_ROUTE_CHANGED")
        position = tuple(snapshot["current_position_m"][k] for k in "xyz")
        goal = tuple(snapshot["goal_position_m"][k] for k in "xyz")
        prefix = "source-waypoint-"
        suffix = state.goal_id.removeprefix(prefix)
        index = (
            int(suffix)
            if state.goal_id.startswith(prefix) and suffix.isascii() and suffix.isdigit()
            else None
        )
        if state.goal_id.startswith(prefix) and index is None:
            raise ValueError("DECISION_TEACHER_WAYPOINT_ID_INVALID")
        target, deviation, remaining = self.route_target(position, goal, waypoint_index=index)
        frame = state.frame.model_dump(mode="json")
        now, margin = frame["observed_at_ms"], state.frame.position_uncertainty_m
        if self.route_clock is not None and not 0 < now - self.route_clock <= 250:
            self.route_blocked = None
        self.route_clock = now
        check.update(
            schema_version="dronedream.decision-route-prefix-check.v2",
            route_sha256=self.route_sha256,
            prefix_world_enu_m=list(target),
            distance_from_route_m=deviation,
            remaining_route_m=remaining,
            minimum_clearance_m=None,
        )
        pose = state.frame.pose_source
        if pose and pose.fresh(now) and margin is not None:
            clearance = self.geometry.clearance([position, target])
            check["minimum_clearance_m"] = clearance
            verified = clearance >= self.clearance_m + 3 * margin
            frame["local_route_verified"] = verified
            braking = self.nominal_speed_mps**2 / (2 * state.braking_acceleration_mps2)
            frame["caution_required"] = clearance < self.clearance_m + 3 * margin + braking + 0.15
            if verified:
                self.route_blocked = None
            elif self.route_blocked is None or self.route_blocked[0] != state.goal_id:
                self.route_blocked = (state.goal_id, now)
            frame["persistent_blockage"] = (
                not verified
                and self.route_blocked is not None
                and now - self.route_blocked[1] >= 2000
            )
            frame["route_source"] = {
                "source_id": "known-map-qualified-route-prefix",
                "evidence_sha256": decision_digest(check),
                "observed_at_ms": pose.observed_at_ms,
                "maximum_age_ms": pose.maximum_age_ms,
            }
            flight = next(
                e
                for e in snapshot["realtime_feature_snapshot"]["encodings"]
                if e["encoder_role"] == "flight-state-encoder"
            )
            if state.frame.goal_relative_body_frd_m is not None:
                frame["goal_relative_body_frd_m"] = enu_to_frd(
                    dict(
                        zip(
                            "xyz",
                            (b - a for a, b in zip(position, target, strict=True)),
                            strict=True,
                        )
                    ),
                    flight["features"][:4],
                )
        else:
            self.route_blocked = None
        return (
            DecisionStateV2.model_validate_json(
                json.dumps({**state.model_dump(mode="json"), "frame": frame})
            ),
            check,
            issues,
        )
