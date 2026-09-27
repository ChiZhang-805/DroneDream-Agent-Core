"""Read-only adapter for existing metric navigation snapshots; no truth substitution."""

import json
import math

from .decision_state_adapter import DecisionStateV2, decision_digest


# 功能：用单位四元数将 ENU 相对向量转换到机体 FRD，明确对接现有 FLU 机体契约。
# 输入：世界向量、world_from_body_FLU 四元数；输出：前/右/下米向量，拒绝非单位姿态。
def enu_to_frd(vector: dict, quaternion: list[float]) -> dict:
    if (len(quaternion) != 4 or any(type(v) not in (int, float)
                                  or not math.isfinite(v) for v in quaternion)
            or abs(math.hypot(*quaternion) - 1) > .001):
        raise ValueError("DECISION_ORIENTATION_INVALID")
    w, x, y, z = quaternion
    vx, vy, vz = (vector[k] for k in "xyz")
    # R^T：先回到已有机体 FLU，然后以 diag(1,-1,-1) 映射 FRD。
    forward = (1 - 2*y*y - 2*z*z)*vx + (2*x*y + 2*z*w)*vy + (2*x*z - 2*y*w)*vz
    left = (2*x*y - 2*z*w)*vx + (1 - 2*x*x - 2*z*z)*vy + (2*y*z + 2*x*w)*vz
    up = (2*x*z + 2*y*w)*vx + (2*y*z - 2*x*w)*vy + (1 - 2*x*x - 2*y*y)*vz
    return {"x": forward, "y": -left, "z": -up}


# 功能：转换已有实测导航快照；所有未有可比证据的字段保留 unknown。
# 输入：已存档快照及独立任务身份、标定摘要、明确制动参数；输出：v2 状态及覆盖问题。
def state_from_navigation_snapshot(snapshot: dict, *, mission_id: str, sequence: int,
                                   calibration_sha256: str, braking_acceleration_mps2: float,
                                   scene: str, task: str = "navigate",
                                   history: tuple = ()) -> tuple[DecisionStateV2, list[str]]:
    if snapshot.get("schema_version") != "dronedream.text-navigation-snapshot.v1":
        raise ValueError("DECISION_NAVIGATION_SCHEMA_INVALID")
    if snapshot.get("source_of_truth") != (
            "metric-range-and-localization-evidence-not-rendered-image"):
        raise ValueError("DECISION_DEPLOYMENT_OBSERVATION_REQUIRED")
    if snapshot.get("snapshot_sha256") != decision_digest(
            {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}):
        raise ValueError("DECISION_NAVIGATION_HASH_MISMATCH")
    now = snapshot["control_reference_observed_at_unix_ms"]
    context = snapshot["strategic_context"]
    encodings = (snapshot.get("realtime_feature_snapshot") or {}).get("encodings", [])
    by_role = {e["encoder_role"]: e for e in encodings}
    if len(by_role) != len(encodings):
        raise ValueError("DECISION_DUPLICATE_ENCODER")
    # 转换器只解释这一版 FLU 四元数定义；未知编码不能猜测方向。
    flight = by_role.get("flight-state-encoder")
    if flight and flight.get("feature_contract_sha256") != (
            "48f9536199319c760afc8e208826a2d349eea2d5b49f2de1437d481b4f831ad9"):
        raise ValueError("DECISION_FLIGHT_ENCODER_CONTRACT_UNSUPPORTED")

    # 功能：保留采集原时间和原时效；输入：编码角色；输出：源戳或未知。
    def source(role):
        value = by_role.get(role)
        if value is None:
            return None
        return {"source_id": role, "evidence_sha256": value["source_sha256"],
                "observed_at_ms": value["observed_at_unix_ms"],
                "maximum_age_ms": value["maximum_age_milliseconds"]}

    pose = source("flight-state-encoder")
    geometry = source("metric-geometry-encoder")
    relative, velocity_body = None, None
    issues = []
    # 现有扇区朝向是目标方向而非机头方向，不能按字段同名直接映射。
    reference = snapshot.get("egocentric_sector_reference")
    if reference != "body-FRD":
        issues.append("GOAL_ALIGNED_SECTORS_NOT_BODY_CLEARANCES")
    flight = by_role.get("flight-state-encoder")
    if flight and all(v == 1 for v in flight.get("valid_mask", [])[:4]) \
            and len(flight.get("valid_mask", [])) >= 4:
        relative = enu_to_frd({k: snapshot["goal_position_m"][k]
                              - snapshot["current_position_m"][k] for k in "xyz"},
                             flight["features"][:4])
        velocity_body = enu_to_frd(snapshot["current_velocity_mps"], flight["features"][:4])
    else:
        issues.append("BODY_ORIENTATION_UNKNOWN")
    static = snapshot["known_static_map"]
    route_hash = static["qualified_route_sha256"]
    health = snapshot["perception_health"]
    covariance = health.get("localization_covariance_m2")
    uncertainty = math.sqrt(covariance) if covariance is not None and covariance >= 0 else None
    frame = {"observed_at_ms": now,
             "position_world_enu_m": snapshot["current_position_m"] if pose else None,
             "velocity_world_enu_mps": snapshot["current_velocity_mps"] if pose else None,
             "velocity_body_frd_mps": velocity_body,
             "goal_relative_body_frd_m": relative,
             "position_uncertainty_m": uncertainty if pose else None,
             "pose_source": pose, "geometry_source": geometry, "route_source": None,
             "local_route_verified": None, "crossing_obstacle": None,
             "persistent_blockage": None, "caution_required": None,
             "can_hold_position": None, "can_brake": None, "payload_attached": None,
             **{f"clearance_{direction}_m": None for direction in
                ("front", "back", "left", "right", "up", "down")}}
    # 不把“有一份已规划路线”转换成当前局部路径仍然可行。
    issues.extend(("LOCAL_ROUTE_STATUS_NOT_WITNESSED", "BEHAVIOR_OUTCOME_LABELS_REQUIRED"))
    state = {"mission_id": mission_id,
             "goal_id": context["task"]["navigation_goal_id"],
             "route_sha256": route_hash, "map_sha256": static["source_sha256"],
             "vehicle_sha256": decision_digest(context["vehicle"]),
             "calibration_sha256": calibration_sha256, "clock_domain": "unix-ms",
             "sequence": sequence, "task": task, "scene": scene,
             "braking_acceleration_mps2": braking_acceleration_mps2,
             "body_radius_m": context["vehicle"]["body_radius_m"],
             "preferred_height_above_floor_m": None, "frame": frame,
             "history": [f.model_dump(mode="json") for f in history]}
    return DecisionStateV2.model_validate_json(json.dumps(state, allow_nan=False)), issues
