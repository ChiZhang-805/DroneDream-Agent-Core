"""Replay bound sensor diagnostics, not teacher actions or fabricated camera frames."""

import math
from copy import deepcopy

from ..contracts import RuntimeLocalSafetyObservation
from ..control_feature_contract import FLIGHT_STATE_MAXIMUM_GAP_MS
from ..hashing import sha256_json
from ..local_advisor_training import LocalAdvisorTrainingSample
from ..local_policy_packages import LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH
from ..local_policy_port import compile_local_policy_features
from ..runtime_sensor_contracts import RuntimeMultimodalSensorSnapshot
from ..temporal_evidence import ObservationHistory

SENSOR_ADVISOR_ROLES = {
    "perception-health-critic", "state-anomaly-detector", "cross-modal-consistency-critic"}


# 功能：
#   1. 将同一时刻的真实安全观测与模态诊断编码，故障期间不继承旧的健康结论。
#   2. 地图只取同目标的最近历史上下文，时序仅由实际飞行状态来源推进。
#   3. 缺少模态诊断时不制造跨模态或综合异常标签；所有输出仅用于感知辅助专家。
# 输入：
#   snapshots：已核验的导航时间线；records：绑定摘要的原始安全循环日志。
#   roles：请求的感知专家；source_identity：物理来源摘要。
# 输出：
#   samples：保留来源和记录身份的去重辅助样本映射。
def replay_teacher_sensor_diagnostics(snapshots, records, *, roles, source_identity):
    roles = set(roles) & SENSOR_ADVISOR_ROLES
    previous_snapshot = -1
    for stamp, snapshot in snapshots:
        if type(stamp) is not int or stamp < previous_snapshot or type(snapshot) is not dict:
            raise ValueError("TEACHER_SENSOR_NAVIGATION_CLOCK_INVALID")
        previous_snapshot = stamp
    history = ObservationHistory(LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH,
                                 maximum_gap_ms=FLIGHT_STATE_MAXIMUM_GAP_MS)
    samples, index, active, previous = {}, 0, None, -1
    for record in records:
        stamp = record.get("recorded_at_unix_ms")
        if type(stamp) is not int or stamp < previous:
            raise ValueError("TEACHER_SENSOR_REPLAY_CLOCK_INVALID")
        previous = stamp
        while index < len(snapshots) and snapshots[index][0] <= stamp:
            active = snapshots[index][1]
            index += 1
        if active is None:
            continue
        snapshot = deepcopy(active)
        task = snapshot["strategic_context"]["task"]
        if record.get("navigation_goal_id") != task.get("navigation_goal_id"):
            history.clear()
            continue
        observation = RuntimeLocalSafetyObservation.model_validate(record.get("observation"))
        if observation.source != "onboard":
            raise ValueError("TEACHER_SENSOR_REPLAY_REQUIRES_ONBOARD_OBSERVATION")
        snapshot["current_position_m"] = observation.current_position_m.model_dump(mode="json")
        snapshot["current_velocity_mps"] = observation.current_velocity_mps.model_dump(mode="json")
        goal = snapshot["goal_position_m"]
        position = observation.current_position_m
        snapshot["goal_distance_m"] = math.dist(
            (position.x, position.y, position.z), (goal["x"], goal["y"], goal["z"]))
        snapshot["control_reference_observed_at_unix_ms"] = stamp
        snapshot["realtime_feature_snapshot"] = record.get("realtime_feature_snapshot")
        # 故障期间的健康由此刻独立日志给出；不能沿用上个健康导航快照的结论。
        healthy = observation.stream_healthy and record.get("identity_accepted") is True
        snapshot["perception_health"] = {
            "stream_healthy": healthy, "stream_age_seconds": observation.stream_age_seconds,
            "localization_covariance_m2": observation.localization_covariance_m2,
            "issue_codes": [] if healthy else ["RECORDED_SENSOR_OR_IDENTITY_UNHEALTHY"],
        }
        raw_sensors = record.get("multimodal_sensor_snapshot")
        sensors = (RuntimeMultimodalSensorSnapshot.model_validate(raw_sensors)
                   if raw_sensors is not None else None)
        snapshot["multimodal_sensor_snapshot"] = (
            sensors.model_dump(mode="json") if sensors is not None else None)
        snapshot.pop("snapshot_sha256", None)
        snapshot["snapshot_sha256"] = sha256_json(snapshot)
        batch = compile_local_policy_features(snapshot, include_candidate_features=False)
        temporal = batch.temporal_evidence
        if ("state-anomaly-detector" not in roles or temporal is None
                or not 0 <= stamp - temporal.observed_at_unix_ms <= FLIGHT_STATE_MAXIMUM_GAP_MS):
            history.clear()
        else:
            history.append(temporal, batch.state_features, batch.payload_features)
        rows = [list(row[0]) for row in history.rows]
        padding = LOCAL_POLICY_TEMPORAL_HISTORY_LENGTH - len(rows)
        state_history = [[0.0] * len(batch.state_features) for _ in range(padding)] + rows
        mask = [0.0] * padding + [1.0] * len(rows)
        targets = {"perception-health-critic": not healthy}
        if sensors is not None and sensors.statuses:
            targets["cross-modal-consistency-critic"] = not sensors.ready_for_motion
            targets["state-anomaly-detector"] = not healthy or not sensors.ready_for_motion
        for role in sorted(roles & targets.keys()):
            risky = targets[role]
            identity = sha256_json({"source": source_identity, "record": sha256_json(record),
                                    "role": role, "kind": "recorded-sensor-diagnostics"})
            samples.setdefault(identity, LocalAdvisorTrainingSample(
                role=role, state_features=list(batch.state_features), state_history=state_history,
                history_mask=mask, maneuver_features=list(batch.maneuver_features),
                payload_features=list(batch.payload_features),
                sensor_features=list(batch.sensor_features),
                risk_target=float(risky), controller_step_scale_target=1.0,
                sample_weight=3.0 if risky else 1.0))
    return samples
