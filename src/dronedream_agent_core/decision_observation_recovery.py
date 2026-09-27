"""Bind observation recovery to original sensor inputs, not a claimed end frame."""

import copy
import math

from .control_feature_contract import encoder_contract_sha256
from .decision_input_evidence import verify_input_evidence, verify_input_outcome_geometry
from .decision_state_adapter import decision_digest
from .realtime_feature_encoders import RealtimeFeatureEncoding


# 功能：核验原始几何编码确有新鲜测量内容，不把目标相对扇区伪装成六向机体净空。
# 输入：已因果重建的末帧证据和来源身份；输出：至少一个有效FRU扇区是否恢复。
# 只证明观测返回；不证明整圈可见、可通行或授予运动权限。
def _geometry_content_present(proof, source):
    if source is None:
        return False
    encodings = proof["snapshots"][-1].get("realtime_feature_snapshot", {}).get("encodings", [])
    matches = [e for e in encodings if e.get("encoder_role") == "metric-geometry-encoder"]
    if len(matches) != 1:
        return False
    try:
        encoding = RealtimeFeatureEncoding.model_validate(matches[0])
    except ValueError:
        return False
    if (
        encoding.feature_contract_sha256 != encoder_contract_sha256("metric-geometry-encoder")
        or encoding.source_sha256 != source.evidence_sha256
        or encoding.observed_at_unix_ms != source.observed_at_ms
        or encoding.maximum_age_milliseconds != source.maximum_age_ms
        or encoding.encoded_at_unix_ms
        > proof["snapshots"][-1]["control_reference_observed_at_unix_ms"]
        or encoding.quality <= 0
        or encoding.coverage <= 0
    ):
        return False
    return any(
        all(v == 1 for v in encoding.valid_mask[i : i + 5])
        and all(0 <= v <= 1 for v in encoding.features[i : i + 5])
        and encoding.features[i + 1] > 0
        and encoding.features[i + 3] > 0
        and encoding.features[i + 4] > 0
        for i in range(0, 130, 5)
    )


# 功能：离线把执行末期真正收到的最后一帧及完整因果前缀装入恢复附件，不生成新观测。
# 输入：原始输入归档、完整选择序列、起始状态及UNIX毫秒窗口；输出：附件或明确缺失错误。
# 不按成功与否挑帧；选择末帧后，独立验证器再决定是否真的恢复。
def capture_observation_recovery(source, selections, state, *, start_ms, end_ms):
    if (
        type(selections) is not list
        or not 1 <= len(selections) <= 512
        or type(start_ms) is not int
        or type(end_ms) is not int
        or start_ms >= end_ms
    ):
        raise ValueError("DECISION_OBSERVATION_RECOVERY_CAPTURE_INVALID")
    chosen = None
    previous_sequence, previous_time = -1, -1
    for selected in selections:
        candidate = selected["state"]
        if hasattr(candidate, "model_dump"):
            candidate = candidate.model_dump(mode="json")
        sequence, observed = candidate["sequence"], candidate["frame"]["observed_at_ms"]
        received = selected.get("selected_at_ms")
        if (
            type(sequence) is not int
            or type(observed) is not int
            or type(received) is not int
            or sequence <= previous_sequence
            or observed <= previous_time
            or received < observed
        ):
            raise ValueError("DECISION_OBSERVATION_RECOVERY_CAPTURE_ORDER_INVALID")
        previous_sequence, previous_time = sequence, observed
        if (
            candidate["sequence"] > state["sequence"]
            and start_ms < candidate["frame"]["observed_at_ms"] <= end_ms
            and received <= end_ms
            and all(
                candidate[k] == state[k]
                for k in ("mission_id", "map_sha256", "goal_id", "route_sha256")
            )
        ):
            chosen = (candidate, selected["snapshot"])
    if chosen is None:
        raise ValueError("DECISION_OBSERVATION_RECOVERY_FRAME_NOT_RECEIVED")
    candidate, snapshot = chosen
    proof = copy.deepcopy(source)
    proof["snapshots"] = proof["snapshots"][: candidate["sequence"] + 1]
    if len(proof["snapshots"]) != candidate["sequence"] + 1 or proof["snapshots"][-1] != snapshot:
        raise ValueError("DECISION_OBSERVATION_RECOVERY_PREFIX_INVALID")
    return {
        "end_decision_frame": copy.deepcopy(candidate["frame"]),
        "observation_recovery": {"state": copy.deepcopy(candidate), "input_evidence": proof},
    }


# 功能：从恢复后的原始导航输入重建状态，验证身份和新鲜内容确实恢复；不把心跳当新观测。
# 输入：原决策样本、实际结果、执行起止UNIX毫秒及独立末位置ENU米；输出：验证后的末帧。
# 恢复证据仅作标签，不回送教师；仍须由调用方先验证实际保持和无碰撞执行。
def verify_observation_recovery(row, document, *, start_ms, end_ms, last_position):
    recovery = document.get("observation_recovery")
    if not isinstance(recovery, dict) or set(recovery) != {"state", "input_evidence"}:
        raise ValueError("DECISION_OBSERVATION_RECOVERY_INPUT_REQUIRED")
    state = row["state"]
    next_state = recovery["state"]
    if (
        not isinstance(next_state, dict)
        or any(
            next_state.get(k) != state[k]
            for k in ("mission_id", "map_sha256", "goal_id", "route_sha256", "calibration_sha256")
        )
        or next_state.get("sequence", -1) <= state["sequence"]
    ):
        raise ValueError("DECISION_OBSERVATION_RECOVERY_IDENTITY_INVALID")
    rebuilt, _ = verify_input_evidence(next_state, recovery["input_evidence"])
    # 不能以另一幅更宽地图或另一个机型的恢复结果替当前任务补齐证据。
    verify_input_outcome_geometry(
        recovery["input_evidence"],
        {k: document[k] for k in ("geometry_sha256", "static_primitives", "envelope")},
    )
    after = rebuilt.frame
    if (
        not start_ms < after.observed_at_ms <= end_ms
        or after.position_world_enu_m is None
        or math.dist(tuple(getattr(after.position_world_enu_m, k) for k in "xyz"), last_position)
        > 0.25
    ):
        raise ValueError("DECISION_OBSERVATION_RECOVERY_TIME_OR_POSITION_INVALID")
    if "end_decision_frame" in document and (
        decision_digest(document["end_decision_frame"])
        != decision_digest(after.model_dump(mode="json"))
    ):
        raise ValueError("DECISION_OBSERVATION_RECOVERY_FRAME_MISMATCH")
    recovered = False
    source_frame = state["frame"]
    for name in ("pose_source", "geometry_source", "route_source"):
        old, new = source_frame[name], getattr(after, name)
        missing = (
            old is None
            or source_frame["observed_at_ms"] - old["observed_at_ms"] > old["maximum_age_ms"]
        )
        content_present = {
            "pose_source": after.position_world_enu_m is not None
            and after.position_uncertainty_m is not None,
            "geometry_source": _geometry_content_present(
                recovery["input_evidence"], after.geometry_source
            ),
            "route_source": after.local_route_verified is not None,
        }[name]
        recovered |= (
            missing
            and new is not None
            and new.fresh(end_ms)
            and new.observed_at_ms > start_ms
            and content_present
            and (old is None or new.evidence_sha256 != old["evidence_sha256"])
        )
    if not recovered:
        raise ValueError("DECISION_OBSERVATION_RECOVERY_NOT_WITNESSED")
    return after
