"""Reconstruct candidate inputs from causal observations, never from outcome truth."""

import hashlib
from collections import deque
from pathlib import Path

from .decision_collection_teacher import CollectionTeacher
from .decision_dataset import decode_object, file_digest
from .decision_state_adapter import decision_digest


# 功能：冻结输入重建算法身份；输入：本版本源码；输出：内容摘要，旧数据须用原版本复核。
def input_algorithm_identity(version=1):
    root = Path(__file__).parent
    if type(version) is not int or version not in (1, 2):
        raise ValueError("DECISION_INPUT_ALGORITHM_VERSION_INVALID")
    return {
        name: file_digest(root / name)
        for name in (
            "decision_collection_teacher.py",
            "decision_runtime_adapter.py",
            "decision_dynamic_context.py",
            "decision_state_adapter.py",
            "training/swept_geometry.py",
        )
        + (("decision_route_teacher.py",) if version == 2 else ())
    }


# 功能：从完整因果前缀重建状态与地图判定，拒绝只有哈希而无原始输入的候选。
# 输入：状态、原地图字节文本、教师参数、按实际顺序的导航快照；输出：重建状态和覆盖缺口。
# 独立仿真真值只能用于结果标签，不得放入本证据的快照或历史。
def verify_input_evidence(state, proof):
    version = 2 if proof.get("schema_version") == "dronedream.decision-input-evidence.v2" else 1
    required = {"schema_version", "algorithm", "semantic_text", "teacher", "identity", "snapshots"}
    if version == 2:
        required.add("qualified_route")
    if (
        set(proof) != required
        or proof["schema_version"] != f"dronedream.decision-input-evidence.v{version}"
        or proof["algorithm"] != input_algorithm_identity(version)
    ):
        raise ValueError("DECISION_INPUT_ALGORITHM_MISMATCH")
    semantic_text = proof["semantic_text"]
    if (
        type(semantic_text) is not str
        or len(semantic_text) > 4 * 1024**2
        or hashlib.sha256(semantic_text.encode("utf-8")).hexdigest() != state["map_sha256"]
    ):
        raise ValueError("DECISION_INPUT_MAP_SOURCE_INVALID")
    semantic = decode_object(semantic_text)
    if semantic.get("schema_version") != "dronedream.map-semantic.v1":
        raise ValueError("DECISION_INPUT_MAP_SCHEMA_INVALID")
    primitives = semantic.get("runtime_collision_primitives", semantic.get("collision_primitives"))
    parameters = proof["teacher"]
    if set(parameters) != {"body_radius_m", "body_height_m", "nominal_speed_mps", "clearance_m"}:
        raise ValueError("DECISION_INPUT_TEACHER_PARAMETERS_INVALID")
    if parameters["body_radius_m"] != state["body_radius_m"]:
        raise ValueError("DECISION_INPUT_VEHICLE_MISMATCH")
    identity = proof["identity"]
    if set(identity) != {"mission_id", "calibration_sha256", "braking_acceleration_mps2", "scene"}:
        raise ValueError("DECISION_INPUT_IDENTITY_INVALID")
    snapshots = proof["snapshots"]
    # 容纳动态障碍的一次完整出现—离开—恢复；原始时间/总文件大小仍独立有界。
    if type(snapshots) is not list or not 1 <= len(snapshots) <= 512:
        raise ValueError("DECISION_INPUT_PREFIX_INVALID")
    if version == 2:
        from .decision_route_teacher import RouteCollectionTeacher

        teacher = RouteCollectionTeacher(
            map_sha256=state["map_sha256"],
            primitives=primitives,
            qualified_route=proof["qualified_route"],
            **parameters,
        )
    else:
        teacher = CollectionTeacher(
            map_sha256=state["map_sha256"], primitives=primitives, **parameters
        )
    history, previous_identity, last_time = deque(maxlen=10), None, -1
    for sequence, snapshot in enumerate(snapshots):
        now = snapshot["control_reference_observed_at_unix_ms"]
        if type(now) is not int or now <= last_time:
            raise ValueError("DECISION_INPUT_PREFIX_CLOCK_INVALID")
        current_identity = (
            snapshot["strategic_context"]["task"]["navigation_goal_id"],
            snapshot["known_static_map"]["qualified_route_sha256"],
        )
        if current_identity != previous_identity:
            history.clear()
        while history and not 0 < now - history[0].observed_at_ms <= 2500:
            history.popleft()
        rebuilt, _, issues = teacher.observe(
            snapshot, sequence=sequence, history=tuple(history), **identity
        )
        history.append(rebuilt.frame)
        previous_identity, last_time = current_identity, now
    if decision_digest(rebuilt.model_dump(mode="json")) != decision_digest(state):
        raise ValueError("DECISION_INPUT_RECONSTRUCTION_MISMATCH")
    return rebuilt, issues


# 功能：标签地图与输入地图必须同源，避免用更空旷的另一幅几何把危险动作评为安全。
# 输入：已验证输入证据、独立结果；输出：无，地图或机体包络不一致时拒绝。
def verify_input_outcome_geometry(proof, outcome):
    semantic = decode_object(proof["semantic_text"])
    primitives = semantic.get("runtime_collision_primitives", semantic.get("collision_primitives"))
    if (
        decision_digest(primitives) != outcome["geometry_sha256"]
        or decision_digest(outcome["static_primitives"]) != outcome["geometry_sha256"]
        or any(
            proof["teacher"][field] != outcome["envelope"][field]
            for field in ("body_radius_m", "body_height_m")
        )
    ):
        raise ValueError("DECISION_INPUT_OUTCOME_GEOMETRY_MISMATCH")
    continuation = outcome.get("continuation")
    if continuation is not None:
        # 恢复片段也是标签依据，其输入必须独立重建；仅有“已恢复”状态和输出哈希不够。
        if (
            not isinstance(continuation, dict)
            or "input_evidence" not in continuation
            or continuation["outcome"].get("continuation") is not None
            or continuation["outcome"]["behavior_application"]["action"]
            not in {"follow_route", "slow_down"}
        ):
            raise ValueError("DECISION_RECOVERY_INPUT_EVIDENCE_REQUIRED")
        verify_input_evidence(continuation["row"]["state"], continuation["input_evidence"])
        verify_input_outcome_geometry(continuation["input_evidence"], continuation["outcome"])
