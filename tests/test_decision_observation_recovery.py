"""Observation recovery must replay sensor evidence rather than accept an invented frame."""

import copy
import json

import pytest
from test_decision_input_evidence import input_fixture

from dronedream_agent_core.decision_collection_teacher import CollectionTeacher
from dronedream_agent_core.decision_observation_recovery import (
    capture_observation_recovery,
    verify_observation_recovery,
)
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：生成定位过期后恢复的两帧因果输入，完全合成且不加入正式训练数据。
# 输入：无；输出：旧状态、含完整恢复前缀的标签附件。
def recovery_fixture(*, geometry_fault=None, missing="pose"):
    _, proof = input_fixture()
    original = copy.deepcopy(proof["snapshots"][0])
    if missing == "geometry":
        from dronedream_agent_core.control_feature_contract import encoder_contract_sha256

        geometry = original["realtime_feature_snapshot"]["encodings"][1]
        geometry.update(
            source_ids=["synthetic-depth"],
            feature_contract_sha256=encoder_contract_sha256("metric-geometry-encoder"),
            encoded_at_unix_ms=9900,
            quality=1.0,
            coverage=1.0,
            uncertainty=0.0,
            features=[0.5, 0.5, 0.5, 1.0, 1.0] * 26,
            valid_mask=[1.0] * 130,
            inference_latency_ms=0.0,
        )
    old = copy.deepcopy(original)
    old["realtime_feature_snapshot"]["encodings"][0 if missing == "pose" else 1][
        "observed_at_unix_ms"
    ] = 8000
    newer = copy.deepcopy(original)
    newer["control_reference_observed_at_unix_ms"] = 12000
    for encoding in newer["realtime_feature_snapshot"]["encodings"]:
        encoding["observed_at_unix_ms"] += 2000
        encoding["source_sha256"] = "f" * 64
        if "encoded_at_unix_ms" in encoding:
            encoding["encoded_at_unix_ms"] += 2000
    if missing == "geometry":
        geometry = newer["realtime_feature_snapshot"]["encodings"][1]
        if geometry_fault == "mask":
            geometry["valid_mask"] = [0.0] * 130
        elif geometry_fault == "contract":
            geometry["feature_contract_sha256"] = "0" * 64
        elif geometry_fault == "confidence":
            geometry["features"] = [0.5, 0.5, 0.5, 0.0, 1.0] * 26
        elif geometry_fault == "coverage":
            geometry["coverage"] = 0.0
        elif geometry_fault == "shape":
            geometry["features"] = [1.0]
        elif geometry_fault == "future-encoding":
            geometry["encoded_at_unix_ms"] = 12001
    for snap in (old, newer):
        snap.pop("snapshot_sha256")
        snap["snapshot_sha256"] = decision_digest(snap)
    proof["snapshots"] = [old, newer]
    primitives = json.loads(proof["semantic_text"])["runtime_collision_primitives"]
    teacher = CollectionTeacher(
        map_sha256=old["known_static_map"]["source_sha256"],
        primitives=primitives,
        **proof["teacher"],
    )
    before, _, _ = teacher.observe(old, sequence=0, **proof["identity"])
    after, _, _ = teacher.observe(newer, sequence=1, history=(before.frame,), **proof["identity"])
    return {"state": before.model_dump(mode="json")}, {
        "observation_recovery": {"state": after.model_dump(mode="json"), "input_evidence": proof},
        "static_primitives": primitives,
        "geometry_sha256": decision_digest(primitives),
        "envelope": {k: proof["teacher"][k] for k in ("body_radius_m", "body_height_m")},
    }


# 功能：真实几何编码恢复不依赖不存在的六向距离；心跳/空掩码/错契约均不算恢复。
# 输入：明确合成的深度编码故障；输出：有效内容通过，其余拒绝，六向净空仍未知。
@pytest.mark.parametrize(
    "fault", [None, "mask", "contract", "confidence", "coverage", "shape", "future-encoding"]
)
def test_geometry_recovery_requires_measured_content(fault):
    row, document = recovery_fixture(missing="geometry", geometry_fault=fault)
    kwargs = dict(start_ms=10000, end_ms=12000, last_position=(1.0, 2.0, 3.0))
    if fault:
        with pytest.raises(ValueError, match="RECOVERY_NOT_WITNESSED"):
            verify_observation_recovery(row, document, **kwargs)
    else:
        frame = verify_observation_recovery(row, document, **kwargs)
        assert frame.geometry_source.fresh(12000)
        assert frame.clearance_front_m is None


# 功能：同一地图/目标、未改写的恢复输入通过；变更目标、位置、声明或源码均拒绝。
# 输入：逐项故障变体；输出：明确验证结论，不产生控制许可。
@pytest.mark.parametrize(
    "fault",
    [None, "missing", "identity", "state", "position", "frame", "clock", "algorithm", "map"],
)
def test_observation_recovery_is_source_bound(fault):
    row, document = recovery_fixture()
    end = 12000
    position = (1.0, 2.0, 3.0)
    recovery = document["observation_recovery"]
    if fault == "missing":
        document.pop("observation_recovery")
    elif fault == "identity":
        recovery["state"]["goal_id"] = "different"
    elif fault == "state":
        recovery["state"]["frame"]["position_world_enu_m"]["x"] += 1
    elif fault == "position":
        position = (4.0, 2.0, 3.0)
    elif fault == "frame":
        document["end_decision_frame"] = row["state"]["frame"]
    elif fault == "clock":
        end = 11900
    elif fault == "algorithm":
        recovery["input_evidence"]["algorithm"] = {}
    elif fault == "map":
        document["geometry_sha256"] = "0" * 64
    if fault is not None:
        with pytest.raises(ValueError):
            verify_observation_recovery(
                row, document, start_ms=10000, end_ms=end, last_position=position
            )
    else:
        frame = verify_observation_recovery(
            row, document, start_ms=10000, end_ms=end, last_position=position
        )
        assert frame.pose_source.fresh(12000)
        assert frame.position_world_enu_m.x == 1.0


# 功能：导出仅复制已经收到的因果前缀，不让后续帧、换目标或错快照修补当前恢复。
# 输入：合成两帧及边界突变；输出：合法导出可重建，错源/未来帧必须拒绝。
@pytest.mark.parametrize(
    "fault", [None, "future", "goal", "snapshot", "received-late", "duplicate"]
)
def test_capture_recovery_keeps_original_prefix(fault):
    row, document = recovery_fixture()
    recovery = document["observation_recovery"]
    source = recovery["input_evidence"]
    selected = {
        "state": copy.deepcopy(recovery["state"]),
        "snapshot": copy.deepcopy(source["snapshots"][-1]),
        "selected_at_ms": 12000,
    }
    selections = [selected]
    if fault == "future":
        selected["state"]["frame"]["observed_at_ms"] += 1
    elif fault == "goal":
        selected["state"]["goal_id"] = "other"
    elif fault == "snapshot":
        selected["snapshot"]["current_position_m"]["x"] += 1
    elif fault == "received-late":
        selected["selected_at_ms"] += 1
    elif fault == "duplicate":
        selections.append(copy.deepcopy(selected))
    if fault:
        with pytest.raises(ValueError, match="OBSERVATION_RECOVERY"):
            capture_observation_recovery(
                source, selections, row["state"], start_ms=10000, end_ms=12000
            )
    else:
        captured = capture_observation_recovery(
            source, [selected], row["state"], start_ms=10000, end_ms=12000
        )
        assert len(captured["observation_recovery"]["input_evidence"]["snapshots"]) == 2
        document.update(captured)
        verify_observation_recovery(
            row, document, start_ms=10000, end_ms=12000, last_position=(1.0, 2.0, 3.0)
        )
        selected["state"]["goal_id"] = "mutated-after-export"
        assert captured["observation_recovery"]["state"]["goal_id"] == row["state"]["goal_id"]
