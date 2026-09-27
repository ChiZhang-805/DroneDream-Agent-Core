"""Input provenance is reconstructed, not accepted because a SHA string exists."""

import copy
import hashlib
import json

import pytest
from test_decision_collection_teacher import fixture

from dronedream_agent_core.decision_collection_teacher import CollectionTeacher
from dronedream_agent_core.decision_input_evidence import (
    input_algorithm_identity,
    verify_input_evidence,
)
from dronedream_agent_core.decision_state_adapter import decision_digest


# 功能：构造可重放的明确合成输入；输入：无；输出：状态与完整来源，禁止计入正式数量。
def input_fixture():
    _, snapshot = fixture()
    primitive = {
        "shape": "box",
        "center_x": 20.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size_x": 1.0,
        "size_y": 1.0,
        "size_z": 1.0,
    }
    semantic = json.dumps(
        {
            "schema_version": "dronedream.map-semantic.v1",
            "runtime_collision_primitives": [primitive],
        }
    )
    map_hash = hashlib.sha256(semantic.encode()).hexdigest()
    snapshot["known_static_map"]["source_sha256"] = map_hash
    snapshot.pop("snapshot_sha256")
    snapshot["snapshot_sha256"] = decision_digest(snapshot)
    parameters = dict(body_radius_m=0.25, body_height_m=0.3, nominal_speed_mps=0.4, clearance_m=0.2)
    identity = dict(
        mission_id="fixture",
        calibration_sha256="a" * 64,
        braking_acceleration_mps2=0.8,
        scene="corridor",
    )
    teacher = CollectionTeacher(map_sha256=map_hash, primitives=[primitive], **parameters)
    state, _, _ = teacher.observe(snapshot, sequence=0, **identity)
    proof = dict(
        schema_version="dronedream.decision-input-evidence.v1",
        algorithm=input_algorithm_identity(),
        semantic_text=semantic,
        teacher=parameters,
        identity=identity,
        snapshots=[snapshot],
    )
    return state.model_dump(mode="json"), proof


# 功能：确认同版本可确定性重建；输入：因果来源；输出：一致状态。
def test_reconstructs_exact_input():
    state, proof = input_fixture()
    rebuilt, _ = verify_input_evidence(state, proof)
    assert rebuilt.model_dump(mode="json") == state


# 功能：重签摘要不能掩盖修改后的状态或缺失来源；输入：逐项篡改；输出：全部拒绝。
@pytest.mark.parametrize("fault", ["state", "map", "truth", "algorithm", "clock", "history"])
def test_rejects_unreconstructible_inputs(fault):
    state, proof = input_fixture()
    if fault == "state":
        state["frame"]["local_route_verified"] = False
    elif fault == "map":
        proof["semantic_text"] += " "
    elif fault == "algorithm":
        proof["algorithm"] = {}
    elif fault == "truth":
        raw = proof["snapshots"][0]
        raw["source_of_truth"] = "simulation-ground-truth"
        raw.pop("snapshot_sha256")
        raw["snapshot_sha256"] = decision_digest(raw)
    elif fault == "clock":
        proof["snapshots"].append(copy.deepcopy(proof["snapshots"][0]))
    else:
        state["sequence"] = 1
    with pytest.raises(ValueError):
        verify_input_evidence(state, proof)


# 功能：恢复状态同样需要原始输入重建，防止只修改crossing标志伪造等待成功。
# 输入：明确合成输入及带恢复附件的几何上下文；输出：完整来源通过，篡改全部拒绝。
@pytest.mark.parametrize("fault", [None, "missing-proof", "state", "geometry", "recursive"])
def test_recovery_input_is_independently_rebuilt(fault):
    from dronedream_agent_core.decision_input_evidence import verify_input_outcome_geometry

    state, proof = input_fixture()
    primitives = json.loads(proof["semantic_text"])["runtime_collision_primitives"]
    outcome = dict(geometry_sha256=decision_digest(primitives), static_primitives=primitives,
                   envelope=dict(body_radius_m=.25, body_height_m=.3),
                   behavior_application=dict(action="follow_route"))
    document = copy.deepcopy(outcome)
    continuation = dict(row=dict(state=state), input_evidence=proof, outcome=outcome)
    document["continuation"] = continuation
    if fault == "missing-proof":
        continuation.pop("input_evidence")
    elif fault == "state":
        state["frame"]["crossing_obstacle"] = not state["frame"]["crossing_obstacle"]
    elif fault == "geometry":
        outcome["geometry_sha256"] = "a"*64
    elif fault == "recursive":
        outcome["continuation"] = {}
    if fault:
        with pytest.raises(ValueError):
            verify_input_outcome_geometry(proof, document)
    else:
        verify_input_outcome_geometry(proof, document)
