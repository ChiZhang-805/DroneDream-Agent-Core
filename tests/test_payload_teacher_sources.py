"""Synthetic measurement receipts exercise admission, not actual flight qualification."""

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

import pytest
from test_executed_demonstration_dataset import source_fixture
from test_payload_teacher_checkpoint import checkpoint_fixture
from test_payload_teacher_curriculum import curriculum_inputs  # noqa: F401

from dronedream_agent_core.contracts import (
    RuntimeActionExecutionContract,
    RuntimeActionExecutionReceipt,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.simulation_payload_runtime import CONTRACT, LIBRARY
from dronedream_agent_core.training.payload_checkpoint import decide_payload_teacher_checkpoint
from dronedream_agent_core.training.payload_curriculum import build_payload_teacher_curriculum
from dronedream_agent_core.training.payload_teacher_sources import (
    read_payload_teacher_source,
    verify_payload_placement_source,
)
from scripts.build_local_advisor_dataset import _source_receipt


# 功能：
#   创建完整且明确为合成的教师测量记录，用于逐项破坏来源和动作绑定的测试。
# 输入：
#   tmp_path、curriculum_inputs：隔离目录、有效的课程编译输入。
# 输出：
#   root：测试数据根目录。
@pytest.fixture
def recorded_payload(tmp_path, curriculum_inputs):  # noqa: F811 - imported pytest fixture
    root = tmp_path / "source"
    source_fixture(root)
    evidence_path = root / "mission_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["status"] = "failed"
    evidence["gates"].update(
        development_payload_collection_absent=False,
        landing_confirmed=True,
        executor_completed=True,
        learning_observation_recording_complete=True,
        teacher_execution_recording_complete=True,
        live_depth_perception_healthy=True,
    )
    evidence_path.write_text(json.dumps(evidence))
    course = build_payload_teacher_curriculum(*curriculum_inputs)
    actions, checkpoints = course["actions"], course["checkpoints"]
    (root / "runtime-actions.json").write_text(actions.model_dump_json())
    (root / "runtime-checkpoints.json").write_text(checkpoints.model_dump_json())
    receipt_root = root / "runtime-actions" / "receipts"
    receipt_root.mkdir(parents=True)
    for index, step in enumerate(actions.steps):
        output = {"confirmed": True, "transport": "gazebo-transport"}
        if index == 1:
            output.update(
                detached=False,
                post_publication_realignments=0,
                state_readback_source="gazebo-detachable-joint-event",
                payload_mount_alignment={
                    "binding_sha256": step.parameters["payload_mount_binding_sha256"],
                    "attachment_pose_readback": {
                        "alignment_error_m": 0.001,
                        "payload_position_world_enu_m": [0.0, 0.0, 1.001],
                        "expected_position_world_enu_m": [0.0, 0.0, 1.0],
                        "maximum_alignment_error_m": 0.02,
                        "accepted": True,
                    },
                },
            )
        elif index == 2:
            output.update(
                custody_state_accepted=True, payload_physics_binding_confirmed=True, detached=False
            )
            output.update(
                {
                    key: step.parameters[key]
                    for key in ("payload_sdf_sha256", "payload_mass_kg", "payload_inertia_kg_m2")
                }
            )
        elif index == 3:
            output.update(
                loaded_hover_stable=True,
                return_authorized=True,
                detached=False,
                position_error_m=0.0,
                speed_mps=0.0,
            )
        stamp = datetime(2026, 9, 20, tzinfo=UTC) + timedelta(seconds=index * 2)
        receipt = RuntimeActionExecutionReceipt(
            execution_contract_sha256=sha256_json(actions),
            step_sha256=sha256_json(step),
            step_id=step.step_id,
            task_id=step.task_id,
            action=step.action,
            adapter_id=step.adapter_id,
            runtime_executor=step.runtime_executor,
            status="accepted",
            attempts=1,
            started_at=stamp,
            completed_at=stamp + timedelta(seconds=1),
            output=output,
            observed_success_evidence=step.required_success_evidence,
            deterministic_gates={
                "driver_confirmed": True,
                "required_evidence_observed": True,
                "output_bound_to_adapter": True,
                "attempt_within_limit": True,
            },
        )
        (receipt_root / f"{step.step_id}.receipt.json").write_text(receipt.model_dump_json())
    (root / "checkpoints").mkdir()
    for checkpoint in checkpoints.checkpoints:
        request, _ = checkpoint_fixture()
        request.contract_id, request.checkpoint = checkpoints.contract_id, checkpoint
        decision = decide_payload_teacher_checkpoint(request, checkpoints)
        (root / "checkpoints" / f"{checkpoint.checkpoint_id}.request.json").write_text(
            request.model_dump_json()
        )
        (root / "checkpoints" / f"{checkpoint.checkpoint_id}.teacher-decision.json").write_text(
            decision.model_dump_json()
        )
    return root


# 功能：
#   核对原生放置来源中的库、资产与动作合同，破坏任一绑定都不能接纳为训练来源。
# 输入：
#   recorded_payload：合成动作回执；mode：要破坏的证据类型。
# 输出：
#   None：通过或拒绝边界的断言结果。
@pytest.mark.parametrize('mode', ['valid', 'library', 'asset', 'mass', 'original', 'spawn'])
def test_native_placement_source_binding(recorded_payload, mode):
    root = recorded_payload
    actions = RuntimeActionExecutionContract.model_validate_json((root / 'runtime-actions.json').read_text())
    receipts = [json.loads((root / 'runtime-actions/receipts' / (s.step_id + '.receipt.json')).read_text())
                for s in actions.steps]
    receipts[1]['output']['payload_mount_alignment']['set_pose'] = {'service': '/world/school/model/parcel/place_detached'}
    placement = root / 'payload-placement'
    placement.mkdir()
    library = b'\x7fELF\x02\x01synthetic-test-only'
    (placement / LIBRARY).write_bytes(library)
    sdf_path = placement / 'payload.sdf'
    tree = ET.Element('sdf', version='1.9')
    model = ET.SubElement(tree, 'model', name='parcel')
    inertial = ET.SubElement(ET.SubElement(model, 'link', name='body'), 'inertial')
    params = actions.steps[2].parameters
    ET.SubElement(inertial, 'mass').text = str(params['payload_mass_kg'] + (1. if mode == 'mass' else 0.))
    inertia = ET.SubElement(inertial, 'inertia')
    for name, value in params['payload_inertia_kg_m2'].items():
        ET.SubElement(inertia, name).text = str(value)
    ET.SubElement(model, 'plugin', name='dronedream::PayloadPlacement', filename=(placement / LIBRARY).as_posix())
    sdf = ET.tostring(tree)
    sdf_path.write_bytes(sdf)
    deployment = dict(contract=CONTRACT, sdf_path=sdf_path.as_posix(),
        original_sha256='0' * 64 if mode == 'original' else params['payload_sdf_sha256'],
        library_sha256=hashlib.sha256(library).hexdigest(), staged_sha256=hashlib.sha256(sdf).hexdigest())
    (placement / 'deployment.json').write_text(json.dumps(deployment))
    spawn = dict(accepted=True, entity_name='parcel', placement_runtime=deployment,
                 sdf_path=deployment['sdf_path'], sdf_sha256=deployment['staged_sha256'])
    if mode == 'spawn':
        spawn['entity_name'] = 'other'
    (root / 'payload_spawn.json').write_text(json.dumps(spawn))
    if mode == 'library':
        (placement / LIBRARY).write_bytes(b'changed')
    elif mode == 'asset':
        sdf_path.write_bytes(sdf + b' ')
    if mode == 'valid':
        assert len(verify_payload_placement_source(root, actions, receipts)) == 4
    else:
        with pytest.raises(ValueError, match='PAYLOAD_TEACHER_PLACEMENT_'):
            verify_payload_placement_source(root, actions, receipts)


# 功能：
#   检查新来源实际贯通数据集入口，但不能成为一般行为或其他专家来源。
# 输入：
#   recorded_payload：完整合成课程。
# 输出：
#   None：原始字节、角色及资格断言通过。
def test_payload_teacher_source_is_explicit_and_role_limited(recorded_payload):
    receipt, observations, cycles, depth, media = _source_receipt(
        recorded_payload,
        require_verified=True,
        allow_verified_fault_recovery=False,
        allow_verified_payload_collection=True,
        allow_verified_payload_recovery=False,
    )
    assert receipt["allowed_roles"] == ["payload-dynamics-adapter"]
    assert receipt["flight_qualification_granted"] is False
    assert receipt["behavior_action_labels_granted"] is False
    assert observations.content == (recorded_payload / "learning-observations.jsonl").read_bytes()
    assert cycles is depth is media is None


# 功能：
#   逐项验证来源损坏、失败运行、错误绑定、假成功和超容差挂接不能成为训练证据。
# 输入：
#   recorded_payload：合成课程；damage：指定需要破坏的字段。
# 输出：
#   None：读取器拒绝每一种损坏。
@pytest.mark.parametrize(
    "damage",
    [
        "failed",
        "hash",
        "contract",
        "rejected",
        "missing-evidence",
        "wrong-step",
        "alignment",
        "teleport",
        "custody",
        "stability",
        "checkpoint",
        "model-call",
        "positions",
        "alignment-shape",
        "mass",
        "speed",
        "nan-position",
    ],
)
def test_payload_teacher_source_rejects_damage(recorded_payload, damage):
    root = recorded_payload
    path = root / "runtime-actions" / "receipts" / "action-002.receipt.json"
    if damage == "hash":
        (root / "learning-observations.jsonl").write_text("{}\n")
    else:
        if damage == "failed":
            path = root / "mission_evidence.json"
        elif damage in {"custody", "mass"}:
            path = path.with_name("action-003.receipt.json")
        elif damage in {"stability", "speed"}:
            path = path.with_name("action-004.receipt.json")
        elif damage in {"checkpoint", "model-call"}:
            path = root / "checkpoints" / "checkpoint-002.teacher-decision.json"
        value = json.loads(path.read_text())
        if damage == "failed":
            value["gates"]["landing_confirmed"] = False
        elif damage == "contract":
            value["execution_contract_sha256"] = "0" * 64
        elif damage == "rejected":
            value["status"] = "rejected"
        elif damage == "missing-evidence":
            value["observed_success_evidence"] = []
        elif damage == "wrong-step":
            value["task_id"] = "other"
        elif damage == "alignment":
            value["output"]["payload_mount_alignment"]["attachment_pose_readback"][
                "alignment_error_m"
            ] = 0.03
        elif damage == "teleport":
            value["output"]["post_publication_realignments"] = 1
        elif damage == "custody":
            value["output"]["custody_state_accepted"] = False
        elif damage == "stability":
            value["output"]["loaded_hover_stable"] = False
        elif damage == "checkpoint":
            value["continue_authorized"] = False
        elif damage == "model-call":
            value["model_call_performed"] = True
        elif damage == "alignment-shape":
            value["output"]["payload_mount_alignment"] = []
        elif damage == "positions":
            value["output"]["payload_mount_alignment"]["attachment_pose_readback"][
                "payload_position_world_enu_m"
            ] = [0.0, 0.0, 2.0]
        elif damage == "nan-position":
            value["output"]["payload_mount_alignment"]["attachment_pose_readback"][
                "payload_position_world_enu_m"
            ] = [float("nan"), 0.0, 1.0]
        elif damage == "mass":
            value["output"]["payload_mass_kg"] = 99.0
        elif damage == "speed":
            value["output"]["speed_mps"] = 99.0
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_payload_teacher_source(root)
