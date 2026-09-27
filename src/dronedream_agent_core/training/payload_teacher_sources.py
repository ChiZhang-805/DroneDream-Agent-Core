"""Admit completed native payload measurement courses, never production flight evidence."""

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from dronedream_plugin_sdk.protocol import decode_json

from ..contracts import (
    RuntimeActionExecutionContract,
    RuntimeActionExecutionReceipt,
    RuntimeCheckpointContract,
    RuntimeCheckpointRequest,
)
from ..control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from ..hashing import sha256_json
from ..plugin_files import read_plugin_file
from ..simulation_payload_runtime import CONTRACT as PLACEMENT_CONTRACT
from ..simulation_payload_runtime import LIBRARY as PLACEMENT_LIBRARY
from ..simulation_teacher_contract import SIMULATION_TEACHER_CONTRACT_SHA256
from .advisor_sources import RecordedSource
from .evidence_files import decode_evidence_rows, read_evidence_dataset, read_evidence_object
from .mission_groups import recorded_mission_group
from .payload_checkpoint import PayloadTeacherCheckpointDecision, decide_payload_teacher_checkpoint


# 功能：
#   核对新原生放置记录的衍生资产、组件字节和实际质量惯量，保留旧物理采集的独立来源。
# 输入：
#   root：运行证据目录；actions：动作合同；receipts：已经验证的动作回执。
# 输出：
#   hashes：本次实际读取的放置组件证据摘要。
def verify_payload_placement_source(root, actions, receipts):
    alignment = receipts[1]['output'].get('payload_mount_alignment', {})
    service = alignment.get('set_pose', {}).get('service', '')
    placement = root / 'payload-placement'
    if not service.endswith('/place_detached') and not placement.exists():
        return {}
    deployment, deployment_sha = read_evidence_object(placement / 'deployment.json')
    spawn, spawn_sha = read_evidence_object(root / 'payload_spawn.json')
    params = actions.steps[2].parameters
    library = read_plugin_file(placement / PLACEMENT_LIBRARY, limit=16 * 1024**2)
    sdf = read_plugin_file(placement / 'payload.sdf', limit=4 * 1024**2)
    library_sha, sdf_sha = hashlib.sha256(library).hexdigest(), hashlib.sha256(sdf).hexdigest()
    if (deployment.get('contract') != PLACEMENT_CONTRACT
            or deployment.get('original_sha256') != params['payload_sdf_sha256']
            or deployment.get('library_sha256') != library_sha
            or not library.startswith(b'\x7fELF\x02\x01')
            or deployment.get('staged_sha256') != sdf_sha
            or spawn.get('placement_runtime') != deployment
            or spawn.get('sdf_sha256') != sdf_sha
            or spawn.get('sdf_path') != deployment.get('sdf_path')
            or spawn.get('accepted') is not True
            or not service.endswith('/model/' + spawn.get('entity_name', '') + '/place_detached')):
        raise ValueError('PAYLOAD_TEACHER_PLACEMENT_BINDING_INVALID')
    if b'<!DOCTYPE' in sdf.upper() or b'<!ENTITY' in sdf.upper():
        raise ValueError('PAYLOAD_TEACHER_PLACEMENT_SDF_INVALID')
    tree = ET.fromstring(sdf)
    plugins, links = list(tree.iter('plugin')), list(tree.iter('link'))
    if (tree.tag != 'sdf' or len(tree.findall('model')) != 1
            or len(list(tree.iter('model'))) != 1 or len(links) != 1 or len(plugins) != 1
            or plugins[0].get('name') != 'dronedream::PayloadPlacement'
            or plugins[0].get('filename') != str(deployment['sdf_path']).rsplit('/', 1)[0]
            + '/' + PLACEMENT_LIBRARY):
        raise ValueError('PAYLOAD_TEACHER_PLACEMENT_SDF_INVALID')
    for xpath, expected in [('inertial/mass', params['payload_mass_kg'])] + [
            ('inertial/inertia/' + key, value) for key, value in params['payload_inertia_kg_m2'].items()]:
        value = float(links[0].findtext(xpath, 'nan'))
        if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('PAYLOAD_TEACHER_PLACEMENT_PHYSICS_CHANGED')
    hashes = {'payload-placement/deployment.json': deployment_sha,
              'payload_spawn.json': spawn_sha, 'payload-placement/payload.sdf': sdf_sha,
              'payload-placement/' + PLACEMENT_LIBRARY: library_sha}
    return hashes


# 功能：
#   检查每个真实设备动作与冻结步骤的身份、依赖、回读证据和先后时间，不接受仅有成功标记。
# 输入：
#   contract：冻结执行契约；receipts：按契约步骤顺序读取的原始回执。
# 输出：
#   verified：通过校验的类型化回执列表。
def verify_payload_teacher_actions(contract, receipts):
    contract = RuntimeActionExecutionContract.model_validate(contract)
    expected_actions = ['delivery.precontact-hold', 'training.payload-attach-fixture',
                        'delivery.confirm-custody', 'delivery.verify-loaded-stability']
    if [step.action for step in contract.steps] != expected_actions or len(receipts) != 4:
        raise ValueError('PAYLOAD_TEACHER_ACTION_COURSE_INVALID')
    verified, completed, prior_time = [], set(), None
    for step, raw in zip(contract.steps, receipts, strict=True):
        receipt = RuntimeActionExecutionReceipt.model_validate(raw)
        if (receipt.execution_contract_sha256 != sha256_json(contract)
                or receipt.step_sha256 != sha256_json(step)
                or any(getattr(receipt, key) != getattr(step, key)
                       for key in ('step_id', 'task_id', 'action', 'adapter_id',
                                   'runtime_executor'))
                or receipt.status != 'accepted' or receipt.issue_codes
                or receipt.attempts > step.max_attempts
                or receipt.output.get('confirmed') is not True
                or receipt.deterministic_gates != {'driver_confirmed': True,
                    'required_evidence_observed': True, 'output_bound_to_adapter': True,
                    'attempt_within_limit': True}
                or not set(step.required_success_evidence).issubset(
                    receipt.observed_success_evidence)
                or not set(step.depends_on) <= completed
                or receipt.started_at > receipt.completed_at
                or (prior_time is not None and receipt.started_at < prior_time)):
            raise ValueError('PAYLOAD_TEACHER_ACTION_RECEIPT_INVALID')
        completed.add(step.task_id)
        prior_time = receipt.completed_at
        verified.append(receipt)
    attach = verified[1].output
    alignment = attach.get('payload_mount_alignment', {})
    if type(alignment) is not dict:
        raise ValueError('PAYLOAD_TEACHER_PHYSICAL_READBACK_INVALID')
    readback = alignment.get('attachment_pose_readback', {})
    if type(readback) is not dict:
        raise ValueError('PAYLOAD_TEACHER_PHYSICAL_READBACK_INVALID')
    positions = [readback.get(key) for key in (
        'payload_position_world_enu_m', 'expected_position_world_enu_m')]
    if any(type(point) is not list or len(point) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in point)
            for point in positions):
        raise ValueError('PAYLOAD_TEACHER_PHYSICAL_READBACK_INVALID')
    error, limit = readback.get('alignment_error_m'), readback.get('maximum_alignment_error_m')
    if (attach.get('transport') != 'gazebo-transport' or attach.get('detached') is not False
            or attach.get('post_publication_realignments') != 0
            or attach.get('state_readback_source') != 'gazebo-detachable-joint-event'
            or alignment.get('binding_sha256')
            != contract.steps[1].parameters.get('payload_mount_binding_sha256')
            or type(error) not in (int, float) or type(limit) not in (int, float)
            or not 0 <= error <= limit <= .1
            or not math.isclose(error, math.dist(*positions), rel_tol=1e-8, abs_tol=1e-9)
            or limit != contract.steps[1].parameters.get('payload_mount_max_alignment_error_m')
            or readback.get('accepted') is not True
            or verified[2].output.get('custody_state_accepted') is not True
            or verified[2].output.get('payload_physics_binding_confirmed') is not True
            or verified[3].output.get('loaded_hover_stable') is not True
            or verified[3].output.get('return_authorized') is not True):
        raise ValueError('PAYLOAD_TEACHER_PHYSICAL_READBACK_INVALID')
    # 标签必须使用合同中的真实质量与惯量，不能仅凭“绑定通过”布尔值生成。
    custody = verified[2].output
    if (custody.get('detached') is not False
            or any(custody.get(key) != contract.steps[2].parameters.get(key)
                   or key not in custody or key not in contract.steps[2].parameters
                   for key in ('payload_sdf_sha256', 'payload_mass_kg', 'payload_inertia_kg_m2'))):
        raise ValueError('PAYLOAD_TEACHER_PHYSICS_BINDING_INVALID')
    stability = verified[3].output
    if stability.get('detached') is not False:
        raise ValueError('PAYLOAD_TEACHER_STABILITY_INVALID')
    for field, bound in (('position_error_m', 'position_tolerance_m'),
                         ('speed_mps', 'speed_tolerance_mps')):
        value, maximum = stability.get(field), contract.steps[3].parameters.get(bound)
        if (type(value) not in (int, float) or type(maximum) not in (int, float)
                or not math.isfinite(value) or not math.isfinite(maximum)
                or not 0 <= value <= maximum):
            raise ValueError('PAYLOAD_TEACHER_STABILITY_INVALID')
    return verified


# 功能：
#   1. 接纳完整落地、仅因明确开发采集而排除的当前负载教师观测。
#   2. 核对物理动作、教师检查点、传感器来源和空间分组，禁止产生行为标签或飞行资格。
# 输入：
#   root：完整采集运行的 simulation 目录。
# 输出：
#   receipt：用途受限的来源记录；observations：摘要绑定的原始观测字节。
def read_payload_teacher_source(root: Path):
    root = root.absolute()
    evidence, digest = read_evidence_object(root / 'mission_evidence.json')
    gates = evidence.get('gates')
    if (evidence.get('schema_version') != 'dronedream.generic-px4-gazebo-run.v1'
            or evidence.get('status') != 'failed' or type(gates) is not dict
            or {key for key, value in gates.items() if value is not True}
            != {'development_payload_collection_absent'}
            or any(gates.get(key) is not True for key in ('landing_confirmed',
                'executor_completed', 'learning_observation_recording_complete',
                'teacher_execution_recording_complete', 'live_depth_perception_healthy'))):
        raise ValueError('PAYLOAD_TEACHER_PHYSICAL_RUN_INVALID')
    learning = evidence.get('measurements', {}).get('simulation_learning', {})
    if (learning.get('observations_recorded') is not True
            or learning.get('deterministic_teacher_control') is not True
            or learning.get('model_control_qualification_granted') is not False
            or learning.get('teacher_contract_sha256') != SIMULATION_TEACHER_CONTRACT_SHA256):
        raise ValueError('PAYLOAD_TEACHER_CONTRACT_INVALID')
    artifacts = evidence.get('artifacts', {})
    files = {'observations': ('learning-observations.jsonl', 'learning_observations_sha256'),
             'summary': ('learning-observation-summary.json',
                         'learning_observation_summary_sha256'),
             'writer': ('runtime-evidence-writer-summary.json',
                        'runtime_evidence_writer_summary_sha256')}
    contents = read_evidence_dataset(root, {key: value[0] for key, value in files.items()},
        {key: artifacts.get(value[1]) for key, value in files.items()},
        error_prefix='PAYLOAD_TEACHER_SOURCE_HASH_MISMATCH')
    summary = decode_json(contents['summary'], limit=2 * 1024**2)
    writer = decode_json(contents['writer'], limit=2 * 1024**2)
    rows = decode_evidence_rows(contents['observations'])
    if (any(value.get('complete') is not True or value.get('issue_code') is not None
            for value in (summary, writer)) or not rows
            or type(summary.get('completed')) is not int
            or summary['completed'] != len(rows) or summary.get('submitted') != len(rows)):
        raise ValueError('PAYLOAD_TEACHER_RECORDING_INCOMPLETE')
    for row in rows:
        snapshot = row.get('snapshot')
        if (row.get('evidence_kind') != 'simulation-observation-only'
                or row.get('model_invoked') is not False
                or row.get('control_authority_granted') is not False
                or row.get('policy_feature_contract_sha256')
                != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
                or type(snapshot) is not dict):
            raise ValueError('PAYLOAD_TEACHER_OBSERVATION_INVALID')
        content = dict(snapshot)
        snapshot_digest = content.pop('snapshot_sha256', None)
        if (snapshot_digest != sha256_json(content)
                or row.get('recorded_at_unix_ms')
                != snapshot.get('control_reference_observed_at_unix_ms')
                or snapshot.get('strategic_context', {}).get('task', {}).get(
                    'simulation_teacher_contract_sha256')
                != SIMULATION_TEACHER_CONTRACT_SHA256):
            raise ValueError('PAYLOAD_TEACHER_SNAPSHOT_BINDING_INVALID')
    raw, action_sha = read_evidence_object(root / 'runtime-actions.json')
    actions = RuntimeActionExecutionContract.model_validate(raw)
    raw, checkpoint_sha = read_evidence_object(root / 'runtime-checkpoints.json')
    checkpoints = RuntimeCheckpointContract.model_validate(raw)
    if actions.contract_id != checkpoints.contract_id or len(checkpoints.checkpoints) != 2:
        raise ValueError('PAYLOAD_TEACHER_COURSE_BINDING_INVALID')
    hashes = {'actions': action_sha, 'checkpoints': checkpoint_sha}
    receipts = []
    for step in actions.steps:
        raw, identity = read_evidence_object(
            root / 'runtime-actions' / 'receipts' / f'{step.step_id}.receipt.json')
        hashes[f'action/{step.step_id}'] = identity
        receipts.append(raw)
    verify_payload_teacher_actions(actions, receipts)
    hashes.update(verify_payload_placement_source(root, actions, receipts))
    for checkpoint in checkpoints.checkpoints:
        raw, identity = read_evidence_object(
            root / 'checkpoints' / f'{checkpoint.checkpoint_id}.request.json')
        request = RuntimeCheckpointRequest.model_validate(raw)
        hashes[f'request/{checkpoint.checkpoint_id}'] = identity
        raw, identity = read_evidence_object(
            root / 'checkpoints' / f'{checkpoint.checkpoint_id}.teacher-decision.json')
        decision = PayloadTeacherCheckpointDecision.model_validate(raw)
        hashes[f'decision/{checkpoint.checkpoint_id}'] = identity
        if (request.checkpoint != checkpoint or decision.continue_authorized is not True
                or decision != decide_payload_teacher_checkpoint(request, checkpoints)):
            raise ValueError('PAYLOAD_TEACHER_CHECKPOINT_RECEIPT_INVALID')
    split = recorded_mission_group(root, artifacts)
    hashes.update({key: hashlib.sha256(value).hexdigest() for key, value in contents.items()})
    receipt = {'run_directory': str(root), 'source_kind': 'verified-payload-teacher-observations',
        'mission_status': 'failed', 'mission_evidence_sha256': digest,
        'snapshot_file_sha256': hashes['observations'], 'cycle_file_sha256': None,
        'depth_safety_history_sha256': None, 'multimodal_dataset_records_sha256': None,
        'semantic_sha256': split.semantic_sha256, 'mission_split': split.model_dump(mode='json'),
        'verification': 'verified-payload-teacher-observations-only',
        'teacher_contract_sha256': SIMULATION_TEACHER_CONTRACT_SHA256,
        'verified_source_files_sha256': hashes, 'allowed_roles': ['payload-dynamics-adapter'],
        'flight_qualification_granted': False, 'behavior_action_labels_granted': False}
    observations = RecordedSource(root / files['observations'][0], contents['observations'])
    return receipt, observations
