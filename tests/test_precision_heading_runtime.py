"""Explicit learned-yaw runtime extension; no production artifact is changed by tests."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from test_causal_packaging import base_package  # noqa: F401
from test_complete_artifact_assembly import recipe  # noqa: F401
from test_heading_availability import live_snapshot
from test_precision_heading_composition import compose_pair, graphs

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256
from dronedream_agent_core.local_policy_packages import (
    LocalPolicyPackageManifest,
    load_local_policy_package,
)
from dronedream_agent_core.local_policy_port import LocalPolicyFeatureBatch, OnnxLocalPolicyBackend
from dronedream_agent_core.pilot_control_mapping import PilotControlLimits
from dronedream_agent_core.precision_heading_input import (
    current_precision_heading_input,
    validate_current_precision_heading_input,
)
from dronedream_agent_core.temporal_evidence import TemporalEvidence
from dronedream_agent_core.training.artifact_assembly import assemble_complete_ensemble


# 功能：
#   生产输入只使用当前新鲜观测，拒绝过期、错误尺度和未核验的历史动作。
# 输入：
#   无。
# 输出：
#   None：当前编码、掩码和失效边界满足断言。
def test_current_heading_input_is_explicitly_without_transport_history():
    values = current_precision_heading_input(live_snapshot(), now_unix_ms=1001)
    assert values[8:] == (0.,) * 15
    assert validate_current_precision_heading_input(values, yaw_limit_dps=20.) == values
    with pytest.raises(ValueError, match='CLOCK_INVALID'):
        current_precision_heading_input(live_snapshot(), now_unix_ms=1300)
    for yaw_limit in (True, 10., 30., float('nan')):
        with pytest.raises(ValueError, match='CURRENT_INPUT_INVALID'):
            validate_current_precision_heading_input(values, yaw_limit_dps=yaw_limit)
    for column in range(8, 23):
        changed = list(values)
        changed[column] = .1
        with pytest.raises(ValueError, match='CURRENT_INPUT_INVALID'):
            validate_current_precision_heading_input(changed, yaw_limit_dps=20.)


# 功能：
#   确认不启用扩展的旧清单序列化保持原身份，且不能无声明地塞入偏航输入。
# 输入：
#   recipe：合成十专家测试配方。
# 输出：
#   None：序列化兼容、未声明输入和错误协议均被正确处理。
def test_manifest_extension_does_not_silently_change_existing_packages(recipe):  # noqa: F811
    payload = recipe.manifest.model_dump()
    assert 'precision_heading_context_sha256' not in payload
    assert 'navigation_heading_context_sha256' not in payload
    artifact = next(a for a in payload['artifacts'] if a['role'] == 'precision-maneuver-policy')
    artifact['input_names'].append('heading_context')
    with pytest.raises(ValueError, match='tensor contract'):
        LocalPolicyPackageManifest.model_validate(payload)
    payload['precision_heading_context_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='PRECISION_HEADING_PACKAGE'):
        LocalPolicyPackageManifest.model_validate(payload)
    payload['precision_heading_context_sha256'] = HEADING_CONTEXT_SHA256
    assert LocalPolicyPackageManifest.model_validate(payload).precision_heading_context_sha256 == HEADING_CONTEXT_SHA256


# 功能：巡航偏航必须同时声明输入与正确契约，不能靠未知输入隐式启用或影响恢复模型。
# 输入：recipe：合成十专家清单。输出：缺声明、缺输入和错误摘要全部拒绝。
def test_navigation_heading_manifest_requires_both_declarations(recipe):  # noqa: F811
    payload = recipe.manifest.model_dump()
    payload['navigation_heading_context_sha256'] = HEADING_CONTEXT_SHA256
    with pytest.raises(ValueError, match='NAVIGATION_HEADING_TENSOR'):
        LocalPolicyPackageManifest.model_validate(payload)
    artifact = next(a for a in payload['artifacts'] if a['role'] == 'local-navigation-policy')
    artifact['input_names'].append('heading_context')
    accepted = LocalPolicyPackageManifest.model_validate(payload)
    assert accepted.heading_context_for_role('precision-maneuver-policy') is None
    payload['navigation_heading_context_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='NAVIGATION_HEADING_PACKAGE'):
        LocalPolicyPackageManifest.model_validate(payload)
    payload.pop('navigation_heading_context_sha256')
    with pytest.raises(ValueError, match='NAVIGATION_HEADING_TENSOR'):
        LocalPolicyPackageManifest.model_validate(payload)


# 功能：
#   在合成临时包中接入真实组合图并执行十专家检查与连续控制推理，额外输入只进入精细专家。
# 输入：
#   recipe、tmp_path：合成配方及隔离目录。
# 输出：
#   None：实际后端的偏航随具名输入变化，缺输入拒绝，其他角色无需该输入。
@pytest.mark.parametrize('role', ['precision-maneuver-policy', 'local-navigation-policy'])
def test_composed_precision_graph_runs_in_actual_backend(recipe, tmp_path, role):  # noqa: F811
    import onnx

    package = assemble_complete_ensemble(recipe=recipe, source_root=tmp_path, output_root=tmp_path/'original')
    path = package.artifact_paths[role]
    _, heading = graphs()
    content, _ = compose_pair(onnx.load_model_from_string(path.read_bytes()), heading)
    path.write_bytes(content)
    payload = package.manifest.model_dump()
    field = ('navigation_heading_context_sha256' if role == 'local-navigation-policy'
             else 'precision_heading_context_sha256')
    payload[field] = HEADING_CONTEXT_SHA256
    artifact = next(a for a in payload['artifacts'] if a['role'] == role)
    artifact.update(sha256=hashlib.sha256(content).hexdigest(), input_names=[*artifact['input_names'], 'heading_context'])
    (package.root/'manifest.json').write_text(json.dumps(payload), encoding='utf-8')
    package = load_local_policy_package(package.root)
    backend = OnnxLocalPolicyBackend(package, execution_providers=['CPUExecutionProvider'],
                                     allow_precomputed_visual_features=True)
    try:
        assert backend.verify_runtime_io()['expert_count'] == 10
        batch = LocalPolicyFeatureBatch(state_features=(0.,) * 46,
            candidate_features=((0.,) * 15,) * 8, candidate_mask=(0.,) * 8, candidate_ids=(),
            realtime_features=(0.,) * 446, realtime_valid_mask=(1.,) * 446,
            realtime_features_ready=True, realtime_snapshot_sha256='a'*64,
            control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
            precomputed_visual_features=(0.,), navigation_expert_role=role,
            temporal_evidence=TemporalEvidence(stream_id='test', sample_sha256='b'*64, observed_at_unix_ms=1000),
            pilot_control_limits=PilotControlLimits(horizontal_speed_mps=.2, vertical_speed_mps=.2, yaw_rate_dps=20.))
        with pytest.raises(ValueError, match='CURRENT_INPUT_INVALID'):
            backend.infer(batch, multimodal=[])
        values = (.5, .5, 1., 0., 0., 0., 0., 0., *(0.,)*15)
        result = backend.infer(replace(batch, precision_heading_context=values), multimodal=[])
        assert result.pilot_control.yaw_axis == pytest.approx(np.tanh(.5), abs=1e-6)
        assert role in result.expert_latency_ms
        other = 'local-navigation-policy' if role == 'precision-maneuver-policy' else 'precision-maneuver-policy'
        general = backend.infer(replace(batch, navigation_expert_role=other), multimodal=[])
        assert other in general.expert_latency_ms
    finally:
        backend.close()
