"""Algorithm/export tests with synthetic observations, never flight receipts."""

import json
from dataclasses import replace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from test_offline_flight_learning import replay
from test_temporal_evidence import evidence

from dronedream_agent_core.causal_control import (
    CONTROL_HISTORY_CONTRACT_SHA256,
    CONTROL_HISTORY_WIDTH,
    CausalControlHistory,
)
from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_policy_port import LocalPolicyFeatureBatch, OnnxLocalPolicyBackend
from dronedream_agent_core.training.causal_policy import (
    CausalPolicyConfig,
    causal_examples,
    example_tensors,
    export_causal_policy,
    train_causal_policy,
)


# 功能：
#   为合成回放附上独立来源身份，构造可分离任务组的训练夹具。
# 输入：
#   start：来源序号的偏移量。
#   stream：所属来源流名称。
# 输出：
#   rows：带合成时序证据的监督记录列表。
def samples(start=0, stream="train"):
    rows = [sample.model_copy(update={"temporal_evidence": evidence(i + start, stream=stream)})
            for i, sample in enumerate(replay())]
    return rows


# 功能：
#   重复轮询不能填满历史，重复来源与缺失任务分组均被拒绝，合法窗口宽度保持部署契约。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_causal_windows_do_not_cross_streams_or_make_polls_into_samples():
    rows = samples()
    history = CausalControlHistory(4)
    row = rows[0]
    for _ in range(20):
        history.append(row.temporal_evidence, row.state_features,
                       row.realtime_features, row.realtime_valid_mask)
    assert not history.ready
    assert history.values()[1] == [0, 0, 0, 1]
    with pytest.raises(ValueError, match="DUPLICATE_SOURCE"):
        causal_examples([*rows, rows[0]], stream_groups={"train": "route-a"}, history_length=4)
    with pytest.raises(ValueError, match="MISSION_GROUP_MISSING"):
        causal_examples(rows, stream_groups={}, history_length=4)
    examples = causal_examples(rows, stream_groups={"train": "route-a"}, history_length=4)
    assert len(examples) == len(rows) - 3
    assert examples[0].history_mask == [1] * 4
    assert len(examples[0].history[0]) == CONTROL_HISTORY_WIDTH


# 功能：
#   无标签观测可以提供上下文，但缺少来源或与当前标签不匹配时不能生成训练窗口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_unlabelled_observations_fill_history_without_manufacturing_action_labels():
    from dronedream_agent_core.local_policy_training import LocalPolicyObservation

    rows = samples()[:4]
    observations = [LocalPolicyObservation(**{name: getattr(row, name)
                    for name in LocalPolicyObservation.model_fields}) for row in rows]
    examples = causal_examples([rows[-1]], history_observations=observations,
                               stream_groups={"train": "route-a"}, history_length=4)
    assert len(examples) == 1
    assert examples[0].sample == rows[-1]
    assert examples[0].source_sha256 == tuple(r.temporal_evidence.sample_sha256 for r in rows)
    assert all(not hasattr(o, "target_pilot_control") for o in observations)
    changed = observations[-1].model_copy(update={"state_features": [1.] * 46})
    with pytest.raises(ValueError, match="LABEL_OBSERVATION_MISMATCH"):
        causal_examples([rows[-1]], history_observations=[*observations[:-1], changed],
                        stream_groups={"train": "route-a"}, history_length=4)
    with pytest.raises(ValueError, match="LABEL_WITHOUT_SOURCE"):
        causal_examples([rows[-1]], history_observations=observations[:-1],
                        stream_groups={"train": "route-a"}, history_length=4)


# 功能：
#   以合成数据实际训练、导出并运行 ONNX，验证历史影响四轴输出且部署不依赖坐标候选。
# 输入：
#   tmp_path：隔离的模型导出及清单目录。
# 输出：
#   None：不返回业务数据。
def test_train_export_and_actual_backend_use_temporal_control(tmp_path):
    config = CausalPolicyConfig(history_length=4, encoder_width=32, recurrent_width=32,
                                head_width=32, epochs=2)
    train = causal_examples(samples(), stream_groups={"train": "route-a"}, history_length=4)
    validation = causal_examples(samples(100, "val"), stream_groups={"val": "route-b"},
                                 history_length=4)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, metrics = train_causal_policy(train, validation, config)
        assert metrics["parameter_count"] == 39_849
        assert metrics["qualified_for_flight"] is False
        path = tmp_path / "navigation.onnx"
        digest = export_causal_policy(model, path)
        tensor_inputs = example_tensors([validation[0]], 0)
        with torch.no_grad():
            expected = model(*tensor_inputs)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feeds = {item.name: tensor.numpy()
                 for item, tensor in zip(session.get_inputs(), tensor_inputs, strict=True)}
        actual = session.run(None, feeds)
        for reference, output in zip(expected, actual, strict=True):
            np.testing.assert_allclose(reference.numpy(), output, atol=1e-5)
        # Same current sensors, different prior dynamics: a genuine learned
        # recurrent input must influence axes, not just annotate a decision.
        perturbed = {**feeds, "control_history": feeds["control_history"].copy()}
        perturbed["control_history"][:, :-1, :46] = 2.
        changed = session.run(["pilot_control"], perturbed)[0]
        assert not np.allclose(changed, actual[3], atol=1e-6)
        manifest = {
            "package_id": "causal-unit", "display_name": "Causal unit test", "scope": "general",
            "vehicle_sha256": "a" * 64, "sensor_contract_sha256": "b" * 64,
            "maximum_inference_latency_ms": 250, "realtime_feature_count": 446,
            "control_feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
            "pilot_control_mode": "normalized-body-velocity",
            "navigation_architecture": "causal-gru-control", "navigation_history_length": 4,
            "navigation_history_contract_sha256": CONTROL_HISTORY_CONTRACT_SHA256,
            "artifacts": [{"role": "local-navigation-policy", "relative_path": path.name,
                           "sha256": digest, "input_names": [i.name for i in session.get_inputs()],
                           "output_names": [i.name for i in session.get_outputs()]}],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        backend = OnnxLocalPolicyBackend(load_local_policy_package(tmp_path))
        first = samples()[0]
        batch = LocalPolicyFeatureBatch(
            state_features=tuple(first.state_features),
            candidate_features=tuple(tuple(row) for row in first.candidate_features),
            candidate_mask=tuple(first.candidate_mask), candidate_ids=(None,) * 8,
            realtime_features=tuple(first.realtime_features),
            realtime_valid_mask=tuple(first.realtime_valid_mask), realtime_features_ready=True,
            realtime_snapshot_sha256="a" * 64,
            control_feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        )
        try:
            for index in range(4):
                batch = replace(batch, temporal_evidence=evidence(index))
                backend.prepare_temporal_context(batch)
                assert backend.motion_history_ready() is (index == 3)
                inferred = backend.infer(batch, multimodal=[])
                assert inferred.pilot_control is not None
            assert backend._control_history.values()[1] == [1.] * 4
        finally:
            backend.close()
        graph = onnx.load(str(path))
        assert all("candidate" not in i.name for i in graph.graph.input)
    finally:
        torch.set_num_threads(original_threads)


# 功能：
#   同一任务组不能同时作为训练集和留出集，防止评价泄漏。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_same_mission_cannot_be_both_training_and_validation():
    train = causal_examples(samples(), stream_groups={"train": "same-route"}, history_length=4)
    validation = causal_examples(samples(100, "val"), stream_groups={"val": "same-route"},
                                 history_length=4)
    with pytest.raises(ValueError, match="MISSION_GROUP_LEAKAGE"):
        train_causal_policy(train, validation, CausalPolicyConfig(history_length=4, epochs=1))


# 功能：
#   精调必须使用独立参数，不修改基座；架构不匹配或非有限初始权重必须拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_refinement_preserves_base_and_rejects_architecture_mismatch():
    from dronedream_agent_core.training.causal_policy import CausalPilotPolicy

    config = CausalPolicyConfig(history_length=4, encoder_width=32, recurrent_width=32,
                                head_width=32, epochs=1, learning_rate=.0001)
    train = causal_examples(samples(), stream_groups={"train": "route-a"}, history_length=4)
    validation = causal_examples(samples(100, "val"), stream_groups={"val": "route-b"},
                                 history_length=4)
    initial = CausalPilotPolicy(config)
    before = {name: tensor.clone() for name, tensor in initial.state_dict().items()}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        refined, metrics = train_causal_policy(train, validation, config, initial_policy=initial)
        assert metrics["initialization"] == "existing-causal-policy"
        assert all(torch.equal(initial.state_dict()[name], value) for name, value in before.items())
        assert any(not torch.equal(refined.state_dict()[name], value)
                   for name, value in before.items())
        incompatible = CausalPilotPolicy(config.model_copy(update={"encoder_width": 64}))
        with pytest.raises(ValueError, match="REFINEMENT_ARCHITECTURE_MISMATCH"):
            train_causal_policy(train, validation, config, initial_policy=incompatible)
        with torch.no_grad():
            initial.mode_head.bias.fill_(float("nan"))
        with pytest.raises(ValueError, match="INITIAL_WEIGHTS_NONFINITE"):
            train_causal_policy(train, validation, config, initial_policy=initial)
    finally:
        torch.set_num_threads(threads)


# 功能：
#   专家切换保留同一来源的真实传感器历史，但只对所选专家的当前标签生成窗口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_role_transition_retains_prior_specialist_sensor_history():
    rows = samples()[:4]
    rows[-1] = rows[-1].model_copy(update={"navigation_expert_role": "precision-maneuver-policy"})
    examples = causal_examples(rows, stream_groups={"train": "route-a"}, history_length=4,
                               navigation_role="precision-maneuver-policy")
    assert len(examples) == 1
    assert examples[0].source_sha256 == tuple(row.temporal_evidence.sample_sha256 for row in rows)


# 功能：
#   控制历史使用真实来源时间差，超过最大间隔后与窗口一起归零，而不是使用轮询时长。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_real_source_intervals_are_input_features_and_reset_with_history():
    row = samples()[0]
    history = CausalControlHistory(4)
    for index, at in enumerate((1000, 1020, 1100, 1150)):
        source = evidence(index).model_copy(update={"observed_at_unix_ms": at})
        history.append(source, row.state_features, row.realtime_features, row.realtime_valid_mask)
    assert [value[-1] for value in history.values()[0]] == pytest.approx([0., .08, .32, .2])
    source = evidence(5).model_copy(update={"observed_at_unix_ms": 1500})
    history.append(source, row.state_features, row.realtime_features, row.realtime_valid_mask)
    assert not history.ready and history.values()[0][-1][-1] == 0.
