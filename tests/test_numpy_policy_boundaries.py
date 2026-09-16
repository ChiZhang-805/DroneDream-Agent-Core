"""Small real NumPy/ONNX checks; synthetic observations never qualify flight."""

from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from test_action_conditioned_risk import _samples as risk_samples
from test_causal_policy import samples

from dronedream_agent_core import local_policy_training as training


# 功能：
#   用小数、越界标签及非法权重复现类别平衡的静默转换问题。
# 输入：
#   targets：应当拒绝的动作类别或其合法对照。
#   weights：与类别对应的样本权重。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "targets,weights",
    [([0.5], [1]), ([12], [1]), ([-1], [1]), ([0], [-1]), ([0], [0]), ([0], [np.inf])],
)
def test_action_balancing_rejects_invalid_labels_and_weights(targets, weights):
    with pytest.raises(ValueError):
        training.action_class_balanced_sample_weights(
            targets, weights, training.LocalPolicyTrainingConfig()
        )


# 功能：
#   风险标签不能把 NaN 当作安全类，也不能接受越界概率和非正权重。
# 输入：
#   targets：风险监督概率。
#   weights：对应样本权重。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "targets,weights", [([np.nan], [1]), ([1.1], [1]), ([-0.1], [1]), ([0], [-1]), ([0], [0])]
)
def test_risk_balancing_rejects_invalid_labels_and_weights(targets, weights):
    with pytest.raises(ValueError):
        training.risk_class_balanced_sample_weights(
            targets, weights, training.LocalPolicyTrainingConfig()
        )


# 功能：
#   实际导出再损坏权重数值、偏置形状或掩码接口，加载器必须拒绝伪装成兼容的模型。
# 输入：
#   tmp_path：独立模型文件目录。
#   role：导航策略或独立风险网络。
#   corruption：本次注入的张量故障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("role", ["policy", "risk"])
@pytest.mark.parametrize("corruption", ["nan", "bias", "mask"])
def test_onnx_loader_checks_weights_and_all_tensor_shapes(tmp_path, role, corruption):
    model = training._new_model(
        training.LocalPolicyTrainingConfig(hidden_feature_count=8),
        realtime_feature_count=training.POLICY_REALTIME_FEATURE_COUNT,
    )
    exporter = (
        training.export_local_policy_onnx
        if role == "policy"
        else training.export_local_risk_critic_onnx
    )
    loader = (
        training.load_local_policy_onnx
        if role == "policy"
        else training.load_local_risk_critic_onnx
    )
    path = tmp_path / "network.onnx"
    exporter(model, path)
    document = onnx.load(path)
    if corruption == "mask":
        tensor = next(item for item in document.graph.input if item.name == "realtime_valid_mask")
        tensor.type.tensor_type.shape.dim[1].dim_value = 1
    else:
        tensor = next(item for item in document.graph.initializer if item.name == "risk_bias")
        value = np.array([np.nan] if corruption == "nan" else [0, 0], dtype=np.float32)
        tensor.CopyFrom(numpy_helper.from_array(value, "risk_bias"))
    onnx.save(document, path)
    with pytest.raises(ValueError):
        loader(path)


# 功能：
#   导出新模型不得覆盖已经存在的目标权重文件。
# 输入：
#   tmp_path：包含既有文件的测试目录。
#   exporter：导航或风险模型导出函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "exporter", [training.export_local_policy_onnx, training.export_local_risk_critic_onnx]
)
def test_export_preserves_existing_file(tmp_path, exporter):
    model = training._new_model(training.LocalPolicyTrainingConfig(hidden_feature_count=8))
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        exporter(model, path)
    assert path.read_bytes() == b"preserved"


# 功能：
#   一次真实的小批次热启动训练不得改写调用方的通用基座参数。
# 输入：
#   无：使用独立生成的合成连续控制样本。
# 输出：
#   None：不返回业务数据。
def test_training_owns_warm_start_parameters():
    config = training.LocalPolicyTrainingConfig(hidden_feature_count=8, epoch_count=1)
    model = training._new_model(
        config,
        realtime_feature_count=training.POLICY_REALTIME_FEATURE_COUNT,
        include_pilot_control=True,
    )
    before = model.input_weight.copy()
    trained, _ = training.train_local_policy(samples(), config, initial_model=model)
    assert np.array_equal(model.input_weight, before)
    assert not np.shares_memory(model.input_weight, trained.input_weight)


# 功能：
#   模型不完整的控制头不能被静默当作未启用，也不能被风险导出绕过。
# 输入：
#   tmp_path：候选文件目录。
# 输出：
#   None：不返回业务数据。
def test_risk_export_rejects_incomplete_control_head(tmp_path):
    model = training._new_model(training.LocalPolicyTrainingConfig(hidden_feature_count=8))
    broken = replace(model, pilot_control_bias=np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError):
        training.export_local_risk_critic_onnx(broken, tmp_path / "broken.onnx")


# 功能：
#   共享样本读取仍接受合法的不带末尾换行符的最后一行。
# 输入：
#   tmp_path：样本测试目录。
# 输出：
#   None：不返回业务数据。
def test_sample_loader_preserves_last_line_without_newline(tmp_path):
    sample = samples()[0]
    path = tmp_path / "samples.jsonl"
    path.write_text(sample.model_dump_json(), encoding="utf-8")
    assert training.load_training_samples(path) == [sample]


# 功能：
#   后一参数梯度损坏或平方溢出时，Adam 不得先提交前一参数和动量。
# 输入：
#   gradient：注入 NaN 或有限但平方溢出的梯度。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("gradient", [np.nan, 1e30])
def test_adam_failure_preserves_whole_step(gradient):
    parameters = [np.ones(2, dtype=np.float32) for _ in range(2)]
    first = [np.zeros_like(value) for value in parameters]
    second = [value.copy() for value in first]
    gradients = [np.ones(2, dtype=np.float32), np.full(2, gradient, dtype=np.float32)]
    with pytest.raises((ValueError, FloatingPointError)):
        training._adam_update(parameters, gradients, first, second, 1, 0.01)
    assert all(np.array_equal(value, np.ones(2)) for value in parameters)
    assert all(not value.any() for value in [*first, *second])


# 功能：
#   实际风险热启动保留调用方参数，失败或后续训练都不能污染共享基座。
# 输入：
#   无：使用明确合成的安全与危险动作标签。
# 输出：
#   None：不返回业务数据。
def test_risk_training_owns_warm_start_parameters():
    config = training.LocalPolicyTrainingConfig(hidden_feature_count=8, epoch_count=1)
    model = training._new_model(
        config,
        realtime_feature_count=training.POLICY_REALTIME_FEATURE_COUNT,
        action_conditioned=True,
    )
    before = model.input_weight.copy()
    trained, _ = training.train_local_risk_critic(risk_samples(), config, initial_model=model)
    assert np.array_equal(model.input_weight, before)
    assert not np.shares_memory(model.risk_weight, trained.risk_weight)


# 功能：
#   未绑定在 ONNX 文件内的外部权重不能通过路径自动加载进入训练器。
# 输入：
#   tmp_path：可写外部权重测试目录。
# 输出：
#   None：不返回业务数据。
def test_onnx_loader_rejects_external_weights(tmp_path):
    model = training._new_model(training.LocalPolicyTrainingConfig(hidden_feature_count=8))
    path = tmp_path / "embedded.onnx"
    training.export_local_policy_onnx(model, path)
    document = onnx.load(path)
    external_path = tmp_path / "external.onnx"
    onnx.save_model(
        document,
        external_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="external.bin",
        size_threshold=0,
    )
    with pytest.raises(ValueError, match="EXTERNAL_WEIGHTS"):
        training.load_local_policy_onnx(external_path)


# 功能：
#   来源在读取后改变时，模型复制及摘要仍使用刚校验过的同一份字节。
# 输入：
#   tmp_path：独立来源及目标目录。
#   monkeypatch：模拟读取完成后来源文件变化。
# 输出：
#   None：不返回业务数据。
def test_graph_copy_binds_validated_bytes(tmp_path, monkeypatch):
    import hashlib

    source, target = tmp_path / "source.onnx", tmp_path / "target.onnx"
    model = training._new_model(training.LocalPolicyTrainingConfig(hidden_feature_count=8))
    training.export_local_policy_onnx(model, source)
    original = source.read_bytes()
    reader = training.read_plugin_file

    # 功能：
    #   在真实有界读取成功后模拟来源变化，不替代 ONNX 图校验。
    # 输入：
    #   path：实际读取路径。
    #   limit：真实读取器的字节上限。
    # 输出：
    #   content：变化前已经读取的字节。
    def changing_source(path, *, limit):
        content = reader(path, limit=limit)
        source.write_bytes(b"changed")
        return content

    monkeypatch.setattr(training, "read_plugin_file", changing_source)
    _, digest = training._copy_embedded_graph(source, target)
    assert target.read_bytes() == original
    assert digest == hashlib.sha256(original).hexdigest()


# 功能：
#   运行过校验的可变样本仍可能被改坏，训练入口必须重新拒绝非正样本权重。
# 输入：
#   无：构造随后被修改的监督实例。
# 输出：
#   None：不返回业务数据。
def test_training_revalidates_mutated_samples():
    rows = samples()
    rows[0] = rows[0].model_copy(update={"sample_weight": -1})
    with pytest.raises(ValueError):
        training.train_local_policy(rows, training.LocalPolicyTrainingConfig(epoch_count=1))


# 功能：
#   实时输入扩展不能反向缩短宽度，只修改元数据会把旧权重重新解释成其他输入。
# 输入：
#   无：使用已包含实时特征的模型。
# 输出：
#   None：不返回业务数据。
def test_realtime_expansion_cannot_shrink_existing_contract():
    model = training._new_model(
        training.LocalPolicyTrainingConfig(hidden_feature_count=8),
        realtime_feature_count=training.REALTIME_CONTROL_FEATURE_COUNT,
    )
    with pytest.raises(ValueError, match="cannot shrink"):
        training.expand_local_policy_realtime_control_inputs(
            model, realtime_feature_count=0, include_pilot_control=False, random_seed=1
        )
