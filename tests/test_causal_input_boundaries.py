"""Synthetic control/history admission tests; these do not qualify a flight model."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_causal_policy import samples
from test_temporal_evidence import evidence

from dronedream_agent_core.causal_control import (
    CONTROL_REALTIME_WIDTH,
    CONTROL_STATE_WIDTH,
    CausalControlHistory,
    control_history_row,
)
from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    causal_examples,
    example_tensors,
    export_causal_policy,
    validate_learning_examples,
)


# 功能：
#   构造四行完整的合成训练窗口，便于逐个破坏来源、标签或历史边界。
# 输入：
#   无。
# 输出：
#   example：带真实格式但非实际飞行证据的测试窗口。
def complete_example():
    example = causal_examples(samples()[:4], stream_groups={"train": "route-a"},
                              history_length=4)[0]
    return example


# 功能：
#   非数值、布尔值和超大数字不能被包装成合法控制历史，拒绝必须是受控校验错误。
# 输入：
#   field：将被破坏的状态、实时特征或掩码行。
#   value：不符合该行契约的单个输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [
    ("state", True), ("state", "1"), ("state", 2**4096),
    ("realtime", False), ("realtime", float("nan")),
    ("mask", True), ("mask", "1"),
])
def test_control_rows_reject_ambiguous_numeric_values(field, value):
    rows = {"state": [0.] * CONTROL_STATE_WIDTH,
            "realtime": [0.] * CONTROL_REALTIME_WIDTH, "mask": [1.] * CONTROL_REALTIME_WIDTH}
    rows[field][0] = value
    with pytest.raises(ValueError):
        control_history_row(**rows)


# 功能：
#   正常 NumPy 实数标量仍可使用，缺测值归零且与独立零掩码一起保存。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_control_row_retains_mask_semantics_and_detached_float_values():
    state = [np.float32(.5)] * CONTROL_STATE_WIDTH
    realtime = [3.] * CONTROL_REALTIME_WIDTH
    mask = [0.] * CONTROL_REALTIME_WIDTH
    row = control_history_row(state, realtime, mask)
    state[0] = 99.
    assert row[0] == .5
    assert all(type(value) is float for value in row)
    assert all(value == 0. for value in row[CONTROL_STATE_WIDTH:])


# 功能：
#   校验来源必须早于 elapsed 运算，错误实例不得污染已接收的有效历史。
# 输入：
#   value：错误来源对象或绕过契约验证的超大时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [None, {}, evidence(1).model_copy(
    update={"observed_at_unix_ms": 2**4096})])
def test_source_validation_precedes_elapsed_arithmetic(value):
    sample = samples()[0]
    history = CausalControlHistory(4)
    history.append(evidence(0), sample.state_features, sample.realtime_features,
                   sample.realtime_valid_mask)
    before = history.values()
    with pytest.raises(ValueError):
        history.append(value, sample.state_features, sample.realtime_features,
                       sample.realtime_valid_mask)
    assert history.values() == before


# 功能：
#   外部配置修改不能悄悄改变已构造网络的时间展开长度和导出身份。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_model_owns_its_validated_configuration():
    config = CausalPolicyConfig(history_length=4, epochs=1)
    model = CausalPilotPolicy(config)
    config.history_length = 8
    assert model.config.history_length == 4


# 功能：
#   配置绕过验证后必须在构造网络之前被拒绝，而不是构造不匹配的网络。
# 输入：
#   value：非法的历史长度或随机种子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [{"history_length": 3}, {"seed": 2**80}])
def test_model_revalidates_mutated_configuration(value):
    config = CausalPolicyConfig().model_copy(update=value)
    with pytest.raises(ValueError):
        CausalPilotPolicy(config)


# 功能：
#   构造训练窗口后原始记录再被修改，不得改写窗口的当前标签或传感器状态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_examples_detach_current_sample_from_caller():
    rows = samples()[:4]
    example = causal_examples(rows, stream_groups={"train": "route-a"}, history_length=4)[0]
    original = example.sample.state_features[0]
    rows[-1].state_features[0] = original + 10.
    assert example.sample.state_features[0] == original


# 功能：
#   已构造样本的非法标签、权重或来源摘要不能进入梯度计算。
# 输入：
#   field：窗口、来源或标签中被破坏的字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["axes", "weight", "digest", "group"])
def test_learning_revalidates_mutable_labels_and_source_metadata(field):
    example = complete_example()
    if field == "axes":
        example.sample.target_pilot_control[0] = float("nan")
    elif field == "weight":
        example = replace(example, sample=example.sample.model_copy(update={"sample_weight": 0.}))
    elif field == "digest":
        example = replace(example, source_sha256=("bad", *example.source_sha256[1:]))
    else:
        example = replace(example, group_id=" route-a ")
    with pytest.raises(ValueError):
        validate_learning_examples([example], 4)


# 功能：
#   历史中的缺测位置不能藏入未遮蔽数值，当前行不能被标为填充数据。
# 输入：
#   field：将被破坏的历史数值或有效行掩码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["sensor_mask", "hidden_sensor", "last_padding"])
def test_tensor_conversion_checks_all_history_rows(field):
    example = complete_example()
    if field == "sensor_mask":
        example.history[0][CONTROL_STATE_WIDTH + CONTROL_REALTIME_WIDTH] = .5
    elif field == "hidden_sensor":
        example.history[0][CONTROL_STATE_WIDTH] = 10.
        example.history[0][CONTROL_STATE_WIDTH + CONTROL_REALTIME_WIDTH] = 0.
    else:
        example.history_mask[-1] = 0.
    with pytest.raises(ValueError):
        example_tensors([example], 0)


# 功能：
#   非有限网络权重不能生成看似有效的 ONNX 发布文件，失败时保留调用者训练模式。
# 输入：
#   tmp_path：隔离的测试导出目录。
# 输出：
#   None：不返回业务数据。
def test_export_rejects_nonfinite_weights_before_publication(tmp_path):
    model = CausalPilotPolicy(CausalPolicyConfig(history_length=4, epochs=1)).train()
    with torch.no_grad():
        model.axes_head.bias.fill_(float("nan"))
    target = tmp_path / "invalid.onnx"
    with pytest.raises(ValueError, match="NONFINITE"):
        export_causal_policy(model, target)
    assert not target.exists() and model.training
