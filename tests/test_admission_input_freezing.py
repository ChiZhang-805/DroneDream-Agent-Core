"""Input identity and current/legacy dataset separation; no flight qualification."""

import argparse

import pytest
from test_local_policy_runtime_staging import _package
from test_mission_groups import split_fixture

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.plugin_files import hash_plugin_file
from dronedream_agent_core.training import admission_inputs


# 功能：
#   创建模型文件及显式输入参数，只测试字节冻结，不把占位数据当成可准入样本。
# 输入：
#   tmp_path：本测试来源和私有输出根。
# 输出：
#   args：完整的离线评估参数对象。
@pytest.fixture
def arguments(tmp_path):
    package = _package(tmp_path / "package")
    data, receipt = tmp_path / "data.jsonl", tmp_path / "dataset.json"
    data.write_bytes(b"original data")
    receipt.write_bytes(b"{}")
    args = argparse.Namespace(
        package=package.root,
        validation_data=data,
        dataset_receipt=receipt,
        visual_encoding_receipt=None,
        composition_receipt=None,
        augmentation_receipt=None,
        rebinding_receipt=None,
        training_receipt=None,
        navigation_training_dataset_receipt=None,
        inherited_admission_receipt=None,
        advisor_validation_data=[],
        advisor_training_receipt=[],
        advisor_dataset_receipt=[],
        output=tmp_path / "admission.json",
    )
    return args


# 功能：
#   冻结后的模型和数据不受原始文件替换影响，且不修改调用方参数对象。
# 输入：
#   arguments：真实文件路径与原始参数。
#   tmp_path：私有快照位置。
# 输出：
#   None：不返回业务数据。
def test_frozen_admission_sources_do_not_follow_later_changes(arguments, tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    original = dict(vars(arguments))
    frozen = admission_inputs.freeze_admission_inputs(arguments, private)
    arguments.validation_data.write_bytes(b"changed data")
    arguments.dataset_receipt.write_bytes(b"changed receipt")
    model = arguments.package / "models/local-navigation-policy.onnx"
    model.write_bytes(b"changed model")
    assert vars(arguments) == original
    assert frozen.validation_data.read_bytes() == b"original data"
    assert frozen.dataset_receipt.read_bytes() == b"{}"
    assert (
        frozen.package / "models/local-navigation-policy.onnx"
    ).read_bytes() != model.read_bytes()
    assert frozen.output == arguments.output


# 功能：
#   复制边界源模型已发生变化时必须拒绝，不能创建带原摘要身份的新快照。
# 输入：
#   arguments：真实测试模型包及数据。
#   tmp_path：私有快照目录。
#   monkeypatch：模拟加载后模型变化的工具。
# 输出：
#   None：不返回业务数据。
def test_freezing_rejects_model_changed_after_load(arguments, tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()

    # 功能：
    #   在第一次模型复制前替换测试权重，再执行实际有界散列复制。
    # 输入：
    #   source：当前测试模型源。
    #   limit：允许读取字节数。
    #   destination：私有新目标。
    # 输出：
    #   digest：实际改写字节的摘要。
    def changed_copy(source, *, limit, destination=None):
        source.write_bytes(b"model changed during evaluation startup")
        digest = hash_plugin_file(source, limit=limit, destination=destination)
        return digest

    monkeypatch.setattr(admission_inputs, "hash_plugin_file", changed_copy)
    with pytest.raises(ValueError, match="MODEL_CHANGED"):
        admission_inputs.freeze_admission_inputs(arguments, private)


# 功能：
#   冻结完整风险专用数据集，原文件变化不会改变后续准入读取的样本和来源身份。
# 输入：
#   arguments：基础输入夹具；tmp_path：独占输出；monkeypatch：只设置数据构建命令行。
# 输出：
#   None：冻结遗漏文件、引用原路径或身份变化时失败。
def test_freezing_copies_all_risk_sources(arguments, tmp_path, monkeypatch):
    from test_native_action_risk_artifacts import dataset

    from dronedream_agent_core.training.action_risk_artifacts import FILES, load_action_risk_dataset

    native = tmp_path / 'native'
    native.mkdir()
    source = dataset(native, monkeypatch)
    original = load_action_risk_dataset(source)
    arguments.risk_validation_data = [source]
    private = tmp_path / 'private'
    private.mkdir()
    frozen = admission_inputs.freeze_admission_inputs(arguments, private)
    copied = frozen.risk_validation_data[0]
    assert copied.is_relative_to(private) and copied != source
    for name in ('dataset-receipt.jsonl', *FILES.values()):
        assert (copied / name).read_bytes() == (source / name).read_bytes()
    (source / 'dataset-receipt.jsonl').write_bytes(b'{}')
    loaded = load_action_risk_dataset(copied)
    assert loaded.receipt_sha256 == original.receipt_sha256
    assert loaded.samples == original.samples


# 功能：
#   构造由当前数据生产者输出的空间分区结构，路线分组由实际几何重新计算。
# 输入：
#   validation_y：验证路线偏移，可模拟独立路线或与训练重叠。
# 输出：
#   receipt：当前因果数据集的合成回执。
def spatial_receipt(validation_y=2.0):
    sources = {}
    for name, y, digest in (("training", 0.0, "a"), ("validation", validation_y, "b")):
        split = split_fixture(y=y)
        sources[name] = {
            "sha256": digest * 64,
            "sources": [
                {
                    "mission_group": split.group_sha256,
                    "mission_split": split.model_dump(),
                    "mission_evidence_sha256": digest * 64,
                }
            ],
        }
    receipt = {
        "label_source": "verified-executed-simulation-teacher-control",
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "split_method": "map-spatial-route",
        "qualified_for_flight": False,
        "outputs": sources,
    }
    return receipt


# 功能：
#   当前数据回执可转换为统一读取结构，但不能把旧任务划分或训练过的路线当成新空间留出。
# 输入：
#   无：使用可重算的独立路线及历史组。
# 输出：
#   None：不返回业务数据。
def test_current_spatial_dataset_contract_is_explicit_and_independent():
    receipt = spatial_receipt()
    history = {split_fixture(y=8.0).group_sha256}
    result = admission_inputs.navigation_dataset_contract(
        receipt, causal=True, historical_groups=history
    )
    assert result["validation_output_sha256"] == "b" * 64
    assert "outputs" in receipt and "validation_output_sha256" not in receipt
    with pytest.raises(ValueError, match="spatial partitions overlap"):
        admission_inputs.navigation_dataset_contract(
            spatial_receipt(0.0), causal=True, historical_groups=history
        )
    with pytest.raises(ValueError, match="training or tuning"):
        admission_inputs.navigation_dataset_contract(
            receipt, causal=True, historical_groups={split_fixture(y=2.0).group_sha256}
        )
    with pytest.raises(ValueError, match="current executed"):
        admission_inputs.navigation_dataset_contract(
            {"split_method": "held-out-source-campaign"}, causal=True, historical_groups=history
        )


# 功能：
#   篡改空间来源摘要后，即使回执仍标记正确分区，也必须被实际路线重算发现。
# 输入：
#   无：使用已知路线和被替换的分组值。
# 输出：
#   None：不返回业务数据。
def test_current_spatial_receipt_revalidates_route_evidence():
    receipt = spatial_receipt()
    receipt["outputs"]["validation"]["sources"][0]["mission_split"]["group_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        admission_inputs.navigation_dataset_contract(
            receipt, causal=True, historical_groups={"d" * 64}
        )
