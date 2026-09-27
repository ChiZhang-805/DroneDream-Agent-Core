"""Exercise heading source freezing and receipt compatibility through real package APIs."""

import argparse

import pytest
from test_causal_packaging import base_package  # noqa: F401
from test_complete_artifact_assembly import recipe  # noqa: F401
from test_heading_admission import source_case
from test_local_policy_packages import _admission, qualified_pilot_metrics
from test_precision_heading_lineage import composed_recipe

from dronedream_agent_core.local_policy_packages import (
    local_policy_receipt_supports_control_contract,
)
from dronedream_agent_core.training.admission_inputs import freeze_admission_inputs
from dronedream_agent_core.training.artifact_assembly import assemble_complete_ensemble
from dronedream_agent_core.training.heading_admission import read_heading_admission_inputs


# 功能：
#   为实际组装的合成十专家包准备所有显式冻结参数，数据内容只用于复制与来源校验。
# 输入：
#   package：本测试的实际模型包；path：独立临时目录；heading_path：原始偏航快照文件。
# 输出：
#   arguments：不会读写产品目录的离线评估参数。
def freeze_arguments(package, path, heading_path):
    data = path / "dummy.jsonl"
    data.write_bytes(b"{}\n")
    arguments = argparse.Namespace(
        package=package.root,
        validation_data=data,
        validation_observations=data,
        stream_groups=data,
        heading_observations=heading_path,
        dataset_receipt=data,
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
        output=path / "output.json",
    )
    return arguments


# 功能：
#   验证完整包连同偏航原始快照一起冻结；原文件随后变化不能改变本次评估输入。
# 输入：
#   recipe：合成十专家配方；tmp_path：测试独占目录。
# 输出：
#   None：冻结字节、来源重算及调用方参数不符合断言时测试失败。
def test_complete_heading_admission_freezes_original_snapshot(recipe, tmp_path):  # noqa: F811
    candidate, _ = composed_recipe(recipe, tmp_path)
    package = assemble_complete_ensemble(
        recipe=candidate,
        source_root=tmp_path,
        output_root=tmp_path / "package-with-heading",
    )
    _, sample, source = source_case(tmp_path)
    content = source.read_bytes()
    arguments = freeze_arguments(package, tmp_path, source)
    private = tmp_path / "private"
    private.mkdir()
    frozen = freeze_admission_inputs(arguments, private)
    source.write_bytes(b"{}\n")
    assert frozen.heading_observations.read_bytes() == content
    assert arguments.heading_observations == source
    assert frozen.heading_observations != source
    features = read_heading_admission_inputs(frozen.heading_observations).features_for(sample)
    assert len(features) == 23 and features[8:] == (0.0,) * 15


# 功能：
#   拒绝未声明扩展却传入侧文件、或声明扩展却遗漏侧文件，防止评估偷偷换输入契约。
# 输入：
#   extended：是否使用显式偏航扩展；recipe、tmp_path：独立合成配方及目录。
# 输出：
#   None：两种契约错配均在模型复制与推理前被拒绝。
@pytest.mark.parametrize("extended", [False, True])
def test_heading_sidecar_must_match_manifest(extended, recipe, tmp_path):  # noqa: F811
    candidate = composed_recipe(recipe, tmp_path)[0] if extended else recipe
    package = assemble_complete_ensemble(
        recipe=candidate,
        source_root=tmp_path,
        output_root=tmp_path / "source-package",
    )
    _, _, source = source_case(tmp_path)
    arguments = freeze_arguments(package, tmp_path, None if extended else source)
    private = tmp_path / "private"
    private.mkdir()
    with pytest.raises(ValueError, match="HEADING_ADMISSION_EXPLICIT_SOURCE_REQUIRED"):
        freeze_admission_inputs(arguments, private)
    assert not (private / "package").exists()


# 功能：
#   核对新偏航包不能继承缺少原始快照绑定的旧离线回执，即使其中动作成绩看起来合格。
# 输入：
#   recipe、tmp_path：合成配方及独占目录。
# 输出：
#   None：回执扩展必须显式匹配；通过这里只表示输入契约一致，不授予真实飞行资格。
def test_composed_heading_requires_receipt_source_identity(recipe, tmp_path):  # noqa: F811
    from test_risk_admission_evidence import synthetic_risk_evidence

    from dronedream_agent_core.training.risk_admission_evidence import action_risk_summary

    candidate, _ = composed_recipe(recipe, tmp_path)
    package = assemble_complete_ensemble(
        recipe=candidate,
        source_root=tmp_path,
        output_root=tmp_path / "receipt-package",
    )
    risk_sha = next(a.sha256 for a in package.manifest.artifacts if a.role == 'risk-critic')
    risk = synthetic_risk_evidence(risk_sha)
    metrics = qualified_pilot_metrics()
    metrics = type(metrics).model_validate({**metrics.model_dump(), **action_risk_summary(risk),
                                           'action_risk_evidence': risk.model_dump()})
    receipt = _admission(package, suffix="a", map_sha256=None).model_copy(
        update={
            "realtime_feature_ready_sample_count": 100,
            "pilot_control_target_sample_count": 100,
            "pilot_control_mean_absolute_error": 0.05,
            "navigation_expert_metrics": {
                role: metrics.model_copy(deep=True)
                for role in (
                    "local-navigation-policy",
                    "precision-maneuver-policy",
                    "recovery-policy",
                )
            },
        }
    )
    assert not local_policy_receipt_supports_control_contract(package, receipt)
    bound = receipt.model_copy(update={"heading_observations_sha256": "f" * 64})
    assert local_policy_receipt_supports_control_contract(package, bound)
    assert "heading_observations_sha256" not in receipt.model_dump()
    assert bound.model_dump()["heading_observations_sha256"] == "f" * 64
