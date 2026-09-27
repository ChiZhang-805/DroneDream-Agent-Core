"""Explicit composed-model provenance; all synthetic artifacts remain in pytest storage."""

import hashlib
import json

import onnx
import pytest
from test_causal_packaging import base_package  # noqa: F401
from test_complete_artifact_assembly import recipe  # noqa: F401
from test_precision_heading_composition import graphs

from dronedream_agent_core.heading_context import HEADING_CONTEXT_SHA256
from dronedream_agent_core.training.artifact_assembly import (
    CompleteEnsembleRecipe,
    assemble_complete_ensemble,
    expert_spatial_groups,
    validate_expert_training_receipt,
)
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from dronedream_agent_core.training.precision_heading_composition import compose_precision_heading
from dronedream_agent_core.training.precision_heading_lineage import (
    make_precision_composition_receipt,
)


# 功能：
#   生成具备可重组来源的合成精细控制配方，不借用真实模型或覆盖产品资源。
# 输入：
#   recipe、tmp_path：十专家测试配方与临时目录。
# 输出：
#   result：新配方与其组合来源记录。
def composed_recipe(source_recipe, tmp_path, role='precision-maneuver-policy'):
    source = next(s for s in source_recipe.sources if s.role == role)
    translation = source.artifact_path.read_bytes()
    training = json.loads(source.training_receipt_path.read_text(encoding="utf-8"))
    _, heading_graph = graphs()
    metadata = {item.key: item.value for item in heading_graph.metadata_props}
    metadata.update(data_sha256="a" * 64, plan_sha256="b" * 64, variant="availability")
    onnx.helper.set_model_props(heading_graph, metadata)
    heading = heading_graph.SerializeToString()
    content, _ = compose_precision_heading(
        translation,
        heading,
        translation_sha256=hashlib.sha256(translation).hexdigest(),
        heading_sha256=hashlib.sha256(heading).hexdigest(),
        role=role,
    )
    lineage = dict(
        purpose="trained-heading-availability-lineage",
        artifact_sha256=hashlib.sha256(heading).hexdigest(),
        feature_contract_sha256=HEADING_CONTEXT_SHA256,
        data_sha256="a" * 64,
        availability_plan_sha256="b" * 64,
        training_plan_sha256="c" * 64,
        trained_checkpoint_sha256=dict(context="d" * 64, circular="e" * 64),
        split_contract=SPATIAL_SPLIT_CONTRACT,
        training_groups=training["metrics"]["training_groups"],
        validation_groups=training["metrics"]["validation_groups"],
        new_training=False,
        qualified_for_flight=False,
    )
    receipt = make_precision_composition_receipt(translation, heading, training, lineage, role=role)
    path, proof = tmp_path / "composed.onnx", tmp_path / "composed.json"
    path.write_bytes(content)
    proof.write_text(json.dumps(receipt), encoding="utf-8")
    payload = source_recipe.model_dump(mode="python")
    field = ('navigation_heading_context_sha256' if role == 'local-navigation-policy'
             else 'precision_heading_context_sha256')
    payload["manifest"][field] = HEADING_CONTEXT_SHA256
    artifact = next(a for a in payload["manifest"]["artifacts"] if a["role"] == source.role)
    artifact.update(
        sha256=hashlib.sha256(content).hexdigest(),
        input_names=[*artifact["input_names"], "heading_context"],
    )
    new_source = next(s for s in payload["sources"] if s["role"] == source.role)
    new_source.update(
        artifact_path=path,
        training_receipt_path=proof,
        training_receipt_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
    )
    result = CompleteEnsembleRecipe.model_validate(payload), receipt
    return result


# 功能：
#   验证新图可经正式组装入口逐分支核验、执行十专家接口探针，并明确保持未获飞行资格。
# 输入：
#   recipe、tmp_path：合成配方与临时目录。
# 输出：
#   None：组装身份、保留来源或资格声明错误时测试失败。
@pytest.mark.parametrize('role', ['precision-maneuver-policy', 'local-navigation-policy'])
def test_composition_assembles_through_real_entry(recipe, tmp_path, role):  # noqa: F811
    candidate, receipt = composed_recipe(recipe, tmp_path, role)
    package = assemble_complete_ensemble(
        recipe=candidate, source_root=tmp_path, output_root=tmp_path / "published"
    )
    saved = json.loads(
        (package.root / f"training-evidence/{role}.json").read_text()
    )
    assert saved == receipt
    assert (
        json.loads((package.root / "assembly-receipt.json").read_text())["qualified_for_flight"]
        is False
    )
    assert expert_spatial_groups(role, receipt) == expert_spatial_groups(
        role, receipt["translation_training_receipt"]
    )


# 功能：同时声明两个偏航分支时，各自来源还原不得移除另一角色的输入契约。
# 输入：recipe、tmp_path：隔离合成配方和目录。输出：两个角色均可经真实组装入口核验。
def test_both_heading_roles_preserve_each_others_contract(recipe, tmp_path):  # noqa: F811
    first = tmp_path / 'precision'
    second = tmp_path / 'navigation'
    first.mkdir()
    second.mkdir()
    precision, _ = composed_recipe(recipe, first, 'precision-maneuver-policy')
    both, _ = composed_recipe(precision, second, 'local-navigation-policy')
    package = assemble_complete_ensemble(recipe=both, source_root=tmp_path,
                                        output_root=tmp_path / 'both-published')
    assert package.manifest.requires_heading_evidence()
    for role in ('local-navigation-policy', 'precision-maneuver-policy'):
        assert package.manifest.heading_context_for_role(role) == HEADING_CONTEXT_SHA256
    assert package.manifest.heading_context_for_role('recovery-policy') is None


# 功能：
#   逐项破坏来源、原图、分工或留出身份，确保不能用旧成绩、假摘要或跨分支泄漏放行新图。
# 输入：
#   damage：损坏方式；recipe、tmp_path：合成输入与临时目录。
# 输出：
#   None：所有破坏在组装前被拒绝。
@pytest.mark.parametrize(
    "damage",
    [
        "original-receipt",
        "source",
        "ownership",
        "old-metrics",
        "heading-data",
        "heading-plan",
        "checkpoint",
        "leakage",
        "qualification",
        "wrong-role",
    ],
)
@pytest.mark.parametrize('role', ['precision-maneuver-policy', 'local-navigation-policy'])
def test_composition_rejects_invalid_lineage(damage, recipe, tmp_path, role):  # noqa: F811
    candidate, receipt = composed_recipe(recipe, tmp_path, role)
    digest = next(
        a.sha256 for a in candidate.manifest.artifacts if a.role == role
    )
    if damage == "original-receipt":
        receipt = receipt["translation_training_receipt"]
    elif damage == "source":
        receipt["heading_graph_base64"] = "YWJj"
    elif damage == "ownership":
        receipt["composition"]["axis_ownership"][0] = "heading"
    elif damage == "old-metrics":
        receipt["translation_training_receipt"]["artifact_sha256"] = digest
    elif damage == "heading-data":
        receipt["heading_training_lineage"]["data_sha256"] = "f" * 64
    elif damage == "heading-plan":
        receipt["heading_training_lineage"]["availability_plan_sha256"] = "f" * 64
    elif damage == "checkpoint":
        receipt["heading_training_lineage"]["trained_checkpoint_sha256"].pop("circular")
    elif damage == "leakage":
        receipt["heading_training_lineage"]["training_groups"] = receipt[
            "heading_training_lineage"
        ]["validation_groups"]
    elif damage == 'wrong-role':
        receipt['purpose'] = ('frozen-precision-heading-composition' if role == 'local-navigation-policy'
                              else 'frozen-navigation-heading-composition')
    else:
        receipt["qualified_for_flight"] = True
    with pytest.raises((ValueError, onnx.checker.ValidationError)):
        validate_expert_training_receipt(
            role, digest, receipt, candidate.manifest
        )
