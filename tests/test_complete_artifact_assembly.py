"""Synthetic ONNX fixtures verify assembly, never aircraft/model competence."""

import hashlib
import json
import sys

import onnx
import pytest
from test_causal_packaging import base_package, replacements  # noqa: F401
from test_local_policy_composition import _write_advisor_and_receipt

from dronedream_agent_core.causal_control import CONTROL_HISTORY_CONTRACT_SHA256
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_expert_harness import NAVIGATION_EXPERT_ROLES
from dronedream_agent_core.local_policy_composition import (
    compose_local_policy_advisors,
    load_local_advisor_artifact_evidence,
)
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_vision_training import LOCAL_VISION_ARCHITECTURE
from dronedream_agent_core.training.artifact_assembly import (
    CompleteEnsembleRecipe,
    assemble_complete_ensemble,
    read_bound_content,
    validate_embedded_graph,
    validate_expert_training_receipt,
)
from dronedream_agent_core.training.causal_replay import REPLAY_FILES
from dronedream_agent_core.training.mission_groups import SPATIAL_SPLIT_CONTRACT
from dronedream_agent_core.training.visual_lineage import VisualInputContract


# 功能：
#   提供十角色合成组装配方，将夹具生命周期交由 pytest 管理。
# 输入：
#   base_package：合成模型基座目录。
#   tmp_path：测试独立目录。
# 输出：
#   value：完整的测试组装配方。
@pytest.fixture
def recipe(base_package, tmp_path):  # noqa: F811 - imported pytest fixture
    value = build_recipe(base_package, tmp_path)
    return value


# 功能：
#   拒绝把缺少机动与速度历史的旧负载接口放进当前完整飞行配方，即使一般读取器能识别旧包。
# 输入：
#   recipe：当前十专家合成配方。
# 输出：
#   None：通过独立的完整配方门槛错误断言给出结果。
def test_complete_recipe_rejects_payload_without_observable_motion(recipe):
    changed = recipe.model_dump(mode='python')
    payload = next(a for a in changed['manifest']['artifacts'] if a['role'] == 'payload-dynamics-adapter')
    payload['input_names'] = ['payload_features', 'payload_history', 'history_mask']
    with pytest.raises(ValueError, match='ENSEMBLE_PAYLOAD_REQUIRES_CURRENT_MOTION_INPUT'):
        CompleteEnsembleRecipe.model_validate(changed)


# 功能：
#   经真实命令行入口组装合成十专家包，并核对输出摘要和不授予飞行资格的声明。
# 输入：
#   recipe：仅用于接口测试的合成模型配方。
#   tmp_path：独立输入与输出目录。
#   monkeypatch：设置本次测试的命令行参数。
#   capsys：读取实际命令行 JSON 输出。
# 输出：
#   None：不返回业务数据。
def test_cli_assembles_bound_recipe(recipe, tmp_path, monkeypatch, capsys):
    from scripts.assemble_complete_control_ensemble import main

    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(recipe.model_dump_json(), encoding="utf-8")
    output = tmp_path / "cli-assembled"
    monkeypatch.setattr(sys, "argv", ["assemble", "--recipe", str(recipe_path),
                                    "--output", str(output)])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["expert_count"] == 10 and result["qualified_for_flight"] is False
    assert load_local_policy_package(output).package_sha256 == result["package_sha256"]


# 功能：
#   拒绝包含重复顶层键的完整配方，避免后一个值静默覆盖前一个来源声明。
# 输入：
#   recipe：语义有效的合成配方。
#   tmp_path：独立文件目录。
#   monkeypatch：本次测试的命令行参数。
# 输出：
#   None：不返回业务数据。
def test_cli_rejects_duplicate_recipe_keys_before_output(recipe, tmp_path, monkeypatch):
    from scripts.assemble_complete_control_ensemble import main

    recipe_path = tmp_path / "duplicate-recipe.json"
    recipe_path.write_text('{"sources":[],' + recipe.model_dump_json()[1:], encoding="utf-8")
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["assemble", "--recipe", str(recipe_path),
                                    "--output", str(output)])
    with pytest.raises(ValueError, match="DUPLICATE"):
        main()
    assert not output.exists()


# 功能：
#   验证超预算配方在构造任何候选模型包之前拒绝。
# 输入：
#   tmp_path：独立文件目录。
#   monkeypatch：本次测试的命令行参数。
# 输出：
#   None：不返回业务数据。
def test_cli_bounds_recipe_bytes_before_output(tmp_path, monkeypatch):
    from scripts.assemble_complete_control_ensemble import main

    recipe_path = tmp_path / "oversized-recipe.json"
    recipe_path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["assemble", "--recipe", str(recipe_path),
                                    "--output", str(output)])
    with pytest.raises(ValueError):
        main()
    assert not output.exists()


# 功能：
#   构造十角色权重和各自格式的来源回执，只用于合成测试，不生成真实训练资格。
# 输入：
#   base_root：提供顾问和视觉接口的合成基座目录。
#   tmp_path：保存合成操纵模型与回执的测试目录。
# 输出：
#   value：通过配方结构验证的完整测试配方。
def build_recipe(base_root, tmp_path):
    from dronedream_agent_core.local_advisor_training import (
        LocalAdvisorTrainingConfig,
        _new_model,
        export_local_advisor_onnx,
    )

    base = load_local_policy_package(base_root)
    paths = {**base.artifact_paths, **replacements(tmp_path)}
    # 通用旧包夹具仍用于兼容性读取测试；完整新配方必须使用真实导出的当前负载接口。
    payload_path = tmp_path / "current-motion-payload.onnx"
    export_local_advisor_onnx(_new_model("payload-dynamics-adapter",
        LocalAdvisorTrainingConfig(hidden_feature_count=8)), payload_path)
    paths["payload-dynamics-adapter"] = payload_path
    manifest = base.manifest.model_dump(mode="json")
    manifest.update(
        package_id="test.complete-current",
        visual_normalization="imagenet",
        navigation_architecture="causal-gru-control",
        navigation_history_length=4,
        navigation_history_contract_sha256=CONTROL_HISTORY_CONTRACT_SHA256,
    )
    sources = []
    for entry in manifest["artifacts"]:
        role = entry["role"]
        path = paths[role]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        graph = onnx.load(str(path))
        entry.update(
            relative_path=f"{role}.onnx",
            sha256=digest,
            input_names=[v.name for v in graph.graph.input],
            output_names=[v.name for v in graph.graph.output],
        )
        common = {"training_data_sha256": "c" * 64, "validation_data_sha256": "d" * 64}
        if role in NAVIGATION_EXPERT_ROLES:
            record = dict(
                artifact_sha256=digest,
                expert_role=role,
                architecture="causal-gru-control",
                split_contract=SPATIAL_SPLIT_CONTRACT,
                replay_artifact_sha256={name: "c" * 64 for name in REPLAY_FILES},
                qualified_for_flight=False,
                feature_contract_sha256=manifest["control_feature_contract_sha256"],
                config={"history_length": 4, "visual_feature_count": 1},
                metrics={
                    "training_window_count": 40,
                    "validation_window_count": 20,
                    "training_groups": ["a" * 64],
                    "validation_groups": ["b" * 64],
                },
                input_sha256={"train": "c" * 64, "validation": "d" * 64},
            )
            visual = VisualInputContract.from_manifest(
                base.manifest, hashlib.sha256(paths["perception-encoder"].read_bytes()).hexdigest()
            )
            visual.visual_normalization = "imagenet"
            record["visual_input_contract"] = visual.model_dump()
        elif role == "risk-critic":
            record = dict(
                purpose="native-action-risk-offline-training",
                split_contract=SPATIAL_SPLIT_CONTRACT,
                training_groups=["a" * 64], validation_groups=["b" * 64],
                model_sha256=digest,
                optimized=True,
                offline_validation_passed=True,
                qualified_for_flight=False,
                independent_physical_validation_required=True,
                teacher_config={"test_only": True},
                teacher_config_sha256=sha256_json({"test_only": True}),
                control_feature_contract_sha256=manifest["control_feature_contract_sha256"],
                dataset_receipts_sha256={"training": ["c" * 64], "validation": ["d" * 64]},
            )
        elif role == "perception-encoder":
            record = dict(
                common,
                artifact_sha256=digest,
                training_accepted=True,
                flight_qualification_granted=False,
                visual_feature_count=1,
                architecture=LOCAL_VISION_ARCHITECTURE,
                embedding_supervision="traversability-scene-quality-through-embedding",
                backbone_initialization="lraspp-mobilenet-v3-large-coco-voc-v1",
                initialization={"source": "lraspp-mobilenet-v3-large-coco-voc-v1",
                                "tensor_sha256": "e" * 64},
                split_method="held-out-complete-flight-and-render-spatial-group",
                spatial_group_disjoint=True,
                config={"width": 32, "height": 32, "pretrained_source": "coco-voc-segmentation"},
            )
        else:
            record = dict(
                common,
                schema_version="dronedream.local-advisor-training-receipt.v1",
                dataset_split_method=SPATIAL_SPLIT_CONTRACT,
                feature_contract_sha256=manifest["control_feature_contract_sha256"],
                training_groups=["a" * 64], validation_groups=["b" * 64],
                training_accepted=True,
                qualification_granted=False,
                fresh_admission_validation_required=True,
                advisors={
                    role: {"artifact": path.name, "artifact_sha256": digest, "accepted": True}
                },
            )
        receipt = tmp_path / f"{role}-training.json"
        receipt.write_text(json.dumps(record), encoding="utf-8")
        sources.append(
            dict(
                role=role,
                artifact_path=str(path),
                training_receipt_path=str(receipt),
                training_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            )
        )
    value = CompleteEnsembleRecipe.model_validate(dict(manifest=manifest, sources=sources))
    return value


# 功能：
#   验证十角色可由明确来源直接组装，所有证据绑定新包且不能覆盖已有输出目录。
# 输入：
#   recipe：完整合成配方。
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_complete_current_package_requires_no_legacy_base(recipe, tmp_path):
    output = tmp_path / "assembled"
    package = assemble_complete_ensemble(recipe=recipe, source_root=tmp_path, output_root=output)
    assert len(package.artifact_paths) == 10
    assert package.manifest.navigation_architecture == "causal-gru-control"
    assert (
        package.manifest.control_feature_contract_sha256
        == recipe.manifest.control_feature_contract_sha256
    )
    receipt = json.loads((output / "assembly-receipt.json").read_text())
    assert receipt["package_sha256"] == package.package_sha256
    assert not receipt["inherited_admission"] and not receipt["qualification_granted"]
    for role, entry in receipt["expert_evidence"].items():
        assert (
            hashlib.sha256((output / entry["training_receipt_path"]).read_bytes()).hexdigest()
            == entry["training_receipt_sha256"]
        )
        assert (
            hashlib.sha256(package.artifact_paths[role].read_bytes()).hexdigest()
            == entry["artifact_sha256"]
        )
    with pytest.raises(FileExistsError):
        assemble_complete_ensemble(recipe=recipe, source_root=tmp_path, output_root=output)
    assert not list(tmp_path.glob(".complete-ensemble-*"))


# 功能：
#   当前视觉生产契约可组装，但旧分割版本、旁路嵌入、缺少初始化身份或混用划分不得进入。
# 输入：
#   recipe、field、value：当前十专家夹具及要注入的不兼容回执字段。
# 输出：
#   None：每种契约漂移均应在加载任何运行端会话前被拒绝。
@pytest.mark.parametrize("field,value", [
    ("architecture", "mobilenet-v3-large-lraspp-v1"),
    ("embedding_supervision", "bypass-embedding"),
    ("split_method", "held-out-complete-flight"),
    ("spatial_group_disjoint", False),
    ("backbone_initialization", "uninitialized-development-only"),
    ("initialization", {"source": "lraspp-mobilenet-v3-large-coco-voc-v1"}),
])
def test_visual_assembly_rejects_obsolete_or_unbound_receipts(recipe, field, value):
    source = next(item for item in recipe.sources if item.role == "perception-encoder")
    receipt = json.loads(source.training_receipt_path.read_bytes())
    artifact = next(item for item in recipe.manifest.artifacts if item.role == source.role)
    receipt[field] = value
    with pytest.raises(ValueError, match="VISUAL_TRAINING_IDENTITY_MISMATCH"):
        validate_expert_training_receipt(source.role, artifact.sha256, receipt, recipe.manifest)


# 功能：
#   即使训练回执声称使用新网络，也不能给旧架构或旧输出模式的清单背书。
# 输入：
#   recipe：含真实 ONNX 格式但无飞行资格的合成测试配方。
#   field：要替换成旧协议值的清单字段。
#   value：待拒绝的旧协议值。
# 输出：
#   None：断言失败时测试报错。
@pytest.mark.parametrize("field,value", [("navigation_architecture", "feedforward"),
                                        ("pilot_control_mode", None)])
def test_training_receipt_cannot_attest_a_different_control_architecture(recipe, field, value):
    source = next(item for item in recipe.sources if item.role == "local-navigation-policy")
    receipt = json.loads(source.training_receipt_path.read_bytes())
    artifact = next(item for item in recipe.manifest.artifacts if item.role == source.role)
    # 模拟绕开配方加载器的调用者；独立的回执核验边界仍须拒绝清单错配。
    manifest = recipe.manifest.model_copy(update={field: value})
    with pytest.raises(ValueError, match="NAVIGATION_TRAINING_IDENTITY_MISMATCH"):
        validate_expert_training_receipt(source.role, artifact.sha256, receipt, manifest)


# 功能：
#   验证重复角色、旧特征契约及占用清单路径的模型文件不能形成有效配方。
# 输入：
#   recipe：完整合成配方。
#   mutation：需要注入的配方错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["missing-role", "old-contract", "path-alias"])
def test_incomplete_or_legacy_recipes_rejected(recipe, mutation):
    raw = recipe.model_dump(mode="json")
    if mutation == "missing-role":
        raw["sources"][-1] = raw["sources"][0]
    elif mutation == "old-contract":
        raw["manifest"]["control_feature_contract_sha256"] = "0" * 64
    else:
        raw["manifest"]["artifacts"][-1]["relative_path"] = "manifest.json"
    with pytest.raises(ValueError):
        CompleteEnsembleRecipe.model_validate(raw)


# 功能：
#   验证损坏权重、错误回执、角色或划分不符、失败风险训练和视觉归一化不符均拒绝发布。
# 输入：
#   recipe：完整合成配方。
#   tmp_path：测试独立目录。
#   failure：需要注入的来源错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "failure", ["weight", "receipt", "wrong-role", "overlap", "risk-failed", "normalization"]
)
def test_bad_sources_never_leave_a_partial_package(recipe, tmp_path, failure):
    role = "risk-critic" if failure == "risk-failed" else "local-navigation-policy"
    source = next(s for s in recipe.sources if s.role == role)
    if failure == "weight":
        source.artifact_path.write_bytes(b"changed weights")
    elif failure == "receipt":
        source.training_receipt_path.write_text("{}")
    elif failure == "normalization":
        recipe.manifest.visual_normalization = "zero-to-one"
    else:
        data = json.loads(source.training_receipt_path.read_text())
        if failure == "wrong-role":
            data["expert_role"] = "recovery-policy"
        elif failure == "overlap":
            data["metrics"]["validation_groups"] = data["metrics"]["training_groups"]
        else:
            data["offline_validation_passed"] = False
        source.training_receipt_path.write_text(json.dumps(data))
        source.training_receipt_sha256 = hashlib.sha256(
            source.training_receipt_path.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError):
        assemble_complete_ensemble(
            recipe=recipe, source_root=tmp_path, output_root=tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".complete-ensemble-*"))


# 功能：
#   验证每个专家内部训练验证不重叠仍不足够，其他专家的留出空间也不能参与训练。
# 输入：
#   recipe：完整合成配方。
#   tmp_path：测试独立目录。
#   role：要注入跨专家泄漏的角色。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("role", ["local-navigation-policy", "risk-critic",
                                  "payload-dynamics-adapter"])
def test_no_expert_can_train_on_another_experts_holdout(recipe, tmp_path, role):
    source = next(s for s in recipe.sources if s.role == role)
    receipt = json.loads(source.training_receipt_path.read_bytes())
    metrics = receipt["metrics"] if role == "local-navigation-policy" else receipt
    metrics["training_groups"] = ["b" * 64]
    metrics["validation_groups"] = ["c" * 64]  # Locally disjoint, globally contaminated.
    source.training_receipt_path.write_text(json.dumps(receipt))
    source.training_receipt_sha256 = hashlib.sha256(
        source.training_receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="CROSS_EXPERT_TRAINING_VALIDATION_LEAKAGE"):
        assemble_complete_ensemble(recipe=recipe, source_root=tmp_path,
                                   output_root=tmp_path / "contaminated")
    assert not (tmp_path / "contaminated").exists()


# 功能：
#   验证缺少当前空间划分契约的历史回执不能被新包继续采用。
# 输入：
#   recipe：完整合成配方。
#   tmp_path：测试独立目录。
#   role：需要移除划分契约的专家角色。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("role", ["local-navigation-policy", "risk-critic",
                                  "payload-dynamics-adapter"])
def test_current_assembly_cannot_reuse_old_split_receipts(recipe, tmp_path, role):
    source = next(s for s in recipe.sources if s.role == role)
    receipt = json.loads(source.training_receipt_path.read_bytes())
    receipt.pop("dataset_split_method" if "advisors" in receipt else "split_contract")
    source.training_receipt_path.write_text(json.dumps(receipt))
    source.training_receipt_sha256 = hashlib.sha256(
        source.training_receipt_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="CURRENT_SPATIAL_SPLIT_CONTRACT"):
        assemble_complete_ensemble(recipe=recipe, source_root=tmp_path,
                                   output_root=tmp_path / "old-split")


# 功能：
#   验证按名字绑定的图允许输入输出顺序变化，但拒绝缺项、重名和错误名称。
# 输入：
#   无：测试自行构造含两个输入输出的 ONNX 图。
# 输出：
#   None：不返回业务数据。
def test_named_graph_bindings_allow_permutation_but_reject_missing_or_duplicate_names():
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Add", ["first", "second"], ["sum"]),
            onnx.helper.make_node("Identity", ["first"], ["echo"]),
        ],
        "named-bindings",
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])
            for name in ("first", "second")
        ],
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1])
            for name in ("sum", "echo")
        ],
    )
    data = onnx.helper.make_model(graph).SerializeToString()
    validate_embedded_graph(data, input_names=["second", "first"], output_names=["echo", "sum"])
    for inputs, outputs in (
        (["first"], ["sum", "echo"]),
        (["first", "first", "second"], ["sum", "echo"]),
        (["first", "second"], ["sum"]),
        (["first", "second"], ["sum", "echo", "echo"]),
        (["first", "other"], ["sum", "echo"]),
    ):
        with pytest.raises(ValueError, match="TENSOR_NAMES"):
            validate_embedded_graph(data, input_names=inputs, output_names=outputs)


# 功能：
#   验证嵌在 Constant 属性中的外部权重引用也会在运行前拒绝。
# 输入：
#   无：测试自行构造带外部张量引用的 ONNX 图。
# 输出：
#   None：不返回业务数据。
def test_external_weights_in_nested_constant_rejected_before_runtime():
    tensor = onnx.helper.make_tensor("nested", onnx.TensorProto.FLOAT, [1], [1.0])
    tensor.ClearField("float_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="not-a-bound-artifact")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["out"], value=tensor)],
        "nested-test",
        [],
        [onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(graph)
    with pytest.raises(ValueError, match="EXTERNAL_WEIGHTS"):
        validate_embedded_graph(model.SerializeToString(), input_names=[], output_names=["out"])


# 功能：
#   验证仅替换顾问时保持操纵模型权重、历史与传感器契约，并重建十角色证据链。
# 输入：
#   recipe：完整合成配方。
#   tmp_path：测试独立目录。
# 输出：
#   None：不返回业务数据。
def test_advisor_replacement_preserves_causal_weights_and_all_contracts(recipe, tmp_path):
    base = assemble_complete_ensemble(
        recipe=recipe, source_root=tmp_path, output_root=tmp_path / "base-current"
    )
    advisor_dir = tmp_path / "new-advisor"
    advisor_dir.mkdir()
    artifact, receipt = _write_advisor_and_receipt(advisor_dir)
    evidence = load_local_advisor_artifact_evidence(
        role="state-anomaly-detector", artifact_path=artifact, training_receipt_path=receipt
    )
    updated = compose_local_policy_advisors(
        base_package_path=base.manifest_path.parent,
        output_package_path=tmp_path / "composed",
        composition_receipt_path=tmp_path / "composition.json",
        package_id="test.changed-advisor",
        display_name="Synthetic replacement",
        additions=(evidence,),
    )
    for role, path in base.artifact_paths.items():
        if role != evidence.role:
            assert path.read_bytes() == updated.artifact_paths[role].read_bytes()
    assert (
        updated.manifest.navigation_history_contract_sha256
        == base.manifest.navigation_history_contract_sha256
    )
    assert (
        updated.manifest.control_feature_contract_sha256
        == base.manifest.control_feature_contract_sha256
    )
    assert updated.manifest.navigation_architecture == "causal-gru-control"
    lineage = json.loads((updated.manifest_path.parent / "assembly-receipt.json").read_bytes())
    assert lineage["package_sha256"] == updated.package_sha256
    assert len(lineage["expert_evidence"]) == 10
    for entry in lineage["expert_evidence"].values():
        content = (updated.manifest_path.parent / entry["training_receipt_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["training_receipt_sha256"]


# 功能：
#   验证顾问组合不能绕过完整来源、当前划分和跨专家独立性要求，失败不发布结果。
# 输入：
#   recipe：完整合成配方。
#   tmp_path：测试独立目录。
#   failure：需要破坏的来源条件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure", ["missing-base-lineage", "old-advisor", "cross-expert"])
def test_replacement_cannot_bypass_complete_package_provenance(recipe, tmp_path, failure):
    base = assemble_complete_ensemble(recipe=recipe, source_root=tmp_path,
                                      output_root=tmp_path / "base-current")
    advisor_dir = tmp_path / "replacement"
    advisor_dir.mkdir()
    artifact, receipt = _write_advisor_and_receipt(advisor_dir)
    if failure == "missing-base-lineage":
        (base.manifest_path.parent / "assembly-receipt.json").unlink()
    else:
        data = json.loads(receipt.read_bytes())
        if failure == "old-advisor":
            data.pop("dataset_split_method")
        else:
            data["training_groups"], data["validation_groups"] = ["b" * 64], ["c" * 64]
        receipt.write_text(json.dumps(data))
    selected = load_local_advisor_artifact_evidence(role="state-anomaly-detector",
        artifact_path=artifact, training_receipt_path=receipt)
    with pytest.raises(ValueError, match="LINEAGE|SPATIAL_SPLIT|HOLDOUT_LEAKAGE"):
        compose_local_policy_advisors(base_package_path=base.manifest_path.parent,
            output_package_path=tmp_path / "rejected",
            composition_receipt_path=tmp_path / "out.json",
            package_id="test.rejected-composition", display_name="Synthetic fixture",
            additions=(selected,))
    assert not (tmp_path / "rejected").exists() and not (tmp_path / "out.json").exists()


# 功能：
#   验证不同训练生产者的必需嵌套字段均为对象，重新计算摘要也不能使错误结构合法。
# 输入：
#   recipe：完整合成配方。
#   role：需要测试的专家角色。
#   field：必须为对象的字段名。
#   invalid：替换该字段的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("role,field", [
    ("local-navigation-policy", "metrics"), ("local-navigation-policy", "config"),
    ("risk-critic", "dataset_receipts_sha256"), ("perception-encoder", "config"),
    ("state-anomaly-detector", "advisors"),
])
@pytest.mark.parametrize("invalid", [None, []])
def test_nested_training_receipt_must_be_an_object(recipe, role, field, invalid):
    source = next(s for s in recipe.sources if s.role == role)
    artifact = next(a for a in recipe.manifest.artifacts if a.role == role)
    receipt = json.loads(source.training_receipt_path.read_bytes())
    receipt[field] = invalid
    with pytest.raises(ValueError, match="ENSEMBLE_.*NOT_OBJECT"):
        validate_expert_training_receipt(role, artifact.sha256, receipt, recipe.manifest)


# 功能：
#   验证非法读取预算在文件读取前拒绝，负数不能触发无限读取。
# 输入：
#   tmp_path：测试独立目录。
#   limit：非法字节预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("limit", [-2, -1, 0, True, 1.5, 2**80])
def test_invalid_bound_is_rejected_before_reading(tmp_path, limit):
    path = tmp_path / "small.bin"
    data = b"test"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SOURCE_BYTE_LIMIT_INVALID"):
        read_bound_content(path, hashlib.sha256(data).hexdigest(), maximum_bytes=limit)


# 功能：
#   验证布尔值即便与整数一数值相等，也不能冒充视觉特征维数。
# 输入：
#   recipe：完整合成配方。
# 输出：
#   None：不返回业务数据。
def test_boolean_visual_dimension_is_not_an_integer_contract(recipe):
    source = next(s for s in recipe.sources if s.role == "local-navigation-policy")
    artifact = next(a for a in recipe.manifest.artifacts if a.role == source.role)
    receipt = json.loads(source.training_receipt_path.read_bytes())
    receipt["config"]["visual_feature_count"] = True  # True == the fixture's one feature.
    with pytest.raises(ValueError, match="NAVIGATION_TRAINING_IDENTITY_MISMATCH"):
        validate_expert_training_receipt(source.role, artifact.sha256, receipt, recipe.manifest)
