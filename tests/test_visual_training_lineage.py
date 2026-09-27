"""Matching feature dimensions never establishes matching embedding semantics."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_causal_policy import samples

from dronedream_agent_core.local_policy_training import LocalPolicyObservation
from dronedream_agent_core.training.visual_lineage import (
    require_matching_visual_input,
    verify_causal_checkpoint_receipt,
    verify_visual_training_inputs,
)


# 功能：
#   验证单独检查点必须绑定当前权重、操纵角色、特征和视觉契约，拒绝错误身份复用。
# 输入：
#   无：测试自行构造检查点字节及回执。
# 输出：
#   None：不返回业务数据。
def test_standalone_simulation_checkpoint_requires_exact_training_identity():
    from dronedream_agent_core.control_feature_contract import (
        CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
    )

    content = b"synthetic-checkpoint-bytes-not-a-product-model"
    contract = {
        key: encoding(b"row\n")[key]
        for key in (
            "perception_encoder_sha256",
            "visual_feature_count",
            "visual_width",
            "visual_height",
            "visual_normalization",
        )
    }
    receipt = {
        "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
        "expert_role": "local-navigation-policy",
        "architecture": "causal-gru-control",
        "feature_contract_sha256": CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        "config": {"visual_feature_count": 1},
        "visual_input_contract": contract,
    }
    assert (
        verify_causal_checkpoint_receipt(
            content, json.dumps(receipt).encode(), expert_role="local-navigation-policy"
        )
        == contract
    )
    for field, value in (
        ("checkpoint_sha256", "0" * 64),
        ("expert_role", "recovery-policy"),
        ("architecture", "feedforward-control"),
        ("feature_contract_sha256", "f" * 64),
        ("visual_input_contract", None),
        ("config", {"visual_feature_count": 2}),
    ):
        with pytest.raises(ValueError, match="CAUSAL_CHECKPOINT"):
            verify_causal_checkpoint_receipt(
                content,
                json.dumps({**receipt, field: value}).encode(),
                expert_role="local-navigation-policy",
            )


# 功能：
#   为给定合成数据建立视觉编码测试回执，只用于验证绑定逻辑。
# 输入：
#   content：用于摘要和行数统计的测试数据字节。
# 输出：
#   receipt：包含编码器、预处理、输出摘要和样本数的合成回执。
def encoding(content):
    receipt = {
        "schema_version": "dronedream.local-policy-visual-encoding-receipt.v1",
        "qualification_granted": False,
        "output_sha256": hashlib.sha256(content).hexdigest(),
        "sample_count": len(content.splitlines()),
        "perception_encoder_sha256": "a" * 64,
        "visual_feature_count": 1,
        "visual_width": 32,
        "visual_height": 32,
        "visual_normalization": "imagenet",
    }
    return receipt


# 功能：
#   验证训练和验证特征绑定实际字节并使用相同编码器及预处理，不能仅靠相同维数复用。
# 输入：
#   无：测试自行构造两份独立数据与回执。
# 输出：
#   None：不返回业务数据。
def test_embedding_receipts_must_bind_actual_inputs_and_identical_encoders():
    train, validation = b"train\n", b"validation\n"
    left, right = encoding(train), encoding(validation)
    args = dict(training_content=train, validation_content=validation, feature_count=1)
    contract = verify_visual_training_inputs(
        **args, receipt_contents=[json.dumps(left).encode(), json.dumps(right).encode()]
    )
    assert contract["perception_encoder_sha256"] == "a" * 64
    for field, value in (
        ("perception_encoder_sha256", "b" * 64),
        ("visual_normalization", "zero-to-one"),
        ("visual_width", 64),
    ):
        with pytest.raises(ValueError, match="DIFFERENT_ENCODERS"):
            verify_visual_training_inputs(
                **args,
                receipt_contents=[
                    json.dumps(left).encode(),
                    json.dumps({**right, field: value}).encode(),
                ],
            )
    with pytest.raises(ValueError, match="DOES_NOT_BIND"):
        verify_visual_training_inputs(
            **{**args, "training_content": b"new training\n"},
            receipt_contents=[json.dumps(left).encode(), json.dumps(right).encode()],
        )
    with pytest.raises(ValueError, match="BOTH_ENCODING_RECEIPTS"):
        verify_visual_training_inputs(**args, receipt_contents=[])
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        require_matching_visual_input(None, contract)


# 功能：
#   验证视觉回执必须为 JSON 对象，错误容器得到校验错误而不是属性访问异常。
# 输入：
#   receipt：需要拒绝的回执根值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("receipt", [None, [], True, "receipt"])
def test_visual_receipt_requires_an_object(receipt):
    with pytest.raises(ValueError, match="VISUAL_ENCODING_RECEIPT"):
        verify_visual_training_inputs(
            training_content=b"row\n", validation_content=b"row\n", feature_count=1,
            receipt_contents=[json.dumps(receipt).encode()] * 2,
        )


# 功能：
#   验证模型视觉维数只能使用限定范围内的整数，不能隐式转换布尔值和浮点值。
# 输入：
#   count：非法特征维数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [False, True, -1, 1.0, 65537])
def test_feature_count_requires_a_bounded_integer(count):
    with pytest.raises(ValueError, match="VISUAL_FEATURE_COUNT"):
        verify_visual_training_inputs(
            training_content=b"", validation_content=b"", feature_count=count,
            receipt_contents=[],
        )


# 功能：
#   验证回执中的样本数和视觉维数也不接受布尔值冒充整数一。
# 输入：
#   field：需要篡改为布尔值的计数字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["sample_count", "visual_feature_count"])
def test_boolean_receipt_counts_cannot_impersonate_one_sample(field):
    receipt = {**encoding(b"row\n"), field: True}
    with pytest.raises(ValueError, match="VISUAL_ENCODING_RECEIPT"):
        verify_visual_training_inputs(
            training_content=b"row\n", validation_content=b"row\n", feature_count=1,
            receipt_contents=[json.dumps(receipt).encode()] * 2,
        )


# 功能：
#   1. 调用实际离线训练脚本验证视觉来源、五文件回放绑定和按轴评价输出。
#   2. 篡改视觉编码器身份后验证继续训练被拒绝，不生成错误候选包。
# 输入：
#   tmp_path：保存合成样本、训练配置和输出的测试目录。
#   regularized：是否通过真实命令行启用并核验独立正则化配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("regularized", [False, True])
def test_actual_role_training_binds_lineage_and_checks_warmstart(tmp_path, regularized):
    repository = Path(__file__).resolve().parents[1]
    for split, start, stream in (("train", 0, "train"), ("validation", 100, "val")):
        labels = [s.model_copy(update={"visual_features": [0.1]}) for s in samples(start, stream)]
        content = ("\n".join(s.model_dump_json() for s in labels) + "\n").encode()
        (tmp_path / f"{split}.jsonl").write_bytes(content)
        history = [
            LocalPolicyObservation.model_validate(
                {name: getattr(row, name) for name in LocalPolicyObservation.model_fields}
            )
            for row in labels
        ]
        (tmp_path / f"{split}-history.jsonl").write_text(
            "\n".join(s.model_dump_json() for s in history), encoding="utf-8"
        )
        (tmp_path / f"{split}-visual.json").write_text(
            json.dumps(encoding(content)), encoding="utf-8"
        )
    from test_mission_groups import group_fixture

    (tmp_path / "groups.json").write_text(group_fixture().model_dump_json())
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                history_length=4,
                visual_feature_count=1,
                encoder_width=32,
                recurrent_width=32,
                head_width=32,
                epochs=1,
            )
        )
    )
    command = [
        sys.executable,
        str(repository / "scripts/train_causal_control_role.py"),
        "--expert-role",
        "local-navigation-policy",
        "--cpu-threads",
        "1",
    ]
    for arg, name in (
        ("train", "train.jsonl"),
        ("validation", "validation.jsonl"),
        ("training-observations", "train-history.jsonl"),
        ("validation-observations", "validation-history.jsonl"),
        ("stream-groups", "groups.json"),
        ("config", "config.json"),
        ("training-visual-receipt", "train-visual.json"),
        ("validation-visual-receipt", "validation-visual.json"),
    ):
        command += ["--" + arg, str(tmp_path / name)]
    settings = {"balance_mission_groups": regularized,
                "visual_block_dropout": 0.35 if regularized else 0.0}
    if regularized:
        regularization_path = tmp_path / "regularization.json"
        regularization_path.write_text(json.dumps(settings), encoding="utf-8")
        command += ["--regularization-config", str(regularization_path)]
    trained = subprocess.run(
        command + ["--output", str(tmp_path / "trained")],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert trained.returncode == 0, trained.stderr
    receipt_path = tmp_path / "trained/training-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["metrics"]["regularization"] == settings
    if regularized:
        assert receipt["input_sha256"]["regularization_config"] == hashlib.sha256(
            regularization_path.read_bytes()).hexdigest()
    else:
        assert "regularization_config" not in receipt["input_sha256"]
    assert receipt["visual_input_contract"]["perception_encoder_sha256"] == "a" * 64
    assert len(receipt["visual_encoding_receipts_sha256"]) == 2
    from dronedream_agent_core.training.causal_replay import REPLAY_FILES, read_bound_replay
    replay = read_bound_replay(
        {name: tmp_path / "trained" / filename for name, filename in REPLAY_FILES.items()}, receipt)
    assert len(replay) == 5
    assert receipt["metrics"]["validation_by_axis"]["yaw"]["positive"]["count"] >= 0
    receipt["visual_input_contract"]["perception_encoder_sha256"] = "b" * 64
    receipt_path.write_text(json.dumps(receipt))
    rejected = subprocess.run(
        command
        + [
            "--output",
            str(tmp_path / "bad-refinement"),
            "--base-policy",
            str(tmp_path / "trained/local-navigation-policy.pt"),
            "--base-training-receipt",
            str(receipt_path),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert rejected.returncode != 0 and "VISUAL_INPUT_IDENTITY_MISMATCH" in rejected.stderr
    assert not (tmp_path / "bad-refinement").exists()
