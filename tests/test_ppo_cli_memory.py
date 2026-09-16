"""Offline replay lifetimes, with synthetic inputs and no simulator process."""

import hashlib
import json
import weakref
from types import SimpleNamespace

import torch
from test_causal_policy import samples
from test_mission_groups import group_fixture

from dronedream_agent_core.control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256
from dronedream_agent_core.local_policy_training import LocalPolicyObservation
from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    save_causal_checkpoint,
)
from dronedream_agent_core.training.causal_replay import (
    CAUSAL_SPLIT_CONTRACT,
    REPLAY_FILES,
    export_replay_bundle,
)
from dronedream_agent_core.training.flight_environment import RewardConfig
from dronedream_agent_core.training.ppo import PPOConfig
from scripts import refine_local_policy_ppo as cli


# 功能：
#   使用真实回放编译器检查大型 Python 样本在进入实时采集前释放，张量及来源摘要仍保留。
# 输入：
#   tmp_path：独立回放与检查点目录。
#   monkeypatch：记录解码对象弱引用的测试工具。
# 输出：
#   None：不返回业务数据。
def test_build_trainer_releases_python_replay_graph_before_live_collection(tmp_path, monkeypatch):
    args, base_content, receipt_content = replay_fixture(tmp_path)
    references = []
    original = cli.decode_replay

    # 功能：
    #   用真实解码器恢复回放，并保留不延长样本生命周期的弱引用。
    # 输入：
    #   content：已绑定原始字节的回放集合。
    # 输出：
    #   result：真实解码器产生的记录、分区清单和路线组。
    def decode(content):
        result = original(content)
        references.extend(weakref.ref(row) for rows in result[0].values() for row in rows)
        return result

    monkeypatch.setattr(cli, "decode_replay", decode)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        trainer = cli.build_trainer(
            args,
            PPOConfig(),
            RewardConfig(),
            base_content=base_content,
            receipt_content=receipt_content,
        )
    finally:
        torch.set_num_threads(previous)
    assert references and all(ref() is None for ref in references)
    assert trainer.replay.inputs.shape[0] > 0
    assert trainer.replay.pilot_controls.shape[1] == 4
    assert len(trainer.replay_hash) == 64
    assert trainer.baseline_evaluation["validation_window_count"] > 0


# 功能：
#   创建当前架构的小型随机权重及合成空间分区回放，不是产品训练或飞行数据。
# 输入：
#   tmp_path：独立文件目录。
# 输出：
#   args：回放路径及当前架构参数。
#   base_content：实际保存的检查点字节。
#   receipt_content：绑定检查点与回放摘要的合成回执字节。
def replay_fixture(tmp_path):
    config = CausalPolicyConfig(
        history_length=4, encoder_width=32, recurrent_width=32, head_width=32, epochs=1
    )
    base = CausalPilotPolicy(config)
    path = tmp_path / "base.pt"
    save_causal_checkpoint(base, path)
    train, validation = samples(), samples(100, "val")

    # 功能：
    #   只提取可观测字段构造无标签历史，避免教师标签进入在线策略输入。
    # 输入：
    #   rows：带监督标签的合成样本列表。
    # 输出：
    #   observations：无标签的真实形状观测列表。
    def history(rows):
        observations = [
            LocalPolicyObservation.model_validate(
                {name: getattr(row, name) for name in LocalPolicyObservation.model_fields}
            )
            for row in rows
        ]
        return observations

    manifest = group_fixture()
    hashes = export_replay_bundle(
        tmp_path,
        training=train,
        validation=validation,
        training_history=history(train),
        validation_history=history(validation),
        manifest=manifest,
    )
    receipt = dict(
        architecture="causal-gru-control",
        expert_role="local-navigation-policy",
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        feature_contract_sha256=CURRENT_POLICY_FEATURE_CONTRACT_SHA256,
        config=config.model_dump(),
        visual_input_contract=None,
        split_contract=CAUSAL_SPLIT_CONTRACT,
        replay_artifact_sha256=hashes,
        metrics={
            "training_groups": [manifest.groups["train"]],
            "validation_groups": [manifest.groups["val"]],
        },
    )
    args = SimpleNamespace(
        architecture="causal-gru-control",
        base_policy=path,
        **{name: tmp_path / f for name, f in REPLAY_FILES.items()},
    )
    base_content, receipt_content = path.read_bytes(), json.dumps(receipt).encode()
    return args, base_content, receipt_content
