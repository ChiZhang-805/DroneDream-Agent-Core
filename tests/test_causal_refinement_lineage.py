"""Synthetic contract/fault tests only; no simulated or actual aircraft."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from test_causal_policy import samples
from test_mission_groups import group_fixture, route, split_fixture
from test_ppo_cli_memory import replay_fixture

from dronedream_agent_core.training.causal_policy import (
    CausalPilotPolicy,
    CausalPolicyConfig,
    FrozenCausalEvaluation,
    causal_examples,
)
from dronedream_agent_core.training.causal_replay import (
    CAUSAL_SPLIT_CONTRACT,
    REPLAY_FILES,
    bind_source_group_metrics,
    protected_validation_groups,
    read_bound_replay,
    require_unseen_validation,
    validate_split_sources,
)
from dronedream_agent_core.training.flight_environment import RewardConfig
from dronedream_agent_core.training.mission_groups import mission_group_evidence
from dronedream_agent_core.training.ppo import PPOConfig
from scripts import refine_local_policy_ppo as cli


# 功能：
#   任一回放文件的实际字节变化都必须在解码或启动物理仿真前被发现。
# 输入：
#   tmp_path：隔离的精调输入目录。
#   field：本次破坏的固定回放文件名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", REPLAY_FILES)
def test_every_exported_input_is_bound_before_decoding_or_starting_physics(tmp_path, field):
    args, base, raw = replay_fixture(tmp_path)
    path = getattr(args, field)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="REPLAY_CONTENT_CHANGED:" + field):
        cli.build_trainer(args, PPOConfig(), RewardConfig(), base_content=base, receipt_content=raw)


# 功能：
#   旧回执缺少当前分组契约时，不能借用新回放文件获得有效来源资格。
# 输入：
#   tmp_path：隔离回放目录。
# 输出：
#   None：不返回业务数据。
def test_old_receipt_cannot_borrow_new_replay_files(tmp_path):
    args, _, raw = replay_fixture(tmp_path)
    receipt = json.loads(raw)
    receipt.pop("split_contract")
    with pytest.raises(ValueError, match="REQUIRES_BOUND_REPLAY"):
        read_bound_replay({n: getattr(args, n) for n in REPLAY_FILES}, receipt)


# 功能：
#   恢复训练必须匹配原留出与采集上下文绑定，合法恢复不额外执行优化。
# 输入：
#   tmp_path：隔离检查点及回放目录。
# 输出：
#   None：不返回业务数据。
def test_resume_cannot_change_holdout_or_rollout_context(tmp_path):
    args, base, raw = replay_fixture(tmp_path)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        trainer = cli.build_trainer(args, PPOConfig(), RewardConfig(),
                                    base_content=base, receipt_content=raw)
        trainer.refinement_context_sha256 = "a" * 64
        path = tmp_path / "resume.pt"
        trainer.save_checkpoint(path)
        trainer.refinement_context_sha256 = "b" * 64
        with pytest.raises(ValueError, match="CHECKPOINT_BINDING_MISMATCH"):
            trainer.restore_checkpoint(path)
        trainer.refinement_context_sha256 = "a" * 64
        trainer.restore_checkpoint(path)
        assert trainer.updates == 0
    finally:
        torch.set_num_threads(old_threads)


# 功能：
#   已用于旧基座训练的路线不能在热启动时改标为独立留出路线。
# 输入：
#   tmp_path：隔离的旧回执目录。
# 输出：
#   None：不返回业务数据。
def test_warmstart_cannot_relabel_previously_trained_routes_as_validation(tmp_path):
    _, _, raw = replay_fixture(tmp_path)
    receipt = json.loads(raw)
    with pytest.raises(ValueError, match="ALREADY_TRAINED_ON_VALIDATION"):
        require_unseen_validation(receipt, set(receipt["metrics"]["training_groups"]))


# 功能：
#   即使无标签留出历史很短、不产生完整训练窗口，也不能流入训练侧。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_short_unlabelled_holdout_history_still_cannot_enter_training():
    train, validation = samples(), samples(100, "val")
    with pytest.raises(ValueError, match="MISSION_GROUP_LEAKAGE"):
        validate_split_sources([*train, validation[0]], validation, group_fixture())


# 功能：
#   全部来源组应独立于完整窗口统计受到保护，补充回执不能改写原统计。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_source_groups_protect_routes_even_without_a_complete_labelled_window():
    original = {"training_groups": ["a"], "validation_groups": ["b"],
                "training_window_count": 30, "validation_window_count": 10}
    actual = bind_source_group_metrics(original, ({"a", "history-train"}, {"b", "history-val"}))
    assert actual["validation_groups"] == ["b", "history-val"]
    assert actual["validation_window_groups"] == ["b"]
    assert actual["training_window_count"] == 30 and original["training_groups"] == ["a"]


# 功能：
#   采集器必须保留当前回执中的有效留出组，缺失、改名或错误类型不能被当成空集合。
# 输入：
#   tmp_path：隔离的回执夹具目录。
# 输出：
#   None：不返回业务数据。
def test_collectors_must_preserve_existing_held_out_groups(tmp_path):
    _, _, raw = replay_fixture(tmp_path)
    receipt = json.loads(raw)
    assert protected_validation_groups(receipt) == {group_fixture().groups["val"]}
    for groups in (None, [], ["renamed"], [False]):
        receipt["metrics"]["validation_groups"] = groups
        with pytest.raises(ValueError, match="CURRENT_HELD_OUT_GROUPS"):
            protected_validation_groups(receipt)


# 功能：
#   多轮迁移继续保护祖先调参路线；已迁移却没有历史记录的旧回执不能当成完整来源。
# 输入：
#   无：使用明确合成的路线摘要及父模型身份。
# 输出：
#   None：当前与祖先组共同保留，历史缺失或格式错误明确拒绝。
def test_ancestral_tuning_groups_remain_protected():
    receipt = dict(split_contract=CAUSAL_SPLIT_CONTRACT, initial_policy_sha256='a' * 64,
        metrics=dict(validation_groups=['b' * 64], historical_validation_groups=['c' * 64]))
    assert protected_validation_groups(receipt) == {'b' * 64, 'c' * 64}
    for value in (None, ['not-a-digest'], [False]):
        receipt['metrics']['historical_validation_groups'] = value
        with pytest.raises(ValueError, match='ANCESTRAL_HELD_OUT_GROUPS'):
            protected_validation_groups(receipt)
    del receipt['metrics']['historical_validation_groups']
    with pytest.raises(ValueError, match='ANCESTRAL_HELD_OUT_GROUPS'):
        protected_validation_groups(receipt)


# 功能：
#   无合法历史训练组来源时拒绝热启动，不能把空值或随意路线名当成摘要。
# 输入：
#   metrics：缺失或损坏的历史训练统计。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("metrics", [None, [], {"training_groups": [""]},
                                     {"training_groups": ["renamed-route"]}])
def test_invalid_training_group_provenance_cannot_pass_warmstart(metrics):
    from dronedream_agent_core.training.causal_replay import CAUSAL_SPLIT_CONTRACT

    with pytest.raises(ValueError, match="CURRENT_SPLIT_PROVENANCE"):
        require_unseen_validation({"split_contract": CAUSAL_SPLIT_CONTRACT,
                                   "metrics": metrics}, {"a" * 64})


# 功能：
#   将合成路线及语义文件写入隔离目录，构造与实际字节绑定的采集配置。
# 输入：
#   tmp_path：隔离的环境文件目录。
#   points：合成路线上的空间点。
# 输出：
#   config：文件路径、任务名及实际文件摘要的配置。
def environment_config(tmp_path, points):
    route_path, semantic_path = tmp_path / "route.json", tmp_path / "semantic.json"
    route_path.write_text(json.dumps(route(points)))
    semantic_path.write_text('{"map":"synthetic"}')
    config = {"route": str(route_path), "semantic": str(semantic_path), "mission_id": "renamed",
        "asset_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("route", route_path), ("semantic", semantic_path))}}
    return config


# 功能：
#   留出路线改名或反向遍历仍被识别为同一空间路线；文件摘要变化也必须在采集前拒绝。
# 输入：
#   tmp_path：隔离地图和路线目录。
# 输出：
#   None：不返回业务数据。
def test_renamed_reversed_validation_route_rejected_before_factory_import(tmp_path):
    config = environment_config(tmp_path, [(2, 0, 1), (1, 0, 1), (0, 0, 1), (2, 0, 1)])
    held_out = split_fixture(semantic=config["asset_sha256"]["semantic"])
    with pytest.raises(ValueError, match="HELD_OUT_SPATIAL_ROUTE"):
        cli.prepare_rollout_config(config, {held_out.group_sha256})
    allowed, split = cli.prepare_rollout_config(config, {split_fixture(y=4).group_sha256})
    assert allowed["held_out_route_groups"] == [split_fixture(y=4).group_sha256]
    assert split == mission_group_evidence((tmp_path / "route.json").read_bytes(),
                                           config["asset_sha256"]["semantic"])
    assert "held_out_route_groups" not in config
    (tmp_path / "semantic.json").write_text("{}")
    with pytest.raises(ValueError, match="ROLLOUT_ASSET_CHANGED:semantic"):
        cli.prepare_rollout_config(config, set())


# 功能：
#   冻结评价不产生梯度或修改权重，未覆盖的模式／方向保持空统计而不是伪造准确率。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_compiled_evaluation_reports_missing_axis_coverage_without_fabricating_accuracy():
    config = CausalPolicyConfig(history_length=4, encoder_width=32, recurrent_width=32,
                               head_width=32, epochs=1)
    examples = causal_examples(samples(), stream_groups={"train": "synthetic"}, history_length=4)
    examples = [replace(e, sample=e.sample.model_copy(update={"target_action_index": 11,
                "target_pilot_control": [.3, -.3, 0., 0.]})) for e in examples]
    evaluation = FrozenCausalEvaluation.compile(examples, config)
    model = CausalPilotPolicy(config).train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        metrics = evaluation.evaluate(model)
    finally:
        torch.set_num_threads(old_threads)
    assert model.training and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    axes = metrics["validation_by_axis"]
    assert axes["forward"]["positive"]["count"] == len(examples)
    assert axes["right"]["negative"]["count"] == len(examples)
    assert axes["yaw"]["positive"] == {"count": 0, "mae": None}
    assert metrics["validation_by_mode"]["hold"] == {"count": 0, "accuracy": None}
    from dronedream_agent_core.training.flight_environment import MODES
    assert tuple(metrics["validation_by_mode"]) == MODES
