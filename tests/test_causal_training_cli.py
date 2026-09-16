"""Strict CLI source boundaries, with synthetic replay and no simulator or paid model."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_ppo_cli_memory import replay_fixture

from dronedream_agent_core.training import ensemble_lineage
from scripts import train_causal_control_ensemble as ensemble
from scripts import train_causal_control_role as role


# 功能：
#   为两个因果训练入口建立相同的合成输入，实际权重来自小型本机测试网络。
# 输入：
#   tmp_path：独立数据与输出目录。
#   implementation：单专家或集合训练入口模块。
# 输出：
#   argv：用于调用实际命令行解析器的参数列表。
#   config_path：可注入歧义内容的配置路径。
#   replay_path：可注入错误记录的训练数据路径。
def arguments(tmp_path, implementation):
    args, _, receipt = replay_fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(json.loads(receipt)["config"]), encoding="utf-8")
    options = {
        "train": args.training_replay,
        "validation": args.validation_replay,
        "training-observations": args.training_observations,
        "validation-observations": args.validation_observations,
        "stream-groups": args.stream_groups,
        "config": config_path,
        "output": tmp_path / "output",
        "cpu-threads": 1,
    }
    if implementation is role:
        options["expert-role"] = "local-navigation-policy"
    else:
        options.update({"base-package": tmp_path / "base", "package-id": "unit"})
    argv = ["training"]
    for name, value in options.items():
        argv.extend(["--" + name, str(value)])
    return argv, config_path, args.training_replay


# 功能：
#   验证两个入口拒绝重复配置键，不以最后一个值覆盖前一个值后继续训练。
# 输入：
#   tmp_path：独立合成输入目录。
#   monkeypatch：隔离集合基座检查并禁止优化器调用的测试工具。
#   implementation：被检查的实际入口模块。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("implementation", [role, ensemble])
def test_duplicate_cli_config_is_rejected_before_optimization(
    tmp_path, monkeypatch, implementation
):
    argv, path, _ = arguments(tmp_path, implementation)
    path.write_bytes(b'{"epochs":2,' + path.read_bytes()[1:])
    optimize = Mock(side_effect=AssertionError("optimizer-must-not-run"))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(implementation, "train_causal_policy", optimize)
    previous_threads = torch.get_num_threads()
    if implementation is ensemble:
        monkeypatch.setattr(ensemble, "validate_causal_base", lambda *a, **k: SimpleNamespace())
        monkeypatch.setattr(ensemble_lineage, "load_base_lineage", lambda *_: None)
    with pytest.raises(ValueError, match="JSON_DUPLICATE_KEY"):
        implementation.main()
    optimize.assert_not_called()
    assert torch.get_num_threads() == previous_threads
    assert not (tmp_path / "output").exists()


# 功能：
#   验证非视觉入口也不能静默跳过空白训练行，禁止把损坏文件当成合格回放。
# 输入：
#   tmp_path：独立数据目录。
#   monkeypatch：禁止优化器运行的测试工具。
# 输出：
#   None：不返回业务数据。
def test_nonvisual_role_rejects_blank_training_rows(tmp_path, monkeypatch):
    argv, _, path = arguments(tmp_path, role)
    path.write_bytes(path.read_bytes() + b"\n")
    optimize = Mock(side_effect=AssertionError("optimizer-must-not-run"))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(role, "train_causal_policy", optimize)
    with pytest.raises(json.JSONDecodeError):
        role.main()
    optimize.assert_not_called()
    assert not (tmp_path / "output").exists()


# 功能：
#   检查整批专家的种子边界，禁止先训练第一名专家再因下一名种子越界中断。
# 输入：
#   tmp_path：独立训练配置目录。
#   monkeypatch：禁止基座读取及优化的测试工具。
#   implementation：单专家或集合入口。
#   seed：单专家自身越界或集合后续专家越界的种子。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("implementation,seed", [(role, 2**63), (ensemble, 2**63 - 2)])
def test_cli_seed_range_is_checked_before_training(tmp_path, monkeypatch, implementation, seed):
    argv, path, _ = arguments(tmp_path, implementation)
    config = json.loads(path.read_bytes())
    config["seed"] = seed
    path.write_text(json.dumps(config), encoding="utf-8")
    optimize = Mock(side_effect=AssertionError("optimizer-must-not-run"))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(implementation, "train_causal_policy", optimize)
    if implementation is ensemble:
        monkeypatch.setattr(
            ensemble, "validate_causal_base", Mock(side_effect=AssertionError("base-must-not-load"))
        )
    with pytest.raises(ValueError, match="SEED_INVALID"):
        implementation.main()
    optimize.assert_not_called()
    assert not (tmp_path / "output").exists()
