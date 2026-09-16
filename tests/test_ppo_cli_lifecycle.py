"""PPO entry-point fault injection; synthetic records never authorize or start flight."""

import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_causal_refinement_lineage import environment_config
from test_ppo_cli_memory import replay_fixture

from dronedream_agent_core.training.causal_training_inputs import training_cpu_threads
from dronedream_agent_core.training.evidence_files import decode_evidence_rows
from dronedream_agent_core.training.flight_environment import RewardConfig
from dronedream_agent_core.training.ppo import PPOConfig
from scripts import refine_local_policy_ppo as cli


# 功能：
#   建立真实小型检查点和回放，但用明确替身替换采集及优化，不启动仿真或飞行。
# 输入：
#   tmp_path：各次测试独立的证据目录。
#   monkeypatch：隔离环境工厂、采集、优化和暂停诊断的工具。
# 输出：
#   case：入口参数、真实训练器及模拟环境的测试上下文。
@pytest.fixture
def refinement_case(tmp_path, monkeypatch):
    args, base, raw = replay_fixture(tmp_path)
    args.base_training_receipt = tmp_path / "base-receipt.json"
    args.base_training_receipt.write_bytes(raw)
    args.config = tmp_path / "ppo.json"
    args.config.write_text(PPOConfig().model_dump_json(), encoding="utf-8")
    args.reward_config = args.resume = None
    args.environment_factory = "test_adapter:create"
    args.environment_config = tmp_path / "environment.json"
    configuration = environment_config(tmp_path, [(0, 8, 1), (1, 8, 1), (2, 8, 1)])
    args.environment_config.write_text(json.dumps(configuration), encoding="utf-8")
    args.output, args.updates = tmp_path / "output", 1
    with training_cpu_threads(1):
        trainer = cli.build_trainer(
            args, PPOConfig(), RewardConfig(), base_content=base, receipt_content=raw
        )
        _, split = cli.prepare_rollout_config(configuration, trainer.held_out_groups)
        env = SimpleNamespace(
            simulation_only=True,
            evidence_kind="px4-gazebo",
            mission_split=split,
            visual=None,
            quiesce=Mock(),
            close=Mock(),
        )
        trainer.collect = Mock(return_value=SimpleNamespace(records=[{"synthetic": True}]))

        # 功能：
        #   只模拟优化计数推进，保留真实网络但不执行梯度更新。
        # 输入：
        #   rollout：明确的合成采集替身。
        # 输出：
        #   metrics：用于检查入口写入和输出顺序的测试统计。
        def update(rollout):
            assert rollout.records == [{"synthetic": True}]
            trainer.updates += 1
            metrics = {"updates": trainer.updates, "synthetic": True}
            return metrics

        trainer.update = Mock(side_effect=update)
        monkeypatch.setattr(cli, "build_trainer", Mock(return_value=trainer))
        factory = Mock(return_value=env)
        monkeypatch.setattr(
            cli, "importlib",
            SimpleNamespace(import_module=Mock(return_value=SimpleNamespace(create=factory))),
        )
        monitor = Mock()
        monitor.close.return_value = {"synthetic": True, "pauses": []}
        monkeypatch.setattr(cli, "InterpreterPauseMonitor", Mock(return_value=monitor))
        case = SimpleNamespace(
            args=args, trainer=trainer, env=env, factory=factory, monitor=monitor
        )
        yield case


# 功能：
#   使用真实序列化、网络导出和回放绑定检查完整入口，最终回执必须晚于环境关闭。
# 输入：
#   refinement_case：合成采集且真实小型模型的隔离上下文。
# 输出：
#   None：不返回业务数据。
def test_refinement_publishes_receipt_only_after_close(refinement_case):
    case = refinement_case
    receipt_path = case.args.output / "training-receipt.json"

    # 功能：
    #   检查关闭时尚未宣告训练完成，且冻结回放已不再占据训练器字节字段。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def close():
        assert not receipt_path.exists()
        assert not hasattr(case.trainer, "frozen_replay_contents")

    case.env.close.side_effect = close
    assert cli.refine(case.args) == 0
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["updates"] == 1 and receipt["qualified_for_flight"] is False
    replay = (case.args.output / "rollouts.jsonl").read_bytes()
    assert decode_evidence_rows(replay) == [{"synthetic": True}]
    assert receipt["input_sha256"]["rollout_evidence"] == hashlib.sha256(replay).hexdigest()
    case.env.close.assert_called_once()


# 功能：
#   环境关闭失败时保留候选证据，但禁止发布完整训练回执。
# 输入：
#   refinement_case：合成采集上下文。
# 输出：
#   None：不返回业务数据。
def test_close_failure_cannot_publish_success(refinement_case):
    case = refinement_case
    case.env.close.side_effect = RuntimeError("close-failed")
    with pytest.raises(RuntimeError, match="close-failed"):
        cli.refine(case.args)
    assert not (case.args.output / "training-receipt.json").exists()


# 功能：
#   即使环境声称同一证据种类，错误仿真标志、路线或缺失停稳接口都必须阻止采集。
# 输入：
#   refinement_case：允许替换环境属性的隔离上下文。
#   field：被破坏的环境接口属性。
#   value：不能被当作有效契约的值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value", [("simulation_only", 1), ("quiesce", None), ("evidence_kind", "real")]
)
def test_invalid_environment_closes_without_collection(refinement_case, field, value):
    case = refinement_case
    setattr(case.env, field, value)
    with pytest.raises(ValueError):
        cli.refine(case.args)
    case.trainer.collect.assert_not_called()
    case.env.close.assert_called_once()
    assert not (case.args.output / "training-receipt.json").exists()


# 功能：
#   环境关闭期间若最终回执路径被占用，发布器必须拒绝覆盖而不是替换外部文件。
# 输入：
#   refinement_case：真实发布器与合成环境的上下文。
# 输出：
#   None：不返回业务数据。
def test_receipt_created_during_cleanup_is_preserved(refinement_case):
    case = refinement_case
    receipt_path = case.args.output / "training-receipt.json"
    case.env.close.side_effect = lambda: receipt_path.write_bytes(b"existing-evidence")
    with pytest.raises(FileExistsError):
        cli.refine(case.args)
    assert receipt_path.read_bytes() == b"existing-evidence"


# 功能：
#   采集错误为主错误，暂停诊断和环境清理的再次失败只能追加说明，不能覆盖原异常。
# 输入：
#   refinement_case：允许注入三个独立失败的上下文。
# 输出：
#   None：不返回业务数据。
def test_collection_error_survives_both_cleanup_failures(refinement_case):
    case = refinement_case
    primary = RuntimeError("collection-failed")
    case.trainer.collect.side_effect = primary
    case.monitor.close.side_effect = RuntimeError("diagnostics-failed")
    case.env.close.side_effect = RuntimeError("close-failed")
    with pytest.raises(RuntimeError) as caught:
        cli.refine(case.args)
    assert caught.value is primary
    assert any("diagnostics-failed" in note for note in primary.__notes__)
    assert any("close-failed" in note for note in primary.__notes__)
    case.trainer.update.assert_not_called()
    assert not (case.args.output / "training-receipt.json").exists()


# 功能：
#   拒绝非对象或非有限值记录，不能写入一个后续读取器无法接受的证据行。
# 输入：
#   record：类型错误或含 NaN 的测试记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("record", [[], None, {"value": float("nan")}])
def test_rollout_writer_rejects_invalid_records(record):
    evidence = io.BytesIO()
    with pytest.raises(ValueError):
        cli.append_rollout_records(evidence, [record])
    assert evidence.getvalue() == b""


# 功能：
#   单行预算必须包含行末换行符，否则刚好达到 JSON 上限的记录无法再次读取。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_rollout_row_budget_includes_newline():
    limit = 4 * 1024 * 1024
    record = {"text": "x" * (limit - len('{"text":""}'))}
    evidence = io.BytesIO()
    with pytest.raises(ValueError):
        cli.append_rollout_records(evidence, [record])
    assert evidence.getvalue() == b""


# 功能：
#   使用虚拟位置和短写故障验证全文件预算，避免为边界测试实际分配大型文件。
# 输入：
#   position：写入前的模拟字节位置。
#   short_write：是否模拟只写入一个字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("position,short_write", [(512 * 1024 * 1024, False), (0, True)])
def test_rollout_write_size_and_short_write(position, short_write):
    evidence = Mock()
    evidence.tell.return_value = position
    evidence.write.return_value = 1
    with pytest.raises(OSError if short_write else ValueError):
        cli.append_rollout_records(evidence, [{"value": 1}])
    if not short_write:
        evidence.write.assert_not_called()
