"""DAgger CLI contracts using grounded synthetic fixtures; no aircraft is started."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_stream_episode_boundaries import grounded_episode

from scripts import annotate_local_policy_dagger as annotation
from scripts import collect_local_policy_dagger as collection


# 功能：
#   用已通过真实离线读取器的合成回合构造命令行参数，明确替换原生运行环境与采集器。
# 输入：
#   tmp_path：独立回合及输出目录。
#   monkeypatch：隔离模型加载、原生环境和采集的工具。
# 输出：
#   case：两个入口共享的参数、回合和故障注入对象。
@pytest.fixture
def cli_case(tmp_path, monkeypatch):
    episode_path, teacher_config, episode = grounded_episode(tmp_path)
    reset = json.loads((episode_path / "reset.json").read_bytes())
    teacher_path, env_path = tmp_path / "teacher.json", tmp_path / "environment.json"
    teacher_path.write_text(teacher_config.model_dump_json(), encoding="utf-8")
    env_path.write_text(json.dumps(reset["config"]), encoding="utf-8")
    policy_path, receipt_path, onnx_path = [
        tmp_path / name for name in ("base.pt", "base.json", "base.onnx")
    ]
    policy_path.write_bytes(b"synthetic-checkpoint")
    onnx_path.write_bytes(b"synthetic-graph")
    receipt_path.write_text('{"synthetic":true}', encoding="utf-8")
    args = SimpleNamespace(
        episode=episode_path,
        teacher_config=teacher_path,
        environment_config=env_path,
        base_policy=policy_path,
        base_training_receipt=receipt_path,
        policy_onnx=onnx_path,
        output=tmp_path / "output",
        steps=1,
        seed=805,
    )
    env = SimpleNamespace(
        visual=None, episode_path=episode_path, evidence_kind="px4-gazebo", close=Mock()
    )
    student = SimpleNamespace(
        artifact_sha256=reset["policy_sha256"], startup_maximum_absolute_error=0.0
    )
    monkeypatch.setattr(
        collection,
        "load_causal_checkpoint",
        Mock(return_value=SimpleNamespace(config=SimpleNamespace(visual_feature_count=0))),
    )
    monkeypatch.setattr(collection, "OnnxCausalStudent", Mock(return_value=student))
    monkeypatch.setattr(collection, "verify_causal_checkpoint_receipt", Mock(return_value=None))
    monkeypatch.setattr(collection, "protected_validation_groups", Mock(return_value=set()))
    factory = Mock(return_value=env)
    collector = Mock(return_value=list(episode.visits))
    monitor = Mock()
    monitor.close.return_value = {"synthetic": True}
    monkeypatch.setattr(collection, "Px4GazeboTrainingEnvironment", factory)
    monkeypatch.setattr(collection, "collect_native_control_stream", collector)
    monkeypatch.setattr(collection, "InterpreterPauseMonitor", Mock(return_value=monitor))
    case = SimpleNamespace(
        args=args, env=env, factory=factory, collector=collector, monitor=monitor
    )
    # 旧入口的线程副作用也不能影响后续测试；新入口需在断言前自己恢复。
    previous = torch.get_num_threads()
    try:
        yield case
    finally:
        torch.set_num_threads(previous)


# 功能：
#   经实际命令行解析器执行指定入口，未模拟参数校验本身。
# 输入：
#   case：参数路径和故障注入上下文。
#   implementation：采集或离线标注模块。
#   monkeypatch：替换当前命令行的工具。
# 输出：
#   exit_code：实际入口的退出码。
def run_case(case, implementation, monkeypatch):
    names = (
        ["episode", "teacher_config", "output"]
        if implementation is annotation
        else [
            "base_policy",
            "policy_onnx",
            "base_training_receipt",
            "environment_config",
            "teacher_config",
            "output",
            "steps",
            "seed",
        ]
    )
    argv = ["dagger"]
    for name in names:
        argv.extend(["--" + name.replace("_", "-"), str(getattr(case.args, name))])
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = implementation.main()
    return exit_code


# 功能：
#   两个入口必须在任何标注或采集开始之前拒绝重复教师配置键。
# 输入：
#   cli_case：有效合成输入。
#   monkeypatch：替换命令行的工具。
#   implementation：被测试的实际入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("implementation", [annotation, collection])
def test_duplicate_teacher_config_rejected_before_output(cli_case, monkeypatch, implementation):
    path = cli_case.args.teacher_config
    data = path.read_bytes()
    key, value = next(iter(json.loads(data).items()))
    path.write_bytes(("{" + json.dumps(key) + ":" + json.dumps(value) + ",").encode() + data[1:])
    with pytest.raises(ValueError, match="JSON_DUPLICATE_KEY"):
        run_case(cli_case, implementation, monkeypatch)
    cli_case.collector.assert_not_called()
    assert not cli_case.args.output.exists()


# 功能：
#   采集失败与诊断、关闭同时失败时保留首个错误，不能被清理异常覆盖。
# 输入：
#   cli_case：可注入异常的采集上下文。
#   monkeypatch：命令行替换工具。
# 输出：
#   None：不返回业务数据。
def test_collection_primary_error_and_cpu_threads_are_preserved(cli_case, monkeypatch):
    case = cli_case
    original_threads = torch.get_num_threads()
    primary = RuntimeError("collection-failed")
    case.collector.side_effect = primary
    case.monitor.close.side_effect = RuntimeError("diagnostics-failed")
    case.env.close.side_effect = RuntimeError("close-failed")
    with pytest.raises(RuntimeError) as caught:
        run_case(case, collection, monkeypatch)
    assert caught.value is primary
    assert torch.get_num_threads() == original_threads
    assert len(primary.__notes__) == 2
    assert not (case.args.output / "collection-receipt.jsonl").exists()


# 功能：
#   已得到合成动作和真实教师标签也不能在关闭失败后保留成功回执。
# 输入：
#   cli_case：真实离线教师和替身环境的上下文。
#   monkeypatch：命令行替换工具。
# 输出：
#   None：不返回业务数据。
def test_collection_close_failure_does_not_publish_completion(cli_case, monkeypatch):
    case = cli_case
    case.env.close.side_effect = RuntimeError("close-failed")
    with pytest.raises(RuntimeError, match="close-failed"):
        run_case(case, collection, monkeypatch)
    assert (case.args.output / "student-visits.jsonl").exists()
    assert not (case.args.output / "collection-receipt.jsonl").exists()


# 功能：
#   标注异常为主错误，保存教师诊断失败只能追加说明，不能掩盖标注问题。
# 输入：
#   cli_case：具有真实合成回合的上下文。
#   monkeypatch：注入标注及保存异常的工具。
# 输出：
#   None：不返回业务数据。
def test_annotation_preserves_primary_failure(cli_case, monkeypatch):
    primary = RuntimeError("label-failed")
    monkeypatch.setattr(annotation, "label_stream_visits", Mock(side_effect=primary))
    monkeypatch.setattr(annotation, "write_rows", Mock(side_effect=OSError("receipt-failed")))
    with pytest.raises(RuntimeError) as caught:
        run_case(cli_case, annotation, monkeypatch)
    assert caught.value is primary
    assert any("receipt-failed" in note for note in primary.__notes__)
    assert not (cli_case.args.output / "annotation-receipt.jsonl").exists()


# 功能：
#   有效合成回合经两个实际入口形成可读数据集，回执记录真正使用的教师配置摘要。
# 输入：
#   cli_case：真实离线读取与教师、明确替身采集的上下文。
#   monkeypatch：替换命令行的工具。
#   implementation：被检查的入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("implementation", [annotation, collection])
def test_successful_dagger_cli_preserves_lineage(cli_case, monkeypatch, implementation):
    import hashlib

    from dronedream_agent_core.training.dagger_artifacts import load_dagger_training_artifacts

    case = cli_case
    previous_threads = torch.get_num_threads()
    assert run_case(case, implementation, monkeypatch) == 0
    name = "annotation" if implementation is annotation else "collection"
    receipt = json.loads((case.args.output / f"{name}-receipt.jsonl").read_bytes())
    assert (
        receipt["teacher_config_sha256"]
        == hashlib.sha256(case.args.teacher_config.read_bytes()).hexdigest()
    )
    assert receipt["qualified_for_flight"] is False
    assert len(load_dagger_training_artifacts(case.args.output)[0]) == 1
    assert torch.get_num_threads() == previous_threads
    if implementation is collection:
        case.env.close.assert_called_once()
        visits_path = case.args.output / "student-visits.jsonl"
        visits_path.write_bytes(visits_path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="CONTENT_CHANGED:student-visits"):
            load_dagger_training_artifacts(case.args.output)


# 功能：
#   未知回合类型不能默认回退为奖励步骤或由目录文件名重新猜测。
# 输入：
#   cli_case：有效输入路径。
#   monkeypatch：替换读取器返回类型的工具。
# 输出：
#   None：不返回业务数据。
def test_unknown_annotation_record_kind_cannot_choose_a_labeler(cli_case, monkeypatch):
    monkeypatch.setattr(
        annotation,
        "load_grounded_imitation_episode",
        Mock(return_value=SimpleNamespace(native_record_kind="unknown")),
    )
    labeler = Mock(side_effect=AssertionError("labeler-must-not-run"))
    monkeypatch.setattr(annotation, "label_student_visits", labeler)
    monkeypatch.setattr(annotation, "label_stream_visits", labeler)
    with pytest.raises(ValueError, match="RECORD_KIND_INVALID"):
        run_case(cli_case, annotation, monkeypatch)
    labeler.assert_not_called()
    assert not cli_case.args.output.exists()


# 功能：
#   运行期间发现的教师错误保留原采集访问，不被诊断写入的第二个错误替代。
# 输入：
#   cli_case：可隔离教师阶段的采集上下文。
#   monkeypatch：注入教师及教师诊断失败的工具。
# 输出：
#   None：不返回业务数据。
def test_collection_teacher_failure_preserves_visits(cli_case, monkeypatch):
    primary = RuntimeError("teacher-failed")
    original = collection.write_rows

    # 功能：
    #   只让教师诊断保存失败，其他证据仍使用真实写入器。
    # 输入：
    #   path：当前证据文件路径。
    #   rows：待写记录迭代器。
    # 输出：
    #   digest：非故障文件实际写入字节的摘要。
    def write(path, rows):
        if path.name == "counterfactual-receipts.jsonl":
            raise OSError("teacher-diagnostics-failed")
        digest = original(path, rows)
        return digest

    monkeypatch.setattr(collection, "write_rows", write)
    monkeypatch.setattr(collection, "label_stream_visits", Mock(side_effect=primary))
    with pytest.raises(RuntimeError) as caught:
        run_case(cli_case, collection, monkeypatch)
    assert caught.value is primary
    assert any("teacher-diagnostics-failed" in note for note in primary.__notes__)
    assert (cli_case.args.output / "student-visits.jsonl").exists()
    assert not (cli_case.args.output / "collection-receipt.jsonl").exists()
    cli_case.env.close.assert_called_once()
