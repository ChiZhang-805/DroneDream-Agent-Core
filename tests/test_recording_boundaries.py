"""Local recording fault fixtures; no sensor or flight qualification."""

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_learning_observation_recorder import request_fixture
from test_runtime_multimodal_dataset import _frame, _snapshot

from dronedream_agent_core import runtime_multimodal_dataset as dataset_module
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.learning_observation_recorder import LearningObservationRecorder
from dronedream_agent_core.runtime_multimodal_dataset import RuntimeMultimodalDatasetRecorder


# 功能：
#   在独立临时目录构造小容量记录器，避免触碰已有训练与飞行证据。
# 输入：
#   tmp_path：测试临时目录。
#   options：本例覆盖的记录策略。
# 输出：
#   recorder：尚无样本的新记录器。
def _recorder(tmp_path, **options):
    recorder = RuntimeMultimodalDatasetRecorder(
        tmp_path / "dataset", flight_id="recording-fixture", map_sha256="c" * 64,
        maximum_bytes=1024 * 1024, **options,
    )
    return recorder


# 功能：
#   提供只用于摘要和文件生命周期测试的样本，不将占位字节当作解码图像。
# 输入：
#   recorder：本例记录器。
#   overrides：要故障注入的字段。
# 输出：
#   record：接受的记录或频率限制返回的空值。
def _record(recorder, **overrides):
    sample = dict(
        rgb_png=b"opaque hash fixture, not a decoded PNG", frame=_frame(),
        sensor_snapshot=_snapshot(), recorded_at_unix_ms=1000,
        recorded_at_monotonic_seconds=10.1, state={},
    )
    sample.update(overrides)
    record = recorder.record(**sample)
    return record


# 功能：
#   非有限、布尔和非数值采样间隔必须在创建输出目录前拒绝。
# 输入：
#   tmp_path：隔离输出根。
#   period：非法策略值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("period", [float("nan"), True, "0.1"])
def test_invalid_period_creates_no_directory(tmp_path, period):
    with pytest.raises(ValueError):
        _recorder(tmp_path, minimum_period_seconds=period)
    assert not (tmp_path / "dataset").exists()


# 功能：
#   记录拒绝非法时间与状态，不先写图像或伪造成功计数。
# 输入：
#   tmp_path：隔离输出根。
#   overrides：非法时间或 JSON 字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("overrides", [
    {"recorded_at_monotonic_seconds": float("inf")},
    {"recorded_at_monotonic_seconds": None},
    {"recorded_at_unix_ms": True},
    {"rgb_sample_monotonic_seconds": 11.0},
    {"state": {"bad": float("nan")}},
    {"state": {1: "not a string key"}},
])
def test_invalid_sample_does_not_publish(tmp_path, overrides):
    recorder = _recorder(tmp_path)
    with pytest.raises(ValueError):
        _record(recorder, **overrides)
    assert recorder.summary()["record_count"] == 0
    assert list(recorder.rgb_root.iterdir()) == []


# 功能：
#   时钟回退应明确拒绝，不被最小采样间隔静默吞掉。
# 输入：
#   tmp_path：隔离输出根。
# 输出：
#   None：不返回业务数据。
def test_recording_clock_reversal_is_not_a_rate_skip(tmp_path):
    recorder = _recorder(tmp_path)
    _record(recorder)
    with pytest.raises(ValueError, match="CLOCK"):
        _record(recorder, recorded_at_monotonic_seconds=10.05)


# 功能：
#   独占图像发布不得覆盖另一发布者已经占用的目标。
# 输入：
#   tmp_path：隔离输出根。
# 输出：
#   None：不返回业务数据。
def test_atomic_image_publication_preserves_existing_file(tmp_path):
    recorder = _recorder(tmp_path)
    path = recorder.rgb_root / ("e" * 64 + ".png")
    path.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        recorder._atomic_bytes(path, b"replacement")
    assert path.read_bytes() == b"preserved"


# 功能：
#   哈希链记录被外部替换时停止追加，保留新文件和失败现场。
# 输入：
#   tmp_path：隔离输出根。
# 输出：
#   None：不返回业务数据。
def test_record_file_replacement_latches_failure(tmp_path):
    recorder = _recorder(tmp_path)
    _record(recorder)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(b"external record\n")
    os.replace(replacement, recorder.records_path)
    with pytest.raises((ValueError, RuntimeError), match="DRIFT|CHANGED"):
        _record(recorder, recorded_at_monotonic_seconds=10.5)
    assert recorder.records_path.read_bytes() == b"external record\n"
    assert recorder.summary()["issue_code"]
    assert recorder.summary()["record_count"] == 1


# 功能：
#   写入中断后不能继续扩展半条记录或继续宣称数据集完整。
# 输入：
#   tmp_path：隔离输出根。
#   monkeypatch：文件同步故障注入。
# 输出：
#   None：不返回业务数据。
def test_sync_failure_prevents_further_records(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    _record(recorder)

    # 功能：
    #   在证据追加后的同步点抛出受控错误。
    # 输入：
    #   descriptor：当前文件描述符。
    # 输出：
    #   None：不返回业务数据。
    def fail_sync(descriptor):
        raise OSError("injected sync failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_sync)
        with pytest.raises(OSError):
            _record(recorder, recorded_at_monotonic_seconds=10.5)
    bytes_after_failure = recorder.records_path.read_bytes()
    with pytest.raises(RuntimeError, match="FAILED"):
        _record(recorder, recorded_at_monotonic_seconds=11.0)
    assert recorder.records_path.read_bytes() == bytes_after_failure
    assert recorder.summary()["record_count"] == 1
    assert recorder.summary()["issue_code"]


# 功能：
#   去重图像被改成异常大文件时必须有界拒绝读取。
# 输入：
#   tmp_path：隔离输出根。
#   monkeypatch：监视无界读取入口。
# 输出：
#   None：不返回业务数据。
def test_deduplication_does_not_use_unbounded_reads(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    record = _record(recorder)
    path = recorder.root / record.rgb_relative_path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record.rgb_sha256
    monkeypatch.setattr(Path, "read_bytes", lambda path: pytest.fail("unbounded read"))
    assert _record(recorder, recorded_at_monotonic_seconds=10.5) is not None


# 功能：
#   线程提交失败不推进来源和提交计数，同一观测之后仍可正常提交。
# 输入：
#   monkeypatch：线程池提交失败注入。
# 输出：
#   None：不返回业务数据。
def test_learning_submission_failure_preserves_retry(monkeypatch):
    request, command = request_fixture()
    observer = LearningObservationRecorder(lambda record: True)

    # 功能：
    #   模拟线程池在接受任务前失败。
    # 输入：
    #   arguments：原提交参数。
    # 输出：
    #   None：不返回业务数据。
    def failed_submit(*arguments):
        raise RuntimeError("injected submission failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(observer._executor, "submit", failed_submit)
            with pytest.raises(RuntimeError):
                observer.submit(request, command)
        assert observer.summary()["submitted"] == 0
        assert observer.submit(request, command)
    finally:
        observer.close()


# 功能：
#   关闭等待只接受有限非负秒数，不让 NaN 或隐式转换破坏退出语义。
# 输入：
#   timeout：非法等待值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), True, -1.0])
def test_learning_rejects_invalid_close_budget(timeout):
    observer = LearningObservationRecorder(lambda record: True)
    try:
        with pytest.raises(ValueError):
            observer.close(timeout_seconds=timeout)
    finally:
        observer.close()


# 功能：
#   输入整数单调时刻经模型转为浮点后，记录摘要仍必须绑定实际回读表示。
# 输入：
#   tmp_path：隔离数据集根。
# 输出：
#   None：不返回业务数据。
def test_integer_clock_hashes_the_serialized_record(tmp_path):
    recorder = _recorder(tmp_path)
    record = _record(recorder, recorded_at_monotonic_seconds=11)
    payload = record.model_dump(mode="json")
    digest = payload.pop("record_sha256")
    assert digest == sha256_json(payload)
    summary = recorder.summary()
    assert summary["bytes_written"] == sum(
        path.stat().st_size for path in recorder.root.rglob("*") if path.is_file()
    )
    assert summary["bytes_accounting_complete"] is True


# 功能：
#   随机暂存名若发生碰撞，不得修改或删除非本次创建的文件。
# 输入：
#   tmp_path：隔离数据集根。
#   monkeypatch：固定随机名以注入碰撞。
# 输出：
#   None：不返回业务数据。
def test_temporary_collision_preserves_foreign_content(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path)
    path = recorder.rgb_root / ("f" * 64 + ".png")
    temporary = path.with_name(f".{path.name}.collision.tmp")
    temporary.write_bytes(b"foreign")
    monkeypatch.setattr(dataset_module, "uuid4", lambda: SimpleNamespace(hex="collision"))
    with pytest.raises(FileExistsError):
        recorder._atomic_bytes(path, b"ours")
    assert temporary.read_bytes() == b"foreign"
    assert not path.exists()


# 功能：
#   文件路径中的链接不得被解析后当成普通数据集目录使用。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：不返回业务数据。
def test_dataset_rejects_linked_parent(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links unavailable for this test account")
    with pytest.raises(ValueError, match="LINK"):
        RuntimeMultimodalDatasetRecorder(
            link / "dataset", flight_id="fixture", map_sha256="a" * 64,
        )
    assert not (target / "dataset").exists()


# 功能：
#   调用方修改已接受样本或接收的返回模型，不得追溯改变记录文件和哈希链基线。
# 输入：
#   tmp_path：隔离数据集根。
# 输出：
#   None：不返回业务数据。
def test_record_owns_mutable_state(tmp_path):
    recorder = _recorder(tmp_path)
    state = {"phase": {"name": "TRACK"}}
    record = _record(recorder, state=state)
    digest = record.record_sha256
    state["phase"]["name"] = "changed"
    assert record.state["phase"]["name"] == "TRACK"
    record.state["phase"]["name"] = "changed returned record"
    record.record_sha256 = "0" * 64
    next_record = _record(recorder, recorded_at_monotonic_seconds=10.5)
    assert next_record.previous_record_sha256 == digest


# 功能：
#   坐标候选开关必须是明确的 False，不接受 0 或空值作为同义授权。
# 输入：
#   flag：非法连续控制模式标记。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("flag", [0, None, ""])
def test_learning_mode_requires_explicit_false(flag):
    request, command = request_fixture()
    observer = LearningObservationRecorder(lambda record: True)
    try:
        with pytest.raises(ValueError, match="COORDINATE_CANDIDATES"):
            observer.submit(replace(request, include_candidate_paths=flag), command)
    finally:
        observer.close()


# 功能：
#   工作自身抛出的 TimeoutError 不应误报为关闭等待超时，也不泄露错误中的状态内容。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_learning_worker_timeout_is_not_drain_timeout():
    request, command = request_fixture()
    observer = LearningObservationRecorder(lambda record: True)

    # 功能：
    #   注入视觉处理自身的超时，并在错误中夹带应被屏蔽的测试标记。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def fail_visual():
        raise TimeoutError("private-fixture-state")

    assert observer.submit(request, command, prepare_visual=fail_visual)
    summary = observer.close()
    assert summary["issue_code"] == "LEARNING_OBSERVATION_FAILED:TimeoutError"
    assert not summary["complete"]
    assert "private-fixture-state" not in str(summary)
