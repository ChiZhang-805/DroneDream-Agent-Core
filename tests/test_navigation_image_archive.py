"""Navigation image admission stays bounded without starving diagnostic publication."""

import ast
import inspect
import threading
from types import SimpleNamespace

import pytest

import scripts.runtime_depth_safety_worker as worker
from dronedream_agent_core.gazebo_adapter import _runtime_snapshot_writer_complete
from dronedream_agent_core.training.px4_environment import require_visual_control_source


# 功能：验证显式数值采集不能渗入生产视觉控制；输入：工作器参数；输出：拒绝或保留。
@pytest.mark.parametrize("override", [
    {"simulation_training_channel": None}, {"local_navigation_provider": "local-policy"},
    {"require_model_control_authority": False}, {"multimodal_dataset_root": None},
    {"rgb_topic": None}, {"native_camera_epoch": None},
    {"render_scene_epoch": "replica"}, {"record_learning_observations": True},
    {},
])
def test_recording_only_rgb_is_explicit_native_learner_only(override):
    config = dict(recording_only_rgb=True, simulation_training_channel="learner",
                  local_navigation_provider="simulation-training", require_model_control_authority=True,
                  multimodal_dataset_root="media", rgb_topic="native-camera",
                  native_camera_epoch="bound-clock", render_scene_epoch=None,
                  record_learning_observations=False)
    if override:
        with pytest.raises(ValueError, match="RECORDING_ONLY_RGB_REQUIRES"):
            worker._control_uses_rgb(SimpleNamespace(**(config | override)))
    else:
        assert not worker._control_uses_rgb(SimpleNamespace(**config))
    config.update(recording_only_rgb=False, local_navigation_provider="local-policy",
                  simulation_training_channel=None)
    assert worker._control_uses_rgb(SimpleNamespace(**config))


# 功能：
#   在启动仿真前拒绝缺少场景时钟的视觉学习器，保留非视觉采集和显式时钟配置。
# 输入：
#   field：独立编码器或整包视觉配置。
# 输出：
#   None：错误配置未拒绝或合法配置被误拒绝时测试失败。
@pytest.mark.parametrize("field", ["visual_encoder", "visual_package"])
def test_visual_student_requires_capture_clock_before_launch(field):
    config = SimpleNamespace(visual_encoder=None, visual_package=None, render_replica_runtime=None)
    require_visual_control_source(config)
    setattr(config, field, "bound-visual-artifact")
    with pytest.raises(ValueError, match="PX4_VISUAL_CONTROL_REQUIRES_SOURCE_CLOCK_RENDERER"):
        require_visual_control_source(config)
    config.render_replica_runtime = "explicit-renderer-verified-later"
    require_visual_control_source(config)


# 功能：原生源钟必须通过制品校验；输入：验证器替身；输出：实际调用与错误传播。
def test_native_source_clock_is_verified_before_launch(monkeypatch, tmp_path):
    from dronedream_agent_core import simulation_camera_clock
    from unittest.mock import Mock

    verifier = Mock()
    monkeypatch.setattr(simulation_camera_clock, 'validate_native_camera_clock', verifier)
    config = SimpleNamespace(visual_encoder='encoder', visual_package=None,
        render_replica_runtime=None, native_camera_clock_runtime=tmp_path)
    require_visual_control_source(config)
    assert verifier.call_args.args == (tmp_path,)
    verifier.side_effect = ValueError('CLOCK_ARTIFACT_CHANGED')
    with pytest.raises(ValueError, match='CLOCK_ARTIFACT_CHANGED'):
        require_visual_control_source(config)


# 功能：
#   在图像磁盘暂停时验证独立队列背压、诊断发布及恢复后真实文件完整性。
# 输入：
#   tmp_path、monkeypatch：隔离目录及仅暂停图像写入的替换器。
# 输出：
#   None：任何图像丢失、诊断受阻或假完整均使测试失败。
def test_full_image_queue_does_not_poison_diagnostics(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = worker._atomic_bytes

    # 功能：
    #   暂停图像发布以稳定构造磁盘背压，其他文件仍真实写入。
    # 输入：
    #   path、payload、kwargs：原子写入的目标、字节与预算。
    # 输出：
    #   None：实际文件发布完成。
    def delayed(path, payload, **kwargs):
        if path.suffix == ".png":
            entered.set()
            assert release.wait(5)
        original(path, payload, **kwargs)

    monkeypatch.setattr(worker, "_atomic_bytes", delayed)
    images = worker._LatestRuntimeSnapshotWriter(tmp_path / "images.json", maximum_pending_paths=1)
    diagnostics = worker._LatestRuntimeSnapshotWriter(tmp_path / "diagnostics.json")
    try:
        worker._persist_navigation_image(provider="local-policy", writer=images,
                                         path=tmp_path / "a.png", png=b"a")
        assert entered.wait(2)
        worker._persist_navigation_image(provider="local-policy", writer=images,
                                         path=tmp_path / "b.png", png=b"b")
        with pytest.raises(ValueError, match="MODEL_IMAGE_ARCHIVE_BACKPRESSURE"):
            worker._persist_navigation_image(provider="local-policy", writer=images,
                                             path=tmp_path / "c.png", png=b"c")
        assert images.issue is None
        assert images.summary()["rejected_count"] == 0
        assert diagnostics.submit_json(tmp_path / "health.json", {"fresh": True})
        assert diagnostics.close(timeout_seconds=2)["complete"]
        assert (tmp_path / "health.json").is_file()
        assert not (tmp_path / "c.png").exists()
    finally:
        release.set()
        summary = images.close(timeout_seconds=2)
        diagnostics.close(timeout_seconds=2)
    assert summary["complete"]
    assert (tmp_path / "a.png").read_bytes() == b"a"
    assert (tmp_path / "b.png").read_bytes() == b"b"


# 功能：
#   验证入队失败不会被忽略为已保存图像。
# 输入：
#   tmp_path：独立归档目标。
# 输出：
#   None：未显式拒绝时测试失败。
def test_rejected_navigation_archive_is_not_reported_ready(tmp_path):
    writer = SimpleNamespace(available_for_new_path=True, submit_bytes=lambda *args: False)
    with pytest.raises(ValueError, match="MODEL_IMAGE_ARCHIVE_REJECTED"):
        worker._persist_navigation_image(provider="local-policy", writer=writer,
                                         path=tmp_path / "frame.png", png=b"frame")


# 功能：
#   验证复合回执独立检查模型图像的计数与关闭状态，不相信外层完整标记。
# 输入：
#   tmp_path：真实后台队列回执目录。
# 输出：
#   None：缺失、嵌套或不完整图像回执被放行时测试失败。
def test_composite_archive_receipt_checks_image_conservation(tmp_path):
    writer = worker._LatestRuntimeSnapshotWriter(tmp_path / "summary.json")
    child = writer.close()
    aggregate = {**child, "schema_version": "dronedream.runtime-snapshot-writer.v2",
                 "model_image_writer": child}
    assert _runtime_snapshot_writer_complete(aggregate)
    for bad in (None, {}, {**child, "complete": False},
                {**child, "submitted_count": 1}, aggregate):
        assert not _runtime_snapshot_writer_complete({**aggregate, "model_image_writer": bad})


# 功能：
#   核对缓存复用和后台预处理共用实际归档时钟，不以是否新编码代替入队结果。
# 输入：
#   无：读取当前工作函数的语法树。
# 输出：
#   None：归档时钟提前推进、选择了诊断写入器或遗漏缓存重试时测试失败。
def test_cached_image_archive_admission_precedes_persisted_clock():
    tree = ast.parse(inspect.getsource(worker._run_worker))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and ast.unparse(node.test) == "new_image"]
    blocks = [node for node in matches if "_persist_navigation_image" in ast.unparse(node)]
    assert len(blocks) == 1
    assert "writer=model_image_writer" in ast.unparse(blocks[0].body[0])
    admitted = blocks[0].body[1]
    assert isinstance(admitted, ast.If)
    assert ast.unparse(admitted.test) == "image_archive_admitted"
    assert ast.unparse(admitted.body[0]).startswith("last_model_image_time =")
    assert "model_image_archive_skipped += 1" in ast.unparse(admitted)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(ast.unparse(target) == "new_image" for target in node.targets)]
    assert len(assignments) == 1
    assert "last_model_image_time" in ast.unparse(assignments[0].value)


# 功能：验证本地内存推理的归档可降级而训练不能借用此例外；输入：队列状态；输出：真实入队标志。
@pytest.mark.parametrize("available,accepted", [(False, False), (True, False), (True, True)])
def test_embedded_local_image_does_not_depend_on_archive_queue(tmp_path, available, accepted):
    calls = []
    writer = SimpleNamespace(available_for_new_path=available,
        submit_bytes=lambda *args: calls.append(args) or accepted)
    admitted = worker._persist_navigation_image(provider="local-policy", writer=writer,
        path=tmp_path / "frame.png", png=b"frame", allow_embedded_only=True)
    assert admitted is (available and accepted)
    assert len(calls) == int(available)
    assert not (tmp_path / "frame.png").exists()  # 入队不是落盘。
    with pytest.raises(ValueError, match="REQUIRES_LOCAL_POLICY"):
        worker._persist_navigation_image(provider="simulation-training", writer=writer,
            path=tmp_path / "frame.png", png=b"frame", allow_embedded_only=True)


# 功能：真实后台归档故障不能污染已认证的内存像素，也不能伪报归档完整。
def test_failed_archive_can_degrade_but_remains_incomplete(tmp_path, monkeypatch):
    original = worker._atomic_bytes

    def broken_image(path, payload, **kwargs):
        if path.suffix == ".png":
            raise OSError("injected archive disk failure")
        original(path, payload, **kwargs)

    monkeypatch.setattr(worker, "_atomic_bytes", broken_image)
    images = worker._LatestRuntimeSnapshotWriter(tmp_path / "summary.json")
    assert worker._persist_navigation_image(provider="local-policy", writer=images,
        path=tmp_path / "first.png", png=b"actual pixels", allow_embedded_only=True)
    summary = images.close(timeout_seconds=2)
    assert summary["complete"] is False
    assert not worker._persist_navigation_image(provider="local-policy", writer=images,
        path=tmp_path / "next.png", png=b"new actual pixels", allow_embedded_only=True)
    with pytest.raises(ValueError, match="BACKPRESSURE"):
        worker._persist_navigation_image(provider="simulation-training", writer=images,
            path=tmp_path / "next.png", png=b"new actual pixels")
