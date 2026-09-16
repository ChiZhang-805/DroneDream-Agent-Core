"""Depth-worker boundary tests; no simulator, actuator or model API is started."""

import contextlib
import errno
import json
import threading
from types import SimpleNamespace

import pytest
from test_native_flight_state import _identity

import scripts.runtime_depth_safety_worker as worker
from dronedream_agent_core.contracts import Vector3


# 功能：
#   验证要求模型授权时，缺少指令不能悄悄退回确定性路线目标。
# 输入：
#   directive：缺失或授权类型不正确的模型指令。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("directive", [None,
    SimpleNamespace(model_navigation_authorized="false", target_m=Vector3(x=9, y=0, z=1)),
    SimpleNamespace(model_navigation_authorized=1, target_m=Vector3(x=9, y=0, z=1)),
])
def test_missing_or_untyped_model_authority_cannot_select_route_motion(directive):
    position = Vector3(x=0, y=0, z=1)
    target = worker._navigation_target_for_control_authority(
        route_target=Vector3(x=9, y=0, z=1), current_position=position,
        model_directive=directive, require_model_control_authority=True,
    )
    assert target == position


# 功能：
#   验证用于局部控制的尺寸不接受非有限数、布尔或不可表示整数。
# 输入：
#   value：非法步长或体素分辨率。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf"),
                                  pytest.param(10**1000, id="oversized-dimension")])
def test_controller_dimensions_are_finite_numbers(value):
    for field in ("maximum_step_m", "world_resolution_m"):
        values = {"maximum_step_m": .9, "world_resolution_m": .25}
        values[field] = value
        with pytest.raises(ValueError):
            worker._controller_step_for_profile(profile="precision", **values)


# 功能：
#   验证控制工况收紧算法不把布尔或字符串当成实测距离、速度和加速度。
# 输入：
#   value：非法物理量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf"),
                                  pytest.param(10**1000, id="oversized-measurement")])
def test_control_profile_measurements_are_strict(value):
    for field in ("observed_goal_distance_m", "observed_speed_mps", "maximum_acceleration_mps2"):
        values = {"observed_goal_distance_m": 1., "observed_speed_mps": .2,
                  "maximum_acceleration_mps2": 1.}
        values[field] = value
        with pytest.raises(ValueError):
            worker._effective_control_profile(requested_profile="cruise",
                                              action_checkpoint_goal=False, **values)


# 功能：
#   验证普通发布失败后只清理自身暂存文件，且不破坏原有目标。
# 输入：
#   tmp_path：独立临时目录。
#   monkeypatch：替换底层重命名以模拟不可重试故障。
# 输出：
#   None：不返回业务数据。
def test_failed_atomic_publication_cleans_owned_temporary(tmp_path, monkeypatch):
    target = tmp_path / "health.json"
    target.write_bytes(b"original")

    # 功能：
    #   模拟文件发布阶段的不可重试 I/O 故障。
    # 输入：
    #   source、destination：暂存及目标路径。
    # 输出：
    #   None：不返回业务数据。
    def fail_replace(source, destination):
        raise OSError(errno.EIO, "test publication fault")

    monkeypatch.setattr(worker.os, "replace", fail_replace)
    with pytest.raises(OSError):
        worker._atomic_json(target, {"healthy": True})
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


# 功能：
#   验证非有限 JSON 在创建目标或暂存文件前就被拒绝。
# 输入：
#   tmp_path：独立临时目录。
# 输出：
#   None：不返回业务数据。
def test_atomic_json_rejects_nonfinite_values_before_filesystem_changes(tmp_path):
    target = tmp_path / "new-directory" / "health.json"
    with pytest.raises(ValueError):
        worker._atomic_json(target, {"age": float("nan")})
    assert not target.parent.exists()


# 功能：
#   验证原生身份回读拒绝重复键，不接受 JSON 解析器静默保留的最后一个值。
# 输入：
#   tmp_path：独立临时目录。
# 输出：
#   None：不返回业务数据。
def test_native_identity_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "identity.json"
    content = json.dumps(_identity())
    path.write_text('{"updated_at_unix_ms": 999,' + content[1:], encoding="utf-8")
    with pytest.raises(ValueError):
        worker._NativeIdentityTracker().evaluate(path, now_unix_ms=1000)


# 功能：
#   验证已验证原生身份内容由跟踪器独立持有，不随调用方后续变更而漂移。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_tracker_owns_admitted_packet():
    payload = _identity()
    tracker = worker._NativeIdentityTracker()
    tracker.admit_payload(payload, now_unix_ms=1000)
    original = json.dumps(tracker.last_validated_payload, sort_keys=True)
    payload["updated_at_unix_ms"] = 999999
    assert json.dumps(tracker.last_validated_payload, sort_keys=True) == original


# 功能：
#   验证重试预算必须为有限非负秒数，不能因 NaN 等值失去时间边界。
# 输入：
#   tmp_path：独立临时目录。
#   value：非法重试预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf"), -1])
def test_replace_retry_budget_rejects_invalid_numbers(tmp_path, value):
    source = tmp_path / "source"
    source.write_bytes(b"owned")
    with pytest.raises(ValueError):
        worker._replace_with_bounded_retry(source, tmp_path / "target", timeout_seconds=value)
    assert source.read_bytes() == b"owned"


# 功能：
#   验证排队容量不能把布尔、浮点和无界值当作有效路径数。
# 输入：
#   tmp_path：独立临时目录。
#   value：非法容量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, 1.5, "2", None, 0, float("inf")])
def test_snapshot_capacity_is_strict(tmp_path, value):
    with pytest.raises(ValueError):
        worker._LatestRuntimeSnapshotWriter(tmp_path / "summary.json", maximum_pending_paths=value)
    assert not list(tmp_path.iterdir())


# 功能：
#   验证后台 PNG 解码的类型错误可见且阻止采集成功，不允许线程悄悄退出。
# 输入：
#   tmp_path：独立临时目录。
#   monkeypatch：替换图像编码器以注入类型故障。
# 输出：
#   None：不返回业务数据。
def test_dataset_type_error_cannot_look_complete(tmp_path, monkeypatch):
    fault_seen = threading.Event()

    # 功能：
    #   模拟后台编码器遇到不兼容图像类型。
    # 输入：
    #   image：用于触发失败的图像。
    # 输出：
    #   None：不返回业务数据。
    def fail_encoding(image):
        fault_seen.set()
        raise TypeError("injected image type fault")

    monkeypatch.setattr(worker, "_gazebo_image_png", fail_encoding)
    recorder = SimpleNamespace(root=tmp_path, summary=lambda: {}, record=lambda **values: None)
    writer = worker._LatestOnlyDatasetWriter(
        recorder, semantic_label_map_sha256=None, semantic_label_class_ids=frozenset())
    try:
        assert writer.submit(rgb_image=object(), frame=object(), sensor_snapshot=object(),
            recorded_at_unix_ms=1000, recorded_at_monotonic_seconds=1.,
            rgb_sample_monotonic_seconds=1., semantic_image=None,
            semantic_sample_monotonic_seconds=None, state={})
        assert fault_seen.wait(1.)
    finally:
        summary = writer.close(timeout_seconds=1.)
    assert summary["writer_complete"] is False
    assert summary["writer_thread_alive"] is False
    assert summary["writer_thread_finished"] is True
    assert "TypeError" in summary["writer_issue"]
    assert summary["writer_submitted_count"] == 1
    assert summary["writer_completed_count"] == 0


# 功能：
#   验证创建后发生异常时，退出栈仍会清理已登记资源；正常关闭不会被重复执行。
# 输入：
#   early_close：是否在异常发生前已经正常关闭资源。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("early_close", [False, True])
def test_registered_resource_closes_once_on_initialization_failure(early_close):
    closes = []
    resource = SimpleNamespace(close=lambda **options: closes.append(options) or {"complete": True})
    with (pytest.raises(ValueError, match="later initialization"),
          contextlib.ExitStack() as cleanup):
        close_once = worker._register_resource_close(cleanup, resource, timeout_seconds=.5)
        if early_close:
            assert close_once() == {"complete": True}
        raise ValueError("later initialization failed")
    assert closes == [{"timeout_seconds": .5}]


# 功能：
#   验证数据集线程尚在排空时，关闭回执不能把等待超时报告为成功。
# 输入：
#   tmp_path：独立临时目录。
#   monkeypatch：用事件控制编码器的完成时间。
# 输出：
#   None：不返回业务数据。
def test_dataset_drain_timeout_is_persistent(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    # 功能：
    #   保持一次编码未完成直到测试释放，不依赖不确定的短睡眠制造竞态。
    # 输入：
    #   image：待编码的测试消息。
    # 输出：
    #   png：用于记录的测试字节。
    def blocked_encoding(image):
        entered.set()
        assert release.wait(2.)
        return b"png"

    monkeypatch.setattr(worker, "_gazebo_image_png", blocked_encoding)
    recorder = SimpleNamespace(root=tmp_path, summary=lambda: {}, record=lambda **values: None)
    writer = worker._LatestOnlyDatasetWriter(
        recorder, semantic_label_map_sha256=None, semantic_label_class_ids=frozenset())
    try:
        assert writer.submit(rgb_image=object(), frame=object(), sensor_snapshot=object(),
            recorded_at_unix_ms=1000, recorded_at_monotonic_seconds=1.,
            rgb_sample_monotonic_seconds=1., semantic_image=None,
            semantic_sample_monotonic_seconds=None, state={})
        assert entered.wait(1.)
        summary = writer.close(timeout_seconds=0.)
        assert summary["writer_complete"] is False
        assert summary["writer_issue"] == "DATASET_WRITER_DRAIN_TIMEOUT"
    finally:
        release.set()
        final = writer.close(timeout_seconds=1.)
    assert final["writer_thread_alive"] is False
    assert final["writer_complete"] is False
