"""Runtime boundary regressions; these checks do not simulate or qualify a flight."""

import ast
import inspect
import os
from types import SimpleNamespace

import pytest

from dronedream_agent_core import contracts
from dronedream_agent_core import gazebo_adapter as adapter
from dronedream_plugin_sdk.protocol import encode_json


# 功能：
#   核对真实适配器生成的嵌套字典字段与下游严格模型一致，不执行或伪造一次飞行。
# 输入：
#   node：源代码中的字典节点；model：对应的跨进程模型。
# 输出：
#   无。
def _assert_declared_fields(node, model):
    assert isinstance(node, ast.Dict)
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            continue
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        assert key.value in model.model_fields, (model.__name__, key.value)
        annotation = model.model_fields[key.value].annotation
        candidates = (annotation, *getattr(annotation, "__args__", ()))
        child = next(
            (
                item
                for item in candidates
                if isinstance(item, type) and issubclass(item, contracts.StrictModel)
            ),
            None,
        )
        if child is not None and isinstance(value, ast.Dict):
            _assert_declared_fields(value, child)


# 功能：
#   防止适配器新增证据或门控字段却未更新消费者模型的集成错误。
# 输入：
#   无。
# 输出：
#   无。
def test_actual_adapter_output_fields_have_strict_consumers():
    source = ast.parse(inspect.getsource(inspect.unwrap(adapter.run_px4_gazebo_track)))
    matched = 0
    for node in ast.walk(source):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "evidence" for target in node.targets
            ) and isinstance(node.value, ast.Dict):
                _assert_declared_fields(node.value, contracts.Px4GazeboRunEvidence)
                matched += 1
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "gates"
                    and isinstance(target.slice, ast.Constant)
                ):
                    assert target.slice.value in contracts.Px4GazeboGates.model_fields
            if isinstance(node.value, ast.Dict) and any(
                isinstance(target, ast.Name) and target.id == "gates" for target in node.targets
            ):
                _assert_declared_fields(node.value, contracts.Px4GazeboGates)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "gates"
            and node.func.attr == "update"
        ):
            _assert_declared_fields(node.args[0], contracts.Px4GazeboGates)
    assert matched == 1


# 功能：
#   完整证据允许读取，损坏尾部、重复键、非有限值或非对象均不能只留下成功前缀。
# 输入：
#   tmp_path：临时目录；tail：追加的损坏记录。
# 输出：
#   无。
@pytest.mark.parametrize("tail", [b'{"ok":true}', b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b"[]\n"])
def test_evidence_stream_rejects_partial_or_invalid_tail(tmp_path, tail):
    path = tmp_path / "records.jsonl"
    adapter._write_bytes_atomic(path, b'{"ok":true}\n' + tail)
    with pytest.raises(ValueError):
        adapter._runtime_rows(path)
    adapter._write_bytes_atomic(path, b'{"ok":true}\n')
    assert adapter._runtime_rows(path) == [{"ok": True}]
    assert adapter._runtime_rows(tmp_path / "missing.jsonl") == []


# 功能：
#   非法时间、进度编号和字符串授权不能延长运行生命期。
# 输入：
#   tmp_path：临时目录；change：待替换的遥测字段。
# 输出：
#   无。
@pytest.mark.parametrize(
    "change",
    [
        {"updated_at_unix_ms": 1001},
        {"updated_at_unix_ms": True},
        {"schedule_index": "2"},
        {"model_navigation_authorized": "false"},
    ],
)
def test_progress_requires_fresh_typed_observations(tmp_path, change):
    payload = {
        "updated_at_unix_ms": 999,
        "schedule_index": 2,
        "state": "tracking",
        "model_navigation_authorized": True,
        **change,
    }
    path = tmp_path / "progress.json"
    adapter._write_json(path, payload)
    assert adapter._fresh_tracking_progress_state(path, now_unix_ms=1000) is None


# 功能：
#   启动失败也恢复进程环境，并拒绝同解释器嵌套仿真串线。
# 输入：
#   monkeypatch：临时环境补丁。
# 输出：
#   无。
def test_transport_environment_is_restored_after_failure(monkeypatch):
    monkeypatch.setenv("GZ_PARTITION", "original")
    monkeypatch.delenv("GZ_IP", raising=False)

    # 功能：
    #   制造运行中异常，检查外层负责恢复环境。
    # 输入：
    #   无。
    # 输出：
    #   无。
    @adapter._isolate_transport_environment
    def fail():
        os.environ["GZ_PARTITION"] = "temporary"
        os.environ["GZ_IP"] = "127.0.0.1"
        with pytest.raises(adapter.SimulationRuntimeError, match="ALREADY_RUNNING"):
            fail()
        raise ValueError("primary failure")

    for _ in range(2):
        with pytest.raises(ValueError, match="primary failure"):
            fail()
        assert os.environ["GZ_PARTITION"] == "original"
        assert "GZ_IP" not in os.environ


# 功能：
#   一个关闭动作失败不阻断其他回收，并保留原始异常身份。
# 输入：
#   无。
# 输出：
#   无。
def test_cleanup_attempts_remaining_resources_and_preserves_primary():
    attempted, errors = [], []

    # 功能：
    #   记录并制造资源关闭失败。
    # 输入：
    #   无。
    # 输出：
    #   无。
    def fail():
        attempted.append("failed")
        raise OSError("cleanup")

    adapter._cleanup_runtime_resource(errors, fail)
    adapter._cleanup_runtime_resource(errors, attempted.append, "closed")
    primary = ValueError("flight failure")
    adapter._finish_runtime_cleanup(errors, primary)
    assert attempted == ["failed", "closed"]
    assert primary.__notes__ == ["Runtime cleanup failures: OSError"]
    with pytest.raises(adapter.SimulationRuntimeError, match="CLEANUP_FAILED"):
        adapter._finish_runtime_cleanup(errors, None)


# 功能：
#   当前 SDF 名称与所选机型启动脚本必须真实匹配，不可回退到学校地图或固定 4001。
# 输入：
#   tmp_path：临时资产目录。
# 输出：
#   无。
def test_world_and_airframe_are_resolved_from_selected_assets(tmp_path):
    path = tmp_path / "world.sdf"
    adapter._write_bytes_atomic(path, b'<sdf><world name="warehouse"/></sdf>')
    assert adapter._sdf_entity_name(path, "world", None) == "warehouse"
    with pytest.raises(adapter.SimulationRuntimeError, match="MISMATCH"):
        adapter._sdf_entity_name(path, "world", "school")
    scripts = tmp_path / "etc/init.d-posix/airframes"
    scripts.mkdir(parents=True)
    adapter._write_bytes_atomic(scripts / "4010_gz_custom", b"# custom\n")
    assert adapter._px4_autostart_id(tmp_path, "custom") == "4010"
    with pytest.raises(adapter.SimulationRuntimeError, match="UNAVAILABLE"):
        adapter._px4_autostart_id(tmp_path, "x500")


# 功能：
#   输入资产变化必须终止本轮证据归档，不能运行后再重新计算摘要掩盖替换。
# 输入：
#   tmp_path：临时资产目录。
# 输出：
#   无。
def test_input_hashes_reject_mid_run_replacement(tmp_path):
    path = tmp_path / "asset.json"
    adapter._write_json(path, {"name": "original"})
    bindings = {path: adapter._sha256(path)}
    adapter._verify_runtime_inputs(bindings)
    adapter._write_json(path, {"name": "replacement"})
    with pytest.raises(adapter.SimulationRuntimeError, match="ASSET_CHANGED"):
        adapter._verify_runtime_inputs(bindings)


# 功能：
#   PNG 编码与模型图像输入使用同一套严格布局规则，不静默修正错误行跨度或类型。
# 输入：
#   change：待替换的图像字段。
# 输出：
#   无。
@pytest.mark.parametrize("change", [{"width": True}, {"step": 1}, {"data": 3}])
def test_image_evidence_rejects_corrupt_layout(change):
    message = SimpleNamespace(
        **{
            "width": 1,
            "height": 1,
            "step": 3,
            "pixel_format_type": 3,
            "data": b"\xff\0\0",
            **change,
        }
    )
    with pytest.raises(ValueError):
        adapter._gazebo_image_png(message)
    with pytest.raises(ValueError):
        adapter._gazebo_semantic_label_png(message)


# 功能：
#   合同新增运行诊断字段必须可严格序列化，且字符串计数不能冒充控制调用次数。
# 输入：
#   无。
# 输出：
#   无。
def test_current_diagnostics_and_authority_types():
    event = contracts.LiveSafetyEvent.model_validate_json(
        encode_json(
            {
                "schema_version": "dronedream.live-safety-event.v1",
                "reason": "GAZEBO_POSE_PROCESSING_FAILED",
                "error_type": "ValueError",
            }
        ),
        strict=True,
    )
    assert event.elapsed_s is None
    with pytest.raises(ValueError):
        contracts.Px4GazeboModelControlAuthorityEvidence.model_validate_json(
            encode_json({"required": "true", "authorized_control_applied_count": "1"}), strict=True
        )


# 功能：
#   扩展参数只能调节有限恢复项，不能替换轨迹、输出目录或绕开已批准的控制链路。
# 输入：
#   arguments：包含越权、重复或非法数值的参数。
# 输出：
#   无。
@pytest.mark.parametrize(
    "arguments",
    [
        ["--track", "old.json"],
        ["--run-dir", "other"],
        ["--base-executor"],
        ["--takeoff-timeout-seconds", "nan"],
        ["--heading-policy", "measured-hold", "--heading-policy", "route-tangent-relative"],
    ],
)
def test_executor_options_cannot_replace_frozen_inputs(arguments):
    with pytest.raises(ValueError):
        adapter._validated_executor_options(arguments)


# 功能：
#   地面名称不是接触豁免；侧撞、斜面和半悬空接触均拒绝，完整上表面接触才符合条件。
# 输入：
#   无。
# 输出：
#   无。
def test_landing_contact_requires_top_surface_support():
    floor = {
        "name": "ground_floor",
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "size_x": 4.0,
        "size_y": 4.0,
        "size_z": 0.2,
    }
    assert adapter._landing_contact_from_above(floor, (0.0, 0.0, 0.19), 0.2, 0.1)
    assert not adapter._landing_contact_from_above(floor, (2.19, 0.0, 0.0), 0.2, 0.1)
    assert not adapter._landing_contact_from_above(floor, (1.9, 0.0, 0.19), 0.2, 0.1)
    assert not adapter._landing_contact_from_above(
        {**floor, "roll_rad": 0.1}, (0.0, 0.0, 0.19), 0.2, 0.1
    )


# 功能：
#   组长已退出时仍回收其进程组，避免子孙进程持有传感器和管道。
# 输入：
#   monkeypatch：进程接口补丁，仅记录信号，不接触真实系统进程。
# 输出：
#   无。
def test_exited_group_leader_does_not_skip_descendant_cleanup(monkeypatch):
    signals, waits = [], []
    process = SimpleNamespace(pid=444, poll=lambda: 0, wait=lambda timeout: waits.append(timeout))
    monkeypatch.setattr(adapter.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    adapter._terminate(process)
    assert [pid for pid, _ in signals] == [-444, -444]
    assert waits == [10, 5]
