from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from dronedream_agent_core.contracts import (
    OnboardPerceptionFrame,
    RangeRayObservation,
    RuntimeActionExecutionContract,
    RuntimeActionExecutionStep,
    RuntimeCheckpoint,
    RuntimeCheckpointRequest,
    Vector3,
)
from dronedream_agent_core.runtime_multimodal_dataset import RuntimeMultimodalDatasetRecorder
from dronedream_agent_core.runtime_sensor_contracts import (
    RuntimeMultimodalSensorSnapshot,
    RuntimeSensorStatus,
)


# 功能：
#   按明确源码路径加载测试模块并登记类型解析所需的模块名。
# 输入：
#   name：本测试使用的模块注册名。
#   path：待加载的本地源码路径。
# 输出：
#   module：已完成初始化的测试模块。
def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 功能：
#   加载当前工作树的基础飞控与检查点执行器，不调用已安装旧副本。
# 输入：
#   无。
# 输出：
#   modules：基础模块与执行器组成的二元组。
def _modules() -> tuple[ModuleType, ModuleType]:
    root = Path(__file__).parents[1]
    base = _load_module(
        "test_payload_transition_base",
        root / "runtime" / "px4_offboard_track_executor.py",
    )
    executor = _load_module(
        "test_payload_transition_executor",
        root / "scripts" / "px4_checkpoint_executor.py",
    )
    modules = base, executor
    return modules


# 功能：
#   构造接触前、载荷交接或挂载后稳定步骤，提供对应物理参数与成功证据要求。
# 输入：
#   operation：需要测试的载荷过渡操作。
# 输出：
#   step：完整类型化的运行步骤。
def _step(*, operation: str) -> RuntimeActionExecutionStep:
    precontact = operation == "precontact"
    postattach = operation == "postattach-stability"
    step = RuntimeActionExecutionStep(
        step_id="action-001",
        task_id="payload-transition",
        action=(
            "delivery.precontact-hold"
            if precontact
            else ("delivery.verify-loaded-stability" if postattach else "delivery.confirm-custody")
        ),
        target_node="pickup",
        trigger="checkpoint",
        checkpoint_id="checkpoint-001",
        runtime_executor=(
            "native.payload.precontact-hold"
            if precontact
            else (
                "native.payload.verify-loaded-stability"
                if postattach
                else "native.payload.confirm-custody"
            )
        ),
        adapter_id=(
            "runtime.payload.loaded-stability"
            if postattach
            else "runtime.payload.transition-guards"
        ),
        driver="payload-transition",
        parameters={
            "operation": operation,
            "output_topic": "/model/my_drone/takeout_payload/state",
            "payload_sdf_sha256": "a" * 64,
            "payload_mass_kg": 0.1,
            "payload_inertia_kg_m2": {
                "ixx": 0.01,
                "iyy": 0.02,
                "izz": 0.03,
                "ixy": 0.0,
                "ixz": 0.0,
                "iyz": 0.0,
            },
            "position_tolerance_m": 0.2,
            "speed_tolerance_mps": 0.15,
            "stable_window_seconds": 0.01,
            "settle_timeout_seconds": 0.2,
            "arguments": {},
        },
        required_success_evidence=(
            ["stable pre-contact hover", "no payload contact"]
            if precontact
            else (
                [
                    "loaded hover stable",
                    "post-attachment dynamics accepted",
                    "return authorized",
                ]
                if postattach
                else [
                    "payload attachment confirmed",
                    "mass and inertia update confirmed",
                    "custody state accepted",
                ]
            )
        ),
        max_attempts=1,
        fallback="hold",
        timeout_seconds=1.0,
        authority="control",
    )
    return step


# 功能：
#   验证载荷容差仅在稳定步骤完成后调整，不提前放宽空载任务或缩小已有较大容差。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_uses_bounded_payload_aware_position_tolerance() -> None:
    _, executor = _modules()
    contract = RuntimeActionExecutionContract(
        contract_id="mission-payload-settle",
        task_graph_sha256="a" * 64,
        domain_action_catalog_sha256="b" * 64,
        adapter_catalog_sha256="c" * 64,
        steps=[_step(operation="postattach-stability")],
    )

    assert executor._payload_aware_waypoint_position_tolerance_m(
        configured_tolerance_m=0.2,
        runtime_action_contract=None,
        completed_action_step_ids=set(),
    ) == pytest.approx(0.2)
    assert executor._payload_aware_waypoint_position_tolerance_m(
        configured_tolerance_m=0.2,
        runtime_action_contract=contract,
        completed_action_step_ids=set(),
    ) == pytest.approx(0.2)
    assert executor._payload_aware_waypoint_position_tolerance_m(
        configured_tolerance_m=0.2,
        runtime_action_contract=contract,
        completed_action_step_ids={"action-001"},
    ) == pytest.approx(0.25)
    assert executor._payload_aware_waypoint_position_tolerance_m(
        configured_tolerance_m=0.3,
        runtime_action_contract=contract,
        completed_action_step_ids={"action-001"},
    ) == pytest.approx(0.3)


# 功能：
#   用实际记录器生成机载 RGB 证据，验证读取端核对记录摘要、图像摘要和来源健康。
# 输入：
#   monkeypatch：把测试相机证据入口指向隔离目录。
#   tmp_path：隔离机载数据集目录。
# 输出：
#   None：不返回业务数据。
def test_simulation_camera_action_uses_fresh_hash_bound_onboard_rgb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, executor = _modules()
    dataset = tmp_path / "multimodal-dataset"
    recorder = RuntimeMultimodalDatasetRecorder(
        dataset,
        flight_id="payload-camera-test",
        map_sha256="a" * 64,
        maximum_bytes=1024 * 1024,
        minimum_period_seconds=0.1,
    )
    observed_at_unix_ms = 10_000
    frame = OnboardPerceptionFrame(
        sensor_id="oakd-lite-depth",
        sequence=1,
        observed_at_unix_ms=observed_at_unix_ms,
        localization_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        localization_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.01,
        range_rays=[
            RangeRayObservation(
                origin_m=Vector3(x=0.0, y=0.0, z=1.0),
                endpoint_m=Vector3(x=1.0, y=0.0, z=1.0),
                hit=False,
                confidence=1.0,
                observed_at_monotonic_seconds=10.0,
            )
        ],
    )
    snapshot = RuntimeMultimodalSensorSnapshot(
        captured_at_monotonic_seconds=10.0,
        contract_set_sha256="b" * 64,
        ready_for_motion=True,
        statuses=[
            RuntimeSensorStatus(
                sensor_id="oakd-lite-depth",
                modality="depth-camera",
                required_for_motion=True,
                latest_sequence=1,
                sample_age_seconds=0.01,
                transport_latency_seconds=0.01,
                quality=1.0,
                coverage=1.0,
                health="healthy",
                payload_sha256="c" * 64,
            ),
            RuntimeSensorStatus(
                sensor_id="oakd-lite-forward-rgb",
                modality="rgb-camera",
                required_for_motion=False,
                latest_sequence=1,
                sample_age_seconds=0.01,
                transport_latency_seconds=0.01,
                quality=1.0,
                coverage=1.0,
                health="healthy",
                payload_sha256="d" * 64,
            ),
        ],
        active_sensor_ids=["oakd-lite-depth", "oakd-lite-forward-rgb"],
    )
    record = recorder.record(
        rgb_png=b"onboard-rgb-frame",
        frame=frame,
        sensor_snapshot=snapshot,
        recorded_at_unix_ms=observed_at_unix_ms,
        recorded_at_monotonic_seconds=10.0,
        state={"phase": "CHECKPOINT"},
    )
    assert record is not None
    monkeypatch.setenv("DRONEDREAM_SIMULATION_ONBOARD_RGB_DATASET", str(dataset))

    capture = executor._verified_simulation_onboard_rgb_capture(
        {"command": "take_photo", "component_id": 100},
        now_unix_ms=10_500,
    )

    assert capture["confirmed"] is True
    assert capture["transport"] == "gazebo-onboard-rgb-evidence"
    assert capture["frame_sha256"] == record.rgb_sha256
    assert capture["record_sha256"] == record.record_sha256


# 功能：
#   验证最新一条完整记录损坏时明确失败，不回退旧图或伪造相机确认。
# 输入：
#   monkeypatch：设置本次隔离数据集路径。
#   tmp_path：存放损坏记录的测试目录。
# 输出：
#   None：不返回业务数据。
def test_simulation_camera_action_rejects_invalid_complete_latest_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, executor = _modules()
    dataset = tmp_path / "multimodal-dataset"
    dataset.mkdir()
    (dataset / "records.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DRONEDREAM_SIMULATION_ONBOARD_RGB_DATASET", str(dataset))

    with pytest.raises(RuntimeError, match="simulation onboard RGB capture failed"):
        executor._verified_simulation_onboard_rgb_capture(
            {"command": "take_photo", "component_id": 100},
            now_unix_ms=10_500,
        )


# 功能：
#   验证接触前门控同时取得稳定悬停与未挂载状态证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_precontact_guard_proves_stable_hover_and_separation() -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    setpoint = base.Setpoint(1.0, 2.0, -3.0, 0.0)
    step = _step(operation="precontact")

    output = asyncio.run(
        executor._invoke_runtime_action_driver(
            base=base,
            client=client,
            step=step,
            setpoint=setpoint,
            rate_hz=100.0,
            runtime_interrupt_probe=None,
        )
    )

    assert output["detached"] is True
    assert executor._observed_runtime_action_evidence(step, output) == [
        "stable pre-contact hover",
        "no payload contact",
    ]


# 功能：
#   验证载荷接触前稳定等待接入实时控制刷新，不脱离当前原生状态链。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_precontact_guard_keeps_live_identity_refresh_active() -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    observed_samples = []
    refresh_count = 0

    # 功能：
    #   记录接触前门控的控制刷新次数，保留传入目标。
    # 输入：
    #   setpoint：本轮待刷新的稳定位置。
    # 输出：
    #   setpoint：本轮测试原样保留的位置。
    async def refresh_setpoint(setpoint):
        nonlocal refresh_count
        refresh_count += 1
        return setpoint

    output = asyncio.run(
        executor._invoke_runtime_action_driver(
            base=base,
            client=client,
            step=_step(operation="precontact"),
            setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
            rate_hz=100.0,
            runtime_interrupt_probe=None,
            setpoint_refresh=refresh_setpoint,
            sample_observer=observed_samples.append,
        )
    )

    assert output["stable_precontact_hover"] is True
    assert refresh_count > 0
    assert observed_samples


# 功能：
#   验证设备动作等待时本地控制刷新仍持续运行，不被设备调用占住。
# 输入：
#   tmp_path：本次终止文件的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_blocking_domain_action_refreshes_live_identity_while_holding(tmp_path: Path) -> None:
    base, executor = _modules()

    class SlowPayloadClient(base.FakeOffboardClient):
        # 功能：
        #   模拟有延迟的挂载设备确认，为外层控制维持留下观察窗口。
        # 输入：
        #   self：隔离设备替身。
        #   parameters：挂载动作参数。
        # 输出：
        #   state：模拟设备明确确认的挂载状态。
        async def execute_payload_command(self, parameters):
            await asyncio.sleep(0.08)
            return {
                "confirmed": True,
                "transport": "gazebo-service",
                "operation": parameters["operation"],
                "detached": False,
            }

    refresh_count = 0

    # 功能：
    #   累计慢速设备操作期间的控制刷新，保持测试目标不变。
    # 输入：
    #   setpoint：外层保护控制目标。
    # 输出：
    #   setpoint：本次刷新后保留的目标。
    async def refresh_setpoint(setpoint):
        nonlocal refresh_count
        refresh_count += 1
        return setpoint

    step = _step(operation="precontact").model_copy(
        update={
            "driver": "gazebo-payload",
            "parameters": {"operation": "attach"},
        }
    )
    output, interruption = asyncio.run(
        executor._await_runtime_action_while_holding(
            base=base,
            client=SlowPayloadClient(),
            setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
            step=step,
            abort_file=tmp_path / "abort.json",
            rate_hz=100.0,
            runtime_interrupt_probe=None,
            setpoint_refresh=refresh_setpoint,
        )
    )

    assert output["confirmed"] is True
    assert interruption is None
    assert refresh_count > 0


# 功能：
#   验证等待载荷归属回读时仍刷新本地控制，不能只在嵌套稳定门控内刷新。
# 输入：
#   tmp_path：终止文件的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_payload_custody_read_refreshes_live_identity_while_holding(tmp_path: Path) -> None:
    base, executor = _modules()

    class SlowCustodyClient(base.FakeOffboardClient):
        # 功能：
        #   延迟返回已挂载状态，用于观察归属回读期间的控制维持。
        # 输入：
        #   self：隔离的载荷状态替身。
        #   output_topic：设备状态主题。
        #   timeout_seconds：调用方允许的回读预算。
        # 输出：
        #   state：明确确认的挂载状态。
        async def sample_payload_state(self, output_topic, timeout_seconds):
            del output_topic, timeout_seconds
            await asyncio.sleep(0.08)
            return {
                "confirmed": True,
                "transport": "ros2-topic",
                "operation": "state",
                "detached": False,
            }

    refresh_count = 0

    # 功能：
    #   累计归属确认回读期间的保护刷新，检查慢速状态接口不会阻塞控制维持。
    # 输入：
    #   setpoint：外层提供的保护目标。
    # 输出：
    #   setpoint：本测试原样保留的目标。
    async def refresh_setpoint(setpoint):
        nonlocal refresh_count
        refresh_count += 1
        return setpoint

    output, interruption = asyncio.run(
        executor._await_runtime_action_while_holding(
            base=base,
            client=SlowCustodyClient(),
            setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
            step=_step(operation="confirm-custody"),
            abort_file=tmp_path / "abort.json",
            rate_hz=100.0,
            runtime_interrupt_probe=None,
            setpoint_refresh=refresh_setpoint,
        )
    )

    assert output["custody_state_accepted"] is True
    assert interruption is None
    assert refresh_count > 0


# 功能：
#   验证虚拟环境调用 ROS 时恢复工作空间与发行版模块路径，并解析实际样式的服务响应。
# 输入：
#   monkeypatch：隔离 ROS 环境、可执行文件发现及进程捕获。
# 输出：
#   None：不返回业务数据。
def test_ros2_domain_action_restores_ros_python_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, executor = _modules()
    captured = {}

    monkeypatch.setenv("PYTHONPATH", "/qualified/app/src")
    monkeypatch.setenv("ROS_DISTRO", "jazzy")
    monkeypatch.setenv("AMENT_PREFIX_PATH", "/qualified/ros-workspace/install")
    monkeypatch.setattr(executor.shutil, "which", lambda name: f"/opt/ros/jazzy/bin/{name}")

    # 功能：
    #   用已回收进程的字节输出模拟 ROS 响应，并记录传入的环境变量。
    # 输入：
    #   command：独立的服务调用参数。
    #   kwargs：共享进程捕获器的时限、环境与资源配置。
    # 输出：
    #   result：模拟进程退出码及原始字节输出。
    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["environment"]
        result = subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                b"response: dronedream_agent_msgs.srv.ExecuteDomainAction_Response("
                b"success=true, evidence=['recipient identity accepted'], "
                b"issue_code='', details_json='{}')"
            ),
            stderr=b"",
        )
        return result

    monkeypatch.setattr(executor, "capture_process", fake_run)
    output = asyncio.run(
        executor._execute_ros2_domain_action(
            {
                "service_name": "/dronedream/domain_actions/payload/verify_recipient",
                "service_type": "dronedream_agent_msgs/srv/ExecuteDomainAction",
                "request": {"contract_id": "mission-test"},
            }
        )
    )

    assert output["confirmed"] is True
    assert output["evidence"] == ["recipient identity accepted"]
    python_path = captured["env"]["PYTHONPATH"].split(executor.os.pathsep)
    assert "/qualified/app/src" in python_path
    assert (
        f"/qualified/ros-workspace/install/lib/python"
        f"{executor.sys.version_info.major}.{executor.sys.version_info.minor}/site-packages"
    ) in python_path
    assert (
        f"/opt/ros/jazzy/lib/python{executor.sys.version_info.major}."
        f"{executor.sys.version_info.minor}/site-packages"
    ) in python_path


# 功能：
#   验证动作超时后设备任务收到取消且已退出，不留下继续运行的后台操作。
# 输入：
#   tmp_path：隔离的终止文件目录。
# 输出：
#   None：不返回业务数据。
def test_domain_action_timeout_cancels_and_awaits_driver(tmp_path: Path) -> None:
    base, executor = _modules()

    class NeverCompletesClient(base.FakeOffboardClient):
        cancelled = False

        # 功能：
        #   模拟迟迟不返回的设备，并记录真实收到取消的状态。
        # 输入：
        #   self：持有取消标记的设备替身。
        #   parameters：本测试不消费的动作参数。
        # 输出：
        #   None：不返回业务数据。
        async def execute_payload_command(self, parameters):
            del parameters
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    client = NeverCompletesClient()
    step = _step(operation="precontact").model_copy(
        update={
            "driver": "gazebo-payload",
            "parameters": {"operation": "attach"},
            "timeout_seconds": 0.04,
        }
    )

    with pytest.raises(TimeoutError, match="runtime action timed out"):
        asyncio.run(
            executor._await_runtime_action_while_holding(
                base=base,
                client=client,
                setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
                step=step,
                abort_file=tmp_path / "abort.json",
                rate_hz=100.0,
                runtime_interrupt_probe=None,
            )
        )

    assert client.cancelled is True


# 功能：
#   验证旧准备任务带入的重复挂载次数不能重新启用整驱动重试，避免重复物理副作用。
# 输入：
#   monkeypatch：让首次挂载执行产生回读超时。
#   tmp_path：拒绝回执的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_executor_never_reenters_retired_payload_retry_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base, executor = _modules()
    calls = 0

    # 功能：
    #   累计驱动尝试次数并模拟未知挂载状态，检查执行器是否错误重试。
    # 输入：
    #   kwargs：当前挂载调用上下文。
    # 输出：
    #   None：不返回业务数据。
    async def reject_payload_attempt(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        raise TimeoutError("payload state readback unavailable")

    monkeypatch.setattr(
        executor,
        "_await_runtime_action_while_holding",
        reject_payload_attempt,
    )
    step = _step(operation="precontact").model_copy(
        update={
            "driver": "gazebo-payload",
            "parameters": {"operation": "attach"},
            # Simulate an older prepared mission. The current executor must not
            # honor its retired whole-driver retry policy.
            "max_attempts": 2,
        }
    )
    contract = RuntimeActionExecutionContract(
        contract_id="mission-retired-payload-retry",
        task_graph_sha256="a" * 64,
        domain_action_catalog_sha256="b" * 64,
        adapter_catalog_sha256="c" * 64,
        steps=[step],
    )

    with pytest.raises(RuntimeError, match="runtime domain action rejected"):
        asyncio.run(
            executor._execute_runtime_action_step(
                base=base,
                client=object(),
                setpoint=object(),
                step=step,
                contract=contract,
                run_dir=tmp_path,
                abort_file=tmp_path / "abort.json",
                rate_hz=100.0,
                runtime_interrupt_probe=None,
            )
        )

    assert calls == 1
    receipt = json.loads(
        (tmp_path / "runtime-actions" / "receipts" / "action-001.receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "rejected"
    assert receipt["attempts"] == 1


# 功能：
#   验证接触前检查发现载荷已经挂载时拒绝继续，不能把已有接触当成未接触。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_precontact_guard_fails_if_payload_is_already_attached() -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    client.payload_detached = False

    with pytest.raises(RuntimeError, match="payload contact or attachment"):
        asyncio.run(
            executor._invoke_runtime_action_driver(
                base=base,
                client=client,
                step=_step(operation="precontact"),
                setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
                rate_hz=100.0,
                runtime_interrupt_probe=None,
            )
        )


# 功能：
#   对照未挂载与已挂载状态，验证只有后者能生成载荷归属和物理绑定证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_custody_guard_requires_attachment_before_accepting_custody() -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    step = _step(operation="confirm-custody")

    with pytest.raises(RuntimeError, match="payload detached"):
        asyncio.run(
            executor._invoke_runtime_action_driver(
                base=base,
                client=client,
                step=step,
                setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
                rate_hz=100.0,
                runtime_interrupt_probe=None,
            )
        )

    client.payload_detached = False
    output = asyncio.run(
        executor._invoke_runtime_action_driver(
            base=base,
            client=client,
            step=step,
            setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
            rate_hz=100.0,
            runtime_interrupt_probe=None,
        )
    )
    assert executor._observed_runtime_action_evidence(step, output) == [
        "payload attachment confirmed",
        "mass and inertia update confirmed",
        "custody state accepted",
    ]


# 功能：
#   验证返回许可必须在载荷确已挂载且稳定悬停检查通过后产生。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_loaded_stability_gate_requires_attached_stable_hover() -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    step = _step(operation="postattach-stability")

    with pytest.raises(RuntimeError, match="payload detached"):
        asyncio.run(
            executor._invoke_runtime_action_driver(
                base=base,
                client=client,
                step=step,
                setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
                rate_hz=100.0,
                runtime_interrupt_probe=None,
            )
        )

    client.payload_detached = False
    output = asyncio.run(
        executor._invoke_runtime_action_driver(
            base=base,
            client=client,
            step=step,
            setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
            rate_hz=100.0,
            runtime_interrupt_probe=None,
        )
    )
    assert output["loaded_hover_stable"] is True
    assert executor._observed_runtime_action_evidence(step, output) == [
        "loaded hover stable",
        "post-attachment dynamics accepted",
        "return authorized",
    ]


# 功能：
#   验证遥测较慢时仍连续发送刷新后的航向，并保留实际位置观测。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_keeps_offboard_heartbeat_during_slow_telemetry() -> None:
    base, executor = _modules()

    class SlowTelemetryClient(base.FakeOffboardClient):
        # 功能：
        #   延迟返回已经稳定的位置样本，模拟遥测比控制周期更慢。
        # 输入：
        #   self：位置遥测替身。
        #   timeout_seconds：本次采样预算。
        # 输出：
        #   sample：目标位置上的静止 NED 样本。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            await asyncio.sleep(0.18)
            return base.PositionVelocityNed(
                north_m=1.0,
                east_m=2.0,
                down_m=-3.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    client = SlowTelemetryClient()
    observed_samples = []

    # 功能：
    #   保留位置但改变航向，用实际发送记录区分刷新控制与原始计划目标。
    # 输入：
    #   setpoint：原始位置与航向。
    # 输出：
    #   refreshed：航向改为十七度的测试目标。
    async def refresh_setpoint(setpoint):
        return base.Setpoint(
            north_m=setpoint.north_m,
            east_m=setpoint.east_m,
            down_m=setpoint.down_m,
            yaw_deg=17.0,
        )

    observed, position_error, speed = asyncio.run(
        executor._wait_checkpoint_stable(
            base=base,
            client=client,
            setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
            rate_hz=20.0,
            timeout_seconds=1.0,
            stable_window_seconds=0.1,
            position_tolerance_m=0.1,
            speed_tolerance_mps=0.1,
            setpoint_refresh=refresh_setpoint,
            sample_observer=observed_samples.append,
        )
    )

    assert observed.down_m == -3.0
    assert position_error == 0.0
    assert speed == 0.0
    assert len(client.setpoints) >= 6
    assert observed_samples
    assert all(setpoint.yaw_deg == 17.0 for setpoint in client.setpoints)


# 功能：
#   验证期限前进入严格位置门槛后，可以补足连续稳定窗口，不把单次达标当成稳定完成。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_finishes_stable_window_after_qualifying_before_deadline() -> None:
    base, executor = _modules()

    class LateQualifyingClient(base.FakeOffboardClient):
        sample_count = 0

        # 功能：
        #   前两次提供超限位置，随后提供满足门槛的静止样本。
        # 输入：
        #   self：持有采样次数的遥测替身。
        #   timeout_seconds：本测试不消费的采样预算。
        # 输出：
        #   sample：本次预定误差对应的 NED 样本。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            del timeout_seconds
            self.sample_count += 1
            await asyncio.sleep(0.03)
            error = 0.25 if self.sample_count < 3 else 0.05
            return base.PositionVelocityNed(
                north_m=error,
                east_m=0.0,
                down_m=0.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    client = LateQualifyingClient()
    observed, position_error, speed = asyncio.run(
        executor._wait_checkpoint_stable(
            base=base,
            client=client,
            setpoint=base.Setpoint(0.0, 0.0, 0.0, 0.0),
            rate_hz=200.0,
            timeout_seconds=0.14,
            stable_window_seconds=0.05,
            position_tolerance_m=0.1,
            speed_tolerance_mps=0.1,
        )
    )

    assert observed.north_m == pytest.approx(0.05)
    assert position_error == pytest.approx(0.05)
    assert speed == 0.0
    assert client.sample_count >= 4


# 功能：
#   验证稳定误差在路线参考系中计算，不把估计器坐标偏移误判为未到达。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_compares_signed_target_in_route_frame() -> None:
    base, executor = _modules()

    class OffsetEstimatorClient(base.FakeOffboardClient):
        # 功能：
        #   提供带固定估计器偏移的静止位置，交给测试坐标解析器恢复。
        # 输入：
        #   self：位置遥测替身。
        #   timeout_seconds：本测试不消费的采样预算。
        # 输出：
        #   sample：估计器参考系中的实际位置样本。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            del timeout_seconds
            await asyncio.sleep(0.01)
            return base.PositionVelocityNed(
                north_m=0.80,
                east_m=1.70,
                down_m=-2.60,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    observed, position_error, speed = asyncio.run(
        executor._wait_checkpoint_stable(
            base=base,
            client=OffsetEstimatorClient(),
            setpoint=base.Setpoint(1.0, 2.0, -3.0, 0.0),
            rate_hz=100.0,
            timeout_seconds=0.2,
            stable_window_seconds=0.03,
            position_tolerance_m=0.05,
            speed_tolerance_mps=0.05,
            target_frame_position_resolver=lambda sample: (
                sample.north_m + 0.20,
                sample.east_m + 0.30,
                sample.down_m - 0.40,
            ),
        )
    )

    assert observed.north_m == pytest.approx(0.80)
    assert position_error == pytest.approx(0.0)
    assert speed == 0.0


# 功能：
#   验证真实位置持续明显收敛时刷新无进展期限，最终仍须通过原位置和速度门槛。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_extends_while_exact_position_is_materially_converging() -> None:
    base, executor = _modules()

    class ConvergingClient(base.FakeOffboardClient):
        sample_count = 0

        # 功能：
        #   依次输出误差减小直至稳定的样本，用独立序列模拟实际收敛。
        # 输入：
        #   self：保存序列位置的遥测替身。
        #   timeout_seconds：本测试不消费的采样预算。
        # 输出：
        #   sample：本次收敛位置和零速度样本。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            del timeout_seconds
            self.sample_count += 1
            await asyncio.sleep(0.02)
            errors = (0.30, 0.28, 0.26, 0.24, 0.20, 0.09, 0.08, 0.08, 0.08)
            error = errors[min(self.sample_count - 1, len(errors) - 1)]
            return base.PositionVelocityNed(
                north_m=error,
                east_m=0.0,
                down_m=0.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    client = ConvergingClient()
    _, position_error, speed = asyncio.run(
        executor._wait_checkpoint_stable(
            base=base,
            client=client,
            setpoint=base.Setpoint(0.0, 0.0, 0.0, 0.0),
            rate_hz=200.0,
            timeout_seconds=0.06,
            stable_window_seconds=0.03,
            position_tolerance_m=0.1,
            speed_tolerance_mps=0.1,
        )
    )

    assert client.sample_count >= 7
    assert position_error <= 0.1
    assert speed == 0.0


# 功能：
#   验证沿本地模型目标绕行可获得有界进展时间，最终仍必须回到准确航点并稳定。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_allows_bounded_model_authorized_detour() -> None:
    base, executor = _modules()

    class DetouringClient(base.FakeOffboardClient):
        sample_count = 0
        errors = (0.15, 0.30, 0.45, 0.40, 0.30, 0.20, 0.09, 0.08, 0.08, 0.08)

        # 功能：
        #   模拟先远离航点后返回的实际轨迹，不用计划点代替测量结果。
        # 输入：
        #   self：持有绕行误差序列和采样计数的替身。
        #   timeout_seconds：本测试不消费的采样预算。
        # 输出：
        #   sample：该次绕行阶段的静止位置样本。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            del timeout_seconds
            await asyncio.sleep(0.012)
            error = self.errors[min(self.sample_count, len(self.errors) - 1)]
            self.sample_count += 1
            return base.PositionVelocityNed(
                north_m=error,
                east_m=0.0,
                down_m=0.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    client = DetouringClient()

    # 功能：
    #   提供与绕行序列相对应的本地引导点，用于验证稳定器的局部进展判定。
    # 输入：
    #   setpoint：本测试不直接追踪的原始航点。
    # 输出：
    #   target：当前绕行阶段的本地引导点。
    async def model_carrot(setpoint):
        del setpoint
        index = min(client.sample_count, len(client.errors) - 1)
        return base.Setpoint(client.errors[index], 0.0, 0.0, 0.0)

    _, position_error, speed = asyncio.run(
        executor._wait_checkpoint_stable(
            base=base,
            client=client,
            setpoint=base.Setpoint(0.0, 0.0, 0.0, 0.0),
            rate_hz=200.0,
            # Leave scheduler headroom on loaded CI/desktop hosts. The test is
            # about bounded detour progress and exact-waypoint settling, not a
            # 40 ms wall-clock benchmark.
            timeout_seconds=0.08,
            stable_window_seconds=0.025,
            position_tolerance_m=0.1,
            speed_tolerance_mps=0.1,
            setpoint_refresh=model_carrot,
        )
    )

    # The stable window is measured in real monotonic time, not sample count.
    # On a loaded host two qualified samples can legitimately span the full
    # window; eight samples still prove the detour reached its 0.45 m apex and
    # returned inside the exact-waypoint gate before stability was accepted.
    assert client.sample_count >= 8
    assert position_error <= 0.1
    assert speed == 0.0


# 功能：
#   用受控时钟验证精细控制只延长有界等待，不降低原有位置及速度验收要求。
# 输入：
#   monkeypatch：提供独立推进的单调时钟。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_precision_hard_window_keeps_strict_gates_while_converging(
    monkeypatch,
) -> None:
    base, executor = _modules()
    clock = [0.0]
    monkeypatch.setattr(executor.time, "monotonic", lambda: clock[0])

    class SlowlyConvergingClient(base.FakeOffboardClient):
        sample_count = 0

        # 功能：
        #   每次推进测试时钟并减少实际位置误差，最终停在严格门槛内。
        # 输入：
        #   self：当前采样次数。
        #   timeout_seconds：本测试不消费的采样预算。
        # 输出：
        #   sample：本次缓慢收敛的 NED 状态。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            del timeout_seconds
            self.sample_count += 1
            clock[0] += 0.01
            error = max(0.08, 0.36 - self.sample_count * 0.012)
            return base.PositionVelocityNed(
                north_m=error,
                east_m=0.0,
                down_m=0.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    client = SlowlyConvergingClient()
    _, position_error, speed = asyncio.run(
        executor._wait_checkpoint_stable(
            base=base,
            client=client,
            setpoint=base.Setpoint(0.0, 0.0, 0.0, 0.0),
            rate_hz=200.0,
            timeout_seconds=0.04,
            absolute_timeout_factor=8.0,
            stable_window_seconds=0.03,
            position_tolerance_m=0.1,
            speed_tolerance_mps=0.1,
        )
    )

    # The aircraft reaches the original strict gate only after the former 5x
    # absolute ceiling, while material progress keeps the ordinary 40 ms stall
    # deadline alive.  The extended ceiling therefore grants time, not accuracy.
    assert clock[0] > 0.20
    assert position_error <= 0.1
    assert speed == 0.0


# 功能：
#   验证持续小幅进展不能无限延长绝对期限，未达标任务最终仍超时退出。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_progress_never_moves_absolute_deadline() -> None:
    base, executor = _modules()

    class ForeverConvergingClient(base.FakeOffboardClient):
        sample_count = 0

        # 功能：
        #   提供每次略有改善但在预算内不能到达门槛的样本。
        # 输入：
        #   self：持有采样计数的替身。
        #   timeout_seconds：本测试不消费的采样预算。
        # 输出：
        #   sample：本次仍偏离目标的位置和零速度。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            del timeout_seconds
            self.sample_count += 1
            # Keep the fake sampler faster than the deliberately tiny
            # no-progress window. The test is about the immutable absolute
            # deadline, not host scheduler latency under a large test batch.
            await asyncio.sleep(0.005)
            return base.PositionVelocityNed(
                north_m=0.50 - self.sample_count * 0.005,
                east_m=0.0,
                down_m=0.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    client = ForeverConvergingClient()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="waypoint stability timeout"):
        asyncio.run(
            executor._wait_checkpoint_stable(
                base=base,
                client=client,
                setpoint=base.Setpoint(0.0, 0.0, 0.0, 0.0),
                rate_hz=200.0,
                timeout_seconds=0.04,
                stable_window_seconds=0.03,
                position_tolerance_m=0.1,
                speed_tolerance_mps=0.1,
            )
        )
    elapsed = time.monotonic() - started
    # Even repeated progress must still terminate below the immutable absolute
    # bound. Do not assume a busy Windows runner can always schedule a
    # particular number of 5 ms fake samples before that deadline.
    assert elapsed < 0.3
    assert client.sample_count >= 2


# 功能：
#   验证始终不满足位置门槛的状态不能借稳定窗口宽限而通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_waypoint_settle_grace_does_not_rescue_unqualified_vehicle() -> None:
    base, executor = _modules()

    class UnqualifiedClient(base.FakeOffboardClient):
        # 功能：
        #   持续提供同一个超出容差的位置，模拟没有实际进展。
        # 输入：
        #   self：位置遥测替身。
        #   timeout_seconds：本测试不消费的采样预算。
        # 输出：
        #   sample：位置误差始终为二十五厘米的状态。
        async def sample_position_velocity_ned(self, timeout_seconds: float):
            del timeout_seconds
            await asyncio.sleep(0.02)
            return base.PositionVelocityNed(
                north_m=0.25,
                east_m=0.0,
                down_m=0.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    with pytest.raises(TimeoutError, match="waypoint stability timeout"):
        asyncio.run(
            executor._wait_checkpoint_stable(
                base=base,
                client=UnqualifiedClient(),
                setpoint=base.Setpoint(0.0, 0.0, 0.0, 0.0),
                rate_hz=100.0,
                timeout_seconds=0.08,
                stable_window_seconds=0.05,
                position_tolerance_m=0.1,
                speed_tolerance_mps=0.1,
            )
        )


# 功能：
#   验证等待模型检查点决定期间继续发布真实位置，未收到决定则明确超时。
# 输入：
#   tmp_path：隔离的决定和终止文件目录。
# 输出：
#   None：不返回业务数据。
def test_checkpoint_decision_wait_keeps_identity_telemetry_fresh(tmp_path: Path) -> None:
    base, executor = _modules()
    client = base.FakeOffboardClient()
    observed_samples = []
    request = RuntimeCheckpointRequest(
        contract_id="mission-test",
        checkpoint=RuntimeCheckpoint(
            checkpoint_id="checkpoint-001",
            segment_id="segment-001",
            task_id="task-001",
            track_point_index=1,
            target_node="pickup",
        ),
        observed_position_ned_m=Vector3(x=0.0, y=0.0, z=0.0),
        observed_velocity_ned_mps=Vector3(x=0.0, y=0.0, z=0.0),
        commanded_position_ned_m=Vector3(x=0.0, y=0.0, z=0.0),
        position_error_m=0.0,
        speed_mps=0.0,
        battery_percent=80.0,
        deterministic_gates={"stable": True},
    )

    with pytest.raises(TimeoutError, match="checkpoint decision timeout"):
        asyncio.run(
            executor._wait_checkpoint_decision(
                base=base,
                client=client,
                setpoint=base.Setpoint(0.0, 0.0, 0.0, 0.0),
                request=request,
                decision_path=tmp_path / "decision.json",
                abort_file=tmp_path / "abort.json",
                rate_hz=100.0,
                timeout_seconds=0.08,
                sample_observer=observed_samples.append,
            )
        )

    assert observed_samples
