from __future__ import annotations

import hashlib
import io
import json
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.asset_runtime_resolver import (
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from dronedream_agent_app.custom_models import ModelConnection
from dronedream_agent_app.plugin_manager import PluginManager
from dronedream_agent_app.runtime_manager import RuntimeBridgeError, RuntimeManager
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.contracts import MapAsset
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.model_harness.boundary import (
    HarnessInputEnvelope,
    HarnessOutputEnvelope,
    autonomy_runtime_envelope,
    compile_autonomy_control_plane_receipt,
    compile_execution_authority,
    harness_input_sha256,
    selections_from_plugin_snapshot,
)
from dronedream_agent_core.model_harness.memory import MemoryOwnerScope
from dronedream_agent_core.simulation_payload_runtime import CONTRACT as PAYLOAD_CONTRACT
from dronedream_agent_core.simulation_sensor_runtime import MAGNETIC_SENSOR_CONTRACT_SHA256

pytestmark = pytest.mark.usefixtures("isolated_wsl_host_paths")

_LOCAL_POLICY_ARTIFACT_CONTRACTS = {
    "local-navigation-policy": (
        ["state_features", "candidate_features", "candidate_mask"],
        ["candidate_scores", "action_scores", "risk_score"],
    ),
    "perception-encoder": (
        ["forward_rgb"],
        [
            "quality_logits",
            "scene_logits",
            "semantic_logits",
            "traversability_logits",
            "visual_features",
        ],
    ),
}


# 功能：
#   建立带完整哈希目录的最小资源夹具；文件内容仅用于校验，不是可运行的飞控或原生库。
# 输入：
#   root：夹具资源根目录。
# 输出：
#   root：已包含 Runtime 目录和文件索引的资源根目录。
def _runtime_resources(root: Path) -> Path:
    runtime = root / "runtime"
    (runtime / "wheels").mkdir(parents=True)
    (runtime / "ros_ws" / "src").mkdir(parents=True)
    files = {
        "requirements-linux.txt": b"pydantic==2.13.4\n",
        "px4_offboard_track_executor.py": b"# executor\n",
        "px4_checkpoint_executor.py": b"# checkpoint\n",
        "runtime_depth_safety_worker.py": b"# depth safety worker\n",
        "wheels/dependency.whl": b"wheel",
        "native-sensors/libdronedream-magnetometer.so": b"unit-test-native-library",
        "native-sensors/magnetic-field-probe": b"unit-test-native-probe",
        "native-sensors/geo_magnetic_tables.hpp": b"unit-test-native-table",
        "payload-placement/libdronedream-payload-placement.so": b"\x7fELF\x02\x01unit-test-only",
        "camera-clock/libdronedream-camera-clock.so": b"unit-test-clock-library",
    }
    files["camera-clock/camera-clock-runtime.json"] = json.dumps({
        "schema_version": "dronedream.native-camera-clock.v1",
        "clock_contract": "exact-native-sim-tick-preupdate-v1",
        "library_sha256": hashlib.sha256(files["camera-clock/libdronedream-camera-clock.so"]).hexdigest(),
        "sources": dict.fromkeys(('camera_clock.cpp', 'capture_clock.hpp', 'CMakeLists.txt',
                                  'capture_clock_test.cpp'), 'a' * 64),
    }).encode()
    files["payload-placement/payload-placement-runtime.json"] = json.dumps({
        "contract": PAYLOAD_CONTRACT,
        "library_sha256": hashlib.sha256(
            files["payload-placement/libdronedream-payload-placement.so"]).hexdigest(),
        "sources": {"CMakeLists.txt": "a" * 64, "PayloadPlacement.cc": "b" * 64},
    }).encode()
    files["native-sensors/native-sensor-runtime.json"] = json.dumps(
        {
            "wire_contract": "px4-gz-fimex-gauss",
            "native_tests_passed": True,
            "sensor_contract_sha256": MAGNETIC_SENSOR_CONTRACT_SHA256,
            "library_sha256": hashlib.sha256(
                files["native-sensors/libdronedream-magnetometer.so"]
            ).hexdigest(),
            "field_probe_sha256": hashlib.sha256(
                files["native-sensors/magnetic-field-probe"]
            ).hexdigest(),
            "px4_magnetic_table_sha256": hashlib.sha256(
                files["native-sensors/geo_magnetic_tables.hpp"]
            ).hexdigest(),
        }
    ).encode()
    entries = []
    for relative, content in files.items():
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    (runtime / "runtime-manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "files": entries}), encoding="utf-8"
    )
    return root


# 功能：
#   使用隔离应用存储创建管理器，不启动运行环境。
# 输入：
#   store_root：测试应用数据目录。
#   resource_root：测试产品资源目录。
# 输出：
#   manager：可替换进程入口的运行管理器。
def _manager(store_root: Path, resource_root: Path) -> RuntimeManager:
    store = AppStore(store_root)
    manager = RuntimeManager(store, resource_root, PluginManager(store))
    return manager


# 功能：
#   构造指定专家角色的包清单和哈希夹具；占位 ONNX 文件不用于真实推理。
# 输入：
#   package：新建测试模型包的目录。
#   roles：需要声明输入输出契约的专家角色。
# 输出：
#   package_sha256：实际加载夹具清单后得到的内容指纹。
def _write_local_policy_package_fixture(
    package: Path,
    *,
    roles: tuple[str, ...] = ("local-navigation-policy",),
) -> str:
    package.mkdir(parents=True)
    artifacts = []
    for role in roles:
        content = f"fixture:{role}".encode()
        filename = f"{role}.onnx"
        (package / filename).write_bytes(content)
        input_names, output_names = _LOCAL_POLICY_ARTIFACT_CONTRACTS[role]
        # 视觉特征只在包内确有对应编码器声明时连到导航角色。
        if role == "local-navigation-policy" and "perception-encoder" in roles:
            input_names = [*input_names, "visual_features"]
        artifacts.append(
            {
                "role": role,
                "relative_path": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "format": "onnx",
                "input_names": input_names,
                "output_names": output_names,
            }
        )
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "package_id": "fixture.general-navigation",
                "display_name": "Fixture general navigation",
                "scope": "general",
                "vehicle_sha256": "a" * 64,
                "sensor_contract_sha256": "b" * 64,
                "state_feature_count": 46,
                "candidate_feature_count": 15,
                "maximum_candidates": 8,
                "maximum_inference_latency_ms": 250,
                **(
                    {
                        "visual_width": 224,
                        "visual_height": 128,
                        "visual_feature_count": 139,
                        "visual_normalization": "imagenet",
                    }
                    if "perception-encoder" in roles
                    else {}
                ),
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    package_sha256 = load_local_policy_package(package).package_sha256
    return package_sha256


# 功能：
#   验证基础 ROS/Gazebo 可用并不代表应用依赖已完成部署。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换 WSL 探测结果的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_status_distinguishes_base_stack_from_provisioning(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "res"))
    results = iter(
        (
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 1, "", ""),
        )
    )
    monkeypatch.setattr(manager, "_bash", lambda _script, timeout: next(results))

    status = manager.status()

    assert status == {
        "distribution": "DroneDreamRuntime",
        "runtime_available": True,
        "resources_ready": True,
        "provisioned": False,
        "issue": "RUNTIME_PROVISION_REQUIRED",
    }


# 功能：
#   验证仿真专用目录把准入回执与正式资格回执分开，且包路径仍限定在资源目录内。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_local_policy_runtime_catalog_is_scoped_and_simulation_explicit(tmp_path):
    resources = _runtime_resources(tmp_path / "resources")
    policy_root = resources / "runtime" / "local-policy"
    package = policy_root / "packages" / "general-navigation"
    receipt = policy_root / "receipts" / "active.json"
    package_sha256 = _write_local_policy_package_fixture(package)
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    (policy_root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-runtime-catalog.v2",
                "deployment_scope": "simulation-only",
                "entries": [
                    {
                        "package": "packages/general-navigation",
                        "package_sha256": package_sha256,
                        "receipt": {
                            "kind": "simulation-admission",
                            "path": "receipts/active.json",
                            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = _manager(tmp_path / "store", resources)

    packages, qualifications, admissions = manager._local_policy_runtime_resources()

    assert packages == (package.resolve(),)
    assert qualifications == ()
    assert admissions == (receipt.resolve(),)


# 功能：
#   验证视觉能力按包清单中的实际角色判断，缺少角色或清单形状错误不能猜测启用。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_local_policy_visual_capability_comes_from_package_manifest(tmp_path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {"role": "local-navigation-policy"},
                    {"role": "perception-encoder"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert RuntimeManager._local_policy_packages_have_role((package,), "perception-encoder")
    assert not RuntimeManager._local_policy_packages_have_role(
        (package,), "payload-dynamics-adapter"
    )
    (package / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeBridgeError, match="RUNTIME_LOCAL_POLICY_MANIFEST_INVALID"):
        RuntimeManager._local_policy_packages_have_role((package,), "perception-encoder")


# 功能：
#   验证本地模型目录不能通过父目录跳转加载资源根之外的模型包。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_local_policy_runtime_catalog_rejects_path_escape(tmp_path):
    resources = _runtime_resources(tmp_path / "resources")
    policy_root = resources / "runtime" / "local-policy"
    policy_root.mkdir(parents=True)
    outside = resources / "outside-package"
    outside.mkdir()
    (policy_root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-runtime-catalog.v2",
                "deployment_scope": "simulation-only",
                "entries": [
                    {
                        "package": "../../outside-package",
                        "package_sha256": "a" * 64,
                        "receipt": {
                            "kind": "simulation-admission",
                            "path": "../../outside-receipt.json",
                            "sha256": "b" * 64,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = _manager(tmp_path / "store", resources)

    with pytest.raises(RuntimeBridgeError, match="RUNTIME_LOCAL_POLICY_PATH_ESCAPE"):
        manager._local_policy_runtime_resources()


# 功能：
#   验证 WSL 不可用时明确报告未就绪，不以本地资源存在代替运行环境可用。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换 WSL 调用的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_status_fails_closed_when_wsl_is_missing(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "res"))

    # 功能：
    #   模拟 Runtime 尚未安装时的调用失败。
    # 输入：
    #   _script：未执行的探测脚本。
    #   timeout：调用方预算。
    # 输出：
    #   None：不返回业务数据。
    def unavailable(_script: str, *, timeout: float):
        raise RuntimeBridgeError("DRONEDREAM_RUNTIME_UNAVAILABLE")

    monkeypatch.setattr(manager, "_bash", unavailable)

    status = manager.status()

    assert status["runtime_available"] is False
    assert status["provisioned"] is False
    assert status["issue"] == "DRONEDREAM_RUNTIME_UNAVAILABLE"


# 功能：
#   验证安装脚本构建并使用合并 ROS 前缀，不回退到旧的隔离安装目录。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：记录脚本而不执行安装的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_provision_builds_and_verifies_one_merged_ros_prefix(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "res"))
    scripts: list[str] = []
    statuses = iter(
        (
            {"runtime_available": True, "provisioned": False},
            {"runtime_available": True, "provisioned": True},
        )
    )
    monkeypatch.setattr(manager, "status", lambda: next(statuses))

    # 功能：
    #   收集各阶段脚本，并提供消息与依赖探测成功的受控输出。
    # 输入：
    #   script：待验证的安装脚本。
    #   timeout：该阶段的调用预算。
    # 输出：
    #   completed：模拟成功的进程结果。
    def fake_bash(script: str, *, timeout: float):
        scripts.append(script)
        output = "ROS_MESSAGES_READY\nAUTONOMY_RUNTIME_READY\n"
        completed = subprocess.CompletedProcess([], 0, output, "")
        return completed

    monkeypatch.setattr(manager, "_bash", fake_bash)

    result = manager.provision()

    assert result["provisioned"] is True
    rendered = "\n".join(scripts)
    assert "build --merge-install" in rendered
    assert '--build-base "$root/ros_ws-merged/build"' in rendered
    assert '--install-base "$root/ros_ws-merged/install"' in rendered
    assert 'source "$root/ros_ws-merged/install/setup.bash"' in rendered
    assert "ros2 interface show dronedream_agent_msgs/msg/MissionObservation" in rendered
    assert "from dronedream_agent_msgs.msg import MissionObservation" in rendered
    assert '"$root/ros_ws/install/setup.bash"' not in rendered


# 功能：
#   验证资格环境版本来自探测输出，并绑定当前 Runtime 清单哈希。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：提供受控运行环境输出的测试工具。
# 输出：
#   None：不返回业务数据。
def test_qualification_versions_are_observed_from_provisioned_runtime(tmp_path, monkeypatch):
    resources = _runtime_resources(tmp_path / "res")
    manager = _manager(tmp_path / "store", resources)
    scripts: list[str] = []
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {
            "runtime_available": True,
            "provisioned": True,
        },
    )

    # 功能：
    #   记录版本探测命令并返回明确的 ROS、Gazebo 和 PX4 版本夹具。
    # 输入：
    #   script：待核对的版本探测脚本。
    #   timeout：调用方的时间预算。
    # 输出：
    #   completed：包含版本键值的模拟进程结果。
    def fake_bash(script, *, timeout):
        scripts.append(script)
        completed = subprocess.CompletedProcess(
            [],
            0,
            f"ros_distribution=jazzy\ngazebo_sim=8.14.0\npx4_commit={'a' * 40}\n",
            "",
        )
        return completed

    monkeypatch.setattr(manager, "_bash", fake_bash)

    versions = manager.qualification_environment_versions()

    assert versions["ros_distribution"] == "jazzy"
    assert versions["gazebo_sim"] == "8.14.0"
    assert versions["px4_commit"] == "a" * 40
    assert "GZ_CONFIG_PATH=/usr/share/gz" in scripts[-1]
    assert (
        versions["runtime_manifest_sha256"]
        == hashlib.sha256((resources / "runtime/runtime-manifest.json").read_bytes()).hexdigest()
    )


# 功能：
#   验证环境未部署时不返回可用于授予资产资格的版本信息。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换部署状态的测试工具。
# 输出：
#   None：不返回业务数据。
def test_qualification_versions_fail_closed_when_runtime_is_not_provisioned(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "res"))
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {
            "runtime_available": True,
            "provisioned": False,
        },
    )

    assert manager.qualification_environment_versions() == {}


# 功能：
#   验证执行器文件被修改后，旧 Runtime 清单不能继续声明资源就绪。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_resource_hash_tampering_is_rejected(tmp_path):
    resources = _runtime_resources(tmp_path / "res")
    manager = _manager(tmp_path / "store", resources)
    (resources / "runtime" / "px4_checkpoint_executor.py").write_text(
        "# changed\n", encoding="utf-8"
    )

    assert manager._resources_ready() is False


# 功能：
#   验证必要原生依赖未列入清单时，即使文件存在也不能继承包的资格。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_unlisted_native_dependency_cannot_inherit_manifest(tmp_path):
    resources = _runtime_resources(tmp_path / "res")
    manifest_path = resources / "runtime/runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = [
        entry
        for entry in manifest["files"]
        if entry["path"] != "native-sensors/magnetic-field-probe"
    ]
    manifest_path.write_text(json.dumps(manifest))
    assert _manager(tmp_path / "store", resources)._resources_ready() is False


# 功能：
#   载荷运行库缺失、字节变化或契约过期时拒绝就绪，不回退旧的位置传送接口。
# 输入：
#   tmp_path：隔离资源根；mode：要破坏的文件或语义边界。
# 输出：
#   None：由就绪拒绝断言给出结果。
@pytest.mark.parametrize('mode', ['missing', 'bytes', 'contract'])
def test_runtime_payload_placement_required(tmp_path, mode):
    resources = _runtime_resources(tmp_path / 'res')
    runtime = resources / 'runtime'
    library = runtime / 'payload-placement/libdronedream-payload-placement.so'
    if mode == 'missing':
        library.unlink()
    elif mode == 'bytes':
        library.write_bytes(b'changed')
    else:
        path = runtime / 'payload-placement/payload-placement-runtime.json'
        receipt = json.loads(path.read_text())
        receipt['contract'] = 'old-set-pose'
        path.write_text(json.dumps(receipt))
        manifest_path = runtime / 'runtime-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest['files']:
            if entry['path'] == 'payload-placement/payload-placement-runtime.json':
                entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             bytes=path.stat().st_size)
        manifest_path.write_text(json.dumps(manifest))
    assert _manager(tmp_path / 'store', resources)._resources_ready() is False


# 功能：
#   验证全部文件大小与哈希一致时，过时的传感器语义契约仍被独立拒绝。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_old_native_semantics_rejected_even_with_matching_file_hashes(tmp_path):
    resources = _runtime_resources(tmp_path / "res")
    receipt_path = resources / "runtime/native-sensors/native-sensor-runtime.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sensor_contract_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    manifest_path = resources / "runtime/runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        if entry["path"] == "native-sensors/native-sensor-runtime.json":
            entry["sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            # 更新大小，让本测试确实到达语义检查，而不是提前因文件大小不同被拦截。
            entry["bytes"] = receipt_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    assert _manager(tmp_path / "store", resources)._resources_ready() is False


# 功能：
#   验证后台安装从排队进入完成状态，只有全部步骤与最终探测通过才报告百分之百。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换实际安装步骤的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_setup_reports_completion_after_verification(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "res"))
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {
            "distribution": "DroneDreamRuntime",
            "runtime_available": True,
            "resources_ready": True,
            "provisioned": True,
            "issue": None,
        },
    )

    # 功能：
    #   按阶段脚本返回预期验收标志，不执行安装命令。
    # 输入：
    #   script：当前安装阶段脚本。
    #   timeout：测试中不使用的时间预算。
    # 输出：
    #   completed：阶段成功的模拟进程结果。
    def successful_step(script: str, *, timeout: float):
        del timeout
        output = (
            "ROS_MESSAGES_READY\nAUTONOMY_RUNTIME_READY\n"
            if "AUTONOMY_RUNTIME_READY" in script
            else ""
        )
        completed = subprocess.CompletedProcess([], 0, output, "")
        return completed

    monkeypatch.setattr(manager, "_bash", successful_step)

    started = manager.start_setup()
    assert started["phase"] == "queued"
    assert started["progress"] == 0
    deadline = time.monotonic() + 2
    while manager.setup_progress()["active"] and time.monotonic() < deadline:
        time.sleep(0.01)

    completed = manager.setup_progress()
    assert completed["phase"] == "completed"
    assert completed["progress"] == 100
    assert completed["active"] is False
    assert completed["error"] is None


# 功能：
#   验证基础环境缺失时停止安装并保留实际失败阶段，不继续推进进度。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：提供缺失环境状态的测试工具。
# 输出：
#   None：不返回业务数据。
def test_runtime_setup_stops_at_failed_base_runtime_check(tmp_path, monkeypatch):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "res"))
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {
            "distribution": "DroneDreamRuntime",
            "runtime_available": False,
            "resources_ready": True,
            "provisioned": False,
            "issue": "DRONEDREAM_RUNTIME_UNAVAILABLE",
        },
    )

    manager.start_setup()
    deadline = time.monotonic() + 2
    while manager.setup_progress()["active"] and time.monotonic() < deadline:
        time.sleep(0.01)

    failed = manager.setup_progress()
    assert failed["phase"] == "failed"
    assert failed["progress"] == 12
    assert failed["active"] is False
    assert failed["error"] == "DRONEDREAM_RUNTIME_UNAVAILABLE"
    assert failed["failed_phase"] == "checkingBaseRuntime"


# 功能：
#   验证连接文件使用 LF 换行，防止 Windows 回车进入 Linux 环境变量值。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_secret_session_uses_wsl_safe_lf_newlines(tmp_path):
    path = tmp_path / "session.env"

    RuntimeManager._write_secret_session(
        path,
        provider="deepseek",
        model_id="deepseek-v4-flash",
        grant="private-value",
        gateway="https://api.deepseek.com",
    )

    payload = path.read_bytes()
    assert b"\r" not in payload
    assert payload.endswith(b"\n")
    assert b"private-value" in payload


# 功能：
#   构造绑定计划、插件与已打包资产的宿主执行夹具；准备计划的类型解析使用替身。
#   此夹具只测试桌面桥接，不把模拟对象或打包证据当作新飞行验收。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换实际环境探测与计划解析的测试工具。
# 输出：
#   fixture：运行管理器与执行参数字典。
def _execution_fixture(tmp_path, monkeypatch):
    store = AppStore(tmp_path / "store")
    repository = Path(__file__).resolve().parents[1]
    bundled_assets = repository / "app" / "desktop" / "src-tauri" / "resources" / "default-assets"
    version_records = AssetImportService(store).seed_bundled_sources(bundled_assets)
    bundled_index = json.loads((bundled_assets / "index.json").read_text(encoding="utf-8"))
    default_packages = {
        str(entry["kind"]): entry for entry in bundled_index["qualified_pair"]["packages"]
    }
    records_by_key = {
        (str(record["asset_id"]), str(record["content_sha256"])): record
        for record in version_records
    }
    map_asset = records_by_key[
        (default_packages["map"]["asset_id"], default_packages["map"]["content_sha256"])
    ]
    vehicle_asset = records_by_key[
        (
            default_packages["vehicle"]["asset_id"],
            default_packages["vehicle"]["content_sha256"],
        )
    ]
    map_selection = resolve_versioned_map(map_asset)
    vehicle_selection = resolve_versioned_vehicle(vehicle_asset)
    pair_qualification = store.qualified_asset_pair(
        map_asset_id=str(map_asset["asset_id"]),
        map_content_sha256=str(map_asset["content_sha256"]),
        vehicle_asset_id=str(vehicle_asset["asset_id"]),
        vehicle_content_sha256=str(vehicle_asset["content_sha256"]),
    )
    assert pair_qualification is not None
    planned_pair_binding = store.verified_asset_pair_execution_binding(pair_qualification)
    plugins = PluginManager(store)
    thread = store.create_thread("runtime layout", "gpt-5.4")
    thread_id = str(thread["thread_id"])
    store.patch_thread(
        thread_id,
        {
            "selected_map_id": map_asset["asset_id"],
            "selected_map_content_sha256": map_asset["content_sha256"],
            "selected_vehicle_id": vehicle_asset["asset_id"],
            "selected_vehicle_content_sha256": vehicle_asset["content_sha256"],
        },
    )
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    (plan_root / "prepared-mission.json").write_text("{}\n", encoding="utf-8")
    snapshot = plugins.snapshot(thread_id=thread_id)
    owner_scope = MemoryOwnerScope(
        owner_account_id="account-runtime",
        source_edition="autonomy",
    )
    control_plane = compile_autonomy_control_plane_receipt(
        selections_from_plugin_snapshot(snapshot),
        effective_maximum_model_calls=16,
        effective_maximum_repair_cycles=4,
    )
    runtime_envelope = autonomy_runtime_envelope(
        owner_scope,
        control_plane,
        snapshot,
        selected_maximum_model_calls=16,
        maximum_intent_rounds=2,
        maximum_planning_rounds=4,
    )
    input_envelope = HarnessInputEnvelope(
        request_id="request-runtime-0001",
        task_id=thread_id,
        thread_id=thread_id,
        owner_binding_sha256=runtime_envelope.owner_binding_sha256,
        tenant_binding_sha256=runtime_envelope.tenant_binding_sha256,
        source_edition="autonomy",
        control_plane_selection_sha256=control_plane.selection_sha256,
        current_request={"message": "runtime layout"},
    )
    output_envelope = HarnessOutputEnvelope(
        request_id=input_envelope.request_id,
        task_id=input_envelope.task_id,
        control_plane_selection_sha256=control_plane.selection_sha256,
        input_envelope_sha256=harness_input_sha256(input_envelope),
        status="validated_proposal",
        structured_result={"contract_id": "mission-layout"},
        model_call_count=1,
        repair_cycle_count=0,
        validation_receipt_ids=("mission-layout",),
    )
    plan_revision_id = "plan-" + "1" * 32
    authority = compile_execution_authority(
        thread_id=thread_id,
        plan_revision_id=plan_revision_id,
        contract_id="mission-layout",
        prepared_mission_sha256="a" * 64,
        runtime=runtime_envelope,
    )
    store.issue_execution_authority(authority)
    (plan_root / "model-harness-execution-authority.json").write_text(
        authority.model_dump_json(), encoding="utf-8"
    )
    store.append_message(
        thread_id,
        role="assistant",
        kind="plan",
        content="prepared",
        metadata={
            "output_dir": str(plan_root),
            "contract_id": "mission-layout",
            "plan_revision_id": plan_revision_id,
            "plugin_snapshot_id": snapshot.snapshot_id,
            "plugin_catalog_sha256": snapshot.catalog_sha256,
            "execution_authority": authority.model_dump(mode="json"),
            "model_harness_control_plane": {
                "receipt": control_plane.model_dump(mode="json"),
                "runtime_envelope": runtime_envelope.model_dump(mode="json"),
                "input_envelope": input_envelope.model_dump(mode="json"),
                "output_envelope": output_envelope.model_dump(mode="json"),
            },
        },
    )
    assert store.latest_plugin_snapshot(thread_id)["snapshot_id"] == snapshot.snapshot_id
    store.set_thread_state(thread_id, "awaiting_confirmation")
    resources = _runtime_resources(tmp_path / "resources")
    policy_root = resources / "runtime" / "local-policy"
    policy_package = policy_root / "packages" / "general-navigation"
    policy_receipt = policy_root / "receipts" / "active.json"
    policy_package_sha256 = _write_local_policy_package_fixture(
        policy_package,
        roles=("local-navigation-policy", "perception-encoder"),
    )
    policy_receipt.parent.mkdir(parents=True)
    policy_receipt.write_text("{}\n", encoding="utf-8")
    (policy_root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.local-policy-runtime-catalog.v2",
                "deployment_scope": "simulation-only",
                "entries": [
                    {
                        "package": "packages/general-navigation",
                        "package_sha256": policy_package_sha256,
                        "receipt": {
                            "kind": "simulation-admission",
                            "path": "receipts/active.json",
                            "sha256": hashlib.sha256(policy_receipt.read_bytes()).hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = RuntimeManager(store, resources, plugins)
    monkeypatch.setattr(manager, "status", lambda: {"provisioned": True})
    qualification = map_asset["manifest"]["qualification"]
    monkeypatch.setattr(
        manager,
        "qualification_environment_versions",
        lambda: dict(qualification["environment_versions"]),
    )
    monkeypatch.setattr(
        "dronedream_agent_app.runtime_manager.PreparedMission.model_validate_json",
        lambda _payload, **_kwargs: SimpleNamespace(
            schema_version="dronedream.prepared-mission.v4",
            contract=SimpleNamespace(
                contract_id="mission-layout",
                conversation_id=thread_id,
                map_asset_id=map_asset["asset_id"],
                vehicle_asset_id=vehicle_asset["asset_id"],
                map_sha256=sha256_json(MapAsset.model_validate_json(map_selection.graph.read_bytes())),
                map_semantic_sha256=hashlib.sha256(map_selection.semantic.read_bytes()).hexdigest(),
                vehicle_sha256=hashlib.sha256(
                    vehicle_selection.vehicle_sdf.read_bytes()
                ).hexdigest(),
            ),
            simulation_capabilities={
                "simulator": {"engine": "gazebo-harmonic"},
                "native_runtime": {
                    "transport": {"implementation": "px4-uxrce-dds"},
                    "watchdog": {
                        "deadline_ms": 250,
                        "startup_deadline_ms": 10_000,
                        "scheduling_jitter_grace_ms": 25,
                        "persistent_miss_ms": 100,
                    },
                },
            },
            plugin_snapshot=snapshot,
            evidence=[
                SimpleNamespace(
                    event_type="mission.asset-pair-qualification",
                    payload=planned_pair_binding.model_dump(mode="json"),
                )
            ],
        ),
    )
    monkeypatch.setattr(
        "dronedream_agent_app.runtime_manager.sha256_json",
        lambda value: "a" * 64 if isinstance(value, SimpleNamespace) else sha256_json(value),
    )
    provider, _plugin_id, capability_id = plugins.model_binding_for_model("gpt-5.4")
    connection = ModelConnection(
        selection_id="gpt-5.4",
        provider=provider,
        model_id="gpt-5.4",
        api_key="private-value",
        base_url="https://example.supabase.co/functions/v1/model-gateway",
        api_style="chat-completions",
        capability_id=capability_id,
        source="default",
    )
    fixture = (
        manager,
        {
            "thread_id": thread_id,
            "connection": connection,
            "plan_revision_id": plan_revision_id,
            "owner_scope": owner_scope,
        },
    )
    return fixture


# 功能：
#   验证宿主日志与一次性连接文件不污染仿真的新证据目录，且本地专家参数正确传递。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换真实进程创建的测试工具。
# 输出：
#   None：不返回业务数据。
def test_execute_keeps_supervisor_files_outside_empty_simulation_run(tmp_path, monkeypatch):
    manager, arguments = _execution_fixture(tmp_path, monkeypatch)
    # 夹具必须保留两种摘要的真实差异，否则再次误用原始文件摘要也可能通过测试。
    selected = manager.store.get_thread(arguments["thread_id"])
    graph_path = resolve_versioned_map(manager.store.get_asset_version(
        selected["selected_map_id"], selected["selected_map_content_sha256"],
    )).graph
    assert hashlib.sha256(graph_path.read_bytes()).hexdigest() != sha256_json(
        MapAsset.model_validate_json(graph_path.read_bytes())
    )

    class RecordingInput(io.StringIO):
        # 功能：
        #   保留已写入的脚本供断言，不关闭内存输入流。
        # 输入：
        #   self：模拟输入流。
        # 输出：
        #   None：不返回业务数据。
        def close(self) -> None:
            pass

    class FakeProcess:
        # 功能：
        #   构造不启动 WSL 的进程替身。
        # 输入：
        #   self：模拟进程。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self) -> None:
            self.stdin = RecordingInput()

        # 功能：
        #   短暂等待后模拟未通过的退出状态，让宿主监视线程完成清理。
        # 输入：
        #   self：模拟进程。
        # 输出：
        #   return_code：模拟失败退出码。
        def wait(self) -> int:
            time.sleep(0.05)
            return_code = 1
            return return_code

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    launched = manager.execute(**arguments)

    execution_root = Path(str(launched["execution_root"]))
    simulation_root = Path(str(launched["run_dir"]))
    assert simulation_root == execution_root / "simulation"
    assert execution_root.is_dir()
    assert not simulation_root.exists()
    assert (execution_root / "desktop-runtime.stdout.log").is_file()
    assert (execution_root / "desktop-runtime.stderr.log").is_file()
    launch_script = process.stdin.getvalue()
    assert "--profile runtime-manager execute-prepared-mission" in launch_script
    assert "--local-navigation-provider local-policy" in launch_script
    assert "--local-navigation-fallback-provider openai" in launch_script
    assert "--local-navigation-model-timeout-seconds 0.25" in launch_script
    assert "--local-navigation-fallback-model-timeout-seconds 10" in launch_script
    assert "--local-navigation-period-seconds 1" in launch_script
    assert "--local-policy-package" in launch_script
    assert "--local-policy-simulation-admission" in launch_script
    assert "--local-navigation-context-id" in launch_script
    assert "--require-local-navigation-control-authority" in launch_script
    assert "--local-navigation-visual-enabled" in launch_script
    assert "--heading-policy route-tangent-relative" in launch_script
    assert "--maximum-yaw-rate-deg-s 20" in launch_script
    assert "-p watchdog_deadline_ms:=250" in launch_script
    assert "-p watchdog_startup_deadline_ms:=10000" in launch_script
    assert "-p watchdog_scheduling_jitter_grace_ms:=25" in launch_script
    assert "-p watchdog_persistent_miss_ms:=100" in launch_script


# 功能：
#   验证资产验收桥接使用固定 Runtime 并读取执行器产出的证据文件；进程为测试替身。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换真实运行环境与进程创建的测试工具。
# 输出：
#   None：不返回业务数据。
def test_asset_qualification_bridge_uses_pinned_runtime_and_returns_runner_evidence(
    tmp_path, monkeypatch
):
    resources = _runtime_resources(tmp_path / "resources")
    manager = _manager(tmp_path / "store", resources)
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {
            "runtime_available": True,
            "resources_ready": True,
            "provisioned": True,
        },
    )
    work_root = tmp_path / "prepared"
    work_root.mkdir()
    (work_root / "qualification-plan.json").write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "runtime-evidence"
    evidence = {"status": "verified", "gates": {"executor_completed": True}}
    captured: dict[str, object] = {}

    class RecordingInput(io.StringIO):
        # 功能：
        #   保留写入的脚本，供进程替身结束后断言调用内容。
        # 输入：
        #   self：模拟的标准输入流。
        # 输出：
        #   None：不返回业务数据。
        def close(self) -> None:
            return None

    class FakeProcess:
        # 功能：
        #   初始化受控输入流与终止标志。
        # 输入：
        #   self：验收进程替身。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self) -> None:
            self.stdin = RecordingInput()
            self.terminated = False

        # 功能：
        #   模拟执行器独占创建证据目录并写入结果，记录宿主传入的脚本和等待预算。
        # 输入：
        #   self：验收进程替身。
        #   timeout：等待预算，单位秒。
        # 输出：
        #   return_code：模拟的成功退出码。
        def wait(self, timeout: float) -> int:
            captured["timeout"] = timeout
            captured["script"] = self.stdin.getvalue()
            captured["run_dir_existed_before_runner"] = run_dir.exists()
            run_dir.mkdir()
            (run_dir / "mission_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            return_code = 0
            return return_code

        # 功能：
        #   记录宿主发起的终止操作。
        # 输入：
        #   self：验收进程替身。
        # 输出：
        #   None：不返回业务数据。
        def terminate(self) -> None:
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)

    result = manager.run_asset_pair_qualification(
        job_id="asset-qualification-job-" + "a" * 24,
        work_root=work_root,
        run_dir=run_dir,
    )

    assert result == evidence
    script = str(captured["script"])
    assert "root=$HOME/.local/share/dronedream-autonomy/v0.1.0" in script
    assert 'exec "$root/venv/bin/python"' in script
    assert "dronedream_agent_core.asset_qualification_cli" in script
    assert 'source "$root/ros_ws-merged/install/setup.bash"' in script
    assert '--ros-workspace "$root/ros_ws-merged"' in script
    assert captured["run_dir_existed_before_runner"] is False
    assert (tmp_path / "runtime-evidence.desktop.stdout.log").is_file()
    assert (tmp_path / "runtime-evidence.desktop.stderr.log").is_file()


# 功能：
#   验证取消请求写入对应验收任务的目录，并保留正确任务标识和原因。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_asset_qualification_cancel_writes_scoped_abort_request(tmp_path):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))
    run_dir = tmp_path / "runtime-evidence"
    run_dir.mkdir()

    class FakeProcess:
        pass

    job_id = "asset-qualification-job-" + "b" * 24
    manager._active_asset_qualification_runs[job_id] = SimpleNamespace(
        run_dir=run_dir,
        process=FakeProcess(),
    )

    assert manager.cancel_asset_pair_qualification(job_id) is True
    request = json.loads((run_dir / "live_abort.request.json").read_text(encoding="utf-8"))
    assert request["source"] == f"desktop:asset-qualification:{job_id}"
    assert request["world_paused"] is False
    assert request["reason"] == "qualification_cancelled_by_user"


# 功能：
#   验证令牌内容相同也不能让另一操作员使用已发放的接管授权。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_operator_control_rejects_identity_different_from_issued_grant(tmp_path):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))
    thread = manager.store.create_thread("operator identity", "gpt-5.4")
    thread_id = str(thread["thread_id"])
    manager._active_runs[thread_id] = SimpleNamespace(
        execution_id="execution-" + "a" * 32,
        run_dir=tmp_path / "run",
        process=SimpleNamespace(),
        started_at="2026-08-24T00:00:00+00:00",
    )
    manager._takeover_grants[thread_id] = SimpleNamespace(
        message_id="runtime-msg-" + "b" * 32,
        execution_id="execution-" + "a" * 32,
        operator_id="account-owner",
        token_sha256=hashlib.sha256(b"grant-token").hexdigest(),
        grant_sha256="c" * 64,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        next_sequence=1,
    )

    with pytest.raises(RuntimeBridgeError, match="TAKEOVER_GRANT_OPERATOR_MISMATCH"):
        manager.submit_operator_control(
            thread_id,
            operator_id="account-attacker",
            message_id="runtime-msg-" + "b" * 32,
            grant_token="grant-token",
            action="release",
            north_mps=0.0,
            east_mps=0.0,
            down_mps=0.0,
            yaw_rate_dps=0.0,
            duration_seconds=0.25,
        )


# 功能：
#   验证活动任务摘要只暴露执行身份和状态，不泄漏本机证据目录。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   None：不返回业务数据。
def test_active_execution_evidence_is_path_free(tmp_path):
    manager = _manager(tmp_path / "store", _runtime_resources(tmp_path / "resources"))
    thread = manager.store.create_thread("evidence", "gpt-5.4")
    thread_id = str(thread["thread_id"])
    run_dir = manager.store.missions_root / thread_id / "execution-test" / "simulation"
    manager._active_runs[thread_id] = SimpleNamespace(
        execution_id="execution-" + "a" * 32,
        run_dir=run_dir,
        process=SimpleNamespace(),
        started_at="2026-08-21T00:00:00+00:00",
    )

    evidence = manager.execution_evidence(thread_id)

    assert evidence == {
        "schema_version": "dronedream.agent-execution-evidence.v1",
        "thread_id": thread_id,
        "execution_id": "execution-" + "a" * 32,
        "state": "executing",
        "started_at": "2026-08-21T00:00:00+00:00",
        "completed_at": None,
        "result": None,
    }
    assert "run_dir" not in evidence


# 功能：
#   静态检查构建脚本固定时间戳并包含当前深度安全工作器，不代替真实双构建比对。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_release_runtime_manifest_is_reproducible_across_rebuilds() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build-autonomy-windows.ps1"
    ).read_text(encoding="utf-8")

    assert '$env:SOURCE_DATE_EPOCH = "946684800"' in script
    assert "source_date_epoch = 946684800" in script
    assert "generated_at_utc" not in script
    assert "scripts\\runtime_depth_safety_worker.py" in script
    assert 'runtimeResources "runtime_depth_safety_worker.py"' in script
