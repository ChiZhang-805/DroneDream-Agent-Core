"""Fail-closed bridge from the Windows desktop app to DroneDreamRuntime WSL2."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from dronedream_agent_core.contracts import (
    MissionAssetPairQualificationBinding,
    PreparedMission,
    RuntimeHoldAcknowledgement,
    RuntimeOperatorControlCommand,
    RuntimeOperatorTakeoverGrant,
    SimulationWorkflowResult,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.model_harness.boundary import (
    HarnessControlPlaneReceipt,
    HarnessInputEnvelope,
    HarnessOutputEnvelope,
    HarnessRuntimeEnvelope,
    ModelHarnessExecutionAuthority,
    owner_scope_bindings,
    validate_output_against_boundaries,
)
from dronedream_agent_core.model_harness.memory import MemoryOwnerScope
from dronedream_agent_core.plugin_contracts import PluginSnapshot
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from dronedream_agent_core.runtime_interrupt import submit_runtime_message
from dronedream_agent_core.simulation_sensor_runtime import validate_native_sensor_runtime
from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .asset_runtime_resolver import (
    AssetRuntimeResolutionError,
    qualification_matches_environment,
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from .custom_models import ModelConnection
from .plugin_manager import PluginManager, PluginManagerError
from .runtime_control_records import (
    RuntimeControlRecordError,
    read_active_session,
    read_control_record,
    read_takeover_evidence,
)
from .runtime_launch_settings import bounded_integer, packaged_executor, positive_seconds
from .runtime_resource_manifest import verify_runtime_resource_manifest
from .storage import AppStore


class RuntimeBridgeError(RuntimeError):
    """The real ROS/Gazebo/PX4 runtime is missing or rejected a launch."""


@dataclass
class ActiveRun:
    execution_id: str
    run_dir: Path
    process: subprocess.Popen[str]
    started_at: str
    tracking_error: str | None = None


@dataclass
class ActiveAssetQualificationRun:
    run_dir: Path
    process: subprocess.Popen[str]


@dataclass
class ActiveTakeoverGrant:
    message_id: str
    execution_id: str
    operator_id: str
    token_sha256: str
    grant_sha256: str
    expires_at: datetime
    next_sequence: int = 1


# 功能：
#   将 Windows 盘符绝对路径映射为 WSL 的 /mnt 路径，拒绝无法对应盘符的路径。
# 输入：
#   path：本机资源或任务文件路径。
# 输出：
#   mapped：供 WSL 命令使用的绝对路径字符串。
def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive or len(drive) != 1:
        raise RuntimeBridgeError("WINDOWS_PATH_CANNOT_MAP_TO_WSL")
    relative = resolved.as_posix().split(":", 1)[1]
    mapped = f"/mnt/{drive}{relative}"
    return mapped


class RuntimeManager:
    distribution = "DroneDreamRuntime"
    runtime_home = "$HOME/.local/share/dronedream-autonomy/v0.1.0"
    # A merged colcon prefix is required here.  With an isolated prefix,
    # rosidl_generator_py leaves a `.catkin` compatibility marker that causes
    # colcon to omit dronedream_agent_msgs from AMENT_PREFIX_PATH even though
    # its Python extension and shared libraries were installed successfully.
    runtime_ros_workspace = f"{runtime_home}/ros_ws-merged"

    # 功能：
    #   初始化运行、接管授权与安装进度的进程内状态，不在构造时启动 WSL。
    # 输入：
    #   self：当前运行管理器。
    #   store：应用任务与配置存储。
    #   resource_root：产品资源根目录，可为空。
    #   plugin_manager：当前插件目录与能力校验器。
    # 输出：
    #   None：不返回业务数据。
    def __init__(
        self, store: AppStore, resource_root: Path | None, plugin_manager: PluginManager
    ) -> None:
        self.store = store
        self.resource_root = resource_root
        self.plugin_manager = plugin_manager
        self._lock = threading.RLock()
        self._provision_lock = threading.Lock()
        self._active_runs: dict[str, ActiveRun] = {}
        self._active_asset_qualification_runs: dict[str, ActiveAssetQualificationRun] = {}
        self._takeover_grants: dict[str, ActiveTakeoverGrant] = {}
        self._setup_thread: threading.Thread | None = None
        self._setup_snapshot: dict[str, object] = {
            "schema_version": "dronedream.autonomy.runtime-setup.v1",
            "operation_id": None,
            "phase": "idle",
            "progress": 0,
            "active": False,
            "error": None,
            "failed_phase": None,
            "started_at": None,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    # 功能：
    #   有界序列化运行契约，经独占临时文件原子替换目标，失败只清理自身临时文件。
    # 输入：
    #   path：调用方确定的运行契约或请求文件路径。
    #   value：需要保存的类型化契约或 JSON 值。
    #   replace_existing：是否允许替换已有记录；一次性授权和有序指令禁止覆盖。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _atomic_json(path: Path, value: object, *, replace_existing: bool = True) -> None:
        value = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        rendered = encode_json(value, limit=16 * 1024 * 1024, node_limit=2_000_000)
        check_plain_plugin_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
        owned = None
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as output:
                owned = os.fstat(output.fileno())
                output.write(rendered + "\n")
                output.flush()
                os.fsync(output.fileno())
            check_plain_plugin_path(path)
            if replace_existing:
                temporary.replace(path)
            else:
                os.link(temporary, path)
        finally:
            if owned is not None:
                with suppress(OSError):
                    if os.path.samestat(owned, temporary.stat(follow_symlinks=False)):
                        temporary.unlink()

    # 功能：
    #   优先使用任务创建时保存的语言；没有该字段的历史任务回退到应用语言。
    # 输入：
    #   self：持有应用存储的运行管理器。
    #   thread_id：需要生成状态消息的任务标识。
    # 输出：
    #   locale：规范语言标识 zh-CN 或 en-US。
    def _thread_locale(self, thread_id: str) -> str:
        thread = self.store.get_thread(thread_id)
        locale = thread.get("locale")
        if locale in {"zh-CN", "en-US"}:
            return locale
        configured = self.store.get_settings().get("locale")
        locale = "en-US" if configured == "en-US" else "zh-CN"
        return locale

    # 功能：
    #   将受控脚本通过 stdin 交给指定 WSL 发行版，等待结果并统一启动／超时错误。
    # 输入：
    #   self：指定 Runtime 发行版的管理器。
    #   script：内部构造的 bash 脚本，不是用户自然语言。
    #   timeout：宿主等待秒数，不代表已证明 Linux 后代同时退出。
    # 输出：
    #   result：进程退出码及文本输出。
    def _bash(self, script: str, *, timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ["wsl.exe", "-d", self.distribution, "--", "bash", "-s", "--"],
                check=False,
                capture_output=True,
                input=script,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            raise RuntimeBridgeError("DRONEDREAM_RUNTIME_UNAVAILABLE") from error
        return result

    # 功能：
    #   检查必需资源、完整发布索引和原生传感器契约，不把文件存在当作版本一致。
    # 输入：
    #   self：持有产品资源目录的运行管理器。
    # 输出：
    #   ready：全部本地资源检查是否通过。
    def _resources_ready(self) -> bool:
        ready = False
        if self.resource_root is None:
            return ready
        runtime = self.resource_root / "runtime"
        required = (
            runtime / "requirements-linux.txt",
            runtime / "px4_offboard_track_executor.py",
            runtime / "px4_checkpoint_executor.py",
            runtime / "runtime_depth_safety_worker.py",
            runtime / "native-sensors" / "native-sensor-runtime.json",
            runtime / "native-sensors" / "libdronedream-magnetometer.so",
            runtime / "native-sensors" / "magnetic-field-probe",
            runtime / "native-sensors" / "geo_magnetic_tables.hpp",
            runtime / "ros_ws" / "src",
            runtime / "wheels",
        )
        manifest_path = runtime / "runtime-manifest.json"
        try:
            if any(not value.exists() for value in required):
                return ready
            if not manifest_path.is_file() or not any((runtime / "wheels").glob("*.whl")):
                return ready
            # pip 和 ROS 会发现目录中的文件，因此未索引的残留物也必须拒绝。
            verify_runtime_resource_manifest(runtime)
            validate_native_sensor_runtime(runtime / "native-sensors")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return ready
        ready = True
        return ready

    # 功能：
    #   1. 解析可选本地专家目录，区分仿真准入与生产资格，复核包与回执摘要。
    #   2. 此方法不授予控制权限，worker 仍需复核回执及当前资产绑定。
    # 输入：
    #   self：持有产品资源根目录的运行管理器。
    # 输出：
    #   resources：模型包路径、生产资格路径、仿真准入路径三个元组。
    def _local_policy_runtime_resources(
        self,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
        resources = (), (), ()
        if self.resource_root is None:
            return resources
        policy_root = self.resource_root / "runtime" / "local-policy"
        catalog_path = policy_root / "catalog.json"
        if not catalog_path.is_file():
            return resources
        try:
            catalog = decode_json(
                read_plugin_file(catalog_path, limit=4 * 1024 * 1024),
                limit=4 * 1024 * 1024,
            )
            policy_root = policy_root.resolve()
        except (OSError, ValueError) as error:
            raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_CATALOG_INVALID") from error
        if (
            not isinstance(catalog, dict)
            or catalog.get("schema_version") != "dronedream.local-policy-runtime-catalog.v2"
            or not isinstance(catalog.get("entries"), list)
            or not 1 <= len(catalog["entries"]) <= 256
            or catalog.get("deployment_scope") not in {"simulation-only", "production-qualified"}
        ):
            raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_CATALOG_INVALID")
        deployment_scope = str(catalog["deployment_scope"])

        # 功能：
        #   校验专家目录中的规范相对路径，在解析前拒绝链接和逃逸路径。
        # 输入：
        #   relative：目录声明中的路径值。
        #   directory：目标应为目录还是普通文件。
        # 输出：
        #   candidate：位于专家资源根目录内的实际路径。
        def scoped_path(relative: object, *, directory: bool) -> Path:
            if not isinstance(relative, str) or not relative.strip():
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_CATALOG_INVALID")
            try:
                portable_plugin_path(relative)
                check_plain_plugin_path(policy_root / relative)
                candidate = (policy_root / relative).resolve()
                candidate.relative_to(policy_root)
            except ValueError as error:
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_PATH_ESCAPE") from error
            exists = candidate.is_dir() if directory else candidate.is_file()
            if not exists:
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_ARTIFACT_MISSING")
            return candidate

        packages: list[Path] = []
        qualifications: list[Path] = []
        simulation_admissions: list[Path] = []
        for raw_entry in catalog["entries"]:
            if not isinstance(raw_entry, dict):
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_CATALOG_INVALID")
            package = scoped_path(raw_entry.get("package"), directory=True)
            package_sha256 = raw_entry.get("package_sha256")
            receipt = raw_entry.get("receipt")
            if (
                not isinstance(package_sha256, str)
                or len(package_sha256) != 64
                or not isinstance(receipt, dict)
                or receipt.get("kind") not in {"qualification", "simulation-admission"}
                or not isinstance(receipt.get("sha256"), str)
                or len(str(receipt["sha256"])) != 64
            ):
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_RECEIPT_INVALID")
            receipt_path = scoped_path(receipt.get("path"), directory=False)
            if hash_plugin_file(receipt_path, limit=16 * 1024 * 1024) != receipt["sha256"]:
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_RECEIPT_HASH_MISMATCH")
            try:
                observed_package = load_local_policy_package(package)
            except (OSError, TypeError, ValueError) as error:
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_PACKAGE_INVALID") from error
            if observed_package.package_sha256 != package_sha256:
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_PACKAGE_HASH_MISMATCH")
            receipt_kind = receipt["kind"]
            if receipt_kind == "qualification":
                if deployment_scope != "production-qualified":
                    raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_SCOPE_MISMATCH")
                qualifications.append(receipt_path)
            else:
                if deployment_scope != "simulation-only":
                    raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_SCOPE_MISMATCH")
                simulation_admissions.append(receipt_path)
            packages.append(package)
        if len(set(packages)) != len(packages):
            raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_PACKAGE_DUPLICATE")
        resources = tuple(packages), tuple(qualifications), tuple(simulation_admissions)
        return resources

    # 功能：
    #   从包清单明确声明的角色判断能力，不根据模型文件名猜测视觉或控制能力。
    # 输入：
    #   packages：需要检查的模型包目录。
    #   role：待查找的模型角色标识。
    # 输出：
    #   found：是否有包声明该角色。
    @staticmethod
    def _local_policy_packages_have_role(
        packages: tuple[Path, ...],
        role: str,
    ) -> bool:
        found = False
        for package in packages:
            manifest_path = package / "manifest.json"
            try:
                manifest = decode_json(
                    read_plugin_file(manifest_path, limit=4 * 1024 * 1024),
                    limit=4 * 1024 * 1024,
                )
            except (OSError, ValueError) as error:
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_MANIFEST_INVALID") from error
            artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
            if not isinstance(artifacts, list) or not all(
                isinstance(artifact, dict) and isinstance(artifact.get("role"), str)
                for artifact in artifacts
            ):
                raise RuntimeBridgeError("RUNTIME_LOCAL_POLICY_MANIFEST_INVALID")
            if any(artifact["role"] == role for artifact in artifacts):
                found = True
                return found
        return found

    # 功能：
    #   计算当前发布清单的实际字节摘要，供内容绑定比较；不单独验证全部资源。
    # 输入：
    #   self：持有产品资源根目录的管理器。
    # 输出：
    #   digest：清单的 SHA-256 摘要。
    def _runtime_manifest_sha256(self) -> str:
        if self.resource_root is None:
            raise RuntimeBridgeError("RUNTIME_RESOURCES_MISSING")
        manifest_path = self.resource_root / "runtime" / "runtime-manifest.json"
        if not manifest_path.is_file():
            raise RuntimeBridgeError("RUNTIME_RESOURCES_MISSING")
        try:
            digest = hash_plugin_file(manifest_path, limit=16 * 1024 * 1024)
        except (OSError, ValueError) as error:
            raise RuntimeBridgeError("RUNTIME_RESOURCE_MANIFEST_INVALID") from error
        return digest

    # 功能：
    #   分别探测本地资源、WSL 基础环境和已部署依赖，只有检查通过才报告就绪。
    # 输入：
    #   self：当前运行管理器。
    # 输出：
    #   status：发行版、可用性、资源就绪、部署状态和问题代码。
    def status(self) -> dict[str, object]:
        resources_ready = self._resources_ready()
        try:
            base_probe = self._bash(
                "set -e; source /opt/ros/jazzy/setup.bash; set -u; "
                "command -v ros2 >/dev/null; command -v gz >/dev/null; "
                "test -x /opt/PX4-Autopilot/build/px4_sitl_default/bin/px4",
                timeout=15,
            )
        except RuntimeBridgeError:
            base_probe = None
        runtime_available = bool(base_probe and base_probe.returncode == 0)
        provisioned = False
        probe_failed = False
        if runtime_available and resources_ready:
            try:
                manifest_sha256 = self._runtime_manifest_sha256()
            except RuntimeBridgeError:
                # 检查到探测之间的文件变化也按资源未就绪处理。
                resources_ready = False
                manifest_sha256 = "missing"
            provision_script = (
                f"set -e; test -x {self.runtime_home}/venv/bin/dronedream-agent; "
                f"test -f {self.runtime_ros_workspace}/install/setup.bash; "
                f'test "$(cat {self.runtime_home}/.runtime-manifest-sha256)" = '
                f"{shlex.quote(manifest_sha256)}; "
                f"set +u; source {self.runtime_ros_workspace}/install/setup.bash; set -u; "
                "ros2 interface show dronedream_agent_msgs/msg/MissionObservation "
                ">/dev/null; "
                "python3 -c 'from dronedream_agent_msgs.msg import MissionObservation'; "
                "ros2 run dronedream_agent_plugin_api plugin_probe | "
                "grep -q PLUGIN_PROBE_READY; "
                "ros2 run dronedream_agent_plugin_api capability_host --self-test | "
                "grep -q PLUGIN_LIFECYCLE_READY; "
                f"{self.runtime_home}/venv/bin/python -c "
                "'import openai,pydantic,mavsdk,onnxruntime; from PIL import Image; "
                'Image.new("RGB", (2,2)).resize((1,1))\''
            )
            try:
                provision_probe = self._bash(provision_script, timeout=15)
                provisioned = resources_ready and provision_probe.returncode == 0
            except RuntimeBridgeError:
                probe_failed = True
        issue = None
        if not resources_ready:
            issue = "RUNTIME_RESOURCES_MISSING"
        elif not runtime_available:
            issue = "DRONEDREAM_RUNTIME_UNAVAILABLE"
        elif probe_failed:
            issue = "RUNTIME_PROVISION_PROBE_FAILED"
        elif not provisioned:
            issue = "RUNTIME_PROVISION_REQUIRED"
        status = {
            "distribution": self.distribution,
            "runtime_available": runtime_available,
            "resources_ready": resources_ready,
            "provisioned": provisioned,
            "issue": issue,
        }
        return status

    # 功能：
    #   读取本地适配器登记的活动摄像头和 GPS 显示来源，忽略无效配置。
    # 输入：
    #   self：持有应用数据目录的运行管理器。
    # 输出：
    #   result：前端可用的来源描述列表。
    def _configured_live_sources(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        source_file = self.store.root / "live-sources.json"
        try:
            value = decode_json(read_plugin_file(source_file, limit=128 * 1024), limit=128 * 1024)
        except (OSError, ValueError):
            return result
        entries = value.get("sources", []) if isinstance(value, dict) else []
        if not isinstance(entries, list) or len(entries) > 256:
            return result
        for index, item in enumerate(entries):
            if not isinstance(item, dict) or item.get("active") is not True:
                continue
            kind = str(item.get("kind", ""))
            url = str(item.get("url", "")).strip()
            default_transport = "video" if kind == "camera" else "json"
            transport = str(item.get("transport", default_transport))
            allowed_transports = {"camera": {"video", "image"}, "gps": {"json"}}
            if (
                kind not in allowed_transports
                or transport not in allowed_transports[kind]
                or not url.startswith(("http://", "https://"))
            ):
                continue
            identifier = re.sub(r"[^a-zA-Z0-9._-]", "-", str(item.get("id", index)))[:80]
            result.append(
                {
                    "id": f"registered-{identifier}",
                    "kind": kind,
                    "label": str(item.get("label", "Camera" if kind == "camera" else "GPS"))[:120],
                    "url": url,
                    "transport": transport,
                    "mode": "hardware",
                    "ready": True,
                }
            )
        return result

    # 功能：
    #   合并当前仿真的显示来源与已登记硬件来源，按实际进程状态确定显示模式。
    # 输入：
    #   self：管理活动进程和本地来源配置的管理器。
    #   thread_id：需要查看的任务标识。
    # 输出：
    #   catalog：任务显示模式和来源列表。
    def live_sources(self, thread_id: str) -> dict[str, object]:
        with self._lock:
            active = self._active_runs.get(thread_id)
        simulation_active = active is not None and active.process.poll() is None
        sources: list[dict[str, object]] = []
        if simulation_active and active is not None:
            sources.extend(
                (
                    {
                        "id": "gazebo-render",
                        "kind": "simulation",
                        "label": "Gazebo",
                        "transport": "agent-core-frame",
                        "mode": "simulation",
                        "ready": (active.run_dir / "live-frame.png").is_file(),
                    },
                    {
                        "id": "gazebo-position",
                        "kind": "gps",
                        "label": "Simulation position",
                        "transport": "agent-core-telemetry",
                        "mode": "simulation",
                        "ready": (active.run_dir / "live-telemetry.json").is_file(),
                    },
                )
            )
        configured = self._configured_live_sources()
        sources.extend(configured)
        mode = "simulation" if simulation_active else ("hardware" if configured else "idle")
        catalog = {
            "schema_version": "dronedream.live-source-catalog.v1",
            "thread_id": thread_id,
            "mode": mode,
            "sources": sources,
        }
        return catalog

    # 功能：
    #   查找活动仿真进程的显示帧；不将预览帧是否存在用作感知控制准入条件。
    # 输入：
    #   self：当前活动运行的管理器。
    #   thread_id：需要查看的任务标识。
    # 输出：
    #   frame：存在的显示帧路径，无活动进程或文件时为空。
    def live_frame(self, thread_id: str) -> Path | None:
        frame = None
        with self._lock:
            active = self._active_runs.get(thread_id)
        if active is None or active.process.poll() is not None:
            return frame
        path = active.run_dir / "live-frame.png"
        frame = path if path.is_file() else None
        return frame

    # 功能：
    #   有界读取活动仿真的前端显示遥测，丢弃损坏、含糊或非对象内容。
    # 输入：
    #   self：当前活动运行的管理器。
    #   thread_id：需要查看的任务标识。
    # 输出：
    #   telemetry：有效的显示遥测字典，无可用数据时为空。
    def live_telemetry(self, thread_id: str) -> dict[str, object] | None:
        telemetry = None
        with self._lock:
            active = self._active_runs.get(thread_id)
        if active is None or active.process.poll() is not None:
            return telemetry
        path = active.run_dir / "live-telemetry.json"
        try:
            value = decode_json(read_plugin_file(path, limit=64 * 1024), limit=64 * 1024)
        except (OSError, ValueError):
            return telemetry
        telemetry = value if isinstance(value, dict) else None
        return telemetry

    # 功能：
    #   实际探测 ROS、Gazebo、PX4 版本及本地资源摘要，不接受客户端自报环境。
    # 输入：
    #   self：负责访问固定 Runtime 的管理器。
    # 输出：
    #   versions：完整版本快照，缺项或探测失败时为空字典。
    def qualification_environment_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        current = self.status()
        if not current["runtime_available"] or not current["provisioned"]:
            return versions
        try:
            probe = self._bash(
                "set -e; set +u; source /opt/ros/jazzy/setup.bash; set -u; "
                "export GZ_CONFIG_PATH=/usr/share/gz:${GZ_CONFIG_PATH:-}; "
                "printf 'ros_distribution=%s\\n' \"$ROS_DISTRO\"; "
                "printf 'gazebo_sim=%s\\n' \"$(gz sim --versions 2>/dev/null | head -n 1)\"; "
                "printf 'px4_commit=%s\\n' "
                '"$(git -C /opt/PX4-Autopilot rev-parse HEAD)"',
                timeout=15,
            )
        except RuntimeBridgeError:
            return versions
        if probe.returncode != 0:
            return versions
        for line in probe.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator and key in {"ros_distribution", "gazebo_sim", "px4_commit"} and value:
                versions[key] = value
        if set(versions) != {"ros_distribution", "gazebo_sim", "px4_commit"}:
            versions = {}
            return versions
        try:
            versions["runtime_manifest_sha256"] = self._runtime_manifest_sha256()
        except RuntimeBridgeError:
            versions = {}
            return versions
        return versions

    # 功能：
    #   1. 在已固定的 WSL Runtime 中执行本次资产配对，返回实际证据文件内容。
    #   2. 启动前后检查取消请求，防止登记子进程前的取消丢失；证据仍须上层验收。
    # 输入：
    #   self：负责当前运行的桌面桥接器。
    #   job_id：配对验收任务标识。
    #   work_root：已经冻结计划与执行材料的目录。
    #   run_dir：本次新的运行证据目录。
    #   timeout_seconds：允许执行的总等待秒数。
    #   cancel_requested：可选的实时取消状态读取器。
    # 输出：
    #   evidence：实际运行写入且通过有界 JSON 解析的证据字典。
    def run_asset_pair_qualification(
        self,
        *,
        job_id: str,
        work_root: Path,
        run_dir: Path,
        timeout_seconds: float = 900.0,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        if cancel_requested is not None and cancel_requested():
            raise RuntimeBridgeError("ASSET_QUALIFICATION_CANCELLED")
        current = self.status()
        if not current["runtime_available"]:
            raise RuntimeBridgeError("DRONEDREAM_RUNTIME_UNAVAILABLE")
        if not current["resources_ready"]:
            raise RuntimeBridgeError("RUNTIME_RESOURCES_MISSING")
        if not current["provisioned"]:
            raise RuntimeBridgeError("RUNTIME_PROVISION_REQUIRED")
        if (
            type(timeout_seconds) not in (int, float)
            or not 60 <= timeout_seconds <= 3_600
            or not math.isfinite(timeout_seconds)
        ):
            raise RuntimeBridgeError("ASSET_QUALIFICATION_TIMEOUT_INVALID")
        if self.resource_root is None:
            raise RuntimeBridgeError("RUNTIME_RESOURCES_MISSING")
        executor = self.resource_root / "runtime" / "px4_offboard_track_executor.py"
        if not executor.is_file():
            raise RuntimeBridgeError("RUNTIME_ADAPTER_RESOURCE_MISSING")
        work_root = work_root.resolve()
        run_dir = run_dir.resolve()
        if not (work_root / "qualification-plan.json").is_file():
            raise RuntimeBridgeError("ASSET_QUALIFICATION_PLAN_MISSING")
        # The core qualification runner deliberately requires a new or empty
        # evidence directory.  Keep the desktop bridge logs beside that
        # directory so opening the log streams cannot make the runner reject
        # its own destination before Gazebo starts.
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir.with_name(f"{run_dir.name}.desktop.stdout.log")
        stderr_path = run_dir.with_name(f"{run_dir.name}.desktop.stderr.log")
        command = [
            "-m",
            "dronedream_agent_core.asset_qualification_cli",
            "--work-root",
            _wsl_path(work_root),
            "--run-dir",
            _wsl_path(run_dir),
            "--px4-root",
            "/opt/PX4-Autopilot",
            "--executor",
            _wsl_path(executor),
        ]
        command_text = " ".join(shlex.quote(value) for value in command)
        script = (
            f"set -eo pipefail; root={self.runtime_home}; set +u; "
            "source /opt/ros/jazzy/setup.bash; "
            'source "$root/ros_ws-merged/install/setup.bash"; set -u; '
            "export ROS_DOMAIN_ID=74 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
            "ROS2_DISABLE_DAEMON=1; "
            f'exec "$root/venv/bin/python" {command_text} '
            '--ros-workspace "$root/ros_ws-merged"'
        )
        with self._lock:
            if job_id in self._active_asset_qualification_runs:
                raise RuntimeBridgeError("ASSET_QUALIFICATION_ALREADY_RUNNING")
        with (
            stdout_path.open("x", encoding="utf-8") as stdout,
            stderr_path.open("x", encoding="utf-8") as stderr,
        ):
            with self._lock:
                # 创建与登记在同一临界区，避免两个调用都通过较早的空闲检查。
                if job_id in self._active_asset_qualification_runs:
                    raise RuntimeBridgeError("ASSET_QUALIFICATION_ALREADY_RUNNING")
                try:
                    process = subprocess.Popen(
                        ["wsl.exe", "-d", self.distribution, "--", "bash", "-s", "--"],
                        stdin=subprocess.PIPE,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                    )
                except OSError as error:
                    raise RuntimeBridgeError("DRONEDREAM_RUNTIME_UNAVAILABLE") from error
                self._active_asset_qualification_runs[job_id] = ActiveAssetQualificationRun(
                    run_dir=run_dir,
                    process=process,
                )
            process_reaped = False
            try:
                assert process.stdin is not None
                # 此时进程等待 stdin，尚未执行飞行脚本；取消不应启动后再等待飞行器响应。
                if cancel_requested is not None and cancel_requested():
                    raise RuntimeBridgeError("ASSET_QUALIFICATION_CANCELLED")
                process.stdin.write(script)
                process.stdin.close()
                try:
                    return_code = process.wait(timeout=timeout_seconds)
                    process_reaped = True
                except subprocess.TimeoutExpired as error:
                    self.cancel_asset_pair_qualification(job_id)
                    try:
                        process.wait(timeout=15)
                        process_reaped = True
                    except subprocess.TimeoutExpired:
                        pass
                    raise RuntimeBridgeError("ASSET_QUALIFICATION_RUNTIME_TIMEOUT") from error
            except BaseException:
                if not process_reaped:
                    try:
                        self._stop_owned_process(process)
                        process_reaped = True
                    except Exception as error:
                        # 未确认结束时保留登记，不能让下一次启动以为此任务已经空闲。
                        raise RuntimeBridgeError("ASSET_QUALIFICATION_CLEANUP_FAILED") from error
                raise
            finally:
                with suppress(OSError, ValueError):
                    if process.stdin is not None:
                        process.stdin.close()
                if process_reaped:
                    with self._lock:
                        self._active_asset_qualification_runs.pop(job_id, None)
        evidence_path = run_dir / "mission_evidence.json"
        if return_code != 0:
            raise RuntimeBridgeError("ASSET_QUALIFICATION_RUNTIME_FAILED")
        if not evidence_path.is_file():
            raise RuntimeBridgeError("ASSET_QUALIFICATION_EVIDENCE_MISSING")
        try:
            evidence = decode_json(
                read_plugin_file(evidence_path, limit=16 * 1024 * 1024),
                limit=16 * 1024 * 1024,
                node_limit=2_000_000,
            )
        except (OSError, ValueError) as error:
            raise RuntimeBridgeError("ASSET_QUALIFICATION_EVIDENCE_INVALID") from error
        if not isinstance(evidence, dict):
            raise RuntimeBridgeError("ASSET_QUALIFICATION_EVIDENCE_INVALID")
        return evidence

    # 功能：
    #   结束并回收本次创建的宿主进程；普通终止无响应时升级为强制终止。
    #   此处不把 wsl.exe 退出当作 Linux 内所有后代已退出的证明。
    # 输入：
    #   process：当前运行拥有的进程对象。
    # 输出：
    #   None：不返回业务数据。
    @staticmethod
    def _stop_owned_process(process: subprocess.Popen[str]) -> None:
        # 进程可能在异常处理前自行退出；仍须 wait 回收其状态。
        with suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    # 功能：
    #   向已登记运行写入带任务身份的中止请求，不把请求送达冒称为已经停止。
    # 输入：
    #   self：持有活动运行索引的桥接器。
    #   job_id：需要中止的配对验收任务标识。
    # 输出：
    #   requested：是否找到了活动运行并写入中止请求。
    def cancel_asset_pair_qualification(self, job_id: str) -> bool:
        with self._lock:
            active = self._active_asset_qualification_runs.get(job_id)
        requested = False
        if active is None:
            return requested
        self._atomic_json(
            active.run_dir / "live_abort.request.json",
            {
                "schema_version": "dronedream.runtime-abort-request.v1",
                "reason": "qualification_cancelled_by_user",
                "job_id": job_id,
                "requested_at": datetime.now(UTC).isoformat(),
            },
        )
        requested = True
        return requested

    # 功能：
    #   在锁内更新安装阶段、百分比、活动状态与错误，保持查询获得一致快照。
    # 输入：
    #   self：持有安装进度的管理器。
    #   phase：当前安装阶段标识。
    #   progress：当前百分比，限制在 0 至 100。
    #   active：安装是否仍在进行。
    #   error：可选错误代码。
    # 输出：
    #   None：不返回业务数据。
    def _set_setup_progress(
        self,
        phase: str,
        progress: int,
        *,
        active: bool = True,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._setup_snapshot.update(
                {
                    "phase": phase,
                    "progress": max(0, min(100, progress)),
                    "active": active,
                    "error": error,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )

    # 功能：
    #   返回独立的安装进度快照，避免调用方直接修改内部记录。
    # 输入：
    #   self：持有安装进度的管理器。
    # 输出：
    #   progress：当前安装状态字典。
    def setup_progress(self) -> dict[str, object]:
        with self._lock:
            progress = dict(self._setup_snapshot)
        return progress

    # 功能：
    #   启动单个后台安装任务；进度由实际阶段更新，启动或安装失败都结束活动状态。
    # 输入：
    #   self：当前运行管理器。
    # 输出：
    #   progress：已有安装任务或新任务的即时进度快照。
    def start_setup(self) -> dict[str, object]:
        with self._lock:
            if self._setup_thread is not None and self._setup_thread.is_alive():
                progress = dict(self._setup_snapshot)
                return progress
            operation_id = f"runtime-setup-{uuid4().hex}"
            now = datetime.now(UTC).isoformat()
            self._setup_snapshot = {
                "schema_version": "dronedream.autonomy.runtime-setup.v1",
                "operation_id": operation_id,
                "phase": "queued",
                "progress": 0,
                "active": True,
                "error": None,
                "failed_phase": None,
                "started_at": now,
                "updated_at": now,
            }

            # 功能：
            #   执行安装步骤，并把失败阶段保存到进度中，不让计时器伪造完成。
            # 输入：
            #   无。
            # 输出：
            #   None：不返回业务数据。
            def worker() -> None:
                try:
                    self._provision_once(self._set_setup_progress)
                except RuntimeBridgeError as error:
                    current = self.setup_progress()
                    self._set_setup_progress(
                        "failed",
                        int(current["progress"]),
                        active=False,
                        error=str(error),
                    )
                    with self._lock:
                        self._setup_snapshot["failed_phase"] = current["phase"]
                except Exception as error:  # pragma: no cover - defensive boundary
                    current = self.setup_progress()
                    self._set_setup_progress(
                        "failed",
                        int(current["progress"]),
                        active=False,
                        error=f"RUNTIME_SETUP_INTERNAL:{type(error).__name__}",
                    )
                    with self._lock:
                        self._setup_snapshot["failed_phase"] = current["phase"]

            self._setup_thread = threading.Thread(
                target=worker,
                name="dronedream-runtime-setup",
                daemon=True,
            )
            try:
                self._setup_thread.start()
            except Exception as error:
                self._setup_thread = None
                self._set_setup_progress(
                    "failed", 0, active=False, error="RUNTIME_SETUP_WORKER_START_FAILED"
                )
                self._setup_snapshot["failed_phase"] = "queued"
                raise RuntimeBridgeError("RUNTIME_SETUP_WORKER_START_FAILED") from error
            progress = dict(self._setup_snapshot)
            return progress

    # 功能：
    #   使用本地已校验资源安装依赖和构建合并 ROS 工作空间，再做真实就绪探测。
    # 输入：
    #   self：负责固定 Runtime 的管理器。
    #   report：可选的阶段进度接收器。
    # 输出：
    #   verified：安装完成后的实际运行环境状态。
    def _perform_provision(self, report=None) -> dict[str, object]:
        # 功能：
        #   将当前阶段里程碑交给可选进度接收器。
        # 输入：
        #   phase：阶段标识。
        #   progress：该阶段报告的百分比。
        # 输出：
        #   None：不返回业务数据。
        def checkpoint(phase: str, progress: int) -> None:
            if report is not None:
                report(phase, progress)

        checkpoint("validatingResources", 5)
        if self.resource_root is None:
            raise RuntimeBridgeError("RUNTIME_RESOURCES_MISSING")
        runtime_resources = self.resource_root / "runtime"
        wheels = runtime_resources / "wheels"
        requirements = runtime_resources / "requirements-linux.txt"
        ros_source = runtime_resources / "ros_ws" / "src"
        if not self._resources_ready():
            raise RuntimeBridgeError("RUNTIME_RESOURCES_MISSING")
        checkpoint("checkingBaseRuntime", 12)
        current = self.status()
        if not current["runtime_available"]:
            raise RuntimeBridgeError(str(current["issue"]))
        wheels_wsl = shlex.quote(_wsl_path(wheels))
        requirements_wsl = shlex.quote(_wsl_path(requirements))
        ros_source_wsl = shlex.quote(_wsl_path(ros_source))
        manifest_sha256 = self._runtime_manifest_sha256()
        root = f"root={self.runtime_home}; "
        steps = (
            (
                "preparingEnvironment",
                25,
                root
                + 'mkdir -p "$root" "$root/ros_ws-merged"; '
                + 'if [ ! -x "$root/venv/bin/python" ]; then '
                + 'python3 -m venv --system-site-packages "$root/venv"; fi',
                90,
                None,
            ),
            (
                "installingCore",
                42,
                root
                + '"$root/venv/bin/pip" install --no-index --no-deps --force-reinstall '
                + f"--find-links {wheels_wsl} dronedream-flight-agent-core",
                180,
                None,
            ),
            (
                "installingDependencies",
                58,
                root
                + '"$root/venv/bin/pip" install --no-index --force-reinstall '
                + f"--find-links {wheels_wsl} -r {requirements_wsl}",
                300,
                None,
            ),
            (
                "buildingRosWorkspace",
                76,
                root
                + 'colcon --log-base "$root/ros_ws-merged/log" build --merge-install '
                + f"--base-paths {ros_source_wsl} "
                + '--build-base "$root/ros_ws-merged/build" '
                + '--install-base "$root/ros_ws-merged/install" '
                + "--cmake-clean-cache",
                600,
                None,
            ),
            (
                "runningHealthChecks",
                92,
                root
                + 'set +u; source "$root/ros_ws-merged/install/setup.bash"; set -u; '
                + "ros2 interface show dronedream_agent_msgs/msg/MissionObservation "
                + ">/dev/null; "
                + 'python3 -c "from dronedream_agent_msgs.msg import '
                + "MissionObservation; print('ROS_MESSAGES_READY')\"; "
                + '"$root/venv/bin/python" -c "import onnxruntime; from PIL import Image; '
                + "Image.new('RGB', (2,2)).resize((1,1)); "
                + "print('LOCAL_POLICY_RUNTIME_READY')\"; "
                + "ros2 run dronedream_agent_plugin_api plugin_probe | "
                + "grep -q PLUGIN_PROBE_READY; "
                + "ros2 run dronedream_agent_plugin_api capability_host --self-test | "
                + "grep -q PLUGIN_LIFECYCLE_READY; "
                + '"$root/venv/bin/python" -c '
                + "\"import openai,pydantic,mavsdk; print('AUTONOMY_RUNTIME_READY')\"; "
                + '"$root/venv/bin/dronedream-agent" --help >/dev/null',
                120,
                "ROS_MESSAGES_READY",
            ),
            (
                "recordingReceipt",
                98,
                root
                + f"printf '%s\\n' {shlex.quote(manifest_sha256)} > "
                + '"$root/.runtime-manifest-sha256"',
                30,
                None,
            ),
        )
        for phase, progress, body, timeout, required_output in steps:
            checkpoint(phase, progress)
            result = self._bash(
                f"set -eo pipefail; source /opt/ros/jazzy/setup.bash; set -u; {body}",
                timeout=timeout,
            )
            if result.returncode != 0 or (
                required_output is not None and required_output not in result.stdout
            ):
                detail = (result.stderr or result.stdout)[-1_000:]
                raise RuntimeBridgeError(f"RUNTIME_PROVISION_FAILED:{phase}:{detail}")
        verified = self.status()
        if not verified["provisioned"]:
            raise RuntimeBridgeError("RUNTIME_PROVISION_VERIFICATION_FAILED")
        checkpoint("completed", 100)
        if report is not None:
            report("completed", 100, active=False)
        return verified

    # 功能：
    #   串行化同步与后台安装入口，拒绝并发修改同一 Runtime 工作空间。
    # 输入：
    #   self：当前运行管理器。
    #   report：可选的安装进度接收器。
    # 输出：
    #   verified：安装及就绪核对后的状态。
    def _provision_once(self, report=None) -> dict[str, object]:
        if not self._provision_lock.acquire(blocking=False):
            raise RuntimeBridgeError("RUNTIME_PROVISION_ALREADY_ACTIVE")
        try:
            verified = self._perform_provision(report)
        finally:
            self._provision_lock.release()
        return verified

    # 功能：
    #   同步执行互斥的部署流程，供内部调用方等待最终验证结果。
    # 输入：
    #   self：当前运行管理器。
    # 输出：
    #   verified：部署后的实际运行环境状态。
    def provision(self) -> dict[str, object]:
        verified = self._provision_once()
        return verified

    # 功能：
    #   将模型连接信息编码为带 shell 转义的环境变量，不记录或显示凭据内容。
    # 输入：
    #   provider：供应商标识。
    #   model_id：模型标识。
    #   grant：本次连接所用凭据。
    #   gateway：模型服务地址。
    #   api_style：接口协议标识。
    # 输出：
    #   lines：以 LF 分行的环境变量导出语句。
    @staticmethod
    def _secret_lines(
        provider: str, model_id: str, grant: str, gateway: str, api_style: str = "chat-completions"
    ) -> str:
        prefix = re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_") or "CUSTOM"
        values = {
            f"{prefix}_API_KEY": grant,
            f"{prefix}_BASE_URL": gateway,
            f"{prefix}_MODEL": model_id,
        }
        if api_style:
            values[f"{prefix}_API_STYLE"] = api_style
        lines = "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())
        return lines

    # 功能：
    #   独占创建供 WSL 单次读取的会话文件，保留 LF 换行，写入失败仅清理自身文件。
    # 输入：
    #   cls：提供连接信息编码方法的管理器类。
    #   path：本次新建的会话文件路径。
    #   provider：供应商标识。
    #   model_id：模型标识。
    #   grant：本次连接所用凭据。
    #   gateway：模型服务地址。
    #   api_style：接口协议标识。
    # 输出：
    #   None：不返回业务数据。
    @classmethod
    def _write_secret_session(
        cls,
        path: Path,
        *,
        provider: str,
        model_id: str,
        grant: str,
        gateway: str,
        api_style: str = "chat-completions",
    ) -> None:
        check_plain_plugin_path(path)
        content = cls._secret_lines(provider, model_id, grant, gateway, api_style)
        owned = None
        try:
            with path.open("x", encoding="utf-8", newline="\n") as output:
                owned = os.fstat(output.fileno())
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            if owned is not None:
                with suppress(OSError):
                    if os.path.samestat(owned, path.stat(follow_symlinks=False)):
                        path.unlink()
            raise

    # 功能：
    #   1. 复核用户确认的计划、账号范围、插件快照、资产资格和准备材料绑定。
    #   2. 将云端规划供应商与本地专家配置交给固定 Runtime，记录执行并监视退出。
    #   3. 配置校验完成后才消费单次执行授权和写入临时连接文件，不调用模型伪造飞行结果。
    # 输入：
    #   self：管理任务、插件、产品资源与活动进程的桥接器。
    #   thread_id：已经等待用户确认的任务标识。
    #   connection：本任务选中的模型连接与凭据。
    #   plan_revision_id：用户确认的精确计划修订标识。
    #   owner_scope：账号与租户权限范围。
    # 输出：
    #   launched：执行标识、当前状态及本次证据目录。
    def execute(
        self,
        *,
        thread_id: str,
        connection: ModelConnection,
        plan_revision_id: str,
        owner_scope: MemoryOwnerScope,
    ) -> dict[str, object]:
        if not self.status()["provisioned"]:
            raise RuntimeBridgeError("RUNTIME_PROVISION_REQUIRED")
        if self.resource_root is None:
            raise RuntimeBridgeError("RUNTIME_RESOURCES_MISSING")
        thread = self.store.get_thread(thread_id)
        if thread["state"] != "awaiting_confirmation":
            raise RuntimeBridgeError("TASK_NOT_AWAITING_CONFIRMATION")
        if connection.selection_id != thread["selected_model"]:
            raise RuntimeBridgeError("EXECUTION_MODEL_MISMATCH")
        with self._lock:
            if thread_id in self._active_runs:
                raise RuntimeBridgeError("TASK_ALREADY_EXECUTING")
        plan = self.store.latest_plan(thread_id)
        try:
            if str(plan["plan_revision_id"]) != plan_revision_id:
                raise RuntimeBridgeError("EXECUTION_PLAN_REVISION_MISMATCH")
            boundary = plan["model_harness_control_plane"]
            if not isinstance(boundary, dict):
                raise ValueError("boundary must be an object")
            control_plane = HarnessControlPlaneReceipt.model_validate(boundary["receipt"])
            runtime_envelope = HarnessRuntimeEnvelope.model_validate(boundary["runtime_envelope"])
            input_envelope = HarnessInputEnvelope.model_validate(boundary["input_envelope"])
            output_envelope = HarnessOutputEnvelope.model_validate(boundary["output_envelope"])
            validate_output_against_boundaries(
                control_plane,
                runtime_envelope,
                output_envelope,
                input_envelope=input_envelope,
            )
            authority = ModelHarnessExecutionAuthority.model_validate(plan["execution_authority"])
        except RuntimeBridgeError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeBridgeError("EXECUTION_MODEL_HARNESS_BINDING_INVALID") from error
        owner_binding, tenant_binding = owner_scope_bindings(owner_scope)
        if (
            authority.thread_id != thread_id
            or authority.plan_revision_id != plan_revision_id
            or authority.contract_id != str(plan["contract_id"])
            or authority.owner_binding_sha256 != owner_binding
            or authority.tenant_binding_sha256 != tenant_binding
            or runtime_envelope.owner_binding_sha256 != owner_binding
            or runtime_envelope.tenant_binding_sha256 != tenant_binding
            or authority.control_plane_selection_sha256 != control_plane.selection_sha256
            or runtime_envelope.control_plane_selection_sha256 != control_plane.selection_sha256
        ):
            raise RuntimeBridgeError("EXECUTION_AUTHORITY_SCOPE_MISMATCH")
        map_content_sha256 = thread.get("selected_map_content_sha256")
        vehicle_content_sha256 = thread.get("selected_vehicle_content_sha256")
        if (map_content_sha256 is None) != (vehicle_content_sha256 is None):
            raise RuntimeBridgeError("EXECUTION_ASSET_VERSION_PAIR_INCOMPLETE")
        try:
            pair_binding: MissionAssetPairQualificationBinding | None = None
            if map_content_sha256 is None or vehicle_content_sha256 is None:
                raise RuntimeBridgeError("EXECUTION_ASSET_VERSION_PAIR_REQUIRED")
            map_record = self.store.get_asset_version(
                str(thread["selected_map_id"]), str(map_content_sha256)
            )
            vehicle_record = self.store.get_asset_version(
                str(thread["selected_vehicle_id"]), str(vehicle_content_sha256)
            )
            pair_qualification = self.store.qualified_asset_pair(
                map_asset_id=str(thread["selected_map_id"]),
                map_content_sha256=str(map_content_sha256),
                vehicle_asset_id=str(thread["selected_vehicle_id"]),
                vehicle_content_sha256=str(vehicle_content_sha256),
            )
            if pair_qualification is None:
                raise RuntimeBridgeError("EXECUTION_ASSET_PAIR_QUALIFICATION_REQUIRED")
            pair_binding = self.store.verified_asset_pair_execution_binding(pair_qualification)
            environment_versions = self.qualification_environment_versions()
            if not environment_versions or not all(
                qualification_matches_environment(record, environment_versions)
                for record in (map_record, vehicle_record)
            ):
                raise RuntimeBridgeError("EXECUTION_ASSET_QUALIFICATION_STALE")
            map_selection = resolve_versioned_map(map_record)
            vehicle_selection = resolve_versioned_vehicle(vehicle_record)
        except (AssetRuntimeResolutionError, KeyError) as error:
            raise RuntimeBridgeError(str(error)) from error
        provider = connection.provider
        provider_capability_id = connection.capability_id
        gateway = connection.base_url
        snapshot_record = self.store.latest_plugin_snapshot(thread_id)
        snapshot = PluginSnapshot.model_validate(snapshot_record["snapshot"])
        if (
            snapshot.snapshot_id != runtime_envelope.plugin_snapshot_id
            or snapshot.catalog_sha256 != runtime_envelope.plugin_catalog_sha256
            or snapshot.snapshot_id != authority.plugin_snapshot_id
            or snapshot.catalog_sha256 != authority.plugin_catalog_sha256
            or snapshot.snapshot_id != str(plan.get("plugin_snapshot_id", ""))
            or snapshot.catalog_sha256 != str(plan.get("plugin_catalog_sha256", ""))
        ):
            raise RuntimeBridgeError("EXECUTION_PLUGIN_SNAPSHOT_DRIFT")
        try:
            _runtime_entry, _runtime_manifest, runtime_capability = (
                self.plugin_manager.capability_for_slot(snapshot, "simulation.runtime-adapter")
            )
            _interrupt_entry, _interrupt_manifest, interrupt_capability = (
                self.plugin_manager.capability_for_slot(snapshot, "safety.interruption-policy")
            )
        except PluginManagerError as error:
            raise RuntimeBridgeError(str(error)) from error
        required_capabilities = {
            runtime_capability.capability_id,
            interrupt_capability.capability_id,
            provider_capability_id,
        }
        try:
            self.plugin_manager.assert_snapshot_active(snapshot, required_capabilities)
        except (KeyError, PluginManagerError) as error:
            raise RuntimeBridgeError(str(error)) from error
        execution_id = f"execution-{uuid4().hex}"
        execution_root = self.store.missions_root / thread_id / execution_id
        run_dir = execution_root / "simulation"
        prepared = Path(str(plan["output_dir"])) / "prepared-mission.json"
        authority_path = Path(str(plan["output_dir"])) / "model-harness-execution-authority.json"
        world = map_selection.world_sdf
        semantic = map_selection.semantic
        map_graph = map_selection.graph
        vehicle = vehicle_selection.vehicle_sdf
        controller = vehicle_selection.controller_params
        vehicle_metadata = vehicle_selection.vehicle_metadata
        required = (
            prepared,
            authority_path,
            world,
            semantic,
            map_graph,
            vehicle,
            controller,
            vehicle_metadata,
        )
        if any(not path.is_file() for path in required):
            raise RuntimeBridgeError("EXECUTION_ARTIFACT_MISSING")
        try:
            prepared_mission = PreparedMission.model_validate(
                decode_json(
                    read_plugin_file(prepared, limit=16 * 1024 * 1024),
                    limit=16 * 1024 * 1024,
                    node_limit=2_000_000,
                )
            )
            if prepared_mission.schema_version != "dronedream.prepared-mission.v4":
                raise RuntimeBridgeError("PREPARED_MISSION_SCHEMA_OBSOLETE")
            authority_artifact = ModelHarnessExecutionAuthority.model_validate(
                decode_json(
                    read_plugin_file(authority_path, limit=1024 * 1024),
                    limit=1024 * 1024,
                    node_limit=100_000,
                )
            )
            if authority_artifact != authority:
                raise RuntimeBridgeError("EXECUTION_AUTHORITY_ARTIFACT_MISMATCH")
            prepared_sha256 = sha256_json(prepared_mission)
            if (
                authority.prepared_mission_sha256 != prepared_sha256
                or prepared_mission.contract.conversation_id != thread_id
                or prepared_mission.contract.contract_id != authority.contract_id
                or prepared_mission.plugin_snapshot.snapshot_id != snapshot.snapshot_id
                or prepared_mission.plugin_snapshot.catalog_sha256 != snapshot.catalog_sha256
                or prepared_mission.contract.map_asset_id != map_selection.asset_id
                or prepared_mission.contract.vehicle_asset_id != vehicle_selection.asset_id
                or prepared_mission.contract.map_sha256
                != hash_plugin_file(map_graph, limit=256 * 1024 * 1024)
                or prepared_mission.contract.map_semantic_sha256
                != hash_plugin_file(semantic, limit=256 * 1024 * 1024)
                or prepared_mission.contract.vehicle_sha256
                != hash_plugin_file(vehicle, limit=256 * 1024 * 1024)
            ):
                raise RuntimeBridgeError("EXECUTION_ASSET_PLAN_BINDING_MISMATCH")
            if pair_binding is not None:
                binding_records = [
                    record
                    for record in prepared_mission.evidence
                    if record.event_type == "mission.asset-pair-qualification"
                ]
                if len(binding_records) != 1:
                    raise RuntimeBridgeError("EXECUTION_ASSET_QUALIFICATION_BINDING_REQUIRED")
                planned_pair_binding = MissionAssetPairQualificationBinding.model_validate(
                    binding_records[0].payload
                )
                if planned_pair_binding != pair_binding:
                    raise RuntimeBridgeError("EXECUTION_ASSET_QUALIFICATION_BINDING_MISMATCH")
            simulation_contract = prepared_mission.simulation_capabilities
            simulator_contract = simulation_contract.get("simulator", {})
            native_contract = simulation_contract.get("native_runtime", {})
            transport_contract = (
                native_contract.get("transport", {}) if isinstance(native_contract, dict) else {}
            )
            watchdog_contract = (
                native_contract.get("watchdog", {}) if isinstance(native_contract, dict) else {}
            )
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeBridgeError("EXECUTION_PLUGIN_CONTRACT_INVALID") from error
        if not isinstance(simulator_contract, dict) or simulator_contract.get("engine") != (
            "gazebo-harmonic"
        ):
            raise RuntimeBridgeError("SIMULATOR_ADAPTER_NOT_EXECUTABLE")
        if (
            not isinstance(transport_contract, dict)
            or transport_contract.get("implementation") != "px4-uxrce-dds"
        ):
            raise RuntimeBridgeError("FLIGHT_TRANSPORT_NOT_EXECUTABLE")
        if not isinstance(watchdog_contract, dict):
            raise RuntimeBridgeError("RUNTIME_WATCHDOG_CONTRACT_INVALID")
        try:
            watchdog_deadline_ms = bounded_integer(
                watchdog_contract.get("deadline_ms", 1_000), 20, 1_000
            )
            watchdog_startup_deadline_ms = bounded_integer(
                watchdog_contract.get("startup_deadline_ms", 10_000), 1_000, 60_000
            )
            watchdog_scheduling_jitter_grace_ms = bounded_integer(
                watchdog_contract.get("scheduling_jitter_grace_ms", 250), 0, 500
            )
            watchdog_persistent_miss_ms = bounded_integer(
                watchdog_contract.get("persistent_miss_ms", 250), 25, 1_000
            )
        except (TypeError, ValueError) as error:
            raise RuntimeBridgeError("RUNTIME_WATCHDOG_CONTRACT_INVALID") from error
        if watchdog_startup_deadline_ms < watchdog_deadline_ms:
            raise RuntimeBridgeError("RUNTIME_WATCHDOG_CONTRACT_INVALID")
        secret_path = execution_root / f".model-session-{uuid4().hex}.env"
        runtime_metadata = runtime_capability.metadata
        interrupt_metadata = interrupt_capability.metadata
        runtime = self.resource_root / "runtime"
        (
            local_policy_packages,
            local_policy_qualifications,
            local_policy_simulation_admissions,
        ) = self._local_policy_runtime_resources()
        local_policy_enabled = bool(local_policy_packages)
        try:
            executor = packaged_executor(runtime, runtime_metadata.get("executor", ""))
            checkpoint_executor = packaged_executor(
                runtime, runtime_metadata.get("checkpoint_executor", "")
            )
        except (OSError, ValueError) as error:
            raise RuntimeBridgeError("RUNTIME_ADAPTER_RESOURCE_INVALID") from error
        runtime_distribution = str(runtime_metadata.get("distribution", self.distribution))
        ros_setup = str(runtime_metadata.get("ros_setup", "/opt/ros/jazzy/setup.bash"))
        px4_root = str(runtime_metadata.get("px4_root", "/opt/PX4-Autopilot"))
        runtime_cli = str(runtime_metadata.get("runtime_cli", "execute-prepared-mission"))
        capability_host = str(
            runtime_metadata.get(
                "capability_host", "ros2 run dronedream_agent_plugin_api capability_host"
            )
        )
        try:
            ros_domain_id = bounded_integer(runtime_metadata.get("ros_domain_id", 74), 0, 232)
        except (TypeError, ValueError) as error:
            raise RuntimeBridgeError("RUNTIME_ROS_DOMAIN_INVALID") from error
        try:
            hold_timeout = positive_seconds(interrupt_metadata.get("hold_timeout_seconds", 12.0))
            replan_hold = positive_seconds(interrupt_metadata.get("replan_hold_seconds", 60.0))
            configured_local_navigation_timeout = positive_seconds(
                runtime_metadata.get("local_navigation_model_timeout_seconds", 10.0)
            )
            configured_local_navigation_period = positive_seconds(
                runtime_metadata.get("local_navigation_period_seconds", 3.0)
            )
        except ValueError as error:
            raise RuntimeBridgeError("RUNTIME_LOCAL_NAVIGATION_POLICY_INVALID") from error
        visual_providers = runtime_metadata.get("local_navigation_visual_providers", [])
        if not isinstance(visual_providers, list) or not all(
            isinstance(value, str) and value for value in visual_providers
        ):
            raise RuntimeBridgeError("RUNTIME_LOCAL_NAVIGATION_POLICY_INVALID")
        local_navigation_provider = "local-policy" if local_policy_enabled else provider
        local_navigation_fallback_provider = provider if local_policy_enabled else None
        local_navigation_timeout = (
            min(configured_local_navigation_timeout, 0.25)
            if local_policy_enabled
            else configured_local_navigation_timeout
        )
        local_navigation_period = (
            min(configured_local_navigation_period, 1.0)
            if local_policy_enabled
            else configured_local_navigation_period
        )
        local_navigation_visual_enabled = (
            self._local_policy_packages_have_role(
                local_policy_packages,
                "perception-encoder",
            )
            if local_policy_enabled
            else provider in visual_providers
        )
        heading_policy = (
            "route-tangent-relative" if local_navigation_visual_enabled else "measured-hold"
        )
        if not executor.is_file() or not checkpoint_executor.is_file():
            raise RuntimeBridgeError("RUNTIME_ADAPTER_RESOURCE_MISSING")
        context_db = self.store.root / "mission-context.sqlite3"
        contract_id = str(plan["contract_id"])
        command = [
            "--profile",
            "runtime-manager",
            runtime_cli,
            "--prepared",
            _wsl_path(prepared),
            "--execution-authority",
            _wsl_path(authority_path),
            "--confirm-contract-id",
            contract_id,
            "--completion-provider",
            provider,
            "--run-dir",
            _wsl_path(run_dir),
            "--world-sdf",
            _wsl_path(world),
            "--semantic",
            _wsl_path(semantic),
            "--map-graph",
            _wsl_path(map_graph),
            "--vehicle-sdf",
            _wsl_path(vehicle),
            "--vehicle-metadata",
            _wsl_path(vehicle_metadata),
            "--controller-params",
            _wsl_path(controller),
            "--px4-root",
            px4_root,
            "--executor",
            _wsl_path(executor),
            "--context-db",
            _wsl_path(context_db),
            "--checkpoint-provider",
            provider,
            "--checkpoint-executor",
            _wsl_path(checkpoint_executor),
            "--runtime-interrupt-provider",
            provider,
            "--runtime-hold-timeout-seconds",
            f"{hold_timeout:g}",
            "--runtime-replan-hold-seconds",
            f"{replan_hold:g}",
            "--local-navigation-provider",
            local_navigation_provider,
            "--local-navigation-model-timeout-seconds",
            f"{local_navigation_timeout:g}",
            "--local-navigation-period-seconds",
            f"{local_navigation_period:g}",
            "--local-navigation-context-id",
            contract_id,
            "--require-local-navigation-control-authority",
            "--heading-policy",
            heading_policy,
            "--maximum-yaw-rate-deg-s",
            "20",
        ]
        if local_navigation_fallback_provider is not None:
            command.extend(
                [
                    "--local-navigation-fallback-provider",
                    local_navigation_fallback_provider,
                    "--local-navigation-fallback-model-timeout-seconds",
                    f"{configured_local_navigation_timeout:g}",
                ]
            )
        for package_path in local_policy_packages:
            command.extend(["--local-policy-package", _wsl_path(package_path)])
        for qualification_path in local_policy_qualifications:
            command.extend(["--local-policy-qualification", _wsl_path(qualification_path)])
        for admission_path in local_policy_simulation_admissions:
            command.extend(["--local-policy-simulation-admission", _wsl_path(admission_path)])
        if local_navigation_visual_enabled:
            command.append("--local-navigation-visual-enabled")
        command_text = " ".join(shlex.quote(value) for value in command)
        secret_wsl = shlex.quote(_wsl_path(secret_path))
        native_preflight_marker_wsl = shlex.quote(
            _wsl_path(run_dir / "native-runtime-preflight-ready.json")
        )
        script = (
            f"set -eo pipefail; source {shlex.quote(ros_setup)}; "
            f'root={self.runtime_home}; source "$root/ros_ws-merged/install/setup.bash"; set -u; '
            f"export ROS_DOMAIN_ID={ros_domain_id} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
            "ROS2_DISABLE_DAEMON=1; "
            'export CYCLONEDDS_URI=\'<CycloneDDS><Domain Id="any"><General>'
            '<Interfaces><NetworkInterface address="127.0.0.1"/></Interfaces>'
            "<AllowMulticast>false</AllowMulticast></General><Discovery>"
            "<ParticipantIndex>auto</ParticipantIndex><Peers>"
            '<Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>\'; '
            f"source {secret_wsl}; rm -f {secret_wsl}; "
            f"( until [ -f {native_preflight_marker_wsl} ]; do sleep 0.1; done; "
            f"exec {capability_host} --ros-args "
            f"-p contract_id:={shlex.quote(contract_id)} "
            f"-p watchdog_deadline_ms:={watchdog_deadline_ms} "
            f"-p watchdog_startup_deadline_ms:={watchdog_startup_deadline_ms} "
            f"-p watchdog_scheduling_jitter_grace_ms:="
            f"{watchdog_scheduling_jitter_grace_ms} "
            f"-p watchdog_persistent_miss_ms:={watchdog_persistent_miss_ms}; ) & "
            "plugin_host_pid=$!; "
            "trap 'kill $plugin_host_pid 2>/dev/null || true; "
            "wait $plugin_host_pid 2>/dev/null || true' EXIT; "
            f'"$root/venv/bin/dronedream-agent" {command_text} '
            '--ros-workspace "$root/ros_ws-merged"'
        )
        # 验证失败不应消耗用户确认或留下密钥；直到脚本及全部配置准备好才提交授权。
        try:
            self.store.consume_execution_authority(authority, execution_id=execution_id)
        except (KeyError, ValueError) as error:
            raise RuntimeBridgeError(str(error)) from error
        execution_root.mkdir(parents=True, exist_ok=False)
        stdout = None
        stderr = None
        process = None
        secret_identity = None

        # 功能：
        #   只移除本次创建且身份未改变的临时连接文件，避免误删其他写入者的文件。
        # 输入：
        #   无。
        # 输出：
        #   None：不返回业务数据。
        def clear_secret() -> None:
            if secret_identity is not None:
                with suppress(OSError):
                    if os.path.samestat(secret_identity, secret_path.stat(follow_symlinks=False)):
                        secret_path.unlink()

        try:
            self._write_secret_session(
                secret_path,
                provider=provider,
                model_id=connection.model_id,
                grant=connection.api_key,
                gateway=gateway,
                api_style=connection.api_style,
            )
            secret_identity = secret_path.stat(follow_symlinks=False)
            stdout = (execution_root / "desktop-runtime.stdout.log").open("x", encoding="utf-8")
            stderr = (execution_root / "desktop-runtime.stderr.log").open("x", encoding="utf-8")
            process = subprocess.Popen(
                ["wsl.exe", "-d", runtime_distribution, "--", "bash", "-s", "--"],
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            assert process.stdin is not None
            process.stdin.write(script)
            process.stdin.close()
        except BaseException:
            try:
                if process is not None:
                    try:
                        self._stop_owned_process(process)
                    except Exception as error:
                        # 清理失败的进程仍归此任务所有，不能从活动索引中凭空消失。
                        with self._lock:
                            self._active_runs[thread_id] = ActiveRun(
                                execution_id=execution_id,
                                run_dir=run_dir,
                                process=process,
                                started_at=datetime.now(UTC).isoformat(),
                            )
                        raise RuntimeBridgeError("EXECUTION_CLEANUP_FAILED") from error
            finally:
                with suppress(OSError, ValueError):
                    if process is not None and process.stdin is not None:
                        process.stdin.close()
                if stdout is not None:
                    stdout.close()
                if stderr is not None:
                    stderr.close()
                clear_secret()
            raise
        started_at = datetime.now(UTC).isoformat()
        with self._lock:
            self._active_runs[thread_id] = ActiveRun(
                execution_id=execution_id,
                run_dir=run_dir,
                process=process,
                started_at=started_at,
            )

        # 功能：
        #   回收运行进程、清理会话文件并核对结果身份，损坏或不匹配的结果按失败记录。
        # 输入：
        #   无。
        # 输出：
        #   None：不返回业务数据。
        def monitor() -> None:
            cleanup_failed = False
            try:
                return_code = process.wait()
            except Exception:
                return_code = -1
                try:
                    self._stop_owned_process(process)
                except Exception:
                    cleanup_failed = True
            finally:
                clear_secret()
                with suppress(OSError, ValueError):
                    stdout.close()
                with suppress(OSError, ValueError):
                    stderr.close()
            result_path = run_dir / "workflow-result.json"
            final_state = "failed"
            if return_code == 0 and result_path.is_file():
                try:
                    result = SimulationWorkflowResult.model_validate(
                        decode_json(
                            read_plugin_file(result_path, limit=16 * 1024 * 1024),
                            limit=16 * 1024 * 1024,
                            node_limit=2_000_000,
                        )
                    )
                    # 零退出或 status 字符串不能替代实际结果契约及本次计划身份。
                    if (
                        result.status == "verified"
                        and result.contract_id == authority.contract_id
                        and result.prepared_mission_sha256 == authority.prepared_mission_sha256
                    ):
                        final_state = "completed"
                except (OSError, ValueError):
                    final_state = "failed"
            metadata = {
                "execution_id": execution_id,
                "return_code": return_code,
                "run_dir": str(run_dir),
                "execution_root": str(execution_root),
                "contract_id": authority.contract_id,
                "prepared_mission_sha256": authority.prepared_mission_sha256,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            tracking_error = "EXECUTION_CLEANUP_FAILED" if cleanup_failed else None
            try:
                # 数据库临时不可写时仍保留本次宿主退出记录，不能伪造任务已成功落库。
                self._atomic_json(
                    execution_root / "host-completion.json",
                    {"state": final_state, "cleanup_failed": cleanup_failed, "metadata": metadata},
                    replace_existing=False,
                )
                self.store.set_thread_state(thread_id, final_state)
                english = self._thread_locale(thread_id) == "en-US"
                self.store.append_message(
                    thread_id,
                    role="assistant",
                    kind="status" if final_state == "completed" else "error",
                    content=(
                        "Simulation completed"
                        if english and final_state == "completed"
                        else "Simulation did not pass acceptance"
                        if english
                        else "仿真任务已完成"
                        if final_state == "completed"
                        else "仿真任务未通过验收"
                    ),
                    metadata=metadata,
                )
            except Exception:
                tracking_error = tracking_error or "EXECUTION_COMPLETION_PERSISTENCE_FAILED"
            with self._lock:
                active = self._active_runs.get(thread_id)
                if active is not None and active.process is process:
                    self._takeover_grants.pop(thread_id, None)
                    if tracking_error is None:
                        self._active_runs.pop(thread_id, None)
                    else:
                        # 保留异常状态供界面查询；不能把未完成回收或落库的任务当作已消失。
                        active.tracking_error = tracking_error

        try:
            self.store.set_thread_state(thread_id, "executing")
            threading.Thread(target=monitor, daemon=True).start()
        except Exception as error:
            reaped = False
            try:
                # 脚本可能已开始运行；先投递中止请求，留出合作退出时间再回收宿主。
                with suppress(OSError, ValueError):
                    self._atomic_json(
                        run_dir / "live_abort.request.json",
                        {
                            "schema_version": "dronedream.runtime-abort-request.v1",
                            "reason": "execution_tracking_failed",
                            "execution_id": execution_id,
                            "requested_at": datetime.now(UTC).isoformat(),
                        },
                    )
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._stop_owned_process(process)
                reaped = True
            except Exception as cleanup_error:
                raise RuntimeBridgeError("EXECUTION_CLEANUP_FAILED") from cleanup_error
            finally:
                clear_secret()
                stdout.close()
                stderr.close()
                if reaped:
                    with self._lock:
                        self._active_runs.pop(thread_id, None)
                        self._takeover_grants.pop(thread_id, None)
                    with suppress(KeyError, RuntimeError):
                        self.store.set_thread_state(thread_id, "failed")
            raise RuntimeBridgeError("EXECUTION_TRACKING_FAILED") from error
        launched = {
            "execution_id": execution_id,
            "state": "executing",
            "run_dir": str(run_dir),
            "execution_root": str(execution_root),
        }
        return launched

    # 功能：
    #   读取本任务本次执行的类型化验收摘要，不向界面暴露原始文件路径。
    # 输入：
    #   self：持有任务记录和运行进程的管理器。
    #   thread_id：要查询验收证据的任务标识。
    # 输出：
    #   summary：执行身份、状态、完成时间和可用的验收指标。
    def execution_evidence(self, thread_id: str) -> dict[str, object]:
        thread = self.store.get_thread(thread_id)
        with self._lock:
            active = self._active_runs.get(thread_id)
        if active is not None:
            tracking_error = getattr(active, "tracking_error", None)
            summary = {
                "schema_version": "dronedream.agent-execution-evidence.v1",
                "thread_id": thread_id,
                "execution_id": active.execution_id,
                "state": "failed" if tracking_error else "executing",
                "started_at": active.started_at,
                "completed_at": None,
                "result": None,
            }
            if tracking_error:
                summary["issue"] = tracking_error
            return summary

        completion_message = next(
            (
                message
                for message in reversed(thread.get("messages", []))
                if isinstance(message, dict)
                and message.get("role") == "assistant"
                and message.get("kind") in {"status", "error"}
                and isinstance(message.get("metadata"), dict)
                and message["metadata"].get("execution_id")
            ),
            None,
        )
        if completion_message is None:
            raise KeyError("EXECUTION_EVIDENCE_NOT_FOUND")
        metadata = completion_message["metadata"]
        execution_id = str(metadata["execution_id"])
        try:
            if not re.fullmatch(r"execution-[0-9a-f]{32}", execution_id):
                raise ValueError("invalid execution identifier")
            raw_dir = Path(str(metadata.get("run_dir", "")))
            expected = self.store.missions_root / thread_id / execution_id / "simulation"
            check_plain_plugin_path(raw_dir)
            check_plain_plugin_path(expected)
            run_dir = raw_dir.resolve()
            if not raw_dir.is_absolute() or run_dir != expected.resolve():
                raise ValueError("evidence belongs to another task or execution")
        except (OSError, ValueError) as error:
            raise RuntimeBridgeError("EXECUTION_EVIDENCE_PATH_INVALID") from error
        result_path = run_dir / "workflow-result.json"
        if not result_path.exists():
            summary = {
                "schema_version": "dronedream.agent-execution-evidence.v1",
                "thread_id": thread_id,
                "execution_id": str(metadata["execution_id"]),
                "state": str(thread["state"]),
                "started_at": None,
                "completed_at": metadata.get("completed_at"),
                "return_code": metadata.get("return_code"),
                "result": None,
            }
            return summary
        try:
            raw_result = read_plugin_file(result_path, limit=16 * 1024 * 1024)
            result = SimulationWorkflowResult.model_validate(
                decode_json(raw_result, limit=16 * 1024 * 1024, node_limit=2_000_000)
            )
            # 历史消息可能没有保存计划身份；存在该字段时必须逐项匹配，不能忽略。
            for name in ("contract_id", "prepared_mission_sha256"):
                if name in metadata and metadata[name] != getattr(result, name):
                    raise ValueError("execution result identity changed")
        except (OSError, ValueError) as error:
            raise RuntimeBridgeError("EXECUTION_EVIDENCE_INVALID") from error
        runtime = result.runtime_evidence
        summary = {
            "schema_version": "dronedream.agent-execution-evidence.v1",
            "thread_id": thread_id,
            "execution_id": str(metadata["execution_id"]),
            "state": str(thread["state"]),
            "started_at": None,
            "completed_at": metadata.get("completed_at"),
            "return_code": metadata.get("return_code"),
            "result": {
                "status": result.status,
                "contract_id": result.contract_id,
                "prepared_mission_sha256": result.prepared_mission_sha256,
                "workflow_result_sha256": hashlib.sha256(raw_result).hexdigest(),
                "workflow_evidence_chain_head": result.workflow_evidence_chain_head,
                "completion_assessment": result.completion_assessment.model_dump(mode="json"),
                "gates": runtime.gates.model_dump(mode="json"),
                "measurements": runtime.measurements.model_dump(mode="json"),
                "artifact_hashes": {
                    "world_sha256": runtime.artifacts.world_sha256,
                    "semantic_sha256": runtime.artifacts.semantic_sha256,
                    "vehicle_sha256": runtime.artifacts.vehicle_sha256,
                    "route_sha256": runtime.artifacts.route_sha256,
                    "track_sha256": runtime.artifacts.track_sha256,
                    "clearance_sha256": runtime.artifacts.clearance_sha256,
                    "controller_params_sha256": runtime.artifacts.controller_params_sha256,
                    "executor_sha256": runtime.artifacts.executor_sha256,
                    "px4_ulog_count": len(runtime.artifacts.px4_ulogs),
                },
                "checkpoint_decision_count": len(result.checkpoint_decisions),
                "runtime_interruption_count": len(result.runtime_interruption_decisions),
                "plugin_hook_receipt_count": len(result.plugin_hook_receipts),
            },
        }
        return summary

    # 功能：
    #   按任务冻结的中断策略投递自然语言消息，记录请求而不直接绕过飞控安全层。
    # 输入：
    #   self：当前运行管理器。
    #   thread_id：接收消息的活动任务标识。
    #   text：需要运行控制器解释的自然语言请求。
    # 输出：
    #   accepted：消息标识及中断策略声明的即时动作。
    def submit_message(self, thread_id: str, text: str) -> dict[str, object]:
        with self._lock:
            active = self._active_runs.get(thread_id)
        if active is None:
            raise RuntimeBridgeError("TASK_NOT_EXECUTING")
        try:
            snapshot = PluginSnapshot.model_validate(
                self.store.latest_plugin_snapshot(thread_id)["snapshot"]
            )
            _entry, _manifest, capability = self.plugin_manager.capability_for_slot(
                snapshot, "safety.interruption-policy"
            )
        except (KeyError, ValueError, PluginManagerError) as error:
            raise RuntimeBridgeError("INTERRUPTION_POLICY_UNAVAILABLE") from error
        policy = str(capability.metadata.get("policy", "safe-hold-before-classification"))
        immediate_action = str(capability.metadata.get("immediate_action", "safe_hold"))
        with self._lock:
            if self._active_runs.get(thread_id) is not active or active.process.poll() is not None:
                raise RuntimeBridgeError("TASK_NOT_EXECUTING")
            try:
                read_active_session(active.run_dir / "runtime-control", thread_id)
            except RuntimeControlRecordError as error:
                raise RuntimeBridgeError("RUNTIME_MESSAGE_SESSION_INVALID") from error
            message = submit_runtime_message(
                control_dir=active.run_dir / "runtime-control", text=text
            )
            self.store.set_thread_state(thread_id, "holding")
            self.store.append_message(
                thread_id,
                role="user",
                kind="status",
                content=text,
                metadata={
                    "message_id": message.message_id,
                    "interrupt_policy": policy,
                    "interrupt_plugin_capability": capability.capability_id,
                },
            )
        accepted = {
            "accepted": True,
            "message_id": message.message_id,
            "immediate_action": immediate_action,
            "message_sha256": sha256_json(message),
            "execution_id": message.execution_id,
        }
        return accepted

    # 功能：
    #   核对任务、消息、悬停及决策链，为当前操作员发放不可覆盖的短期接管授权。
    # 输入：
    #   self：当前运行管理器。
    #   thread_id：正在执行的桌面任务标识。
    #   message_id：已处理的人工接管请求标识。
    #   operator_id：经应用认证的操作员标识。
    #   duration_seconds：授权有效期，范围 10 至 600 秒。
    # 输出：
    #   issued：一次性返回的令牌、截止时间及速度和偏航角速度限制。
    def issue_takeover_grant(
        self,
        thread_id: str,
        *,
        message_id: str,
        operator_id: str,
        duration_seconds: int,
    ) -> dict[str, object]:
        with self._lock:
            active = self._active_runs.get(thread_id)
        if active is None:
            raise RuntimeBridgeError("TASK_NOT_EXECUTING")
        if type(duration_seconds) is not int or not 10 <= duration_seconds <= 600:
            raise RuntimeBridgeError("TAKEOVER_DURATION_INVALID")
        control_dir = active.run_dir / "runtime-control"
        try:
            session, message, acknowledgement, decision, gates = read_takeover_evidence(
                control_dir, thread_id, message_id
            )
        except RuntimeControlRecordError as error:
            raise RuntimeBridgeError(f"TAKEOVER_GRANT_REJECTED:{error}") from error
        raw_token = secrets.token_urlsafe(48)
        token_sha256 = hashlib.sha256(raw_token.encode()).hexdigest()
        now = datetime.now(UTC)
        grant = RuntimeOperatorTakeoverGrant(
            message_id=message.message_id,
            execution_id=message.execution_id,
            operator_id=operator_id,
            message_sha256=sha256_json(message),
            hold_ack_sha256=sha256_json(acknowledgement),
            decision_sha256=sha256_json(decision),
            grant_token_sha256=token_sha256,
            maximum_horizontal_speed_mps=1.5,
            maximum_vertical_speed_mps=1.0,
            maximum_yaw_rate_dps=90.0,
            deterministic_gates=gates,
            issued_at=now,
            expires_at=now + timedelta(seconds=duration_seconds),
        )
        grant_sha256 = sha256_json(grant)
        with self._lock:
            # 文件读取之后任务可能已经结束，不能把旧任务授权挂到新进程上。
            if self._active_runs.get(thread_id) is not active or active.process.poll() is not None:
                raise RuntimeBridgeError("TAKEOVER_TASK_CHANGED")
            try:
                current_session = read_active_session(control_dir, thread_id, message.execution_id)
                if current_session != session:
                    raise RuntimeControlRecordError("RUNTIME_CONTROL_SESSION_CHANGED")
                self._atomic_json(
                    control_dir / "takeover-grants" / f"{message_id}.json",
                    grant,
                    replace_existing=False,
                )
            except (RuntimeControlRecordError, FileExistsError) as error:
                raise RuntimeBridgeError(
                    "TAKEOVER_GRANT_ALREADY_ISSUED_OR_SESSION_CHANGED"
                ) from error
            self._takeover_grants[thread_id] = ActiveTakeoverGrant(
                message_id=message_id,
                execution_id=message.execution_id,
                operator_id=operator_id,
                token_sha256=token_sha256,
                grant_sha256=grant_sha256,
                expires_at=grant.expires_at,
            )
        issued = {
            "accepted": True,
            "message_id": message_id,
            "grant_token": raw_token,
            "expires_at": grant.expires_at.isoformat(),
            "maximum_horizontal_speed_mps": grant.maximum_horizontal_speed_mps,
            "maximum_vertical_speed_mps": grant.maximum_vertical_speed_mps,
            "maximum_yaw_rate_dps": grant.maximum_yaw_rate_dps,
        }
        return issued

    # 功能：
    #   验证操作员、令牌和当前会话，按序发布限幅且短时有效的速度控制指令。
    # 输入：
    #   self：当前运行管理器。
    #   thread_id：控制指令所属任务标识。
    #   operator_id：经应用认证的操作员标识。
    #   message_id：授权对应的接管请求标识。
    #   grant_token：发放授权时返回的令牌。
    #   action：velocity 为速度控制，release 为交还控制权。
    #   north_mps：北向速度，单位米每秒。
    #   east_mps：东向速度，单位米每秒。
    #   down_mps：向下速度，单位米每秒，负值表示上升。
    #   yaw_rate_dps：偏航角速度，单位度每秒。
    #   duration_seconds：本条指令有效时长，大于零且不超过 0.5 秒。
    # 输出：
    #   accepted：成功入队的指令序号和动作类型。
    def submit_operator_control(
        self,
        thread_id: str,
        *,
        operator_id: str,
        message_id: str,
        grant_token: str,
        action: str,
        north_mps: float,
        east_mps: float,
        down_mps: float,
        yaw_rate_dps: float,
        duration_seconds: float,
    ) -> dict[str, object]:
        with self._lock:
            active = self._active_runs.get(thread_id)
            grant_state = self._takeover_grants.get(thread_id)
        if active is None or grant_state is None:
            raise RuntimeBridgeError("TAKEOVER_GRANT_NOT_ACTIVE")
        if not isinstance(operator_id, str) or not hmac.compare_digest(
            operator_id.encode("utf-8"), grant_state.operator_id.encode("utf-8")
        ):
            raise RuntimeBridgeError("TAKEOVER_GRANT_OPERATOR_MISMATCH")
        if not isinstance(grant_token, str) or len(grant_token) > 256:
            raise RuntimeBridgeError("TAKEOVER_GRANT_TOKEN_INVALID")
        supplied_hash = hashlib.sha256(grant_token.encode()).hexdigest()
        if not hmac.compare_digest(supplied_hash, grant_state.token_sha256):
            raise RuntimeBridgeError("TAKEOVER_GRANT_TOKEN_INVALID")
        if message_id != grant_state.message_id or datetime.now(UTC) >= grant_state.expires_at:
            raise RuntimeBridgeError("TAKEOVER_GRANT_EXPIRED_OR_MISMATCHED")
        values = (north_mps, east_mps, down_mps, yaw_rate_dps, duration_seconds)
        try:
            valid_numbers = all(
                type(value) in (int, float) and math.isfinite(value) for value in values
            )
        except OverflowError:
            valid_numbers = False
        if not valid_numbers:
            raise RuntimeBridgeError("TAKEOVER_COMMAND_NONFINITE_OR_INVALID")
        horizontal = math.hypot(north_mps, east_mps)
        if horizontal > 1.5 or abs(down_mps) > 1.0 or abs(yaw_rate_dps) > 90.0:
            raise RuntimeBridgeError("TAKEOVER_COMMAND_OUTSIDE_GRANT_ENVELOPE")
        with self._lock:
            if (
                self._active_runs.get(thread_id) is not active
                or self._takeover_grants.get(thread_id) is not grant_state
                or active.process.poll() is not None
                or datetime.now(UTC) >= grant_state.expires_at
            ):
                raise RuntimeBridgeError("TAKEOVER_GRANT_NOT_ACTIVE")
            try:
                read_active_session(
                    active.run_dir / "runtime-control", thread_id, grant_state.execution_id
                )
            except RuntimeControlRecordError as error:
                raise RuntimeBridgeError("TAKEOVER_SESSION_NOT_ACTIVE") from error
            sequence = grant_state.next_sequence
            command = RuntimeOperatorControlCommand(
                message_id=message_id,
                execution_id=grant_state.execution_id,
                grant_sha256=grant_state.grant_sha256,
                sequence=sequence,
                action=action,
                velocity_ned_mps=Vector3(x=north_mps, y=east_mps, z=down_mps),
                yaw_rate_dps=yaw_rate_dps,
                duration_seconds=duration_seconds,
                issued_at=datetime.now(UTC),
            )
            # 同一把锁覆盖取号、验证与发布，磁盘失败不能留下执行器永远等不到的序号。
            self._atomic_json(
                active.run_dir / "runtime-control" / "operator-commands" / f"{sequence:08d}.json",
                command,
                replace_existing=False,
            )
            grant_state.next_sequence += 1
        accepted = {"accepted": True, "sequence": sequence, "action": action}
        return accepted

    # 功能：
    #   按活动任务冻结的插件快照执行停用策略，需要悬停时核对本次请求的真实回执。
    # 输入：
    #   self：当前运行管理器。
    #   plugin_id：准备停用的插件标识。
    #   swap_policy：插件清单声明的替换策略。
    # 输出：
    #   held：已取得匹配悬停回执的任务标识列表。
    def prepare_plugin_disable(self, plugin_id: str, swap_policy: str) -> list[str]:
        with self._lock:
            active_runs = list(self._active_runs.items())
        affected: list[tuple[str, ActiveRun]] = []
        for thread_id, active in active_runs:
            try:
                record = self.store.latest_plugin_snapshot(thread_id)
                snapshot = PluginSnapshot.model_validate(record["snapshot"])
            except (KeyError, ValueError):
                raise RuntimeBridgeError("ACTIVE_TASK_PLUGIN_SNAPSHOT_INVALID") from None
            if plugin_id not in {item.plugin_id for item in snapshot.plugins}:
                continue
            affected.append((thread_id, active))
        held: list[str] = []
        if not affected or swap_policy in {"anytime", "next-mission"}:
            return held
        if swap_policy in {"restart", "certified-update"}:
            raise RuntimeBridgeError(
                f"PLUGIN_SWAP_REQUIRES_IDLE:{swap_policy}:"
                + ",".join(thread_id for thread_id, _active in affected)
            )
        if swap_policy != "safe-hold":
            raise RuntimeBridgeError(f"PLUGIN_SWAP_POLICY_UNSUPPORTED:{swap_policy}")
        for thread_id, active in affected:
            with self._lock:
                if self._active_runs.get(thread_id) is not active:
                    raise RuntimeBridgeError("PLUGIN_DRAIN_TASK_CHANGED")
            english = self._thread_locale(thread_id) == "en-US"
            result = self.submit_message(
                thread_id,
                (
                    f"Plugin {plugin_id} is being disabled. Enter a safe hold immediately "
                    "and wait for the capability change."
                    if english
                    else f"系统正在停用插件 {plugin_id}，请立即进入安全悬停并等待能力变更。"
                ),
            )
            acknowledgement_path = (
                active.run_dir / "runtime-control" / "acks" / f"{result['message_id']}.json"
            )
            deadline = time.monotonic() + 15.0
            while not acknowledgement_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not acknowledgement_path.is_file():
                raise RuntimeBridgeError("PLUGIN_DRAIN_HOLD_ACK_TIMEOUT")
            try:
                acknowledgement = read_control_record(
                    acknowledgement_path, RuntimeHoldAcknowledgement
                )
            except RuntimeControlRecordError as error:
                raise RuntimeBridgeError("PLUGIN_DRAIN_HOLD_ACK_REJECTED") from error
            if (
                acknowledgement.message_id != result["message_id"]
                or acknowledgement.message_sha256 != result["message_sha256"]
                or acknowledgement.execution_id != result["execution_id"]
                or not acknowledgement.side_effects_inhibited
                or not all(acknowledgement.deterministic_gates.values())
            ):
                raise RuntimeBridgeError("PLUGIN_DRAIN_HOLD_ACK_REJECTED")
            with self._lock:
                if self._active_runs.get(thread_id) is not active:
                    raise RuntimeBridgeError("PLUGIN_DRAIN_TASK_CHANGED")
            held.append(thread_id)
        return held

    # 功能：
    #   请求各运行合作退出并逐个回收宿主进程，单个失败不妨碍其他清理。
    #   宿主回收不等同于真机已降落；无法确认回收的进程保留登记并报告失败。
    # 输入：
    #   self：持有执行与资产验收进程的运行管理器。
    # 输出：
    #   None：不返回业务数据。
    def shutdown(self) -> None:
        with self._lock:
            active = list(self._active_runs.items())
            active_qualifications = list(self._active_asset_qualification_runs.items())
        for job_id, _run in active_qualifications:
            with suppress(KeyError, RuntimeError, OSError, ValueError):
                self.cancel_asset_pair_qualification(job_id)
        for thread_id, _run in active:
            try:
                self.submit_message(
                    thread_id,
                    (
                        "The application is shutting down. Enter a safe hold and land immediately."
                        if self._thread_locale(thread_id) == "en-US"
                        else "应用正在关闭，请立即进入安全悬停并降落"
                    ),
                )
            except (KeyError, RuntimeError, OSError, ValueError):
                # 中断消息链不可用时仍投递已有执行器支持的退出请求，不能直接跳过。
                with suppress(OSError, ValueError):
                    self._atomic_json(
                        _run.run_dir / "live_abort.request.json",
                        {"reason": "application_shutdown", "execution_id": _run.execution_id},
                    )
        deadline = time.monotonic() + 5.0
        failures = []
        for identifier, run in [*active, *active_qualifications]:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                try:
                    run.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    self._stop_owned_process(run.process)
            except Exception:
                failures.append(identifier)
                continue
            with self._lock:
                if self._active_runs.get(identifier) is run:
                    self._active_runs.pop(identifier, None)
                    self._takeover_grants.pop(identifier, None)
                if self._active_asset_qualification_runs.get(identifier) is run:
                    self._active_asset_qualification_runs.pop(identifier, None)
        if failures:
            raise RuntimeBridgeError("RUNTIME_SHUTDOWN_CLEANUP_FAILED:" + ",".join(failures))
