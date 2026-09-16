"""Checkpoint verdicts and bounded evidence exports; no direct actuation authority."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from dronedream_agent_core.contracts import (
    PreparedMission,
    Px4GazeboRunEvidence,
    RuntimeCheckpointRequest,
)
from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_files import check_plain_plugin_path
from dronedream_agent_core.runtime_plugins import all_required_gates_passed
from dronedream_plugin_sdk.protocol import MAX_MESSAGE_BYTES, copy_json

from ._helpers import bounded_number, hook_plugin, policy_number

_EXPORT_PUBLICATION_LOCK = threading.Lock()


# 功能：
#   核对位置、速度、电量的有限性与物理范围，并要求执行器明确通过非空的确定性门控。
# 输入：
#   request：当前检查点的遥测及执行器证据。
#   _：完整性检测器不使用的扩展参数。
# 输出：
#   verdict：检查结论、分项门控和问题码。
def _telemetry_integrity(*, request: RuntimeCheckpointRequest, **_: Any) -> dict[str, object]:
    vectors = (
        request.observed_position_ned_m,
        request.observed_velocity_ned_mps,
        request.commanded_position_ned_m,
    )
    values = [component for vector in vectors for component in (vector.x, vector.y, vector.z)]
    values.extend([request.position_error_m, request.speed_mps, request.battery_percent])
    finite = all(bounded_number(value, -math.inf, math.inf) for value in values)
    physical = (
        bounded_number(request.position_error_m, 0, math.inf)
        and bounded_number(request.speed_mps, 0, math.inf)
        and bounded_number(request.battery_percent, 0, 100)
    )
    deterministic = all_required_gates_passed(request.deterministic_gates)
    accepted = finite and physical and deterministic
    verdict = {
        "detector": "telemetry-integrity",
        "accepted": accepted,
        "gates": {
            "all_values_finite": finite,
            "physical_ranges_valid": physical,
            "executor_deterministic_gates_passed": deterministic,
        },
        "issue_codes": [] if accepted else ["RUNTIME_TELEMETRY_INTEGRITY_REJECTED"],
    }
    return verdict


# 功能：
#   按合法策略核对检查点位置误差和速度，不修改控制器参数或覆盖其他安全否决。
# 输入：
#   request：包含位置误差与速度的检查点请求。
#   configuration：可选的位置误差上限和速度上限配置；仅 None 表示缺省。
#   _：稳定性检测器不使用的扩展参数。
# 输出：
#   verdict：分项结论、实际输入读数及拒绝原因。
def _tracking_stability(
    *,
    request: RuntimeCheckpointRequest,
    configuration: dict[str, object] | None = None,
    **_: Any,
) -> dict[str, object]:
    configured = {} if configuration is None else configuration
    if not isinstance(configured, dict) or set(configured) - {
        "maximum_position_error_m",
        "maximum_checkpoint_speed_mps",
    }:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    maximum_error = policy_number(configured, "maximum_position_error_m", 1.5, 0.05, 20)
    maximum_speed = policy_number(configured, "maximum_checkpoint_speed_mps", 3.0, 0, 20)
    gates = {
        "position_error_within_policy": bounded_number(request.position_error_m, 0, maximum_error),
        "checkpoint_speed_within_policy": bounded_number(request.speed_mps, 0, maximum_speed),
    }
    accepted = all(gates.values())
    verdict = {
        "detector": "tracking-stability",
        "accepted": accepted,
        "gates": gates,
        "observed_position_error_m": request.position_error_m,
        "observed_speed_mps": request.speed_mps,
        "issue_codes": [] if accepted else ["RUNTIME_TRACKING_STABILITY_REJECTED"],
    }
    return verdict


# 功能：
#   核对电量是否达到已配置的最低百分比，并拒绝物理上不可能的电量值。
# 输入：
#   request：包含电量读数的检查点请求。
#   configuration：可选最低电量配置；错误类型或未知键均不能回退到默认值。
#   _：电量检测器不使用的扩展参数。
# 输出：
#   verdict：电量结论、读数、策略阈值及问题码。
def _battery_reserve(
    *,
    request: RuntimeCheckpointRequest,
    configuration: dict[str, object] | None = None,
    **_: Any,
) -> dict[str, object]:
    configured = {} if configuration is None else configuration
    if not isinstance(configured, dict) or set(configured) - {"minimum_battery_percent"}:
        raise ValueError("PLUGIN_POLICY_CONFIGURATION_INVALID")
    threshold = policy_number(configured, "minimum_battery_percent", 20, 1, 80)
    accepted = bounded_number(request.battery_percent, threshold, 100)
    verdict = {
        "detector": "battery-reserve",
        "accepted": accepted,
        "battery_percent": request.battery_percent,
        "minimum_battery_percent": threshold,
        "issue_codes": [] if accepted else ["RUNTIME_BATTERY_RESERVE_LOW"],
    }
    return verdict


# 功能：
#   验证运行已确认、必需门控齐全且已报告的所有门控明确为真，不接受空证据。
# 输入：
#   runtime：上游绑定的 PX4/Gazebo 运行证据。
#   _：完整性检查不使用的扩展参数。
# 输出：
#   verdict：运行状态、实际报告的门控及完整性结论。
def _runtime_gate_integrity(*, runtime: Px4GazeboRunEvidence, **_: Any) -> dict[str, object]:
    # 未启用可选能力的保守 False 默认值不是本次运行结果；但显式报告的 False 必须否决。
    # 用 fields_set 保留这一区别，并另查必需字段，不能只依赖过滤后字典的 all()。
    gates = runtime.gates.model_dump(exclude_unset=True)
    required = {
        name for name, field in type(runtime.gates).model_fields.items() if field.is_required()
    }
    accepted = (
        runtime.status == "verified"
        and required.issubset(gates)
        and all_required_gates_passed(gates)
    )
    verdict = {
        "evaluation": "runtime-gate-integrity",
        "accepted": accepted,
        "runtime_status": runtime.status,
        "gates": gates,
        "issue_codes": [] if accepted else ["RUNTIME_GATE_INTEGRITY_REJECTED"],
    }
    return verdict


# 功能：
#   汇总核心提供的制品绑定检查，空映射、非布尔真值或任何否决均不能通过。
# 输入：
#   binding_gates：核心计算的制品身份门控结果。
#   _：绑定汇总器不使用的扩展参数。
# 输出：
#   verdict：绑定检查结论及独立的门控映射。
def _artifact_binding(*, binding_gates: dict[str, bool], **_: Any) -> dict[str, object]:
    accepted = all_required_gates_passed(binding_gates)
    verdict = {
        "evaluation": "artifact-binding",
        "accepted": accepted,
        "gates": dict(binding_gates),
        "issue_codes": [] if accepted else ["RUNTIME_ARTIFACT_BINDING_REJECTED"],
    }
    return verdict


# 功能：
#   1. 有界写入、落盘并原子替换命名摘要，不覆盖原始飞行证据文件。
#   2. 发布前复核独占暂存文件身份，退出时仅清理仍属于本次的文件。
#   3. 拒绝静态链接和重解析路径；这些检查不替代恶意进程竞争下的系统隔离。
# 输入：
#   path：允许重新导出的汇总文件路径。
#   rendered：待发布的完整 UTF-8 文本。
# 输出：
#   None：不返回业务数据。
def _atomic_text(path: Path, rendered: str) -> None:
    if len(rendered) > MAX_MESSAGE_BYTES or len(rendered.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("RUNTIME_EVIDENCE_EXPORT_TOO_LARGE")
    check_plain_plugin_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    identity: os.stat_result | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            identity = os.fstat(stream.fileno())
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows 并发替换可能冲突，仅串行化发布阶段，不把渲染和落盘纳入全局锁。
        with _EXPORT_PUBLICATION_LOCK:
            check_plain_plugin_path(path)
            check_plain_plugin_path(temporary)
            if not os.path.samestat(identity, temporary.stat()):
                raise ValueError("RUNTIME_EVIDENCE_EXPORT_STAGING_CHANGED")
            temporary.replace(path)
    finally:
        if temporary is not None and identity is not None:
            with suppress(OSError, ValueError):
                check_plain_plugin_path(temporary)
                if os.path.samestat(identity, temporary.stat()):
                    temporary.unlink()


# 功能：
#   导出任务身份、运行状态与插件检查摘要；文件写入成功不能覆盖运行失败结论。
# 输入：
#   run_dir：本次运行的证据目录。
#   prepared：冻结的待执行任务。
#   runtime：实际运行证据。
#   binding_gates：核心制品绑定结果。
#   plugin_evaluations：插件评估结果列表。
#   _：摘要导出器不使用的扩展参数。
# 输出：
#   receipt：导出器名称、写入状态和摘要路径。
def _export_summary(
    *,
    run_dir: Path,
    prepared: PreparedMission,
    runtime: Px4GazeboRunEvidence,
    binding_gates: dict[str, bool],
    plugin_evaluations: list[dict[str, object]],
    **_: Any,
) -> dict[str, object]:
    destination = run_dir / "plugin-evidence" / "mission-summary.json"
    payload = {
        "schema_version": "dronedream.plugin-mission-summary.v1",
        "contract_id": prepared.contract.contract_id,
        "world": runtime.world,
        "vehicle": runtime.vehicle,
        "runtime_status": runtime.status,
        "binding_gates": binding_gates,
        "plugin_evaluations": plugin_evaluations,
    }
    _atomic_text(
        destination,
        json.dumps(copy_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    receipt = {"accepted": True, "exporter": "mission-summary-json", "path": str(destination)}
    return receipt


# 功能：
#   导出有限标量测量，避免电子表格把文本值解释为公式；非有限值在渲染前拒绝。
# 输入：
#   run_dir：本次运行证据目录。
#   runtime：包含原始测量字段的运行证据。
#   binding_gates：核心绑定检查结果。
#   _：CSV 导出器不使用的扩展参数。
# 输出：
#   receipt：导出状态、文件路径及数据行数。
def _export_metrics_csv(
    *,
    run_dir: Path,
    runtime: Px4GazeboRunEvidence,
    binding_gates: dict[str, bool],
    **_: Any,
) -> dict[str, object]:
    destination = run_dir / "plugin-evidence" / "runtime-metrics.csv"
    rows: list[tuple[str, object]] = [
        (f"gate.{name}", value) for name, value in sorted(binding_gates.items())
    ]
    rows.extend(
        (f"measurement.{name}", value)
        for name, value in sorted(runtime.measurements.model_dump(mode="python").items())
        if isinstance(value, (str, int, float, bool)) or value is None
    )
    # 在模型 JSON 序列化可能把 NaN 变成 null 之前，先验证 Python 原始数值。
    validated = copy_json([list(row) for row in rows])
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["metric", "value"])
    writer.writerows([_csv_cell(value) for value in row] for row in validated)
    _atomic_text(destination, stream.getvalue())
    receipt = {
        "accepted": True,
        "exporter": "runtime-metrics-csv",
        "path": str(destination),
        "row_count": len(rows),
    }
    return receipt


# 功能：
#   对文本公式前缀和控制字符加保护前缀，真实数值（包括负数）保持数值语义。
# 输入：
#   value：已通过有限 JSON 检查的单元格值。
# 输出：
#   cell：必要时加保护前缀的单元格值。
def _csv_cell(value: object) -> object:
    cell = value
    if isinstance(value, str) and (
        value.lstrip().startswith(("=", "+", "-", "@"))
        or any(character in value for character in ("\t", "\r", "\n"))
    ):
        cell = "'" + value
    return cell


# 功能：
#   保存本地 ENU 航迹属性；未绑定 WGS84 地理参考时 geometry 为 null，不能把米当经纬度。
# 输入：
#   run_dir：本次运行证据目录。
#   prepared：包含冻结合同及源世界坐标轨迹的任务。
#   _：航迹导出器不使用的扩展参数。
# 输出：
#   receipt：导出路径、点数及明确为假的地理参考可用标记。
def _export_track_geojson(
    *, run_dir: Path, prepared: PreparedMission, **_: Any
) -> dict[str, object]:
    destination = run_dir / "plugin-evidence" / "planned-track.geojson"
    coordinates = [
        [point.east_m, point.north_m, point.up_m]
        for point in prepared.px4_track.source_world_points
    ]
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "contract_id": prepared.contract.contract_id,
                    "coordinate_frame": "Gazebo ENU",
                    "point_count": len(coordinates),
                    "local_coordinates_enu_m": coordinates,
                    "georeference_available": False,
                },
                "geometry": None,
            }
        ],
    }
    _atomic_text(
        destination,
        json.dumps(copy_json(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    receipt = {
        "accepted": True,
        "exporter": "planned-track-geojson",
        "path": str(destination),
        "point_count": len(coordinates),
        "georeference_available": False,
    }
    return receipt


# 功能：
#   注册运行异常检测、证据门控及导出器，严格区分拒绝执行的门控与不授予权限的导出。
# 输入：
#   无。
# 输出：
#   definitions：带独立失败策略及替换时机的插件列表。
def plugin_definitions() -> list[PluginDefinition]:
    definitions: list[PluginDefinition] = []
    anomaly_plugins = [
        (
            "runtime.anomaly-telemetry",
            "遥测完整性检测",
            "拒绝非有限数值或执行器确定性门失败的检查点。",
            _telemetry_integrity,
            {},
        ),
        (
            "runtime.anomaly-tracking",
            "跟踪稳定性检测",
            "按可配置位置误差与检查点速度限制决定是否继续。",
            _tracking_stability,
            {
                "type": "object",
                "properties": {
                    "maximum_position_error_m": {
                        "type": "number",
                        "minimum": 0.05,
                        "maximum": 20,
                    },
                    "maximum_checkpoint_speed_mps": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 20,
                    },
                },
                "additionalProperties": False,
            },
        ),
        (
            "runtime.anomaly-battery",
            "电量余量检测",
            "在每个运行检查点实施可配置最低电量门。",
            _battery_reserve,
            {
                "type": "object",
                "properties": {
                    "minimum_battery_percent": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 80,
                    }
                },
                "additionalProperties": False,
            },
        ),
    ]
    for index, (plugin_id, name, description, handler, schema) in enumerate(
        anomaly_plugins, start=1
    ):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.detect",
                capability_kind="anomaly-detector",
                capability_name=name,
                capability_description=description,
                category_id="runtime",
                category_label="运行与接管",
                slot_id="runtime.anomaly-detectors",
                slot_label="运行异常检测器",
                activation_mode="multiple",
                category_order=70,
                slot_order=30,
                plugin_order=index * 10,
                hooks={"evaluate_checkpoint": handler},
                default_enabled=True,
                failure_mode="fail-closed",
                swap_policy="safe-hold",
                configuration_schema=schema,
                permissions=["mission.read", "telemetry.read", "configuration.read"],
            )
        )
    runtime_gates = [
        (
            "evaluation.runtime-gates",
            "运行门完整性",
            "验收 PX4/Gazebo 运行状态与全部原生运行门。",
            _runtime_gate_integrity,
        ),
        (
            "evaluation.artifact-binding",
            "工件绑定完整性",
            "验收运行文件、任务合同、地图和无人机哈希绑定。",
            _artifact_binding,
        ),
    ]
    for index, (plugin_id, name, description, handler) in enumerate(runtime_gates, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.evaluate",
                capability_kind="evaluator",
                capability_name=name,
                capability_description=description,
                category_id="evaluation",
                category_label="证据与评测",
                slot_id="evaluation.runtime-gates",
                slot_label="运行验收门",
                activation_mode="multiple",
                category_order=90,
                slot_order=20,
                plugin_order=index * 10,
                hooks={"evaluate_runtime": handler},
                default_enabled=True,
                failure_mode="fail-closed",
                swap_policy="next-mission",
                permissions=["mission.read", "telemetry.read"],
            )
        )
    exporters = [
        (
            "evidence.summary-json",
            "任务摘要 JSON",
            "导出合同、运行门和插件评测的机器可读摘要。",
            _export_summary,
        ),
        (
            "evidence.metrics-csv",
            "运行指标 CSV",
            "导出运行测量与绑定门，便于数据分析和批量评测。",
            _export_metrics_csv,
        ),
        (
            "evidence.track-geojson",
            "航迹 GeoJSON",
            "导出本地 ENU 计划航迹属性；未绑定地理参考时不生成经纬度几何。",
            _export_track_geojson,
        ),
    ]
    for index, (plugin_id, name, description, handler) in enumerate(exporters, start=1):
        definitions.append(
            hook_plugin(
                module_name=__name__,
                plugin_id=plugin_id,
                name=name,
                description=description,
                capability_id=f"{plugin_id}.export",
                capability_kind="evidence-exporter",
                capability_name=name,
                capability_description=description,
                category_id="evaluation",
                category_label="证据与评测",
                slot_id="evidence.exporters",
                slot_label="证据导出器",
                activation_mode="multiple",
                category_order=90,
                slot_order=30,
                plugin_order=index * 10,
                hooks={"export_evidence": handler},
                default_enabled=plugin_id != "evidence.track-geojson",
                failure_mode="isolate",
                swap_policy="next-mission",
                permissions=["mission.read", "telemetry.read", "evidence.write"],
            )
        )
    return definitions
