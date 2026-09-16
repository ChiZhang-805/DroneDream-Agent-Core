"""Boundary regressions for builtin verdicts, geometry proposals and evidence exports."""

from __future__ import annotations

import csv
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import (
    CoveragePlanRequest,
    Px4GazeboGates,
    Px4GazeboMeasurements,
    Vector3,
)
from dronedream_agent_plugins import runtime_quality_plugins as quality
from dronedream_agent_plugins.native_runtime_plugins import _descriptor, _watchdog
from dronedream_agent_plugins.runtime_coverage_plugins import (
    _edge_distance,
    _lawnmower,
    _region,
    _spiral,
)
from dronedream_agent_plugins.runtime_quality_plugins import (
    _artifact_binding,
    _atomic_text,
    _battery_reserve,
    _export_metrics_csv,
    _export_summary,
    _export_track_geojson,
    _runtime_gate_integrity,
    _telemetry_integrity,
    _tracking_stability,
)


# 功能：
#   验证原生存活检测的毫秒配置不强制转换布尔值、字符串、浮点数或越界值。
# 输入：
#   value：非法心跳间隔。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, "25", 25.1, float("nan"), -1, 251])
def test_watchdog_rejects_coercion_and_out_of_range_heartbeat(value: object) -> None:
    with pytest.raises(ValueError, match="NATIVE_WATCHDOG_CONFIGURATION_INVALID"):
        _watchdog(configuration={"heartbeat_ms": value})


# 功能：
#   验证预期间隔不能长于失联截止时间，也不能通过额外配置键忽略失联。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_watchdog_does_not_allow_a_heartbeat_later_than_deadline() -> None:
    with pytest.raises(ValueError, match="HEARTBEAT_EXCEEDS_DEADLINE"):
        _watchdog(configuration={"heartbeat_ms": 100, "deadline_ms": 50})
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _watchdog(configuration={"ignore_miss": True})


# 功能：
#   验证原生能力描述返回独立列表，且必须另取核心授权，不能冒称已执行通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_native_descriptor_does_not_lend_mutable_capabilities() -> None:
    capabilities = ["depth"]
    result = _descriptor("perception", capabilities)
    result["capabilities"].clear()
    assert capabilities == ["depth"]
    assert result["core_authorization_required"] is True
    assert "accepted" not in result


# 功能：
#   构造可被故意损坏的检查点输入，直接验证检测器而不依赖模型构造时的拦截。
# 输入：
#   changes：替换默认合法遥测或门控的字段。
# 输出：
#   request：仅含检测器所需字段的测试检查点。
def _checkpoint(**changes: object) -> SimpleNamespace:
    values = dict(
        observed_position_ned_m=Vector3(x=0, y=0, z=-1),
        observed_velocity_ned_mps=Vector3(x=0, y=0, z=0),
        commanded_position_ned_m=Vector3(x=0, y=0, z=-1),
        position_error_m=0.0,
        speed_mps=0.0,
        battery_percent=50.0,
        deterministic_gates={"fresh": True},
    )
    request = SimpleNamespace(**(values | changes))
    return request


# 功能：
#   验证空门控、非布尔真值及明确失败都不能通过遥测完整性或制品绑定评估。
# 输入：
#   gates：需要拒绝的门控集合。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("gates", [{}, {"fresh": "false"}, {"fresh": 1}, {"fresh": False}])
def test_verdicts_require_nonempty_literal_true_gates(gates: dict) -> None:
    assert not _telemetry_integrity(request=_checkpoint(deterministic_gates=gates))["accepted"]
    assert not _artifact_binding(binding_gates=gates)["accepted"]


# 功能：
#   验证物理范围外、非有限及错误类型电量同时被完整性和储备门控拒绝。
# 输入：
#   value：非法电量百分比。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [-1, 101, True, "50", float("inf"), float("nan"), 10**400])
def test_invalid_battery_never_passes_a_reserve_gate(value: object) -> None:
    request = _checkpoint(battery_percent=value)
    assert not _telemetry_integrity(request=request)["accepted"]
    assert not _battery_reserve(request=request)["accepted"]


# 功能：
#   验证负速度、非有限位置误差和非法阈值不能得到跟踪或电量通过结果。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_tracking_rejects_mutated_physical_values_and_invalid_policy() -> None:
    assert not _tracking_stability(request=_checkpoint(speed_mps=-1))["accepted"]
    assert not _tracking_stability(request=_checkpoint(position_error_m=float("nan")))["accepted"]
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _tracking_stability(request=_checkpoint(), configuration={"maximum_position_error_m": "20"})
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _battery_reserve(request=_checkpoint(), configuration={"minimum_battery_percent": 0})


# 功能：
#   验证运行期异常门控不将错误类型或未知配置键当成缺省策略。
# 输入：
#   detector：位置速度或电量异常检测器。
#   configuration：需要拒绝的配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("detector", [_tracking_stability, _battery_reserve])
@pytest.mark.parametrize("configuration", [False, [], 0, "", {"misspelled_threshold": 1}])
def test_runtime_detectors_reject_falsey_or_unknown_configuration(detector, configuration):
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        detector(request=_checkpoint(), configuration=configuration)


# 功能：
#   验证发布前暂存名称被另一文件替换时，不把替换内容当作导出结果，也不删除它。
# 输入：
#   tmp_path：隔离的导出目录。
#   monkeypatch：在临时流关闭后注入名称替换。
# 输出：
#   None：不返回业务数据。
def test_export_neither_publishes_nor_deletes_replaced_staging(tmp_path, monkeypatch):
    target = tmp_path / "summary.json"
    target.write_text("old", encoding="utf-8")
    create = quality.tempfile.NamedTemporaryFile
    replaced = []

    # 功能：
    #   保留原导出文件后另建同名文件，避免依赖删除后文件标识可能立即复用的行为。
    # 输入：
    #   kwargs：生产代码指定的临时流选项。
    # 输出：
    #   stream：原始独占临时流。
    @contextmanager
    def replace_after_close(**kwargs):
        with create(**kwargs) as stream:
            yield stream
        path = Path(stream.name)
        path.rename(tmp_path / "retained-export")
        path.write_text("another writer", encoding="utf-8")
        replaced.append(path)

    monkeypatch.setattr(quality.tempfile, "NamedTemporaryFile", replace_after_close)
    with pytest.raises(ValueError, match="STAGING_CHANGED"):
        _atomic_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert replaced[0].read_text(encoding="utf-8") == "another writer"


# 功能：
#   验证发布成功后另一写入者重新占用暂存名称时，退出清理不能删除该文件。
# 输入：
#   tmp_path：隔离的导出目录。
#   monkeypatch：在原子替换完成后注入新文件。
# 输出：
#   None：不返回业务数据。
def test_export_cleanup_preserves_new_owner_after_publication(tmp_path, monkeypatch):
    target = tmp_path / "summary.json"
    replace = Path.replace
    recreated = []

    # 功能：
    #   执行真实原子替换，然后在已空出的源名称创建另一份文件。
    # 输入：
    #   source：当前导出的暂存路径。
    #   destination：汇总目标路径。
    # 输出：
    #   result：原 Path.replace 返回的目标路径。
    def occupy_after_replace(source, destination):
        result = replace(source, destination)
        source.write_text("new owner", encoding="utf-8")
        recreated.append(source)
        return result

    monkeypatch.setattr(Path, "replace", occupy_after_replace)
    _atomic_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert recreated[0].read_text(encoding="utf-8") == "new owner"


# 功能：
#   验证必需门控不可缺失；未报告的可选默认值不参加验收，显式失败或错误类型仍须否决。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_runtime_verdict_requires_required_gates_but_not_unset_optional_defaults() -> None:
    required = {
        name: True for name, field in Px4GazeboGates.model_fields.items() if field.is_required()
    }
    runtime = SimpleNamespace(status="verified", gates=Px4GazeboGates(**required))
    assert _runtime_gate_integrity(runtime=runtime)["accepted"]
    runtime.gates.model_fields_set.clear()
    assert not _runtime_gate_integrity(runtime=runtime)["accepted"]
    runtime.gates = Px4GazeboGates(**required, live_depth_perception_healthy=False)
    assert not _runtime_gate_integrity(runtime=runtime)["accepted"]
    object.__setattr__(runtime.gates, "live_depth_perception_healthy", "false")
    assert not _runtime_gate_integrity(runtime=runtime)["accepted"]


# 功能：
#   验证并发导出只发布其中一份完整摘要，不争用固定暂存名，也不清理无关文件。
# 输入：
#   tmp_path：隔离的并发导出目录。
# 输出：
#   None：不返回业务数据。
def test_export_uses_unique_tempfiles_and_publishes_only_complete_text(tmp_path: Path) -> None:
    target = tmp_path / "summary.json"
    unrelated = target.with_suffix(".json.tmp")
    unrelated.write_text("preserved", encoding="utf-8")
    payloads = [str(index) * 10_000 for index in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda value: _atomic_text(target, value), payloads))
    assert target.read_text(encoding="utf-8") in payloads
    assert unrelated.read_text(encoding="utf-8") == "preserved"
    assert not list(tmp_path.glob(".summary.json.*.tmp"))


# 功能：
#   验证发布失败时保留原目标摘要，并回收本次独占创建的暂存文件。
# 输入：
#   tmp_path：隔离的导出目录。
#   monkeypatch：仅对本次测试注入替换失败。
# 输出：
#   None：不返回业务数据。
def test_failed_export_preserves_destination_and_cleans_only_its_own_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "summary.json"
    target.write_text("old", encoding="utf-8")

    # 功能：
    #   在暂存内容完成写入与落盘后，使发布入口抛出 I/O 错误。
    # 输入：
    #   _：原替换调用的位置参数。
    # 输出：
    #   None：不返回业务数据。
    def fail_replace(*_: object) -> None:
        raise OSError("injected")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        _atomic_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["summary.json"]


# 功能：
#   验证文本公式被转义而数值负号保留，后来损坏为 NaN 的测量不能导出成正常空值。
# 输入：
#   tmp_path：CSV 导出的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_csv_escapes_formulas_but_preserves_numeric_values(tmp_path: Path) -> None:
    runtime = SimpleNamespace(
        measurements=Px4GazeboMeasurements(
            pose_sample_count=1,
            ros_observation_rows=1,
            abort_reason=" =1+1",
            executor_return_code=-1,
        )
    )
    result = _export_metrics_csv(run_dir=tmp_path, runtime=runtime, binding_gates={"bound": True})
    rows = dict(list(csv.reader(io.StringIO(Path(result["path"]).read_text(encoding="utf-8"))))[1:])
    assert rows["measurement.abort_reason"] == "'=1+1"
    assert rows["measurement.executor_return_code"] == "-1"
    object.__setattr__(runtime.measurements, "minimum_goal_distance_m", float("nan"))
    with pytest.raises(ValueError, match="JSON_NUMBER_INVALID"):
        _export_metrics_csv(run_dir=tmp_path, runtime=runtime, binding_gates={"bound": True})


# 功能：
#   验证本地米制航迹不作为经纬度几何发布，成功写入摘要也不改变实际运行失败状态。
# 输入：
#   tmp_path：航迹及摘要导出目录。
# 输出：
#   None：不返回业务数据。
def test_local_track_never_claims_metres_are_longitude_latitude(tmp_path: Path) -> None:
    prepared = SimpleNamespace(
        contract=SimpleNamespace(contract_id="test"),
        px4_track=SimpleNamespace(
            source_world_points=[
                SimpleNamespace(east_m=5000, north_m=2000, up_m=10),
                SimpleNamespace(east_m=5001, north_m=2001, up_m=10),
            ]
        ),
    )
    result = _export_track_geojson(run_dir=tmp_path, prepared=prepared)
    feature = json.loads(Path(result["path"]).read_text(encoding="utf-8"))["features"][0]
    assert feature["geometry"] is None
    assert feature["properties"]["local_coordinates_enu_m"][0] == [5000, 2000, 10]
    assert result["georeference_available"] is False
    # 文件成功写入不是飞行成功，摘要必须保留原始运行失败状态。
    result = _export_summary(
        run_dir=tmp_path,
        prepared=prepared,
        runtime=SimpleNamespace(world="test", vehicle="test", status="failed"),
        binding_gates={"bound": False},
        plugin_evaluations=[],
    )
    assert json.loads(Path(result["path"]).read_text())["runtime_status"] == "failed"


# 功能：
#   构造小型米制覆盖区域，显式设置高度、间距和安全边距，供路径边界测试使用。
# 输入：
#   polygon：可选多边形顶点；未提供时使用矩形区域。
#   changes：本例覆盖的请求字段。
# 输出：
#   request：经过契约构造的覆盖规划请求。
def _coverage(
    polygon: list[tuple[float, float]] | None = None, **changes: object
) -> CoveragePlanRequest:
    values = dict(
        center_enu_m=Vector3(x=0, y=0, z=0),
        width_m=8,
        height_m=6,
        lane_spacing_m=1,
        boundary_margin_m=0.25,
        altitude_m=2,
        polygon_enu_m=[Vector3(x=x, y=y, z=0) for x, y in polygon or []],
    )
    request = CoveragePlanRequest(**(values | changes))
    return request


# 功能：
#   验证两类覆盖提案的段内部也满足倾斜边界和边距，不只检查返回顶点。
# 输入：
#   planner：往复式或内缩环式提案函数。
#   polygon：矩形或两种顶点顺序的三角形。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("planner", [_lawnmower, _spiral])
@pytest.mark.parametrize("polygon", [None, [(0, 0), (8, 0), (0, 6)], [(0, 6), (8, 0), (0, 0)]])
def test_coverage_segments_respect_oblique_edges_and_margin(planner, polygon) -> None:
    request = _coverage(polygon)
    result = planner(request=request)
    _, vertices = _region(request)
    points = [(p["x"], p["y"]) for p in result["points_enu_m"]]
    for a, b in zip(points, points[1:], strict=False):
        # 沿连接段采样，避免顶点在区域内却穿越边界的路径被误判为合格。
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            p = (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))
            for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True):
                assert _edge_distance(p, start, end) >= request.boundary_margin_m - 1e-8
    assert result["estimated_area_m2"] < (48 if polygon is None else 24)
    assert all(result["deterministic_gates"].values())


# 功能：
#   验证未做凸分解的凹区域被明确拒绝，而不是穿越缺口连接航点。
# 输入：
#   planner：当前被测覆盖提案函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("planner", [_lawnmower, _spiral])
def test_coverage_rejects_concave_regions_instead_of_connecting_across_a_notch(planner) -> None:
    request = _coverage([(0, 0), (6, 0), (6, 6), (4, 6), (4, 2), (2, 2), (2, 6), (0, 6)])
    with pytest.raises(ValueError, match="NONCONVEX_REGION_REQUIRES_DECOMPOSITION"):
        planner(request=request)


# 功能：
#   验证退化区域、过大边距、超量采样和构造后损坏的间距均在生成前被拒绝。
# 输入：
#   planner：当前被测覆盖提案函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("planner", [_lawnmower, _spiral])
def test_coverage_rejects_degenerate_excessive_and_mutated_inputs(planner) -> None:
    with pytest.raises(ValueError, match="DEGENERATE"):
        planner(request=_coverage([(0, 0), (1, 1), (2, 2)]))
    with pytest.raises(ValueError, match="MARGIN_COLLAPSES"):
        planner(request=_coverage(boundary_margin_m=50))
    with pytest.raises(ValueError, match="POINT_LIMIT"):
        planner(request=_coverage(width_m=5000, height_m=5000, lane_spacing_m=0.2))
    request = _coverage()
    object.__setattr__(request, "lane_spacing_m", 0)
    with pytest.raises(ValueError):
        planner(request=request)
    object.__setattr__(request, "lane_spacing_m", math.nan)
    with pytest.raises(ValueError, match="UNBOUNDED"):
        planner(request=request)


# 功能：
#   验证远离坐标原点的凸区域仍能产生有限且数量受限的提案，面积计算不因平移失真。
# 输入：
#   planner：当前被测覆盖提案函数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("planner", [_lawnmower, _spiral])
def test_large_translated_convex_region_remains_finite(planner) -> None:
    polygon = [
        (1e6 + 10 * math.cos(i * math.pi / 8), 1e6 + 10 * math.sin(i * math.pi / 8))
        for i in range(16)
    ]
    result = planner(request=_coverage(polygon))
    assert 2 <= len(result["points_enu_m"]) <= 10_000
    assert 200 < result["estimated_area_m2"] < 315
