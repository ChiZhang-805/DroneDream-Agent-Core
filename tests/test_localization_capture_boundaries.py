"""Offline capture and readiness boundaries; no flight or external model calls."""

import json

import pytest
from test_localization_observations import _publish, _scan
from test_localization_truth_capture import contract, message, publish
from test_navigation_readiness import _graph_and_catalog, _vehicle

from dronedream_agent_core.contracts import NavigationReadinessReport
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.localization_observations import GeometryObservationCapture
from dronedream_agent_core.localization_truth_capture import LocalizationTruthCapture, _serialize
from dronedream_agent_core.native_preflight import (
    NativePerceptionReadiness,
    _read_native_health,
    assert_native_preflight_current,
)
from dronedream_agent_core.navigation_readiness import (
    _semantic,
    assess_navigation_readiness,
    enforce_environment_readiness,
)


# 功能：
#   非法发布器应在独占领取文件前拒绝，不留下无法发布回执的半初始化采集器。
# 输入：
#   tmp_path：测试私有目录。
#   kind：几何记录或独立真值记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["geometry", "truth"])
def test_publisher_is_validated_before_capture_claim(tmp_path, kind):
    with pytest.raises(ValueError):
        if kind == "geometry":
            GeometryObservationCapture(tmp_path, map_sha256="b" * 64, summary_publisher=None)
        else:
            LocalizationTruthCapture(tmp_path, frames=contract(), summary_publisher=None)
    assert not list(tmp_path.iterdir())


# 功能：
#   时钟补充字段必须是有限有界对象，不能将错误键、巨大内容或列表作为成功观测排队。
# 输入：
#   tmp_path：测试私有目录。
#   clock：非法来源时钟描述。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("clock", [[1], {1: "coerced"}, {"age": float("nan")},
                                 {"text": "x" * 8193}], ids=["list", "key", "nan", "size"])
def test_bad_clock_never_enters_geometry_fifo(tmp_path, clock):
    capture = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                         summary_publisher=_publish)
    scan, kwargs = _scan()
    try:
        assert not capture.record(scan, **{**kwargs, "source_clock": clock})
    finally:
        result = capture.close()
    assert not result["complete"] and result["accepted_count"] == 0


# 功能：
#   即使记录错误发生在关闭前，每一份已发布的完成回执也必须保留采集失败状态。
# 输入：
#   tmp_path：测试私有目录。
#   kind：被破坏的采集器类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["geometry", "truth"])
def test_writer_never_publishes_transient_false_success(tmp_path, kind):
    published = []

    # 功能：
    #   捕获每次真实发布，而不只检查关闭最后返回的对象。
    # 输入：
    #   path：采集器回执路径。
    #   value：本次即将发布的回执。
    # 输出：
    #   None：不返回业务数据。
    def publisher(path, value):
        published.append(json.loads(json.dumps(value)))
        publish(path, value)

    if kind == "geometry":
        capture = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                             summary_publisher=publisher)
        scan, kwargs = _scan()
        assert not capture.record(scan, **{**kwargs, "calibration_sha256": "bad"})
    else:
        capture = LocalizationTruthCapture(tmp_path, frames=contract(), summary_publisher=publisher)
        assert not capture.record(message(duplicate=True), received_monotonic=1.,
                                  received_unix_ms=1000)
    assert not capture.close()["complete"]
    assert published and all(row["complete"] is False for row in published)


# 功能：
#   自洽摘要不能替代安装几何和字段类型检查，错误契约不得先占用输出文件。
# 输入：
#   tmp_path：测试私有目录。
#   fault：合法摘要下的错误内容类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["missing", "rotation", "name", "keys"])
def test_truth_contract_checks_content_before_claim(tmp_path, fault):
    frames = contract()
    frames.pop("record_sha256")
    if fault == "missing":
        frames.pop("canonical_at_rest")
    elif fault == "rotation":
        frames["canonical_at_rest"]["orientation_wxyz"] = [0, 0, 0, 0]
    elif fault == "name":
        frames["canonical_link_name"] = "nested::link"
    else:
        frames["extra"] = {1: "stringified-key"}
    frames["record_sha256"] = sha256_json(frames)
    with pytest.raises(ValueError):
        LocalizationTruthCapture(tmp_path, frames=frames, summary_publisher=publish)
    assert not list(tmp_path.iterdir())


# 功能：
#   序列化器不能把数字键悄悄转为字符串，也不能重复包裹已有摘要形成不一致记录。
# 输入：
#   record：错误键或已有摘要的候选记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("record", [{1: "value"}, {"record_sha256": "a" * 64}])
def test_truth_serializer_rejects_ambiguous_records(record):
    with pytest.raises(ValueError):
        _serialize(record)


# 功能：
#   仿真见证的本地墙钟必须是有界整数，过大时刻不能进入成功记录。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_truth_rejects_out_of_range_wall_clock(tmp_path):
    capture = LocalizationTruthCapture(tmp_path, frames=contract(), summary_publisher=publish)
    try:
        assert not capture.record(message(), received_monotonic=1., received_unix_ms=2**64)
    finally:
        result = capture.close()
    assert not result["complete"] and result["accepted_count"] == 0


# 功能：
#   就绪检查的 JSON 输入拒绝重复键，避免健康或地图声明被末项覆盖。
# 输入：
#   tmp_path：测试私有目录。
#   kind：健康证据或地图语义。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["health", "semantic"])
def test_readiness_readers_reject_duplicate_keys(tmp_path, kind):
    path = tmp_path / "input.json"
    path.write_text('{"ready":false,"ready":true}', encoding="utf-8")
    if kind == "health":
        with pytest.raises(ValueError):
            _read_native_health(path)
    else:
        assert _semantic(path) == {}


# 功能：
#   前置健康读取只接收普通文件，不沿静态链接借用另一份数据充当当前运行证据。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_preflight_reader_refuses_symlink(tmp_path):
    target, link = tmp_path / "target.json", tmp_path / "health.json"
    target.write_text('{}', encoding="utf-8")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("host does not permit unprivileged symbolic links")
    with pytest.raises(ValueError):
        _read_native_health(link)


# 功能：
#   修改报告总开关不能跳过其定位、碰撞和感知依赖项。
# 输入：
#   mode：要求的动态环境模式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["known-map-with-dynamic-obstacles", "unknown-indoor-environment"])
def test_summary_flag_cannot_override_missing_dependencies(mode):
    report = NavigationReadinessReport(static_map_planning_ready=False,
        static_collision_geometry_ready=False, occupancy_esdf_ready=False,
        indoor_localization_ready=False, onboard_obstacle_perception_ready=False,
        dynamic_obstacle_tracking_ready=False, qualified_static_simulation_ready=False,
        known_dynamic_map_autonomy_ready=True, arbitrary_indoor_autonomy_ready=True)
    with pytest.raises(ValueError, match="NOT_READY"):
        enforce_environment_readiness(mode, report)


# 功能：
#   被绕过模型校验的传感器列表或拓扑标志不能隐式转成有效配置。
# 输入：
#   tmp_path：测试私有目录。
#   fault：被篡改的配置字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["sensors", "topology"])
def test_readiness_revalidates_mutable_config_fields(tmp_path, fault):
    graph, catalog = _graph_and_catalog()
    vehicle = _vehicle(["imu"])
    if fault == "sensors":
        vehicle = vehicle.model_copy(update={"sensors": [1]})
    else:
        catalog = catalog.model_copy(update={"topology_available": "false"})
    with pytest.raises(ValueError, match="ASSET_INPUT_INVALID"):
        assess_navigation_readiness(graph, catalog, tmp_path / "missing", vehicle)


# 功能：
#   见证发布器失败时，关闭返回不能仍声称完成，即使记录流为空也一样。
# 输入：
#   tmp_path：测试私有目录。
#   kind：见证采集器类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["geometry", "truth"])
def test_summary_publication_failure_is_retained(tmp_path, kind):
    # 功能：
    #   模拟回执目标写入失败，错误详情不得成为成功依据。
    # 输入：
    #   path：待发布的测试路径。
    #   value：本次回执对象。
    # 输出：
    #   None：不返回业务数据。
    def fail(path, value):
        raise OSError("test-only publication failure")

    if kind == "geometry":
        capture = GeometryObservationCapture(tmp_path, map_sha256="b" * 64,
                                             summary_publisher=fail)
    else:
        capture = LocalizationTruthCapture(tmp_path, frames=contract(), summary_publisher=fail)
    result = capture.close()
    assert not result["complete"]
    assert result["issue_code"] == "RUNTIME_EVIDENCE_SUMMARY_FAILED:OSError"


# 功能：
#   极大整数不能作为有效 UNIX 毫秒时刻，通过减法一致性伪装成新鲜回执。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_preflight_bounds_integer_clock_domain():
    now = 2**64
    gate = NativePerceptionReadiness()
    assert not gate.observe({}, now_unix_ms=now)
    with pytest.raises(ValueError):
        assert_native_preflight_current({"ready": True, "source_observed_at_unix_ms": now,
                                         "valid_until_unix_ms": now + 250}, now_unix_ms=now)
