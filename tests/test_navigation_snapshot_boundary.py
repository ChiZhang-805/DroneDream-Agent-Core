"""Owned navigation snapshots; fixtures are not live flight qualification."""

from dataclasses import replace
from time import perf_counter

import pytest
from test_learning_observation_recorder import request_fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.navigation_snapshot import (
    bind_navigation_control_output_contract,
    compile_navigation_snapshot,
)


# 功能：
#   请求持有的特征、传感器和视觉容器改变后，已编译快照及其摘要不能漂移。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_compiled_snapshot_does_not_alias_input_messages():
    request, _ = request_fixture()
    request = replace(request, visual_evidence=[{"confidence": 0.5}],
                      multimodal_sensor_snapshot={"available": [True]})
    result = compile_navigation_snapshot(request)
    expected = result["snapshot_sha256"]
    request.visual_evidence[0]["confidence"] = 0.9
    request.multimodal_sensor_snapshot["available"].append(False)
    request.realtime_feature_snapshot["source"] = "changed"
    assert result["snapshot_sha256"] == expected
    assert sha256_json({key: value for key, value in result.items()
                        if key != "snapshot_sha256"}) == expected


# 功能：
#   连续轴控制的输入不能混入坐标候选模式，开关与来源时钟必须使用精确类型。
# 输入：
#   field、value：非法请求字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("field", "value"), [
    ("include_candidate_paths", 0), ("include_candidate_paths", True),
    ("control_reference_observed_at_unix_ms", True),
    ("control_reference_observed_at_unix_ms", -1),
    ("maximum_snapshot_planning_seconds", float("nan")),
    ("maximum_snapshot_planning_seconds", True),
    ("vehicle_radius_m", float("inf")), ("required_clearance_m", True),
])
def test_snapshot_rejects_bad_controls_before_planning(field, value):
    request, _ = request_fixture()
    with pytest.raises(ValueError):
        compile_navigation_snapshot(replace(request, **{field: value}))


# 功能：
#   非字符串输出模式应明确拒绝，不在集合查找时触发不可哈希类型异常。
# 输入：
#   value：损坏的模式字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [{}, [], True])
def test_mode_rejects_invalid_type(value):
    with pytest.raises(ValueError):
        bind_navigation_control_output_contract(
            {}, {"task": {"local_navigation_output_mode": value}})


# 功能：
#   变异后的健康字段不能经 JSON 序列化悄悄转换成 null 再写入已散列快照。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_health_is_revalidated_before_serialization():
    request, _ = request_fixture()
    request = replace(request, health=request.health.model_copy(
        update={"stream_age_seconds": float("nan")}))
    with pytest.raises(ValueError):
        compile_navigation_snapshot(request)


# 功能：
#   编译不更新来源时钟，不自动授予运行控制权。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_snapshot_keeps_original_clock_and_axis_contract():
    request, _ = request_fixture()
    result = compile_navigation_snapshot(request)
    assert result["control_reference_observed_at_unix_ms"] == 1000
    assert result["model_authority"]["may_output"]["axes"] == ["forward", "right", "up", "yaw"]
    assert "metric coordinates" in result["model_authority"]["may_not_author"]


# 功能：
#   协调器使用不可变元组冻结动态目标时，编译结果应与列表输入完全相同。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_frozen_dynamic_obstacle_tuple_is_supported():
    request, _ = request_fixture()
    expected = compile_navigation_snapshot(request)
    request = replace(request, frame=request.frame.model_copy(
        update={"dynamic_obstacles": tuple(request.frame.dynamic_obstacles)}))
    assert compile_navigation_snapshot(request) == expected


# 功能：
#   记录小型合成地图上的编译延迟，分开标明此测量不含感知、推理和飞行控制。
# 输入：
#   record_property：把样本数及分位数写入本次 JUnit 证据。
# 输出：
#   None：不返回业务数据。
def test_snapshot_compile_timing_is_recorded_without_refreshing_source(record_property):
    request, _ = request_fixture()
    for _ in range(5):
        compile_navigation_snapshot(request)
    latencies = []
    for _ in range(100):
        started = perf_counter()
        snapshot = compile_navigation_snapshot(request)
        latencies.append((perf_counter() - started) * 1000)
        assert snapshot["control_reference_observed_at_unix_ms"] == 1000
    latencies.sort()
    record_property("scope", "small synthetic map snapshot compile only; not flight acceptance")
    record_property("samples", len(latencies))
    record_property("median_ms", latencies[50])
    record_property("p95_ms", latencies[94])
    record_property("p99_ms", latencies[98])
    assert all(value >= 0 for value in latencies)
