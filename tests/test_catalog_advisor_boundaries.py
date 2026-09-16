"""Malformed declarations must not become successful qualifications or invented measurements."""

from copy import deepcopy

import pytest
from jsonschema import validate

from dronedream_agent_core.contracts import Px4Track
from dronedream_agent_plugins.asset_qualification_checks import (
    check_physics_completeness,
    check_runtime_interfaces,
    check_source_provenance,
)
from dronedream_agent_plugins.general_constraint_coverage import _coverage
from dronedream_agent_plugins.general_mission_risk_profile import _profile
from dronedream_agent_plugins.notification_plugins import _planning_metrics
from dronedream_agent_plugins.simulation_campaign_plugins import _obstacle_fault
from dronedream_agent_plugins.simulation_runtime_plugins import _descriptor, _physics
from dronedream_agent_plugins.simulation_runtime_plugins import (
    plugin_definitions as simulator_plugins,
)


# 功能：
#   构造最小合法声明供附加检查使用，不将测试字典称为已经认证的生产资产包。
# 输入：
#   无。
# 输出：
#   inputs：地图、无人机、门控与环境声明的独立测试数据。
def _asset_inputs():
    shared = {
        "source": {"adapter_id": "source.test", "source_format": "sdf", "source_sha256": "a" * 64},
        "geometry": {"collision_count": 1},
        "simulation_targets": [{"simulator": "gazebo-harmonic"}],
    }
    map_ir, vehicle_ir = deepcopy(shared), deepcopy(shared)
    map_ir["semantics"] = {"navigation_graph_path": "map/graph.json"}
    vehicle_ir["interfaces"] = {"px4_profiles": ["test-profile"]}
    vehicle_ir["physics"] = {
        "collision_complete": True,
        "mass_complete": True,
        "inertia_complete": True,
    }
    inputs = {
        "map_asset_ir": map_ir,
        "vehicle_asset_ir": vehicle_ir,
        "plan": {"required_runtime_gates": ["telemetry_valid"]},
        "runtime_evidence": {"gates": {"telemetry_valid": True}},
        "environment_versions": {"gazebo": "test-fixture"},
    }
    return inputs


# 功能：
#   验证合法声明可通过附加检查，返回证据不共享可变引用，也不冒称资产已提升。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_additive_checks_accept_well_formed_declarations_and_detach_details():
    inputs = _asset_inputs()
    for check in (check_source_provenance, check_physics_completeness, check_runtime_interfaces):
        assert check(**inputs)["accepted"] is True
    result = check_physics_completeness(**inputs)
    result["details"]["vehicle_physics"]["mass_complete"] = False
    assert inputs["vehicle_asset_ir"]["physics"]["mass_complete"] is True
    assert "promoted" not in result


# 功能：
#   验证长度、字符或类型错误的来源摘要不能作为有效来源声明。
# 输入：
#   digest：需要拒绝的伪摘要。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("digest", ["x", "", " " * 64, True, "a" * 63, "g" * 64])
def test_provenance_rejects_non_hash_strings_and_other_types(digest):
    inputs = _asset_inputs()
    inputs["map_asset_ir"]["source"]["source_sha256"] = digest
    assert check_source_provenance(**inputs)["accepted"] is False


# 功能：
#   验证碰撞几何必须有范围内的实际整数计数，不能依赖真值或类型转换。
# 输入：
#   count：非法碰撞几何计数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [True, "1", 1.2, None, 0, -1, 1_000_001])
def test_physics_count_requires_exact_positive_ir_integer(count):
    inputs = _asset_inputs()
    inputs["vehicle_asset_ir"]["geometry"]["collision_count"] = count
    assert check_physics_completeness(**inputs)["accepted"] is False


# 功能：
#   验证必要门控集合非空且条目类型正确，避免空遍历导致错误通过。
# 输入：
#   required：非法必要门控声明。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("required", [[], "telemetry_valid", [""], [True], None])
def test_empty_or_malformed_required_gate_set_cannot_pass(required):
    inputs = _asset_inputs()
    inputs["plan"]["required_runtime_gates"] = required
    assert check_runtime_interfaces(**inputs)["accepted"] is False


# 功能：
#   分别破坏运行门控、飞控配置、仿真目标、环境和导航声明，检查均不能放行。
# 输入：
#   field：本例替换的声明部分。
#   value：替换后的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("gate", "true"),
        ("profiles", "test-profile"),
        ("targets", [{"simulator": True}]),
        ("versions", {"gazebo": ""}),
        ("navigation", True),
    ],
)
def test_runtime_interface_requires_typed_bindings(field, value):
    inputs = _asset_inputs()
    if field == "gate":
        inputs["runtime_evidence"]["gates"]["telemetry_valid"] = value
    elif field == "profiles":
        inputs["vehicle_asset_ir"]["interfaces"]["px4_profiles"] = value
    elif field == "targets":
        inputs["map_asset_ir"]["simulation_targets"] = value
        inputs["vehicle_asset_ir"]["simulation_targets"] = value
    elif field == "versions":
        inputs["environment_versions"] = value
    else:
        inputs["map_asset_ir"]["semantics"]["navigation_graph_path"] = value
    assert check_runtime_interfaces(**inputs)["accepted"] is False


# 功能：
#   验证普通室外导航不误判为载荷或室内任务，风险提示也不附带执行批准。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_navigation_is_not_payload_and_outdoor_is_not_indoor():
    result = _profile(
        {"goal": "go outdoors for breakfast", "payload_action": "navigate", "constraints": []}
    )
    assert result["risk_factors"] == []
    assert result["risk_level"] == "normal"
    # 定性标签既不是安全概率，也不是允许执行的证明。
    assert "accepted" not in result


# 功能：
#   验证无额外约束时仍分析目标文字，未知领域动作单独提示而不一概视为载荷。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_advisor_uses_goal_even_when_constraints_are_empty():
    result = _profile({"goal": "fly fast indoors", "payload_action": "pickup", "constraints": []})
    assert result["risk_factors"] == [
        "payload interaction",
        "constrained indoor geometry",
        "time pressure",
    ]
    unknown = _profile({"goal": "inspect", "payload_action": "camera.inspect", "constraints": []})
    assert unknown["risk_factors"] == ["unclassified domain action"]


# 功能：
#   验证两类顾问都拒绝错误类型、超量或超长约束，不将任意对象转换为提示文字。
# 输入：
#   constraints：待拒绝的约束输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("constraints", [None, "safe", [True], [{}], ["x"] * 33, ["x" * 4001]])
def test_advisors_do_not_coerce_arbitrary_constraint_objects(constraints):
    for hook in (_coverage, _profile):
        with pytest.raises(ValueError, match="CONSTRAINTS_INVALID"):
            hook({"goal": "go", "payload_action": "navigate", "constraints": constraints})


# 功能：
#   验证提及全部维度只能得到文字覆盖率，不能产生已验证或允许执行字段。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_coverage_is_only_a_dimension_mention_report():
    result = _coverage({"constraints": ["safe return speed confirm"]})
    assert result["coverage_ratio"] == 1
    assert "accepted" not in result
    assert "safe_to_execute" not in result


# 功能：
#   验证物理描述器拒绝非法步长、非有限值、错误类型、未知求解器和配置键。
# 输入：
#   configuration：需要拒绝的 DART 配置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "configuration",
    [
        {"step_seconds": True},
        {"step_seconds": 0},
        {"step_seconds": -1},
        {"real_time_factor_target": float("inf")},
        {"contact_tolerance_m": "0.001"},
        {"solver": "unknown"},
        {"ignore_gate": True},
    ],
)
def test_physics_descriptor_rejects_invalid_values_without_launching_engine(configuration):
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _physics(configuration=configuration)


# 功能：
#   验证物理默认描述符合其 Schema，且仿真器界面选项与运行探测要求保持一致。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_physics_configuration_schema_and_runtime_descriptor_agree():
    definition = next(
        item for item in simulator_plugins() if item.manifest.plugin_id == "simulation.physics-dart"
    )
    result = _physics(configuration={})
    validate(result, definition.manifest.configuration_schema)
    assert result["step_seconds"] == 0.001
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _descriptor("gazebo", "ros-gz", configuration={"headless": "false"})
    assert _descriptor("gazebo", "ros-gz", configuration={})["runtime_probe_required"] is True


# 功能：
#   验证直接生成仿真器描述时也拒绝拼错或多余配置，不能静默改用无界面模式。
# 输入：
#   configuration：与仿真器配置 Schema 不符的输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "configuration", [{"headles": False}, {"headless": False, "extra": 1}, [], False]
)
def test_simulator_descriptor_rejects_unknown_configuration(configuration):
    with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
        _descriptor("gazebo", "ros-gz", configuration=configuration)


# 功能：
#   构造不同东向采样间距的轨迹，隔离障碍位置插值测试与真实运行器。
# 输入：
#   eastings：世界坐标中依次经过的东向米制坐标。
# 输出：
#   track：保留 ENU 源点的最小测试轨迹。
def _track(eastings):
    track = Px4Track.model_validate(
        {
            "coordinate_contract": {
                "model_root_world_enu_m": [0, 0, 0],
                "collision_center_offset_model_m": [0, 0, 0.2],
            },
            "points": [
                {"x": 0, "y": east, "z": 1, "phase": "transit", "speed_limit_mps": 0.5}
                for east in eastings
            ],
            "source_world_points": [{"east_m": east, "north_m": 0, "up_m": 1} for east in eastings],
            "waypoint_hold_seconds": 0.4,
        }
    )
    return track


# 功能：
#   验证不均匀采样及重复点不改变按路径距离定位的中点，规格不伪造故障执行记录。
# 输入：
#   eastings：总长相同但采样分布不同的东向坐标列表。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("eastings", [[0, 1, 10], [0, 0, 1, 10], [0, 9, 10]])
def test_obstacle_fault_uses_route_distance_not_waypoint_index(eastings):
    fault = _obstacle_fault(px4_track=_track(eastings))
    assert fault["parameters"]["east_m"] == pytest.approx(5)
    assert fault["parameters"]["north_m"] == 0
    assert fault["trigger"]["route_fraction"] == 0.5
    assert "executed" not in fault


# 功能：
#   验证零长轨迹没有可定义的半程触发位置，不能生成误导性障碍规格。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_obstacle_fault_rejects_zero_length_track():
    with pytest.raises(ValueError, match="ROUTE_INVALID"):
        _obstacle_fault(px4_track=_track([0, 0]))


# 功能：
#   验证通知不把数字字符串、布尔值或负计数显示成有效的规划指标。
# 输入：
#   field：本例损坏的指标名。
#   value：替换指标的非法值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "field,value",
    [
        ("minimum_clearance_m", True),
        ("minimum_clearance_m", "1"),
        ("model_calls", "1"),
        ("model_calls", True),
        ("planning_attempts", -1),
    ],
)
def test_notification_does_not_format_coerced_values_as_measured_metrics(field, value):
    summary = {
        "minimum_clearance_m": 1.2,
        "model_calls": 1,
        "planning_attempts": 1,
        "plugin_catalog_sha256": "a" * 64,
        "locale": "en-US",
    }
    summary[field] = value
    with pytest.raises(ValueError, match="METRICS_INVALID"):
        _planning_metrics(summary=summary)
