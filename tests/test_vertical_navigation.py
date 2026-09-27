"""竖直控制集成与负向场景；数值测试不是实机资格或 Gazebo 飞行验收。"""

import hashlib
import json
import math
from itertools import product

import pytest

from dronedream_agent_core.collision import primitive_bounds
from dronedream_agent_core.contracts import (
    LocalPlannerRequest,
    RuntimeLocalSafetyObservation,
    Vector3,
    VehicleAsset,
)
from dronedream_agent_core.dynamic_safety import predictive_safety_decision
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety
from dronedream_agent_core.vertical_navigation import (
    CeilingTransition,
    FlightDynamicsEnvelope,
    VerticalMotionGuard,
    height_interval,
    load_flight_dynamics_envelope,
    payload_motion_issues,
)


# 功能：
#   构造单位明确的合成载具，用于边界测试，不声明实际飞机已通过测量。
# 输入：
#   无：固定测试参数。
# 输出：
#   vehicle：测试机体。
def vehicle():
    result = VehicleAsset(asset_id="test-uav", name="fixture", dry_mass_kg=2.,
        max_takeoff_mass_kg=2.8, body_radius_m=.25, body_height_m=.4, max_speed_mps=2.,
        max_acceleration_mps2=1., qualified_range_m=100., reserve_battery_percent=30.,
        max_pickup_payload_kg=1., sensors=["depth-camera"])
    return result


# 功能：
#   为测试提供完整的质量、电压、姿态及制动约束，不写入产品资格文件。
# 输入：
#   updates：本用例需要覆盖的包络字段。
# 输出：
#   envelope：合成测试动力学包络。
def envelope(**updates):
    fields = dict(vehicle_sha256=sha256_json(vehicle()), evidence_sha256="a" * 64,
        domain="simulation", minimum_total_mass_kg=2., maximum_total_mass_kg=2.8,
        maximum_climb_mps=.4, maximum_descent_mps=.3, maximum_horizontal_mps=1.,
        maximum_command_acceleration_mps2=1., braking_deceleration_lower_bound_mps2=.5,
        response_delay_upper_bound_seconds=.1, minimum_battery_voltage_v=14., maximum_tilt_deg=25.,
        maximum_actuator_fraction=.95,
        collision_radius_m=.35, collision_height_m=.8)
    result = FlightDynamicsEnvelope(**{**fields, **updates})
    return result


# 功能：
#   生成带新鲜电压和姿态的负载背景，供正常及污染场景共同使用。
# 输入：
#   mass：实测负载公斤数。
# 输出：
#   payload：测试装载完成后的状态。
def payload(mass=.5):
    result = {"state": "loaded-stable", "observed_payload_mass_kg": mass,
              "dynamics": {"ready": True, "voltage_v": 16., "roll_deg": 0., "pitch_deg": 0.,
                           "actuator_max_abs": .5, "actuator_normalization_ready": True,
                           "source_sample_age_seconds": {"battery": .1}}}
    return result


# 功能：
#   创建标准轴对齐障碍盒供门檐、楼板及车辆场景使用。
# 输入：
#   center、size：中心及三个方向的尺寸。
# 输出：
#   primitive：米制碰撞图元。
def box(center, size):
    result = dict(zip(("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"),
                      (*center, *size), strict=True))
    return result


# 功能：
#   创建当前位置两米高的预测输入，并可替换速度或目标。
# 输入：
#   updates：用例需替换的字段。
# 输出：
#   request：校验后的局部预测请求。
def request(**updates):
    fields = dict(current_position_m=Vector3(x=0., y=0., z=2.),
        current_velocity_mps=Vector3(x=0., y=0., z=0.),
        target_position_m=Vector3(x=3., y=0., z=3.), vehicle_radius_m=.25,
        vehicle_height_m=.4, max_speed_mps=1., max_acceleration_mps2=1.,
        required_clearance_m=.15, prediction_horizon_seconds=3., prediction_step_seconds=.2)
    result = LocalPlannerRequest(**{**fields, **updates})
    return result


# 功能：
#   构建实际保护器，仅覆盖函数可按测试需要替换以隔离各项约束。
# 输入：
#   updates：几何、覆盖来源或负载等测试参数。
# 输出：
#   guard：单周期竖直保护器。
def guard(**updates):
    fields = dict(vehicle=vehicle(), payload={"state": "detached"}, primitives=[],
                  position=(0., 0., 2.), source_seconds=1., transition=CeilingTransition(),
                  clearance=.15, coverage_check=lambda _path, _margin: True)
    result = VerticalMotionGuard(**{**fields, **updates})
    return result


# 功能：
#   验证碰撞中心高度、离地高度及楼层世界坐标不会混淆，顶部区间包含机身尺寸。
# 输入：
#   offset：不同楼层的世界高度。
# 输出：
#   None：断言所有楼层得到相同的局部净空。
@pytest.mark.parametrize("offset", [-10., 0., 17., 60.])
def test_height_interval_uses_local_floor_and_whole_body(offset):
    bounds = [primitive_bounds(box((0., 0., offset - .1), (10., 10., .2))),
              primitive_bounds(box((0., 0., offset + 3.1), (10., 10., .2)))]
    band = height_interval((0., 0., offset + 1.), bounds, .25, .2, .15)
    assert band.floor_z_m == pytest.approx(offset)
    assert band.minimum_center_z_m == pytest.approx(offset + .35)
    assert band.maximum_center_z_m == pytest.approx(offset + 2.65)


# 功能：
#   只有机身和余量整体离开屋檐投影，才开始放宽顶部区间。
# 输入：
#   无：固定一米长的出口屋檐。
# 输出：
#   None：断言中心越过边缘不等于整机已通过。
def test_exit_checks_body_radius_not_center_only():
    bounds = [primitive_bounds(box((0., 0., 3.), (2., 2., .2)))]
    assert height_interval((1.1, 0., 2.), bounds, .25, .2, .15).ceiling_z_m is not None
    assert height_interval((1.41, 0., 2.), bounds, .25, .2, .15).ceiling_z_m is None


# 功能：
#   拒绝复用帧推进过渡时钟；短暂离开又回来时重新开始稳定计时。
# 输入：
#   无：人工排列的有效观测序列。
# 输出：
#   None：验证收紧立即生效且放宽具有迟滞。
def test_transition_is_asymmetric_and_does_not_renew_repeated_frames():
    state = CeilingTransition(.4)
    assert state.update(2.5, 1.) == 2.5
    for _ in range(20):
        assert state.update(None, 1.) == 2.5
    assert state.update(None, 1.1) == 2.5
    assert state.update(None, 1.3) == 2.5
    assert state.update(2.4, 1.4) == 2.4
    assert state.update(None, 1.5) == 2.4
    assert state.update(None, 1.7) == 2.4
    assert state.update(None, 1.91) is None
    assert state.update(2.2, 2.) == 2.2
    with pytest.raises(ValueError, match="CLOCK_REGRESSED"):
        state.update(None, 1.9)


# 功能：
#   同时刻的更窄门檐打断退出计时，避免旧的放宽计时替新门檐解除限制。
# 输入：
#   无：包含重复时间戳和收紧边界的序列。
# 输出：
#   None：断言必须重新积累稳定的新观测。
def test_repeated_frame_tightening_resets_exit_timer():
    state = CeilingTransition(.4)
    state.update(2.5, 1.)
    state.update(None, 1.1)
    state.update(None, 1.3)
    assert state.update(2.3, 1.3) == 2.3
    assert state.update(None, 1.51) == 2.3
    assert state.update(None, 1.72) == 2.3
    assert state.update(None, 1.93) is None


# 功能：
#   检验质量、适用环境和机型绑定，额定挂载量不覆盖起飞总重限制。
# 输入：
#   mass、code：各负载用例及应触发原因。
# 输出：
#   None：断言越界输入被直接阻断。
@pytest.mark.parametrize("mass,code", [(None, "PAYLOAD_MASS_UNKNOWN"),
    (True, "PAYLOAD_MASS_UNKNOWN"), (float("nan"), "PAYLOAD_MASS_UNKNOWN"),
    (-.1, "PAYLOAD_MASS_UNKNOWN"), (.9, "PAYLOAD_EXCEEDS_VEHICLE_LIMIT")])
def test_payload_bounds(mass, code):
    assert code in payload_motion_issues(vehicle(), payload(mass), envelope(), "simulation")


# 功能：
#   验证低压、旧电压、姿态超界和模型无实测包络时不能取得负载运动授权。
# 输入：
#   无：分别污染有效负载背景。
# 输出：
#   None：确认各条件独立生效。
def test_payload_qualification_and_live_operating_conditions():
    assert payload_motion_issues(vehicle(), payload(), envelope(), "simulation") == []
    assert "LOADED_MOTION_DYNAMICS_UNQUALIFIED" in payload_motion_issues(
        vehicle(), payload(), None, "simulation")
    assert "FLIGHT_DYNAMICS_BINDING_MISMATCH" in payload_motion_issues(
        vehicle(), payload(), envelope(), "hardware")
    conditions = [("voltage_v", 12., "BATTERY"), ("roll_deg", 30., "ATTITUDE"),
                  ("ready", False, "UNREADY"), ("actuator_max_abs", 1., "ACTUATOR")]
    for field, value, expected in conditions:
        data = payload()
        data["dynamics"][field] = value
        assert any(expected in s for s in payload_motion_issues(
            vehicle(), data, envelope(), "simulation"))
    data = payload()
    data["dynamics"]["source_sample_age_seconds"]["battery"] = 4.
    assert "FLIGHT_DYNAMICS_BATTERY_OUTSIDE_QUALIFICATION" in payload_motion_issues(
        vehicle(), data, envelope(), "simulation")


# 功能：
#   在有空间依据时允许前进同时爬升，头顶证据缺失时拒绝原动作并保留制动。
# 输入：
#   无：相同模型请求分别面对已知和未知空间。
# 输出：
#   None：断言保护器接入实际候选决策，而非只输出建议。
def test_forward_and_climb_is_actuated_only_with_swept_coverage():
    proposed = request(requested_velocity_mps=Vector3(x=.3, y=0., z=.2))
    allowed = predictive_safety_decision(proposed, [], motion_check=guard().check)
    assert allowed.selected_velocity_mps.x > 0 and allowed.selected_velocity_mps.z > 0
    assert allowed.control_source == "local-model-body-control"
    blocked = predictive_safety_decision(proposed, [], motion_check=guard(
        coverage_check=lambda _path, _margin: False).check)
    assert blocked.action == "hold"
    assert "SWEPT_SPACE_UNOBSERVED_OR_BLOCKED" in blocked.issue_codes


# 功能：
#   在屋檐下阻断持续上升；候选替代必须仍经完整碰撞及高度检查。
# 输入：
#   无：三米高屋檐下的上升请求。
# 输出：
#   None：验证任何获准路径均不越过实际顶部中心界限。
def test_canopy_and_upward_inertia_do_not_use_endpoint_only():
    geometry = [box((0., 0., 3.), (4., 4., .2))]
    protection = guard(primitives=geometry)
    result = predictive_safety_decision(request(requested_velocity_mps=Vector3(x=.2, y=0., z=.8)),
                                       geometry, motion_check=protection.check)
    if result.action != "hold":
        assert max(p.z for p in result.predicted_path_m) <= protection.ceiling + 1e-9
    assert protection.check(request(), (0., 0., .5), [(0., 0., 2.), (0., 0., 2.7)])


# 功能：
#   比较相同当前速度下的强弱制动包络；短预测终点不能代替停止距离。
# 输入：
#   无：远端两米处不可通行的合成覆盖边界。
# 输出：
#   None：验证较弱的合格制动能力确实需要更大停止空间。
def test_loaded_braking_checks_current_momentum_and_response_delay():
    # 功能：
    #   用明确的两米边界隔离停止距离计算。
    # 输入：
    #   path、margin：候选路径及本用例不使用的额外膨胀。
    # 输出：
    #   covered：路径是否保持在测试空间内。
    def coverage(path, margin):
        covered = max(p[0] for p in path) < 2.
        return covered
    state = request(current_velocity_mps=Vector3(x=1., y=0., z=0.))
    path = [(0., 0., 2.), (.2, 0., 2.)]
    strong = guard(payload=payload(), envelope=envelope(), coverage_check=coverage)
    weak = guard(payload=payload(), envelope=envelope(braking_deceleration_lower_bound_mps2=.1),
                 coverage_check=coverage)
    assert strong.check(state, (.1, 0., 0.), path) == []
    assert "BRAKING_SWEPT_SPACE_UNOBSERVED_OR_BLOCKED" in weak.check(state, (.1, 0., 0.), path)


# 功能：
#   验证竖直背景与实际运行观测成对绑定，缺少保护器或换位置后不能复用摘要。
# 输入：
#   无：一条已绑定背景的实时观测。
# 输出：
#   None：断言真实运行入口执行限制及摘要检查。
def test_runtime_binding_and_direct_payload_veto():
    protection = guard(payload=payload())
    observation = RuntimeLocalSafetyObservation(sequence=1, observed_at_unix_ms=1000,
        source="onboard", stream_healthy=True, stream_age_seconds=.01,
        localization_covariance_m2=0.,
        current_position_m=Vector3(x=0., y=0., z=2.),
        current_velocity_mps=Vector3(x=.1, y=0., z=0.),
        target_position_m=Vector3(x=3., y=0., z=3.), motion_context_sha256=protection.sha256)
    kwargs = dict(observation=observation, vehicle=vehicle(), static_primitives=[],
                  required_clearance_m=.15, generated_at_unix_ms=1010)
    with pytest.raises(ValueError, match="CONTEXT_BINDING"):
        evaluate_runtime_local_safety(**kwargs)
    command = evaluate_runtime_local_safety(**kwargs, motion_guard=protection)
    assert command.decision.action == "hold"
    assert command.observation_sha256 == sha256_json(observation)
    assert "LOADED_MOTION_DYNAMICS_UNQUALIFIED" in command.decision.issue_codes
    observation.current_position_m.x = 1.
    with pytest.raises(ValueError, match="POSITION_BINDING"):
        evaluate_runtime_local_safety(**kwargs, motion_guard=protection)


# 功能：
#   区分已资格化配置空间与原始测距空间，防止前向单束证据允许无观测的上升。
# 输入：
#   无：小型体素地图及一条静态认证走廊。
# 输出：
#   None：检查未认证空地、角点穿越与边界外空间均不能冒充静态走廊。
def test_qualified_coverage_requires_bound_route_and_every_crossed_cell():
    world = MetricVoxelMap(resolution_m=.5, minimum_bound_m=Vector3(x=-3., y=-3., z=0.),
                           maximum_bound_m=Vector3(x=3., y=3., z=5.))
    path = [Vector3(x=.25, y=.25, z=2.25), Vector3(x=1.25, y=.25, z=2.25)]
    world.seed_known_static_region([(world.center_for(k), False)
        for k in product(range(6, 11), range(6, 11), range(0, 10))], source_sha256="a" * 64)
    assert not world.qualified_static_path_covered(path)
    world.bind_qualified_route(path, route_sha256="b" * 64, minimum_clearance_m=.2)
    assert world.qualified_static_path_covered(path)
    assert not world.qualified_static_path_covered([path[0], Vector3(x=-1., y=.25, z=2.25)])
    assert not world.qualified_static_path_covered([path[0], Vector3(x=.25, y=.25, z=6.)])
    assert not world.qualified_static_path_covered(path, extra_clearance_m=.3)
    with pytest.raises(ValueError, match="coverage extra clearance"):
        world.qualified_static_path_covered(path, extra_clearance_m=-.1)


# 功能：
#   将首次负载测量与产品授权分开；缺失包络只在显式仿真采集中豁免且仍限速。
# 输入：
#   无：正常、未知质量、坏遥测及真机越权用例。
# 输出：
#   None：断言首批数据可采集但不能绕过其他必要约束。
def test_measurement_mode_is_simulation_only_and_never_production_qualification():
    protection = guard(payload=payload(), measurement_only=True)
    assert protection.context["measurement_only"] is True
    assert protection.context["dynamics_sha256"] is None
    assert protection.check(request(), (.1, 0., .1), [(0., 0., 2.), (.1, 0., 2.1)]) == []
    assert protection.acceleration_limit(1.) == .2
    assert protection.check(request(), (.3, 0., 0.), [(0., 0., 2.)]) == ["MEASUREMENT_SPEED_LIMIT"]
    assert "LOADED_MOTION_DYNAMICS_UNQUALIFIED" in guard(payload=payload()).issues
    assert "PAYLOAD_MASS_UNKNOWN" in guard(payload=payload(None), measurement_only=True).issues
    bad = payload()
    bad["dynamics"]["ready"] = False
    assert "MEASUREMENT_DYNAMICS_TELEMETRY_UNREADY" in guard(
        payload=bad, measurement_only=True).issues
    with pytest.raises(ValueError, match="MEASUREMENT_DOMAIN"):
        guard(payload=payload(), measurement_only=True, domain="hardware")
    assert "HARDWARE_MOTION_DYNAMICS_UNQUALIFIED" in guard(domain="hardware").issues


# 功能：
#   验证包络里的负载碰撞尺寸实际参与顶面净空，而不是只存入说明字段。
# 输入：
#   无：空载可通过但载荷包围高度过大的屋檐。
# 输出：
#   None：断言载荷增大时允许中心高度下降。
def test_loaded_collision_shape_tightens_ceiling():
    geometry = [box((0., 0., 3.), (4., 4., .2))]
    empty = guard(primitives=geometry)
    loaded = guard(primitives=geometry, envelope=envelope(), payload=payload())
    assert loaded.radius == .35
    assert loaded.height == .8
    assert loaded.ceiling == pytest.approx(empty.ceiling - .2)


# 功能：
#   检查斜楼板、俯仰墙体的所有实际角点均落在规划索引包围盒内。
# 输入：
#   angle：楼板绕 Y 轴倾角。
# 输出：
#   None：断言规划与碰撞模块共用包围盒，不漏掉真实端点。
@pytest.mark.parametrize("angle", [math.pi / 6, math.pi / 4, math.pi / 2, -math.pi / 3])
def test_tilted_stair_slab_planner_bounds_cover_geometry(angle):
    from dronedream_agent_core.known_map_planner import _primitive_bounds
    primitive = {**box((0., 0., 3.), (8., 2., .2)), "pitch_rad": angle}
    low, high = _primitive_bounds(primitive, .1)
    assert (low, high) == primitive_bounds(primitive, .1)
    for x, y, z in product((-4., 4.), (-1., 1.), (-.1, .1)):
        point = (math.cos(angle) * x + math.sin(angle) * z, y,
                 3. - math.sin(angle) * x + math.cos(angle) * z)
        assert all(low[i] <= point[i] <= high[i] for i in range(3))


# 功能：
#   检查连续制动扫掠遇到横穿车辆时的否决，车辆不能仅因当前不在无人机前方而被忽略。
# 输入：
#   无：交叉运动的合成轨迹。
# 输出：
#   None：断言整段停止空间冲突被识别。
def test_braking_sweep_includes_crossing_vehicle():
    from dronedream_agent_core.contracts import DynamicObstacleObservation
    car = DynamicObstacleObservation(obstacle_id="car", position_m=Vector3(x=1., y=2., z=1.5),
        velocity_mps=Vector3(x=0., y=-1., z=0.), radius_m=.4, height_m=1.5,
        confidence=1., age_seconds=.01)
    state = request(current_velocity_mps=Vector3(x=1., y=0., z=0.), dynamic_obstacles=[car])
    protection = guard(payload=payload(), envelope=envelope())
    assert "BRAKING_SWEPT_SPACE_CONFLICT" in protection.check(
        state, (.5, 0., 0.), [(0., 0., 2.), (.1, 0., 2.)])


# 功能：
#   确认测量文件缺失、跨包络复用和轨迹摘要变化不能靠 accepted 文本蒙混过关。
# 输入：
#   tmp_path：测试专用目录，不写入产品资格路径。
# 输出：
#   None：断言完整测试夹具可读，污染后立即拒绝。
def test_dynamics_loader_verifies_report_and_raw_sources(tmp_path):
    source_dir = tmp_path / "evidence"
    source_dir.mkdir()
    sources = []
    for split in ("calibration", "validation"):
        data = ("TEST-ONLY " + split).encode()
        (source_dir / (split + ".csv")).write_bytes(data)
        sources.append({"file": split + ".csv", "sha256": hashlib.sha256(data).hexdigest(),
                        "split": split})
    limits = envelope().model_dump(mode="json", exclude={"evidence_sha256"})
    report = {"schema_version": "dronedream.flight-dynamics-qualification.v1", "status": "accepted",
              "limits": limits, "measurement_sources": sources,
              "independent_validation_passed": True}
    raw = json.dumps(report).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (source_dir / (digest + ".json")).write_bytes(raw)
    path = tmp_path / "envelope.json"
    path.write_text(envelope(evidence_sha256=digest).model_dump_json(), encoding="utf-8")
    assert load_flight_dynamics_envelope(path).evidence_sha256 == digest
    (source_dir / "validation.csv").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="MEASUREMENT_HASH"):
        load_flight_dynamics_envelope(path)


# 功能：
#   验证背景能进入实际战略上下文，不因新增字段深度或宽度而在模型调用前崩溃。
# 输入：
#   无：采用本轮真实新增的背景结构。
# 输出：
#   None：断言云端和本地专家能读取同一竖直背景。
def test_vertical_context_passes_model_context_contract():
    from dronedream_agent_core.navigation_context import build_navigation_context
    protection = guard(payload=payload(), envelope=envelope())
    result = build_navigation_context(task={}, map_context={"vertical_motion": protection.context},
        sensor_context={}, vehicle=vehicle(),
        payload={**payload(), "vertical_motion": protection.context},
        rgb_enabled=True)
    assert result["map"]["vertical_motion"]["maximum_center_z_m"] is None
