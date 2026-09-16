from dronedream_agent_core.contracts import GraphRoute, Vector3, VehicleAsset
from dronedream_agent_plugins.planning_quality_plugins import _energy_reserve_gate


# 功能：
#   构造已包含电量储备约束的合格航程资产，用于核对门控不会重复扣除储备。
# 输入：
#   无。
# 输出：
#   vehicle：合格航程为 400 米、储备为百分之二十的测试无人机。
def _vehicle() -> VehicleAsset:
    vehicle = VehicleAsset(
        asset_id="vehicle-a",
        name="vehicle",
        dry_mass_kg=1,
        max_takeoff_mass_kg=2,
        body_radius_m=0.2,
        body_height_m=0.2,
        max_speed_mps=2,
        max_acceleration_mps2=2,
        qualified_range_m=400,
        reserve_battery_percent=20,
        max_pickup_payload_kg=0.5,
        sensors=["camera"],
    )
    return vehicle


# 功能：
#   构造指定长度的单边直线路径，隔离能耗门控与寻路实现。
# 输入：
#   length_m：路径总长，单位米。
# 输出：
#   route：端点距离与声明长度一致的测试路径。
def _route(length_m: float) -> GraphRoute:
    route = GraphRoute(
        start_node="start",
        goal_node="goal",
        node_ids=["start", "goal"],
        edge_ids=["edge"],
        positions_m=[Vector3(x=0, y=0, z=1), Vector3(x=length_m, y=0, z=1)],
        route_length_m=length_m,
        all_edges_flight_verified=True,
    )
    return route


# 功能：
#   验证刚好等于资产合格航程的路径可通过，不再重复扣除已计入的电量储备。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_energy_gate_uses_vehicle_qualified_range_without_double_reserving() -> None:
    result = _energy_reserve_gate(route=_route(400), vehicle=_vehicle())

    assert result["accepted"] is True
    assert result["usable_range_m"] == 400


# 功能：
#   验证边界比较容忍序列化舍入误差，但不放过真实的航程超限。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_energy_gate_accepts_only_sub_micrometre_serialization_noise() -> None:
    serialized_equal = _energy_reserve_gate(
        route=_route(400.00000000000006),
        vehicle=_vehicle(),
    )
    real_overrun = _energy_reserve_gate(
        route=_route(400.0001),
        vehicle=_vehicle(),
    )

    assert serialized_equal["accepted"] is True
    assert real_overrun["accepted"] is False


# 功能：
#   验证配置只能缩短可用航程或提高储备，不能放大资产认证的航程包络。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_energy_gate_configuration_can_tighten_but_never_expand_asset_envelope() -> None:
    expanded = _energy_reserve_gate(
        route=_route(401),
        vehicle=_vehicle(),
        configuration={"qualified_range_m": 1000, "reserve_fraction": 0.05},
    )
    tightened = _energy_reserve_gate(
        route=_route(350),
        vehicle=_vehicle(),
        configuration={"qualified_range_m": 400, "reserve_fraction": 0.4},
    )

    assert expanded["accepted"] is False
    assert expanded["qualified_range_m"] == 400
    assert expanded["reserve_fraction"] == 0.2
    assert tightened["accepted"] is False
    assert tightened["usable_range_m"] == 300
