"""Fault injection for the independent simulation witness, without Gazebo."""

from types import SimpleNamespace

import pytest
from test_training_outcome_channel import entity

from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.training import gazebo_witness as witness_module


# 功能：
#   建立仅收集内存帧的真值采集器，保留可修改的测试时钟以触发时钟故障。
# 输入：
#   monkeypatch：替换真值模块时钟的测试工具。
# 输出：
#   fixture：采集器、已发布帧与当前时钟组成的三元组。
@pytest.fixture
def observer(monkeypatch):
    rows, clock = [], [10.0]
    monkeypatch.setattr(witness_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    instance = witness_module.GazeboOutcomeWitness(
        publisher=SimpleNamespace(send=rows.append),
        resolve_pose=lambda poses: ((0.0, 0.0, 1.0),),
        collision_offset=[0.0, 0.0, 0.0],
        target=Vector3(x=1.0, y=0.0, z=1.0),
    )
    instance.dynamic_geometry = {"person_worker": 0.8, "person_A": 0.8, "person_a": 0.8}
    fixture = instance, rows, clock
    return fixture


# 功能：
#   验证时钟在首次读取、年龄计算或退出统计时失败，均不能逃逸或遗留互斥锁。
# 输入：
#   observer：独立采集器及输出。
#   monkeypatch：注入指定次序的时钟故障。
#   failure_index：发生故障的时钟调用序号，从零开始。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_clock_failure_never_leaks_callback_lock(observer, monkeypatch, failure_index):
    instance, rows, _ = observer
    readings = [10.0, 10.001, 10.002]
    readings[failure_index] = RuntimeError("clock unavailable")

    # 功能：
    #   按调用顺序返回时钟或抛出故障，后续调用保持可用。
    # 输入：
    #   无；读取外层 readings 队列。
    # 输出：
    #   reading：当前单调时刻，秒。
    def monotonic():
        reading = readings.pop(0) if readings else 10.003
        if isinstance(reading, Exception):
            raise reading
        return reading

    monkeypatch.setattr(witness_module, "time", SimpleNamespace(monotonic=monotonic))
    instance.receive([], 10.0, 1000, simulation_time_ns=0)
    assert instance.error is not None and "clock unavailable" in instance.error
    assert not instance._lock.locked()
    assert len(rows) <= (1 if failure_index == 2 else 0)


# 功能：
#   验证重复原始名称及大小写归一后冲突的名称被拒绝，不静默覆盖障碍。
# 输入：
#   observer：独立采集器与输出。
#   names：两种有歧义的动态实体身份。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("names", [("person_worker", "person_worker"), ("person_A", "person_a")])
def test_ambiguous_dynamic_identity_is_rejected(observer, names):
    instance, rows, _ = observer
    instance.receive([entity(name) for name in names], 10.0, 1000, simulation_time_ns=0)
    assert instance.error is not None and "DYNAMIC_IDENTITY" in instance.error
    assert rows == []


# 功能：
#   验证解析器遍历一次性位姿迭代器后，障碍物信息仍来自同一完整帧。
# 输入：
#   observer：待检查的独立采集器。
# 输出：
#   None：不返回业务数据。
def test_pose_iterator_not_consumed_before_obstacles(observer):
    instance, rows, _ = observer
    instance.resolve_pose = lambda poses: (tuple(0.0 for _ in list(poses)[:3]),)
    instance.receive(
        iter([entity("drone"), entity("link"), entity("person_worker")]),
        10.0,
        1000,
        simulation_time_ns=0,
    )
    assert instance.error is None
    assert [item.obstacle_id for item in rows[0].dynamic_obstacles] == ["person_worker"]


# 功能：
#   验证缺少已声明碰撞几何的实体不能自动变成半径 0.35 米、高 1.7 米的行人。
# 输入：
#   observer：缺少车体几何声明的采集器。
# 输出：
#   None：不返回业务数据。
def test_missing_geometry_never_guesses_human_shape(observer):
    instance, rows, _ = observer
    instance.receive([entity("vehicle_dynamic_cart")], 10.0, 1000, simulation_time_ns=0)
    assert instance.error is not None and "GEOMETRY" in instance.error
    assert rows == []


# 功能：
#   验证构造后修改调用方碰撞偏移列表不会改变观测位置。
# 输入：
#   observer：提供测试时钟的夹具。
# 输出：
#   None：不返回业务数据。
def test_constructor_owns_collision_offset(observer):
    _, rows, _ = observer
    offset = [0.0, 0.0, 0.2]
    instance = witness_module.GazeboOutcomeWitness(
        publisher=SimpleNamespace(send=rows.append),
        resolve_pose=lambda poses: ((0.0, 0.0, 0.0),),
        collision_offset=offset,
        target=Vector3(x=1.0, y=0.0, z=1.0),
    )
    offset[2] = 100.0
    instance.receive([], 10.0, 1000, simulation_time_ns=0)
    assert rows[0].current_position_m.z == 0.2


# 功能：
#   验证源帧处理期间主机时钟倒退或变成 NaN，不把负年龄截零伪装成健康帧。
# 输入：
#   observer：待检查采集器。
#   monkeypatch：替换顺序时钟。
#   second_reading：处理末端非法时刻。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("second_reading", [9.0, float("nan")])
def test_processing_clock_regression_does_not_rejuvenate(observer, monkeypatch, second_reading):
    instance, rows, _ = observer
    readings = iter([10.0, second_reading, 10.001])
    monkeypatch.setattr(witness_module, "time", SimpleNamespace(monotonic=lambda: next(readings)))
    instance.receive([], 10.0, 1000, simulation_time_ns=0)
    assert instance.error is not None and rows == []


# 功能：
#   验证首次观测仅建立速度基线，第二帧才有物理速度；发布拒绝后不可继续伪装成功。
# 输入：
#   observer：采集器与可步进时钟。
# 输出：
#   None：不返回业务数据。
def test_velocity_readiness_and_failed_publish_are_explicit(observer):
    instance, rows, clock = observer
    instance.receive([entity("person_worker")], 10.0, 1000, simulation_time_ns=0)
    assert not rows[0].stream_healthy and rows[0].dynamic_obstacles[0].confidence == 0
    clock[0] = 10.1
    instance.receive([entity("person_worker", x=0.2)], 10.1, 1100, simulation_time_ns=100_000_000)
    assert rows[1].stream_healthy and rows[1].dynamic_obstacles[0].velocity_mps.x == 2
    instance.publisher.send = lambda observation: False
    clock[0] = 10.2
    instance.receive([], 10.2, 1200, simulation_time_ns=200_000_000)
    assert "PUBLISH_REJECTED" in instance.error
    assert instance._sequence == 2 and not instance._lock.locked()


# 功能：
#   验证解析器原地改变动态位置时，已经固定的该帧证据不受影响。
# 输入：
#   observer：采集器与输出帧。
# 输出：
#   None：不返回业务数据。
def test_resolver_does_not_change_frozen_dynamic_position(observer):
    instance, rows, _ = observer

    # 功能：
    #   模拟不纯的解析回调，在返回机体位置之前改变输入物体。
    # 输入：
    #   poses：实体帧。
    # 输出：
    #   resolved：有效机体位置结果。
    def resolver(poses):
        poses[0].position.z = 100.0
        resolved = ((0.0, 0.0, 1.0),)
        return resolved

    instance.resolve_pose = resolver
    instance.receive([entity("person_worker", z=2.0)], 10.0, 1000, simulation_time_ns=0)
    assert rows[0].dynamic_obstacles[0].position_m.z == 2.0
