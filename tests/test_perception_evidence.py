"""Immutable internal reads must keep public schemas, clocks and safety intact."""

import threading

import pytest
from pydantic import ValidationError
from test_perception_runtime import _frame, _tracked_frame, _world

from dronedream_agent_core.contracts import OnboardPerceptionFrame
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.perception_evidence import FrozenPerceptionFrame
from dronedream_agent_core.perception_runtime import RuntimePerceptionFusion


# 功能：
#   构造只接纳测试前向雷达的独立融合器，不连接仿真或真实设备。
# 输入：
#   无。
# 输出：
#   runtime：持有独立体素地图的融合器。
def fusion():
    runtime = RuntimePerceptionFusion(world=_world(), accepted_sensor_ids={"front-lidar"})
    return runtime


# 功能：
#   构造带明确原始时钟和移动目标的单帧证据。
# 输入：
#   无。
# 输出：
#   frame：时间为一千 Unix 毫秒的合成观测。
def tracked():
    frame = _tracked_frame(
        sequence=1, observed_at_unix_ms=1000, obstacle_x=1.0, obstacle_speed_x=0.2
    )
    return frame


# 功能：
#   验证内部冻结表示与原始线协议的所有 JSON 数值和摘要完全一致。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_frozen_representation_keeps_complete_json_values_and_hash():
    source = tracked()
    frozen = FrozenPerceptionFrame.model_validate(source.model_dump(mode="python"))
    assert frozen.model_dump(mode="json") == source.model_dump(mode="json")
    assert sha256_json(frozen) == sha256_json(source)
    assert OnboardPerceptionFrame.model_validate_json(frozen.model_dump_json()) == source
    assert isinstance(frozen.range_rays, tuple)
    assert isinstance(frozen.dynamic_obstacles, tuple)


# 功能：
#   验证顶层字段及嵌套向量、射线、动态目标都不能经普通赋值改变。
# 输入：
#   path：待定位的嵌套对象路径。
#   field、value：尝试改写的字段和值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "path,field,value",
    [
        ((), "sequence", 42),
        (("localization_position_m",), "x", 42.0),
        (("localization_velocity_mps",), "y", 42.0),
        (("range_rays", 0), "confidence", 0.1),
        (("range_rays", 0, "endpoint_m"), "z", 42.0),
        (("range_rays", 0, "origin_m"), "x", 42.0),
        (("dynamic_obstacles", 0), "age_seconds", 42.0),
        (("dynamic_obstacles", 0, "position_m"), "x", 42.0),
        (("dynamic_obstacles", 0, "velocity_mps"), "y", 42.0),
    ],
)
def test_all_nested_values_reject_assignment(path, field, value):
    frame = FrozenPerceptionFrame.model_validate(tracked().model_dump(mode="python"))
    digest = sha256_json(frame)
    node = frame
    for step in path:
        node = node[step] if isinstance(step, int) else getattr(node, step)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(node, field, value)
    assert sha256_json(frame) == digest


# 功能：
#   验证射线集合及默认动态集合为不可变元组，不能追加或替换元素。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_collections_and_default_dynamic_list_are_immutable():
    frame = FrozenPerceptionFrame.model_validate(_frame().model_dump(mode="python"))
    with pytest.raises(AttributeError):
        frame.range_rays.append(frame.range_rays[0])
    with pytest.raises(TypeError):
        frame.range_rays[0] = frame.range_rays[0]
    source = _frame().model_dump(mode="python")
    source.pop("dynamic_obstacles")
    assert FrozenPerceptionFrame.model_validate(source).dynamic_obstacles == ()


# 功能：
#   验证冻结容器仍遵守公共字段的必填、容量和有限数值约束。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_original_container_constraints_remain_authoritative():
    for name in ("range_rays", "dynamic_obstacles"):
        original = OnboardPerceptionFrame.model_fields[name]
        frozen = FrozenPerceptionFrame.model_fields[name]
        assert frozen.metadata == original.metadata
        assert frozen.is_required() == original.is_required()
    source = _frame().model_dump(mode="python")
    source["range_rays"] = []
    with pytest.raises(ValidationError):
        FrozenPerceptionFrame.model_validate(source)
    source = _frame().model_dump(mode="python")
    source["range_rays"][0]["endpoint_m"]["x"] = float("nan")
    with pytest.raises(ValidationError):
        FrozenPerceptionFrame.model_validate(source)


# 功能：
#   验证原始帧赋值动态目标时重验新目标，但不无故重建未变化的射线列表。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_owned_frame_dynamic_assignment_validates_without_rebuilding_rays():
    source = _frame()
    rays = source.range_rays
    source.dynamic_obstacles = tracked().dynamic_obstacles
    assert source.range_rays is rays
    assert len(source.dynamic_obstacles) == 1
    with pytest.raises(ValidationError):
        source.dynamic_obstacles = [{"obstacle_id": "incomplete"}]


# 功能：
#   验证来源对象、公共可变副本和下一帧替换都不能改变已接纳或已保留的历史证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_source_and_public_copies_cannot_mutate_accepted_or_retained_evidence():
    runtime = fusion()
    source = tracked()
    runtime.ingest(source, now_unix_ms=1020, now_monotonic_seconds=1.02)
    original = runtime.frozen_frame
    digest = sha256_json(original)
    assert runtime.frozen_frame is original  # Read is not another frame clone.
    source.range_rays[0].endpoint_m.x += 0.1
    source.dynamic_obstacles[0].position_m.x = 8.0
    source.dynamic_obstacles.clear()
    public = runtime.latest_frame
    public.range_rays.clear()
    public.localization_position_m.x = 90.0
    public.dynamic_obstacles[0].velocity_mps.x = 9.0
    assert sha256_json(runtime.frozen_frame) == digest
    # Track prediction state must also own its vectors, not just its record.
    track = next(iter(runtime._dynamic_tracks.values()))
    assert track.position_m.x == 1.0
    assert track.position_m is track.observation.position_m
    next_frame = _tracked_frame(
        sequence=2, observed_at_unix_ms=1100, obstacle_x=1.02, obstacle_speed_x=0.2
    )
    runtime.ingest(next_frame, now_unix_ms=1120, now_monotonic_seconds=1.12)
    assert runtime.frozen_frame is not original
    assert sha256_json(original) == digest
    assert original.observed_at_unix_ms == 1000
    assert runtime.frozen_frame.observed_at_unix_ms == 1100


# 功能：
#   验证绕过赋值验证的非法置信度在提交地图或发布帧之前被拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_invalid_source_fails_before_world_or_source_commit():
    runtime = fusion()
    source = _frame()
    # Deliberately bypass a caller's assignment validation: the fusion trust
    # boundary must still validate before changing the accepted state or map.
    object.__setattr__(source.range_rays[0], "confidence", -1.0)
    with pytest.raises(ValidationError):
        runtime.ingest(source, now_unix_ms=1020, now_monotonic_seconds=1.02)
    assert runtime.frozen_frame is None
    assert runtime.world.observation_count == 0


# 功能：
#   用真实线程验证读取者可保留旧不可变帧，生产者发布新帧不会改写旧证据。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_reader_can_retain_exact_frame_while_producer_replaces_it():
    runtime = fusion()
    runtime.ingest(_frame(), now_unix_ms=1020, now_monotonic_seconds=1.02)
    ready, release = threading.Event(), threading.Event()
    observed = []

    # 功能：
    #   固定读入的帧并在生产者替换后再次核对摘要，结果写入本测试列表。
    # 输入：
    #   无；通过闭包访问融合器与同步事件。
    # 输出：
    #   None：不返回业务数据。
    def reader():
        frame = runtime.frozen_frame
        digest = sha256_json(frame)
        ready.set()
        if release.wait(2):
            observed.append((frame.sequence, digest == sha256_json(frame)))

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert ready.wait(2)
        source = _frame().model_dump(mode="python")
        source.update(sequence=2, observed_at_unix_ms=1100)
        for ray in source["range_rays"]:
            ray["observed_at_monotonic_seconds"] = 1.1
        runtime.ingest(
            OnboardPerceptionFrame.model_validate(source),
            now_unix_ms=1120,
            now_monotonic_seconds=1.12,
        )
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert observed == [(1, True)]


# 功能：
#   验证融合入口在执行跟踪或地图计算之前拒绝伪类型射线，而不是把字符串转换成测量。
# 输入：
#   field、value：绕过赋值验证后注入的错误射线字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,value", [("hit", "false"), ("confidence", True)])
def test_fusion_rejects_coerced_ray_types_before_state_changes(field, value):
    runtime = fusion()
    source = _frame()
    object.__setattr__(source.range_rays[0], field, value)
    with pytest.raises(ValueError):
        runtime.ingest(source, now_unix_ms=1020, now_monotonic_seconds=1.02)
    assert runtime.frozen_frame is None
    assert runtime.world.observation_count == 0


# 功能：
#   验证融合过程中原始可变帧的更新不会改写已进入本次处理链的射线。
# 输入：
#   monkeypatch：在动态验证阶段修改调用方原始帧。
# 输出：
#   None：不返回业务数据。
def test_fusion_owns_rays_before_tracking_and_reuses_one_frozen_graph(monkeypatch):
    runtime = fusion()
    source = _frame()
    expected = source.range_rays[0].endpoint_m.model_dump()
    original = runtime._validated_dynamic_tracks
    admitted_rays = []

    # 功能：
    #   模拟生产者在跟踪验证期间更新自己的端点，并保留消费者实际收到的射线图。
    # 输入：
    #   frame：融合器送入验证阶段的帧。
    # 输出：
    #   result：原始动态验证结果。
    def mutate_original(frame):
        admitted_rays.append(frame.range_rays)
        source.range_rays[0].endpoint_m.x = 9.0
        result = original(frame)
        return result

    monkeypatch.setattr(runtime, "_validated_dynamic_tracks", mutate_original)
    runtime.ingest(source, now_unix_ms=1020, now_monotonic_seconds=1.02)
    assert runtime.frozen_frame.range_rays[0].endpoint_m.model_dump() == expected
    assert runtime.frozen_frame.range_rays is admitted_rays[0]


# 功能：
#   验证无传感器注册表时，真假值也不能冒充融合或健康查询的本机时钟。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_boolean_clock_never_becomes_fusion_time():
    runtime = fusion()
    with pytest.raises(ValueError, match="CLOCK_INVALID"):
        runtime.ingest(_frame(), now_unix_ms=1020, now_monotonic_seconds=True)
    with pytest.raises(ValueError, match="CLOCK_INVALID"):
        runtime.health(now_unix_ms=1020, now_monotonic_seconds=True)
