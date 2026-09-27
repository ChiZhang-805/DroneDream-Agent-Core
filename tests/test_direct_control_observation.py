"""Equivalent structured observations without moving the live map into a worker."""

import random
from dataclasses import replace
from unittest.mock import Mock

import pytest
from test_adaptive_sensor_admission import coordinator, schedule

import dronedream_agent_core.perception_runtime as runtime
from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.local_world_model import MetricVoxelMap


# 功能：
#   核对不同位置和目标下，全图同步摘要与十米独占地图的八米摘要完全一致，包含原始摘要哈希。
# 输入：
#   resolution：体素分辨率。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize('resolution', [.25, .37, 1.])
def test_direct_metric_observation_matches_cropped_map_exactly(resolution):
    world = MetricVoxelMap(resolution_m=resolution,
        minimum_bound_m=Vector3(x=-25., y=-25., z=-25.),
        maximum_bound_m=Vector3(x=25., y=25., z=25.))
    rng = random.Random(70)
    edge = int(50 / resolution)
    for _ in range(9000):
        key = tuple(rng.randrange(edge) for _ in range(3))
        world._update_log_odds(key, measurement=rng.choice([-4., 0., 4.]), observed_at=1.)
    world.seed_known_static_region([(Vector3(x=0., y=0., z=0.), True),
                                    (Vector3(x=1., y=1., z=1.), False)], source_sha256='a' * 64)
    for _ in range(10):
        point = Vector3(**dict(zip(('x', 'y', 'z'), (rng.uniform(-10., 10.) for _ in range(3)))))
        goal = Vector3(**dict(zip(('x', 'y', 'z'), (rng.uniform(-20., 20.) for _ in range(3)))))
        args = dict(current_position_m=point, current_velocity_mps=Vector3(x=.1, y=-.2, z=.3),
                    goal_position_m=goal, required_clearance_m=.4, include_candidate_paths=False)
        clone = world.navigation_clone(center_m=point, radius_m=10.)
        assert world.text_navigation_snapshot(**args) == clone.text_navigation_snapshot(**args)


# 功能：
#   核对连续控制在地图所有者线程直接编译同一完整输入，后台仅接收独立 JSON 而不携带地图。
# 输入：
#   monkeypatch：监视编译输入和后台提交的夹具。
# 输出：
#   None：不返回业务数据。
def test_continuous_worker_gets_identical_owned_snapshot_without_map(monkeypatch):
    instance, port, executor = coordinator(monkeypatch)
    original = runtime.compile_navigation_snapshot
    compiled = []

    # 功能：
    #   用实际编译器比较直接读取与旧独占地图读取的全部输出，不用简化观测冒充等价。
    # 输入：
    #   request：同步编译输入。
    # 输出：
    #   snapshot：已核对的完整观测。
    def verify_compilation(request):
        assert request.world is instance.fusion.world
        assert not request.include_candidate_paths
        clone = request.world.navigation_clone(center_m=request.frame.localization_position_m, radius_m=10.)
        snapshot = original(request)
        assert snapshot == original(replace(request, world=clone))
        compiled.append(snapshot)
        return snapshot

    monkeypatch.setattr(runtime, 'compile_navigation_snapshot', verify_compilation)
    try:
        assert schedule(instance) is None
        args = executor.submit.call_args.kwargs
        assert args['request'] is None
        assert args['prepared_snapshot'] is compiled[0]
        assert instance.input_preparation_timing['world_freeze_ms'] == 0.
        before = runtime.sha256_json(compiled[0])
        instance.fusion.world._update_log_odds((0, 0, 0), measurement=4., observed_at=2.)
        assert runtime.sha256_json(compiled[0]) == before
    finally:
        instance.close()


# 功能：
#   验证后台不接受两种输入同时存在或同时缺失，避免错误分支绕过预期编译流程。
# 输入：
#   monkeypatch：阻止非法输入触发模型或编译器的夹具。
#   both：是否同时提供两种互斥输入。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize('both', [False, True])
def test_worker_rejects_ambiguous_input_ownership(monkeypatch, both):
    compiler, invoke = Mock(), Mock()
    monkeypatch.setattr(runtime, 'compile_navigation_snapshot', compiler)
    monkeypatch.setattr(runtime, 'request_text_navigation_decision', invoke)
    result = runtime._compile_and_request_navigation_decision(
        request=object() if both else None, prepared_snapshot={} if both else None,
        port=Mock(), context_id=None, multimodal=[])
    assert result.failure_reason == 'METRIC_NAVIGATION_SNAPSHOT_UNAVAILABLE'
    compiler.assert_not_called()
    invoke.assert_not_called()
