import pytest
from types import SimpleNamespace
from dronedream_agent_core.contracts import HybridControlLease, RuntimeLocalSafetyCommand

from dronedream_agent_core.hybrid_control import BoundedHybridArbiter
from dronedream_agent_core.hybrid_control import require_hybrid_provider


# 功能：仿真采集可显式使用已有有界衔接，不扩大云端或无授权教师权限；输出：组合校验。
@pytest.mark.parametrize("provider,channel,authority,teacher,allowed", [
    ("local-policy", None, True, False, True),
    ("simulation-training", "owned-channel", True, False, True),
    ("simulation-training", None, True, False, False),
    ("cloud", "owned-channel", True, False, False),
    ("simulation-training", "owned-channel", False, False, False),
    ("simulation-training", "owned-channel", True, True, False),
])
def test_hybrid_provider_scope(provider, channel, authority, teacher, allowed):
    arguments = dict(provider=provider, training_channel=channel,
                     model_authority=authority, teacher_control=teacher)
    if allowed:
        require_hybrid_provider(**arguments)
    else:
        with pytest.raises(ValueError, match="HYBRID_REQUIRES"):
            require_hybrid_provider(**arguments)


# 功能：构造同一路线和新鲜独立观测，不把测试桩当作真实飞行证明。
# 输入：时间、模型状态和需要覆盖的边界字段。
# 输出：仲裁结果。
def tick(arbiter, t, **changes):
    data = dict(now=t, now_unix_ms=1000 + round(t * 1000), position=(0., 0., 0.),
        route_sha256='a' * 64, goal_id='office', sensors_and_route_ready=True,
        model_live=False, model_call_id=None, model_reason='model-lease-expired')
    data.update(changes)
    data.setdefault("model_speed_mps", .25 if data["model_live"] else None)
    return arbiter.update(**data)


# 功能：衔接不加速越过最新模型低速要求，也不把零速度当默认巡航许可。
# 输入：合成独立安全状态和已归一化物理速度；输出：低速保持、零速/缺失证据撤销。
@pytest.mark.parametrize("speed", [.03, .15, .4])
def test_bridge_preserves_model_speed_ceiling(speed):
    a = BoundedHybridArbiter()
    for index, t in enumerate((0., .15, .31)):
        tick(a, t, model_live=True, model_call_id=str(index), model_speed_mps=speed)
    assert tick(a, .4).maximum_speed_mps == min(.25, speed)
    tick(a, .5, model_live=True, model_call_id="zero", model_speed_mps=0.)
    assert tick(a, .6) is None


# 功能：未知或坏速度不继承旧高速度授权；输入：无/非法标量；输出：撤销并拒绝。
@pytest.mark.parametrize("speed", [None, True, float("nan"), float("inf"), -1., 21.])
def test_unknown_model_speed_revokes_bridge(speed):
    a = BoundedHybridArbiter()
    arm(a)
    with pytest.raises(ValueError, match="MODEL_SPEED_REQUIRED"):
        tick(a, .4, model_live=True, model_call_id="bad", model_speed_mps=speed)
    assert tick(a, .5) is None


# 功能：用不同的新鲜模型响应建立恢复窗口。
# 输入：仲裁器及起始时刻。
# 输出：无；不授予接管许可。
def arm(a, start=0):
    for index, dt in enumerate((0, .15, .31)):
        assert tick(a, start + dt, model_live=True, model_call_id=str(index)) is None


def test_never_bootstrap_route_motion_without_model():
    a = BoundedHybridArbiter()
    assert tick(a, 0) is None
    assert tick(a, 100) is None


def test_lease_has_fixed_deadline_and_cannot_renew_after_timeout():
    a = BoundedHybridArbiter()
    arm(a)
    first = tick(a, .4)
    assert first.maximum_speed_mps == .25
    assert tick(a, .8).expires_at_unix_ms == first.expires_at_unix_ms
    assert tick(a, 1.9) is None
    assert tick(a, 2.0) is None


@pytest.mark.parametrize('reason', ['model-requested-hold', 'model-lease-closed',
    'model-lease-goal-changed', 'model-lease-not-established', 'MODEL_NAVIGATION_INVOCATION_FAILED'])
def test_rejection_or_unknown_error_is_not_delay(reason):
    a = BoundedHybridArbiter()
    arm(a)
    assert tick(a, .4, model_reason=reason) is None
    assert tick(a, .5) is None


@pytest.mark.parametrize('changes', [dict(sensors_and_route_ready=False),
    dict(goal_id='return'), dict(route_sha256='b' * 64), dict(now_unix_ms=0)])
def test_bad_inputs_or_context_change_revoke(changes):
    a = BoundedHybridArbiter()
    arm(a)
    assert tick(a, .4, **changes) is None


def test_path_length_counts_return_motion_and_model_interruption():
    a = BoundedHybridArbiter()
    arm(a)
    assert tick(a, .4)
    assert tick(a, .5, position=(.2, 0., 0.), model_live=True, model_call_id='x') is None
    assert tick(a, .6, position=(0., 0., 0.)) is None
    assert tick(a, .7) is None


def test_isolated_model_response_does_not_reset_episode():
    a = BoundedHybridArbiter()
    arm(a)
    first = tick(a, .4)
    tick(a, .5, model_live=True, model_call_id='x')
    assert tick(a, .6).episode == first.episode
    assert tick(a, 1.9) is None


def test_same_call_cannot_rearm_and_stable_new_calls_can():
    a = BoundedHybridArbiter()
    for t in (0., .2, .4):
        tick(a, t, model_live=True, model_call_id='one')
    assert tick(a, .5) is None
    arm(a, 1.)
    assert tick(a, 1.4)


def test_frequent_good_responses_do_not_overflow_call_set():
    a = BoundedHybridArbiter()
    for i in range(40):
        tick(a, i * .01, model_live=True, model_call_id=str(i))
    assert tick(a, .4)


def test_short_delay_between_real_responses_does_not_prevent_arming():
    """功能：允许短延迟贯穿恢复；输入：三个真实响应夹少量间隙；输出：有界租期。"""
    a = BoundedHybridArbiter()
    for index, t in enumerate((0., .2, .4)):
        tick(a, t, model_live=True, model_call_id=f'call-{index}')
        if index < 2:
            assert tick(a, t + .1) is None
    assert tick(a, .5) is not None
    assert a.snapshot()['reason'] == 'bounded-delay-bridge'


def test_old_model_history_cannot_arm_after_stall():
    """功能：防止旧预热回执复用；输入：长时间无新响应；输出：不接管。"""
    a = BoundedHybridArbiter()
    arm(a)
    assert tick(a, 5.) is None
    a.invalidate('worker-state-unavailable')
    assert a.snapshot()['reason'] == 'worker-state-unavailable'


@pytest.mark.parametrize('code', ['LOCAL_POLICY_CONTROL_DEADLINE_INVALID',
    'MODEL_NAVIGATION_INVOCATION_FAILED', 'UNKNOWN_FAILURE'])
def test_model_contract_failure_cannot_hide_behind_expired_lease(code):
    """功能：新硬故障撤销旧接管条件；输入：故障后许可过期；输出：不接管。"""
    a = BoundedHybridArbiter()
    arm(a)
    assert a.observe_model_failure(code)
    assert tick(a, .4) is None


def test_explicit_inference_expiration_allows_bounded_delay_check():
    """功能：延迟不伪装硬故障；输入：已武装后的真实输入过期；输出：仍需有界许可。"""
    a = BoundedHybridArbiter()
    arm(a)
    assert not a.observe_model_failure('CONTROL_SOURCE_EXPIRED_DURING_INFERENCE')
    assert tick(a, .4)


# 功能：构造经过实际合同验证的接管命令，用来检验消费者拒绝伪造授权。
# 输入：固定测试时刻；输出：短租期命令，不是飞行数据。
def bridge_command(now=1000):
    from test_runtime_commands import _safety_command_fixture
    return _safety_command_fixture(navigation_control_authority='bounded-hybrid',
        navigation_goal_id='office', generated_at_unix_ms=now, valid_until_unix_ms=now + 200,
        evaluated_target_position_m=dict(x=1., y=0., z=1.),
        hybrid_lease=HybridControlLease(route_sha256='a' * 64, navigation_goal_id='office',
            episode=1, started_at_unix_ms=now, expires_at_unix_ms=now + 1500,
            maximum_speed_mps=.25, maximum_distance_m=.35),
        observation_budget=dict(source_observed_at_unix_ms=now - 10,
            control_deadline_unix_ms=now + 200, disposition='control-eligible',
            reason='fresh-test-observation', clearance_margin_m=1., uncertainty_margin_m=.1,
            downstream_reserve_ms=40),
        decision=SimpleNamespace(action='continue', selected_velocity_mps=dict(x=.1, y=.1, z=0.)))


@pytest.mark.parametrize('changes', [dict(hybrid_lease=None), dict(navigation_goal_id='return'),
    dict(source='simulation-ground-truth'), dict(model_navigation_authorized=True),
    dict(navigation_control_authority='route-fallback'), dict(observation_budget=None),
    dict(valid_until_unix_ms=2600), dict(estimator_to_world_position_offset_m=dict(x=.1, y=0., z=0.))])
def test_hybrid_contract_rejects_forged_binding(changes):
    data = bridge_command().model_dump()
    data.update(changes)
    with pytest.raises(ValueError):
        RuntimeLocalSafetyCommand.model_validate(data)


def test_hybrid_contract_caps_full_3d_speed():
    data = bridge_command().model_dump()
    data['decision']['selected_velocity_mps'] = dict(x=.2, y=.2, z=.2)
    with pytest.raises(ValueError, match='HYBRID_COMMAND_SPEED'):
        RuntimeLocalSafetyCommand.model_validate(data)


# 功能：建立已有三个不同模型实际接收回执的执行器测试状态。
# 输入：无；输出：独立授权检查所需的固定测试上下文。
def executor_args():
    return SimpleNamespace(bounded_hybrid_control=True, require_model_control_authority=True,
        hybrid_route_sha256='a' * 64,
        _hybrid_model_receipts=(('1', 'office', 700), ('2', 'office', 800), ('3', 'office', 900)))


@pytest.mark.parametrize('changes', [dict(bounded_hybrid_control=False),
    dict(hybrid_route_sha256='b' * 64), dict(_hybrid_model_receipts=()),
    dict(_hybrid_model_receipts=(('1', 'office', 700),) * 3)])
def test_executor_requires_explicit_mode_route_and_actual_model_receipts(changes):
    from test_runtime_commands import _load_executor
    args = executor_args()
    vars(args).update(changes)
    assert not _load_executor()._hybrid_transport_admissible(args,
        SimpleNamespace(north_m=0., east_m=0., down_m=-1.), bridge_command(), now=10., now_unix_ms=1000)


def test_executor_distance_and_deadline_are_independent_of_producer():
    from test_runtime_commands import _load_executor
    guard = _load_executor()._hybrid_transport_admissible
    args = executor_args()
    cmd = bridge_command()
    pos = SimpleNamespace(north_m=0., east_m=0., down_m=-1.)
    assert guard(args, pos, cmd, now=10., now_unix_ms=1000)
    pos.north_m = .2
    assert guard(args, pos, None, now=10.1, now_unix_ms=1100)
    pos.north_m = 0.
    assert not guard(args, pos, cmd, now=10.2, now_unix_ms=1200)
    assert not guard(args, pos, cmd, now=12., now_unix_ms=1100)


def test_executor_sends_only_verified_velocity_and_records_hybrid(tmp_path):
    import asyncio
    import time
    from test_runtime_commands import _load_executor
    from dronedream_agent_core.contracts import Px4CoordinateContract
    executor = _load_executor()
    now = int(time.time() * 1000)
    command = bridge_command(now)
    args = executor_args()
    args._hybrid_model_receipts = tuple((str(i), 'office', now - 300 + i * 100) for i in range(3))
    args.local_safety_target = None
    args.local_safety_command = tmp_path / 'command.json'
    args.local_safety_repair_timeout_seconds = 1.
    args.setpoint_rate_hz = 50.
    observed = SimpleNamespace(north_m=0., east_m=0., down_m=-1.)

    async def refresh(**kwargs):
        return observed

    class Client:
        def __init__(self):
            self.sent = []

        def latest_dynamics_telemetry(self, age):
            return {'sources': {'attitude': {'yaw_deg': 12., 'sample_age_seconds': .01}}}

        async def set_velocity_ned(self, velocity):
            self.sent.append(velocity)

    executor._read_local_safety_command = lambda args: command
    executor._refresh_px4_identity_telemetry = refresh
    client = Client()
    asyncio.run(executor._apply_local_safety(args=args,
        base=SimpleNamespace(Setpoint=lambda **kw: SimpleNamespace(**kw),
            VelocitySetpoint=lambda **kw: SimpleNamespace(**kw)), client=client,
        planned_setpoint=SimpleNamespace(north_m=50., east_m=40., down_m=-8., yaw_deg=90.),
        coordinate_contract=Px4CoordinateContract(model_root_world_enu_m=[0., 0., 0.],
            collision_center_offset_model_m=[0., 0., .2]),
        phase_path=tmp_path / 'phase.json', navigation_goal_id='office'))
    assert len(client.sent) == 1
    assert args._model_control_application_counts == {'bounded-hybrid-bridge': 1}
    assert getattr(args, '_model_authorized_control_applied_count', 0) == 0
    assert client.sent[0].north_m_s == pytest.approx(.1)
    assert client.sent[0].east_m_s == pytest.approx(.1)
    assert client.sent[0].yaw_deg == 12.


@pytest.mark.parametrize('healthy,covered', [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize('lease_start', [1000, 1010])
def test_bridge_runs_real_safety_solver_and_keeps_vetoes(healthy, covered, lease_start):
    from test_vertical_navigation import guard, vehicle
    from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation, Vector3
    from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety
    protection = guard(coverage_check=lambda path, margin: covered)
    observation = RuntimeLocalSafetyObservation(sequence=1, observed_at_unix_ms=1000,
        source='onboard', stream_healthy=healthy, stream_age_seconds=.01,
        localization_covariance_m2=0., current_position_m=Vector3(x=0., y=0., z=2.),
        current_velocity_mps=Vector3(x=0., y=0., z=0.),
        target_position_m=Vector3(x=1., y=0., z=2.), motion_context_sha256=protection.sha256)
    command = evaluate_runtime_local_safety(observation=observation, vehicle=vehicle(),
        static_primitives=[], required_clearance_m=.15, generated_at_unix_ms=1010,
        navigation_control_authority='bounded-hybrid', navigation_goal_id='office',
        hybrid_lease=bridge_command(lease_start).hybrid_lease, motion_guard=protection)
    assert command.navigation_control_authority == 'bounded-hybrid'
    assert command.model_navigation_authorized is False
    if healthy and covered:
        assert command.decision.action in {'continue', 'slow'}
        velocity = command.decision.selected_velocity_mps
        assert (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** .5 <= .25 + 1e-9
        assert command.valid_until_unix_ms <= command.observation_budget.control_deadline_unix_ms
    else:
        assert command.decision.action == 'hold'
