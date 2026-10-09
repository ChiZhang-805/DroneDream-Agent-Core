import asyncio
import importlib.util
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dronedream_agent_core.contracts import (
    BodyFrameControlIntent,
    ModelCallRecord,
    Px4CoordinateContract,
    Px4Track,
    RuntimeAmendmentDirective,
    RuntimeCommandAdoption,
    RuntimeHoldAcknowledgement,
    RuntimeInterruptionDecision,
    RuntimeLocalSafetyCommand,
    RuntimeMessageClassification,
    RuntimeUserMessage,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.runtime_commands import build_runtime_command


# 功能：
#   用完整安全命令契约构造执行测试输入，保留真实序列化与摘要，不再使用残缺命名空间。
# 输入：
#   values：本测试明确覆盖的指令字段；decision 为对应安全决策的字段命名空间。
# 输出：
#   command：通过正式字段校验的独立测试指令，不来自实际飞行。
def _safety_command_fixture(**values):
    decision = vars(values.pop('decision'))
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    zero = Vector3(x=0., y=0., z=0.)
    data = dict(observation_sha256='f' * 64, observation_sequence=1,
                generated_at_unix_ms=now_ms, valid_until_unix_ms=now_ms + 1500,
                source='onboard', command_position_m=zero)
    data.update(values)
    data['decision'] = dict(selected_velocity_mps=zero, predicted_path_m=[zero],
        minimum_predicted_clearance_m=2., time_to_minimum_clearance_seconds=0.,
        evaluated_candidate_count=1,
        control_source='deterministic-brake' if decision['action'] == 'hold' else 'route-target')
    data['decision'].update(decision)
    command = RuntimeLocalSafetyCommand.model_validate(data)
    return command


def _load_executor() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "px4_checkpoint_executor.py"
    spec = importlib.util.spec_from_file_location("test_px4_checkpoint_executor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_depth_worker() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "runtime_depth_safety_worker.py"
    spec = importlib.util.spec_from_file_location("test_runtime_depth_safety_worker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _two_segment_track() -> Px4Track:
    return Px4Track.model_validate(
        {
            "coordinate_contract": {
                "model_root_world_enu_m": [0.0, 0.0, 0.0],
                "collision_center_offset_model_m": [0.0, 0.0, 0.3],
            },
            "points": [
                {"x": 0.0, "y": 0.0, "z": 1.0, "phase": "launch", "speed_limit_mps": 0.2},
                {"x": 1.0, "y": 0.0, "z": 1.0, "phase": "transit", "speed_limit_mps": 0.2},
                {"x": 2.0, "y": 0.0, "z": 1.0, "phase": "land", "speed_limit_mps": 0.2},
            ],
            "source_world_points": [
                {"east_m": 0.0, "north_m": 0.0, "up_m": 1.0},
                {"east_m": 0.0, "north_m": 1.0, "up_m": 1.0},
                {"east_m": 0.0, "north_m": 2.0, "up_m": 1.0},
            ],
            "waypoint_hold_seconds": 0.2,
        }
    )


def test_px4_world_transform_uses_complete_vehicle_collision_offset() -> None:
    executor = _load_executor()
    coordinate = Px4CoordinateContract(
        model_root_world_enu_m=[10.0, 20.0, 30.0],
        collision_center_offset_model_m=[0.4, -0.2, 0.5],
    )
    setpoint = SimpleNamespace(north_m=2.0, east_m=3.0, down_m=-4.0)

    world = executor._setpoint_world_enu(setpoint, coordinate)

    assert world == Vector3(x=13.4, y=21.8, z=34.5)
    base = SimpleNamespace(
        Setpoint=lambda **values: SimpleNamespace(**values),
    )
    restored = executor._world_enu_setpoint(
        base=base,
        world=world,
        yaw_deg=17.0,
        coordinate_contract=coordinate,
    )
    assert restored.north_m == pytest.approx(2.0)
    assert restored.east_m == pytest.approx(3.0)
    assert restored.down_m == pytest.approx(-4.0)
    assert restored.yaw_deg == pytest.approx(17.0)


# 功能：
#   未发送候选和迟到回执不推进偏航；同命令的及时纯速度接收才提交有界积分。
# 输入：
#   无。
# 输出：
#   无。
def test_model_body_yaw_rate_is_integrated_and_bounded_at_control_rate() -> None:
    executor = _load_executor()
    base = SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values))
    args = SimpleNamespace(setpoint_rate_hz=20.0, maximum_yaw_rate_deg_s=20.0)
    intent = BodyFrameControlIntent(
        source_expert="local-navigation-policy",
        model_call_id="model-" + "a" * 24,
        navigation_snapshot_sha256="b" * 64,
        generated_at_unix_ms=1_000,
        valid_until_unix_ms=1_500,
        forward_velocity_mps=0.5,
        right_velocity_mps=0.0,
        up_velocity_mps=0.0,
        yaw_rate_dps=40.0,
        task_reference_sha256="d" * 64,
        maximum_acceleration_mps2=1.0,
        maximum_jerk_mps3=5.0,
    )
    command = RuntimeLocalSafetyCommand.model_validate(
        {
            "observation_sha256": "c" * 64,
            "observation_sequence": 1,
            "generated_at_unix_ms": 1_010,
            "valid_until_unix_ms": 1_500,
            "source": "onboard",
            "evaluated_target_position_m": {"x": 1.0, "y": 0.0, "z": 1.0},
            "navigation_goal_id": "goal-1",
            "navigation_control_authority": "model-required",
            "model_navigation_authorized": True,
            "model_call_id": intent.model_call_id,
            "model_selected_candidate_id": "candidate-1",
            "model_path_sha256": intent.task_reference_sha256,
            "model_navigation_snapshot_sha256": intent.navigation_snapshot_sha256,
            "model_authority_reason": "model-path-lease-active",
            "requested_control_intent": intent,
            "command_position_m": {"x": 0.1, "y": 0.0, "z": 1.0},
            "decision": {
                "action": "continue",
                "selected_velocity_mps": {"x": 0.5, "y": 0.0, "z": 0.0},
                "selected_yaw_rate_dps": 40.0,
                "maximum_acceleration_mps2": 1.0,
                "maximum_jerk_mps3": 5.0,
                "control_source": "local-model-body-control",
                "predicted_path_m": [{"x": 0.1, "y": 0.0, "z": 1.0}],
                "minimum_predicted_clearance_m": 2.0,
                "time_to_minimum_clearance_seconds": 0.2,
                "evaluated_candidate_count": 1,
            },
        }
    )
    setpoint = SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)

    first = executor._setpoint_with_model_body_yaw(
        base=base,
        args=args,
        setpoint=setpoint,
        command=command,
    )
    second = executor._setpoint_with_model_body_yaw(
        base=base,
        args=args,
        setpoint=setpoint,
        command=command,
    )

    assert first.yaw_deg == pytest.approx(1.0)
    assert second.yaw_deg == pytest.approx(1.0)
    assert getattr(args, "_model_body_control_yaw_deg", None) is None
    assert not executor._record_model_control_application(
        args, command, velocity_ned_mps=(0., .5, 0.), yaw_deg=second.yaw_deg,
        accepted_at_unix_ms=1501,
    )
    assert getattr(args, "_model_body_control_yaw_deg", None) is None
    executor._record_model_control_application(
        args, command, velocity_ned_mps=(0., .5, 0.), yaw_deg=second.yaw_deg,
        accepted_at_unix_ms=1100,
    )
    third = executor._setpoint_with_model_body_yaw(
        base=base, args=args, setpoint=setpoint, command=command,
    )
    assert third.yaw_deg == pytest.approx(2.0)
    assert args._model_body_control_yaw_deg == pytest.approx(1.0)


def test_local_safety_target_uses_stable_content_identity_without_route_context(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    target_path = tmp_path / "local-safety-target.json"
    setpoint = SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-0.7, yaw_deg=0.0)
    coordinate_contract = _two_segment_track().coordinate_contract

    executor._publish_local_safety_target(
        path=target_path,
        setpoint=setpoint,
        coordinate_contract=coordinate_contract,
    )
    first = json.loads(target_path.read_text(encoding="utf-8"))
    executor._publish_local_safety_target(
        path=target_path,
        setpoint=setpoint,
        coordinate_contract=coordinate_contract,
    )
    second = json.loads(target_path.read_text(encoding="utf-8"))

    assert first["navigation_goal_id"].startswith("fixed-target-")
    assert first["navigation_goal_id"] == second["navigation_goal_id"]
    assert first["navigation_goal_position_m"] == first["target_position_m"]


@pytest.mark.parametrize("mode", ["model-rate", "route-heading-assist"])
def test_zero_model_yaw_hands_over_only_with_explicit_assistance(mode: str) -> None:
    executor = _load_executor()
    base = SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values))
    args = SimpleNamespace(
        setpoint_rate_hz=20.0,
        maximum_yaw_rate_deg_s=20.0,
        _model_body_control_yaw_deg=77.0,
    )
    intent = BodyFrameControlIntent(
        source_expert="local-navigation-policy",
        model_call_id="model-" + "a" * 24,
        navigation_snapshot_sha256="b" * 64,
        generated_at_unix_ms=1_000,
        valid_until_unix_ms=1_500,
        forward_velocity_mps=0.5,
        right_velocity_mps=0.0,
        up_velocity_mps=0.0,
        yaw_rate_dps=0.0,
        yaw_control_mode=mode,
        task_reference_sha256="d" * 64,
        maximum_acceleration_mps2=1.0,
        maximum_jerk_mps3=5.0,
    )
    command = RuntimeLocalSafetyCommand.model_validate(
        {
            "observation_sha256": "c" * 64,
            "observation_sequence": 1,
            "generated_at_unix_ms": 1_010,
            "valid_until_unix_ms": 1_500,
            "source": "onboard",
            "evaluated_target_position_m": {"x": 1.0, "y": 0.0, "z": 1.0},
            "navigation_goal_id": "goal-1",
            "navigation_control_authority": "model-required",
            "model_navigation_authorized": True,
            "model_call_id": intent.model_call_id,
            "model_path_sha256": intent.task_reference_sha256,
            "model_navigation_snapshot_sha256": intent.navigation_snapshot_sha256,
            "model_authority_reason": "model-path-lease-active",
            "requested_control_intent": intent,
            "command_position_m": {"x": 0.1, "y": 0.0, "z": 1.0},
            "decision": {
                "action": "continue",
                "selected_velocity_mps": {"x": 0.5, "y": 0.0, "z": 0.0},
                "selected_yaw_rate_dps": 0.0,
                "maximum_acceleration_mps2": 1.0,
                "maximum_jerk_mps3": 5.0,
                "control_source": "local-model-body-control",
                "predicted_path_m": [{"x": 0.1, "y": 0.0, "z": 1.0}],
                "minimum_predicted_clearance_m": 2.0,
                "time_to_minimum_clearance_seconds": 0.2,
                "evaluated_candidate_count": 1,
            },
        }
    )
    route_setpoint = SimpleNamespace(
        north_m=0.0,
        east_m=0.0,
        down_m=-1.0,
        yaw_deg=-35.0,
    )

    observed = executor._setpoint_with_model_body_yaw(
        base=base,
        args=args,
        setpoint=route_setpoint,
        command=command,
    )

    if mode == "route-heading-assist":
        assert observed is route_setpoint
        assert args._model_body_control_yaw_deg is None
    else:
        assert observed.yaw_deg == pytest.approx(77.0)
        assert args._model_body_control_yaw_deg == pytest.approx(77.0)


def test_tracking_segment_policy_is_hash_bound_and_context_selected(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    track = _two_segment_track()
    policy_path = tmp_path / "tracking-corridor-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.tracking-corridor-policy.v1",
                "track_sha256": sha256_json(track),
                "segment_policies": [
                    {
                        "segment_index": 0,
                        "tracking_lag_limit_m": 0.05,
                        "tracking_rejoin_tolerance_m": 0.04,
                        "control_profile": "cruise",
                    },
                    {
                        "segment_index": 1,
                        "tracking_lag_limit_m": 0.22,
                        "tracking_rejoin_tolerance_m": 0.18,
                        "control_profile": "precision",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    policies = executor._load_tracking_segment_policies(policy_path, track=track)
    args = SimpleNamespace(
        tracking_lag_limit_m=0.03,
        tracking_rejoin_tolerance_m=0.02,
        _tracking_segment_policies=policies,
    )

    assert executor._tracking_limits_for_context(
        args,
        {"navigation_goal_id": "source-waypoint-0002"},
    ) == (0.22, 0.18, 1)
    assert (
        executor._tracking_control_profile_for_context(
            args,
            {"navigation_goal_id": "source-waypoint-0002"},
        )
        == "precision"
    )
    assert executor._tracking_limits_for_context(
        args,
        {
            "navigation_goal_id": "source-waypoint-0002",
            "replacement_sequence": 1,
        },
    ) == (0.03, 0.02, None)
    assert (
        executor._tracking_control_profile_for_context(
            args,
            {
                "navigation_goal_id": "source-waypoint-0002",
                "replacement_sequence": 1,
            },
        )
        == "cruise"
    )

    tampered_track = track.model_copy(
        update={"waypoint_hold_seconds": track.waypoint_hold_seconds + 0.1}
    )
    with pytest.raises(ValueError, match="not bound"):
        executor._load_tracking_segment_policies(policy_path, track=tampered_track)


# 功能：验证局部安全目标只取得当前原路线段的实测净空，不借用其他段或固定目标的预算。
# 输入：无：使用两段不同净空的冻结策略。
# 输出：None：返回当前段身份与净空，非路线目标不伪造预算。
def test_tracking_localization_context_selects_current_measured_corridor() -> None:
    executor = _load_executor()
    args = SimpleNamespace(
        _tracking_segment_policies=(
            {"minimum_route_clearance_m": 0.76},
            {"minimum_route_clearance_m": 0.49},
        )
    )
    assert executor._tracking_localization_context_for_goal(
        args, "source-waypoint-0002"
    ) == (1, pytest.approx(0.49))
    assert executor._tracking_localization_context_for_goal(args, "takeoff") == (None, None)
    with pytest.raises(ValueError, match="MEASURED_CLEARANCE_REQUIRED"):
        executor._tracking_localization_context_for_goal(
            SimpleNamespace(_tracking_segment_policies=({},)),
            "source-waypoint-0001",
        )


def test_tracking_segment_policy_rejects_missing_current_control_profile(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    track = _two_segment_track()
    policy_path = tmp_path / "tracking-corridor-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "dronedream.tracking-corridor-policy.v1",
                "track_sha256": sha256_json(track),
                "segment_policies": [
                    {
                        "segment_index": index,
                        "tracking_lag_limit_m": 0.22,
                        "tracking_rejoin_tolerance_m": 0.18,
                    }
                    for index in range(len(track.points) - 1)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="control profile is invalid"):
        executor._load_tracking_segment_policies(policy_path, track=track)


def test_spawn_relative_client_rebases_combined_position_velocity_command() -> None:
    executor = _load_executor()

    class Client:
        def __init__(self) -> None:
            self.position: Any = None
            self.velocity: Any = None

        async def set_position_velocity_ned(self, position: Any, velocity: Any) -> None:
            self.position = position
            self.velocity = velocity

    async def exercise() -> Client:
        client = Client()
        wrapped = executor.SpawnRelativeOffboardClient(
            client,
            SimpleNamespace(north_m=10.0, east_m=-4.0, down_m=2.0),
            heading_hold_deg=78.0,
        )
        await wrapped.set_position_velocity_ned(
            SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-3.0, yaw_deg=5.0),
            SimpleNamespace(
                north_m_s=0.1,
                east_m_s=0.2,
                down_m_s=-0.3,
                yaw_deg=5.0,
            ),
        )
        return client

    client = asyncio.run(exercise())
    assert vars(client.position) == {
        "north_m": 11.0,
        "east_m": -2.0,
        "down_m": -1.0,
        "yaw_deg": 78.0,
    }
    assert vars(client.velocity) == {
        "north_m_s": 0.1,
        "east_m_s": 0.2,
        "down_m_s": -0.3,
        "yaw_deg": 78.0,
    }


def test_spawn_relative_client_forwards_velocity_without_origin_translation() -> None:
    executor = _load_executor()

    class Client:
        def __init__(self) -> None:
            self.velocity: Any = None

        async def set_velocity_ned(self, velocity: Any) -> None:
            self.velocity = velocity

    async def exercise() -> Client:
        client = Client()
        wrapped = executor.SpawnRelativeOffboardClient(
            client,
            SimpleNamespace(north_m=10.0, east_m=-4.0, down_m=2.0),
            heading_hold_deg=78.0,
        )
        await wrapped.set_velocity_ned(
            SimpleNamespace(
                north_m_s=0.1,
                east_m_s=0.2,
                down_m_s=-0.3,
                yaw_deg=5.0,
            )
        )
        return client

    client = asyncio.run(exercise())
    assert vars(client.velocity) == {
        "north_m_s": 0.1,
        "east_m_s": 0.2,
        "down_m_s": -0.3,
        "yaw_deg": 78.0,
    }


def test_local_control_hold_preserves_bounded_executor_phase(tmp_path: Path) -> None:
    executor = _load_executor()
    phase_path = tmp_path / "runtime-phase.json"
    phase_path.write_text(
        json.dumps(
            {
                "phase": "ACTION",
                "checkpoint_id": "checkpoint-001",
                "trigger": "checkpoint",
            }
        ),
        encoding="utf-8",
    )

    executor._publish_local_control_phase(
        phase_path,
        local_phase="PERCEPTION_REFRESH_HOLD",
        details={
            "local_safety_action": "hold",
            "schedule_advancement_authorized": False,
        },
    )

    published = json.loads(phase_path.read_text(encoding="utf-8"))
    assert published["phase"] == "ACTION"
    assert published["checkpoint_id"] == "checkpoint-001"
    assert published["local_control_phase"] == "PERCEPTION_REFRESH_HOLD"
    assert published["schedule_advancement_authorized"] is False

    executor._clear_local_control_phase(phase_path)
    restored = json.loads(phase_path.read_text(encoding="utf-8"))
    assert restored == {
        "phase": "ACTION",
        "checkpoint_id": "checkpoint-001",
        "trigger": "checkpoint",
    }


def test_local_control_hold_restores_original_tracking_state_after_repeated_updates(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    phase_path = tmp_path / "runtime-phase.json"
    original = {
        "phase": "TRACK",
        "checkpoint_id": None,
        "schedule_index": 41,
    }
    phase_path.write_text(json.dumps(original), encoding="utf-8")

    executor._publish_local_control_phase(
        phase_path,
        local_phase="MODEL_AUTHORITY_HOLD",
        details={
            "model_authority_reason": "model-lease-expired",
            "schedule_advancement_authorized": False,
        },
    )
    executor._publish_local_control_phase(
        phase_path,
        local_phase="MODEL_AUTHORITY_HOLD",
        details={
            "model_authority_reason": "model-lease-not-established",
            "schedule_advancement_authorized": False,
        },
    )
    held = json.loads(phase_path.read_text(encoding="utf-8"))
    assert held["phase"] == "MODEL_AUTHORITY_HOLD"
    assert held["local_control_phase"] == "MODEL_AUTHORITY_HOLD"
    assert held["enclosing_executor_state"] == original

    executor._clear_local_control_phase(phase_path)
    assert json.loads(phase_path.read_text(encoding="utf-8")) == original


def test_tracking_recovery_window_refreshes_only_for_material_progress() -> None:
    executor = _load_executor()

    state, expired = executor._advance_tracking_recovery_window(
        now=100.0,
        timeout_seconds=20.0,
        state=None,
        tracking_error_m=0.20,
        model_goal_distance_m=2.0,
    )
    assert expired is False

    state, expired = executor._advance_tracking_recovery_window(
        now=119.0,
        timeout_seconds=20.0,
        state=state,
        tracking_error_m=0.20,
        model_goal_distance_m=1.98,
    )
    assert expired is False
    assert state["stall_deadline_monotonic"] == pytest.approx(139.0)
    assert state["progress_revision"] == 1
    assert state["last_progress_evidence"] == "semantic_goal_distance_decreased"

    state, expired = executor._advance_tracking_recovery_window(
        now=139.0,
        timeout_seconds=20.0,
        state=state,
        tracking_error_m=0.20,
        model_goal_distance_m=1.975,
    )
    assert expired is True
    assert state["progress_revision"] == 1


def test_semantic_progress_window_requests_recovery_before_bounded_abort() -> None:
    executor = _load_executor()
    state, expired = executor._advance_model_semantic_progress_window(
        now=0.0,
        recovery_after_seconds=20.0,
        abort_after_seconds=60.0,
        state=None,
        navigation_goal_id="goal-a",
        model_goal_distance_m=3.0,
    )
    assert expired is False
    assert state["recovery_requested"] is False

    state, expired = executor._advance_model_semantic_progress_window(
        now=20.0,
        recovery_after_seconds=20.0,
        abort_after_seconds=60.0,
        state=state,
        navigation_goal_id="goal-a",
        model_goal_distance_m=2.98,
    )
    assert expired is False
    assert state["recovery_requested"] is True
    first_recovery_episode_id = state["recovery_episode_id"]
    assert str(first_recovery_episode_id).startswith("recovery-")
    assert state["recovery_request_count"] == 1

    state, expired = executor._advance_model_semantic_progress_window(
        now=30.0,
        recovery_after_seconds=20.0,
        abort_after_seconds=60.0,
        state=state,
        navigation_goal_id="goal-a",
        model_goal_distance_m=2.90,
    )
    assert expired is False
    assert state["recovery_requested"] is False
    assert state["recovery_episode_id"] is None
    assert state["progress_revision"] == 1

    state, expired = executor._advance_model_semantic_progress_window(
        now=90.0,
        recovery_after_seconds=20.0,
        abort_after_seconds=60.0,
        state=state,
        navigation_goal_id="goal-a",
        model_goal_distance_m=2.89,
    )
    assert expired is True
    assert state["recovery_requested"] is True
    assert state["recovery_request_count"] == 2
    assert state["recovery_episode_id"] != first_recovery_episode_id

    reset, expired = executor._advance_model_semantic_progress_window(
        now=90.0,
        recovery_after_seconds=20.0,
        abort_after_seconds=60.0,
        state=state,
        navigation_goal_id="goal-b",
        model_goal_distance_m=4.0,
    )
    assert expired is False
    assert reset["recovery_requested"] is False
    assert reset["recovery_episode_id"] is None
    assert reset["recovery_request_count"] == 0
    assert reset["navigation_goal_id"] == "goal-b"


def test_authorized_schedule_advance_keeps_semantic_window_alive() -> None:
    executor = _load_executor()
    state, expired = executor._advance_model_semantic_progress_window(
        now=0.0,
        recovery_after_seconds=5.0,
        abort_after_seconds=30.0,
        state=None,
        navigation_goal_id="long-segment-goal",
        model_goal_distance_m=0.22,
    )

    for now in (5.0, 10.0, 20.0, 35.0):
        state, expired = executor._advance_model_semantic_progress_window(
            now=now,
            recovery_after_seconds=5.0,
            abort_after_seconds=30.0,
            state=state,
            navigation_goal_id="long-segment-goal",
            model_goal_distance_m=0.22,
            authorized_schedule_advance=True,
        )
        assert expired is False
        assert state["recovery_requested"] is False

    assert state["authorized_schedule_revision"] == 4
    assert state["progress_revision"] == 4

    state, expired = executor._advance_model_semantic_progress_window(
        now=65.0,
        recovery_after_seconds=5.0,
        abort_after_seconds=30.0,
        state=state,
        navigation_goal_id="long-segment-goal",
        model_goal_distance_m=0.22,
    )
    assert expired is True
    assert state["recovery_requested"] is True


def test_action_checkpoint_enters_precision_profile_before_final_settle() -> None:
    executor = _load_executor()

    assert (
        executor._navigation_control_profile(
            planned_goal_distance_m=3.5,
            action_checkpoint_goal=True,
            semantic_approach_damping_active=False,
        )
        == "precision"
    )
    assert (
        executor._navigation_control_profile(
            planned_goal_distance_m=5.0,
            action_checkpoint_goal=True,
            semantic_approach_damping_active=False,
        )
        == "cruise"
    )
    assert (
        executor._navigation_control_profile(
            planned_goal_distance_m=0.7,
            action_checkpoint_goal=False,
            semantic_approach_damping_active=True,
        )
        == "precision"
    )
    assert (
        executor._navigation_control_profile(
            planned_goal_distance_m=10.0,
            action_checkpoint_goal=False,
            semantic_approach_damping_active=False,
            tight_clearance_segment=True,
        )
        == "precision"
    )


def test_tracking_recovery_window_accepts_controller_convergence() -> None:
    executor = _load_executor()
    state, _ = executor._advance_tracking_recovery_window(
        now=5.0,
        timeout_seconds=20.0,
        state=None,
        tracking_error_m=0.20,
        model_goal_distance_m=None,
    )

    state, expired = executor._advance_tracking_recovery_window(
        now=24.0,
        timeout_seconds=20.0,
        state=state,
        tracking_error_m=0.18,
        model_goal_distance_m=None,
    )

    assert expired is False
    assert state["stall_deadline_monotonic"] == pytest.approx(44.0)
    assert state["last_progress_evidence"] == "controller_error_decreased"


def test_tracking_recovery_progress_scales_to_narrow_rejoin_corridor() -> None:
    executor = _load_executor()
    progress_epsilon_m = 0.04752 * 0.10
    state, _ = executor._advance_tracking_recovery_window(
        now=0.0,
        timeout_seconds=20.0,
        state=None,
        tracking_error_m=0.05715,
        model_goal_distance_m=None,
        progress_epsilon_m=progress_epsilon_m,
    )

    state, expired = executor._advance_tracking_recovery_window(
        now=20.0,
        timeout_seconds=20.0,
        state=state,
        tracking_error_m=0.04803,
        model_goal_distance_m=None,
        progress_epsilon_m=progress_epsilon_m,
    )

    assert expired is False
    assert state["progress_revision"] == 1
    assert state["stall_deadline_monotonic"] == pytest.approx(40.0)
    assert state["last_progress_evidence"] == "controller_error_decreased"


def test_tracking_recovery_counts_convergence_from_a_recent_error_peak() -> None:
    executor = _load_executor()
    state, _ = executor._advance_tracking_recovery_window(
        now=0.0,
        timeout_seconds=20.0,
        state=None,
        tracking_error_m=0.048,
        model_goal_distance_m=None,
        progress_epsilon_m=0.004,
    )

    state, expired = executor._advance_tracking_recovery_window(
        now=19.0,
        timeout_seconds=20.0,
        state=state,
        tracking_error_m=0.060,
        model_goal_distance_m=None,
        progress_epsilon_m=0.004,
    )
    assert expired is False
    assert state["progress_revision"] == 0

    state, expired = executor._advance_tracking_recovery_window(
        now=19.5,
        timeout_seconds=20.0,
        state=state,
        tracking_error_m=0.054,
        model_goal_distance_m=None,
        progress_epsilon_m=0.004,
    )

    assert expired is False
    assert state["best_tracking_error_m"] == pytest.approx(0.048)
    assert state["tracking_error_progress_anchor_m"] == pytest.approx(0.054)
    assert state["stall_deadline_monotonic"] == pytest.approx(39.5)
    assert state["absolute_deadline_monotonic"] == pytest.approx(60.0)
    assert state["progress_revision"] == 1
    assert state["last_progress_evidence"] == "controller_error_decreased"


def test_tracking_recovery_window_never_extends_absolute_deadline() -> None:
    executor = _load_executor()
    state, _ = executor._advance_tracking_recovery_window(
        now=0.0,
        timeout_seconds=20.0,
        state=None,
        tracking_error_m=0.30,
        model_goal_distance_m=3.0,
    )

    for now, distance in ((19.0, 2.9), (38.0, 2.8), (57.0, 2.7)):
        state, expired = executor._advance_tracking_recovery_window(
            now=now,
            timeout_seconds=20.0,
            state=state,
            tracking_error_m=0.30,
            model_goal_distance_m=distance,
        )
        assert expired is False
    assert state["stall_deadline_monotonic"] == pytest.approx(60.0)

    _, expired = executor._advance_tracking_recovery_window(
        now=60.0,
        timeout_seconds=20.0,
        state=state,
        tracking_error_m=0.30,
        model_goal_distance_m=2.6,
    )
    assert expired is True


# 功能：
#   验证恢复超时回执读取字典字段，避免飞行中把字典误当对象访问而触发异常降落。
# 输入：
#   无：使用与恢复窗口相同形状的最小状态。
# 输出：
#   None：不返回业务数据。
def test_tracking_recovery_rollover_details_accept_dictionary_state() -> None:
    executor = _load_executor()

    details = executor._tracking_recovery_rollover_details(
        {
            "started_at_monotonic": 10.0,
            "best_tracking_error_m": 0.125,
        },
        now=12.5,
    )

    assert details == {
        "elapsed_seconds": pytest.approx(2.5),
        "best_tracking_error_m": pytest.approx(0.125),
    }


def test_atomic_json_retries_a_transient_windows_reader_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _load_executor()
    destination = tmp_path / "runtime-phase.json"
    original_replace = Path.replace
    attempts = 0

    def transiently_locked(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated Windows reader lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", transiently_locked)
    executor._atomic_json(
        destination,
        {"phase": "checkpoint"},
        replace_timeout_seconds=0.1,
        replace_retry_seconds=0.0,
    )

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {"phase": "checkpoint"}


def test_depth_worker_atomic_json_retries_a_transient_drvfs_reader_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _load_depth_worker()
    destination = tmp_path / "depth-local-safety-command.json"
    original_replace = worker.os.replace
    attempts = 0

    def transiently_locked(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated DrvFS reader lock")
        original_replace(source, target)

    monkeypatch.setattr(worker.os, "replace", transiently_locked)
    worker._atomic_json(
        destination,
        {"sequence": 42},
        replace_timeout_seconds=0.1,
        replace_retry_seconds=0.0,
    )

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {"sequence": 42}


def test_local_safety_slowdown_advances_after_one_safe_controller_tick(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        decision=SimpleNamespace(
            action="slow",
            threat_obstacle_id="person-crossing",
            minimum_predicted_clearance_m=0.42,
            selected_velocity_mps=Vector3(x=0.2, y=0.1, z=0.0),
        ),
        command_position_m=Vector3(x=0.1, y=0.2, z=1.5),
        observation_sequence=9,
    )
    executor._read_local_safety_command = lambda _args: command

    async def ignore_identity_refresh(**_kwargs: Any) -> None:
        return None

    executor._refresh_px4_identity_telemetry = ignore_identity_refresh

    class Client:
        commands: list[Any]

        def __init__(self) -> None:
            self.commands = []
            self.velocities: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

        async def set_position_velocity_ned(self, setpoint: Any, velocity: Any) -> None:
            self.commands.append(setpoint)
            self.velocities.append(velocity)

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=tmp_path / "command.json",
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=100.0,
            ),
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=0.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
        return client

    client = asyncio.run(exercise())
    assert len(client.commands) == 1
    assert vars(client.velocities[0]) == {
        "north_m_s": 0.1,
        "east_m_s": 0.2,
        "down_m_s": -0.0,
        "yaw_deg": 0.0,
    }
    assert (
        json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))["local_safety_action"]
        == "slow"
    )


def test_local_safety_recovery_window_rolls_over_without_terminal_landing(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        decision=SimpleNamespace(
            action="hold",
            threat_obstacle_id="temporary-depth-track",
            minimum_predicted_clearance_m=0.05,
            selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
            issue_codes=["HOLD_AND_REQUEST_REPLAN"],
        ),
        command_position_m=Vector3(x=0.1, y=0.2, z=1.5),
        observation_sequence=12,
    )
    executor._read_local_safety_command = lambda _args: command

    async def ignore_identity_refresh(**_kwargs: Any) -> None:
        return None

    executor._refresh_px4_identity_telemetry = ignore_identity_refresh

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

    async def exercise() -> tuple[Client, Any]:
        client = Client()
        applied = await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=tmp_path / "command.json",
                local_safety_repair_timeout_seconds=15.0,
                local_safety_repair_absolute_timeout_seconds=1e-9,
                setpoint_rate_hz=1000.0,
            ),
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=0.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
        return client, applied

    client, applied = asyncio.run(exercise())
    assert 1 <= len(client.commands) <= 2
    assert applied is client.commands[-1]
    phase = json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))
    assert phase["local_control_phase"] == "RECOVERY_RETRY"
    assert phase["recovery_rollover_reason"] == "absolute-window-elapsed"


def test_static_clearance_recovery_freezes_schedule_until_margin_restored(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    commands = iter(
        (
            _safety_command_fixture(
                decision=SimpleNamespace(
                    action="slow",
                    threat_obstacle_id="static:door-leaf",
                    minimum_predicted_clearance_m=0.01,
                    selected_velocity_mps=Vector3(x=0.0, y=0.05, z=0.02),
                    issue_codes=["STATIC_CLEARANCE_RECOVERY"],
                ),
                command_position_m=Vector3(x=0.0, y=0.05, z=1.02),
                estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
                observation_sequence=1,
            ),
            _safety_command_fixture(
                decision=SimpleNamespace(action="continue"),
                estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
            ),
        )
    )
    executor._read_local_safety_command = lambda _args: next(commands)
    observations = iter(
        (
            SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-0.8),
            SimpleNamespace(north_m=0.03, east_m=0.0, down_m=-0.81),
        )
    )

    async def observed_identity(**_kwargs: Any) -> Any:
        return next(observations)

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_velocity_ned(self, setpoint: Any, _velocity: Any) -> None:
            self.commands.append(setpoint)

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=tmp_path / "target.json",
                local_safety_command=tmp_path / "command.json",
                local_safety_repair_timeout_seconds=15.0,
                local_safety_repair_absolute_timeout_seconds=60.0,
                setpoint_rate_hz=1_000.0,
            ),
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=1.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
        return client

    client = asyncio.run(exercise())
    assert len(client.commands) == 2
    assert client.commands[0].north_m == pytest.approx(0.05)
    assert client.commands[1].north_m == pytest.approx(1.0)
    target = json.loads((tmp_path / "target.json").read_text(encoding="utf-8"))
    assert target["target_position_m"] == {"x": 0.0, "y": 1.0, "z": 1.2}


def test_local_safety_continue_preserves_controller_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _load_executor()
    sleeps: list[float] = []
    clock = [10.0]

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    pacer = executor.ControlTickPacer(20., clock=lambda: clock[0], sleep=record_sleep)

    async def ignore_identity_refresh(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(executor, "_refresh_px4_identity_telemetry", ignore_identity_refresh)
    executor._read_local_safety_command = lambda _args: _safety_command_fixture(
        decision=SimpleNamespace(
            action="continue",
            threat_obstacle_id=None,
            minimum_predicted_clearance_m=2.0,
        ),
        command_position_m=Vector3(x=1.0, y=0.0, z=1.0),
        estimator_to_world_position_offset_m=Vector3(x=0.2, y=-0.1, z=0.05),
        observation_sequence=10,
    )

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []
            self.velocities: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

        async def set_position_velocity_ned(self, setpoint: Any, velocity: Any) -> None:
            self.commands.append(setpoint)
            self.velocities.append(velocity)

    async def exercise() -> Client:
        client = Client()
        kwargs = dict(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=tmp_path / "command.json",
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=20.0,
                _local_control_pacer=pacer,
            ),
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=0.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(0.6, -0.2, 0.1),
        )
        await executor._apply_local_safety(**kwargs)
        assert not sleeps  # The first command need not wait for an empty tick.
        clock[0] += .020  # Work in the caller must count toward the same period.
        await executor._apply_local_safety(**kwargs)
        return client

    client = asyncio.run(exercise())
    assert len(client.commands) == 2
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(.030)
    assert client.commands[0].east_m == pytest.approx(-0.2)
    assert client.commands[0].north_m == pytest.approx(0.1)
    assert client.commands[0].down_m == pytest.approx(-0.95)
    assert vars(client.velocities[0]) == {
        "north_m_s": 0.6,
        "east_m_s": -0.2,
        "down_m_s": 0.1,
        "yaw_deg": 0.0,
    }


# 功能：
#   验证固定目标恢复只使用有界径向速度与实测速度阻尼，不沿用路线切向前馈。
# 输入：
#   tmp_path：本次控制目标与证据目录。
# 输出：
#   None：不返回业务数据。
def test_tracking_recovery_settles_fixed_target_with_bounded_radial_feedforward(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    executor._read_local_safety_command = lambda _args: _safety_command_fixture(
        decision=SimpleNamespace(
            action="continue",
            selected_velocity_mps=Vector3(x=0.25, y=-0.4, z=0.15),
        ),
        estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
        tracking_recovery_active=True,
        evaluated_target_position_m=Vector3(x=2.0, y=1.0, z=1.2),
    )

    # 功能：
    #   提供明确位置及零速度的原生遥测替身，不再把缺失速度隐式当作静止。
    # 输入：
    #   _kwargs：执行器要求的遥测刷新上下文。
    # 输出：
    #   observed：位置偏离恢复点、速度为零的完整状态。
    async def observed_identity(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            north_m=0.9,
            east_m=1.9,
            down_m=-0.95,
            north_m_s=0.0,
            east_m_s=0.0,
            down_m_s=0.0,
        )

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.velocity = None

        async def set_position_velocity_ned(self, _setpoint: Any, velocity: Any) -> None:
            self.velocity = velocity

        async def set_position_ned(self, _setpoint: Any) -> None:
            raise AssertionError("recovery should use position+velocity")

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=tmp_path / "target.json",
                local_safety_command=tmp_path / "command.json",
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=1_000.0,
            ),
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=1.0,
                east_m=2.0,
                down_m=-1.0,
                yaw_deg=30.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(0.8, 0.0, 0.0),
            tracking_recovery_active=True,
        )
        return client

    client = asyncio.run(exercise())
    assert vars(client.velocity) == {
        "north_m_s": pytest.approx(0.053333333333333344),
        "east_m_s": pytest.approx(0.05333333333333339),
        "down_m_s": pytest.approx(-0.026666666666666696),
        "yaw_deg": 30.0,
    }
    assert math.sqrt(
        client.velocity.north_m_s**2 + client.velocity.east_m_s**2 + client.velocity.down_m_s**2
    ) == pytest.approx(0.08)
    target = json.loads((tmp_path / "target.json").read_text(encoding="utf-8"))
    assert target["tracking_recovery_active"] is True


def test_ordinary_safety_lease_is_compatible_only_when_entering_slower_recovery() -> None:
    executor = _load_executor()
    contract = Px4CoordinateContract(
        model_root_world_enu_m=[10.0, 20.0, 1.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )
    planned = SimpleNamespace(
        north_m=1.0,
        east_m=2.0,
        down_m=-0.5,
        yaw_deg=0.0,
    )
    planned_world = Vector3(x=12.0, y=21.0, z=1.7)

    ordinary = SimpleNamespace(
        tracking_recovery_active=False,
        navigation_control_authority="route-fallback",
        evaluated_target_position_m=planned_world,
    )
    assert executor._local_safety_command_matches_control_context(
        command=ordinary,
        planned_setpoint=planned,
        coordinate_contract=contract,
        tracking_recovery_active=True,
    )

    recovery = SimpleNamespace(
        tracking_recovery_active=True,
        navigation_control_authority="route-fallback",
        evaluated_target_position_m=planned_world,
    )
    assert not executor._local_safety_command_matches_control_context(
        command=recovery,
        planned_setpoint=planned,
        coordinate_contract=contract,
        tracking_recovery_active=False,
    )
    assert not executor._local_safety_command_matches_control_context(
        command=SimpleNamespace(
            tracking_recovery_active=False,
            navigation_control_authority="route-fallback",
            evaluated_target_position_m=Vector3(x=12.03, y=21.0, z=1.7),
        ),
        planned_setpoint=planned,
        coordinate_contract=contract,
        tracking_recovery_active=True,
    )

    compatible = executor._local_safety_control_context_diagnostic(
        command=ordinary,
        planned_setpoint=planned,
        coordinate_contract=contract,
        tracking_recovery_active=True,
    )
    assert compatible["compatible"] is True
    assert compatible["context_reason"] == "same-target-proof-compatible-with-recovery"
    assert compatible["evaluated_target_distance_m"] == pytest.approx(0.0)

    rejected = executor._local_safety_control_context_diagnostic(
        command=recovery,
        planned_setpoint=planned,
        coordinate_contract=contract,
        tracking_recovery_active=False,
    )
    assert rejected == {
        "compatible": False,
        "context_reason": "recovery-proof-cannot-authorize-ordinary-tracking",
        "command_tracking_recovery_active": True,
        "executor_tracking_recovery_active": False,
        "evaluated_target_distance_m": None,
    }


def test_stationary_route_hold_completes_only_the_already_reached_setpoint() -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        decision=SimpleNamespace(
            action="hold",
            selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        ),
        navigation_control_authority="route-fallback",
        evaluated_target_position_m=Vector3(x=6.35198, y=11.64242, z=1.4),
        command_position_m=Vector3(x=6.31237, y=11.60313, z=1.41301),
    )
    observed = SimpleNamespace(
        north_m_s=0.001,
        east_m_s=0.012,
        down_m_s=-0.013,
    )

    assert executor._stationary_route_setpoint_is_complete(
        command=command,
        observed=observed,
        tracking_recovery_active=False,
    )

    command.command_position_m = Vector3(x=6.1, y=11.4, z=1.4)
    assert not executor._stationary_route_setpoint_is_complete(
        command=command,
        observed=observed,
        tracking_recovery_active=False,
    )


def test_stationary_route_hold_never_bypasses_recovery_or_model_authority() -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        decision=SimpleNamespace(
            action="hold",
            selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        ),
        navigation_control_authority="route-fallback",
        evaluated_target_position_m=Vector3(x=1.0, y=2.0, z=1.0),
        command_position_m=Vector3(x=1.0, y=2.0, z=1.0),
    )
    stopped = SimpleNamespace(north_m_s=0.0, east_m_s=0.0, down_m_s=0.0)

    assert not executor._stationary_route_setpoint_is_complete(
        command=command,
        observed=stopped,
        tracking_recovery_active=True,
    )
    command.navigation_control_authority = "model-required"
    assert not executor._stationary_route_setpoint_is_complete(
        command=command,
        observed=stopped,
        tracking_recovery_active=False,
    )


def test_stationary_route_hold_requires_low_speed_and_finite_complete_state() -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        decision=SimpleNamespace(
            action="hold",
            selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        ),
        navigation_control_authority="route-fallback",
        evaluated_target_position_m=Vector3(x=1.0, y=2.0, z=1.0),
        command_position_m=Vector3(x=1.0, y=2.0, z=1.0),
    )

    assert not executor._stationary_route_setpoint_is_complete(
        command=command,
        observed=SimpleNamespace(north_m_s=0.06, east_m_s=0.0, down_m_s=0.0),
        tracking_recovery_active=False,
    )
    assert not executor._stationary_route_setpoint_is_complete(
        command=command,
        observed=SimpleNamespace(
            north_m_s=float("nan"), east_m_s=0.0, down_m_s=0.0
        ),
        tracking_recovery_active=False,
    )


# 功能：短时低帧率空窗可在签名静态走廊内桥接，不能把其他异常降级为运动许可。
# 输入：纯感知时效保持命令、健康帧时刻、PX4 状态和路线净空；输出：有界窗口内允许桥接。
def test_static_corridor_perception_bridge_accepts_only_recent_stale_camera_gap() -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        navigation_control_authority="route-fallback",
        source="onboard",
        decision=SimpleNamespace(
            action="hold",
            issue_codes=[
                "PERCEPTION_STREAM_UNHEALTHY",
                "PERCEPTION_STREAM_STALE",
                "HOLD_UNTIL_FRESH_LOCAL_WORLD",
            ],
        ),
    )
    args = SimpleNamespace(
        independent_route_control=True,
        _last_fresh_local_safety_monotonic=10.0,
    )
    observed = SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0)

    assert executor._bounded_static_corridor_perception_bridge_allowed(
        args=args,
        command=command,
        observed=observed,
        tracking_recovery_active=True,
        action_checkpoint_goal=False,
        qualified_route_clearance_m=0.395,
        now=11.2,
    )

    for overrides in (
        {"now": 14.01},
        {"action_checkpoint_goal": True},
        {"qualified_route_clearance_m": None},
        {"observed": None},
    ):
        values = {
            "args": args,
            "command": command,
            "observed": observed,
            "tracking_recovery_active": True,
            "action_checkpoint_goal": False,
            "qualified_route_clearance_m": 0.395,
            "now": 11.2,
        }
        values.update(overrides)
        assert not executor._bounded_static_corridor_perception_bridge_allowed(**values)

    # 动作检查点在真实飞行中仍拒绝桥接；显式 Gazebo 身份只授权低速接近，
    # 真正载荷动作仍由检查点的新鲜感知确认独立把关。
    args.gazebo_vehicle_model_name = "x500_depth_0"
    assert executor._bounded_static_corridor_perception_bridge_allowed(
        args=args,
        command=command,
        observed=observed,
        tracking_recovery_active=True,
        action_checkpoint_goal=True,
        qualified_route_clearance_m=0.395,
        now=11.2,
    )
    del args.gazebo_vehicle_model_name

    # Ordinary route progress uses the same bounded bridge. It must not first
    # manufacture a tracking-error episode merely because a camera frame was
    # delayed.
    assert executor._bounded_static_corridor_perception_bridge_allowed(
        args=args,
        command=command,
        observed=observed,
        tracking_recovery_active=False,
        action_checkpoint_goal=False,
        qualified_route_clearance_m=0.395,
        now=11.2,
    )


# 功能：已完成碰撞预测但错过发令预算的帧可由当前 PX4 状态和静态几何短暂接续。
# 输入：仅含处理预算耗尽原因的保持指令；输出：短窗允许，其他时效原因拒绝。
def test_static_corridor_bridge_accepts_only_processing_budget_exhaustion() -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        navigation_control_authority="route-fallback",
        source="onboard",
        decision=SimpleNamespace(
            action="hold",
            issue_codes=[
                "OBSERVATION_CONTROL_BUDGET_EXHAUSTED",
                "OBSERVATION_PROCESSING_OR_RESPONSE_BUDGET_EXHAUSTED",
                "CANDIDATE_TIME_BUDGET_EXHAUSTED",
                "NO_SAFE_LOCAL_VELOCITY",
                "HOLD_AND_REQUEST_REPLAN",
            ],
        ),
    )
    values = dict(
        args=SimpleNamespace(
            independent_route_control=True,
            _last_fresh_local_safety_monotonic=10.0,
        ),
        command=command,
        observed=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        tracking_recovery_active=False,
        action_checkpoint_goal=False,
        qualified_route_clearance_m=0.395,
        now=11.2,
    )
    assert executor._bounded_static_corridor_perception_bridge_allowed(**values)
    command.decision.issue_codes[1] = "OBSERVATION_NO_TEMPORAL_CLEARANCE_MARGIN"
    assert not executor._bounded_static_corridor_perception_bridge_allowed(**values)


# 功能：
#   验证异步局部安全结果之间可沿同一签名静态走廊连续控制，而非按毫秒租期反复刹停。
# 输入：
#   最近健康继续裁决、当前 PX4 状态、相同语义目标和路线净空。
# 输出：
    #   短暂计算空窗返回独立副本；真实动作点、目标变化、异常裁决或超时全部拒绝，
    #   显式仿真身份允许低速接近动作点。
def test_static_corridor_control_continuity_uses_only_recent_clean_same_goal_seed() -> None:
    executor = _load_executor()
    seed = _safety_command_fixture(
        navigation_control_authority="route-fallback",
        navigation_goal_id="source-waypoint-0001",
        source="onboard",
        decision=SimpleNamespace(
            action="continue",
            issue_codes=[],
            selected_velocity_mps=Vector3(x=0.3, y=0.0, z=0.0),
        ),
    )
    args = SimpleNamespace(
        independent_route_control=True,
        _last_fresh_local_safety_monotonic=10.0,
        _last_fresh_local_safety_command=seed,
    )
    values = dict(
        args=args,
        observed=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        navigation_goal_id="source-waypoint-0001",
        action_checkpoint_goal=False,
        qualified_route_clearance_m=0.395,
        now=11.0,
    )

    accepted = executor._bounded_static_corridor_control_continuity_seed(**values)
    assert accepted == seed
    assert accepted is not seed

    for overrides in (
        {"now": 14.01},
        {"action_checkpoint_goal": True},
        {"navigation_goal_id": "source-waypoint-0002"},
        {"qualified_route_clearance_m": None},
        {"observed": None},
    ):
        attempt = dict(values)
        attempt.update(overrides)
        assert executor._bounded_static_corridor_control_continuity_seed(**attempt) is None

    args.gazebo_vehicle_model_name = "x500_depth_0"
    checkpoint_attempt = dict(values)
    checkpoint_attempt["action_checkpoint_goal"] = True
    accepted_checkpoint = executor._bounded_static_corridor_control_continuity_seed(
        **checkpoint_attempt
    )
    assert accepted_checkpoint == seed
    assert accepted_checkpoint is not seed
    del args.gazebo_vehicle_model_name

    args._hybrid_rejection_reason = "model-contract-mismatch"
    assert executor._bounded_static_corridor_control_continuity_seed(**values) is None
    args._hybrid_rejection_reason = None
    seed.decision.issue_codes = ["DYNAMIC_TRACK_EVIDENCE_UNRELIABLE"]
    assert executor._bounded_static_corridor_control_continuity_seed(**values) is None


# 功能：动态障碍或定位异常与单纯相机帧老化不同，任何附加异常都必须阻止静态走廊桥接。
# 输入：含动态证据异常的保持命令；输出：拒绝桥接并保留悬停语义。
def test_static_corridor_perception_bridge_rejects_non_perception_issue() -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        navigation_control_authority="route-fallback",
        source="onboard",
        decision=SimpleNamespace(
            action="hold",
            issue_codes=[
                "PERCEPTION_STREAM_STALE",
                "DYNAMIC_TRACK_EVIDENCE_UNRELIABLE",
                "HOLD_UNTIL_FRESH_LOCAL_WORLD",
            ],
        ),
    )

    assert not executor._bounded_static_corridor_perception_bridge_allowed(
        args=SimpleNamespace(
            independent_route_control=True,
            _last_fresh_local_safety_monotonic=10.0,
        ),
        command=command,
        observed=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        tracking_recovery_active=True,
        action_checkpoint_goal=False,
        qualified_route_clearance_m=0.395,
        now=11.0,
    )


# 功能：验证短时感知桥接使用当前实测位置到目标的机体包络净空，而非只复用原路线报告。
# 输入：机型、坐标合同、定位对齐量和一次可控的净空计算。
# 输出：计算路线从实测世界点出发；拒绝报告不授权桥接运动。
def test_static_corridor_bridge_revalidates_actual_segment(monkeypatch, tmp_path) -> None:
    executor = _load_executor()
    captured = []

    def clearance(route, semantic_path, **kwargs):
        captured.append((route, semantic_path, kwargs))
        return SimpleNamespace(accepted=True, minimum_clearance_m=0.21)

    monkeypatch.setattr(executor, "validate_route_clearance", clearance)
    vehicle = executor.VehicleAsset(
        asset_id="test-vehicle",
        name="Test Vehicle",
        dry_mass_kg=1.0,
        max_takeoff_mass_kg=2.0,
        body_radius_m=0.2,
        body_height_m=0.3,
        max_speed_mps=2.0,
        max_acceleration_mps2=3.0,
        qualified_range_m=100.0,
        reserve_battery_percent=20.0,
        max_pickup_payload_kg=0.2,
        sensors=["depth"],
    )
    args = SimpleNamespace(
        semantic=tmp_path / "semantic.json",
        _runtime_vehicle_asset=vehicle,
        _last_estimator_to_world_position_offset_m=Vector3(x=0.1, y=-0.2, z=0.3),
    )
    contract = Px4CoordinateContract(
        model_root_world_enu_m=[10.0, 20.0, 30.0],
        collision_center_offset_model_m=[0.4, -0.2, 0.5],
    )
    target = Vector3(x=12.0, y=22.0, z=33.0)

    report = executor._static_corridor_bridge_clearance(
        args=args,
        observed=SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-1.0),
        target_world=target,
        coordinate_contract=contract,
    )

    assert report.minimum_clearance_m == 0.21
    route, semantic_path, kwargs = captured[0]
    assert route.positions_m[0] == Vector3(x=12.5, y=20.6, z=31.8)
    assert route.positions_m[1] == target
    assert semantic_path == args.semantic
    assert kwargs["vehicle_diameter_m"] == 0.4
    assert kwargs["vehicle_height_m"] == 0.3
    monkeypatch.setattr(
        executor,
        "validate_route_clearance",
        lambda *args, **kwargs: SimpleNamespace(accepted=False),
    )
    assert executor._static_corridor_bridge_clearance(
        args=args,
        observed=SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-1.0),
        target_world=target,
        coordinate_contract=contract,
    ) is None


def test_controlled_landing_holds_xy_until_near_ground_handoff(monkeypatch) -> None:
    executor = _load_executor()
    monkeypatch.setattr(executor, "_CONTROLLED_LANDING_DESCENT_RATE_MPS", 20.0)
    monkeypatch.setattr(executor, "_CONTROLLED_LANDING_STABLE_WINDOW_SECONDS", 0.01)

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint) -> None:
            self.commands.append(setpoint)

        async def sample_position_velocity_ned(self, _timeout_seconds: float):
            command = self.commands[-1]
            return SimpleNamespace(
                north_m=command.north_m,
                east_m=command.east_m,
                down_m=command.down_m,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    client = Client()
    base = SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values))
    terminal = SimpleNamespace(north_m=1.25, east_m=-2.5, down_m=-0.92, yaw_deg=37.0)
    timing: dict[str, Any] = {}

    handoff = asyncio.run(
        executor._controlled_offboard_landing_handoff(
            base=base,
            client=client,
            landing_setpoint=terminal,
            rate_hz=100.0,
            timeout_seconds=1.0,
            timing=timing,
        )
    )

    assert len(client.commands) >= 2
    assert all(command.north_m == pytest.approx(1.25) for command in client.commands)
    assert all(command.east_m == pytest.approx(-2.5) for command in client.commands)
    assert handoff.down_m == pytest.approx(-executor._CONTROLLED_LANDING_HANDOFF_HEIGHT_M)
    assert timing["controlled_offboard_landing"]["status"] == "handoff-ready"


def test_controlled_landing_timeout_scales_with_route_altitude() -> None:
    executor = _load_executor()
    terminal = SimpleNamespace(down_m=-5.52)

    timeout_seconds = executor._controlled_landing_handoff_timeout_seconds(
        landing_setpoint=terminal,
        landing_timeout_seconds=90.0,
    )

    assert timeout_seconds > 30.0
    assert timeout_seconds < 40.0
    with pytest.raises(ValueError, match="cannot fund controlled descent"):
        executor._controlled_landing_handoff_timeout_seconds(
            landing_setpoint=terminal,
            landing_timeout_seconds=15.0,
        )


def test_model_safety_lease_rejects_a_previous_navigation_goal_epoch() -> None:
    executor = _load_executor()
    contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )
    planned = SimpleNamespace(
        north_m=1.0,
        east_m=2.0,
        down_m=-0.5,
        yaw_deg=0.0,
    )
    command = SimpleNamespace(
        tracking_recovery_active=True,
        navigation_control_authority="model-required",
        navigation_goal_id="source-waypoint-0056",
        evaluated_target_position_m=Vector3(x=2.0, y=1.0, z=0.5),
    )

    diagnostic = executor._local_safety_control_context_diagnostic(
        command=command,
        planned_setpoint=planned,
        coordinate_contract=contract,
        tracking_recovery_active=True,
        navigation_goal_id="source-waypoint-0057",
    )

    assert diagnostic["compatible"] is False
    assert diagnostic["context_reason"] == "navigation-goal-epoch-mismatch"
    assert diagnostic["command_navigation_goal_id"] == "source-waypoint-0056"
    assert diagnostic["executor_navigation_goal_id"] == "source-waypoint-0057"


def test_tracking_recovery_radial_feedforward_has_target_deadband() -> None:
    executor = _load_executor()
    target = SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-0.5)

    assert executor._recovery_position_velocity_feedforward(
        observed=SimpleNamespace(north_m=1.002, east_m=1.998, down_m=-0.499),
        target_setpoint=target,
    ) == (0.0, 0.0, 0.0)
    assert executor._recovery_position_velocity_feedforward(
        observed=None,
        target_setpoint=target,
    ) == (0.0, 0.0, 0.0)


def test_tracking_recovery_feedforward_damps_velocity_toward_target() -> None:
    executor = _load_executor()
    target = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-0.5)

    damped = executor._recovery_position_velocity_feedforward(
        observed=SimpleNamespace(
            north_m=0.9,
            east_m=0.0,
            down_m=-0.5,
            north_m_s=0.1,
            east_m_s=0.0,
            down_m_s=0.0,
        ),
        target_setpoint=target,
    )
    assert damped == pytest.approx((0.03, 0.0, 0.0))

    bounded_away_correction = executor._recovery_position_velocity_feedforward(
        observed=SimpleNamespace(
            north_m=0.9,
            east_m=0.0,
            down_m=-0.5,
            north_m_s=-0.1,
            east_m_s=0.0,
            down_m_s=0.0,
        ),
        target_setpoint=target,
    )
    assert bounded_away_correction == pytest.approx((0.08, 0.0, 0.0))


def test_local_repair_progress_renews_stall_window_but_not_without_motion() -> None:
    executor = _load_executor()
    state = executor._advance_local_repair_progress(
        now=0.0,
        observed=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=0.07,
        stall_timeout_seconds=15.0,
        state=None,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )
    assert state["stall_deadline"] == 15.0
    assert state["progress_revision"] == 0

    state = executor._advance_local_repair_progress(
        now=14.0,
        observed=SimpleNamespace(north_m=0.03, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=0.07,
        stall_timeout_seconds=15.0,
        state=state,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )
    assert state["stall_deadline"] == 29.0
    assert state["progress_revision"] == 1
    assert state["last_progress_evidence"] == "vehicle-closed-on-local-repair-target"

    state = executor._advance_local_repair_progress(
        now=28.0,
        observed=SimpleNamespace(north_m=0.031, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=0.071,
        stall_timeout_seconds=15.0,
        state=state,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )
    assert state["stall_deadline"] == 29.0
    assert state["progress_revision"] == 1

    state = executor._advance_local_repair_progress(
        now=28.5,
        observed=SimpleNamespace(north_m=0.031, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=0.076,
        stall_timeout_seconds=15.0,
        state=state,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )
    assert state["stall_deadline"] == 43.5
    assert state["progress_revision"] == 2
    assert state["last_progress_evidence"] == "predicted-clearance-increased"


def test_local_repair_counts_recovery_from_a_recent_clearance_trough() -> None:
    executor = _load_executor()
    state = executor._advance_local_repair_progress(
        now=0.0,
        observed=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=0.17,
        stall_timeout_seconds=15.0,
        state=None,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )
    state = executor._advance_local_repair_progress(
        now=8.0,
        observed=SimpleNamespace(north_m=0.001, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=0.04,
        stall_timeout_seconds=15.0,
        state=state,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )
    assert state["progress_revision"] == 0
    assert state["clearance_progress_anchor_m"] == pytest.approx(0.04)
    assert state["best_minimum_clearance_m"] == pytest.approx(0.17)

    state = executor._advance_local_repair_progress(
        now=14.0,
        observed=SimpleNamespace(north_m=0.002, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=0.13,
        stall_timeout_seconds=15.0,
        state=state,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )
    assert state["stall_deadline"] == pytest.approx(29.0)
    assert state["progress_revision"] == 1
    assert state["last_progress_evidence"] == "predicted-clearance-increased"
    assert state["best_minimum_clearance_m"] == pytest.approx(0.17)


# 功能：
#   验证悬停噪声或背离目标的往返漂移不会被计为局部修复进展。
# 输入：
#   无：构造超过旧位移阈值、但与固定恢复目标距离不减的观测。
# 输出：
#   None：停滞期限和进展版本保持不变。
def test_local_repair_does_not_count_arbitrary_drift_as_progress() -> None:
    executor = _load_executor()
    state = executor._advance_local_repair_progress(
        now=0.0,
        observed=SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        minimum_clearance_m=-0.03,
        stall_timeout_seconds=15.0,
        state=None,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )

    state = executor._advance_local_repair_progress(
        now=14.0,
        observed=SimpleNamespace(north_m=-0.04, east_m=0.04, down_m=-0.95),
        minimum_clearance_m=-0.031,
        stall_timeout_seconds=15.0,
        state=state,
        target_position_ned_m=(1.0, 0.0, -1.0),
    )

    assert state["stall_deadline"] == 15.0
    assert state["progress_revision"] == 0
    assert state["last_progress_evidence"] == "repair-started"

def test_local_repair_refresh_preserves_progress_stall_trigger(tmp_path: Path) -> None:
    executor = _load_executor()
    actions = iter(("hold", "slow"))

    def read_command(_args: Any) -> Any:
        action = next(actions)
        return SimpleNamespace(
            decision=SimpleNamespace(
                action=action,
                selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
                threat_obstacle_id=None,
                minimum_predicted_clearance_m=2.0,
            ),
            estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
            command_position_m=Vector3(x=2.0, y=1.0, z=1.2),
            observation_sequence=1,
            tracking_recovery_active=False,
            evaluated_target_position_m=Vector3(x=2.0, y=1.0, z=1.2),
        )

    executor._read_local_safety_command = read_command

    async def observed_identity(**_kwargs: Any) -> None:
        return None

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        async def set_position_velocity_ned(self, _setpoint: Any, _velocity: Any) -> None:
            return None

        async def set_position_ned(self, _setpoint: Any) -> None:
            return None

    async def exercise() -> None:
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=tmp_path / "target.json",
                local_safety_command=tmp_path / "command.json",
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=1_000.0,
            ),
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=Client(),
            planned_setpoint=SimpleNamespace(
                north_m=1.0,
                east_m=2.0,
                down_m=-1.0,
                yaw_deg=30.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            navigation_goal_position_m=Vector3(x=4.0, y=3.0, z=1.2),
            navigation_goal_id="goal-stalled",
            decision_trigger="progress-stalled",
            recovery_episode_id="recovery-0123456789abcdef01234567",
        )

    asyncio.run(exercise())
    target = json.loads((tmp_path / "target.json").read_text(encoding="utf-8"))
    assert target["decision_trigger"] == "progress-stalled"
    assert target["recovery_episode_id"] == "recovery-0123456789abcdef01234567"


def test_required_local_safety_staleness_holds_measured_position_until_refresh(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command_path = tmp_path / "command.json"
    command_path.write_text("{}", encoding="utf-8")
    calls = 0

    def read_command(_args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return _safety_command_fixture(
            decision=SimpleNamespace(action="continue"),
            estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
        )

    executor._read_local_safety_command = read_command
    executor._read_stale_local_safety_command = lambda _args: SimpleNamespace(
        valid_until_unix_ms=int(executor.time.time() * 1_000) - 50
    )

    async def observed_identity(**_kwargs: Any) -> Any:
        return SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-0.8)

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=command_path,
                local_safety_required=True,
                local_safety_command_grace_seconds=1.0,
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=20.0,
            ),
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=8.0,
                east_m=9.0,
                down_m=-2.0,
                yaw_deg=30.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
        return client

    client = asyncio.run(exercise())
    assert len(client.commands) == 2
    assert (client.commands[0].north_m, client.commands[0].east_m) == (1.0, 2.0)
    phase = json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))
    assert phase["phase"] == "PERCEPTION_REFRESH_HOLD"
    assert phase["schedule_advancement_authorized"] is False


def test_required_local_safety_refresh_hold_latches_first_measured_position(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command_path = tmp_path / "command.json"
    command_path.write_text("{}", encoding="utf-8")
    command_calls = 0

    def read_command(_args: Any) -> Any:
        nonlocal command_calls
        command_calls += 1
        if command_calls <= 2:
            return None
        return _safety_command_fixture(
            decision=SimpleNamespace(action="continue"),
            estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
        )

    executor._read_local_safety_command = read_command
    executor._read_stale_local_safety_command = lambda _args: SimpleNamespace(
        valid_until_unix_ms=int(executor.time.time() * 1_000) - 25
    )
    observations = iter(
        (
            SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-0.8),
            SimpleNamespace(north_m=4.0, east_m=5.0, down_m=-0.4),
            SimpleNamespace(north_m=7.0, east_m=8.0, down_m=-0.2),
        )
    )

    async def observed_identity(**_kwargs: Any) -> Any:
        return next(observations)

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=command_path,
                local_safety_required=True,
                local_safety_runtime_stale_grace_seconds=1.0,
                local_safety_command_grace_seconds=8.0,
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=1_000.0,
            ),
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=9.0,
                east_m=10.0,
                down_m=-1.5,
                yaw_deg=30.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
        return client

    client = asyncio.run(exercise())
    assert len(client.commands) == 3
    assert vars(client.commands[0]) == vars(client.commands[1])
    assert (client.commands[0].north_m, client.commands[0].east_m) == (1.0, 2.0)
    assert (client.commands[2].north_m, client.commands[2].east_m) == (9.0, 10.0)


def test_local_safety_hold_latches_first_repair_position(tmp_path: Path) -> None:
    executor = _load_executor()
    hold_decision = SimpleNamespace(
        action="hold",
        threat_obstacle_id="static:office-frame-east",
        minimum_predicted_clearance_m=0.1,
        selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
    )
    commands = iter(
        (
            _safety_command_fixture(
                decision=hold_decision,
                command_position_m=Vector3(x=1.0, y=2.0, z=1.2),
                estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
                observation_sequence=1,
            ),
            _safety_command_fixture(
                decision=hold_decision,
                command_position_m=Vector3(x=5.0, y=6.0, z=2.0),
                estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
                observation_sequence=2,
            ),
            _safety_command_fixture(
                decision=SimpleNamespace(action="continue"),
                estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
            ),
        )
    )
    executor._read_local_safety_command = lambda _args: next(commands)

    async def observed_identity(**_kwargs: Any) -> None:
        return None

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=tmp_path / "command.json",
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=1_000.0,
            ),
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=9.0,
                east_m=10.0,
                down_m=-1.5,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
        return client

    client = asyncio.run(exercise())
    assert len(client.commands) == 3
    assert vars(client.commands[0]) == vars(client.commands[1])
    assert (client.commands[0].north_m, client.commands[0].east_m) == (2.0, 1.0)
    assert (client.commands[2].north_m, client.commands[2].east_m) == (9.0, 10.0)


def test_required_local_safety_startup_holds_measured_position_until_first_command(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command_path = tmp_path / "not-created-yet.json"
    calls = 0

    def read_command(_args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return _safety_command_fixture(
            decision=SimpleNamespace(action="continue"),
            estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
        )

    executor._read_local_safety_command = read_command

    async def observed_identity(**_kwargs: Any) -> Any:
        return SimpleNamespace(north_m=1.25, east_m=-0.5, down_m=-0.9)

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=command_path,
                local_safety_required=True,
                local_safety_command_grace_seconds=1.0,
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=20.0,
            ),
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=8.0,
                east_m=9.0,
                down_m=-2.0,
                yaw_deg=30.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
        return client

    client = asyncio.run(exercise())
    assert (client.commands[0].north_m, client.commands[0].east_m) == (1.25, -0.5)
    phase = json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))
    assert phase["phase"] == "PERCEPTION_STARTUP_HOLD"
    assert phase["schedule_advancement_authorized"] is False


def test_established_local_safety_missing_file_brakes_and_recovers_within_grace(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command_path = tmp_path / "temporarily-missing-command.json"
    calls = 0

    def read_command(_args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return _safety_command_fixture(
            decision=SimpleNamespace(action="continue"),
            estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
            tracking_recovery_active=False,
            observation_sequence=2,
            valid_until_unix_ms=int(executor.time.time() * 1_000) + 500,
        )

    executor._read_local_safety_command = read_command

    async def observed_identity(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            north_m=1.25,
            east_m=-0.5,
            down_m=-0.9,
            north_m_s=0.2,
            east_m_s=0.0,
            down_m_s=0.0,
        )

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)
            if len(self.commands) == 1:
                command_path.write_text("{}", encoding="utf-8")

    args = SimpleNamespace(
        local_safety_target=None,
        local_safety_command=command_path,
        local_safety_observation=tmp_path / "observation.json",
        local_safety_required=True,
        local_safety_command_grace_seconds=0.01,
        local_safety_runtime_stale_grace_seconds=1.0,
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=1_000.0,
        run_dir=tmp_path,
        _local_safety_command_established=True,
    )
    args._queued_executor_events = []
    args._control_application_writer = SimpleNamespace(
        submit=lambda _path, row: args._queued_executor_events.append(row)
    )
    client = Client()
    planned = SimpleNamespace(
        north_m=8.0,
        east_m=9.0,
        down_m=-2.0,
        yaw_deg=30.0,
    )
    asyncio.run(
        executor._apply_local_safety(
            args=args,
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=planned,
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
    )

    assert len(client.commands) == 2
    assert (client.commands[0].north_m, client.commands[0].east_m) == (1.25, -0.5)
    assert (client.commands[1].north_m, client.commands[1].east_m) == (8.0, 9.0)
    events = args._queued_executor_events
    assert events[0]["status"] == "command-missing"


def test_required_local_safety_staleness_keeps_braking_across_recovery_windows(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command_path = tmp_path / "command.json"
    command_path.write_text("{}", encoding="utf-8")
    executor._read_local_safety_command = lambda _args: None
    executor._read_stale_local_safety_command = lambda _args: SimpleNamespace(
        valid_until_unix_ms=int(executor.time.time() * 1_000) - 3_000
    )

    async def observed_identity(**_kwargs: Any) -> Any:
        return SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0)

    executor._refresh_px4_identity_telemetry = observed_identity

    events: list[dict[str, Any]] = []
    executor._record_local_safety_executor_event = (
        lambda _args, **event: events.append(event)
    )

    class BrakeObserved(Exception):
        pass

    async def set_position_ned(_setpoint: Any) -> None:
        raise BrakeObserved()

    with pytest.raises(BrakeObserved):
        asyncio.run(
            executor._apply_local_safety(
                args=SimpleNamespace(
                    local_safety_target=None,
                    local_safety_command=command_path,
                    local_safety_required=True,
                    local_safety_runtime_stale_grace_seconds=2.0,
                    local_safety_command_grace_seconds=0.5,
                    local_safety_repair_timeout_seconds=15.0,
                    setpoint_rate_hz=20.0,
                ),
                base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
                client=SimpleNamespace(set_position_ned=set_position_ned),
                planned_setpoint=SimpleNamespace(
                    north_m=0.0,
                    east_m=0.0,
                    down_m=-1.0,
                    yaw_deg=0.0,
                ),
                coordinate_contract=Px4CoordinateContract(
                    model_root_world_enu_m=[0.0, 0.0, 0.0],
                    collision_center_offset_model_m=[0.0, 0.0, 0.2],
                ),
                phase_path=tmp_path / "phase.json",
            )
        )
    assert any(
        event["status"] == "command-stale-recovery-window-rolled-over"
        for event in events
    )


def test_runtime_stale_default_holds_through_measured_publication_jitter(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command_path = tmp_path / "command.json"
    command_path.write_text("{}", encoding="utf-8")
    calls = 0

    def read_command(_args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return _safety_command_fixture(
            decision=SimpleNamespace(action="continue"),
            estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
            tracking_recovery_active=False,
            observation_sequence=2,
            valid_until_unix_ms=int(executor.time.time() * 1_000) + 500,
        )

    executor._read_local_safety_command = read_command
    executor._read_stale_local_safety_command = lambda _args: SimpleNamespace(
        valid_until_unix_ms=int(executor.time.time() * 1_000) - 2_500
    )

    async def observed_identity(**_kwargs: Any) -> Any:
        return SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-1.5)

    executor._refresh_px4_identity_telemetry = observed_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

    client = Client()
    asyncio.run(
        executor._apply_local_safety(
            args=SimpleNamespace(
                local_safety_target=None,
                local_safety_command=command_path,
                local_safety_required=True,
                local_safety_command_grace_seconds=8.0,
                local_safety_repair_timeout_seconds=15.0,
                setpoint_rate_hz=1_000.0,
            ),
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=8.0,
                east_m=9.0,
                down_m=-2.0,
                yaw_deg=30.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
        )
    )

    assert (client.commands[0].north_m, client.commands[0].east_m) == (1.0, 2.0)
    assert (client.commands[-1].north_m, client.commands[-1].east_m) == (8.0, 9.0)
    phase = json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))
    assert phase["phase"] == "PERCEPTION_REFRESH_HOLD"
    assert phase["schedule_advancement_authorized"] is False


def test_local_safety_refreshes_px4_identity_telemetry_during_holds(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class Client:
        async def sample_position_velocity_ned(self, timeout_seconds: float) -> Any:
            assert timeout_seconds == 0.25
            return SimpleNamespace(
                north_m=2.0,
                east_m=3.0,
                down_m=-1.5,
                north_m_s=0.1,
                east_m_s=0.2,
                down_m_s=0.0,
            )

        def latest_dynamics_telemetry(self, max_age_seconds: float) -> dict[str, Any]:
            assert max_age_seconds == 3.0
            return {
                "schema_version": "dronedream.px4-dynamics-telemetry.v1",
                "ready_for_payload_inference": True,
                "sources": {},
                "issue_codes": [],
            }

    asyncio.run(
        executor._refresh_px4_identity_telemetry(
            args=SimpleNamespace(
                tracking_telemetry_timeout_seconds=0.25,
                run_dir=tmp_path,
            ),
            client=Client(),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[10.0, 20.0, 1.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
        )
    )

    payload = json.loads(
        (tmp_path / "runtime-state" / "px4-identity-telemetry.json").read_text(encoding="utf-8")
    )
    assert payload["observed_world_collision_center_m"] == {
        "x": 13.0,
        "y": 22.0,
        "z": 2.7,
    }
    assert payload["dynamics"]["ready_for_payload_inference"] is True


def test_identity_telemetry_waits_for_active_stream_and_requires_fresh_sample(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class Client:
        def __init__(self) -> None:
            self.sample_count = 0
            self.restart_count = 0
            self.sample_timeouts = []

        async def sample_position_velocity_ned(self, timeout_seconds: float) -> Any:
            self.sample_timeouts.append(timeout_seconds)
            self.sample_count += 1
            if self.sample_count == 1:
                raise TimeoutError("stalled stream")
            return SimpleNamespace(
                north_m=2.0,
                east_m=3.0,
                down_m=-1.5,
                north_m_s=0.1,
                east_m_s=0.2,
                down_m_s=0.0,
            )

        async def restart_position_velocity_ned_stream(self) -> None:
            self.restart_count += 1

        def position_velocity_stream_needs_restart(self) -> bool:
            return False

    client = Client()
    observed = asyncio.run(
        executor._refresh_px4_identity_telemetry(
            args=SimpleNamespace(
                tracking_telemetry_timeout_seconds=0.01,
                tracking_telemetry_recovery_timeout_seconds=0.2,
                run_dir=tmp_path,
            ),
            client=client,
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[10.0, 20.0, 1.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
        )
    )

    assert observed.north_m == 2.0
    assert client.restart_count == 0
    assert client.sample_timeouts[0] == 0.01
    assert 0.01 < client.sample_timeouts[1] <= 0.2
    recovery = json.loads(
        (tmp_path / "runtime-state" / "px4-telemetry-recovery.json").read_text(encoding="utf-8")
    )
    assert recovery["outage_count"] == 1
    assert recovery["recovered_outage_count"] == 1
    assert recovery["failed_outage_count"] == 0
    assert [event["status"] for event in recovery["events"]] == [
        "holding",
        "recovered",
    ]


def test_identity_telemetry_fails_closed_after_bounded_recovery_window(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class Client:
        restart_count = 0

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            raise TimeoutError("still stalled")

        async def restart_position_velocity_ned_stream(self) -> None:
            self.restart_count += 1

    client = Client()
    with pytest.raises(TimeoutError, match="did not recover within"):
        asyncio.run(
            executor._refresh_px4_identity_telemetry(
                args=SimpleNamespace(
                    tracking_telemetry_timeout_seconds=0.005,
                    tracking_telemetry_recovery_timeout_seconds=0.02,
                    run_dir=tmp_path,
                ),
                client=client,
                coordinate_contract=Px4CoordinateContract(
                    model_root_world_enu_m=[0.0, 0.0, 0.0],
                    collision_center_offset_model_m=[0.0, 0.0, 0.2],
                ),
            )
        )

    assert client.restart_count == 1
    recovery = json.loads(
        (tmp_path / "runtime-state" / "px4-telemetry-recovery.json").read_text(encoding="utf-8")
    )
    assert recovery["outage_count"] == 1
    assert recovery["recovered_outage_count"] == 0
    assert recovery["failed_outage_count"] == 1
    assert recovery["last_status"] == "failed"


def test_identity_telemetry_replaces_live_but_stalled_stream_after_grace(
    tmp_path: Path,
) -> None:
    """A live collector that stops delivering must not consume the whole recovery window."""

    executor = _load_executor()

    class Client:
        def __init__(self) -> None:
            self.sample_count = 0
            self.restart_count = 0

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            self.sample_count += 1
            if self.sample_count < 3:
                raise TimeoutError("live task has no fresh gRPC sample")
            return SimpleNamespace(
                north_m=4.0,
                east_m=5.0,
                down_m=-1.5,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

        async def restart_position_velocity_ned_stream(self) -> None:
            self.restart_count += 1

        def position_velocity_stream_needs_restart(self) -> bool:
            return False

    client = Client()
    observed = asyncio.run(
        executor._refresh_px4_identity_telemetry(
            args=SimpleNamespace(
                tracking_telemetry_timeout_seconds=0.01,
                tracking_telemetry_recovery_timeout_seconds=0.2,
                run_dir=tmp_path,
            ),
            client=client,
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
        )
    )

    assert observed.north_m == 4.0
    assert client.sample_count == 3
    assert client.restart_count == 1
    recovery = json.loads(
        (tmp_path / "runtime-state" / "px4-telemetry-recovery.json").read_text(
            encoding="utf-8"
        )
    )
    assert recovery["last_status"] == "recovered"
    assert recovery["events"][-1]["stream_restarted"] is True


# 功能：
#   验证 50 Hz 稠密参考共享 0.5 Hz 局部规划目标，同时航点和检查点仍是不可跨越的硬边界。
# 输入：
#   无；使用固定索引、频率和边界集合。
# 输出：
#   None：不返回业务数据。
def test_grouped_control_target_reuses_bounded_lookahead_without_crossing_barriers():
    executor = _load_executor()
    barriers = {5, 117, 229}

    assert executor._grouped_control_target_index(
        schedule_index=0,
        schedule_length=300,
        setpoint_rate_hz=50.0,
        barrier_indices=barriers,
    ) == 5
    assert executor._grouped_control_target_index(
        schedule_index=1,
        schedule_length=300,
        setpoint_rate_hz=50.0,
        barrier_indices=barriers,
    ) == 5
    assert executor._grouped_control_target_index(
        schedule_index=6,
        schedule_length=300,
        setpoint_rate_hz=50.0,
        barrier_indices=barriers,
    ) == 99
    assert executor._grouped_control_target_index(
        schedule_index=10,
        schedule_length=300,
        setpoint_rate_hz=50.0,
        barrier_indices=barriers,
    ) == 99
    assert executor._grouped_control_target_index(
        schedule_index=299,
        schedule_length=300,
        setpoint_rate_hz=50.0,
        barrier_indices=barriers,
    ) == 299

    # The end of takeoff is also a hard direction-change boundary even though
    # it is not one of the source-route waypoint arrivals.
    assert executor._grouped_control_target_index(
        schedule_index=115,
        schedule_length=300,
        setpoint_rate_hz=50.0,
        barrier_indices={128, 299},
    ) == 128


# 功能：
#   非法分组索引和越界屏障必须在发出任何控制前被拒绝。
# 输入：
#   无；构造越界索引及屏障。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "values",
    [
        {"schedule_index": -1, "schedule_length": 10, "barrier_indices": set()},
        {"schedule_index": 0, "schedule_length": 0, "barrier_indices": set()},
        {"schedule_index": 0, "schedule_length": 10, "barrier_indices": {10}},
    ],
)
def test_grouped_control_target_rejects_invalid_indices(values):
    executor = _load_executor()

    with pytest.raises(ValueError, match="grouped control target"):
        executor._grouped_control_target_index(setpoint_rate_hz=50.0, **values)


# 功能：
#   验证运动相位边界覆盖移动转悬停、悬停转移动和转弯，但不把同向加速误判为边界。
# 输入：
#   无；构造含同向变速、悬停和直角转弯的稠密参考序列。
# 输出：
#   None：不返回业务数据。
def test_schedule_motion_barriers_preserve_hold_and_turn_phases():
    executor = _load_executor()
    schedule = [
        SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0),
        SimpleNamespace(north_m=0.1, east_m=0.0, down_m=-1.0),
        SimpleNamespace(north_m=0.3, east_m=0.0, down_m=-1.0),
        SimpleNamespace(north_m=0.3, east_m=0.0, down_m=-1.0),
        SimpleNamespace(north_m=0.3, east_m=0.0, down_m=-1.0),
        SimpleNamespace(north_m=0.3, east_m=0.2, down_m=-1.0),
        SimpleNamespace(north_m=0.3, east_m=0.4, down_m=-1.0),
        SimpleNamespace(north_m=0.1, east_m=0.4, down_m=-1.0),
    ]

    assert executor._schedule_motion_barrier_indices(schedule) == {0, 2, 4, 6, 7}
    assert executor._grouped_control_target_index(
        schedule_index=3,
        schedule_length=len(schedule),
        setpoint_rate_hz=50.0,
        barrier_indices=executor._schedule_motion_barrier_indices(schedule),
    ) == 4


# 功能：
#   非有限坐标不得进入运动相位分析，避免错误边界静默污染真实控制目标。
# 输入：
#   无；构造含无穷坐标的参考点。
# 输出：
#   None：不返回业务数据。
def test_schedule_motion_barriers_reject_nonfinite_coordinates():
    executor = _load_executor()

    with pytest.raises(ValueError, match="not finite"):
        executor._schedule_motion_barrier_indices(
            [SimpleNamespace(north_m=math.inf, east_m=0.0, down_m=-1.0)]
        )


# 功能：
#   验证稠密参考结算只跳过飞行器已实测越过的点，并保留组末硬边界供下一轮执行。
# 输入：
#   无；构造朝同一语义航点前进的五个 NED 参考点。
# 输出：
#   None：不返回业务数据。
def test_measured_dense_schedule_catch_up_preserves_group_boundary():
    executor = _load_executor()
    schedule = [
        SimpleNamespace(north_m=float(north), east_m=0.0, down_m=0.0)
        for north in range(5)
    ]
    coordinate_contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )

    assert executor._measured_dense_schedule_catch_up_index(
        schedule=schedule,
        current_index=0,
        maximum_index=4,
        coordinate_contract=coordinate_contract,
        navigation_goal_position_m=Vector3(x=0.0, y=10.0, z=0.2),
        measured_goal_distance_m=7.2,
    ) == 2


# 功能：
#   非法实测距离或反向索引范围必须被拒绝，不能产生未经证据支持的参考跳跃。
# 输入：
#   无；构造负距离和反向索引。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "current_index,maximum_index,measured_goal_distance_m",
    [(2, 1, 1.0), (0, 1, -0.1), (0, 1, math.nan)],
)
def test_measured_dense_schedule_catch_up_rejects_invalid_inputs(
    current_index,
    maximum_index,
    measured_goal_distance_m,
):
    executor = _load_executor()
    schedule = [SimpleNamespace(north_m=0.0, east_m=0.0, down_m=0.0)] * 3

    with pytest.raises(ValueError, match="measured dense schedule catch-up"):
        executor._measured_dense_schedule_catch_up_index(
            schedule=schedule,
            current_index=current_index,
            maximum_index=maximum_index,
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            navigation_goal_position_m=Vector3(x=0.0, y=1.0, z=0.2),
            measured_goal_distance_m=measured_goal_distance_m,
        )


def test_tracking_gate_holds_schedule_until_vehicle_rejoins(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []
            self.samples = [
                SimpleNamespace(
                    north_m=0.0,
                    east_m=0.0,
                    down_m=-1.0,
                    north_m_s=0.2,
                    east_m_s=0.0,
                    down_m_s=0.0,
                ),
                SimpleNamespace(
                    north_m=0.72,
                    east_m=0.0,
                    down_m=-1.0,
                    north_m_s=0.1,
                    east_m_s=0.0,
                    down_m_s=0.0,
                ),
            ]

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return self.samples.pop(0)

    args = SimpleNamespace(
        run_dir=tmp_path,
        local_safety_target=None,
        local_safety_command=None,
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=1_000.0,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.75,
        tracking_rejoin_tolerance_m=0.35,
    )
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)
    coordinate_contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )

    async def exercise() -> tuple[Client, tuple[bool, float | None], tuple[bool, float | None]]:
        client = Client()
        first = await executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=planned,
            coordinate_contract=coordinate_contract,
            phase_path=tmp_path / "phase.json",
            schedule_index=17,
            sample_now=True,
            recovery_active=False,
        )
        second = await executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=client,
            planned_setpoint=planned,
            coordinate_contract=coordinate_contract,
            phase_path=tmp_path / "phase.json",
            schedule_index=17,
            sample_now=True,
            recovery_active=True,
        )
        return client, first, second

    client, first, second = asyncio.run(exercise())
    assert first == (False, 1.0)
    assert second[0] is True
    assert second[1] == pytest.approx(0.28)
    assert len(client.commands) == 2
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert state["state"] == "tracking"
    assert state["advance_threshold_m"] == 0.35


def test_tracking_gate_accepts_small_settled_recovery_residual(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class Client:
        async def set_position_ned(self, _setpoint: Any) -> None:
            return None

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=0.918,
                east_m=0.0,
                down_m=-1.0,
                north_m_s=0.004,
                east_m_s=0.002,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path,
        local_safety_target=None,
        local_safety_command=None,
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=1_000.0,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.076,
        tracking_rejoin_tolerance_m=0.068,
    )
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)
    coordinate_contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )

    advanced, error_m = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(Setpoint=lambda **values: SimpleNamespace(**values)),
            client=Client(),
            planned_setpoint=planned,
            coordinate_contract=coordinate_contract,
            phase_path=tmp_path / "phase.json",
            schedule_index=431,
            sample_now=True,
            recovery_active=True,
        )
    )

    assert advanced is True
    assert error_m == pytest.approx(0.082)
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["advance_threshold_m"] == pytest.approx(0.093)


def test_tracking_gate_compares_px4_telemetry_in_route_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    estimator_offset = Vector3(x=0.1, y=0.2, z=0.3)
    corrected_setpoint = SimpleNamespace(
        north_m=-0.2,
        east_m=-0.1,
        down_m=-0.7,
        yaw_deg=0.0,
    )

    async def apply_local_safety(**kwargs: Any) -> Any:
        args = kwargs["args"]
        args._last_model_control_required = False
        args._last_model_control_authorized = False
        args._last_estimator_to_world_position_offset_m = estimator_offset
        return corrected_setpoint

    monkeypatch.setattr(executor, "_apply_local_safety", apply_local_safety)

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=-0.2,
                east_m=-0.1,
                down_m=-0.7,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.05,
        tracking_rejoin_tolerance_m=0.025,
    )
    planned = SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)

    result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=planned,
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            schedule_index=11,
            sample_now=True,
            recovery_active=False,
        )
    )

    assert result == (True, pytest.approx(0.0))
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert state["controller_position_error_m"] == pytest.approx(0.0)
    assert state["route_schedule_position_error_m"] == pytest.approx(0.0)
    assert state["estimator_to_world_position_offset_m"] == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
    }
    assert state["observed_world_collision_center_m"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": 1.2}
    )


def test_tracking_gate_reuses_fresh_local_safety_telemetry_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    cached = SimpleNamespace(
        north_m=1.0,
        east_m=0.0,
        down_m=-1.0,
        north_m_s=0.0,
        east_m_s=0.0,
        down_m_s=0.0,
    )

    async def apply_local_safety(**kwargs: Any) -> Any:
        args = kwargs["args"]
        args._last_model_control_required = False
        args._last_model_control_authorized = False
        args._last_estimator_to_world_position_offset_m = Vector3(x=0.0, y=0.0, z=0.0)
        args._last_px4_control_observation = cached
        return kwargs["planned_setpoint"]

    monkeypatch.setattr(executor, "_apply_local_safety", apply_local_safety)

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            raise AssertionError("tracking gate awaited a duplicate PX4 telemetry frame")

    result = asyncio.run(
        executor._tracking_gate_tick(
            args=SimpleNamespace(
                run_dir=tmp_path,
                tracking_telemetry_timeout_seconds=0.1,
                tracking_lag_limit_m=0.05,
                tracking_rejoin_tolerance_m=0.025,
            ),
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=SimpleNamespace(
                north_m=1.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            schedule_index=11,
            sample_now=True,
            recovery_active=False,
        )
    )

    assert result == (True, pytest.approx(0.0))


def test_tracking_gate_tolerates_bounded_along_route_lag_but_not_cross_track_error(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class Client:
        def __init__(self, *, east_m: float) -> None:
            self.east_m = east_m

        async def set_position_ned(self, _setpoint: Any) -> None:
            return None

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=0.94,
                east_m=self.east_m,
                down_m=-1.0,
                north_m_s=0.15,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path / "along",
        local_safety_target=None,
        local_safety_command=None,
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=1_000.0,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.045,
        tracking_rejoin_tolerance_m=0.04,
    )
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)
    contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )

    along_result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(east_m=0.0),
            planned_setpoint=planned,
            coordinate_contract=contract,
            phase_path=tmp_path / "along-phase.json",
            schedule_index=11,
            sample_now=True,
            recovery_active=False,
            planned_velocity_ned_mps=(0.15, 0.0, 0.0),
        )
    )

    assert along_result == (True, pytest.approx(0.0))
    along_state = json.loads(
        (args.run_dir / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert along_state["controller_position_error_m"] == pytest.approx(0.06)
    assert along_state["route_cross_track_error_m"] == pytest.approx(0.0)
    assert along_state["route_along_track_lag_m"] == pytest.approx(0.06)
    assert along_state["route_along_track_lag_limit_m"] == pytest.approx(0.15)
    assert along_state["route_along_track_lag_rejoin_limit_m"] == pytest.approx(0.1125)

    args.run_dir = tmp_path / "cross"
    cross_result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(east_m=0.06),
            planned_setpoint=planned,
            coordinate_contract=contract,
            phase_path=tmp_path / "cross-phase.json",
            schedule_index=11,
            sample_now=True,
            recovery_active=False,
            planned_velocity_ned_mps=(0.15, 0.0, 0.0),
        )
    )

    assert cross_result == (False, pytest.approx(0.06))
    cross_state = json.loads(
        (args.run_dir / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert cross_state["route_cross_track_error_m"] == pytest.approx(0.06)
    assert cross_state["schedule_advancement_authorized"] is False


def test_tracking_gate_requires_tighter_along_route_rejoin_after_recovery(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class Client:
        def __init__(self, north_m: float) -> None:
            self.north_m = north_m

        async def set_position_ned(self, _setpoint: Any) -> None:
            return None

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=self.north_m,
                east_m=0.0,
                down_m=-1.0,
                north_m_s=0.15,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path / "still-recovering",
        local_safety_target=None,
        local_safety_command=None,
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=1_000.0,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.045,
        tracking_rejoin_tolerance_m=0.04,
    )
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)
    contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )

    still_recovering = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(north_m=0.87),
            planned_setpoint=planned,
            coordinate_contract=contract,
            phase_path=tmp_path / "still-recovering-phase.json",
            schedule_index=12,
            sample_now=True,
            recovery_active=True,
            planned_velocity_ned_mps=(0.15, 0.0, 0.0),
        )
    )
    assert still_recovering == (False, pytest.approx(0.13))

    args.run_dir = tmp_path / "rejoined"
    rejoined = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(north_m=0.89),
            planned_setpoint=planned,
            coordinate_contract=contract,
            phase_path=tmp_path / "rejoined-phase.json",
            schedule_index=12,
            sample_now=True,
            recovery_active=True,
            planned_velocity_ned_mps=(0.15, 0.0, 0.0),
        )
    )
    assert rejoined == (True, pytest.approx(0.0))


def test_tracking_gate_separates_along_lag_hold_from_cross_track_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    applied_contexts: list[bool] = []

    async def apply_local_safety(**kwargs: Any) -> Any:
        args = kwargs["args"]
        applied_contexts.append(bool(kwargs["tracking_recovery_active"]))
        args._last_model_control_required = False
        args._last_model_control_authorized = False
        args._last_estimator_to_world_position_offset_m = Vector3(x=0.0, y=0.0, z=0.0)
        return kwargs["planned_setpoint"]

    monkeypatch.setattr(executor, "_apply_local_safety", apply_local_safety)

    class Client:
        def __init__(self, *, north_m: float, east_m: float) -> None:
            self.north_m = north_m
            self.east_m = east_m

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=self.north_m,
                east_m=self.east_m,
                down_m=-1.0,
                north_m_s=0.15,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path / "along",
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.045,
        tracking_rejoin_tolerance_m=0.04,
    )
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)
    contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )

    along_hold = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(north_m=0.84, east_m=0.0),
            planned_setpoint=planned,
            coordinate_contract=contract,
            phase_path=tmp_path / "along-phase.json",
            schedule_index=13,
            sample_now=True,
            recovery_active=True,
            local_recovery_control_active=False,
            planned_velocity_ned_mps=(0.15, 0.0, 0.0),
            context={"semantic_approach_damping_active": True},
        )
    )
    assert along_hold == (False, pytest.approx(0.16))
    assert args._last_tracking_local_recovery_control_required is False

    args.run_dir = tmp_path / "cross"
    cross_hold = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(north_m=1.0, east_m=0.06),
            planned_setpoint=planned,
            coordinate_contract=contract,
            phase_path=tmp_path / "cross-phase.json",
            schedule_index=13,
            sample_now=True,
            recovery_active=True,
            local_recovery_control_active=True,
            planned_velocity_ned_mps=(0.15, 0.0, 0.0),
        )
    )
    assert cross_hold == (False, pytest.approx(0.06))
    assert args._last_tracking_local_recovery_control_required is True
    assert applied_contexts == [False, True]


def test_tracking_recovery_targets_current_reference_not_grouped_lookahead(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    applied_targets: list[float] = []

    async def apply_local_safety(**kwargs: Any) -> Any:
        args = kwargs["args"]
        applied_targets.append(float(kwargs["planned_setpoint"].north_m))
        args._last_model_control_required = False
        args._last_model_control_authorized = False
        args._last_estimator_to_world_position_offset_m = Vector3(x=0.0, y=0.0, z=0.0)
        return kwargs["planned_setpoint"]

    monkeypatch.setattr(executor, "_apply_local_safety", apply_local_safety)

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=1.0,
                east_m=0.2,
                down_m=-1.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.15,
        tracking_rejoin_tolerance_m=0.10,
    )
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)
    lookahead = SimpleNamespace(north_m=2.0, east_m=0.0, down_m=-1.0, yaw_deg=0.0)
    contract = Px4CoordinateContract(
        model_root_world_enu_m=[0.0, 0.0, 0.0],
        collision_center_offset_model_m=[0.0, 0.0, 0.2],
    )

    asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=planned,
            control_setpoint=lookahead,
            coordinate_contract=contract,
            phase_path=tmp_path / "phase.json",
            schedule_index=14,
            sample_now=True,
            recovery_active=True,
            local_recovery_control_active=True,
            planned_velocity_ned_mps=(0.15, 0.0, 0.0),
        )
    )

    assert applied_targets == [1.0]


def test_schedule_velocity_feedforward_is_bounded_and_zero_during_recovery() -> None:
    executor = _load_executor()
    current = SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0)
    following = SimpleNamespace(north_m=0.04, east_m=0.03, down_m=-1.0)

    velocity = executor._schedule_velocity_feedforward(
        current_setpoint=current,
        next_setpoint=following,
        rate_hz=20.0,
        speed_limit_mps=0.5,
        recovery_active=False,
    )
    assert velocity == pytest.approx((0.4, 0.3, 0.0))
    assert executor._schedule_velocity_feedforward(
        current_setpoint=current,
        next_setpoint=following,
        rate_hz=20.0,
        speed_limit_mps=0.5,
        recovery_active=True,
    ) == (0.0, 0.0, 0.0)


def test_route_velocity_feedforward_is_suppressed_when_vehicle_leads_reference() -> None:
    executor = _load_executor()
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0)
    observed = SimpleNamespace(north_m=1.08, east_m=0.0, down_m=-1.0)

    velocity, suppressed, lead_m = executor._lead_aware_route_velocity_feedforward(
        observed=observed,
        planned_setpoint=planned,
        planned_velocity_ned_mps=(0.5, 0.0, 0.0),
        estimator_offset_world_enu_m=Vector3(x=0.0, y=0.0, z=0.0),
    )

    assert velocity == (0.0, 0.0, 0.0)
    assert suppressed is True
    assert lead_m == pytest.approx(0.08)


def test_route_velocity_feedforward_remains_available_behind_reference() -> None:
    executor = _load_executor()
    planned = SimpleNamespace(north_m=1.0, east_m=0.0, down_m=-1.0)
    observed = SimpleNamespace(north_m=0.92, east_m=0.0, down_m=-1.0)

    velocity, suppressed, lead_m = executor._lead_aware_route_velocity_feedforward(
        observed=observed,
        planned_setpoint=planned,
        planned_velocity_ned_mps=(0.5, 0.0, 0.0),
        estimator_offset_world_enu_m=Vector3(x=0.0, y=0.0, z=0.0),
    )

    assert velocity == (0.5, 0.0, 0.0)
    assert suppressed is False
    assert lead_m == 0.0


def test_model_required_continue_applies_model_command_instead_of_route_setpoint(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        valid_until_unix_ms=int(datetime.now(UTC).timestamp() * 1000) + 1_500,
        decision=SimpleNamespace(
            action="continue",
            threat_obstacle_id=None,
            minimum_predicted_clearance_m=1.0,
            selected_velocity_mps=Vector3(x=0.3, y=0.2, z=0.0),
            control_source="route-target",
        ),
        command_position_m=Vector3(x=1.0, y=2.0, z=1.2),
        estimator_to_world_position_offset_m=Vector3(x=0.0, y=0.0, z=0.0),
        observation_sequence=11,
        navigation_control_authority="model-required",
        model_navigation_authorized=True,
        model_call_id="call-authority",
        model_selected_candidate_id="candidate-authority",
        navigation_goal_id="goal-authority",
        evaluated_target_position_m=Vector3(x=9.0, y=8.0, z=3.0),
        requested_control_intent=None,
        model_path_sha256="b" * 64,
        model_authority_reason="model-path-lease-active",
    )
    executor._read_local_safety_command = lambda _args: command

    async def ignore_identity_refresh(**_kwargs: Any) -> None:
        return None

    executor._refresh_px4_identity_telemetry = ignore_identity_refresh

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []
            self.velocities: list[Any] = []

        def latest_dynamics_telemetry(self, _max_age_seconds: float) -> dict[str, Any]:
            return {"sources": {"attitude": {"yaw_deg": 0., "sample_age_seconds": .01}}}

        async def set_velocity_ned(self, velocity: Any) -> None:
            self.velocities.append(velocity)

    args = SimpleNamespace(
        local_safety_target=None,
        local_safety_command=tmp_path / "command.json",
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=1_000.0,
    )

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=args,
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=9.0,
                east_m=8.0,
                down_m=-3.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(0.8, 0.0, 0.0),
        )
        return client

    client = asyncio.run(exercise())
    assert client.commands == []
    assert client.velocities[0].north_m_s == pytest.approx(0.2)
    assert client.velocities[0].east_m_s == pytest.approx(0.3)
    assert args._model_authorized_control_applied_count == 1
    assert args._model_control_application_counts == {"route-derived": 1}


def test_required_model_authority_rejects_route_compatibility_command() -> None:
    executor = _load_executor()
    route_command = SimpleNamespace(navigation_control_authority="route-fallback")
    model_command = SimpleNamespace(navigation_control_authority="model-required")

    assert not executor._local_safety_command_matches_required_authority(
        args=SimpleNamespace(require_model_control_authority=True),
        command=route_command,
    )
    assert executor._local_safety_command_matches_required_authority(
        args=SimpleNamespace(require_model_control_authority=True),
        command=model_command,
    )
    assert executor._local_safety_command_matches_required_authority(
        args=SimpleNamespace(require_model_control_authority=False),
        command=route_command,
    )


# 功能：验证独立航线控制只接受固定部署坐标，Gazebo 见证偏移不能进入执行器。
# 输入：分别携带零偏移和非零偏移的本地安全命令；输出：仅零偏移命令匹配权限合同。
def test_independent_route_control_rejects_truth_fitted_offset() -> None:
    executor = _load_executor()
    args = SimpleNamespace(
        independent_route_control=True,
        bounded_hybrid_control=False,
        require_model_control_authority=False,
    )

    def command(offset: Vector3) -> SimpleNamespace:
        return SimpleNamespace(
            navigation_control_authority="route-fallback",
            model_navigation_authorized=False,
            requested_control_intent=None,
            source="onboard",
            estimator_to_world_position_offset_m=offset,
        )

    assert executor._local_safety_command_matches_required_authority(
        args=args,
        command=command(Vector3(x=0.0, y=0.0, z=0.0)),
    )
    assert not executor._local_safety_command_matches_required_authority(
        args=args,
        command=command(Vector3(x=0.0, y=0.0, z=0.01)),
    )


def test_model_authority_gap_holds_live_px4_position_not_corrected_gazebo_pose(
    tmp_path: Path,
) -> None:
    executor = _load_executor()
    command = _safety_command_fixture(
        decision=SimpleNamespace(
            action="hold",
            control_source="deterministic-brake",
            threat_obstacle_id=None,
            minimum_predicted_clearance_m=0.3,
            selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        ),
        # Deliberately far from the measured PX4 position. A fail-closed model
        # gap must never fly toward this separately corrected Gazebo value.
        command_position_m=Vector3(x=20.0, y=30.0, z=40.0),
        estimator_to_world_position_offset_m=Vector3(x=0.5, y=0.5, z=0.5),
        observation_sequence=12,
        navigation_control_authority="model-required",
        model_navigation_authorized=False,
        model_call_id=None,
        model_selected_candidate_id=None,
        model_path_sha256=None,
        model_authority_reason="model-lease-not-established",
        requested_control_intent=None,
        # A runtime command always carries its immutable input deadline.
        # Keep this hold-position fixture valid without bypassing that field.
        valid_until_unix_ms=int(datetime.now(UTC).timestamp() * 1000) + 1_500,
    )
    executor._read_local_safety_command = lambda _args: command

    measured_positions = iter(
        (
            SimpleNamespace(
                north_m=1.25,
                east_m=-0.75,
                down_m=-0.9,
                north_m_s=0.18,
                east_m_s=0.0,
                down_m_s=0.0,
            ),
            # While the vehicle is still moving, the zero-velocity braking
            # target follows measured PX4 position instead of pulling back to
            # the first point and creating a confined-space return arc.
            SimpleNamespace(
                north_m=1.55,
                east_m=-0.95,
                down_m=-1.1,
                north_m_s=0.08,
                east_m_s=0.0,
                down_m_s=0.0,
            ),
            # The first low-speed sample becomes the fixed drift-rejection
            # point for the remainder of this authority gap.
            SimpleNamespace(
                north_m=1.62,
                east_m=-0.98,
                down_m=-1.08,
                north_m_s=0.02,
                east_m_s=0.0,
                down_m_s=0.0,
            ),
            SimpleNamespace(
                north_m=1.70,
                east_m=-1.02,
                down_m=-1.06,
                north_m_s=0.01,
                east_m_s=0.0,
                down_m_s=0.0,
            ),
        )
    )

    async def measured_identity(**_kwargs: Any) -> Any:
        return next(measured_positions)

    executor._refresh_px4_identity_telemetry = measured_identity

    class Client:
        def __init__(self) -> None:
            self.commands: list[Any] = []
            self.velocities: list[Any] = []

        def latest_dynamics_telemetry(self, max_age_seconds: float) -> dict[str, Any]:
            assert max_age_seconds == .25
            return {"sources": {"attitude": {"yaw_deg": 114., "sample_age_seconds": .02}}}

        async def set_position_velocity_ned(self, setpoint: Any, velocity: Any) -> None:
            self.commands.append(setpoint)
            self.velocities.append(velocity)

    args = SimpleNamespace(
        local_safety_target=None,
        local_safety_command=tmp_path / "command.json",
        local_safety_repair_timeout_seconds=15.0,
        setpoint_rate_hz=1_000.0,
    )

    async def exercise() -> Client:
        client = Client()
        await executor._apply_local_safety(
            args=args,
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=9.0,
                east_m=8.0,
                down_m=-3.0,
                yaw_deg=17.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(0.8, 0.0, 0.0),
        )
        await executor._apply_local_safety(
            args=args,
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=9.0,
                east_m=8.0,
                down_m=-3.0,
                yaw_deg=17.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(0.8, 0.0, 0.0),
        )
        await executor._apply_local_safety(
            args=args,
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=9.0,
                east_m=8.0,
                down_m=-3.0,
                yaw_deg=17.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(0.8, 0.0, 0.0),
        )
        await executor._apply_local_safety(
            args=args,
            base=SimpleNamespace(
                Setpoint=lambda **values: SimpleNamespace(**values),
                VelocitySetpoint=lambda **values: SimpleNamespace(**values),
            ),
            client=client,
            planned_setpoint=SimpleNamespace(
                north_m=9.0,
                east_m=8.0,
                down_m=-3.0,
                yaw_deg=17.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            planned_velocity_ned_mps=(0.8, 0.0, 0.0),
        )
        return client

    client = asyncio.run(exercise())
    assert len(client.commands) == 4
    assert [
        (setpoint.north_m, setpoint.east_m, setpoint.down_m) for setpoint in client.commands
    ] == pytest.approx(
        [
            (1.25, -0.75, -0.9),
            (1.55, -0.95, -1.1),
            (1.62, -0.98, -1.08),
            (1.62, -0.98, -1.08),
        ]
    )
    for velocity in client.velocities:
        assert (
            velocity.north_m_s,
            velocity.east_m_s,
            velocity.down_m_s,
        ) == pytest.approx((0.0, 0.0, 0.0))
    assert args._model_authority_hold_setpoint is client.commands[2]
    assert [item.yaw_deg for item in client.commands] == [114.] * 4
    assert args._model_body_control_yaw_deg == 114.
    phase = json.loads((tmp_path / "phase.json").read_text(encoding="utf-8"))
    assert phase["phase"] == "MODEL_AUTHORITY_HOLD"
    assert phase["hold_source"] == "px4-measured-position"
    assert phase["hold_latched_for_authority_gap"] is True
    assert phase["hold_observed_speed_mps"] == pytest.approx(0.01)
    assert phase["hold_latch_speed_threshold_mps"] == pytest.approx(0.05)


def test_model_required_missing_lease_holds_schedule_without_route_recovery(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    async def hold_without_authority(*, args: Any, planned_setpoint: Any, **_kwargs: Any) -> Any:
        args._last_model_control_required = True
        args._last_model_control_authorized = False
        args._last_model_call_id = None
        args._last_model_selected_candidate_id = None
        args._last_model_path_sha256 = None
        args._last_estimator_to_world_position_offset_m = Vector3(x=0.1, y=0.2, z=0.3)
        args._last_px4_control_observation = SimpleNamespace(
            north_m=1.0,
            east_m=2.0,
            down_m=-1.0,
            north_m_s=0.1,
            east_m_s=0.2,
            down_m_s=0.0,
        )
        return planned_setpoint

    executor._apply_local_safety = hold_without_authority

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            raise AssertionError("an authority hold must not sample or release route progress")

    args = SimpleNamespace(
        run_dir=tmp_path,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.75,
        tracking_rejoin_tolerance_m=0.35,
    )
    result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=SimpleNamespace(
                north_m=5.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            schedule_index=25,
            sample_now=False,
            recovery_active=False,
        )
    )
    assert result == (False, None)
    assert args._model_authority_schedule_hold_count == 1
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert state["state"] == "model-authority-hold"
    assert state["schedule_advancement_authorized"] is False
    assert state["observed_position_ned"] == {
        "north_m": 1.0,
        "east_m": 2.0,
        "down_m": -1.0,
    }
    assert state["observed_velocity_ned_mps"] == {
        "north_m_s": 0.1,
        "east_m_s": 0.2,
        "down_m_s": 0.0,
    }
    assert state["observed_world_collision_center_m"] == pytest.approx(
        {"x": 2.1, "y": 1.2, "z": 1.5}
    )
    assert state["speed_mps"] == pytest.approx(math.sqrt(0.05))


def test_model_required_local_target_cannot_race_route_schedule(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    async def apply_authorized_local_target(*, args: Any, **_kwargs: Any) -> Any:
        args._last_model_control_required = True
        args._last_model_control_authorized = True
        args._last_model_call_id = "call-progress-gate"
        args._last_model_selected_candidate_id = "candidate-progress-gate"
        args._last_model_path_sha256 = "c" * 64
        return SimpleNamespace(
            north_m=0.1,
            east_m=0.0,
            down_m=-1.0,
            yaw_deg=0.0,
        )

    executor._apply_local_safety = apply_authorized_local_target

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=0.1,
                east_m=0.0,
                down_m=-1.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.2,
        tracking_rejoin_tolerance_m=0.1,
        _model_goal_progress_state={
            "navigation_goal_id": "goal-progress-gate",
            "initial_goal_distance_m": 2.0,
            "accepted_schedule_distance_m": 0.0,
            "last_advanced_planned_ned": (0.0, 0.0, -1.0),
        },
    )
    result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=SimpleNamespace(
                north_m=2.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            schedule_index=26,
            sample_now=True,
            recovery_active=False,
            context={
                "navigation_goal_id": "goal-progress-gate",
                "navigation_goal_position_m": {"x": 0.0, "y": 2.0, "z": 1.2},
            },
        )
    )

    assert result == (False, None)
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert state["controller_position_error_m"] == pytest.approx(0.0)
    assert state["route_schedule_position_error_m"] == pytest.approx(1.9)
    assert state["model_goal_progress_m"] == pytest.approx(0.1)
    assert state["model_candidate_schedule_progress_m"] == pytest.approx(2.0)
    assert state["model_schedule_progress_authorized"] is False
    assert state["schedule_advancement_authorized"] is False


def test_model_required_semantic_lead_allows_dense_reference_to_catch_up(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    async def apply_authorized_local_target(*, args: Any, **_kwargs: Any) -> Any:
        args._last_model_control_required = True
        args._last_model_control_authorized = True
        args._last_model_call_id = "call-semantic-lead"
        args._last_model_selected_candidate_id = "candidate-semantic-lead"
        args._last_model_path_sha256 = "e" * 64
        return SimpleNamespace(
            north_m=1.5,
            east_m=0.0,
            down_m=-1.0,
            yaw_deg=0.0,
        )

    executor._apply_local_safety = apply_authorized_local_target

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=1.0,
                east_m=0.0,
                down_m=-1.0,
                north_m_s=0.1,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.2,
        tracking_rejoin_tolerance_m=0.1,
        model_progress_slack_m=0.2,
    )
    result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=SimpleNamespace(
                north_m=0.5,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            schedule_index=27,
            sample_now=True,
            recovery_active=True,
            context={
                "navigation_goal_id": "goal-semantic-lead",
                "navigation_goal_position_m": {"x": 0.0, "y": 3.0, "z": 1.2},
            },
        )
    )

    assert result == (True, pytest.approx(0.5))
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert state["model_goal_distance_m"] == pytest.approx(2.0)
    assert state["planned_goal_distance_m"] == pytest.approx(2.5)
    assert state["controller_position_error_m"] == pytest.approx(0.5)
    assert state["model_controller_target_reached"] is False
    assert state["model_reference_catch_up_authorized"] is True
    assert state["schedule_advancement_authorized"] is True


def test_model_required_reference_catch_up_rejects_vehicle_behind_with_only_slack(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    async def apply_authorized_local_target(*, args: Any, **_kwargs: Any) -> Any:
        args._last_model_control_required = True
        args._last_model_control_authorized = True
        args._last_model_call_id = "call-semantic-behind"
        args._last_model_selected_candidate_id = "candidate-semantic-behind"
        args._last_model_path_sha256 = "f" * 64
        return SimpleNamespace(
            north_m=1.5,
            east_m=0.0,
            down_m=-1.0,
            yaw_deg=0.0,
        )

    executor._apply_local_safety = apply_authorized_local_target

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=1.0,
                east_m=0.0,
                down_m=-1.0,
                north_m_s=0.1,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.2,
        tracking_rejoin_tolerance_m=0.1,
        model_progress_slack_m=0.2,
    )
    result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=SimpleNamespace(
                north_m=1.2,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            schedule_index=28,
            sample_now=True,
            recovery_active=True,
            context={
                "navigation_goal_id": "goal-semantic-behind",
                "navigation_goal_position_m": {"x": 0.0, "y": 3.0, "z": 1.2},
            },
        )
    )

    assert result == (False, pytest.approx(0.5))
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert state["model_goal_distance_m"] == pytest.approx(2.0)
    assert state["planned_goal_distance_m"] == pytest.approx(1.8)
    assert state["model_schedule_progress_authorized"] is True
    assert state["model_controller_target_reached"] is False
    assert state["model_reference_catch_up_authorized"] is False
    assert state["schedule_advancement_authorized"] is False


def test_model_progress_gate_does_not_deadlock_on_reference_path_stretch(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    async def apply_authorized_local_target(*, args: Any, **_kwargs: Any) -> Any:
        args._last_model_control_required = True
        args._last_model_control_authorized = True
        args._last_model_call_id = "call-reference-stretch"
        args._last_model_selected_candidate_id = "candidate-reference-stretch"
        args._last_model_path_sha256 = "d" * 64
        return SimpleNamespace(
            north_m=1.72,
            east_m=0.0,
            down_m=-1.0,
            yaw_deg=0.0,
        )

    executor._apply_local_safety = apply_authorized_local_target

    class Client:
        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=1.72,
                east_m=0.0,
                down_m=-1.0,
                north_m_s=0.0,
                east_m_s=0.0,
                down_m_s=0.0,
            )

    args = SimpleNamespace(
        run_dir=tmp_path,
        tracking_telemetry_timeout_seconds=0.1,
        tracking_lag_limit_m=0.145,
        tracking_rejoin_tolerance_m=0.1,
        model_progress_slack_m=0.2,
        _model_goal_progress_state={
            "navigation_goal_id": "goal-reference-stretch",
            "initial_goal_distance_m": 2.0,
            "initial_planned_goal_distance_m": 2.0,
            # A curved reference can have more arc length than the direct
            # reduction in goal distance.  This historical value must not
            # prevent release once vehicle and reference have converged.
            "accepted_schedule_distance_m": 2.108,
            "last_advanced_planned_ned": (2.0, 0.0, -1.0),
        },
    )
    result = asyncio.run(
        executor._tracking_gate_tick(
            args=args,
            base=SimpleNamespace(),
            client=Client(),
            planned_setpoint=SimpleNamespace(
                north_m=2.0,
                east_m=0.0,
                down_m=-1.0,
                yaw_deg=0.0,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[0.0, 0.0, 0.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
            phase_path=tmp_path / "phase.json",
            schedule_index=237,
            sample_now=True,
            recovery_active=False,
            context={
                "navigation_goal_id": "goal-reference-stretch",
                "navigation_goal_position_m": {"x": 0.0, "y": 2.0, "z": 1.2},
            },
        )
    )

    assert result[0] is True
    assert result[1] == pytest.approx(0.0)
    state = json.loads(
        (tmp_path / "runtime-state" / "closed-loop-tracking.json").read_text(encoding="utf-8")
    )
    assert state["model_goal_distance_m"] == pytest.approx(0.28)
    assert state["planned_goal_distance_m"] == pytest.approx(0.0)
    assert state["model_progress_slack_m"] == pytest.approx(0.2)
    assert state["model_schedule_progress_authorized"] is True
    assert state["schedule_advancement_authorized"] is True


def test_checkpoint_executor_keeps_all_logic_spawn_relative() -> None:
    executor = _load_executor()

    class RawClient:
        def __init__(self) -> None:
            self.command = None

        async def set_position_ned(self, setpoint: Any) -> None:
            self.command = setpoint

        async def sample_position_velocity_ned(self, _timeout_seconds: float) -> Any:
            return SimpleNamespace(
                north_m=9.25,
                east_m=-3.5,
                down_m=-0.15,
                north_m_s=0.1,
                east_m_s=-0.2,
                down_m_s=0.05,
            )

    async def exercise() -> tuple[RawClient, Any]:
        raw = RawClient()
        client = executor.SpawnRelativeOffboardClient(
            raw,
            SimpleNamespace(north_m=9.0, east_m=-4.0, down_m=0.1),
            heading_hold_deg=79.2,
        )
        await client.set_position_ned(
            SimpleNamespace(north_m=2.0, east_m=1.0, down_m=-1.0, yaw_deg=45.0)
        )
        return raw, await client.sample_position_velocity_ned(1.0)

    raw, observed = asyncio.run(exercise())

    assert vars(raw.command) == {
        "north_m": 11.0,
        "east_m": -3.0,
        "down_m": -0.9,
        "yaw_deg": 79.2,
    }
    assert vars(observed) == {
        "north_m": 0.25,
        "east_m": 0.5,
        "down_m": -0.25,
        "north_m_s": 0.1,
        "east_m_s": -0.2,
        "down_m_s": 0.05,
    }


def test_spawn_relative_client_records_fresh_observed_heading_without_resubscribing(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class RawClient:
        def __init__(self) -> None:
            self.timestamp_us = 10
            self.observed_yaw_deg = 8.0
            self.commands: list[Any] = []

        async def set_position_ned(self, setpoint: Any) -> None:
            self.commands.append(setpoint)

        def latest_dynamics_telemetry(self, max_age_seconds: float) -> dict[str, Any]:
            assert max_age_seconds == pytest.approx(1.0)
            return {
                "sources": {
                    "attitude": {
                        "yaw_deg": self.observed_yaw_deg,
                        "sample_age_seconds": 0.02,
                        "timestamp_us": self.timestamp_us,
                    }
                }
            }

    async def exercise() -> dict[str, Any]:
        raw = RawClient()
        evidence_path = tmp_path / "runtime-state" / "px4-heading-tracking.json"
        client = executor.SpawnRelativeOffboardClient(
            raw,
            SimpleNamespace(north_m=0.0, east_m=0.0, down_m=0.0),
            heading_policy="route-tangent-relative",
            heading_evidence_path=evidence_path,
            heading_evidence_flush_interval_seconds=0.0,
        )
        command = SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-1.0, yaw_deg=10.0)
        await client.set_position_ned(command)
        await client.set_position_ned(command)
        raw.timestamp_us = 20
        raw.observed_yaw_deg = 20.0
        command.yaw_deg = 25.0
        await client.set_position_ned(command)
        evidence = await client.flush_heading_tracking_evidence()
        assert json.loads(evidence_path.read_text(encoding="utf-8")) == evidence
        return evidence

    evidence = asyncio.run(exercise())

    assert evidence["status"] == "observed"
    assert evidence["policy"] == "route-tangent-relative"
    assert evidence["sample_count"] == 2
    assert evidence["absolute_error_deg"]["mean"] == pytest.approx(3.5)
    assert evidence["absolute_error_deg"]["maximum"] == pytest.approx(5.0)
    assert evidence["within_15_deg_count"] == 2
    assert evidence["writer_issue"] is None


def test_spawn_relative_heading_evidence_writer_failure_does_not_interrupt_control(
    tmp_path: Path,
) -> None:
    executor = _load_executor()

    class RawClient:
        async def set_position_ned(self, _setpoint: Any) -> None:
            return None

        def latest_dynamics_telemetry(self, _max_age_seconds: float) -> dict[str, Any]:
            return {
                "sources": {
                    "attitude": {
                        "yaw_deg": 4.0,
                        "sample_age_seconds": 0.01,
                        "timestamp_us": 1,
                    }
                }
            }

    async def exercise() -> dict[str, Any]:
        occupied = tmp_path / "occupied"
        occupied.write_text("not-a-directory", encoding="utf-8")
        client = executor.SpawnRelativeOffboardClient(
            RawClient(),
            SimpleNamespace(north_m=0.0, east_m=0.0, down_m=0.0),
            heading_evidence_path=occupied / "heading.json",
            heading_evidence_flush_interval_seconds=0.0,
        )
        await client.set_position_ned(
            SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-1.0, yaw_deg=5.0)
        )
        return await client.flush_heading_tracking_evidence()

    evidence = asyncio.run(exercise())

    assert evidence["sample_count"] == 1
    assert evidence["writer_issue"] in {"FileExistsError", "NotADirectoryError"}


def _message() -> RuntimeUserMessage:
    return RuntimeUserMessage(
        message_id="runtime-msg-" + "a" * 32,
        conversation_id="conversation-a",
        mission_id="mission-" + "b" * 32,
        plan_revision_id="plan-" + "c" * 32,
        contract_id="mission-contract-a",
        execution_id="execution-" + "d" * 32,
        text="运行时操作",
        submitted_at=datetime.now(UTC),
    )


def _ack(message: RuntimeUserMessage) -> RuntimeHoldAcknowledgement:
    now = datetime.now(UTC)
    return RuntimeHoldAcknowledgement(
        message_sha256=sha256_json(message),
        message_id=message.message_id,
        execution_id=message.execution_id,
        interrupted_phase="TRACK",
        schedule_index=7,
        detected_at=now,
        detection_latency_ms=20,
        stable_at=now,
        stabilization_latency_ms=900,
        frozen_command_ned_m=Vector3(x=1, y=2, z=-3),
        hold_command_ned_m=Vector3(x=1, y=2, z=-3),
        observed_position_ned_m=Vector3(x=1, y=2, z=-3),
        observed_velocity_ned_mps=Vector3(x=0, y=0, z=0),
        position_error_m=0,
        speed_mps=0,
        deterministic_gates={
            "telemetry_finite": True,
            "position_stable": True,
            "velocity_stable": True,
            "old_plan_inhibited": True,
        },
    )


def _decision(
    message: RuntimeUserMessage,
    acknowledgement: RuntimeHoldAcknowledgement,
    *,
    action: str,
    parameters: dict[str, Any],
) -> RuntimeInterruptionDecision:
    return RuntimeInterruptionDecision(
        message_sha256=sha256_json(message),
        hold_ack_sha256=sha256_json(acknowledgement),
        classification=RuntimeMessageClassification(
            message_kind="motion_adjustment",
            requested_action=action,  # type: ignore[arg-type]
            requires_plan_revision=False,
            summary="Execute a bounded command.",
            parameters=parameters,
        ),
        model_call=ModelCallRecord(
            call_id="model-" + "e" * 24,
            role="runtime_message_classifier",
            attempt=1,
            input_sha256="1" * 64,
            output_sha256="2" * 64,
            output_schema="RuntimeMessageClassification",
            provider="test",
            model="test",
            latency_ms=1,
            created_at=datetime.now(UTC),
        ),
        authorized_action="apply_command",
        authorization_gates={"stable_hold": True},
        decision_reason="Stable hold permits the command.",
        amendment_directive=RuntimeAmendmentDirective(
            action=action,
            parameters=parameters,
            requires_plan_revision=False,
        ),
    )


class _Base:
    @staticmethod
    def _raise_if_external_abort_requested(_: Path) -> None:
        return None


class _Client:
    def __init__(self) -> None:
        self.hold_count = 0

    async def set_position_ned(self, _: Any) -> None:
        self.hold_count += 1

    async def execute_camera_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"confirmed": True, "transport": "test-camera", **parameters}

    async def execute_payload_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"confirmed": True, "transport": "test-payload", **parameters}

    async def execute_avoidance_command(self, enabled: bool) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"confirmed": True, "transport": "test-param", "enabled": enabled}


class _UnconfirmedClient(_Client):
    async def execute_camera_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"confirmed": False, "transport": "test-camera", **parameters}


def test_runtime_hold_tick_refreshes_px4_identity_telemetry(tmp_path: Path) -> None:
    executor = _load_executor()

    class Base:
        @staticmethod
        def _raise_if_external_abort_requested(_: Path) -> None:
            return None

        @staticmethod
        async def _await_with_setpoint_keepalive(
            _client: Any,
            operation: Any,
            **_: Any,
        ) -> Any:
            return await operation

    class Client(_Client):
        async def sample_position_velocity_ned(self, _: float) -> Any:
            return SimpleNamespace(
                north_m=2.0,
                east_m=3.0,
                down_m=-1.5,
                north_m_s=0.1,
                east_m_s=0.2,
                down_m_s=0.0,
            )

        def latest_dynamics_telemetry(self, _: float) -> None:
            return None

    client = Client()
    observed = asyncio.run(
        executor._runtime_hold_tick(
            base=Base,
            client=client,
            hold_setpoint=SimpleNamespace(north_m=2.0, east_m=3.0, down_m=-1.5),
            abort_file=tmp_path / "abort.json",
            rate_hz=20.0,
            telemetry_args=SimpleNamespace(
                tracking_telemetry_timeout_seconds=0.1,
                tracking_telemetry_recovery_timeout_seconds=0.2,
                run_dir=tmp_path,
            ),
            coordinate_contract=Px4CoordinateContract(
                model_root_world_enu_m=[10.0, 20.0, 1.0],
                collision_center_offset_model_m=[0.0, 0.0, 0.2],
            ),
        )
    )

    telemetry = json.loads(
        (tmp_path / "runtime-state" / "px4-identity-telemetry.json").read_text(encoding="utf-8")
    )
    assert observed.north_m == 2.0
    assert client.hold_count == 1
    assert telemetry["observed_world_collision_center_m"] == {
        "x": 13.0,
        "y": 22.0,
        "z": 2.7,
    }


@pytest.mark.parametrize(
    ("action", "parameters"),
    [
        ("camera_control", {"command": "take_photo", "component_id": 100}),
        (
            "payload_control",
            {
                "operation": "detach",
                "protocol": "gazebo-transport",
                "topic": "/model/my_drone/payload/detach",
                "output_topic": "/model/my_drone/payload/state",
            },
        ),
        ("set_avoidance", {"enabled": True}),
    ],
)
def test_runtime_command_is_hash_bound_executed_and_adopted(
    tmp_path: Path, action: str, parameters: dict[str, Any]
) -> None:
    executor = _load_executor()
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(message, acknowledgement, action=action, parameters=parameters)
    command = build_runtime_command(
        message=message,
        acknowledgement=acknowledgement,
        decision=decision,
        # Disposable simulated endpoint: production gets this binding from the
        # prepared, hash-bound action contract compiled from the selected asset.
        prepared=SimpleNamespace(
            contract=SimpleNamespace(contract_id=message.contract_id),
            runtime_actions=SimpleNamespace(
                contract_id=message.contract_id,
                steps=[SimpleNamespace(driver="gazebo-payload", authority="actuate",
                                       parameters=dict(parameters))],
            ),
        ),
    )
    control_dir = tmp_path / "runtime-control"
    command_path = control_dir / "commands" / f"{message.message_id}.json"
    command_path.parent.mkdir(parents=True)
    command_path.write_text(command.model_dump_json(indent=2), encoding="utf-8")
    interruption = executor.RuntimeInterruptDetected(
        message,
        control_dir / "claimed" / f"{message.message_id}.json",
        datetime.now(UTC),
    )
    client = _Client()
    hold = SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-3.0)

    loaded = asyncio.run(
        executor._wait_runtime_command(
            base=_Base,
            client=client,
            hold_setpoint=hold,
            interruption=interruption,
            acknowledgement=acknowledgement,
            decision=decision,
            control_dir=control_dir,
            abort_file=tmp_path / "abort.json",
            rate_hz=20.0,
            timeout_seconds=1.0,
        )
    )
    outcome = asyncio.run(
        executor._execute_runtime_command(
            base=_Base,
            client=client,
            hold_setpoint=hold,
            interruption=interruption,
            command=loaded,
            control_dir=control_dir,
            abort_file=tmp_path / "abort.json",
            rate_hz=20.0,
            timeout_seconds=1.0,
        )
    )

    assert outcome == "resume_original"
    assert client.hold_count >= 2
    adoption = RuntimeCommandAdoption.model_validate_json(
        (control_dir / "adoptions" / f"{message.message_id}.json").read_text(encoding="utf-8")
    )
    assert adoption.command_sha256 == sha256_json(command)
    assert adoption.observed_result["confirmed"] is True
    assert (control_dir / "command-results" / f"{message.message_id}.json").is_file()


def test_invalid_runtime_command_never_reaches_executor() -> None:
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(
        message,
        acknowledgement,
        action="camera_control",
        parameters={"command": "delete_all", "component_id": 100},
    )
    with pytest.raises(Exception, match="command_parameters_valid"):
        build_runtime_command(
            message=message,
            acknowledgement=acknowledgement,
            decision=decision,
        )


def test_unconfirmed_device_readback_fails_closed_and_writes_evidence(tmp_path: Path) -> None:
    executor = _load_executor()
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(
        message,
        acknowledgement,
        action="camera_control",
        parameters={"command": "take_photo", "component_id": 100},
    )
    command = build_runtime_command(
        message=message,
        acknowledgement=acknowledgement,
        decision=decision,
    )
    control_dir = tmp_path / "runtime-control"
    interruption = executor.RuntimeInterruptDetected(
        message,
        control_dir / "claimed" / f"{message.message_id}.json",
        datetime.now(UTC),
    )

    with pytest.raises(executor.UserDirectedLanding, match="positive device readback"):
        asyncio.run(
            executor._execute_runtime_command(
                base=_Base,
                client=_UnconfirmedClient(),
                hold_setpoint=SimpleNamespace(north_m=1.0, east_m=2.0, down_m=-3.0),
                interruption=interruption,
                command=command,
                control_dir=control_dir,
                abort_file=tmp_path / "abort.json",
                rate_hz=20.0,
                timeout_seconds=1.0,
            )
        )

    failure = json.loads(
        (control_dir / "command-failures" / f"{message.message_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["observed_result"]["confirmed"] is False


def test_depth_worker_publishes_safety_command_before_model_orchestration() -> None:
    """Keep provider/RGB work outside the deterministic command deadline."""

    worker = (Path(__file__).parents[1] / "scripts" / "runtime_depth_safety_worker.py").read_text(
        encoding="utf-8"
    )
    loop = worker[worker.index("    while not stop_requested.is_set():") :]

    command_publish = loop.index("_atomic_json(args.command, command.model_dump")
    early_poll = loop.index("completed_model_cycle = _poll_ready_navigation_cycle(")
    directive = loop.index("model_directive, now_unix_ms = _current_navigation_directive(")
    model_poll = loop.index("completed_model_cycle if early_model_poll else")
    rgb_encode = loop.index("model_image_cache.prepare(")

    assert early_poll < directive < command_publish < model_poll
    assert command_publish < rgb_encode
    assert 'if local_navigation_output_mode == "normalized-body-velocity":' in loop[:early_poll]
    helper = worker[worker.index("def _poll_ready_navigation_cycle("):].split("\ndef ", 1)[0]
    assert "_fresh_perception_can_finalize_model_cycle(" in helper
    assert "current_health = fusion.health(" in helper
    assert "now_monotonic_seconds=time.monotonic()" in helper


def test_waypoint_settle_refresh_uses_damped_local_safety() -> None:
    executor = (Path(__file__).parents[1] / "scripts" / "px4_checkpoint_executor.py").read_text(
        encoding="utf-8"
    )
    settle_refreshes = executor.split("async def live_settle_setpoint_refresh(")[1:]

    assert len(settle_refreshes) == 2
    for refresh in settle_refreshes:
        body = refresh.split("\n        def ", 1)[0]
        assert "tracking_recovery_active=True" in body
        assert "navigation_goal_position_m=" in body
        assert "navigation_goal_id=" in body
        assert '"current-control-setpoint"' not in body


def test_semantic_approach_damping_uses_physical_stopping_envelope() -> None:
    executor = _load_executor()

    assert executor._semantic_approach_damping_required(
        planned_goal_distance_m=0.8,
        planned_speed_mps=0.6,
        maximum_acceleration_mps2=0.8,
        waypoint_position_tolerance_m=0.2,
    )
    assert not executor._semantic_approach_damping_required(
        planned_goal_distance_m=0.81,
        planned_speed_mps=0.6,
        maximum_acceleration_mps2=0.8,
        waypoint_position_tolerance_m=0.2,
    )
    assert executor._semantic_approach_damping_required(
        planned_goal_distance_m=1.249,
        planned_speed_mps=1.2,
        maximum_acceleration_mps2=0.8,
        waypoint_position_tolerance_m=0.2,
    )


def test_depth_worker_rejects_retired_truth_only_identity(tmp_path: Path) -> None:
    worker = _load_depth_worker()
    telemetry = tmp_path / "px4-identity-telemetry.json"
    telemetry.write_text(json.dumps({
        "observed_world_collision_center_m": {"x": 0, "y": 0, "z": 1},
        "updated_at_unix_ms": 1_000,
    }), encoding="utf-8")
    tracker = worker._NativeIdentityTracker()
    with pytest.raises(ValueError):
        tracker.evaluate(telemetry, now_unix_ms=1_000)


def test_depth_worker_does_not_time_shift_estimator_to_fit_truth(tmp_path: Path) -> None:
    from test_native_flight_state import _identity

    worker = _load_depth_worker()
    telemetry = tmp_path / "px4-identity-telemetry.json"
    payload = _identity()
    payload["observed_velocity_ned_mps"] = {
        "north_m_s": -0.0125, "east_m_s": -0.7088, "down_m_s": -0.0314,
    }
    payload["observed_world_collision_center_m"] = {"x": 46.1017, "y": .8983, "z": 1.3666}
    telemetry.write_text(json.dumps(payload), encoding="utf-8")
    tracker = worker._NativeIdentityTracker()
    native = tracker.evaluate(telemetry, now_unix_ms=1_020)
    assert native.position_world_enu_m == Vector3(x=0, y=0, z=0)
    assert native.velocity_world_enu_mps == Vector3(x=-.7088, y=-.0125, z=.0314)
    assert native.observed_at_unix_ms == 1_000


def test_depth_worker_cached_position_expiry_is_not_renewed(tmp_path: Path) -> None:
    from test_native_flight_state import _identity

    worker = _load_depth_worker()
    telemetry = tmp_path / "px4-identity-telemetry.json"
    payload = _identity()
    telemetry.write_text(json.dumps(payload), encoding="utf-8")
    tracker = worker._NativeIdentityTracker()
    assert tracker.evaluate(telemetry, now_unix_ms=1_000)
    payload["updated_at_unix_ms"] = 1_260
    telemetry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)expired"):
        tracker.evaluate(telemetry, now_unix_ms=1_260)
