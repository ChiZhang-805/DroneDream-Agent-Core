"""Production boundary integration with injected timing faults, not flight proof."""
import hashlib
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from control_fixtures import complete_feature_snapshot
from test_perception_runtime import _frame, _world
from test_runtime_local_safety import _vehicle

import dronedream_agent_core.perception_runtime as runtime
from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation, Vector3
from dronedream_agent_core.realtime_feature_encoders import (
    RealtimeFeatureEncoding,
    fuse_realtime_features,
)
from dronedream_agent_core.runtime_local_safety import evaluate_runtime_local_safety


def coordinator(monkeypatch):
    fusion = runtime.RuntimePerceptionFusion(world=_world(),
        accepted_sensor_ids={"front-lidar"}, minimum_rays_per_frame=8)
    fusion.ingest(_frame(), now_unix_ms=1020)
    port = SimpleNamespace(prime_multimodal=Mock(), call=Mock())
    instance = runtime.EventDrivenIndoorNavigationCoordinator(fusion=fusion, port=port,
        required_clearance_m=0., control_output_mode="normalized-body-velocity",
        include_candidate_paths=False)
    executor = Mock()
    executor.submit.return_value = Future()
    instance._executor.shutdown(wait=True)
    monkeypatch.setattr(instance, "_executor", executor)
    return instance, port, executor


def schedule(instance, **overrides):
    args = dict(goal_position_m=Vector3(x=3.75, y=.25, z=.25), now_unix_ms=1020,
                trigger="initial",
                realtime_feature_snapshot=complete_feature_snapshot().model_dump())
    args.update(overrides)
    return instance.schedule(**args)


@pytest.mark.parametrize("basis,source,reason", [
    ("host-receive-not-hardware-exposure", 1000, "VISUAL_CONTROL_SOURCE_NOT_READY"),
    ("simulation-scene-capture", 800, "VISUAL_CONTROL_SOURCE_NOT_READY"),
    ("simulation-scene-capture", 1100, "VISUAL_CONTROL_SOURCE_NOT_READY"),
])
def test_expired_unknown_future_images_do_not_prime_or_invoke(monkeypatch, basis, source, reason):
    instance, port, executor = coordinator(monkeypatch)
    try:
        result = schedule(instance, multimodal=[dict(timestamp_basis=basis,
            scene_source_unix_ns=source*1_000_000, host_received_unix_ns=1010*1_000_000)])
        assert result.hold_reason == reason
        port.prime_multimodal.assert_not_called()
        executor.submit.assert_not_called()
    finally:
        instance.close()


def test_live_image_has_its_own_older_deadline_even_with_new_state(monkeypatch):
    instance, port, executor = coordinator(monkeypatch)
    frame = b"owned-frame"
    try:
        assert schedule(instance, multimodal=[dict(kind="image-file", content_bytes=frame,
            content_sha256=hashlib.sha256(frame).hexdigest(),
            timestamp_basis="simulation-scene-capture", scene_source_unix_ns=950_000_000,
            host_received_unix_ns=1010_000_000)]) is None
        port.prime_multimodal.assert_called_once()
        executor.submit.assert_called_once()
        evidence = executor.submit.call_args.kwargs["visual_evidence"][0]
        assert evidence["source_observed_at_unix_ms"] == 950
        assert evidence["control_deadline_unix_ms"] == 1200
        assert instance.continuous_request_remaining_ms(now_unix_ms=1020) == 180
        assert instance.continuous_request_remaining_ms(now_unix_ms=1200) == 0
    finally:
        instance.close()


@pytest.mark.parametrize("stage", ["compilation", "inference"])
def test_time_spent_before_or_inside_inference_cannot_be_renewed(monkeypatch, stage):
    instance, port, executor = coordinator(monkeypatch)
    clock = [1.]
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    try:
        assert schedule(instance) is None
        args = executor.submit.call_args.kwargs

        def compile_snapshot(_):
            if stage == "compilation":
                clock[0] += .2
            return {"snapshot_sha256": "a"*64}

        def invoke(**_):
            clock[0] += .2
            return SimpleNamespace(record={"actual-inference": True}), None

        invocation = Mock(side_effect=invoke)
        monkeypatch.setattr(runtime, "compile_navigation_snapshot", compile_snapshot)
        monkeypatch.setattr(runtime, "request_text_navigation_decision", invocation)
        result = runtime._compile_and_request_navigation_decision(**args)
        assert result.model_result is None
        assert result.failure_reason == ("CONTROL_SOURCE_EXPIRED_DURING_PREPARATION"
            if stage == "compilation" else "CONTROL_SOURCE_EXPIRED_DURING_INFERENCE")
        assert invocation.call_count == (0 if stage == "compilation" else 1)
        assert result.discarded_call_record == (
            None if stage == "compilation" else {"actual-inference": True})
    finally:
        instance.close()


def test_rotational_aging_uses_original_geometry_not_refreshed_state():
    source = complete_feature_snapshot()
    encodings = []
    for encoding in source.encodings:
        data = encoding.model_dump()
        if encoding.encoder_role == "flight-state-encoder":
            data["features"][11] = .2  # 1 rad/s => 174 ms, not 250 ms.
            data.update(observed_at_unix_ms=1100, encoded_at_unix_ms=1100)
        encodings.append(RealtimeFeatureEncoding.model_validate(data))
    snapshot = fuse_realtime_features(encodings, captured_at_unix_ms=1100)
    assert snapshot.control_deadline_unix_ms(now_unix_ms=1100) == 1174
    assert snapshot.control_deadline_unix_ms(now_unix_ms=1180) == 0


def observation(**updates):
    data = dict(sequence=1, observed_at_unix_ms=10_000, source="onboard",
        stream_healthy=True, stream_age_seconds=.01, localization_covariance_m2=.0001,
        current_position_m=Vector3(x=0., y=0., z=1.),
        current_velocity_mps=Vector3(x=.1, y=0., z=0.),
        target_position_m=Vector3(x=1., y=0., z=1.))
    data.update(updates)
    return RuntimeLocalSafetyObservation(**data)


def test_real_safety_adapter_binds_motion_to_source_not_evaluation_time():
    source = observation()
    command = evaluate_runtime_local_safety(observation=source, vehicle=_vehicle(),
        static_primitives=[], required_clearance_m=.1, generated_at_unix_ms=10_020)
    assert command.decision.action != "hold"
    assert command.valid_until_unix_ms == 10_250
    # Same source, computational stall: the old observation cannot justify
    # motion despite its old, cached stream_healthy flag and small age field.
    expired = evaluate_runtime_local_safety(observation=source, vehicle=_vehicle(),
        static_primitives=[], required_clearance_m=.1, generated_at_unix_ms=10_220)
    assert expired.decision.action == "hold"
    assert expired.command_position_m == source.current_position_m
    assert "OBSERVATION_CONTROL_BUDGET_EXHAUSTED" in expired.decision.issue_codes
    assert expired.decision.braking_prediction_velocity_mps is not None
    assert expired.observation_budget.reason == "processing-or-response-budget-exhausted"
    assert "PERCEPTION_STREAM_UNHEALTHY" not in expired.decision.issue_codes
    assert source.stream_healthy and source.stream_age_seconds == .01


def test_action_specific_margin_veto_preserves_threat_forecast(monkeypatch):
    import dronedream_agent_core.runtime_local_safety as safety

    real = safety.predictive_safety_decision
    source = observation()

    def tight_forecast(request, primitives):
        result = real(request, primitives)
        if result.action != "hold":
            result.minimum_predicted_clearance_m = .13  # Only 3 cm residual slack.
        return result

    monkeypatch.setattr(safety, "predictive_safety_decision", tight_forecast)
    command = safety.evaluate_runtime_local_safety(observation=source, vehicle=_vehicle(),
        static_primitives=[], required_clearance_m=.1, generated_at_unix_ms=10_020)
    assert command.decision.action == "hold"
    assert "OBSERVATION_CONTROL_BUDGET_EXHAUSTED" in command.decision.issue_codes
    assert command.decision.predicted_path_m  # Not erased or claimed measured safe.
    assert command.observation_budget.reason == "no-temporal-clearance-margin"
    assert command.observation_budget.clearance_margin_m == pytest.approx(.03)
    assert command.observation_budget.uncertainty_margin_m == pytest.approx(.03)
    assert "PERCEPTION_STREAM_UNHEALTHY" not in command.decision.issue_codes


def test_observation_budget_survives_wire_round_trip_and_cannot_extend_motion():
    from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand

    command = evaluate_runtime_local_safety(observation=observation(), vehicle=_vehicle(),
        static_primitives=[], required_clearance_m=.1, generated_at_unix_ms=10_020)
    wire = command.model_dump(mode="json")
    assert RuntimeLocalSafetyCommand.model_validate(wire) == command
    wire["valid_until_unix_ms"] = command.observation_budget.control_deadline_unix_ms + 1
    with pytest.raises(ValueError, match="exceeds its observation budget"):
        RuntimeLocalSafetyCommand.model_validate(wire)
    wire = command.model_dump(mode="json")
    wire["observation_budget"]["disposition"] = "context-only"
    with pytest.raises(ValueError, match="exceeds its observation budget"):
        RuntimeLocalSafetyCommand.model_validate(wire)


def test_absent_budget_does_not_rewrite_historical_command_identity():
    from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand
    from dronedream_agent_core.hashing import sha256_json

    command = evaluate_runtime_local_safety(observation=observation(), vehicle=_vehicle(),
        static_primitives=[], required_clearance_m=.1, generated_at_unix_ms=10_020)
    wire = command.model_dump(mode="json")
    wire.pop("observation_budget")
    restored = RuntimeLocalSafetyCommand.model_validate(wire)
    assert "observation_budget" not in restored.model_dump(mode="json")
    assert sha256_json(restored) == sha256_json(wire)


@pytest.mark.parametrize("confidence,age", [(.1, .01), (.95, 1.1), (.1, 10.)])
def test_lost_track_cannot_be_dropped_by_safety_even_if_caller_claims_healthy(confidence, age):
    from test_dynamic_safety import _request

    from dronedream_agent_core.contracts import DynamicObstacleObservation
    from dronedream_agent_core.dynamic_safety import predictive_safety_decision

    obstacle = DynamicObstacleObservation(obstacle_id="retained-person",
        position_m=Vector3(x=.5, y=0., z=1.), velocity_mps=Vector3(x=.1, y=0., z=0.),
        radius_m=.3, height_m=1.7, confidence=confidence, age_seconds=age)
    decision = predictive_safety_decision(_request(obstacles=[obstacle]), [])
    assert decision.action == "hold"
    assert decision.threat_obstacle_id == "retained-person"
    assert decision.minimum_predicted_clearance_m < .1
    assert "DYNAMIC_TRACK_EVIDENCE_UNRELIABLE" in decision.issue_codes
