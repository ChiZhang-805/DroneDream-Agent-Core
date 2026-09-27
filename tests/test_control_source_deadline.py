"""Original action clocks remain stricter than contextual feature retention."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from control_fixtures import complete_feature_snapshot

from dronedream_agent_core.control_timing import LOCAL_DISPATCH_RESERVE_MS
from dronedream_agent_core.perception_runtime import EventDrivenIndoorNavigationCoordinator
from dronedream_agent_core.realtime_feature_encoders import (
    RealtimeFeatureEncoding,
    fuse_realtime_features,
)
from dronedream_agent_core.runtime_scheduling import ReadyControlScheduler
from dronedream_agent_core.simulation_teacher import teacher_input_deadline


def retained_features(*, captured=1200, geometry_age_limit=350):
    source = complete_feature_snapshot()
    encodings = []
    for encoding in source.encodings:
        data = encoding.model_dump()
        if encoding.encoder_role == "flight-state-encoder":
            data.update(observed_at_unix_ms=captured, encoded_at_unix_ms=captured)
        else:
            data["maximum_age_milliseconds"] = (
                geometry_age_limit if encoding.encoder_role == "metric-geometry-encoder" else 500
            )
        encodings.append(RealtimeFeatureEncoding.model_validate(data))
    return fuse_realtime_features(encodings, captured_at_unix_ms=captured)


@pytest.mark.parametrize("now,expected", [(1199, 0), (1200, 1250), (1249, 1250), (1250, 0),
                                         (1251, 0), (1300, 0)])
def test_context_retention_cannot_extend_action_source_deadline(now, expected):
    # At expiry there is no time left to dispatch; retained context is not authority.
    features = retained_features()
    assert features.fresh_at(1300)  # Context is intentionally still usable.
    assert features.control_deadline_unix_ms(now_unix_ms=now) == expected
    assert teacher_input_deadline(features, now_ms=now) == expected


def test_new_state_and_fusion_clock_do_not_refresh_geometry_action_authority():
    earlier, refreshed = retained_features(), retained_features(captured=1240)
    assert earlier.snapshot_sha256 != refreshed.snapshot_sha256
    assert earlier.control_deadline_unix_ms(now_unix_ms=1240) == 1250
    assert refreshed.control_deadline_unix_ms(now_unix_ms=1240) == 1250
    assert refreshed.encodings[0].observed_at_unix_ms == 1000


def test_stricter_sensor_specific_limit_is_not_raised_to_control_limit():
    features = retained_features(captured=1100, geometry_age_limit=150)
    assert features.control_deadline_unix_ms(now_unix_ms=1100) == 1150
    assert teacher_input_deadline(features, now_ms=1100) == 1150


def test_missing_required_encoding_has_no_control_deadline():
    source = complete_feature_snapshot()
    incomplete = fuse_realtime_features(source.encodings[:-1], captured_at_unix_ms=1000)
    assert not incomplete.ready_for_control
    assert incomplete.control_deadline_unix_ms(now_unix_ms=1000) == 0


def pending_coordinator(*, submitted=1000, age=.25, done=False):
    coordinator = object.__new__(EventDrivenIndoorNavigationCoordinator)
    coordinator.control_output_mode = "normalized-body-velocity"
    future = Future()
    if done:
        future.set_result(None)
    coordinator._pending = SimpleNamespace(
        submitted_at_unix_ms=submitted, maximum_decision_age_seconds=age, future=future,
    )
    return coordinator


@pytest.mark.parametrize("done", [False, True])
def test_pending_budget_retains_original_request_time_even_after_completion(done):
    coordinator = pending_coordinator(submitted=1100, age=.08, done=done)
    pending = coordinator._pending
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1099) == 0
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1100) == 80
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1130) == 50
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1180) == 0
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1300) == 0
    assert coordinator._pending is pending
    coordinator.control_output_mode = "legacy-candidate-selection"
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1100) == 0
    coordinator.control_output_mode = "normalized-body-velocity"
    coordinator._pending = None
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1100) == 0


@pytest.mark.parametrize("clock", [-1, 1.0, True, None])
def test_invalid_clock_cannot_create_a_scheduling_budget(clock):
    for clock_consumer in (retained_features().control_deadline_unix_ms,
                           pending_coordinator().continuous_request_remaining_ms):
        with pytest.raises(ValueError, match="CONTROL_DEADLINE_CLOCK_INVALID"):
            clock_consumer(now_unix_ms=clock)


@pytest.mark.parametrize("now", [1210, 1211, 1240])
def test_wait_for_pending_model_cannot_consume_new_scan_time_after_action_lease(now):
    features = retained_features(captured=1100)
    coordinator = pending_coordinator()
    # The old freshness check passed even when less than 40 ms of the action
    # budget remained. This used to allow another 35 ms wait for that reply.
    assert features.fresh_at(now + LOCAL_DISPATCH_RESERVE_MS)
    budget_ready = (
        features.control_deadline_unix_ms(now_unix_ms=now) - now >= LOCAL_DISPATCH_RESERVE_MS
        and coordinator.continuous_request_remaining_ms(now_unix_ms=now)
        >= LOCAL_DISPATCH_RESERVE_MS
    )
    scheduler = ReadyControlScheduler()
    waiting = scheduler.await_pending_result(
        now_monotonic=2., depth_sequence=1, continuous_pending=True, result_ready=False,
        perception_healthy=True, features_retain_dispatch_budget=budget_ready,
        depth_processing_failed=False, fault_active=False,
    )
    assert waiting is (now == 1210)


def test_fresh_features_cannot_promote_older_pending_request():
    features = complete_feature_snapshot(1200)
    coordinator = pending_coordinator(submitted=1000, age=.20)
    assert features.control_deadline_unix_ms(now_unix_ms=1200) == 1450
    assert coordinator.continuous_request_remaining_ms(now_unix_ms=1200) == 0
