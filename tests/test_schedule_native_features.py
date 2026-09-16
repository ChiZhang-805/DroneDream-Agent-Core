import pytest
from control_fixtures import complete_feature_snapshot

from dronedream_agent_core.realtime_feature_encoders import refresh_flight_state_features
from dronedream_agent_core.simulation_teacher import teacher_input_deadline


def test_only_measured_new_native_state_is_selected_and_geometry_clock_is_preserved():
    before = complete_feature_snapshot(1000)
    old_dump = before.model_dump(mode="json")
    later = before.encodings[-1].model_copy(deep=True, update={
        "observed_at_unix_ms": 1080, "encoded_at_unix_ms": 1090,
        "source_sha256": "b" * 64,
    })
    result = refresh_flight_state_features(before, later, captured_at_unix_ms=1100)
    assert result.ready_for_control
    assert result.encodings[-1] == later
    assert result.encodings[:2] == before.encodings[:2]
    assert before.model_dump(mode="json") == old_dump
    assert result.snapshot_sha256 != before.snapshot_sha256
    # Geometry still owns the earliest original deadline; refresh cannot renew it.
    assert teacher_input_deadline(result, now_ms=1100) == 1250
    result.encodings[0].features[0] = .5
    result.encodings[-1].features[1] = .8
    assert before.model_dump(mode="json") == old_dump and later.features[1] == 0.


def test_refresh_does_not_make_stale_geometry_or_repeated_state_fresh():
    before = complete_feature_snapshot(1000)
    same = refresh_flight_state_features(before, before.encodings[-1], captured_at_unix_ms=1200)
    assert same.encodings[-1].observed_at_unix_ms == 1000
    assert teacher_input_deadline(same, now_ms=1200) == 1250
    later = before.encodings[-1].model_copy(update={
        "observed_at_unix_ms": 1300, "encoded_at_unix_ms": 1300,
    })
    stale = refresh_flight_state_features(before, later, captured_at_unix_ms=1300)
    assert not stale.ready_for_control
    assert "REQUIRED_ENCODER_STALE:metric-geometry-encoder" in stale.issue_codes


@pytest.mark.parametrize("change, clock", [
    ({"source_ids": ["different-aircraft"]}, 1100),
    ({"feature_contract_sha256": "f" * 64}, 1100),
    ({"observed_at_unix_ms": 999}, 1100),
    ({"encoded_at_unix_ms": 1200}, 1100),
    ({}, 999),
])
def test_wrong_binding_contract_or_clock_cannot_refresh(change, clock):
    before = complete_feature_snapshot(1000)
    new = before.encodings[-1].model_copy(update=change)
    with pytest.raises(ValueError, match="CONTROL_REFRESH_NATIVE_"):
        refresh_flight_state_features(before, new, captured_at_unix_ms=clock)


def test_geometry_cannot_be_substituted_as_native_state():
    before = complete_feature_snapshot(1000)
    with pytest.raises(ValueError, match="SOURCE_OR_CONTRACT_CHANGED"):
        refresh_flight_state_features(before, before.encodings[0], captured_at_unix_ms=1100)
