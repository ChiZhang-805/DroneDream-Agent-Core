import gc
import hashlib
import json
from collections import deque
from dataclasses import replace

import pytest
from test_native_corrections import fixture

from dronedream_agent_core.contracts import RuntimeLocalSafetyCommand, RuntimeLocalSafetyObservation
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.training.capture_archive import (
    MAXIMUM_CAPTURE_BYTES,
    PackedNativeTransition,
    pack_native_transition,
)
from dronedream_agent_core.training.flight_environment import FlightObservation, FlightStep
from dronedream_agent_core.training.native_transition import CapturedNativeTransition
from dronedream_agent_core.training.policy_exchange import TrainingProposal
from dronedream_agent_core.training.px4_environment import Px4GazeboTrainingEnvironment


def capture_fixture(tmp_path):
    _, _, path, _ = fixture(tmp_path)
    row = json.loads(path.read_bytes())
    step = FlightStep.model_validate(row["step"])
    return CapturedNativeTransition(
        FlightObservation.model_validate(row["source_observation"]), step.observation,
        row["source_snapshot"], TrainingProposal.model_validate(row["proposal"]),
        RuntimeLocalSafetyCommand.model_validate(row["command"]),
        ControlApplicationRecord.model_validate(row["application"]), None,
        step.applied_action, False,
        tuple(RuntimeLocalSafetyObservation.model_validate(r)
              for r in row["outcome_receipt"]["observations"]), False,
    )


def test_packing_preserves_exact_source_action_witness_and_deadlines(tmp_path):
    capture = capture_fixture(tmp_path)
    packed = pack_native_transition(capture)
    assert type(packed.content) is bytes and not gc.is_tracked(packed.content)
    assert packed.unpack() == capture
    capture.snapshot["goal_position_m"]["x"] += 3.
    capture.source_observation.sample.state_features[0] += 1.
    assert packed.unpack() != capture
    assert set(vars(packed)) == {"content", "sha256"}


def test_previous_actual_application_and_terminal_flags_survive_packing(tmp_path):
    capture = capture_fixture(tmp_path)
    capture = replace(capture, previous_application=capture.application,
                      truncated=True, deadline_intervened=True)
    assert pack_native_transition(capture).unpack() == capture


def test_changed_or_invalid_capture_is_not_silently_recovered(tmp_path):
    packed = pack_native_transition(capture_fixture(tmp_path))
    with pytest.raises(ValueError, match="CONTENT_CHANGED"):
        replace(packed, content=packed.content + b" ").unpack()
    invalid = b'{}'
    with pytest.raises(ValueError):
        PackedNativeTransition(invalid, hashlib.sha256(invalid).hexdigest()).unpack()
    with pytest.raises(ValueError, match="TYPE_INVALID"):
        pack_native_transition({})


@pytest.mark.parametrize("content", [b"", b" " * (MAXIMUM_CAPTURE_BYTES + 1), "{}"],
                         ids=["empty", "oversized", "text"])
def test_unpack_rejects_unbounded_or_nonbyte_content(content):
    with pytest.raises(ValueError, match="SIZE_INVALID"):
        PackedNativeTransition(content, "a" * 64).unpack()


def test_pack_bound_does_not_disable_memory_collection(tmp_path):
    capture = capture_fixture(tmp_path)
    capture.snapshot["oversized"] = "a" * MAXIMUM_CAPTURE_BYTES
    was_enabled = gc.isenabled()
    with pytest.raises(ValueError, match="SIZE_INVALID"):
        pack_native_transition(capture)
    assert gc.isenabled() is was_enabled


def test_episode_byte_limit_fails_without_dropping_earlier_captures(tmp_path, monkeypatch):
    capture = capture_fixture(tmp_path)
    env = object.__new__(Px4GazeboTrainingEnvironment)
    env._interface_timings = deque(maxlen=512)
    first = env.retain_capture(capture)
    count = env._retained_capture_bytes
    monkeypatch.setattr("dronedream_agent_core.training.px4_environment.MAXIMUM_ARCHIVE_BYTES",
                        count * 2 - 1)
    with pytest.raises(ValueError, match="ARCHIVE_FULL"):
        env.retain_capture(capture)
    assert env._retained_capture_bytes == count
    assert first.unpack() == capture
    assert len(env._interface_timings) == 1


def test_unpack_and_verification_still_require_confirmed_grounding(tmp_path):
    env = object.__new__(Px4GazeboTrainingEnvironment)
    env._quiesced, env._process = False, object()
    with pytest.raises(ValueError, match="CONFIRMED_QUIESCENCE"):
        env.finalize_captures([PackedNativeTransition(b"invalid", "a" * 64)])
    env._quiesced, env._process = True, None
    with pytest.raises(ValueError, match="CAPTURE_NOT_PACKED"):
        env.finalize_captures([capture_fixture(tmp_path)])
