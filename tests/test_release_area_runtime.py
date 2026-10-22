from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

from dronedream_agent_core.contracts import (
    PredictiveSafetyDecision,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    Vector3,
)
from dronedream_agent_core.hashing import sha256_json


def _executor():
    path = Path(__file__).parents[1] / "scripts" / "px4_checkpoint_executor.py"
    spec = importlib.util.spec_from_file_location("test_release_area_executor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_area_uses_current_onboard_clearance(tmp_path: Path) -> None:
    executor = _executor()
    now = int(time.time() * 1000)
    target = Vector3(x=12.0, y=-4.0, z=2.0)
    observation = RuntimeLocalSafetyObservation(
        sequence=17,
        observed_at_unix_ms=now,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.02,
        localization_covariance_m2=0.02,
        current_position_m=Vector3(x=12.04, y=-4.03, z=1.98),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=target,
    )
    command = RuntimeLocalSafetyCommand(
        observation_sha256=sha256_json(observation),
        observation_sequence=observation.sequence,
        generated_at_unix_ms=now,
        valid_until_unix_ms=now + 1_500,
        source="onboard",
        evaluated_target_position_m=target,
        navigation_goal_id="delivery-zone",
        command_position_m=target,
        decision=PredictiveSafetyDecision(
            action="continue",
            selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
            selected_yaw_rate_dps=0.0,
            control_source="route-target",
            predicted_path_m=[target],
            minimum_predicted_clearance_m=2.4,
            time_to_minimum_clearance_seconds=0.0,
            evaluated_candidate_count=1,
        ),
    )
    (tmp_path / "depth-local-safety-observation.json").write_text(
        observation.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "depth-local-safety-command.json").write_text(
        command.model_dump_json(), encoding="utf-8"
    )

    result = asyncio.run(
        executor._verify_release_area_from_local_safety(
            run_dir=tmp_path,
            parameters={
                "operation": "verify-release-area",
                "minimum_clearance_m": 0.75,
                "maximum_age_seconds": 1.0,
                "maximum_position_error_m": 0.35,
            },
            setpoint=object(),
            setpoint_refresh=None,
        )
    )

    assert result["confirmed"] is True
    assert result["transport"] == "onboard-local-safety"
    assert result["evidence"] == ["release area clear"]
    assert result["observation_sequence"] == 17
    assert result["minimum_predicted_clearance_m"] == 2.4


def test_release_area_reads_atomic_pair_from_live_history(tmp_path: Path) -> None:
    executor = _executor()
    now = int(time.time() * 1000)
    target = Vector3(x=8.0, y=3.0, z=2.0)
    observation = RuntimeLocalSafetyObservation(
        sequence=22,
        observed_at_unix_ms=now,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.03,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=8.02, y=3.01, z=2.01),
        current_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        target_position_m=target,
    )
    command = RuntimeLocalSafetyCommand(
        observation_sha256=sha256_json(observation),
        observation_sequence=observation.sequence,
        generated_at_unix_ms=now,
        valid_until_unix_ms=now + 1_500,
        source="onboard",
        evaluated_target_position_m=target,
        navigation_goal_id="delivery-zone",
        command_position_m=target,
        decision=PredictiveSafetyDecision(
            action="continue",
            selected_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
            selected_yaw_rate_dps=0.0,
            control_source="route-target",
            predicted_path_m=[target],
            minimum_predicted_clearance_m=1.8,
            time_to_minimum_clearance_seconds=0.0,
            evaluated_candidate_count=1,
        ),
    )
    stale_observation = observation.model_copy(update={"sequence": 21})
    (tmp_path / "depth-local-safety-observation.json").write_text(
        stale_observation.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "depth-local-safety-command.json").write_text(
        command.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "depth-local-safety-history.jsonl").write_text(
        json.dumps(
            {
                "recorded_at_unix_ms": now,
                "observation": observation.model_dump(mode="json"),
                "command": command.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        executor._verify_release_area_from_local_safety(
            run_dir=tmp_path,
            parameters={
                "operation": "verify-release-area",
                "minimum_clearance_m": 0.75,
                "maximum_age_seconds": 1.0,
                "maximum_position_error_m": 0.35,
            },
            setpoint=object(),
            setpoint_refresh=None,
        )
    )

    assert result["confirmed"] is True
    assert result["observation_sequence"] == 22
    assert result["minimum_predicted_clearance_m"] == 1.8
