"""Execute a prepared mission in real Gazebo/PX4 with the ROS plugin host active."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from dronedream_agent_app.asset_runtime_resolver import (
    assert_development_map_binding,
    resolve_development_mission_input,
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.contracts import MapAsset, PreparedMission, VehicleAsset
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.model_harness.boundary import (
    HarnessRuntimeEnvelope,
    compile_execution_authority,
)
from dronedream_agent_core.ros_workspace_provenance import (
    verify_ros_workspace_provenance,
)
from dronedream_agent_core.runtime_interrupt import submit_runtime_message
from dronedream_agent_core.windows_paths import windows_drive_path_to_wsl

# The child executor owns the motion deadline (at most six hours), semantic
# no-progress watchdog, recovery windows, and landing timeout.  This outer
# process limit is deliberately larger so it remains a final stuck-process
# guard instead of terminating a healthy low-real-time-factor depth run.
DEFAULT_RUNTIME_TIMEOUT_SECONDS = 28_800.0
MINIMUM_RUNTIME_TIMEOUT_SECONDS = 900.0
MAXIMUM_RUNTIME_TIMEOUT_SECONDS = 43_200.0
OUTER_RUNTIME_LANDING_GRACE_SECONDS = 900.0
MINIMUM_ROUTE_TANGENT_HEADING_SAMPLE_COUNT = 20
MAXIMUM_ROUTE_TANGENT_P99_HEADING_ERROR_DEG = 45.0
REQUIRED_DYNAMICS_TELEMETRY_RATES_HZ = {
    "position_velocity": 50.0,
    "imu": 50.0,
    "attitude": 50.0,
    "odometry": 50.0,
    "battery": 2.0,
}
MINIMUM_DEVELOPMENT_PAYLOAD_MOTION_SAMPLES = 50


# 功能：
#   使用与桌面入口相同的盘符映射规则，支持扩展长度路径并拒绝不支持的命名空间。
# 输入：
#   path：本机任务或资源路径。
# 输出：
#   mapped：供 WSL 使用的绝对路径。
def _wsl_path(path: Path) -> str:
    mapped = windows_drive_path_to_wsl(str(path.resolve()))
    return mapped


def _secret_lines(provider: str, model: str, api_key: str) -> str:
    settings = {
        "deepseek": ("DEEPSEEK", "https://api.deepseek.com"),
        "kimi": ("KIMI", "https://api.moonshot.ai/v1"),
        "openai": ("OPENAI", "https://api.openai.com/v1"),
    }
    prefix, base_url = settings[provider]
    values = {
        f"{prefix}_API_KEY": api_key,
        f"{prefix}_BASE_URL": base_url,
        f"{prefix}_MODEL": model,
    }
    if provider == "openai":
        values["OPENAI_API_STYLE"] = "chat-completions"
    return "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())


def _provider_key_env(provider: str) -> str:
    return {
        "deepseek": "DEEPSEEK_API_KEY",
        "kimi": "KIMI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[provider]


def _provider_default_model(provider: str) -> str:
    return {
        "deepseek": "deepseek-v4-flash",
        "kimi": "kimi-k2.6",
        "openai": "gpt-5.4",
    }[provider]


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _is_applied_model_motion(
    cycle: dict[str, Any],
    *,
    call: dict[str, Any] | None,
) -> bool:
    """Recognize both legacy path leases and current continuous pilot control.

    A direct pilot-control decision intentionally has no selected candidate.  It
    is applied only when the cycle carries a controller target and the bound
    model-call trace proves that four-axis pilot control was actually emitted.
    This prevents qualification tooling from silently treating the current
    control architecture as a fail-closed hold merely because it no longer
    chooses one of the old discrete candidate endpoints.
    """

    if not isinstance(cycle.get("controller_target_m"), dict):
        return False
    action = cycle.get("model_action")
    if action == "select-candidate":
        return cycle.get("selected_candidate_id") is not None
    if action != "pilot-control" or not isinstance(call, dict):
        return False
    trace = call.get("local_expert_trace")
    return bool(
        isinstance(trace, dict)
        and trace.get("navigation_action") == "pilot-control"
        and isinstance(trace.get("pilot_control"), dict)
    )


def _validated_development_payload_collection_evidence(
    simulation_root: Path,
    *,
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Prove reduced-step, fresh-dynamics payload motion without granting qualification."""

    gates = runtime_evidence.get("gates")
    measurements = runtime_evidence.get("measurements")
    if not isinstance(gates, dict) or not isinstance(measurements, dict):
        raise RuntimeError("development payload collection runtime evidence is incomplete")
    allowed_false_gates = {"development_payload_collection_absent"}
    model_navigation = measurements.get("model_navigation")
    if isinstance(model_navigation, dict) and model_navigation.get("visual_enabled") is False:
        allowed_false_gates.add("model_navigation_visual_frame_recorded")
    multimodal_dataset = measurements.get("multimodal_dataset")
    if (
        isinstance(multimodal_dataset, dict)
        and multimodal_dataset.get("enabled") is True
        and multimodal_dataset.get("semantic_topic") is None
    ):
        allowed_false_gates.add("semantic_supervision_recorded_for_every_sample")
    false_gates = {key for key, value in gates.items() if value is not True}
    if false_gates != allowed_false_gates:
        raise RuntimeError("development payload collection has unrelated failed runtime gates")
    configured = measurements.get("development_payload_collection")
    if (
        not isinstance(configured, dict)
        or configured.get("enabled") is not True
        or configured.get("development_only") is not True
        or configured.get("flight_qualification_granted") is not False
        or configured.get("maximum_controller_step_scale") != 0.2
        or configured.get("fresh_px4_dynamics_required") is not True
    ):
        raise RuntimeError("development payload collection contract is not bound")

    snapshots_path = simulation_root / "model-navigation-snapshots.jsonl"
    cycles_path = simulation_root / "model-navigation-cycles.jsonl"
    calls_path = simulation_root / "model-navigation-model-calls.jsonl"
    for path in (snapshots_path, cycles_path, calls_path):
        if not path.is_file():
            raise RuntimeError("development payload collection model evidence is missing")
    snapshots = {
        str(snapshot.get("snapshot_sha256")): snapshot
        for record in _jsonl_objects(snapshots_path)
        if isinstance((snapshot := record.get("snapshot")), dict)
    }
    calls = {str(record.get("call_id")): record for record in _jsonl_objects(calls_path)}
    payload_cycles = 0
    fresh_payload_cycles = 0
    bounded_motion_cycles = 0
    stale_dynamics_motion_cycles = 0
    oversized_motion_cycles = 0
    for cycle in _jsonl_objects(cycles_path):
        snapshot = snapshots.get(str(cycle.get("snapshot_sha256", "")))
        if not isinstance(snapshot, dict):
            continue
        strategic = snapshot.get("strategic_context")
        payload = strategic.get("payload") if isinstance(strategic, dict) else None
        if not isinstance(payload, dict) or str(payload.get("state", "")).lower() not in {
            "attached",
            "custody-confirmed",
            "loaded-stable",
        }:
            continue
        payload_cycles += 1
        dynamics = payload.get("dynamics")
        dynamics_ready = bool(
            isinstance(dynamics, dict)
            and dynamics.get("available") is True
            and dynamics.get("ready") is True
        )
        if dynamics_ready:
            fresh_payload_cycles += 1
        step_scale = cycle.get("controller_step_scale")
        call = calls.get(str(cycle.get("model_call_id", "")))
        # A model may finish after the active goal changes. The coordinator
        # records its advisory action for audit, but fail-closes with no
        # controller target. That is a HOLD, not applied payload motion.
        if not _is_applied_model_motion(cycle, call=call):
            continue
        trace = call.get("local_expert_trace") if isinstance(call, dict) else None
        trace_scale = trace.get("controller_step_scale") if isinstance(trace, dict) else None
        bounded = bool(
            isinstance(step_scale, int | float)
            and not isinstance(step_scale, bool)
            and float(step_scale) <= 0.2 + 1e-9
            and isinstance(trace_scale, int | float)
            and not isinstance(trace_scale, bool)
            and float(trace_scale) <= 0.2 + 1e-9
        )
        if not dynamics_ready:
            stale_dynamics_motion_cycles += 1
        if not bounded:
            oversized_motion_cycles += 1
        if dynamics_ready and bounded:
            bounded_motion_cycles += 1
    if payload_cycles < MINIMUM_DEVELOPMENT_PAYLOAD_MOTION_SAMPLES:
        raise RuntimeError("development payload collection has too few payload cycles")
    if stale_dynamics_motion_cycles or oversized_motion_cycles:
        raise RuntimeError("development payload collection applied unsafe payload motion")
    if bounded_motion_cycles < MINIMUM_DEVELOPMENT_PAYLOAD_MOTION_SAMPLES:
        raise RuntimeError("development payload collection has too few bounded motion cycles")
    return {
        "status": "verified-development-data",
        "development_only": True,
        "flight_qualification_granted": False,
        "model_navigation_snapshots_sha256": hashlib.sha256(
            snapshots_path.read_bytes()
        ).hexdigest(),
        "model_navigation_cycles_sha256": hashlib.sha256(cycles_path.read_bytes()).hexdigest(),
        "model_navigation_model_calls_sha256": hashlib.sha256(calls_path.read_bytes()).hexdigest(),
        "payload_cycle_count": payload_cycles,
        "fresh_payload_cycle_count": fresh_payload_cycles,
        "bounded_motion_cycle_count": bounded_motion_cycles,
        "maximum_controller_step_scale": 0.2,
        "stale_dynamics_motion_cycle_count": stale_dynamics_motion_cycles,
        "oversized_motion_cycle_count": oversized_motion_cycles,
    }


def _validated_runtime_timeout_seconds(value: float) -> float:
    timeout_seconds = float(value)
    if not MINIMUM_RUNTIME_TIMEOUT_SECONDS <= timeout_seconds <= (MAXIMUM_RUNTIME_TIMEOUT_SECONDS):
        raise ValueError(
            "runtime acceptance timeout must be between "
            f"{MINIMUM_RUNTIME_TIMEOUT_SECONDS:g} and "
            f"{MAXIMUM_RUNTIME_TIMEOUT_SECONDS:g} seconds"
        )
    return timeout_seconds


def _resolve_ros_workspace(
    resource_root: Path,
    override: Path | None,
) -> str:
    """Resolve ROS without silently mixing current and installed source generations."""

    runtime_manifest = resource_root / "runtime" / "runtime-manifest.json"
    development_runtime = False
    if runtime_manifest.is_file():
        try:
            manifest = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("RUNTIME_RESOURCE_MANIFEST_INVALID") from error
        if not isinstance(manifest, dict):
            raise RuntimeError("RUNTIME_RESOURCE_MANIFEST_INVALID")
        development_runtime = manifest.get("development_only") is True

    if override is None:
        if development_runtime:
            raise ValueError("DEVELOPMENT_RUNTIME_REQUIRES_CURRENT_ROS_WORKSPACE")
        return "/home/dronedream/.local/share/dronedream-autonomy/v0.1.0/ros_ws-merged"

    workspace = override.resolve()
    if (workspace / "setup.bash").is_file() and workspace.name == "install":
        raise ValueError("ROS_WORKSPACE_OVERRIDE_MUST_NAME_WORKSPACE_ROOT")
    setup = workspace / "install" / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(f"ROS workspace setup is missing: {setup}")
    if development_runtime:
        verify_ros_workspace_provenance(Path(__file__).resolve().parents[1], workspace)
    return _wsl_path(workspace)


def _request_outer_timeout_landing(simulation_root: Path) -> Path:
    """Ask the live executor to land before the outer process guard exits."""

    abort_path = simulation_root / "live_abort.request.json"
    if abort_path.exists():
        return abort_path
    payload = {
        "schema_version": "dronedream.runtime-abort-request.v1",
        "reason": "OUTER_RUNTIME_TIMEOUT",
        "world_paused": False,
        "requested_at_unix_ms": int(time.time() * 1_000),
    }
    temporary = abort_path.with_name(f".{abort_path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        temporary.replace(abort_path)
    except OSError:
        temporary.unlink(missing_ok=True)
        if not abort_path.exists():
            raise
    return abort_path


def _run_wsl_runtime(
    *,
    script: str,
    simulation_root: Path,
    timeout_seconds: float,
    landing_grace_seconds: float = OUTER_RUNTIME_LANDING_GRACE_SECONDS,
) -> int:
    """Run the WSL child and fail closed through its controlled-landing path."""

    process = subprocess.Popen(
        ["wsl.exe", "-d", "DroneDreamRuntime", "--", "bash", "-s", "--"],
        stdin=subprocess.PIPE,
    )
    try:
        process.communicate(input=script.encode("utf-8"), timeout=timeout_seconds)
    except subprocess.TimeoutExpired as timeout_error:
        _request_outer_timeout_landing(simulation_root)
        try:
            process.communicate(timeout=landing_grace_seconds)
        except subprocess.TimeoutExpired as landing_error:
            process.kill()
            process.communicate()
            raise RuntimeError(
                "runtime exceeded its outer timeout and did not finish controlled "
                "landing within the cleanup grace period"
            ) from landing_error
        raise RuntimeError(
            "runtime exceeded its outer timeout; controlled landing cleanup completed"
        ) from timeout_error
    return process.returncode


def _validated_heading_control(
    heading_policy: str,
    maximum_yaw_rate_deg_s: float,
) -> tuple[str, float]:
    if heading_policy not in {"measured-hold", "route-tangent-relative"}:
        raise ValueError("unsupported PX4 heading policy")
    yaw_rate = float(maximum_yaw_rate_deg_s)
    if not 1.0 <= yaw_rate <= 90.0:
        raise ValueError("maximum yaw rate must be between 1 and 90 deg/s")
    return heading_policy, yaw_rate


def _validated_heading_evidence(
    timing: dict[str, Any],
    *,
    heading_policy: str,
    maximum_yaw_rate_deg_s: float,
) -> dict[str, Any]:
    heading_control = timing.get("takeoff_gate", {}).get("heading_control")
    if not isinstance(heading_control, dict):
        raise RuntimeError("PX4 heading-control evidence is missing")
    expected_heading_policy = {
        "measured-hold": "measured_prearm_heading_hold",
        "route-tangent-relative": "measured_origin_route_tangent_rate_limited",
    }[heading_policy]
    if heading_control.get("policy") != expected_heading_policy:
        raise RuntimeError("PX4 heading-control policy binding changed")
    expected_yaw_rate = None if heading_policy == "measured-hold" else maximum_yaw_rate_deg_s
    recorded_yaw_rate = heading_control.get("maximum_yaw_rate_deg_s")
    if expected_yaw_rate is None:
        if recorded_yaw_rate is not None:
            raise RuntimeError("measured-hold unexpectedly recorded a yaw-rate authority")
    elif (
        not isinstance(recorded_yaw_rate, (int, float))
        or abs(float(recorded_yaw_rate) - expected_yaw_rate) > 1e-6
    ):
        raise RuntimeError("PX4 maximum yaw-rate binding changed")
    return heading_control


def _validated_heading_tracking_evidence(
    timing: dict[str, Any],
    *,
    heading_policy: str,
) -> dict[str, Any] | None:
    evidence = timing.get("heading_tracking")
    if heading_policy == "measured-hold" and not isinstance(evidence, dict):
        return None
    if not isinstance(evidence, dict):
        raise RuntimeError("PX4 observed heading-tracking evidence is missing")
    if evidence.get("policy") != heading_policy:
        raise RuntimeError("PX4 observed heading-tracking policy binding changed")
    if heading_policy == "measured-hold":
        return evidence
    if evidence.get("status") != "observed":
        raise RuntimeError("PX4 route-tangent heading was not observed")
    sample_count = int(evidence.get("sample_count", 0))
    if sample_count < MINIMUM_ROUTE_TANGENT_HEADING_SAMPLE_COUNT:
        raise RuntimeError("PX4 route-tangent heading has too few observed samples")
    maximum_sample_age_seconds = evidence.get("maximum_sample_age_seconds")
    if not isinstance(maximum_sample_age_seconds, (int, float)) or not (
        0.0 <= float(maximum_sample_age_seconds) <= 1.0
    ):
        raise RuntimeError("PX4 route-tangent heading samples are stale")
    error = evidence.get("absolute_error_deg")
    if not isinstance(error, dict):
        raise RuntimeError("PX4 route-tangent heading error evidence is missing")
    p99_error_deg = error.get("p99")
    if not isinstance(p99_error_deg, (int, float)) or not (
        0.0 <= float(p99_error_deg) <= MAXIMUM_ROUTE_TANGENT_P99_HEADING_ERROR_DEG
    ):
        raise RuntimeError("PX4 route-tangent P99 heading error exceeds the safety bound")
    return evidence


def _validated_px4_dynamics_evidence(
    simulation_root: Path,
    *,
    required: bool,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    telemetry_path = simulation_root / "runtime-state" / "px4-identity-telemetry.json"
    if not telemetry_path.is_file():
        if required:
            raise RuntimeError("PX4 dynamics telemetry evidence is missing")
        return None
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    dynamics = telemetry.get("dynamics") if isinstance(telemetry, dict) else None
    if not isinstance(dynamics, dict):
        if required:
            raise RuntimeError("PX4 dynamics telemetry evidence is missing")
        return None
    if not required:
        return dynamics
    if dynamics.get("schema_version") != "dronedream.px4-dynamics-telemetry.v1":
        raise RuntimeError("PX4 dynamics telemetry schema changed")
    sources = dynamics.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("PX4 dynamics telemetry sources are missing")
    required_sources = {"imu", "attitude", "battery"}
    missing_sources = required_sources - set(sources)
    if missing_sources:
        raise RuntimeError(
            "PX4 dynamics telemetry lacks required sources: " + ", ".join(sorted(missing_sources))
        )
    freshness_bound = dynamics.get("maximum_sample_age_seconds")
    if not isinstance(freshness_bound, (int, float)) or not (0.0 < float(freshness_bound) <= 3.0):
        raise RuntimeError("PX4 dynamics telemetry freshness bound changed")
    preflight = timing.get("preflight_connection") if isinstance(timing, dict) else None
    attempts = preflight.get("attempts") if isinstance(preflight, dict) else None
    successful_attempt = (
        preflight.get("successful_attempt") if isinstance(preflight, dict) else None
    )
    attempt = next(
        (
            item
            for item in attempts or []
            if isinstance(item, dict)
            and item.get("attempt") == successful_attempt
            and item.get("status") == "ready"
        ),
        None,
    )
    rate_evidence = attempt.get("dynamics_telemetry_rates") if isinstance(attempt, dict) else None
    rate_sources = rate_evidence.get("sources") if isinstance(rate_evidence, dict) else None
    if (
        not isinstance(rate_evidence, dict)
        or rate_evidence.get("required_rate_requests_succeeded") is not True
        or not isinstance(rate_sources, dict)
    ):
        raise RuntimeError("PX4 dynamics telemetry rate evidence is missing")
    for source_name, expected_rate_hz in REQUIRED_DYNAMICS_TELEMETRY_RATES_HZ.items():
        source_rate = rate_sources.get(source_name)
        if (
            not isinstance(source_rate, dict)
            or source_rate.get("status") != "requested"
            or not isinstance(source_rate.get("requested_rate_hz"), (int, float))
            or abs(float(source_rate["requested_rate_hz"]) - expected_rate_hz) > 1e-6
        ):
            raise RuntimeError(f"PX4 dynamics telemetry rate was not established: {source_name}")
    if dynamics.get("ready_for_payload_inference") is True:
        for source_name in required_sources:
            source = sources[source_name]
            age_seconds = source.get("sample_age_seconds") if isinstance(source, dict) else None
            if (
                not isinstance(age_seconds, (int, float))
                or isinstance(age_seconds, bool)
                or not (
                    math.isfinite(float(age_seconds))
                    and 0.0 <= float(age_seconds) <= float(freshness_bound)
                )
            ):
                raise RuntimeError(f"PX4 dynamics source is stale: {source_name}")
        return dynamics

    # Streams are deliberately stopped after a confirmed landing.  A terminal
    # snapshot can therefore become stale during evidence finalization even
    # though every payload-expert inference used fresh telemetry in flight.
    # Accept that narrow case only when both the terminal lifecycle and the
    # hash-bound snapshot -> model call -> control-cycle ledger prove it.
    terminal_phase_path = simulation_root / "runtime-phase.json"
    terminal_lifecycle_path = simulation_root / "native-terminal-lifecycle.json"
    if not terminal_phase_path.is_file() or not terminal_lifecycle_path.is_file():
        raise RuntimeError("PX4 dynamics telemetry is not ready for payload inference")
    terminal_phase = json.loads(terminal_phase_path.read_text(encoding="utf-8"))
    terminal_lifecycle = json.loads(terminal_lifecycle_path.read_text(encoding="utf-8"))
    if (
        not isinstance(terminal_phase, dict)
        or terminal_phase.get("phase") != "COMPLETE"
        or not isinstance(terminal_lifecycle, dict)
        or terminal_lifecycle.get("terminal_state") != "ON_GROUND"
        or terminal_lifecycle.get("executor_return_code") != 0
        or terminal_lifecycle.get("publisher_exit_code") != 0
        or terminal_lifecycle.get("landing_confirmed") is not True
        or terminal_lifecycle.get("safe_to_stop_watchdog") is not True
    ):
        raise RuntimeError("PX4 dynamics telemetry is not ready for payload inference")
    inference_evidence = _validated_payload_inference_dynamics_evidence(simulation_root)
    return {
        "schema_version": "dronedream.px4-dynamics-telemetry.v1",
        "ready_for_payload_inference": True,
        "maximum_sample_age_seconds": inference_evidence["maximum_sample_age_seconds"],
        "sources": inference_evidence["sources"],
        "issue_codes": [],
        "evidence_source": "hash-bound-payload-expert-inference-ledger",
        "payload_inference_evidence": inference_evidence,
        "terminal_snapshot": {
            "ready_for_payload_inference": False,
            "issue_codes": dynamics.get("issue_codes", []),
            "recorded_after_confirmed_landing": True,
        },
    }


def _validated_payload_inference_dynamics_evidence(
    simulation_root: Path,
) -> dict[str, Any]:
    """Validate every dynamics sample actually consumed by the payload expert."""

    snapshots_path = simulation_root / "model-navigation-snapshots.jsonl"
    cycles_path = simulation_root / "model-navigation-cycles.jsonl"
    calls_path = simulation_root / "model-navigation-model-calls.jsonl"
    for path in (snapshots_path, cycles_path, calls_path):
        if not path.is_file():
            raise RuntimeError("payload inference dynamics evidence is missing")

    snapshots: dict[str, dict[str, Any]] = {}
    for record in _jsonl_objects(snapshots_path):
        snapshot = record.get("snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeError("payload inference snapshot record is malformed")
        digest = snapshot.get("snapshot_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("payload inference snapshot hash is malformed")
        unhashed = dict(snapshot)
        unhashed.pop("snapshot_sha256", None)
        if sha256_json(unhashed) != digest:
            raise RuntimeError("payload inference snapshot hash does not match")
        if digest in snapshots:
            raise RuntimeError("payload inference snapshot hash is duplicated")
        snapshots[digest] = snapshot

    calls: dict[str, dict[str, Any]] = {}
    payload_call_ids: set[str] = set()
    for record in _jsonl_objects(calls_path):
        call_id = record.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError("payload inference model call id is malformed")
        if call_id in calls:
            raise RuntimeError("payload inference model call id is duplicated")
        calls[call_id] = record
        trace = record.get("local_expert_trace")
        expert_latency = trace.get("expert_latency_ms") if isinstance(trace, dict) else None
        if isinstance(expert_latency, dict) and "payload-dynamics-adapter" in (expert_latency):
            advisory_scores = trace.get("advisory_risk_scores")
            if (
                record.get("provider") != "local-policy"
                or record.get("role") != "local_navigation_advisor"
                or trace.get("schema_version") != "dronedream.local-expert-inference-trace.v1"
                or not isinstance(advisory_scores, dict)
                or "payload-dynamics-adapter" not in advisory_scores
            ):
                raise RuntimeError("payload inference model-call trace is malformed")
            payload_call_ids.add(call_id)

    observed_payload_call_ids: set[str] = set()
    telemetry_hashes: set[str] = set()
    maximum_source_ages = {
        source_name: 0.0 for source_name in ("actuator_output", "imu", "attitude", "battery")
    }
    minimum_cycle_sequence: int | None = None
    maximum_cycle_sequence: int | None = None
    maximum_freshness_bound = 0.0
    for cycle in _jsonl_objects(cycles_path):
        call_id = cycle.get("model_call_id")
        if call_id not in payload_call_ids:
            continue
        if call_id in observed_payload_call_ids:
            raise RuntimeError("payload inference model call is bound to multiple cycles")
        call = calls[call_id]
        snapshot_digest = cycle.get("snapshot_sha256")
        snapshot = snapshots.get(str(snapshot_digest))
        if not isinstance(snapshot, dict):
            raise RuntimeError("payload inference cycle has no bound snapshot")
        if call.get("input_sha256") != sha256_json({"text_navigation_snapshot": snapshot}):
            raise RuntimeError("payload inference call input is not bound to its snapshot")

        strategic = snapshot.get("strategic_context")
        payload = strategic.get("payload") if isinstance(strategic, dict) else None
        if not isinstance(payload, dict) or str(payload.get("state", "")).lower() not in {
            "attached",
            "custody-confirmed",
            "loaded-stable",
        }:
            raise RuntimeError("payload expert was invoked without an attached payload")
        dynamics = payload.get("dynamics")
        if (
            not isinstance(dynamics, dict)
            or dynamics.get("available") is not True
            or dynamics.get("ready") is not True
            or dynamics.get("telemetry_schema_version") != "dronedream.px4-dynamics-telemetry.v1"
        ):
            raise RuntimeError("payload expert consumed unavailable dynamics telemetry")
        issue_codes = dynamics.get("issue_codes")
        if not isinstance(issue_codes, list) or issue_codes:
            raise RuntimeError("payload expert consumed dynamics telemetry with issues")
        freshness_bound = dynamics.get("maximum_sample_age_seconds")
        if (
            not isinstance(freshness_bound, (int, float))
            or isinstance(freshness_bound, bool)
            or not math.isfinite(float(freshness_bound))
            or not 0.0 < float(freshness_bound) <= 3.0
        ):
            raise RuntimeError("payload inference dynamics freshness bound changed")
        source_ages = dynamics.get("source_sample_age_seconds")
        if not isinstance(source_ages, dict):
            raise RuntimeError("payload inference dynamics source ages are missing")
        for source_name in maximum_source_ages:
            age = source_ages.get(source_name)
            if (
                not isinstance(age, (int, float))
                or isinstance(age, bool)
                or not math.isfinite(float(age))
                or not 0.0 <= float(age) <= float(freshness_bound)
            ):
                raise RuntimeError(f"payload expert consumed stale dynamics source: {source_name}")
            maximum_source_ages[source_name] = max(maximum_source_ages[source_name], float(age))
        telemetry_digest = dynamics.get("telemetry_payload_sha256")
        if (
            not isinstance(telemetry_digest, str)
            or len(telemetry_digest) != 64
            or any(character not in "0123456789abcdef" for character in telemetry_digest)
        ):
            raise RuntimeError("payload inference telemetry hash is malformed")
        telemetry_hashes.add(telemetry_digest)
        observed_payload_call_ids.add(call_id)
        sequence = cycle.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise RuntimeError("payload inference cycle sequence is malformed")
        minimum_cycle_sequence = (
            sequence if minimum_cycle_sequence is None else min(minimum_cycle_sequence, sequence)
        )
        maximum_cycle_sequence = (
            sequence if maximum_cycle_sequence is None else max(maximum_cycle_sequence, sequence)
        )
        maximum_freshness_bound = max(maximum_freshness_bound, float(freshness_bound))

    if not payload_call_ids:
        raise RuntimeError("payload dynamics expert was never invoked")
    if observed_payload_call_ids != payload_call_ids:
        raise RuntimeError("payload inference model calls lack control-cycle bindings")
    return {
        "schema_version": "dronedream.payload-inference-dynamics-evidence.v1",
        "payload_expert_call_count": len(payload_call_ids),
        "unique_dynamics_sample_count": len(telemetry_hashes),
        "first_cycle_sequence": minimum_cycle_sequence,
        "last_cycle_sequence": maximum_cycle_sequence,
        "maximum_sample_age_seconds": maximum_freshness_bound,
        "sources": {
            source_name: {"maximum_observed_sample_age_seconds": age}
            for source_name, age in maximum_source_ages.items()
        },
        "model_navigation_snapshots_sha256": hashlib.sha256(
            snapshots_path.read_bytes()
        ).hexdigest(),
        "model_navigation_cycles_sha256": hashlib.sha256(cycles_path.read_bytes()).hexdigest(),
        "model_navigation_model_calls_sha256": hashlib.sha256(calls_path.read_bytes()).hexdigest(),
    }


def _validated_runtime_cpu_affinity_evidence(
    simulation_root: Path,
    *,
    local_navigation_provider: str,
) -> dict[str, Any] | None:
    if local_navigation_provider != "local-policy":
        return None
    preflight_path = simulation_root / "native-runtime-preflight-ready.json"
    if not preflight_path.is_file():
        raise RuntimeError("local-policy runtime CPU-affinity evidence is missing")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    affinity = preflight.get("runtime_cpu_affinity") if isinstance(preflight, dict) else None
    if not isinstance(affinity, dict) or affinity.get("enabled") is not True:
        raise RuntimeError("local-policy runtime CPU affinity was not enabled")
    allowed = affinity.get("allowed_cpu_ids")
    general = affinity.get("general_cpu_ids")
    local_model = affinity.get("local_model_cpu_ids")
    if not all(isinstance(values, list) and values for values in (allowed, general, local_model)):
        raise RuntimeError("local-policy runtime CPU-affinity sets are incomplete")
    allowed_ids = {int(value) for value in allowed}
    general_ids = {int(value) for value in general}
    local_model_ids = {int(value) for value in local_model}
    if general_ids & local_model_ids:
        raise RuntimeError("local-policy and simulator CPU-affinity sets overlap")
    if general_ids | local_model_ids != allowed_ids:
        raise RuntimeError("runtime CPU-affinity sets do not cover the allowed CPUs")
    minimum_model_cpu_count = 4 if len(allowed_ids) >= 16 else 2 if len(allowed_ids) >= 8 else 1
    if len(local_model_ids) < minimum_model_cpu_count:
        raise RuntimeError("local-policy runtime CPU reservation is below the host-tier minimum")
    return affinity


def _validated_multimodal_dataset_evidence(
    measurements: dict[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    evidence = measurements.get("multimodal_dataset")
    if not isinstance(evidence, dict):
        if required:
            raise RuntimeError("multimodal flight dataset evidence is missing")
        return None
    if not required:
        return evidence
    if evidence.get("enabled") is not True:
        raise RuntimeError("multimodal flight dataset recording was not enabled")
    record_count = int(evidence.get("record_count", 0))
    if record_count < 1:
        raise RuntimeError("multimodal flight dataset contains no synchronized records")
    summary = evidence.get("summary")
    if not isinstance(summary, dict) or int(summary.get("record_count", -1)) != record_count:
        raise RuntimeError("multimodal flight dataset summary is inconsistent")
    if summary.get("issue_code") is not None:
        raise RuntimeError("multimodal flight dataset reported an issue")
    if summary.get("qualification_granted") is not False:
        raise RuntimeError("raw multimodal flight data incorrectly claims qualification")
    return evidence


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _execution_authority(
    *, planning_root: Path, run_root: Path, prepared_payload: dict[str, Any]
) -> Path:
    """Derive one exact, single-use authority from the persisted planning boundary.

    The standalone acceptance planner does not own the desktop AppStore that normally
    issues this sidecar.  Its accepted evidence still persists the exact runtime
    envelope and lifecycle revision, so the acceptance bridge can issue the same
    hash-bound authority without weakening any execution gate.
    """

    prepared = PreparedMission.model_validate(prepared_payload)
    if prepared.schema_version != "dronedream.prepared-mission.v4":
        raise RuntimeError("PREPARED_MISSION_SCHEMA_OBSOLETE")
    runtime_payload: dict[str, Any] | None = None
    for record in prepared_payload.get("evidence", []):
        if not isinstance(record, dict):
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        input_metadata = payload.get("input_metadata")
        if not isinstance(input_metadata, dict):
            continue
        candidate = input_metadata.get("model_harness_runtime")
        if isinstance(candidate, dict):
            runtime_payload = candidate
            break
    if runtime_payload is None:
        raise RuntimeError("prepared mission has no persisted model harness runtime envelope")

    lifecycle_path = planning_root / "plan" / "mission-lifecycle.json"
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    try:
        plan_revision_id = str(lifecycle["plan_revision"]["plan_revision_id"])
    except (KeyError, TypeError) as error:
        raise RuntimeError("prepared mission lifecycle has no plan revision") from error

    authority = compile_execution_authority(
        thread_id=prepared.contract.conversation_id,
        plan_revision_id=plan_revision_id,
        contract_id=prepared.contract.contract_id,
        # Match the durable artifact identity used by the execution boundary.
        # New reader versions may add defaulted compatibility fields; those
        # must not change the hash of an already prepared mission.
        prepared_mission_sha256=sha256_json(prepared.model_dump(mode="json", exclude_unset=True)),
        runtime=HarnessRuntimeEnvelope.model_validate(runtime_payload),
    )
    path = run_root / "model-harness-execution-authority.json"
    _atomic_json(path, authority.model_dump(mode="json"))
    return path


def _inject_runtime_message(
    *,
    run_root: Path,
    message_text: str,
    phase: str = "TRACK",
    timeout_seconds: float = 240.0,
) -> None:
    """Wait for a real execution phase, then enqueue one exact UTF-8 user message."""

    phase_path = run_root / "simulation" / "runtime-phase.json"
    control_dir = run_root / "simulation" / "runtime-control"
    evidence_path = run_root / "runtime-message-injection.json"
    started = time.monotonic()
    last_phase: str | None = None
    try:
        while time.monotonic() - started < timeout_seconds:
            if phase_path.is_file():
                try:
                    phase_payload = json.loads(phase_path.read_text(encoding="utf-8"))
                    last_phase = str(phase_payload.get("phase", ""))
                except (OSError, json.JSONDecodeError):
                    last_phase = None
                if last_phase == phase and (control_dir / "session.json").is_file():
                    message = submit_runtime_message(
                        control_dir=control_dir,
                        text=message_text,
                    )
                    _atomic_json(
                        evidence_path,
                        {
                            "status": "submitted",
                            "target_phase": phase,
                            "observed_phase": last_phase,
                            "message_id": message.message_id,
                            "message_sha256": sha256_json(message),
                            "text_sha256": hashlib.sha256(message_text.encode("utf-8")).hexdigest(),
                            "utf8_round_trip": (
                                message_text.encode("utf-8").decode("utf-8") == message_text
                            ),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
                    return
            time.sleep(0.1)
        raise TimeoutError(
            f"runtime did not reach {phase!r} before injection timeout; last phase={last_phase!r}"
        )
    except Exception as exc:
        _atomic_json(
            evidence_path,
            {
                "status": "failed",
                "target_phase": phase,
                "last_phase": last_phase,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started,
            },
        )


def _summarize(
    run_root: Path,
    *,
    contract_id: str,
    provider: str,
    model: str,
    watchdog_deadline_ms: int,
    watchdog_startup_deadline_ms: int,
    watchdog_scheduling_jitter_grace_ms: int,
    watchdog_persistent_miss_ms: int,
    local_navigation_provider: str,
    local_navigation_visual_enabled: bool,
    heading_policy: str,
    maximum_yaw_rate_deg_s: float,
    require_px4_dynamics_telemetry: bool,
    record_multimodal_dataset: bool,
    asset_input_mode: str,
    asset_input_qualification_granted: bool,
    development_payload_collection: bool = False,
) -> dict[str, object]:
    simulation_root = run_root / "simulation"
    workflow_path = simulation_root / "workflow-result.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    plugin_log = (run_root / "plugin-host.log").read_text(encoding="utf-8", errors="replace")
    if not development_payload_collection and workflow.get("status") != "verified":
        raise RuntimeError("runtime workflow did not reach verified status")
    if development_payload_collection and workflow.get("status") == "verified":
        raise RuntimeError("development payload collection incorrectly reached verified status")
    if "activated plugin runtime.safe-hold" not in plugin_log:
        raise RuntimeError("ROS safe-hold capability was not activated")
    if "forced fail-closed" in plugin_log or "plugin configure failed" in plugin_log:
        raise RuntimeError("ROS native capability host entered its fail-closed path")
    runtime_evidence = workflow["runtime_evidence"]
    development_payload_evidence = (
        _validated_development_payload_collection_evidence(
            simulation_root,
            runtime_evidence=runtime_evidence,
        )
        if development_payload_collection
        else None
    )
    measurements = runtime_evidence["measurements"]
    timing = json.loads((simulation_root / "offboard_timing.json").read_text(encoding="utf-8"))
    heading_control = _validated_heading_evidence(
        timing,
        heading_policy=heading_policy,
        maximum_yaw_rate_deg_s=maximum_yaw_rate_deg_s,
    )
    heading_tracking = _validated_heading_tracking_evidence(
        timing,
        heading_policy=heading_policy,
    )
    dynamics_telemetry = _validated_px4_dynamics_evidence(
        simulation_root,
        required=require_px4_dynamics_telemetry,
        timing=timing,
    )
    cpu_affinity = _validated_runtime_cpu_affinity_evidence(
        simulation_root,
        local_navigation_provider=local_navigation_provider,
    )
    multimodal_dataset = _validated_multimodal_dataset_evidence(
        measurements,
        required=record_multimodal_dataset,
    )
    model_navigation = measurements.get("model_navigation", {})
    if not isinstance(model_navigation, dict) or not model_navigation.get("enabled"):
        raise RuntimeError("continuous local model navigation was not enabled")
    if int(model_navigation.get("provider_call_count", 0)) < 1:
        raise RuntimeError("continuous local model navigation made no provider calls")
    if int(model_navigation.get("authorized_control_applied_count", 0)) < 1:
        raise RuntimeError("continuous local model navigation never authorized motion")
    if model_navigation.get("provider") != local_navigation_provider:
        raise RuntimeError("continuous local navigation provider binding changed")
    if bool(model_navigation.get("visual_enabled")) != local_navigation_visual_enabled:
        raise RuntimeError("continuous local navigation visual binding changed")
    if not bool(model_navigation.get("model_control_authority_required")):
        raise RuntimeError("continuous model control authority was not required")
    checkpoints = workflow.get("checkpoint_decisions", [])
    injection_path = run_root / "runtime-message-injection.json"
    runtime_message: dict[str, Any] | None = None
    if injection_path.is_file():
        runtime_message = json.loads(injection_path.read_text(encoding="utf-8"))
        if runtime_message.get("status") != "submitted":
            raise RuntimeError(f"runtime message injection failed: {runtime_message}")
        message_id = str(runtime_message["message_id"])
        control_root = simulation_root / "runtime-control"
        required_message_evidence = {
            # The coordinator atomically moves a claimed inbox item into the
            # processed ledger after the decision is finalized.  Requiring the
            # transient claimed path would reject the successful terminal state.
            "processed": control_root / "processed" / f"{message_id}.json",
            "detected": control_root / "detected" / f"{message_id}.json",
            "acknowledgement": control_root / "acks" / f"{message_id}.json",
            "decision": control_root / "decisions" / f"{message_id}.json",
        }
        missing_runtime_evidence = [
            name for name, path in required_message_evidence.items() if not path.is_file()
        ]
        if missing_runtime_evidence:
            raise RuntimeError(
                "runtime message lacks closed-loop evidence: " + ", ".join(missing_runtime_evidence)
            )
        adoption_candidates = (
            control_root / "adoptions" / f"{message_id}.json",
            control_root / "command-results" / f"{message_id}.json",
            control_root / "takeover-adoptions" / f"{message_id}.json",
        )
        if not any(path.is_file() for path in adoption_candidates):
            raise RuntimeError("runtime message was decided but never adopted by the executor")
        runtime_message["closed_loop_evidence"] = True
    summary = {
        "schema_version": "dronedream.pluginized-runtime-acceptance.v1",
        "status": (
            "verified-development-payload-collection"
            if development_payload_collection
            else workflow["status"]
        ),
        "flight_qualification_granted": (
            not development_payload_collection and asset_input_qualification_granted
        ),
        "asset_input_mode": asset_input_mode,
        "asset_input_qualification_granted": asset_input_qualification_granted,
        "contract_id": contract_id,
        "provider": provider,
        "model": model,
        "plugin_host": "runtime.safe-hold",
        "plugin_host_activated": True,
        "ros_domain_id": 74,
        "rmw_implementation": "rmw_cyclonedds_cpp",
        "watchdog_deadline_ms": watchdog_deadline_ms,
        "watchdog_startup_deadline_ms": watchdog_startup_deadline_ms,
        "watchdog_scheduling_jitter_grace_ms": watchdog_scheduling_jitter_grace_ms,
        "watchdog_persistent_miss_ms": watchdog_persistent_miss_ms,
        "plugin_host_log_sha256": hashlib.sha256(
            (run_root / "plugin-host.log").read_bytes()
        ).hexdigest(),
        "workflow_sha256": hashlib.sha256(workflow_path.read_bytes()).hexdigest(),
        "executor_return_code": measurements["executor_return_code"],
        "landing_state": measurements["landing_state"],
        "pose_sample_count": measurements["pose_sample_count"],
        "ros_observation_rows": measurements["ros_observation_rows"],
        "minimum_goal_distance_m": measurements["minimum_goal_distance_m"],
        "checkpoint_calls": len(checkpoints),
        "checkpoints_authorized": all(
            bool(item.get("continue_authorized")) for item in checkpoints
        ),
        "completion_accepted": bool(workflow["completion_assessment"]["accepted"]),
        "continuous_model_navigation": model_navigation,
        "heading_control": heading_control,
        "heading_tracking": heading_tracking,
        "px4_dynamics_telemetry": dynamics_telemetry,
        "px4_dynamics_telemetry_required": require_px4_dynamics_telemetry,
        "runtime_cpu_affinity": cpu_affinity,
        "multimodal_dataset": multimodal_dataset,
        "multimodal_dataset_required": record_multimodal_dataset,
        "runtime_message": runtime_message,
        "development_payload_collection": development_payload_evidence,
    }
    (run_root / "runtime-acceptance-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run(
    planning_root: Path,
    run_root: Path,
    resource_root: Path,
    *,
    provider: str,
    model: str,
    local_navigation_provider: str | None = None,
    local_navigation_model: str | None = None,
    local_navigation_fallback_provider: str | None = None,
    local_navigation_fallback_model: str | None = None,
    local_navigation_visual_enabled: bool = False,
    heading_policy: str = "route-tangent-relative",
    maximum_yaw_rate_deg_s: float = 20.0,
    require_px4_dynamics_telemetry: bool = False,
    record_multimodal_dataset: bool = False,
    multimodal_dataset_maximum_mib: int = 5_120,
    multimodal_record_period_seconds: float = 0.1,
    local_navigation_model_timeout_seconds: float = 10.0,
    local_navigation_fallback_model_timeout_seconds: float = 10.0,
    local_navigation_period_seconds: float = 3.0,
    runtime_timeout_seconds: float = DEFAULT_RUNTIME_TIMEOUT_SECONDS,
    runtime_message_file: Path | None = None,
    development_input: Path | None = None,
    ros_workspace_override: Path | None = None,
    local_policy_package_paths: tuple[Path, ...] = (),
    local_policy_qualification_paths: tuple[Path, ...] = (),
    local_policy_simulation_admission_paths: tuple[Path, ...] = (),
    development_payload_collection: bool = False,
) -> dict[str, object]:
    runtime_timeout_seconds = _validated_runtime_timeout_seconds(runtime_timeout_seconds)
    heading_policy, maximum_yaw_rate_deg_s = _validated_heading_control(
        heading_policy,
        maximum_yaw_rate_deg_s,
    )
    prepared_path = planning_root / "plan" / "prepared-mission.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    prepared_mission = PreparedMission.model_validate(prepared)
    if prepared_mission.schema_version != "dronedream.prepared-mission.v4":
        raise RuntimeError("PREPARED_MISSION_SCHEMA_OBSOLETE")
    contract_id = prepared_mission.contract.contract_id
    planning_summary_path = planning_root / "acceptance-summary.json"
    try:
        planning_summary = json.loads(planning_summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("CURRENT_PLANNING_INPUT_BINDING_REQUIRED") from error
    asset_input_mode = planning_summary.get("asset_input_mode")
    asset_input_qualification_granted = planning_summary.get("flight_qualification_granted")
    if asset_input_mode not in {
        "qualified-bundled-pair",
        "current-development-mission",
    } or not isinstance(asset_input_qualification_granted, bool):
        raise RuntimeError("CURRENT_PLANNING_INPUT_BINDING_INVALID")
    simulation_contract = prepared.get("simulation_capabilities")
    if not isinstance(simulation_contract, dict):
        raise RuntimeError("prepared mission has no simulation capability contract")
    simulator_contract = simulation_contract.get("simulator")
    native_contract = simulation_contract.get("native_runtime")
    if not isinstance(simulator_contract, dict) or simulator_contract.get("engine") != (
        "gazebo-harmonic"
    ):
        raise RuntimeError("prepared mission does not select the Gazebo Harmonic adapter")
    if not isinstance(native_contract, dict):
        raise RuntimeError("prepared mission has no native runtime contract")
    transport_contract = native_contract.get("transport")
    watchdog_contract = native_contract.get("watchdog")
    if (
        not isinstance(transport_contract, dict)
        or transport_contract.get("implementation") != "px4-uxrce-dds"
    ):
        raise RuntimeError("prepared mission does not select the PX4 uXRCE-DDS transport")
    if not isinstance(watchdog_contract, dict):
        raise RuntimeError("prepared mission has no native watchdog contract")
    watchdog_deadline_ms = int(watchdog_contract.get("deadline_ms", 1_000))
    watchdog_startup_deadline_ms = int(watchdog_contract.get("startup_deadline_ms", 10_000))
    watchdog_scheduling_jitter_grace_ms = int(
        watchdog_contract.get("scheduling_jitter_grace_ms", 250)
    )
    watchdog_persistent_miss_ms = int(watchdog_contract.get("persistent_miss_ms", 250))
    if not 20 <= watchdog_deadline_ms <= 1_000:
        raise RuntimeError("prepared mission watchdog deadline is outside the safe range")
    if (
        not 1_000 <= watchdog_startup_deadline_ms <= 60_000
        or watchdog_startup_deadline_ms < watchdog_deadline_ms
    ):
        raise RuntimeError("prepared mission watchdog startup deadline is outside the safe range")
    if not 0 <= watchdog_scheduling_jitter_grace_ms <= 500:
        raise RuntimeError("prepared mission watchdog scheduling grace is outside the safe range")
    if not 25 <= watchdog_persistent_miss_ms <= 1_000:
        raise RuntimeError("prepared mission watchdog persistent miss is outside the safe range")
    local_navigation_provider = local_navigation_provider or provider
    if local_navigation_provider == "local-policy":
        if not local_policy_package_paths or not (
            local_policy_qualification_paths or local_policy_simulation_admission_paths
        ):
            raise ValueError(
                "local policy navigation requires packages and qualification "
                "or simulation admission receipts"
            )
        local_navigation_model = "local-policy"
    else:
        if (
            local_policy_package_paths
            or local_policy_qualification_paths
            or local_policy_simulation_admission_paths
        ):
            raise ValueError("local policy artifacts require the local-policy provider")
        local_navigation_model = local_navigation_model or (
            model
            if local_navigation_provider == provider
            else _provider_default_model(local_navigation_provider)
        )
    if local_navigation_provider == provider and local_navigation_model != model:
        raise ValueError("one provider cannot select two models in the same runtime session")
    if local_navigation_fallback_provider == local_navigation_provider:
        raise ValueError("local navigation fallback provider must differ from primary")
    if development_payload_collection:
        if local_navigation_provider != "local-policy":
            raise ValueError("development payload collection requires local-policy")
        if not local_policy_simulation_admission_paths or local_policy_qualification_paths:
            raise ValueError("development payload collection requires simulation admission only")
        if not record_multimodal_dataset:
            raise ValueError("development payload collection requires multimodal recording")
        if not require_px4_dynamics_telemetry:
            raise ValueError("development payload collection requires PX4 dynamics telemetry")
    if not 1.0 <= local_navigation_model_timeout_seconds <= 120.0:
        raise ValueError("local navigation model timeout must be between 1 and 120 seconds")
    if not 1.0 <= local_navigation_fallback_model_timeout_seconds <= 120.0:
        raise ValueError(
            "local navigation fallback model timeout must be between 1 and 120 seconds"
        )
    minimum_navigation_period_seconds = 0.2 if local_navigation_provider == "local-policy" else 1.0
    if not minimum_navigation_period_seconds <= local_navigation_period_seconds <= 120.0:
        raise ValueError(
            "local navigation period must be between "
            f"{minimum_navigation_period_seconds:g} and 120 seconds for "
            f"{local_navigation_provider}"
        )
    if local_navigation_fallback_provider is not None:
        local_navigation_fallback_model = (
            local_navigation_fallback_model
            or _provider_default_model(local_navigation_fallback_provider)
        )
    if run_root.exists():
        if (run_root / "simulation" / "workflow-result.json").is_file():
            return _summarize(
                run_root,
                contract_id=contract_id,
                provider=provider,
                model=model,
                watchdog_deadline_ms=watchdog_deadline_ms,
                watchdog_startup_deadline_ms=watchdog_startup_deadline_ms,
                watchdog_scheduling_jitter_grace_ms=(
                    watchdog_scheduling_jitter_grace_ms
                ),
                watchdog_persistent_miss_ms=watchdog_persistent_miss_ms,
                local_navigation_provider=local_navigation_provider,
                local_navigation_visual_enabled=local_navigation_visual_enabled,
                heading_policy=heading_policy,
                maximum_yaw_rate_deg_s=maximum_yaw_rate_deg_s,
                require_px4_dynamics_telemetry=require_px4_dynamics_telemetry,
                record_multimodal_dataset=record_multimodal_dataset,
                asset_input_mode=str(asset_input_mode),
                asset_input_qualification_granted=asset_input_qualification_granted,
                development_payload_collection=development_payload_collection,
            )
        raise FileExistsError(f"run directory already exists: {run_root}")
    provider_models: dict[str, str] = {}
    selections = [(provider, model)]
    if local_navigation_provider != "local-policy":
        selections.append((local_navigation_provider, local_navigation_model))
    if local_navigation_fallback_provider is not None:
        selections.append(
            (
                local_navigation_fallback_provider,
                str(local_navigation_fallback_model),
            )
        )
    for selected_provider, selected_model in selections:
        existing_model = provider_models.get(selected_provider)
        if existing_model is not None and existing_model != selected_model:
            raise ValueError("one provider cannot select two models in the same runtime session")
        provider_models[selected_provider] = selected_model
    provider_secrets: dict[str, str] = {}
    for selected_provider in provider_models:
        key_env = _provider_key_env(selected_provider)
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"{key_env} is not configured")
        provider_secrets[selected_provider] = api_key
    runtime_message_text: str | None = None
    if runtime_message_file is not None:
        runtime_message_text = runtime_message_file.read_text(encoding="utf-8").strip()
        if not runtime_message_text:
            raise ValueError("runtime message file is empty")
    if not 1 <= multimodal_dataset_maximum_mib <= 20 * 1_024:
        raise ValueError("multimodal dataset quota is outside the safe range")
    if not 0.05 <= multimodal_record_period_seconds <= 10.0:
        raise ValueError("multimodal recording period is outside the safe range")

    map_asset_id = planning_summary.get("map_asset_id")
    map_content_sha256 = planning_summary.get("map_content_sha256")
    vehicle_asset_id = planning_summary.get("vehicle_asset_id")
    vehicle_content_sha256 = planning_summary.get("vehicle_content_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (
            map_asset_id,
            map_content_sha256,
            vehicle_asset_id,
            vehicle_content_sha256,
        )
    ):
        raise RuntimeError("CURRENT_PLANNING_ASSET_IDENTITY_INVALID")
    store = AppStore(planning_root / "app-store")
    try:
        map_selection = resolve_versioned_map(
            store.get_asset_version(str(map_asset_id), str(map_content_sha256))
        )
        if asset_input_mode == "qualified-bundled-pair":
            if development_input is not None:
                raise ValueError("DEVELOPMENT_INPUT_NOT_AUTHORIZED_BY_PLAN")
            vehicle_selection = resolve_versioned_vehicle(
                store.get_asset_version(str(vehicle_asset_id), str(vehicle_content_sha256))
            )
            vehicle_sdf_path = vehicle_selection.vehicle_sdf
            vehicle_metadata_path = vehicle_selection.vehicle_metadata
            controller_params_path = vehicle_selection.controller_params
        else:
            if development_input is None:
                raise ValueError("DEVELOPMENT_INPUT_REQUIRED_BY_PLAN")
            development_mission = resolve_development_mission_input(development_input)
            assert_development_map_binding(development_mission, map_selection)
            if (
                development_mission.asset_id != vehicle_asset_id
                or development_mission.manifest_sha256 != vehicle_content_sha256
            ):
                raise ValueError("DEVELOPMENT_INPUT_PLAN_MISMATCH")
            vehicle_sdf_path = development_mission.vehicle_sdf
            vehicle_metadata_path = development_mission.vehicle_metadata
            controller_params_path = development_mission.controller_params
    finally:
        store.close()
    world_sdf_path = map_selection.world_sdf
    semantic_path = map_selection.semantic
    map_graph_path = map_selection.graph
    resolved_graph = MapAsset.model_validate_json(map_graph_path.read_text(encoding="utf-8"))
    resolved_vehicle = VehicleAsset.model_validate_json(
        vehicle_metadata_path.read_text(encoding="utf-8")
    )
    if (
        sha256_json(resolved_graph) != prepared_mission.contract.map_sha256
        or hashlib.sha256(semantic_path.read_bytes()).hexdigest()
        != prepared_mission.contract.map_semantic_sha256
        or hashlib.sha256(vehicle_sdf_path.read_bytes()).hexdigest()
        != prepared_mission.contract.vehicle_sha256
        or resolved_vehicle.asset_id != prepared_mission.contract.vehicle_asset_id
    ):
        raise RuntimeError("CURRENT_RUNTIME_ASSET_CONTRACT_MISMATCH")
    runtime = resource_root / "runtime"
    resource_src = resource_root / "src"
    checkpoint_executor = runtime / "px4_checkpoint_executor.py"
    if not checkpoint_executor.is_file():
        checkpoint_executor = resource_root / "scripts" / "px4_checkpoint_executor.py"
    required = (
        prepared_path,
        world_sdf_path,
        semantic_path,
        map_graph_path,
        vehicle_sdf_path,
        vehicle_metadata_path,
        controller_params_path,
        runtime / "px4_offboard_track_executor.py",
        checkpoint_executor,
    )
    missing = [str(path) for path in required if not path.is_file()]
    missing.extend(
        str(path)
        for path in (
            *local_policy_package_paths,
            *local_policy_qualification_paths,
            *local_policy_simulation_admission_paths,
        )
        if not path.exists()
    )
    if missing:
        raise FileNotFoundError(f"acceptance inputs are missing: {missing}")

    run_root.mkdir(parents=True)
    simulation_root = run_root / "simulation"
    ros_workspace_wsl = _resolve_ros_workspace(resource_root, ros_workspace_override)
    authority_path = _execution_authority(
        planning_root=planning_root,
        run_root=run_root,
        prepared_payload=prepared,
    )
    secret_path = run_root / f".model-session-{uuid4().hex}.env"
    secret_path.write_text(
        "".join(
            _secret_lines(
                selected_provider,
                selected_model,
                provider_secrets[selected_provider],
            )
            for selected_provider, selected_model in provider_models.items()
        ),
        encoding="utf-8",
        newline="\n",
    )
    run_wsl = shlex.quote(_wsl_path(run_root))
    native_preflight_marker_wsl = shlex.quote(
        _wsl_path(simulation_root / "native-runtime-preflight-ready.json")
    )
    secret_wsl = shlex.quote(_wsl_path(secret_path))
    command = [
        "--profile",
        "runtime-manager",
        "execute-prepared-mission",
        "--prepared",
        _wsl_path(prepared_path),
        "--execution-authority",
        _wsl_path(authority_path),
        "--confirm-contract-id",
        contract_id,
        "--completion-provider",
        provider,
        "--run-dir",
        _wsl_path(simulation_root),
        "--world-sdf",
        _wsl_path(world_sdf_path),
        "--semantic",
        _wsl_path(semantic_path),
        "--map-graph",
        _wsl_path(map_graph_path),
        "--vehicle-sdf",
        _wsl_path(vehicle_sdf_path),
        "--vehicle-metadata",
        _wsl_path(vehicle_metadata_path),
        "--controller-params",
        _wsl_path(controller_params_path),
        "--px4-root",
        "/opt/PX4-Autopilot",
        "--executor",
        _wsl_path(runtime / "px4_offboard_track_executor.py"),
        "--context-db",
        _wsl_path(run_root / "mission-context.sqlite3"),
        "--checkpoint-provider",
        provider,
        "--checkpoint-executor",
        _wsl_path(checkpoint_executor),
        "--runtime-interrupt-provider",
        provider,
        "--runtime-hold-timeout-seconds",
        "12",
        "--runtime-replan-hold-seconds",
        "60",
        "--local-navigation-provider",
        local_navigation_provider,
        "--local-navigation-model-timeout-seconds",
        f"{local_navigation_model_timeout_seconds:g}",
        "--local-navigation-period-seconds",
        f"{local_navigation_period_seconds:g}",
        "--local-navigation-context-id",
        contract_id,
        "--require-local-navigation-control-authority",
        "--heading-policy",
        heading_policy,
        "--maximum-yaw-rate-deg-s",
        f"{maximum_yaw_rate_deg_s:g}",
    ]
    if local_navigation_fallback_provider is not None:
        command.extend(
            [
                "--local-navigation-fallback-provider",
                local_navigation_fallback_provider,
                "--local-navigation-fallback-model-timeout-seconds",
                f"{local_navigation_fallback_model_timeout_seconds:g}",
            ]
        )
    for package_path in local_policy_package_paths:
        command.extend(["--local-policy-package", _wsl_path(package_path.resolve())])
    for qualification_path in local_policy_qualification_paths:
        command.extend(["--local-policy-qualification", _wsl_path(qualification_path.resolve())])
    for admission_path in local_policy_simulation_admission_paths:
        command.extend(
            [
                "--local-policy-simulation-admission",
                _wsl_path(admission_path.resolve()),
            ]
        )
    if local_navigation_visual_enabled:
        command.append("--local-navigation-visual-enabled")
    if record_multimodal_dataset:
        command.extend(
            [
                "--multimodal-dataset-root",
                _wsl_path(simulation_root / "multimodal-dataset"),
                "--multimodal-flight-id",
                contract_id,
                "--multimodal-dataset-maximum-mib",
                str(multimodal_dataset_maximum_mib),
                "--multimodal-record-period-seconds",
                f"{multimodal_record_period_seconds:g}",
            ]
        )
    if development_payload_collection:
        command.append("--development-payload-collection")
    command_text = " ".join(shlex.quote(value) for value in command)
    source_path_export = ""
    if resource_src.is_dir():
        source_path_export = (
            f"export PYTHONPATH={shlex.quote(_wsl_path(resource_src))}"
            "${PYTHONPATH:+:$PYTHONPATH}\n"
        )
    script = (
        "set -eo pipefail\n"
        "source /opt/ros/jazzy/setup.bash\n"
        'root="$HOME/.local/share/dronedream-autonomy/v0.1.0"\n'
        f"ros_workspace={shlex.quote(ros_workspace_wsl)}\n"
        'source "$ros_workspace/install/setup.bash"\n'
        "set -u\n"
        f"{source_path_export}"
        "export ROS_DOMAIN_ID=74 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp "
        "ROS2_DISABLE_DAEMON=1\n"
        'export CYCLONEDDS_URI=\'<CycloneDDS><Domain Id="any"><General>'
        '<Interfaces><NetworkInterface address="127.0.0.1"/></Interfaces>'
        "<AllowMulticast>false</AllowMulticast></General><Discovery>"
        "<ParticipantIndex>auto</ParticipantIndex><Peers>"
        '<Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>\'\n'
        f"source {secret_wsl}\n"
        f"rm -f {secret_wsl}\n"
        "(\n"
        f"  until [ -f {native_preflight_marker_wsl} ]; do sleep 0.1; done\n"
        "  exec ros2 run dronedream_agent_plugin_api capability_host "
        "--ros-args "
        f"-p contract_id:={shlex.quote(contract_id)} "
        f"-p watchdog_deadline_ms:={watchdog_deadline_ms} "
        f"-p watchdog_startup_deadline_ms:={watchdog_startup_deadline_ms} "
        f"-p watchdog_scheduling_jitter_grace_ms:="
        f"{watchdog_scheduling_jitter_grace_ms} "
        f"-p watchdog_persistent_miss_ms:={watchdog_persistent_miss_ms} "
        f">{run_wsl}/plugin-host.log 2>&1\n"
        ") &\n"
        "plugin_host_pid=$!\n"
        "trap 'kill $plugin_host_pid 2>/dev/null || true; "
        "wait $plugin_host_pid 2>/dev/null || true' EXIT\n"
        f'"$root/venv/bin/dronedream-agent" {command_text} '
        '--ros-workspace "$ros_workspace"\n'
    )
    injection_thread: threading.Thread | None = None
    if runtime_message_text is not None:
        injection_thread = threading.Thread(
            target=_inject_runtime_message,
            kwargs={"run_root": run_root, "message_text": runtime_message_text},
            name="runtime-message-injector",
            daemon=True,
        )
        injection_thread.start()
    try:
        return_code = _run_wsl_runtime(
            script=script,
            simulation_root=simulation_root,
            # Complete model-guided round trips can take substantially longer
            # than their simulated flight time when depth rendering lowers the
            # real-time factor. If an explicitly shorter outer guard does fire,
            # it must still use the executor's controlled-landing cleanup path.
            timeout_seconds=runtime_timeout_seconds,
        )
    finally:
        secret_path.unlink(missing_ok=True)
        if injection_thread is not None:
            injection_thread.join(timeout=5.0)
    expected_return_codes = {0, 2} if development_payload_collection else {0}
    if return_code not in expected_return_codes:
        raise RuntimeError(f"pluginized runtime failed with exit code {return_code}")

    return _summarize(
        run_root,
        contract_id=contract_id,
        provider=provider,
        model=model,
        watchdog_deadline_ms=watchdog_deadline_ms,
        watchdog_startup_deadline_ms=watchdog_startup_deadline_ms,
        watchdog_scheduling_jitter_grace_ms=watchdog_scheduling_jitter_grace_ms,
        watchdog_persistent_miss_ms=watchdog_persistent_miss_ms,
        local_navigation_provider=local_navigation_provider,
        local_navigation_visual_enabled=local_navigation_visual_enabled,
        heading_policy=heading_policy,
        maximum_yaw_rate_deg_s=maximum_yaw_rate_deg_s,
        require_px4_dynamics_telemetry=require_px4_dynamics_telemetry,
        record_multimodal_dataset=record_multimodal_dataset,
        asset_input_mode=str(asset_input_mode),
        asset_input_qualification_granted=asset_input_qualification_granted,
        development_payload_collection=development_payload_collection,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("planning_root", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("resource_root", type=Path)
    parser.add_argument("--provider", choices=("deepseek", "kimi", "openai"), default="deepseek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument(
        "--local-navigation-provider",
        choices=("deepseek", "kimi", "openai", "local-policy"),
        help="defaults to the mission provider",
    )
    parser.add_argument("--local-navigation-model")
    parser.add_argument(
        "--local-navigation-fallback-provider",
        choices=("deepseek", "kimi", "openai"),
    )
    parser.add_argument("--local-navigation-fallback-model")
    parser.add_argument(
        "--local-navigation-model-timeout-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--local-navigation-fallback-model-timeout-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--local-navigation-period-seconds",
        type=float,
        default=3.0,
    )
    parser.add_argument("--local-navigation-visual-enabled", action="store_true")
    parser.add_argument(
        "--heading-policy",
        choices=("measured-hold", "route-tangent-relative"),
        default="route-tangent-relative",
    )
    parser.add_argument("--maximum-yaw-rate-deg-s", type=float, default=20.0)
    parser.add_argument(
        "--require-px4-dynamics-telemetry",
        action="store_true",
        help=(
            "fail unless fresh PX4 IMU, attitude, and battery evidence is ready "
            "for payload inference"
        ),
    )
    parser.add_argument("--record-multimodal-dataset", action="store_true")
    parser.add_argument("--development-payload-collection", action="store_true")
    parser.add_argument("--multimodal-dataset-maximum-mib", type=int, default=5_120)
    parser.add_argument("--multimodal-record-period-seconds", type=float, default=0.1)
    parser.add_argument("--local-policy-package", type=Path, action="append", default=[])
    parser.add_argument("--local-policy-qualification", type=Path, action="append", default=[])
    parser.add_argument(
        "--local-policy-simulation-admission",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--development-input",
        type=Path,
        help=(
            "The same hash-pinned current-development manifest frozen by planning; "
            "free-form file overrides are intentionally unsupported."
        ),
    )
    parser.add_argument(
        "--ros-workspace-override",
        type=Path,
        help=(
            "ROS workspace root containing install/setup.bash. Development Runtime "
            "resources require this explicit binding and reject the install directory itself."
        ),
    )
    parser.add_argument(
        "--runtime-message-file",
        type=Path,
        help="UTF-8 natural-language message injected after the real runtime reaches TRACK",
    )
    parser.add_argument(
        "--runtime-timeout-seconds",
        type=float,
        default=DEFAULT_RUNTIME_TIMEOUT_SECONDS,
        help=(
            "outer stuck-process guard; motion and no-progress deadlines remain "
            "owned by the child runtime"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.planning_root,
                args.run_root,
                args.resource_root,
                provider=args.provider,
                model=args.model,
                local_navigation_provider=args.local_navigation_provider,
                local_navigation_model=args.local_navigation_model,
                local_navigation_fallback_provider=(args.local_navigation_fallback_provider),
                local_navigation_fallback_model=args.local_navigation_fallback_model,
                local_navigation_visual_enabled=(args.local_navigation_visual_enabled),
                heading_policy=args.heading_policy,
                maximum_yaw_rate_deg_s=args.maximum_yaw_rate_deg_s,
                require_px4_dynamics_telemetry=(args.require_px4_dynamics_telemetry),
                record_multimodal_dataset=args.record_multimodal_dataset,
                development_payload_collection=args.development_payload_collection,
                multimodal_dataset_maximum_mib=(args.multimodal_dataset_maximum_mib),
                multimodal_record_period_seconds=(args.multimodal_record_period_seconds),
                local_navigation_model_timeout_seconds=(
                    args.local_navigation_model_timeout_seconds
                ),
                local_navigation_fallback_model_timeout_seconds=(
                    args.local_navigation_fallback_model_timeout_seconds
                ),
                local_navigation_period_seconds=args.local_navigation_period_seconds,
                runtime_timeout_seconds=args.runtime_timeout_seconds,
                runtime_message_file=args.runtime_message_file,
                development_input=args.development_input,
                ros_workspace_override=args.ros_workspace_override,
                local_policy_package_paths=tuple(args.local_policy_package),
                local_policy_qualification_paths=tuple(args.local_policy_qualification),
                local_policy_simulation_admission_paths=tuple(
                    args.local_policy_simulation_admission
                ),
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
