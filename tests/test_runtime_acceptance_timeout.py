from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.ros_workspace_provenance import (
    write_ros_workspace_provenance,
)


def _load_runtime_acceptance_module():
    path = Path(__file__).parents[1] / "scripts" / "run_pluginized_runtime_acceptance.py"
    spec = importlib.util.spec_from_file_location("test_runtime_acceptance_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outer_runtime_timeout_outlives_child_executor_cap() -> None:
    module = _load_runtime_acceptance_module()

    # The Gazebo adapter caps motion at six hours.  The outer process guard
    # must leave room for takeoff, checkpoints, actions, landing, and evidence.
    assert module.DEFAULT_RUNTIME_TIMEOUT_SECONDS > 21_600.0
    assert module._validated_runtime_timeout_seconds(
        module.DEFAULT_RUNTIME_TIMEOUT_SECONDS
    ) == pytest.approx(module.DEFAULT_RUNTIME_TIMEOUT_SECONDS)


@pytest.mark.parametrize("value", [899.0, 43_201.0, float("nan"), float("inf")])
def test_outer_runtime_timeout_rejects_unbounded_or_unsafe_values(value: float) -> None:
    module = _load_runtime_acceptance_module()

    with pytest.raises(ValueError, match="runtime acceptance timeout"):
        module._validated_runtime_timeout_seconds(value)


def test_development_runtime_requires_explicit_current_ros_workspace(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    runtime = tmp_path / "resources" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "runtime-manifest.json").write_text(
        json.dumps({"development_only": True}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="DEVELOPMENT_RUNTIME_REQUIRES_CURRENT_ROS_WORKSPACE",
    ):
        module._resolve_ros_workspace(tmp_path / "resources", None)


# 功能：
#   验证显式工作区先通过来源与安装校验，只隔离宿主 Git 元数据和 Windows 盘符映射。
# 输入：
#   tmp_path、monkeypatch、isolated_source_repository：测试目录、替换工具和当前源码副本。
# 输出：
#   None：无返回值。
def test_ros_workspace_override_validates_root(tmp_path, monkeypatch, isolated_source_repository):
    module = _load_runtime_acceptance_module()
    monkeypatch.setattr(module, "__file__", str(
        isolated_source_repository / "scripts/run_pluginized_runtime_acceptance.py"))
    if os.name != "nt":
        monkeypatch.setattr(module, "_wsl_path", Path.as_posix)
    resources = tmp_path / "resources"
    runtime = resources / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "runtime-manifest.json").write_text(
        json.dumps({"development_only": True}),
        encoding="utf-8",
    )
    workspace = tmp_path / "ros-workspace"
    install = workspace / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# current workspace\n", encoding="utf-8")
    write_ros_workspace_provenance(isolated_source_repository, workspace)

    assert module._resolve_ros_workspace(resources, workspace) == module._wsl_path(
        workspace
    )
    with pytest.raises(
        ValueError,
        match="ROS_WORKSPACE_OVERRIDE_MUST_NAME_WORKSPACE_ROOT",
    ):
        module._resolve_ros_workspace(resources, install)


def test_ros_workspace_override_requires_install_setup(tmp_path: Path) -> None:
    module = _load_runtime_acceptance_module()

    with pytest.raises(FileNotFoundError, match="ROS workspace setup is missing"):
        module._resolve_ros_workspace(tmp_path / "resources", tmp_path / "empty")


def test_outer_timeout_requests_landing_before_forced_process_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runtime_acceptance_module()
    events: list[str] = []

    class FakeProcess:
        returncode = 2

        def __init__(self) -> None:
            self.calls = 0

        def communicate(self, input=None, timeout=None):
            self.calls += 1
            events.append(
                "abort-present" if (tmp_path / "live_abort.request.json").is_file() else "no-abort"
            )
            if self.calls == 1:
                raise module.subprocess.TimeoutExpired("wsl.exe", timeout)
            return b"", b""

        def kill(self) -> None:
            events.append("killed")

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    with pytest.raises(RuntimeError, match="controlled landing cleanup completed"):
        module._run_wsl_runtime(
            script="exit 0\n",
            simulation_root=tmp_path,
            timeout_seconds=900.0,
            landing_grace_seconds=1.0,
        )

    assert events == ["no-abort", "abort-present"]
    request = json.loads((tmp_path / "live_abort.request.json").read_text())
    assert request["reason"] == "OUTER_RUNTIME_TIMEOUT"
    assert request["world_paused"] is False


def test_runtime_acceptance_heading_control_is_explicit_and_bounded() -> None:
    module = _load_runtime_acceptance_module()
    assert module._validated_heading_control("measured-hold", 20) == (
        "measured-hold",
        20.0,
    )
    assert module._validated_heading_control("route-tangent-relative", 12.5) == (
        "route-tangent-relative",
        12.5,
    )
    with pytest.raises(ValueError, match="unsupported PX4 heading policy"):
        module._validated_heading_control("unbounded", 20)
    for value in (0, 90.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="maximum yaw rate"):
            module._validated_heading_control("route-tangent-relative", value)


def test_runtime_acceptance_requires_matching_px4_heading_evidence() -> None:
    module = _load_runtime_acceptance_module()
    timing = {
        "takeoff_gate": {
            "heading_control": {
                "policy": "measured_origin_route_tangent_rate_limited",
                "maximum_yaw_rate_deg_s": 15.0,
            }
        }
    }
    assert module._validated_heading_evidence(
        timing,
        heading_policy="route-tangent-relative",
        maximum_yaw_rate_deg_s=15.0,
    ) == timing["takeoff_gate"]["heading_control"]
    with pytest.raises(RuntimeError, match="policy binding changed"):
        module._validated_heading_evidence(
            timing,
            heading_policy="measured-hold",
            maximum_yaw_rate_deg_s=20.0,
        )
    with pytest.raises(RuntimeError, match="maximum yaw-rate binding changed"):
        module._validated_heading_evidence(
            timing,
            heading_policy="route-tangent-relative",
            maximum_yaw_rate_deg_s=20.0,
        )


def test_runtime_acceptance_requires_observed_route_tangent_heading_accuracy() -> None:
    module = _load_runtime_acceptance_module()
    evidence = {
        "policy": "route-tangent-relative",
        "status": "observed",
        "sample_count": 20,
        "maximum_sample_age_seconds": 0.1,
        "absolute_error_deg": {"p99": 12.0},
    }
    timing = {"heading_tracking": evidence}

    assert module._validated_heading_tracking_evidence(
        timing,
        heading_policy="route-tangent-relative",
    ) == evidence
    assert (
        module._validated_heading_tracking_evidence(
            {},
            heading_policy="measured-hold",
        )
        is None
    )

    with pytest.raises(RuntimeError, match="too few observed samples"):
        module._validated_heading_tracking_evidence(
            {
                "heading_tracking": {
                    **evidence,
                    "sample_count": 19,
                }
            },
            heading_policy="route-tangent-relative",
        )
    with pytest.raises(RuntimeError, match="P99 heading error"):
        module._validated_heading_tracking_evidence(
            {
                "heading_tracking": {
                    **evidence,
                    "absolute_error_deg": {"p99": 45.1},
                }
            },
            heading_policy="route-tangent-relative",
        )


def test_runtime_acceptance_can_require_fresh_px4_dynamics(tmp_path: Path) -> None:
    module = _load_runtime_acceptance_module()
    state_root = tmp_path / "runtime-state"
    state_root.mkdir()
    dynamics = {
        "schema_version": "dronedream.px4-dynamics-telemetry.v1",
        "maximum_sample_age_seconds": 3.0,
        "ready_for_payload_inference": True,
        "sources": {
            "imu": {"sample_age_seconds": 0.01},
            "attitude": {"sample_age_seconds": 0.02},
            "battery": {"sample_age_seconds": 0.03},
        },
        "issue_codes": [],
        "restart_counts": {},
    }
    telemetry_path = state_root / "px4-identity-telemetry.json"
    telemetry_path.write_text(json.dumps({"dynamics": dynamics}), encoding="utf-8")
    timing = {
        "preflight_connection": {
            "successful_attempt": 2,
            "attempts": [
                {
                    "attempt": 2,
                    "status": "ready",
                    "dynamics_telemetry_rates": {
                        "required_rate_requests_succeeded": True,
                        "sources": {
                            "position_velocity": {"requested_rate_hz": 50.0, "status": "requested"},
                            "odometry": {"requested_rate_hz": 50.0, "status": "requested"},
                            "imu": {
                                "requested_rate_hz": 50.0,
                                "status": "requested",
                            },
                            "attitude": {
                                "requested_rate_hz": 50.0,
                                "status": "requested",
                            },
                            "battery": {
                                "requested_rate_hz": 2.0,
                                "status": "requested",
                            },
                        },
                    },
                }
            ],
        }
    }

    assert module._validated_px4_dynamics_evidence(
        tmp_path,
        required=True,
        timing=timing,
    ) == dynamics

    with pytest.raises(RuntimeError, match="rate evidence is missing"):
        module._validated_px4_dynamics_evidence(
            tmp_path,
            required=True,
        )
    battery_rate = timing["preflight_connection"]["attempts"][0][
        "dynamics_telemetry_rates"
    ]["sources"]["battery"]
    battery_rate["requested_rate_hz"] = 1.0
    with pytest.raises(RuntimeError, match="rate was not established: battery"):
        module._validated_px4_dynamics_evidence(
            tmp_path,
            required=True,
            timing=timing,
        )
    battery_rate["requested_rate_hz"] = 2.0

    telemetry_path.write_text(
        json.dumps(
            {
                "dynamics": {
                    **dynamics,
                    "ready_for_payload_inference": False,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not ready for payload inference"):
        module._validated_px4_dynamics_evidence(
            tmp_path,
            required=True,
            timing=timing,
        )
    assert module._validated_px4_dynamics_evidence(
        tmp_path,
        required=False,
    )["ready_for_payload_inference"] is False

    telemetry_path.unlink()
    with pytest.raises(RuntimeError, match="evidence is missing"):
        module._validated_px4_dynamics_evidence(tmp_path, required=True)
    assert module._validated_px4_dynamics_evidence(tmp_path, required=False) is None


def _write_payload_inference_dynamics_evidence(
    root: Path,
    *,
    stale_source: str | None = None,
    break_call_binding: bool = False,
) -> None:
    snapshot = {
        "schema_version": "dronedream.text-navigation-snapshot.v1",
        "strategic_context": {
            "payload": {
                "state": "loaded-stable",
                "dynamics": {
                    "available": True,
                    "ready": True,
                    "issue_codes": [],
                    "maximum_sample_age_seconds": 3.0,
                    "source_sample_age_seconds": {
                        "actuator_output": 0.1,
                        "imu": 0.2,
                        "attitude": 0.3,
                        "battery": 3.1 if stale_source == "battery" else 0.4,
                    },
                    "telemetry_payload_sha256": "a" * 64,
                    "telemetry_schema_version": (
                        "dronedream.px4-dynamics-telemetry.v1"
                    ),
                },
            }
        },
    }
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    call_id = "model-payload-dynamics-proof"
    call_input = {"text_navigation_snapshot": snapshot}
    records = {
        "model-navigation-snapshots.jsonl": [{"snapshot": snapshot}],
        "model-navigation-cycles.jsonl": [
            {
                "sequence": 4,
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "model_call_id": call_id,
            }
        ],
        "model-navigation-model-calls.jsonl": [
            {
                "call_id": call_id,
                "provider": "local-policy",
                "role": "local_navigation_advisor",
                "input_sha256": (
                    "b" * 64 if break_call_binding else sha256_json(call_input)
                ),
                "local_expert_trace": {
                    "schema_version": "dronedream.local-expert-inference-trace.v1",
                    "expert_latency_ms": {"payload-dynamics-adapter": 0.04},
                    "advisory_risk_scores": {"payload-dynamics-adapter": 0.2},
                },
            }
        ],
    }
    for name, values in records.items():
        (root / name).write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )


def test_runtime_acceptance_uses_hash_bound_inflight_dynamics_after_landing(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    state_root = tmp_path / "runtime-state"
    state_root.mkdir()
    (state_root / "px4-identity-telemetry.json").write_text(
        json.dumps(
            {
                "dynamics": {
                    "schema_version": "dronedream.px4-dynamics-telemetry.v1",
                    "maximum_sample_age_seconds": 3.0,
                    "ready_for_payload_inference": False,
                    "sources": {
                        "imu": {"sample_age_seconds": 0.1},
                        "attitude": {"sample_age_seconds": 0.2},
                        "battery": {"sample_age_seconds": 3.2},
                    },
                    "issue_codes": ["battery:SAMPLE_STALE"],
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime-phase.json").write_text(
        json.dumps({"phase": "COMPLETE"}), encoding="utf-8"
    )
    (tmp_path / "native-terminal-lifecycle.json").write_text(
        json.dumps(
            {
                "terminal_state": "ON_GROUND",
                "executor_return_code": 0,
                "publisher_exit_code": 0,
                "landing_confirmed": True,
                "safe_to_stop_watchdog": True,
            }
        ),
        encoding="utf-8",
    )
    _write_payload_inference_dynamics_evidence(tmp_path)
    timing = {
        "preflight_connection": {
            "successful_attempt": 1,
            "attempts": [
                {
                    "attempt": 1,
                    "status": "ready",
                    "dynamics_telemetry_rates": {
                        "required_rate_requests_succeeded": True,
                        "sources": {
                            "position_velocity": {"requested_rate_hz": 50.0, "status": "requested"},
                            "odometry": {"requested_rate_hz": 50.0, "status": "requested"},
                            "imu": {"requested_rate_hz": 50.0, "status": "requested"},
                            "attitude": {
                                "requested_rate_hz": 50.0,
                                "status": "requested",
                            },
                            "battery": {
                                "requested_rate_hz": 2.0,
                                "status": "requested",
                            },
                        },
                    },
                }
            ],
        }
    }

    result = module._validated_px4_dynamics_evidence(
        tmp_path, required=True, timing=timing
    )

    assert result["ready_for_payload_inference"] is True
    assert result["evidence_source"] == "hash-bound-payload-expert-inference-ledger"
    assert result["payload_inference_evidence"]["payload_expert_call_count"] == 1
    assert result["terminal_snapshot"]["recorded_after_confirmed_landing"] is True


@pytest.mark.parametrize(
    ("stale_source", "break_call_binding", "error"),
    [
        ("battery", False, "consumed stale dynamics source: battery"),
        (None, True, "call input is not bound to its snapshot"),
    ],
)
def test_payload_inference_dynamics_proof_rejects_unsafe_or_unbound_evidence(
    tmp_path: Path,
    stale_source: str | None,
    break_call_binding: bool,
    error: str,
) -> None:
    module = _load_runtime_acceptance_module()
    _write_payload_inference_dynamics_evidence(
        tmp_path,
        stale_source=stale_source,
        break_call_binding=break_call_binding,
    )

    with pytest.raises(RuntimeError, match=error):
        module._validated_payload_inference_dynamics_evidence(tmp_path)


def _write_development_payload_collection_evidence(
    root: Path,
    *,
    stale_first_sample: bool = False,
    sample_count: int = 50,
    direct_pilot_control: bool = False,
) -> dict[str, object]:
    snapshots = []
    cycles = []
    calls = []
    for index in range(sample_count):
        digest = f"{index + 1:064x}"
        call_id = f"model-{index:024d}"
        snapshots.append(
            {
                "snapshot": {
                    "snapshot_sha256": digest,
                    "strategic_context": {
                        "payload": {
                            "state": "loaded-stable",
                            "dynamics": {
                                "available": True,
                                "ready": not (stale_first_sample and index == 0),
                            },
                        }
                    },
                }
            }
        )
        cycle = {
                "snapshot_sha256": digest,
                "model_action": (
                    "pilot-control" if direct_pilot_control else "select-candidate"
                ),
                "model_call_id": call_id,
                "controller_step_scale": 0.2,
                "selected_candidate_id": (
                    None
                    if direct_pilot_control
                    else f"candidate-{index:024d}"
                ),
                "controller_target_m": {"x": 1.0, "y": 0.0, "z": 1.0},
            }
        cycles.append(cycle)
        calls.append(
            {
                "call_id": call_id,
                "local_expert_trace": {
                    "controller_step_scale": 0.2,
                    "navigation_action": (
                        "pilot-control" if direct_pilot_control else "select-candidate"
                    ),
                    "pilot_control": (
                        {
                            "forward": 0.4,
                            "right": -0.1,
                            "up": 0.05,
                            "yaw": 0.0,
                        }
                        if direct_pilot_control
                        else None
                    ),
                },
            }
        )
    for path, records in (
        (root / "model-navigation-snapshots.jsonl", snapshots),
        (root / "model-navigation-cycles.jsonl", cycles),
        (root / "model-navigation-model-calls.jsonl", calls),
    ):
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    return {
        "gates": {"development_payload_collection_absent": False},
        "measurements": {
            "development_payload_collection": {
                "enabled": True,
                "development_only": True,
                "flight_qualification_granted": False,
                "maximum_controller_step_scale": 0.2,
                "fresh_px4_dynamics_required": True,
            }
        },
    }


def test_development_payload_collection_proves_bounded_fresh_motion(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    runtime_evidence = _write_development_payload_collection_evidence(tmp_path)

    result = module._validated_development_payload_collection_evidence(
        tmp_path,
        runtime_evidence=runtime_evidence,
    )

    assert result["status"] == "verified-development-data"
    assert result["flight_qualification_granted"] is False
    assert result["bounded_motion_cycle_count"] == 50


def test_development_payload_collection_counts_direct_pilot_control_motion(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    runtime_evidence = _write_development_payload_collection_evidence(
        tmp_path,
        direct_pilot_control=True,
    )

    result = module._validated_development_payload_collection_evidence(
        tmp_path,
        runtime_evidence=runtime_evidence,
    )

    assert result["bounded_motion_cycle_count"] == 50


def test_development_payload_collection_allows_disabled_optional_visual_gates(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    runtime_evidence = _write_development_payload_collection_evidence(tmp_path)
    runtime_evidence["gates"].update(
        {
            "model_navigation_visual_frame_recorded": False,
            "semantic_supervision_recorded_for_every_sample": False,
        }
    )
    runtime_evidence["measurements"].update(
        {
            "model_navigation": {"visual_enabled": False},
            "multimodal_dataset": {"enabled": True, "semantic_topic": None},
        }
    )

    result = module._validated_development_payload_collection_evidence(
        tmp_path,
        runtime_evidence=runtime_evidence,
    )

    assert result["status"] == "verified-development-data"


def test_development_payload_collection_rejects_stale_dynamics_motion(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    runtime_evidence = _write_development_payload_collection_evidence(
        tmp_path,
        stale_first_sample=True,
    )

    with pytest.raises(RuntimeError, match="unsafe payload motion"):
        module._validated_development_payload_collection_evidence(
            tmp_path,
            runtime_evidence=runtime_evidence,
        )


def test_development_payload_collection_does_not_count_fail_closed_hold_as_motion(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    runtime_evidence = _write_development_payload_collection_evidence(
        tmp_path,
        stale_first_sample=True,
        sample_count=51,
    )
    cycles_path = tmp_path / "model-navigation-cycles.jsonl"
    cycles = [json.loads(line) for line in cycles_path.read_text().splitlines()]
    cycles[0].update(
        {
            "controller_step_scale": None,
            "selected_candidate_id": None,
            "controller_target_m": None,
            "hold_reason": "MODEL_NAVIGATION_GOAL_CHANGED_DURING_CALL",
        }
    )
    cycles_path.write_text(
        "".join(json.dumps(record) + "\n" for record in cycles),
        encoding="utf-8",
    )

    result = module._validated_development_payload_collection_evidence(
        tmp_path,
        runtime_evidence=runtime_evidence,
    )

    assert result["bounded_motion_cycle_count"] == 50
    assert result["stale_dynamics_motion_cycle_count"] == 0
    assert result["oversized_motion_cycle_count"] == 0


def test_development_payload_collection_rejects_unrelated_failed_gate(
    tmp_path: Path,
) -> None:
    module = _load_runtime_acceptance_module()
    runtime_evidence = _write_development_payload_collection_evidence(tmp_path)
    runtime_evidence["gates"]["landing_confirmed"] = False

    with pytest.raises(RuntimeError, match="unrelated failed runtime gates"):
        module._validated_development_payload_collection_evidence(
            tmp_path,
            runtime_evidence=runtime_evidence,
        )


def test_runtime_acceptance_requires_disjoint_host_tier_cpu_sets(tmp_path: Path) -> None:
    module = _load_runtime_acceptance_module()
    preflight_path = tmp_path / "native-runtime-preflight-ready.json"
    affinity = {
        "enabled": True,
        "allowed_cpu_ids": list(range(32)),
        "general_cpu_ids": list(range(28)),
        "local_model_cpu_ids": [28, 29, 30, 31],
    }
    preflight_path.write_text(
        json.dumps({"runtime_cpu_affinity": affinity}),
        encoding="utf-8",
    )

    assert module._validated_runtime_cpu_affinity_evidence(
        tmp_path,
        local_navigation_provider="local-policy",
    ) == affinity
    assert (
        module._validated_runtime_cpu_affinity_evidence(
            tmp_path,
            local_navigation_provider="kimi",
        )
        is None
    )

    affinity["general_cpu_ids"] = list(range(30))
    affinity["local_model_cpu_ids"] = [30, 31]
    preflight_path.write_text(
        json.dumps({"runtime_cpu_affinity": affinity}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="below the host-tier minimum"):
        module._validated_runtime_cpu_affinity_evidence(
            tmp_path,
            local_navigation_provider="local-policy",
        )


def test_runtime_acceptance_can_require_synchronized_multimodal_records() -> None:
    module = _load_runtime_acceptance_module()
    evidence = {
        "enabled": True,
        "record_count": 42,
        "summary": {
            "record_count": 42,
            "issue_code": None,
            "qualification_granted": False,
        },
    }

    assert module._validated_multimodal_dataset_evidence(
        {"multimodal_dataset": evidence},
        required=True,
    ) == evidence
    assert (
        module._validated_multimodal_dataset_evidence({}, required=False) is None
    )

    with pytest.raises(RuntimeError, match="contains no synchronized records"):
        module._validated_multimodal_dataset_evidence(
            {
                "multimodal_dataset": {
                    **evidence,
                    "record_count": 0,
                }
            },
            required=True,
        )
