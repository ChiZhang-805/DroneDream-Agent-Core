"""Run a real PX4/Gazebo School Map route under live depth safety.

The input metric route is produced from qualified collision geometry.  This
script turns either a prefix of that route or the complete route into a closed
round trip, revalidates the complete vehicle envelope against the exact
semantic artifact, and only then starts the native PX4/Gazebo executor.  The
executor is configured to require the live x500 depth-safety sidecar; losing
that sidecar therefore lands instead of silently falling back to a blind run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from dronedream_agent_core.collision import validate_route_clearance
from dronedream_agent_core.contracts import (
    GraphRoute,
    MapAsset,
    MapEdge,
    MapNode,
    VehicleAsset,
)
from dronedream_agent_core.gazebo_adapter import run_px4_gazebo_track
from dronedream_agent_core.px4_track import route_to_px4_track
from dronedream_agent_core.runtime_scheduling import (
    configure_sensor_thread_handoff,
    retained_interpreter_baseline,
)
from dronedream_agent_core.simulation_camera_profile import SIMULATION_CAMERA_CHOICES


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dynamic_obstacle_challenge(
    receipt_path: Path,
    *,
    world_sdf: Path,
) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version")
        != "dronedream.dynamic-obstacle-world-receipt.v1"
        or receipt.get("intended_use") != "simulation-recovery-challenge"
        or receipt.get("qualification_granted") is not False
        or receipt.get("output_world_sha256") != _sha256(world_sdf)
    ):
        raise ValueError("dynamic obstacle challenge receipt is invalid")
    obstacles = receipt.get("obstacles")
    if not isinstance(obstacles, list) or not obstacles:
        raise ValueError("dynamic obstacle challenge has no obstacles")
    entity_names: list[str] = []
    for obstacle in obstacles:
        if not isinstance(obstacle, dict):
            raise ValueError("dynamic obstacle challenge entry is invalid")
        entity_name = obstacle.get("entity_name")
        if (
            not isinstance(entity_name, str)
            or not entity_name.startswith("dronedream_dynamic_")
            or obstacle.get("physical_collision") is not True
            or obstacle.get("visible_to_camera") is not True
            or obstacle.get("published_as_dynamic_obstacle") is not True
        ):
            raise ValueError("dynamic obstacle challenge contract is incomplete")
        entity_names.append(entity_name)
    if len(entity_names) != len(set(entity_names)):
        raise ValueError("dynamic obstacle challenge entity names are not unique")
    required_encounter_distance_m = float(
        receipt.get("required_encounter_distance_m", 12.0)
    )
    if not (
        math.isfinite(required_encounter_distance_m)
        and 0.5 <= required_encounter_distance_m <= 50.0
    ):
        raise ValueError("dynamic obstacle challenge encounter distance is invalid")
    return {
        "challenge_id": receipt.get("challenge_id"),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "world_sha256": receipt["output_world_sha256"],
        "entity_names": entity_names,
        "required_encounter_distance_m": required_encounter_distance_m,
    }


def _dynamic_obstacle_observation_metrics(
    history_path: Path,
    *,
    entity_names: list[str],
) -> dict[str, dict[str, int | float | None]]:
    metrics: dict[str, dict[str, int | float | None]] = {
        name: {
            "observation_count": 0,
            "threat_count": 0,
            "minimum_horizontal_distance_m": None,
        }
        for name in entity_names
    }
    if not history_path.is_file():
        return metrics
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        observation = record.get("observation")
        obstacles = (
            observation.get("dynamic_obstacles")
            if isinstance(observation, dict)
            else None
        )
        if not isinstance(obstacles, list):
            continue
        current_position = observation.get("current_position_m")
        for item in obstacles:
            if not isinstance(item, dict):
                continue
            name = item.get("obstacle_id")
            if name not in metrics:
                continue
            entry = metrics[name]
            entry["observation_count"] = int(entry["observation_count"] or 0) + 1
            obstacle_position = item.get("position_m")
            if not (
                isinstance(current_position, dict)
                and isinstance(obstacle_position, dict)
            ):
                continue
            try:
                distance_m = math.hypot(
                    float(current_position["x"]) - float(obstacle_position["x"]),
                    float(current_position["y"]) - float(obstacle_position["y"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            previous = entry["minimum_horizontal_distance_m"]
            if previous is None or distance_m < float(previous):
                entry["minimum_horizontal_distance_m"] = distance_m
        command = record.get("command")
        decision = command.get("decision") if isinstance(command, dict) else None
        threat_name = (
            decision.get("threat_obstacle_id")
            if isinstance(decision, dict)
            else None
        )
        if threat_name in metrics:
            entry = metrics[threat_name]
            entry["threat_count"] = int(entry["threat_count"] or 0) + 1
    return metrics


def _dynamic_obstacle_observation_counts(
    history_path: Path,
    *,
    entity_names: list[str],
) -> dict[str, int]:
    metrics = _dynamic_obstacle_observation_metrics(
        history_path,
        entity_names=entity_names,
    )
    return {
        name: int(entry["observation_count"] or 0)
        for name, entry in metrics.items()
    }


def _load_metric_route(path: Path) -> GraphRoute:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metric route evidence is not an object")
    route_payload = payload.get("route", payload)
    if not isinstance(route_payload, dict):
        raise ValueError("metric route evidence has no route object")
    return GraphRoute.model_validate(route_payload)


def _closed_qualification_route(
    outbound: GraphRoute,
    *,
    point_count: int | None,
    speed_limit_mps: float,
    round_trip_repetitions: int = 1,
) -> tuple[GraphRoute, MapAsset]:
    selected = list(outbound.positions_m)
    if point_count is not None:
        if point_count < 2:
            raise ValueError("short qualification needs at least two outbound points")
        selected = selected[:point_count]
    if len(selected) < 2:
        raise ValueError("metric route has fewer than two points")

    outbound_ids = ["qualification-launch"] + [
        f"qualification-out-{index:03d}" for index in range(1, len(selected))
    ]
    one_round_trip_node_ids = [*outbound_ids, *reversed(outbound_ids[:-1])]
    one_round_trip_positions = [*selected, *reversed(selected[:-1])]
    node_ids = list(one_round_trip_node_ids)
    positions = list(one_round_trip_positions)
    for _ in range(1, round_trip_repetitions):
        # The preceding trip already ends at launch.  Excluding the repeated
        # first point avoids a zero-length edge while retaining a real stable
        # arrival at launch before the next departure.
        node_ids.extend(one_round_trip_node_ids[1:])
        positions.extend(one_round_trip_positions[1:])
    edge_ids = [
        f"qualification-edge-{index:03d}" for index in range(len(positions) - 1)
    ]
    nodes = []
    for index, (node_id, position) in enumerate(zip(outbound_ids, selected, strict=True)):
        if index == 0:
            semantic = "launch"
        elif index == len(selected) - 1 and point_count is None:
            semantic = "pickup"
        else:
            semantic = "corridor"
        nodes.append(
            MapNode(
                node_id=node_id,
                label=("Qualification launch" if index == 0 else f"Metric route point {index}"),
                position_m=position,
                semantic=semantic,
            )
        )
    edges = []
    for index, (first, second) in enumerate(zip(node_ids, node_ids[1:], strict=False)):
        first_position = positions[index]
        second_position = positions[index + 1]
        edges.append(
            MapEdge(
                edge_id=edge_ids[index],
                from_node=first,
                to_node=second,
                distance_m=math.dist(
                    (first_position.x, first_position.y, first_position.z),
                    (second_position.x, second_position.y, second_position.z),
                ),
                minimum_clearance_m=0.4,
                speed_limit_mps=speed_limit_mps,
                bidirectional=False,
                qualification="geometry-derived",
            )
        )
    route = GraphRoute(
        start_node="qualification-launch",
        goal_node="qualification-launch",
        node_ids=node_ids,
        edge_ids=edge_ids,
        positions_m=positions,
        route_length_m=sum(edge.distance_m for edge in edges),
        all_edges_flight_verified=False,
    )
    graph = MapAsset(
        asset_id="dronedream.school-map.depth-qualification",
        name="School Map live-depth qualification route",
        nodes=nodes,
        edges=edges,
        named_entities={
            "qualification-launch": "qualification-launch",
            "qualification-turn": outbound_ids[-1],
        },
    )
    return route, graph


def _load_vehicle(path: Path, *, vehicle_sdf: Path) -> VehicleAsset:
    """Load the exact vehicle contract that will be flown.

    Qualification must not reconstruct vehicle geometry, payload limits, or
    range from constants that belonged to an older development asset.
    """

    vehicle = VehicleAsset.model_validate_json(path.read_text(encoding="utf-8"))
    sdf = vehicle_sdf.read_text(encoding="utf-8")
    if "oakd-lite-depth" not in vehicle.sensors or "model://x500_depth" not in sdf:
        raise ValueError("qualification requires the current x500 depth vehicle")
    return vehicle


def _validate_training_capture(*, training: bool, dataset: Path | None,
                               learner_channel: Path | None) -> None:
    # A PPO rollout retains model-camera evidence and causal transition receipts.
    # Recording a second full-resolution vision-training corpus is optional;
    # ordinary teacher collection must still record that corpus explicitly.
    if training and dataset is None and learner_channel is None:
        raise ValueError("training requires a multimodal dataset or an explicit learner channel")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metric_route", type=Path)
    parser.add_argument("semantic", type=Path)
    parser.add_argument("world_sdf", type=Path)
    parser.add_argument("vehicle_sdf", type=Path)
    parser.add_argument("vehicle_metadata", type=Path)
    parser.add_argument("controller_params", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--short-outbound-points",
        type=int,
        help="Use only this many outbound points before returning to launch.",
    )
    parser.add_argument(
        "--round-trip-repetitions",
        type=int,
        default=1,
        help=(
            "Repeat the selected closed route within one training-data flight. "
            "Production qualification requires exactly one round trip."
        ),
    )
    parser.add_argument("--speed-limit-mps", type=float, default=0.8)
    parser.add_argument("--required-clearance-m", type=float, default=0.4)
    parser.add_argument(
        "--training-data-collection",
        action="store_true",
        help=(
            "Permit an explicitly non-qualifying narrow-passage collection run; "
            "the complete vehicle envelope must still remain collision-free."
        ),
    )
    parser.add_argument(
        "--teacher-observation-collection",
        action="store_true",
        help=(
            "Record current deployment observations in a non-qualifying teacher "
            "simulation without loading obsolete policy weights or invoking a model."
        ),
    )
    parser.add_argument("--learning-image-size", type=int, nargs=2, default=(224, 128))
    parser.add_argument("--simulation-camera-profile",
                        choices=SIMULATION_CAMERA_CHOICES,
                        default="native")
    parser.add_argument("--camera-source-model-sha256")
    parser.add_argument("--native-sensor-runtime", type=Path,
        help="Explicit native sensor build; training verifies it against current sources.")
    parser.add_argument("--render-replica-runtime", type=Path,
                        help="Source-bound isolated sensor renderer for explicit visual training")
    parser.add_argument("--preflight-render-warmup", action="store_true",
        help="Explicit training-only camera preparation before spawning the aircraft.")
    parser.add_argument("--preflight-depth-warmup", action="store_true",
        help="Also prepare depth; retain the unsubscribed diagnostic rig until Gazebo exits.")
    parser.add_argument("--render-preparation-runtime", type=Path,
        help="Explicit verified native render preparation build; no installed files are changed.")
    parser.add_argument("--render-cache-bundle", type=Path,
        help="Source-matched cache from a stopped simulation; never an automatic latest path.")
    parser.add_argument("--takeoff-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--heading-policy",
        choices=("measured-hold", "route-tangent-relative"),
        default="route-tangent-relative",
    )
    parser.add_argument("--maximum-yaw-rate-deg-s", type=float, default=20.0)
    parser.add_argument(
        "--local-safety-runtime-stale-grace-seconds",
        type=float,
        default=8.0,
        help=(
            "Maximum bounded position-hold interval for a transient local-safety "
            "lease outage before controlled landing."
        ),
    )
    parser.add_argument("--local-navigation-provider")
    parser.add_argument("--simulation-training-channel", type=Path)
    parser.add_argument("--batch-static-world-visuals", action="store_true",
        help="Batch eligible static visuals without changing collisions or sensors.")
    parser.add_argument("--local-navigation-fallback-provider")
    parser.add_argument(
        "--local-policy-package",
        type=Path,
        action="append",
        default=[],
        help="Content-bound local policy package directory; repeat for rollback candidates.",
    )
    parser.add_argument(
        "--local-policy-qualification",
        type=Path,
        action="append",
        default=[],
        help="Production qualification receipt; repeat when several packages are supplied.",
    )
    parser.add_argument(
        "--local-policy-simulation-admission",
        type=Path,
        action="append",
        default=[],
        help="Simulation-only admission receipt; never grants production eligibility.",
    )
    parser.add_argument("--local-navigation-model-timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--local-navigation-fallback-model-timeout-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument("--local-navigation-period-seconds", type=float, default=3.0)
    parser.add_argument(
        "--semantic-progress-recovery-timeout-seconds",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--semantic-progress-abort-timeout-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument("--local-navigation-visual-enabled", action="store_true")
    parser.add_argument("--multimodal-dataset-root", type=Path)
    parser.add_argument("--multimodal-flight-id")
    parser.add_argument("--semantic-label-topic")
    parser.add_argument("--semantic-label-map", type=Path)
    parser.add_argument("--multimodal-dataset-maximum-mib", type=int, default=5_120)
    parser.add_argument("--multimodal-record-period-seconds", type=float, default=0.1)
    parser.add_argument("--development-depth-drop-after-seconds", type=float)
    parser.add_argument("--development-depth-drop-duration-seconds", type=float)
    parser.add_argument(
        "--dynamic-obstacle-challenge-receipt",
        type=Path,
        help=(
            "Content-bound receipt for physical Gazebo recovery obstacles. "
            "Challenge missions are simulation training evidence only."
        ),
    )
    parser.add_argument(
        "--development-payload-collection",
        action="store_true",
        help=(
            "Collect a non-qualifying payload dynamics corpus using a bounded "
            "general-policy fallback until a payload adapter can be trained."
        ),
    )
    parser.add_argument("--px4-root", type=Path, default=Path("/opt/PX4-Autopilot"))
    parser.add_argument(
        "--ros-workspace",
        type=Path,
        default=(
            Path.home()
            / ".local/share/dronedream-autonomy/v0.1.0/ros_ws-merged"
        ),
    )
    args = parser.parse_args()
    if not 0.1 <= args.speed_limit_mps <= 1.2:
        raise ValueError("qualification speed must be within [0.1, 1.2] m/s")
    minimum_requested_clearance_m = 0.12 if args.training_data_collection else 0.35
    if args.required_clearance_m < minimum_requested_clearance_m:
        raise ValueError(
            "School Map requested clearance is below the active campaign bound"
        )
    _validate_training_capture(training=args.training_data_collection,
                               dataset=args.multimodal_dataset_root,
                               learner_channel=args.simulation_training_channel)
    if args.simulation_training_channel is not None:
        if (not args.training_data_collection or args.teacher_observation_collection
                or args.local_navigation_provider not in {None, "simulation-training"}):
            raise ValueError("offline learner channel is restricted to simulation training")
        args.local_navigation_provider = "simulation-training"
    if args.teacher_observation_collection and not args.training_data_collection:
        raise ValueError(
            "teacher observation collection requires --training-data-collection"
        )
    if args.teacher_observation_collection and (
        args.local_navigation_provider is not None
        or args.local_policy_package or args.local_policy_simulation_admission
        or args.local_policy_qualification
    ):
        raise ValueError(
            "teacher observations are collected without a policy provider or model package"
        )
    if args.dynamic_obstacle_challenge_receipt and not args.training_data_collection:
        raise ValueError(
            "dynamic obstacle challenges require --training-data-collection"
        )
    if not 1 <= args.round_trip_repetitions <= 10:
        raise ValueError("round-trip repetitions must be within [1, 10]")
    if args.round_trip_repetitions > 1 and not args.training_data_collection:
        raise ValueError(
            "repeated round trips are restricted to training data collection"
        )
    dynamic_obstacle_challenge = (
        _load_dynamic_obstacle_challenge(
            args.dynamic_obstacle_challenge_receipt,
            world_sdf=args.world_sdf,
        )
        if args.dynamic_obstacle_challenge_receipt
        else None
    )
    if args.development_payload_collection:
        if not args.training_data_collection:
            raise ValueError(
                "development payload collection requires --training-data-collection"
            )
        if args.local_navigation_provider != "local-policy":
            raise ValueError("development payload collection requires local-policy")
        if not args.local_policy_simulation_admission or args.local_policy_qualification:
            raise ValueError(
                "development payload collection requires simulation admission only"
            )
    if not 90.0 <= args.takeoff_timeout_seconds <= 600.0:
        raise ValueError("takeoff timeout must be within [90, 600] seconds")
    if not 1.0 <= args.maximum_yaw_rate_deg_s <= 45.0:
        raise ValueError("qualification yaw rate must be within [1, 45] deg/s")
    if not 1.0 <= args.local_safety_runtime_stale_grace_seconds <= 30.0:
        raise ValueError("local safety stale grace must be within [1, 30] seconds")
    if args.local_navigation_model_timeout_seconds <= 0.0:
        raise ValueError("local navigation model timeout must be positive")
    if args.local_navigation_fallback_model_timeout_seconds <= 0.0:
        raise ValueError("local navigation fallback model timeout must be positive")
    if args.local_navigation_period_seconds <= 0.0:
        raise ValueError("local navigation period must be positive")
    if args.semantic_progress_recovery_timeout_seconds <= 0.0:
        raise ValueError("semantic progress recovery timeout must be positive")
    if (
        args.semantic_progress_abort_timeout_seconds
        <= args.semantic_progress_recovery_timeout_seconds
    ):
        raise ValueError(
            "semantic progress abort timeout must exceed the recovery timeout"
        )
    if (args.development_depth_drop_after_seconds is None) != (
        args.development_depth_drop_duration_seconds is None
    ):
        raise ValueError("development depth-drop injection requires after and duration")
    if (
        args.development_depth_drop_after_seconds is not None
        and args.development_depth_drop_after_seconds < 0.0
    ):
        raise ValueError("development depth-drop delay must be non-negative")
    if (
        args.development_depth_drop_duration_seconds is not None
        and args.development_depth_drop_duration_seconds <= 0.0
    ):
        raise ValueError("development depth-drop duration must be positive")
    if (args.local_navigation_visual_enabled and not args.local_navigation_provider
            and not args.teacher_observation_collection):
        raise ValueError("visual local navigation requires --local-navigation-provider")
    if (args.multimodal_dataset_root is None) != (args.multimodal_flight_id is None):
        raise ValueError("multimodal recording requires dataset root and flight identity")
    if (args.semantic_label_topic is None) != (args.semantic_label_map is None):
        raise ValueError("semantic supervision requires topic and label map")
    if args.semantic_label_topic is not None and args.multimodal_dataset_root is None:
        raise ValueError("semantic supervision requires multimodal recording")
    if not 1 <= args.multimodal_dataset_maximum_mib <= 20 * 1024:
        raise ValueError("multimodal dataset quota is outside the safe range")
    if not 0.05 <= args.multimodal_record_period_seconds <= 10.0:
        raise ValueError("multimodal recording period is outside the safe range")
    if args.local_navigation_fallback_provider == "local-policy":
        raise ValueError("local-policy is supported only as the primary local provider")
    local_policy_artifacts_supplied = bool(
        args.local_policy_package
        or args.local_policy_qualification
        or args.local_policy_simulation_admission
    )
    if args.local_navigation_provider == "local-policy":
        if not args.local_policy_package or not (
            args.local_policy_qualification
            or args.local_policy_simulation_admission
        ):
            raise ValueError(
                "local-policy navigation requires a package and qualification "
                "or simulation-admission receipt"
            )
    elif local_policy_artifacts_supplied:
        raise ValueError("local policy artifacts require the local-policy provider")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {args.output_root}")

    outbound = _load_metric_route(args.metric_route)
    route, graph = _closed_qualification_route(
        outbound,
        point_count=args.short_outbound_points,
        speed_limit_mps=args.speed_limit_mps,
        round_trip_repetitions=args.round_trip_repetitions,
    )
    vehicle = _load_vehicle(args.vehicle_metadata, vehicle_sdf=args.vehicle_sdf)
    clearance = validate_route_clearance(
        route,
        args.semantic,
        vehicle_diameter_m=vehicle.body_radius_m * 2.0,
        vehicle_height_m=vehicle.body_height_m,
        sample_interval_m=0.05,
    )
    if not clearance.accepted:
        raise RuntimeError("closed qualification route intersects School Map collision geometry")
    if clearance.minimum_clearance_m < args.required_clearance_m - 0.03:
        raise RuntimeError(
            "closed qualification route does not preserve the operational clearance"
        )
    track = route_to_px4_track(
        route,
        graph,
        args.semantic,
        vehicle=vehicle,
        waypoint_hold_seconds=0.2,
    )

    plan_root = args.output_root / "plan"
    route_path = plan_root / "route.json"
    graph_path = plan_root / "qualification-graph.json"
    vehicle_path = plan_root / "vehicle.json"
    clearance_path = plan_root / "clearance.json"
    track_path = plan_root / "track.json"
    _write_json(route_path, route.model_dump(mode="json"))
    _write_json(graph_path, graph.model_dump(mode="json"))
    _write_json(vehicle_path, vehicle.model_dump(mode="json"))
    _write_json(clearance_path, clearance.model_dump(mode="json"))
    _write_json(track_path, track.model_dump(mode="json"))
    _write_json(
        plan_root / "qualification-plan.json",
        {
            "schema_version": "dronedream.school-map-depth-qualification-plan.v1",
            "mode": (
                "training-data-collection"
                if args.training_data_collection
                else (
                    "short-round-trip"
                    if args.short_outbound_points is not None
                    else "full-round-trip"
                )
            ),
            "operational_qualification_requested": not args.training_data_collection,
            "round_trip_repetitions": args.round_trip_repetitions,
            "live_depth_required": True,
            "text_model_input": (
                "structured-metric-state-plus-forward-rgb"
                if args.local_navigation_visual_enabled
                else "structured-metric-state-no-image-required"
            ),
            "local_navigation_provider": args.local_navigation_provider,
            "local_navigation_model_required": args.local_navigation_provider is not None,
            "local_navigation_control_authority_required": (
                args.local_navigation_provider is not None
                and not args.teacher_observation_collection
            ),
            "teacher_observation_collection": {
                "enabled": args.teacher_observation_collection,
                "observation_recorder_has_control_authority": False,
                "observation_recorder_invokes_models": False,
                "flight_qualification_granted": False,
            },
            "local_navigation_visual_enabled": args.local_navigation_visual_enabled,
            "multimodal_dataset_recording_enabled": (
                args.multimodal_dataset_root is not None
            ),
            "multimodal_flight_id": args.multimodal_flight_id,
            "semantic_supervision_enabled": args.semantic_label_topic is not None,
            "semantic_label_topic": args.semantic_label_topic,
            "multimodal_dataset_maximum_mib": args.multimodal_dataset_maximum_mib,
            "multimodal_record_period_seconds": args.multimodal_record_period_seconds,
            "local_safety_runtime_stale_grace_seconds": (
                args.local_safety_runtime_stale_grace_seconds
            ),
            "heading_policy": args.heading_policy,
            "maximum_yaw_rate_deg_s": args.maximum_yaw_rate_deg_s,
            "semantic_progress_recovery_timeout_seconds": (
                args.semantic_progress_recovery_timeout_seconds
            ),
            "semantic_progress_abort_timeout_seconds": (
                args.semantic_progress_abort_timeout_seconds
            ),
            "local_policy_package_count": len(args.local_policy_package),
            "local_policy_qualification_receipt_count": len(
                args.local_policy_qualification
            ),
            "local_policy_simulation_admission_receipt_count": len(
                args.local_policy_simulation_admission
            ),
            "development_depth_fault_injection": (
                {
                    "development_only": True,
                    "drop_after_first_accepted_frame_seconds": (
                        args.development_depth_drop_after_seconds
                    ),
                    "drop_duration_seconds": args.development_depth_drop_duration_seconds,
                }
                if args.development_depth_drop_after_seconds is not None
                else None
            ),
            "dynamic_obstacle_challenge": dynamic_obstacle_challenge,
            "development_payload_collection": {
                "enabled": args.development_payload_collection,
                "development_only": args.development_payload_collection,
                "flight_qualification_granted": False
                if args.development_payload_collection
                else None,
                "maximum_controller_step_scale": (
                    0.2 if args.development_payload_collection else None
                ),
            },
            "route_length_m": route.route_length_m,
            "route_point_count": len(route.positions_m),
            "minimum_clearance_m": clearance.minimum_clearance_m,
            "required_clearance_m": args.required_clearance_m,
        },
    )

    resource_root = Path(__file__).resolve().parents[1]
    evidence = run_px4_gazebo_track(
        run_dir=args.output_root / "simulation",
        world_sdf=args.world_sdf,
        semantic_path=args.semantic,
        vehicle_sdf=args.vehicle_sdf,
        vehicle_metadata_path=vehicle_path,
        route_path=route_path,
        track_path=track_path,
        clearance_path=clearance_path,
        controller_params_path=args.controller_params,
        px4_root=args.px4_root,
        executor_path=resource_root / "scripts" / "px4_checkpoint_executor.py",
        ros_workspace=args.ros_workspace,
        contract_id=(
            "school-map-training-data-collection"
            if args.training_data_collection
            else "school-map-live-depth-qualification"
        ),
        executor_extra_args=[
            "--base-executor",
            str(resource_root / "runtime" / "px4_offboard_track_executor.py"),
            "--takeoff-timeout-seconds",
            f"{args.takeoff_timeout_seconds:g}",
            "--local-safety-command-grace-seconds",
            # An expired command immediately holds the measured PX4 position.
            # Eight seconds covers qualified full-route static-map startup.
            # The executor holds measured PX4 position throughout this window;
            # a persistent outage still lands safely.
            "8",
            "--local-safety-runtime-stale-grace-seconds",
            # A context change invalidates the old motion command immediately.
            # The executor sends zero-velocity position hold throughout this
            # bounded recovery window. This absorbs host scheduling jitter
            # without authorizing an expired motion command; a persistent
            # outage still escalates to controlled landing.
            f"{args.local_safety_runtime_stale_grace_seconds:g}",
            "--semantic-progress-recovery-timeout-seconds",
            f"{args.semantic_progress_recovery_timeout_seconds:g}",
            "--semantic-progress-abort-timeout-seconds",
            f"{args.semantic_progress_abort_timeout_seconds:g}",
            "--heading-policy",
            args.heading_policy,
            "--maximum-yaw-rate-deg-s",
            f"{args.maximum_yaw_rate_deg_s:g}",
        ],
        # The qualification authority is the vehicle-mounted depth sensor.
        # A second 1280x720 observer camera adds render load but no safety
        # evidence, so keep it out of this physical closed-loop test.
        live_camera_enabled=False,
        local_navigation_provider=args.local_navigation_provider,
        local_navigation_fallback_provider=args.local_navigation_fallback_provider,
        local_navigation_model_timeout_seconds=(
            args.local_navigation_model_timeout_seconds
        ),
        local_navigation_fallback_model_timeout_seconds=(
            args.local_navigation_fallback_model_timeout_seconds
        ),
        local_navigation_period_seconds=args.local_navigation_period_seconds,
        local_navigation_context_id=None,
        local_navigation_visual_enabled=args.local_navigation_visual_enabled,
        local_navigation_control_authority_required=(
            args.local_navigation_provider is not None
            and not args.teacher_observation_collection
        ),
        local_navigation_omit_coordinate_candidates=(
            args.teacher_observation_collection
        ),
        record_learning_observations=args.teacher_observation_collection,
        learning_image_size=tuple(args.learning_image_size),
        simulation_teacher_control=args.teacher_observation_collection,
        simulation_training_channel=args.simulation_training_channel,
        batch_static_world_visuals=args.batch_static_world_visuals,
        simulation_camera_profile=args.simulation_camera_profile,
        preflight_render_warmup=args.preflight_render_warmup,
        preflight_depth_warmup=args.preflight_depth_warmup,
        render_preparation_runtime=args.render_preparation_runtime,
        render_cache_bundle=args.render_cache_bundle,
        render_replica_runtime=args.render_replica_runtime,
        native_sensor_runtime=args.native_sensor_runtime,
        camera_source_model_sha256=args.camera_source_model_sha256,
        heading_policy=args.heading_policy,
        maximum_yaw_rate_deg_s=args.maximum_yaw_rate_deg_s,
        multimodal_dataset_root=args.multimodal_dataset_root,
        multimodal_flight_id=args.multimodal_flight_id,
        semantic_label_topic=args.semantic_label_topic,
        semantic_label_map_path=args.semantic_label_map,
        multimodal_dataset_maximum_mib=args.multimodal_dataset_maximum_mib,
        multimodal_record_period_seconds=args.multimodal_record_period_seconds,
        local_policy_package_paths=tuple(args.local_policy_package),
        local_policy_qualification_paths=tuple(args.local_policy_qualification),
        local_policy_simulation_admission_paths=tuple(
            args.local_policy_simulation_admission
        ),
        development_depth_drop_after_seconds=(
            args.development_depth_drop_after_seconds
        ),
        development_depth_drop_duration_seconds=(
            args.development_depth_drop_duration_seconds
        ),
        development_payload_collection=args.development_payload_collection,
    )
    if dynamic_obstacle_challenge is not None:
        entity_names = list(dynamic_obstacle_challenge["entity_names"])
        observation_metrics = _dynamic_obstacle_observation_metrics(
            args.output_root / "simulation" / "local-safety-history.jsonl",
            entity_names=entity_names,
        )
        required_encounter_distance_m = float(
            dynamic_obstacle_challenge["required_encounter_distance_m"]
        )
        challenge_evidence = {
            **dynamic_obstacle_challenge,
            "observation_metrics": observation_metrics,
            "all_entities_observed": all(
                int(observation_metrics[name]["observation_count"] or 0) > 0
                for name in entity_names
            ),
            "all_entities_encountered": all(
                observation_metrics[name]["minimum_horizontal_distance_m"]
                is not None
                and float(
                    observation_metrics[name]["minimum_horizontal_distance_m"]
                )
                <= required_encounter_distance_m
                for name in entity_names
            ),
            "all_entities_selected_as_threat": all(
                int(observation_metrics[name]["threat_count"] or 0) > 0
                for name in entity_names
            ),
        }
        evidence["measurements"]["dynamic_obstacle_challenge"] = challenge_evidence
        evidence["gates"]["dynamic_obstacle_challenge_receipt_recorded"] = True
        evidence["gates"]["dynamic_obstacle_challenge_entities_observed"] = bool(
            challenge_evidence["all_entities_observed"]
        )
        evidence["gates"]["dynamic_obstacle_challenge_entities_encountered"] = bool(
            challenge_evidence["all_entities_encountered"]
        )
        evidence["gates"]["dynamic_obstacle_challenge_entities_became_threat"] = bool(
            challenge_evidence["all_entities_selected_as_threat"]
        )
        evidence["artifacts"]["dynamic_obstacle_challenge_receipt_sha256"] = (
            dynamic_obstacle_challenge["receipt_sha256"]
        )
        evidence["status"] = (
            "verified" if all(evidence["gates"].values()) else "failed"
        )
        _write_json(
            args.output_root / "simulation" / "mission_evidence.json",
            evidence,
        )
    summary = {
        "schema_version": "dronedream.school-map-depth-qualification-summary.v1",
        "status": evidence["status"],
        "gates": evidence["gates"],
        "measurements": evidence["measurements"],
        "training_data_collection": args.training_data_collection,
        "development_payload_collection": args.development_payload_collection,
        "operational_qualification_granted": bool(
            not args.training_data_collection and evidence["status"] == "verified"
        ),
    }
    _write_json(args.output_root / "qualification-summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "route_length_m": route.route_length_m,
                "route_point_count": len(route.positions_m),
                "minimum_clearance_m": clearance.minimum_clearance_m,
                "failed_gates": [
                    key for key, accepted in evidence["gates"].items() if not accepted
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["status"] == "verified" else 1


if __name__ == "__main__":
    # Dedicated simulation process only, before loading mission graphs or
    # creating any observers. Normal collection of new cycles remains active.
    configure_sensor_thread_handoff()
    with retained_interpreter_baseline():
        raise SystemExit(main())
