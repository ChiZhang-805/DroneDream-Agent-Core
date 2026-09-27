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
import os
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
from dronedream_agent_core.payload_collection_contract import payload_collection_mode
from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_agent_core.px4_track import route_to_px4_track
from dronedream_agent_core.recovery_obstacle_evidence import (
    recovery_clearance_metrics,
    recovery_obstacle_metrics,
)
from dronedream_agent_core.runtime_scheduling import (
    configure_sensor_thread_handoff,
    retained_interpreter_baseline,
)
from dronedream_agent_core.simulation_camera_profile import SIMULATION_CAMERA_CHOICES
from dronedream_agent_core.training.payload_curriculum import build_payload_teacher_curriculum
from dronedream_plugin_sdk.protocol import decode_json


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 功能：
#   读取有限大小的严格挑战回执并绑定实际世界，拒绝歧义 JSON 与隐式数值转换。
# 输入：
#   receipt_path：挑战生成器保存的回执路径。
#   world_sdf：本轮明确选定的派生仿真世界。
# 输出：
#   challenge：唯一实体名称、场景摘要和相遇距离约束。
def _load_dynamic_obstacle_challenge(receipt_path: Path, *, world_sdf: Path) -> dict[str, object]:
    content = read_plugin_file(receipt_path, limit=4 * 1024 * 1024)
    receipt = decode_json(content, limit=4 * 1024 * 1024)
    if (
        type(receipt) is not dict or receipt.get("schema_version")
        != "dronedream.dynamic-obstacle-world-receipt.v1"
        or receipt.get("intended_use") != "simulation-recovery-challenge"
        or receipt.get("qualification_granted") is not False
        or receipt.get("output_world_sha256") != _sha256(world_sdf)
    ):
        raise ValueError("dynamic obstacle challenge receipt is invalid")
    obstacles = receipt.get("obstacles")
    if not isinstance(obstacles, list) or not 1 <= len(obstacles) <= 32:
        raise ValueError("dynamic obstacle challenge has no obstacles")
    entity_names: list[str] = []
    moving_entities = []
    for obstacle in obstacles:
        if not isinstance(obstacle, dict):
            raise ValueError("dynamic obstacle challenge entry is invalid")
        entity_name = obstacle.get("entity_name")
        if (
            not isinstance(entity_name, str)
            or not entity_name.startswith("dronedream_dynamic_")
            or len(entity_name) > 256
            or obstacle.get("physical_collision") is not True
            or obstacle.get("visible_to_camera") is not True
            or obstacle.get("published_as_dynamic_obstacle") is not True
        ):
            raise ValueError("dynamic obstacle challenge contract is incomplete")
        entity_names.append(entity_name)
        motion = obstacle.get("motion")
        if motion is not None:
            if (type(motion) is not dict or motion.get("kind") != "circle"
                    or obstacle.get("stationary") is not False):
                raise ValueError("dynamic obstacle motion contract is invalid")
            moving_entities.append(entity_name)
    if len(entity_names) != len(set(entity_names)):
        raise ValueError("dynamic obstacle challenge entity names are not unique")
    required_encounter_distance_m = receipt.get("required_encounter_distance_m")
    if not (
        type(required_encounter_distance_m) in (int, float)
        and 0.5 <= required_encounter_distance_m <= 50.0
    ):
        raise ValueError("dynamic obstacle challenge encounter distance is invalid")
    challenge = {
        "challenge_id": receipt.get("challenge_id"),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": hashlib.sha256(content).hexdigest(),
        "world_sha256": receipt["output_world_sha256"],
        "entity_names": entity_names,
        "moving_entity_names": moving_entities,
        "required_encounter_distance_m": required_encounter_distance_m,
    }
    return challenge


# 功能：
#   1. 先落盘挑战未通过状态，再核对原生感知与独立见证，防止后处理异常留下通过回执。
#   2. 只把完整证据中的真实关联计入观察、接近和威胁门槛，不恢复旧真值控制历史。
# 输入：
#   root：已经停止执行的仿真目录。
#   evidence：基础物理运行证据。
#   challenge：启动前验证的挑战资产信息。
#   vehicle：本轮加载的机体模型，用于独立包围体距离检查。
# 输出：
#   result：包含挑战门槛与检查错误的最终证据。
def _finalize_recovery_challenge(root: Path, evidence: dict, challenge: dict, vehicle: VehicleAsset) -> dict:
    result = {**evidence, "gates": dict(evidence["gates"]),
        "measurements": dict(evidence["measurements"]), "artifacts": dict(evidence["artifacts"])}
    checks = {"evidence_complete": False, "motion_observed": False, "entities_observed": False,
        "entities_encountered": False, "entities_became_threat": False, "sampled_clearance_respected": False}
    result["status"] = "failed"
    result["gates"].update({"dynamic_obstacle_challenge_" + key: value for key, value in checks.items()})
    result["measurements"]["dynamic_obstacle_challenge"] = {**challenge, "issue": "CHECK_PENDING"}
    target = root / "mission_evidence.json"
    _write_json(target, result)
    details = {**challenge, "issue": None}
    try:
        names = list(challenge["entity_names"])
        metrics = recovery_obstacle_metrics(root, names)
        clearance = recovery_clearance_metrics(root, names, vehicle,
            evidence["measurements"]["local_safety"]["required_clearance_m"])
        artifacts = {}
        for name in ("recovery-obstacle-witness.jsonl", "recovery-obstacle-witness-summary.json"):
            artifacts[name.replace(".", "_").replace("-", "_") + "_sha256"] = _sha256(root / name)
        artifacts["dynamic_obstacle_challenge_receipt_sha256"] = challenge["receipt_sha256"]
        details["observation_metrics"] = metrics
        details["clearance_metrics"] = clearance
        checks = {
            "evidence_complete": True,
            "motion_observed": all(metrics[name]["moving_observation_count"] >= 3
                for name in challenge.get("moving_entity_names", [])),
            "entities_observed": all(metrics[name]["observation_count"] > 0 for name in names),
            "entities_encountered": all(metrics[name]["minimum_horizontal_distance_m"] is not None
                and metrics[name]["minimum_horizontal_distance_m"] <= challenge["required_encounter_distance_m"] for name in names),
            "entities_became_threat": all(metrics[name]["threat_count"] > 0 for name in names),
            "sampled_clearance_respected": clearance["sampled_clearance_respected"],
        }
        result["artifacts"].update(artifacts)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        # 错误类型用于排查；不把任意底层异常文本（可能含环境信息）写进公开回执。
        details["issue"] = "RECOVERY_CHALLENGE_EVIDENCE_INVALID"
        details["error_type"] = type(error).__name__
    result["gates"].update({"dynamic_obstacle_challenge_" + key: value for key, value in checks.items()})
    result["measurements"]["dynamic_obstacle_challenge"] = details
    result["status"] = "verified" if all(result["gates"].values()) else "failed"
    _write_json(target, result)
    return result


def _load_metric_route(path: Path) -> GraphRoute:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metric route evidence is not an object")
    route_payload = payload.get("route", payload)
    if not isinstance(route_payload, dict):
        raise ValueError("metric route evidence has no route object")
    return GraphRoute.model_validate(route_payload)


# 功能：
#   将明确给出的米制路线转换为几何检查图；默认往返，显式训练接近课程可在终点落地。
# 输入：
#   outbound：原始路线；point_count：可选出程点数；speed_limit_mps：任务速度上限。
#   round_trip_repetitions：往返次数；return_to_launch：是否返回起点。
# 输出：
#   route、graph：绑定同一位置和边的路线及地图图结构。
def _qualification_route(
    outbound: GraphRoute,
    *,
    point_count: int | None,
    speed_limit_mps: float,
    round_trip_repetitions: int = 1,
    return_to_launch: bool = True,
) -> tuple[GraphRoute, MapAsset]:
    if type(return_to_launch) is not bool or (not return_to_launch and round_trip_repetitions != 1):
        raise ValueError("one-way training cannot repeat a closed route")
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
    one_round_trip_node_ids = [*outbound_ids, *reversed(outbound_ids[:-1])] if return_to_launch else list(outbound_ids)
    one_round_trip_positions = [*selected, *reversed(selected[:-1])] if return_to_launch else list(selected)
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
        goal_node=node_ids[-1],
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


# 功能：
#   要求训练有明确证据出口；教师观测已保存状态、模型尺寸相机图片及执行回执，
#   不强制同步生成第二份全分辨率视觉语料，以免采集本身破坏实时性。
# 输入：
#   training：是否为非资格训练采集。
#   dataset：可选的独立多模态视觉语料目录。
#   learner_channel：可选的显式学习器通道。
#   teacher_observations：是否启用当前教师观测与执行回执采集。
# 输出：
#   None：无证据出口的训练被拒绝。
def _validate_training_capture(*, training: bool, dataset: Path | None,
                               learner_channel: Path | None, teacher_observations: bool = False) -> None:
    if type(teacher_observations) is not bool:
        raise ValueError("teacher observation mode must be an explicit boolean")
    if training and dataset is None and learner_channel is None and not teacher_observations:
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
    parser.add_argument("--one-way-training", action="store_true",
                        help="Explicit teacher curriculum ending at the outbound goal; not round-trip qualification")
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
    parser.add_argument("--native-source-clock-domain", default=None)
    parser.add_argument("--live-localization-source", action="store_true",
                        help="Publish run-scoped measurements; never grants pose/control authority")
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
            "Collect non-qualifying payload dynamics using explicit recorded teacher "
            "control or an admitted current model package; never a legacy-policy fallback."
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
    parser.add_argument('--experimental-map-fusion', action='store_true',
                        help='Owned source-matched localization experiment; training only, no qualification')
    parser.add_argument('--bounded-hybrid-control', action='store_true',
                        help='Explicit bounded delay bridge; independent controller attribution retained')
    args = parser.parse_args()
    if args.experimental_map_fusion and (
            not args.training_data_collection or args.simulation_training_channel is None
            or args.native_sensor_runtime is None or args.simulation_camera_profile != 'responsive-control'
            or args.multimodal_dataset_root is None):
        raise ValueError('MAP_FUSION_EXPERIMENT_REQUIRES_NATIVE_RECORDED_TRAINING')
    if not 0.1 <= args.speed_limit_mps <= 1.2:
        raise ValueError("qualification speed must be within [0.1, 1.2] m/s")
    minimum_requested_clearance_m = 0.12 if args.training_data_collection else 0.35
    if args.required_clearance_m < minimum_requested_clearance_m:
        raise ValueError(
            "School Map requested clearance is below the active campaign bound"
        )
    _validate_training_capture(training=args.training_data_collection,
                               dataset=args.multimodal_dataset_root,
                               learner_channel=args.simulation_training_channel,
                               teacher_observations=args.teacher_observation_collection)
    if args.simulation_training_channel is not None:
        if (not args.training_data_collection or args.teacher_observation_collection
                or args.local_navigation_provider not in {None, "simulation-training"}):
            raise ValueError("offline learner channel is restricted to simulation training")
        args.local_navigation_provider = "simulation-training"
    if args.bounded_hybrid_control and (
            not args.training_data_collection or args.simulation_training_channel is None
            or args.teacher_observation_collection):
        raise ValueError("DECISION_HYBRID_REQUIRES_EXPLICIT_TRAINING_CHANNEL")
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
    if args.one_way_training and (not args.training_data_collection
            or not args.teacher_observation_collection or args.round_trip_repetitions != 1):
        raise ValueError("one-way curriculum requires a single explicit teacher training run")
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
        payload_collection_mode(
            provider=args.local_navigation_provider, teacher=args.teacher_observation_collection,
            recording=args.teacher_observation_collection,
            model_authority=(args.local_navigation_provider is not None
                             and not args.teacher_observation_collection),
            model_packages=bool(args.local_policy_package),
            admission=bool(args.local_policy_simulation_admission),
            qualification=bool(args.local_policy_qualification),
            multimodal=args.multimodal_dataset_root is not None,
            fallback=args.local_navigation_fallback_provider is not None)
        if args.teacher_observation_collection and (
                args.one_way_training or args.round_trip_repetitions != 1
                or args.short_outbound_points is not None):
            raise ValueError("payload teacher requires one complete closed measurement route")
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
    route, graph = _qualification_route(
        outbound,
        point_count=args.short_outbound_points,
        speed_limit_mps=args.speed_limit_mps,
        round_trip_repetitions=args.round_trip_repetitions,
        return_to_launch=not args.one_way_training,
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
        raise RuntimeError("qualification route intersects School Map collision geometry")
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
    payload_curriculum = (
        build_payload_teacher_curriculum(route, graph, vehicle, args.vehicle_sdf,
                                         clearance.semantic_sha256)
        if args.teacher_observation_collection and args.development_payload_collection else None
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
    checkpoint_path = action_path = None
    if payload_curriculum is not None:
        for name, artifact in payload_curriculum.items():
            _write_json(plan_root / f"payload-teacher-{name}.json", artifact.model_dump(mode="json"))
        checkpoint_path = plan_root / "payload-teacher-checkpoints.json"
        action_path = plan_root / "payload-teacher-actions.json"
        _write_json(plan_root / "payload-teacher-purpose.json", {
            "simulation_only": True, "model_call_performed": False,
            "flight_qualification_granted": False, "purpose": "physical-payload-measurement"})
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
            "route_completion_mode": "outbound-goal-landing" if args.one_way_training else "return-to-launch",
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
    base_executor = resource_root / 'runtime' / 'px4_offboard_track_executor.py'
    if args.experimental_map_fusion:
        from dronedream_agent_core.map_fusion_experiment import prepare_map_fusion_experiment

        inputs_path, inputs = prepare_map_fusion_experiment(
            args.output_root/'simulation', args.world_sdf, args.semantic)
        os.environ['DRONEDREAM_MAP_FUSION_INPUTS'] = str(inputs_path)
        args.native_source_clock_domain = 'px4-gz-sitl:' + inputs['run_name']
        args.live_localization_source = True
        base_executor = resource_root/'runtime'/'px4_map_fusion_experiment_executor.py'
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
        checkpoint_contract_path=checkpoint_path,
        runtime_action_contract_path=action_path,
        contract_id=(
            payload_curriculum["mission"].contract_id if payload_curriculum is not None else (
                "school-map-training-data-collection" if args.training_data_collection
                else "school-map-live-depth-qualification")
        ),
        executor_extra_args=[
            "--base-executor",
            str(base_executor),
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
        bounded_hybrid_control=args.bounded_hybrid_control,
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
        native_source_clock_domain=args.native_source_clock_domain,
        local_reference_prearm=args.experimental_map_fusion,
        localization_source_channel=(
            args.output_root / "simulation/runtime-state/localization-source.json"
            if args.live_localization_source else None),
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
        evidence = _finalize_recovery_challenge(args.output_root / "simulation", evidence, dynamic_obstacle_challenge, vehicle)
    summary = {
        "schema_version": "dronedream.school-map-depth-qualification-summary.v1",
        "status": evidence["status"],
        "gates": evidence["gates"],
        "measurements": evidence["measurements"],
        "training_data_collection": args.training_data_collection,
        "route_completion_mode": "outbound-goal-landing" if args.one_way_training else "return-to-launch",
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
