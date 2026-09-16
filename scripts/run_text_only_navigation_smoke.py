"""Exercise DeepSeek-style text-only local navigation against metric evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    OnboardPerceptionFrame,
    RangeRayObservation,
    Vector3,
)
from dronedream_agent_core.local_world_model import MetricVoxelMap
from dronedream_agent_core.model_harness.model_port import StructuredModelPort
from dronedream_agent_core.perception_runtime import (
    EventDrivenIndoorNavigationCoordinator,
    RuntimePerceptionFusion,
)


def run(
    *,
    provider: str,
    output: Path,
    dynamic_obstacle: bool = False,
    image: Path | None = None,
) -> dict[str, object]:
    world = MetricVoxelMap(
        resolution_m=0.5,
        minimum_bound_m=Vector3(x=0.0, y=0.0, z=0.0),
        maximum_bound_m=Vector3(x=6.0, y=2.0, z=1.5),
    )
    world.mark_box(
        minimum_m=Vector3(x=0.25, y=0.25, z=0.25),
        maximum_m=Vector3(x=5.75, y=1.75, z=1.25),
        occupied=False,
        observed_at_monotonic_seconds=1.0,
    )
    current = Vector3(x=0.75, y=0.75, z=0.75)
    goal = Vector3(x=5.25, y=1.25, z=0.75)
    ray = RangeRayObservation(
        origin_m=current,
        endpoint_m=goal,
        hit=False,
        confidence=0.95,
        observed_at_monotonic_seconds=1.0,
    )
    observed_at_unix_ms = int(time.time() * 1_000)
    frame = OnboardPerceptionFrame(
        sensor_id="front-lidar",
        sequence=1,
        observed_at_unix_ms=observed_at_unix_ms,
        localization_position_m=current,
        localization_velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
        localization_covariance_m2=0.01,
        range_rays=[ray.model_copy(deep=True) for _ in range(8)],
        dynamic_obstacles=(
            [
                DynamicObstacleObservation(
                    obstacle_id="person-crossing",
                    position_m=Vector3(x=3.0, y=1.0, z=0.75),
                    velocity_mps=Vector3(x=0.0, y=0.0, z=0.0),
                    radius_m=0.4,
                    height_m=1.7,
                    confidence=0.95,
                    age_seconds=0.0,
                )
            ]
            if dynamic_obstacle
            else []
        ),
    )
    fusion = RuntimePerceptionFusion(
        world=world,
        accepted_sensor_ids={"front-lidar"},
        minimum_rays_per_frame=8,
    )
    health = fusion.ingest(frame, now_unix_ms=observed_at_unix_ms + 10)
    port = StructuredModelPort(provider, max_attempts=1, timeout_seconds=120.0)
    coordinator = EventDrivenIndoorNavigationCoordinator(
        fusion=fusion,
        port=port,
        required_clearance_m=0.0,
        maximum_decision_age_seconds=125.0,
    )
    try:
        coordinator.schedule(
            goal_position_m=goal,
            now_unix_ms=observed_at_unix_ms + 20,
            trigger="initial",
            context_id="indoor-navigation-smoke",
            multimodal=(
                [
                    {
                        "kind": "image-file",
                        "path": str(image),
                        "content_type": "image/png",
                    }
                ]
                if image is not None
                else None
            ),
            strategic_context={
                "task": {
                    "phase": "TRACK",
                    "navigation_goal_id": "pickup",
                    "decision_trigger": "initial",
                },
                "map": {
                    "coordinate_frame": "metric world ENU",
                    "qualified_route_point_count": 2,
                },
                "sensor_contract": {
                    "metric_authority": "calibrated range plus localization",
                    "rgb_role": (
                        "supplementary semantic evidence"
                        if image is not None
                        else "not enabled"
                    ),
                    "unknown_space_policy": "blocked",
                },
                "vehicle": {
                    "body_radius_m": 0.38,
                    "body_height_m": 0.43,
                },
                "payload": {"state": "detached"},
            },
        )
        deadline = time.monotonic() + 125.0
        receipt = None
        while receipt is None and time.monotonic() < deadline:
            receipt = coordinator.poll(now_unix_ms=observed_at_unix_ms + 30)
            if receipt is None:
                time.sleep(0.02)
        if receipt is None:
            raise TimeoutError("navigation smoke timed out")
        model_call = coordinator.pop_model_call_record()
        submitted_snapshot = coordinator.pop_submitted_snapshot()
    finally:
        coordinator.close()
    payload: dict[str, object] = {
        "schema_version": "dronedream.text-only-navigation-smoke.v1",
        "provider": provider,
        "input_mode": "metric-plus-forward-rgb" if image is not None else "metric-text-only",
        "visual_evidence_path": str(image) if image is not None else None,
        "scenario": "dynamic-person-crossing" if dynamic_obstacle else "static-free-corridor",
        "perception_health": health.model_dump(mode="json"),
        "cycle": receipt.model_dump(mode="json"),
        "verified": bool(
            receipt.controller_target_m is not None
            and receipt.deterministic_metric_path_revalidated
            and receipt.dynamic_path_revalidated
        ),
        "model_call": (
            model_call.model_dump(mode="json") if model_call is not None else None
        ),
        "strategic_context_bound": bool(
            submitted_snapshot and submitted_snapshot.get("strategic_context")
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dynamic-obstacle", action="store_true")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    payload = run(
        provider=args.provider,
        output=args.output,
        dynamic_obstacle=args.dynamic_obstacle,
        image=args.image,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
