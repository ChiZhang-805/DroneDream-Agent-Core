#!/usr/bin/env python3
"""Verify a live Gazebo depth frame and project it into DroneDream metric rays."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from dronedream_agent_core.contracts import (
    CalibratedRangeSensorMount,
    QuaternionWxyz,
    RawMetricRangeScan,
    Vector3,
)
from dronedream_agent_core.depth_projection import (
    DepthProjectionCalibration,
    project_float32_depth_image,
)
from dronedream_agent_core.gazebo_adapter import _gazebo_image_png
from dronedream_agent_core.sensor_bridge import MetricRangeSensorBridge


def _vehicle_model_uri(value: str) -> str:
    prefix = "model://"
    name = value.removeprefix(prefix)
    if not value.startswith(prefix) or not name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in name
    ):
        raise ValueError("vehicle model URI must be model:// followed by a safe model name")
    return value


def _world_sdf(vehicle_model_uri: str = "model://x500_depth") -> str:
    vehicle_model_uri = _vehicle_model_uri(vehicle_model_uri)
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="depth_qualification">
    <gravity>0 0 -9.80665</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <model name="depth_rig">
      <static>true</static>
      <include merge="true">
        <uri>{vehicle_model_uri}</uri>
      </include>
      <pose>0 0 1 0 0 0</pose>
    </model>
    <model name="depth_target">
      <static>true</static>
      <pose>3 0 1.2 0 0 0</pose>
      <link name="target_link">
        <collision name="target_collision">
          <geometry><box><size>0.4 2.0 2.0</size></box></geometry>
        </collision>
        <visual name="target_visual">
          <geometry><box><size>0.4 2.0 2.0</size></box></geometry>
          <material><ambient>0.8 0.2 0.2 1</ambient></material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
"""


def _mount() -> CalibratedRangeSensorMount:
    # OakD-Lite optical center relative to the qualified X500 collision-
    # envelope center (the runtime localization reference).
    return CalibratedRangeSensorMount(
        sensor_id="oakd-lite-depth",
        translation_body_m=Vector3(x=0.13233, y=0.0, z=0.03278),
        orientation_body_from_sensor=QuaternionWxyz(w=1.0, x=0.0, y=0.0, z=0.0),
        minimum_range_m=0.2,
        maximum_range_m=27.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--px4-model-root",
        type=Path,
        default=Path("/opt/PX4-Autopilot/Tools/simulation/gz/models"),
    )
    parser.add_argument(
        "--vehicle-model-root",
        type=Path,
        help="Qualified vehicle package root whose models/ directory shadows PX4 models",
    )
    parser.add_argument("--vehicle-model-uri", default="model://x500_depth")
    parser.add_argument(
        "--depth-topic",
        help="Require this exact Gazebo depth topic instead of auto-discovering one",
    )
    parser.add_argument("--require-rgb", action="store_true")
    parser.add_argument(
        "--rgb-topic",
        help="Require this exact Gazebo RGB topic instead of auto-discovering IMX214",
    )
    args = parser.parse_args()

    try:
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node
    except ModuleNotFoundError as error:
        raise SystemExit(f"Gazebo Python transport is unavailable: {error}") from error

    result: dict[str, object] = {
        "schema_version": "dronedream.gazebo-depth-camera-verification.v1",
        "status": "failed",
        "gates": {},
    }
    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="dronedream-depth-") as temporary_name:
            temporary = Path(temporary_name)
            world = temporary / "world.sdf"
            world.write_text(_world_sdf(args.vehicle_model_uri), encoding="utf-8")
            environment = dict(os.environ)
            existing = environment.get("GZ_SIM_RESOURCE_PATH", "")
            resource_roots: list[Path] = []
            if args.vehicle_model_root is not None:
                if not args.vehicle_model_root.is_dir():
                    raise RuntimeError("QUALIFIED_VEHICLE_MODEL_ROOT_MISSING")
                qualified_models = args.vehicle_model_root / "models"
                if qualified_models.is_dir():
                    resource_roots.append(qualified_models)
                resource_roots.append(args.vehicle_model_root)
            resource_roots.append(args.px4_model_root)
            environment["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
                [*(str(path) for path in resource_roots), *([existing] if existing else [])]
            )
            environment.setdefault("QT_QPA_PLATFORM", "offscreen")
            process = subprocess.Popen(
                ["gz", "sim", "-s", "-r", "--headless-rendering", str(world)],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            node = Node()
            deadline = time.monotonic() + args.timeout_seconds
            topic = ""
            while time.monotonic() < deadline and process.poll() is None:
                topics = [str(value) for value in node.topic_list()]
                matches = (
                    [args.depth_topic]
                    if args.depth_topic is not None and args.depth_topic in topics
                    else [
                        value
                        for value in topics
                        if args.depth_topic is None
                        and "depth_camera" in value
                        and not value.endswith("camera_info")
                    ]
                )
                if matches:
                    topic = sorted(matches, key=len)[0]
                    break
                time.sleep(0.1)
            if not topic:
                raise RuntimeError("LIVE_DEPTH_TOPIC_NOT_ADVERTISED")

            received = threading.Event()
            frame: dict[str, object] = {}

            def on_image(message: Image) -> None:
                if received.is_set():
                    return
                frame.update(
                    width=int(message.width),
                    height=int(message.height),
                    step=int(message.step),
                    data=bytes(message.data),
                    pixel_format_type=int(message.pixel_format_type),
                )
                received.set()

            node.subscribe(Image, topic, on_image)
            remaining = max(0.1, deadline - time.monotonic())
            if not received.wait(remaining):
                raise RuntimeError("LIVE_DEPTH_FRAME_NOT_RECEIVED")

            rgb_topic = ""
            rgb_frame_path: Path | None = None
            if args.require_rgb:
                topics = [str(value) for value in node.topic_list()]
                rgb_matches = (
                    [args.rgb_topic]
                    if args.rgb_topic is not None and args.rgb_topic in topics
                    else [
                        value
                        for value in topics
                        if args.rgb_topic is None
                        and "IMX214" in value
                        and value.endswith("/image")
                    ]
                )
                if not rgb_matches:
                    raise RuntimeError("LIVE_RGB_TOPIC_NOT_ADVERTISED")
                rgb_topic = sorted(rgb_matches, key=len)[0]
                rgb_received = threading.Event()
                rgb_frame: dict[str, bytes] = {}

                def on_rgb(message: Image) -> None:
                    if rgb_received.is_set():
                        return
                    rgb_frame["png"] = _gazebo_image_png(message)
                    rgb_received.set()

                node.subscribe(Image, rgb_topic, on_rgb)
                remaining = max(0.1, deadline - time.monotonic())
                if not rgb_received.wait(remaining):
                    raise RuntimeError("LIVE_RGB_FRAME_NOT_RECEIVED")
                rgb_frame_path = args.output.with_name(f"{args.output.stem}-forward-rgb.png")
                rgb_frame_path.parent.mkdir(parents=True, exist_ok=True)
                rgb_frame_path.write_bytes(rgb_frame["png"])

            width = int(frame["width"])
            height = int(frame["height"])
            samples = project_float32_depth_image(
                data=bytes(frame["data"]),
                row_step_bytes=int(frame["step"]),
                calibration=DepthProjectionCalibration(
                    width=width,
                    height=height,
                    horizontal_fov_rad=1.274,
                    minimum_depth_m=0.2,
                    maximum_depth_m=19.1,
                    sample_stride_pixels=16,
                ),
            )
            scan = RawMetricRangeScan(
                sensor_id="oakd-lite-depth",
                sequence=1,
                observed_at_unix_ms=int(time.time() * 1_000),
                observed_at_monotonic_seconds=time.monotonic(),
                body_position_world_enu_m=Vector3(x=0.0, y=0.0, z=1.228),
                body_orientation_world_from_body=QuaternionWxyz(
                    w=1.0, x=0.0, y=0.0, z=0.0
                ),
                body_velocity_world_enu_mps=Vector3(x=0.0, y=0.0, z=0.0),
                localization_covariance_m2=0.0,
                samples=samples,
            )
            perception = MetricRangeSensorBridge(_mount()).assemble(scan)
            hit_ranges = [sample.range_m for sample in samples if sample.hit]
            finite_ranges = [sample.range_m for sample in samples if math.isfinite(sample.range_m)]
            result.update(
                status="verified",
                topic=topic,
                rgb_topic=rgb_topic or None,
                rgb_frame_path=str(rgb_frame_path) if rgb_frame_path is not None else None,
                vehicle_model_uri=args.vehicle_model_uri,
                vehicle_model_root=(
                    str(args.vehicle_model_root.resolve())
                    if args.vehicle_model_root is not None
                    else None
                ),
                image={
                    "width": width,
                    "height": height,
                    "step": int(frame["step"]),
                    "pixel_format_type": int(frame["pixel_format_type"]),
                },
                measurements={
                    "projected_ray_count": len(samples),
                    "hit_ray_count": len(hit_ranges),
                    "minimum_hit_range_m": min(hit_ranges) if hit_ranges else None,
                    "maximum_projected_range_m": max(finite_ranges),
                    "world_ray_count": len(perception.range_rays),
                },
                gates={
                    "live_depth_topic_advertised": True,
                    "live_depth_frame_received": True,
                    "metric_ray_projection_valid": len(samples) >= 8,
                    "world_enu_sensor_bridge_valid": len(perception.range_rays) == len(samples),
                    "known_obstacle_detected": bool(hit_ranges) and min(hit_ranges) < 4.0,
                    **(
                        {
                            "live_rgb_topic_advertised": bool(rgb_topic),
                            "live_rgb_frame_received": bool(
                                rgb_frame_path is not None and rgb_frame_path.is_file()
                            ),
                        }
                        if args.require_rgb
                        else {}
                    ),
                },
            )
    except Exception as error:  # evidence boundary: serialize exact failure
        result["error"] = f"{type(error).__name__}:{error}"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if process is not None and process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            if stderr.strip():
                result["gazebo_stderr_tail"] = stderr[-4_000:]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    gates = result.get("gates")
    verified = (
        result.get("status") == "verified"
        and isinstance(gates, dict)
        and all(gates.values())
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
