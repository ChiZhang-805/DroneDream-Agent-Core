#!/usr/bin/env python3
"""Compare actual Gazebo RGB/depth in the same static world before/after batching.

This diagnostic never starts PX4 or arms a vehicle. Each side runs alone in a
new Gazebo partition; camera configuration and pose are identical. Neither an
image similarity pass nor render timing grants control/flight qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from dronedream_agent_core.simulation_graphics_evidence import graphics_request_evidence
from dronedream_agent_core.static_render_batching import (
    copy_relative_render_resources,
    prepare_static_render_world,
)


def add_camera_rigs(content: bytes, poses: list[list[float]]) -> bytes:
    import math

    if (not 1 <= len(poses) <= 8 or any(len(pose) != 6 or any(
        isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
        for v in pose) for pose in poses)):
        raise ValueError("render verification requires 1–8 finite XYZ/RPY poses")
    root = ET.fromstring(content)
    world = root.find("world")
    if world is None:
        raise ValueError("world required")
    for suffix, name in (("physics", "Physics"), ("sensors", "Sensors")):
        full_name = f"gz::sim::systems::{name}"
        if not any(p.get("name") == full_name for p in world.findall("plugin")):
            plugin = ET.SubElement(world, "plugin", filename=f"gz-sim-{suffix}-system",
                                   name=full_name)
            if name == "Sensors":
                ET.SubElement(plugin, "render_engine").text = "ogre2"
    for i, pose in enumerate(poses):
        model = ET.SubElement(world, "model", name=f"render_probe_{i}")
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = " ".join(format(v, ".17g") for v in pose)
        link = ET.SubElement(model, "link", name="camera_link")
        sensor = ET.SubElement(link, "sensor", name="camera", type="rgbd_camera")
        ET.SubElement(sensor, "topic").text = f"/dronedream/render-probe/{i}"
        ET.SubElement(sensor, "update_rate").text = "10"
        ET.SubElement(sensor, "always_on").text = "true"
        camera = ET.SubElement(sensor, "camera")
        ET.SubElement(camera, "horizontal_fov").text = "1.274"
        image = ET.SubElement(camera, "image")
        for name, value in (("width", "224"), ("height", "128"), ("format", "R8G8B8")):
            ET.SubElement(image, name).text = value
        clip = ET.SubElement(camera, "clip")
        ET.SubElement(clip, "near").text = "0.2"
        ET.SubElement(clip, "far").text = "27"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def capture(args) -> int:
    import numpy as np
    from gz.msgs10.image_pb2 import Image
    from gz.transport13 import Node

    from dronedream_agent_core.gazebo_adapter import _gazebo_image_png
    from dronedream_agent_core.gazebo_subscriptions import (
        GazeboSubscriptions,
        subscription_shutdown_is_complete,
    )
    from dronedream_agent_core.native_process_diagnostics import NativeSourceStallProbe
    from dronedream_agent_core.render_camera_sweep import CameraSweepClient, prepare_camera_sweep

    validate_capture_options(args.camera_count, args.warmup_seconds, args.sample_seconds)
    if not math.isfinite(args.sweep_yaw_rate_dps) or not 0 <= args.sweep_yaw_rate_dps <= 30:
        raise ValueError("render sweep yaw rate must be finite and in [0, 30] degrees/second")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    graphics_request = graphics_request_evidence(os.environ)
    render_world, sweep = args.world, None
    if args.sweep_yaw_rate_dps:
        content = args.world.read_bytes()
        prepared, world_name, cameras = prepare_camera_sweep(content, args.camera_count)
        copy_relative_render_resources(content, args.world.parent, root)
        render_world = root / "sweep-world.sdf"
        render_world.write_bytes(prepared)
        sweep = world_name, cameras
    source_probe = None
    lock, latest, counts, gaps, previous = threading.Lock(), {}, {}, {}, {}
    steady_gaps, steady_counts = {}, {}
    callback_errors = []
    measurement_started = None
    node = Node()
    # Each owner has only the two native subscriptions belonging to one camera.
    subscriptions = [GazeboSubscriptions(node) for _ in range(args.camera_count)]

    def callback(key):
        def on_frame(message):
            now = time.monotonic()
            if source_probe is not None and key == "0-depth":
                source_probe.source_received(now)
            copied = Image()
            copied.CopyFrom(message)
            with lock:
                if key in previous:
                    gaps.setdefault(key, []).append((now - previous[key]) * 1000)
                    # CLI bounds the acquisition to < 6,000 frames per stream.
                    if len(gaps[key]) > 8192:
                        gaps[key].pop()
                        if not callback_errors:
                            callback_errors.append("RENDER_PROBE_TIMING_CAPACITY_EXCEEDED")
                        return
                    if (measurement_started is not None
                            and measurement_started <= previous[key] < now
                            <= measurement_started + args.sample_seconds):
                        steady_gaps.setdefault(key, []).append((now - previous[key]) * 1000)
                if (measurement_started is not None and measurement_started <= now
                        <= measurement_started + args.sample_seconds):
                    steady_counts[key] = steady_counts.get(key, 0) + 1
                previous[key] = now
                counts[key] = counts.get(key, 0) + 1
                latest[key] = copied
        return on_frame

    process = None
    motion_requests = 0
    maximum_motion_request_ms = 0.
    next_motion_at = 0.
    last_motion_result = None
    failure = None
    sweep_client = None
    try:
        if os.environ.get("DRONEDREAM_SIMULATION_CPU_SAMPLES") == "1":
            source_probe = NativeSourceStallProbe()
        if sweep is not None:
            sweep_client = CameraSweepClient(prepared, args.camera_count)
        for i in range(args.camera_count):
            for kind, suffix in (("rgb", "image"), ("depth", "depth_image")):
                topic = f"/dronedream/render-probe/{i}/{suffix}"
                subscriptions[i].subscribe(Image, topic, callback(f"{i}-{kind}"))
        with (root / "gazebo.log").open("xb") as log:
            process = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering",
                                        "-v", "3", str(render_world)],
                                       stdout=log, stderr=subprocess.STDOUT)
            if source_probe is not None:
                source_probe.register("gazebo", process.pid)
            deadline = time.monotonic() + 90 + args.warmup_seconds + args.sample_seconds
            ready_at = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("render probe Gazebo exited")
                with lock:
                    if callback_errors:
                        raise RuntimeError(callback_errors[0])
                    ready = len(counts) == args.camera_count * 2 and min(counts.values()) >= 5
                if ready:
                    if ready_at is None:
                        ready_at = time.monotonic()
                        with lock:
                            measurement_started = ready_at + args.warmup_seconds
                    if time.monotonic() >= measurement_started + args.sample_seconds:
                        break
                    if sweep_client is not None and time.monotonic() >= next_motion_at:
                        now = time.monotonic()
                        yaw_delta = math.radians(args.sweep_yaw_rate_dps) * (now - ready_at)
                        last_motion_result = sweep_client.request(yaw_delta)
                        maximum_motion_request_ms = max(maximum_motion_request_ms,
                            (time.monotonic() - now) * 1000)
                        if (not last_motion_result["transport_accepted"]
                                or last_motion_result["reply"] is not True):
                            raise RuntimeError("render camera sweep request failed:"
                                               + json.dumps(last_motion_result))
                        motion_requests += 1
                        next_motion_at = now + .1
                time.sleep(.05)
            else:
                raise TimeoutError(f"render probe frames missing:{counts}")
            with lock:
                frames, received = dict(latest), dict(counts)
                timings = {key: tuple(values) for key, values in gaps.items()}
                steady_timings = {key: tuple(values) for key, values in steady_gaps.items()}
                measured_counts = dict(steady_counts)
            if (len(steady_timings) != args.camera_count * 2
                    or min(measured_counts.values(), default=0) < 2):
                raise ValueError("render probe has no complete steady measurement")
            for i in range(args.camera_count):
                rgb, depth = frames[f"{i}-rgb"], frames[f"{i}-depth"]
                float32 = Image.DESCRIPTOR.fields_by_name["pixel_format_type"].enum_type
                float32 = float32.values_by_name["R_FLOAT32"].number
                if (depth.pixel_format_type != float32 or depth.width != 224 or depth.height != 128
                        or depth.step != depth.width * 4):
                    raise ValueError("render probe requires packed FLOAT32 metric depth")
                (root / f"rgb-{i}.png").write_bytes(_gazebo_image_png(rgb))
                metric = np.frombuffer(depth.data, dtype="<f4").reshape(128, 224)
                np.save(root / f"depth-{i}.npy", metric, allow_pickle=False)
            receipt = {"received_frames": received, "gaps_ms": gap_summaries(timings),
                "steady_received_frames": measured_counts,
                "steady_gaps_ms": gap_summaries(steady_timings),
                "warmup_seconds": args.warmup_seconds, "sample_seconds": args.sample_seconds,
                "requested_graphics_adapter": os.environ.get("MESA_D3D12_DEFAULT_ADAPTER_NAME"),
                "requested_gallium_driver": os.environ.get("GALLIUM_DRIVER"),
                "graphics_request": graphics_request,
                "sweep_yaw_rate_dps": args.sweep_yaw_rate_dps,
                "camera_motion_requests": motion_requests,
                "maximum_motion_request_ms": maximum_motion_request_ms,
                "camera_count": args.camera_count,
                "world_sha256": hashlib.sha256(args.world.read_bytes()).hexdigest(),
                "render_world_sha256": hashlib.sha256(render_world.read_bytes()).hexdigest(),
                "flight_qualification_granted": False}
    except BaseException as error:
        failure = type(error).__name__ + ":" + str(error)
        raise
    finally:
        shutdown = [owner.close() for owner in subscriptions]
        motion_shutdown = sweep_client.close() if sweep_client else None
        source_diagnostics = source_probe.close() if source_probe is not None else None
        # This process contains only static geometry/cameras, no flight controller.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if failure is not None:
            (root / "failure.json").write_text(json.dumps({
                "error": failure, "received_frames": counts,
                "camera_motion_requests": motion_requests,
                "maximum_motion_request_ms": maximum_motion_request_ms,
                "last_motion_result": last_motion_result,
                "native_subscriptions_shutdown": shutdown,
                "camera_motion_client_shutdown": motion_shutdown,
                "source_stall_diagnostics": source_diagnostics,
                "flight_qualification_granted": False,
            }, indent=2), encoding="utf-8")
    receipt["native_subscriptions_shutdown"] = shutdown
    receipt["camera_motion_client_shutdown"] = motion_shutdown
    receipt["source_stall_diagnostics"] = source_diagnostics
    receipt["capture_complete"] = (all(subscription_shutdown_is_complete(row) for row in shutdown)
                                   and (motion_shutdown is None or motion_shutdown["complete"]))
    (root / "capture.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if not receipt["capture_complete"]:
        raise RuntimeError("render probe subscriptions did not close completely")
    return 0


def validate_capture_options(count: int, warmup: float, sample: float) -> None:
    import math

    if (type(count) is not int or not 1 <= count <= 8
            or any(isinstance(value, bool) or not isinstance(value, int | float)
                   or not math.isfinite(value) for value in (warmup, sample))
            or not 0 <= warmup <= 120 or not 5 <= sample <= 120):
        raise ValueError("render probe requires 1–8 cameras, 0–120 s warmup and 5–120 s sample")


def gap_summaries(timings: dict) -> dict:
    import numpy as np

    return {key: {"interval_count": len(values), "p50": float(np.percentile(values, 50)),
                  "p99": float(np.percentile(values, 99)), "maximum": max(values),
                  "above_150_ms": sum(value > 150 for value in values),
                  "above_250_ms": sum(value > 250 for value in values)}
            for key, values in timings.items() if values}


def compare(original: Path, batched: Path, count: int) -> dict:
    import numpy as np
    from PIL import Image

    result = []
    for i in range(count):
        with Image.open(original / f"rgb-{i}.png") as left:
            a = np.asarray(left.convert("RGB"), dtype=np.int16)
        with Image.open(batched / f"rgb-{i}.png") as right:
            b = np.asarray(right.convert("RGB"), dtype=np.int16)
        da = np.load(original / f"depth-{i}.npy", allow_pickle=False)
        db = np.load(batched / f"depth-{i}.npy", allow_pickle=False)
        if a.shape != b.shape or da.shape != db.shape:
            raise ValueError("render comparison dimensions differ")
        fa, fb = np.isfinite(da), np.isfinite(db)
        common = fa & fb
        rgb_error = np.abs(a - b)
        metrics = {"camera_index": i, "rgb_mean_absolute_error": float(rgb_error.mean()),
            "rgb_fraction_above_8": float(np.mean(np.any(rgb_error > 8, axis=-1))),
            "depth_validity_mismatch_fraction": float(np.mean(fa != fb)),
            "depth_common_fraction": float(np.mean(common)),
            "depth_absolute_p99_m": float(np.percentile(np.abs(da[common] - db[common]), 99))
                if common.any() else None}
        metrics["passed"] = (
            metrics["rgb_mean_absolute_error"] <= 2.
            and metrics["rgb_fraction_above_8"] <= .02
            and metrics["depth_validity_mismatch_fraction"] <= .005
            and metrics["depth_common_fraction"] >= .1
            and metrics["depth_absolute_p99_m"] <= .0001)
        result.append(metrics)
    return {"views": result, "passed": all(row["passed"] for row in result),
            "flight_qualification_granted": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--poses", type=Path, help="JSON list of metric XYZ / radian RPY poses")
    parser.add_argument("--capture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--camera-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--warmup-seconds", type=float, default=0,
                        help="Warmup after every camera has delivered five frames")
    parser.add_argument("--sample-seconds", type=float, default=5,
                        help="Separate steady measurement duration, 5–120 seconds")
    parser.add_argument("--sweep-yaw-rate-dps", type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.capture:
        return capture(args)
    if args.sweep_yaw_rate_dps:
        parser.error("camera sweep is diagnostic-only; use --capture, not image comparison")
    if args.poses is None:
        parser.error("--poses required")
    poses = json.loads(args.poses.read_text(encoding="utf-8"))
    validate_capture_options(len(poses), args.warmup_seconds, args.sample_seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    original = args.output / "probe-world.sdf"
    source = args.world.read_bytes()
    original.write_bytes(add_camera_rigs(source, poses))
    copy_relative_render_resources(source, args.world.parent, args.output)
    batched, receipt = prepare_static_render_world(original, args.output / "batched-world")
    for label, world in (("original", original), ("batched", batched)):
        env = os.environ.copy()
        env["GZ_PARTITION"] = "render-verification-" + uuid.uuid4().hex
        env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
            [str(world.parent), str(args.world.parent), env.get("GZ_SIM_RESOURCE_PATH", "")])
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(world),
            str(args.output / label), "--capture", "--camera-count", str(len(poses)),
            "--warmup-seconds", str(args.warmup_seconds),
            "--sample-seconds", str(args.sample_seconds)],
            env=env, check=True)
    result = {**compare(args.output / "original", args.output / "batched", len(poses)),
        "source_world_sha256": hashlib.sha256(args.world.read_bytes()).hexdigest(),
        "poses": poses, "batching": receipt}
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    brief = {key: value for key, value in result.items() if key != "batching"}
    print(json.dumps(brief), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
