"""Unarmed full-map isolated-camera probe: fixed sensor rig, no PX4 or airframe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from dronedream_agent_core.gazebo_adapter import _spawn_entity, _wait_for_world
from dronedream_agent_core.render_replica_runtime import file_hash
from dronedream_agent_core.simulation_camera_profile import prepare_camera_profile
from dronedream_agent_core.simulation_render_replica import (
    prepare_render_replica,
    wait_for_replica_images,
)
from dronedream_agent_core.static_render_batching import prepare_static_render_world


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--world-sha256", required=True)
    parser.add_argument("--camera-source-sha256", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--px4-root", type=Path, default=Path("/opt/PX4-Autopilot"))
    args = parser.parse_args()
    if file_hash(args.world) != args.world_sha256:
        raise ValueError("PROBE_WORLD_IDENTITY_MISMATCH")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = prepare_camera_profile(source_models=args.px4_root / "Tools/simulation/gz/models",
        output=output / "camera", expected_source_sha256=args.camera_source_sha256)
    render_world, batching = prepare_static_render_world(args.world, output / "render-world")
    rig = output / "sensor-rig.sdf"
    with rig.open("x", encoding="utf-8") as stream:
        stream.write('<sdf version="1.9"><model name="my_drone">'
            '<include merge="true"><uri>model://OakD-Lite</uri></include>'
            '<static>true</static></model></sdf>')
    resources = (Path(profile["model_resource_root"]), render_world.parent,
                 args.world.parent, args.px4_root / "Tools/simulation/gz/models")
    deployment = prepare_render_replica(runtime_root=args.runtime,
        server_config=args.px4_root / "src/modules/simulation/gz_bridge/server.config",
        world_sdf=render_world, vehicle_sdf=rig,
        camera_sdf=resources[0] / "OakD-Lite/model.sdf", resource_paths=resources,
        world_name="school_map_world", vehicle_name="my_drone", output=output / "isolation")
    env = {**os.environ, **deployment["environment"],
        "GZ_PARTITION": "dronedream-map-probe-" + deployment["epoch"],
        "GZ_SIM_RESOURCE_PATH": ":".join(str(p) for p in resources),
        "GZ_SIM_SYSTEM_PLUGIN_PATH": str(args.px4_root /
            "build/px4_sitl_default/src/modules/simulation/gz_plugins"),
        "GZ_IP": "127.0.0.1", "HEADLESS": "1"}
    # Gazebo transport reads these in this diagnostic process before Node().
    os.environ["GZ_PARTITION"], os.environ["GZ_IP"] = env["GZ_PARTITION"], env["GZ_IP"]
    physics = renderer = None
    failure, readiness = None, {}
    try:
        with (output / "physics.log").open("xb") as physics_log, (
            output / "renderer.log"
        ).open("xb") as render_log:
            physics = subprocess.Popen(["gz", "sim", "-r", "-s", str(render_world)],
                env=env, stdout=physics_log, stderr=subprocess.STDOUT, start_new_session=True)
            _wait_for_world("gz", "school_map_world", env, 90)
            renderer = subprocess.Popen(deployment["command"], env=env, stdout=render_log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            spawn = _spawn_entity("gz", world_name="school_map_world", entity_name="my_drone",
                sdf_path=rig, pose=(-42.25, 15.3, 8.15), env=env)
            write_new(output / "rig-spawn.json", spawn)
            readiness = wait_for_replica_images(deployment)
            write_new(output / "readiness.json", readiness)
            if not readiness["ready"]:
                raise RuntimeError("PROBE_FULL_MAP_CAMERA_NOT_READY")
            until = time.monotonic() + 20
            while time.monotonic() < until:
                if physics.poll() is not None or renderer.poll() is not None:
                    raise RuntimeError("PROBE_PROCESS_EXITED")
                time.sleep(.1)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        failure = type(error).__name__ + ":" + str(error)[:512]
    finally:
        # These owned groups contain only the fixed rig/world, not an aircraft.
        for process in (renderer, physics):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
    def receipt(name):
        path = output / "isolation" / name
        return json.loads(path.read_text()) if path.is_file() else {"receipt_missing": True}
    result = {"fixture": "actual-school-map-fixed-OakD-rig-no-airframe-no-PX4",
              "qualification_granted": False, "failure": failure,
              "readiness": readiness, "source": receipt("source.json"),
              "renderer": receipt("replica.json"), "world_sha256": file_hash(args.world),
              "rig_sha256": hashlib.sha256(rig.read_bytes()).hexdigest(),
              "render_batching": batching,
              "renderer_exit": renderer.returncode if renderer else None,
              "physics_exit": physics.returncode if physics else None}
    result["probe_passed"] = (failure is None and readiness.get("ready") is True
        and result["renderer"].get("failed") is False and result["renderer_exit"] == 0
        and result["source"].get("failed") is False)
    write_new(output / "probe.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "render_batching"}))
    return 0 if result["probe_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
