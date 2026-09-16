"""No-aircraft native rendering isolation probe; never a flight qualification.

Creates only a fixed camera rig and a moving geometric box in a private Gazebo
partition. Fault injection stalls the replica render thread, not physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from xml.sax.saxutils import escape


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def fixture_world(source: Path, output: Path, epoch: str) -> str:
    cameras = "".join(
        f'<sensor name="{name}" type="{kind}"><topic>/replica_probe/{name}</topic>'
        '<update_rate>20</update_rate><camera><horizontal_fov>1.2</horizontal_fov>'
        '<image><width>160</width><height>120</height></image>'
        '<clip><near>0.1</near><far>20</far></clip></camera></sensor>'
        for name, kind in [("rgb", "camera"), ("depth", "depth_camera")]
    )
    return f'''<sdf version="1.9"><world name="replica_probe">
      <gravity>0 0 0</gravity>
      <physics name="default" type="ignored"><max_step_size>0.004</max_step_size>
        <real_time_factor>1</real_time_factor></physics>
      <scene><ambient>0.5 0.5 0.5 1</ambient><background>0.2 0.3 0.4 1</background></scene>
      <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
      <plugin filename="{escape(str(source))}" name="dronedream::RenderSceneSource">
        <epoch>{epoch}</epoch><snapshot_topic>/replica_probe/scene</snapshot_topic>
        <receipt>{escape(str(output / 'source.json'))}</receipt></plugin>
      <model name="camera_rig"><static>true</static><pose>0 0 1 0 0 0</pose>
        <link name="cameras">{cameras}</link></model>
      <model name="moving_box"><pose>3 -1 1 0 0 0</pose><link name="box">
        <inertial><mass>1</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
          </inertia></inertial><collision name="body"><geometry><box><size>0.5 0.5 0.5</size>
          </box></geometry></collision><visual name="body"><geometry><box><size>0.5 0.5 0.5</size>
          </box></geometry><material><ambient>0.8 0.1 0.1 1</ambient>
          <diffuse>0.8 0.1 0.1 1</diffuse></material></visual></link>
        <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
          <initial_linear>0 0.06 0</initial_linear></plugin></model>
    </world></sdf>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=40, choices=range(10, 121))
    parser.add_argument("--render-stall-ms", type=int, default=500, choices=[0, 500])
    parser.add_argument("--debugger", action="store_true", help="GDB on this fixture only")
    parser.add_argument("--development-unfrozen", action="store_true",
                        help="Explicitly probe an unqualified development binary")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    epoch = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    source = args.runtime.resolve() / "libdronedream-render-source.so"
    replica = args.runtime.resolve() / "dronedream-render-replica"
    from dronedream_agent_core.render_replica_runtime import validate_replica_runtime

    runtime_identity = (None if args.development_unfrozen
                        else validate_replica_runtime(args.runtime.resolve()))
    sensors = Path(runtime_identity["sensors_plugin"] if runtime_identity else
        "/usr/lib/x86_64-linux-gnu/gz-sim-8/plugins/libgz-sim8-sensors-system.so.8.14.0")
    identities = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (source, replica, sensors)}
    world = output / "probe.sdf"
    with world.open("x", encoding="utf-8") as stream:
        stream.write(fixture_world(source, output, epoch))
    os.environ["GZ_PARTITION"] = "dronedream-render-probe-" + epoch
    os.environ["GZ_TRANSPORT_SNDHWM"] = "2"
    os.environ["GZ_TRANSPORT_RCVHWM"] = "2"
    from gz.msgs10.image_pb2 import Image
    from gz.transport13 import Node

    from dronedream_agent_core.gazebo_subscriptions import GazeboSubscriptions
    from dronedream_agent_core.sensor_frame_clock import SensorFrameClock

    node = Node()
    subscriptions = GazeboSubscriptions(node)
    lock = threading.Lock()
    rows: dict[str, list] = {"rgb": [], "depth": [], "errors": []}
    last_images = {}
    clocks = {kind: SensorFrameClock(expected_scene_epoch=epoch) for kind in ("rgb", "depth")}

    def received(kind, message):
        now = time.time_ns()
        try:
            mono = time.monotonic()
            with lock:
                clock = clocks[kind].admit(message, received_unix_ns=now,
                                          received_monotonic_seconds=mono)
            row = {"age_ms": clock.age_at_receipt_ns / 1e6,
                   "sha256": hashlib.sha256(message.data).hexdigest(),
                   "sequence": clock.sequence}
            with lock:
                rows[kind].append(row)
                last_images[kind] = message.SerializeToString()
        except (KeyError, ValueError) as error:
            with lock:
                rows["errors"].append(str(error))

    subscriptions.subscribe(Image, "/replica_probe/rgb", lambda message: received("rgb", message))
    subscriptions.subscribe(Image, "/replica_probe/depth",
                            lambda message: received("depth", message))
    child = server = None
    try:
        with (output / "replica.log").open("xb") as replica_log, (
            output / "physics.log"
        ).open("xb") as physics_log:
            command = [
                str(replica), "--epoch", epoch, "--world", "replica_probe",
                "--snapshot-topic", "/replica_probe/scene", "--rgb-topic", "/replica_probe/rgb",
                "--depth-topic", "/replica_probe/depth", "--sensors-plugin", str(sensors),
                "--receipt", str(output / "replica.json"), "--duration-seconds", str(args.duration),
                "--fault-stall-ms", str(args.render_stall_ms), "--fault-every", "30",
            ]
            if args.debugger:
                command = ["gdb", "--batch", "-ex", "run", "-ex", "thread apply all bt",
                           "--args", *command]
            child = subprocess.Popen(command, stdout=replica_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            server = subprocess.Popen(["gz", "sim", "-s", "-r", "-v", "2", str(world)],
                stdout=physics_log, stderr=subprocess.STDOUT, start_new_session=True)
            child.wait(timeout=args.duration + 30)
    except (OSError, subprocess.TimeoutExpired) as error:
        rows["errors"].append(type(error).__name__ + ":" + str(error)[:512])
    finally:
        # These groups contain only our non-aircraft fixture, never an installed
        # product, an existing user simulator, or a physical flight controller.
        for process in (child, server):
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
        subscription_receipt = subscriptions.close()
    for kind, payload in last_images.items():
        with (output / (kind + ".pb")).open("xb") as stream:
            stream.write(payload)
    if "rgb" in last_images:
        from PIL import Image as PILImage
        rgb = Image.FromString(last_images["rgb"])
        PILImage.frombytes("RGB", (rgb.width, rgb.height), rgb.data,
                          "raw", "RGB", rgb.step).save(output / "rgb.png")
    def receipt(name):
        try:
            return json.loads((output / name).read_text())
        except (OSError, ValueError) as error:
            rows["errors"].append(name + ":" + type(error).__name__)
            return {"failed": True, "receipt_unavailable": True}

    source_receipt, replica_receipt = receipt("source.json"), receipt("replica.json")
    if identities != {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in identities}:
        rows["errors"].append("probe-binaries-changed-during-run")
    with lock:
        summary = {
            "fixture": "fixed-camera-rig-and-moving-box-no-aircraft",
            "epoch": epoch, "binaries": identities,
            "world_sha256": hashlib.sha256(world.read_bytes()).hexdigest(),
            "qualification_granted": False,
            "replica_exit": child.returncode if child is not None else None,
            "source": source_receipt, "replica": replica_receipt,
            "subscription_shutdown": subscription_receipt, "debugger": args.debugger,
            "source_bound_runtime": runtime_identity is not None,
            "errors": rows["errors"],
            "images": {kind: {"count": len(rows[kind]),
                "unique_images": len({row["sha256"] for row in rows[kind]}),
                "maximum_arrival_age_ms": max((row["age_ms"] for row in rows[kind]), default=None),
                "expired_at_arrival": sum(row["age_ms"] > 250 for row in rows[kind])}
                for kind in ("rgb", "depth")},
        }
    summary["isolation_probe_passed"] = (
        not summary["errors"] and summary["replica_exit"] == 0 and subscription_receipt["complete"]
        and not args.debugger
        and not summary["source"]["failed"] and not summary["replica"]["failed"]
        and summary["source"]["maximum_source_interval_ms"] < 150
        and all(item["count"] >= 20 and item["unique_images"] >= 2
                for item in summary["images"].values())
        and (not args.render_stall_ms or summary["replica"]["injected_render_stalls"] >= 2)
    )
    write_new(output / "probe.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["isolation_probe_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
