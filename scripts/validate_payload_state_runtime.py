"""Bounded physics-only parcel integration probe; never starts PX4 or a flight."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from dronedream_agent_core.gazebo_adapter import _detach_payload_before_flight
from dronedream_agent_core.payload_state_query import query_attachment_state


# 功能：在独立世界故意不订阅事件，验证附着、分离、重复分离和再次附着均可真实查询。
# 输入：已构建原生库和全新证据目录；输出：物理回执，退出时清理本次进程组。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GZ_PARTITION="parcel-state-probe-" + uuid.uuid4().hex)
    root = ET.fromstring("""<sdf version="1.9"><world name="parcel_probe">
      <physics name="p" type="ignored"><max_step_size>0.001</max_step_size>
        <real_time_factor>1</real_time_factor></physics>
      <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
      <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
      <model name="carrier"><static>true</static><link name="base_link"/>
        <plugin filename="gz-sim-detachable-joint-system" name="gz::sim::systems::DetachableJoint">
          <parent_link>base_link</parent_link><child_model>parcel</child_model><child_link>payload_link</child_link>
          <attach_topic>/parcel/attach</attach_topic><detach_topic>/parcel/detach</detach_topic><output_topic>/parcel/state</output_topic>
        </plugin></model>
      <model name="parcel"><link name="payload_link"><gravity>false</gravity>
        <inertial><mass>0.2</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
        <collision name="c"><geometry><box><size>0.1 0.1 0.1</size></box></geometry></collision>
      </link></model></world></sdf>""")
    parcel = root.find("./world/model[@name='parcel']")
    ET.SubElement(
        parcel, "plugin", filename=str(args.library.resolve()), name="dronedream::PayloadPlacement"
    )
    world = args.output / "world.sdf"
    ET.ElementTree(root).write(world, encoding="utf-8", xml_declaration=True)
    service = "/world/parcel_probe/model/parcel/attachment_state"
    with (args.output / "gazebo.log").open("w") as log:
        process = subprocess.Popen(
            ["gz", "sim", "-s", "-r", "-v", "4", str(world)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError("probe simulator exited")
                try:
                    initial = query_attachment_state("gz", service=service, env=env)
                    if initial["detached"] is False:
                        break
                except (RuntimeError, subprocess.TimeoutExpired):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("initial actual attachment not observed")
                time.sleep(0.2)
            detached = _detach_payload_before_flight(
                "gz",
                detach_topic="/parcel/detach",
                output_topic="/parcel/state",
                state_service=service,
                env=env,
            )
            repeated = _detach_payload_before_flight(
                "gz",
                detach_topic="/parcel/detach",
                output_topic="/parcel/state",
                state_service=service,
                env=env,
            )
            client_receipt = None
            if args.executor:
                spec = importlib.util.spec_from_file_location(
                    "probe_installed_executor", args.executor
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                preflight_path = args.output / "payload-preflight.json"
                preflight_path.write_text(
                    json.dumps(
                        dict(
                            schema_version="dronedream.payload-preflight-observation",
                            world="parcel_probe",
                            partition=env["GZ_PARTITION"],
                            observed_at_unix_ms=int(time.time() * 1000),
                            observation=repeated,
                        )
                    )
                )
                os.environ.update(
                    GZ_PARTITION=env["GZ_PARTITION"],
                    PX4_GAZEBO_WORLD_NAME="parcel_probe",
                    PX4_GAZEBO_PAYLOAD_PREFLIGHT_PATH=str(preflight_path),
                    PX4_GAZEBO_PAYLOAD_PREFLIGHT_SHA256=hashlib.sha256(
                        preflight_path.read_bytes()
                    ).hexdigest(),
                )

                # 功能：直接验证发布版异步客户端的重复解除和回读；不连接 MAVSDK，不解锁飞控。
                # 输入：当前原生世界和运行绑定回执；输出：真实关节执行回执。
                async def client_probe():
                    client = module.MavsdkOffboardClient()
                    try:
                        await client._prime_payload_observer()
                        command = await client.execute_payload_command(
                            dict(
                                protocol="gazebo-transport",
                                operation="detach",
                                topic="/parcel/detach",
                                output_topic="/parcel/state",
                            )
                        )
                        state = await client.sample_payload_state("/parcel/state", 3)
                        assert command["confirmed"] and state["detached"] is True
                        assert state["observation_kind"] == "request-bound-physics-snapshot"
                        return dict(command=command, state=state)
                    finally:
                        await client.close()

                client_receipt = asyncio.run(client_probe())
            deadline = time.monotonic() + 15
            while True:
                subprocess.run(
                    ["gz", "topic", "-t", "/parcel/attach", "-m", "gz.msgs.Empty", "-p", ""],
                    env=env,
                    check=True,
                    timeout=3,
                    capture_output=True,
                )
                attached = query_attachment_state("gz", service=service, env=env)
                if attached["detached"] is False:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("reattachment not confirmed")
            result = dict(
                initial=initial,
                detached=detached,
                repeated_detach=repeated,
                reattached=attached,
                flight_started=False,
                preflight_subscribed_to_state_events=False,
                executor_client=client_receipt,
            )
            (args.output / "receipt.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print("REAL_PHYSICS_PAYLOAD_STATE_PROBE_PASSED", flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
