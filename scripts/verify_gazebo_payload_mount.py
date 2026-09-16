"""Exercise the production Gazebo payload mount alignment against a live world."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


def _load_executor(path: Path):
    spec = importlib.util.spec_from_file_location("dronedream_payload_mount_acceptance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load runtime executor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _run(args: argparse.Namespace) -> dict[str, object]:
    module = _load_executor(args.executor)
    client = object.__new__(module.MavsdkOffboardClient)
    os.environ["PX4_GAZEBO_WORLD_NAME"] = args.world
    parameters = {
        "protocol": "gazebo-transport",
        "operation": "attach",
        "topic": f"/model/{args.vehicle}/{args.payload}/attach",
        "output_topic": f"/model/{args.vehicle}/{args.payload}/state",
        "vehicle_model_name": args.vehicle,
        "payload_model_name": args.payload,
        "payload_mount_offset_model_m": [0.0, 0.0, args.offset_z_m],
        "payload_mount_max_alignment_error_m": args.maximum_error_m,
        "payload_mount_binding_sha256": "0" * 64,
    }
    command = await client.execute_payload_command(parameters)
    sampled = await client._sample_named_gazebo_poses(
        world_name=args.world,
        model_names=(args.vehicle, args.payload),
    )
    vehicle_pose = sampled[args.vehicle]
    payload_pose = sampled[args.payload]
    offset_world = module._rotate_gazebo_model_vector(
        vehicle_pose,
        (0.0, 0.0, args.offset_z_m),
    )
    expected = (
        vehicle_pose.x + offset_world[0],
        vehicle_pose.y + offset_world[1],
        vehicle_pose.z + offset_world[2],
    )
    observed = (payload_pose.x, payload_pose.y, payload_pose.z)
    error_m = math.dist(expected, observed)
    if error_m > args.maximum_error_m:
        raise RuntimeError(
            "payload mount alignment exceeded tolerance: "
            f"observed={error_m:.6f}m limit={args.maximum_error_m:.6f}m"
        )
    return {
        "schema_version": "dronedream.gazebo-payload-mount-acceptance.v1",
        "accepted": True,
        "world": args.world,
        "vehicle_model": args.vehicle,
        "payload_model": args.payload,
        "expected_payload_position_world_enu_m": list(expected),
        "observed_payload_position_world_enu_m": list(observed),
        "alignment_error_m": error_m,
        "maximum_alignment_error_m": args.maximum_error_m,
        "command": command,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executor", type=Path)
    parser.add_argument("--world", default="school_map_world")
    parser.add_argument("--vehicle", default="my_drone")
    parser.add_argument("--payload", default="takeout_payload")
    parser.add_argument("--offset-z-m", type=float, default=0.12)
    parser.add_argument("--maximum-error-m", type=float, default=0.02)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
