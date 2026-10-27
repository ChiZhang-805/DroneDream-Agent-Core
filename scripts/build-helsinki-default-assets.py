from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.asset_pair_qualification import (
    bind_qualification_receipt,
    build_pair_qualification_receipt,
    prepare_asset_pair_qualification,
)
from dronedream_agent_core.collision import validate_route_clearance
from dronedream_agent_core.contracts import GraphRoute, Px4Track, VehicleAsset

ENVIRONMENT_VERSIONS = {
    "gazebo_sim": "8.15.0",
    "px4_commit": "6ea3539157ca358c70a515878b77077af7d4611d",
    "ros_distribution": "jazzy",
    "runtime_manifest_sha256": ("11d99efcff2b096d252765e7c9fd83ec4ee8ed79fb8c5c90c2bb71f24aa7e36c"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bind verified Helsinki PX4/Gazebo runs into the bundled map catalog."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--default-assets", required=True, type=Path)
    parser.add_argument("--backend-assets", required=True, type=Path)
    parser.add_argument("--qualification-runs", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--staging", required=True, type=Path)
    return parser.parse_args()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vehicle_contract(package: Path) -> VehicleAsset:
    with zipfile.ZipFile(package) as bundle:
        return VehicleAsset.model_validate_json(bundle.read("normalized/vehicle/vehicle.json"))


def build_pair(
    *,
    item: dict[str, Any],
    map_source: Path,
    vehicle_source: Path,
    semantic: Path,
    run_dir: Path,
    staging: Path,
) -> dict[str, Any]:
    evidence_path = run_dir / "mission_evidence.json"
    evidence = read_json(evidence_path)
    if (
        evidence.get("schema_version") != "dronedream.helsinki-map-px4-mission-evidence.v1"
        or evidence.get("asset_id") != item["asset_id"]
        or evidence.get("accepted") is not True
        or evidence.get("executor_exit_code") != 0
        or evidence.get("returned_to_launch") is not True
        or evidence.get("landed") is not True
        or evidence.get("unsafe_collision_count") != 0
    ):
        raise ValueError(f"{item['key']}: real PX4/Gazebo qualification did not pass")

    route_path = run_dir / "mission-route.json"
    track_path = run_dir / "reference_track.json"
    route = GraphRoute.model_validate_json(route_path.read_bytes())
    Px4Track.model_validate_json(track_path.read_bytes())
    vehicle = vehicle_contract(vehicle_source)
    clearance = validate_route_clearance(
        route,
        semantic,
        vehicle_diameter_m=vehicle.body_radius_m * 2,
        vehicle_height_m=vehicle.body_height_m,
        sample_interval_m=0.25,
    )
    if not clearance.accepted or clearance.collision_count != 0:
        raise ValueError(f"{item['key']}: flown route failed package clearance validation")
    clearance_path = run_dir / "clearance.json"
    write_json(clearance_path, clearance.model_dump(mode="json"))

    with tempfile.TemporaryDirectory(prefix=f"{item['key']}-", dir=staging) as temporary_name:
        temporary = Path(temporary_name)
        qualification_root = temporary / "qualification"
        plan = prepare_asset_pair_qualification(
            map_archive=map_source,
            vehicle_archive=vehicle_source,
            work_root=qualification_root,
            recorded_route_path=route_path,
            recorded_track_path=track_path,
            recorded_clearance_path=clearance_path,
            maximum_translation_m=1_000.0,
        )
        artifacts = {
            "world_sha256": sha256(qualification_root / plan.inputs.world_sdf),
            "semantic_sha256": sha256(qualification_root / plan.inputs.semantic),
            "vehicle_sha256": sha256(qualification_root / plan.inputs.vehicle_sdf),
            "controller_params_sha256": sha256(qualification_root / plan.inputs.controller_params),
            "route_sha256": plan.route_sha256,
            "track_sha256": plan.track_sha256,
            "clearance_sha256": plan.clearance_sha256,
            "mission_evidence_sha256": sha256(evidence_path),
            "natural_language_mission_sha256": sha256(run_dir / "natural_language_mission.json"),
            "gazebo_pose_samples_sha256": sha256(run_dir / "gazebo_pose_samples.csv"),
            "px4_log_sha256": sha256(run_dir / "px4.log"),
            "gazebo_log_sha256": sha256(run_dir / "gazebo.log"),
        }
        runtime_evidence = {
            "schema_version": "dronedream.helsinki-map-pair-runtime-evidence.v1",
            "status": "verified",
            "source": "real-gazebo-harmonic-px4-sitl",
            "gates": {gate: True for gate in plan.required_runtime_gates},
            "measurements": {
                "pose_sample_count": evidence["pose_sample_count"],
                "maximum_model_root_z_m": evidence["maximum_model_root_z_m"],
                "route_distance_m": evidence["route_distance_m"],
                "route_waypoint_count": evidence["route_waypoint_count"],
                "unsafe_collision_count": evidence["unsafe_collision_count"],
                "returned_to_launch": evidence["returned_to_launch"],
                "landed": evidence["landed"],
            },
            "artifacts": artifacts,
        }
        qualified_at = datetime.fromtimestamp(evidence_path.stat().st_mtime, tz=UTC)
        receipt = build_pair_qualification_receipt(
            plan=plan,
            work_root=qualification_root,
            runtime_evidence=runtime_evidence,
            environment_versions=ENVIRONMENT_VERSIONS,
            qualified_at=qualified_at,
        )
        map_name = f"{item['key']}.ddpkg"
        vehicle_name = f"my-drone-{item['key']}.ddpkg"
        map_target = staging / map_name
        vehicle_target = staging / vehicle_name
        bind_qualification_receipt(
            source_archive=map_source,
            destination_archive=map_target,
            receipt=receipt,
        )
        bind_qualification_receipt(
            source_archive=vehicle_source,
            destination_archive=vehicle_target,
            receipt=receipt,
        )
        map_source_inspected = inspect_ddpkg(map_source)
        vehicle_source_inspected = inspect_ddpkg(vehicle_source)
        map_qualified = inspect_ddpkg(map_target)
        vehicle_qualified = inspect_ddpkg(vehicle_target)
        receipt_sha256 = hashlib.sha256(json_bytes(receipt.model_dump(mode="json"))).hexdigest()
        pair = {
            "schema_version": "dronedream.bundled-qualified-pair.v1",
            "resource_id": item["resource_id"],
            "qualification_id": receipt.qualification_id,
            "receipt_sha256": receipt_sha256,
            "packages": [
                {
                    "kind": "map",
                    "asset_id": item["asset_id"],
                    "file": map_name,
                    "sha256": sha256(map_target),
                    "source_content_sha256": map_source_inspected.manifest.content_sha256,
                    "content_sha256": map_qualified.manifest.content_sha256,
                },
                {
                    "kind": "vehicle",
                    "asset_id": vehicle_qualified.manifest.asset_id,
                    "file": vehicle_name,
                    "sha256": sha256(vehicle_target),
                    "source_content_sha256": vehicle_source_inspected.manifest.content_sha256,
                    "content_sha256": vehicle_qualified.manifest.content_sha256,
                },
            ],
        }
        # TemporaryDirectory owns its children, so copy completed packages into
        # stable staging paths before it is released.
        stable_map = staging / f"ready-{map_name}"
        stable_vehicle = staging / f"ready-{vehicle_name}"
        shutil.copyfile(map_target, stable_map)
        shutil.copyfile(vehicle_target, stable_vehicle)
        return {
            "pair": pair,
            "map_staged": stable_map,
            "vehicle_staged": stable_vehicle,
        }


def main() -> None:
    args = parse_args()
    config = read_json(args.config.resolve())
    maps = config.get("maps")
    if not isinstance(maps, list) or len(maps) != 5:
        raise ValueError("the first-party Helsinki set must contain exactly five maps")
    default_assets = args.default_assets.resolve()
    staging = args.staging.resolve()
    if staging.exists():
        if not staging.is_dir() or any(staging.iterdir()):
            raise FileExistsError(f"staging directory is not empty: {staging}")
    else:
        staging.mkdir(parents=True)
    old_index = read_json(default_assets / "index.json")
    current_pair = old_index["qualified_pair"]
    vehicle_source = default_assets / "my-drone.ddpkg"
    results: list[dict[str, Any]] = []
    for item in maps:
        key = item["key"]
        results.append(
            build_pair(
                item=item,
                map_source=default_assets / f"{key}.ddpkg",
                vehicle_source=vehicle_source,
                semantic=args.backend_assets.resolve() / key / "gazebo" / "semantic.json",
                run_dir=args.qualification_runs.resolve() / key,
                staging=staging,
            )
        )

    for result in results:
        pair = result["pair"]
        for package in pair["packages"]:
            source = result["map_staged"] if package["kind"] == "map" else result["vehicle_staged"]
            destination = default_assets / package["file"]
            shutil.copyfile(source, destination)

    qualified_pairs = [current_pair, *[result["pair"] for result in results]]
    write_json(
        default_assets / "index.json",
        {
            "schema_version": "dronedream.bundled-assets.v3",
            "default_qualification_id": current_pair["qualification_id"],
            "qualified_pair": current_pair,
            "qualified_pairs": qualified_pairs,
            "superseded_qualified_pairs": old_index.get("superseded_qualified_pairs", []),
        },
    )

    catalog = read_json(args.catalog.resolve())
    pair_by_resource = {result["pair"]["resource_id"]: result["pair"] for result in results}
    for resource in catalog["resources"]:
        pair = pair_by_resource[resource["resource_id"]]
        map_entry = next(package for package in pair["packages"] if package["kind"] == "map")
        resource["source"]["expected_sha256"] = map_entry["sha256"]
        resource["analysis"]["status"] = "qualified"
        resource["readiness"] = {
            "catalog": "ready",
            "planning": "ready",
            "simulation": "ready",
            "flight": "qualified",
            "qualification_id": pair["qualification_id"],
            "required_inputs": [],
        }
    write_json(args.catalog.resolve(), catalog)
    print(
        json.dumps(
            {
                "qualified_maps": [
                    {
                        "resource_id": result["pair"]["resource_id"],
                        "qualification_id": result["pair"]["qualification_id"],
                    }
                    for result in results
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
