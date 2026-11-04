from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dronedream_agent_app.vehicle_resource_catalog import vehicle_resource_catalog
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
    "runtime_manifest_sha256": "11d99efcff2b096d252765e7c9fd83ec4ee8ed79fb8c5c90c2bb71f24aa7e36c",
}
MAPS = {
    "kumpula-campus": "dronedream-school-map",
    "meilahti-hospital-campus": "dronedream-meilahti-hospital-map",
    "kalasatama-waterfront": "dronedream-kalasatama-waterfront-map",
    "vuosaari-harbour": "dronedream-vuosaari-harbour-map",
    "suomenlinna-sea-fortress": "dronedream-suomenlinna-map",
    "uutela-coastal-forest": "dronedream-uutela-coastal-forest-map",
}
VEHICLES = {
    "x500-depth": {
        "resource_id": "px4-x500-depth",
        "asset_id": "dronedream.my-drone.v1",
        "source_file": "my-drone.ddpkg",
    },
    "x500-lidar-front": {
        "resource_id": "px4-x500-lidar-front",
        "asset_id": "x500-lidar-front",
        "profile": "x500_lidar_front",
        "name": "X500 Forward-Lidar Quadcopter",
        "dry_mass_kg": 2.085307692,
        "body_radius_m": 0.38,
        "body_height_m": 0.288,
        "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.157},
    },
    "x500-gimbal": {
        "resource_id": "px4-x500-gimbal",
        "asset_id": "x500-gimbal",
        "profile": "x500_gimbal",
        "name": "X500 Gimbal Quadcopter",
        "dry_mass_kg": 2.464307692,
        "body_radius_m": 0.38,
        "body_height_m": 0.288,
        "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.157},
    },
    "standard-vtol": {
        "resource_id": "px4-standard-vtol",
        "asset_id": "standard-vtol",
        "profile": "standard_vtol",
        "name": "Standard VTOL",
        "dry_mass_kg": 5.08600003,
        "body_radius_m": 1.2,
        "body_height_m": 0.298,
        "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.3},
        "max_pickup_payload_kg": 0.04,
        "extra_sensors": ["camera", "depth_camera"],
    },
    "tiltrotor": {
        "resource_id": "px4-tiltrotor",
        "asset_id": "tiltrotor",
        "profile": "tiltrotor",
        "px4_sitl_model": "standard_vtol",
        "name": "Tiltrotor (Multicopter Map Mode)",
        "dry_mass_kg": 5.13500004,
        "body_radius_m": 1.2,
        "body_height_m": 0.178,
        "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.230},
        "extra_sensors": ["camera", "depth_camera"],
    },
    "quadtailsitter": {
        "resource_id": "px4-quadtailsitter",
        "asset_id": "quadtailsitter",
        "profile": "quadtailsitter",
        "name": "Quad Tailsitter",
        "dry_mass_kg": 1.635,
        "body_radius_m": 0.42,
        "body_height_m": 0.287,
        "collision_center_offset_model_m": {"x": 0.0, "y": 0.0, "z": 0.343},
        "extra_sensors": ["camera", "depth_camera"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote accepted real PX4/Gazebo aircraft-map runs into bundled pairs."
    )
    parser.add_argument("--default-assets", required=True, type=Path)
    parser.add_argument(
        "--map-source-root",
        required=True,
        type=Path,
        help="Current product map source root containing one directory per map key.",
    )
    parser.add_argument(
        "--qualification-runs",
        required=True,
        type=Path,
        action="append",
        help="Qualification evidence root. Repeat to overlay newer exact runs.",
    )
    parser.add_argument("--staging", required=True, type=Path)
    return parser.parse_args()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
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


def source_package_writer() -> Any:
    path = Path(__file__).with_name("build-default-assets.py")
    spec = importlib.util.spec_from_file_location("dronedream_build_default_assets", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("default asset builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._write_current_source_package


def vehicle_contract(package: Path) -> VehicleAsset:
    with zipfile.ZipFile(package) as bundle:
        return VehicleAsset.model_validate_json(bundle.read("normalized/vehicle/vehicle.json"))


def accepted_runs(root: Path) -> dict[tuple[str, str], Path]:
    runs: dict[tuple[str, str], Path] = {}
    vehicle_aliases = {
        "x500_depth / My Drone": "x500-depth",
        "x500_depth": "x500-depth",
    }
    for evidence_path in sorted(root.rglob("mission_evidence.json")):
        evidence = read_json(evidence_path)
        vehicle_key = vehicle_aliases.get(str(evidence.get("vehicle")), evidence.get("vehicle"))
        map_key = evidence.get("scene_id")
        if (
            evidence.get("schema_version") == "dronedream.helsinki-map-px4-mission-evidence.v1"
            and evidence.get("accepted") is True
            and evidence.get("executor_exit_code") == 0
            and evidence.get("returned_to_launch") is True
            and evidence.get("landed") is True
            and evidence.get("unsafe_collision_count") == 0
            and vehicle_key in VEHICLES
            and map_key in MAPS
        ):
            runs[(str(vehicle_key), str(map_key))] = evidence_path.parent
    return runs


def build_vehicle_source(
    *, vehicle_key: str, vehicle_runs: list[Path], staging: Path, writer: Any
) -> Path:
    profile = VEHICLES[vehicle_key]
    source_run = vehicle_runs[0]
    source_root = staging / f"source-{vehicle_key}"
    gazebo = source_root / "gazebo"
    gazebo.mkdir(parents=True)
    for source in (source_run / "my-drone").iterdir():
        if source.is_file():
            shutil.copyfile(source, gazebo / source.name)
    shutil.copyfile(source_run / "controller_params.json", source_root / "controller_params.json")
    maximum_distance = max(
        float(read_json(run / "mission_evidence.json")["route_distance_m"]) for run in vehicle_runs
    )
    catalog = next(
        item
        for item in vehicle_resource_catalog()["resources"]
        if item["resource_id"] == profile["resource_id"]
    )
    sensors = list(
        dict.fromkeys(
            [
                *catalog["analysis"]["resolved_sensor_types"],
                *profile.get("extra_sensors", []),
                "odometry",
            ]
        )
    )
    summary = read_json(gazebo / "summary.json") if (gazebo / "summary.json").is_file() else {}
    summary.update(
        {
            "schema_version": "dronedream.vehicle-gazebo-artifact.v1",
            "model_name": "my_drone",
            "source_model": f"model://{profile['profile']}",
            "px4_sitl_model": profile.get("px4_sitl_model", profile["profile"]),
            "control_interface": "mavsdk",
            "dry_mass_kg": profile["dry_mass_kg"],
            "collision_envelope": {
                "body_radius_m": profile["body_radius_m"],
                "body_height_m": profile["body_height_m"],
                "collision_center_offset_model_m": profile["collision_center_offset_model_m"],
            },
        }
    )
    write_json(
        gazebo / "summary.json",
        summary,
    )
    controller = read_json(source_root / "controller_params.json")
    write_json(
        source_root / "vehicle.json",
        {
            "schema_version": "dronedream.vehicle.v1",
            "asset_id": profile["asset_id"],
            "name": profile["name"],
            "coordinate_frame": "base_link_frd",
            "dry_mass_kg": profile["dry_mass_kg"],
            "max_takeoff_mass_kg": profile["dry_mass_kg"]
            + float(profile.get("max_pickup_payload_kg", 0.0)),
            "body_radius_m": profile["body_radius_m"],
            "body_height_m": profile["body_height_m"],
            "collision_center_offset_model_m": profile["collision_center_offset_model_m"],
            "max_speed_mps": float(controller["vel_limit"]),
            "qualified_cruise_max_mps": float(controller["vel_limit"]),
            "max_acceleration_mps2": float(controller["accel_limit"]),
            "qualified_range_m": maximum_distance,
            "reserve_battery_percent": 30.0,
            "max_pickup_payload_kg": float(profile.get("max_pickup_payload_kg", 0.0)),
            "sensors": sensors,
        },
    )
    evidence_path = source_run / "mission_evidence.json"
    return writer(
        source_root=source_root,
        destination=staging / f"source-{vehicle_key}.ddpkg",
        kind="vehicle",
        asset_id=profile["asset_id"],
        name=profile["name"],
        created_at=datetime.fromtimestamp(evidence_path.stat().st_mtime, tz=UTC),
    )


def build_pair(
    *,
    vehicle_key: str,
    map_key: str,
    map_source: Path,
    vehicle_source: Path,
    run_dir: Path,
    staging: Path,
) -> dict[str, Any]:
    evidence_path = run_dir / "mission_evidence.json"
    evidence = read_json(evidence_path)
    inputs_path = run_dir / "qualification-inputs.json"
    if not inputs_path.is_file():
        inputs_path = run_dir / "natural_language_mission.json"
    inputs = read_json(inputs_path)
    if inputs.get("natural_language_acceptance") is True:
        raise ValueError(f"{vehicle_key}/{map_key}: physical evidence was misclassified")
    route_path = run_dir / "mission-route.json"
    track_path = run_dir / "reference_track.json"
    clearance_path = run_dir / "clearance.json"
    route = GraphRoute.model_validate_json(route_path.read_bytes())
    Px4Track.model_validate_json(track_path.read_bytes())
    vehicle = vehicle_contract(vehicle_source)
    recorded_clearance = read_json(clearance_path)
    qualification_root = staging / "qualification" / vehicle_key / map_key
    plan = prepare_asset_pair_qualification(
        map_archive=map_source,
        vehicle_archive=vehicle_source,
        work_root=qualification_root,
        recorded_route_path=route_path,
        recorded_track_path=track_path,
        recorded_clearance_path=clearance_path,
        maximum_translation_m=1_000.0,
    )
    clearance = validate_route_clearance(
        route,
        qualification_root / plan.inputs.semantic,
        vehicle_diameter_m=vehicle.body_radius_m * 2,
        vehicle_height_m=vehicle.body_height_m,
        sample_interval_m=0.05,
    )
    if (
        not clearance.accepted
        or clearance.collision_count != 0
        or recorded_clearance.get("accepted") is not True
        or recorded_clearance.get("collision_count") != 0
    ):
        raise ValueError(f"{vehicle_key}/{map_key}: flown route failed clearance validation")
    artifacts = {
        "world_sha256": sha256(qualification_root / plan.inputs.world_sdf),
        "semantic_sha256": sha256(qualification_root / plan.inputs.semantic),
        "vehicle_sha256": sha256(qualification_root / plan.inputs.vehicle_sdf),
        "controller_params_sha256": sha256(qualification_root / plan.inputs.controller_params),
        "route_sha256": plan.route_sha256,
        "track_sha256": plan.track_sha256,
        "clearance_sha256": plan.clearance_sha256,
        "mission_evidence_sha256": sha256(evidence_path),
        "qualification_inputs_sha256": sha256(inputs_path),
        "gazebo_pose_samples_sha256": sha256(run_dir / "gazebo_pose_samples.csv"),
        "px4_log_sha256": sha256(run_dir / "px4.log"),
        "gazebo_log_sha256": sha256(run_dir / "gazebo.log"),
    }
    runtime_evidence = {
        "schema_version": "dronedream.aircraft-map-physical-evidence.v1",
        "status": "verified",
        "source": "real-gazebo-harmonic-px4-sitl",
        "natural_language_acceptance": False,
        "gates": {gate: True for gate in plan.required_runtime_gates},
        "measurements": {
            key: evidence[key]
            for key in (
                "pose_sample_count",
                "maximum_model_root_z_m",
                "route_distance_m",
                "route_waypoint_count",
                "unsafe_collision_count",
                "returned_to_launch",
                "landed",
            )
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
    default_pair = vehicle_key == "x500-depth" and map_key == "kumpula-campus"
    map_name = "school-map.ddpkg" if default_pair else f"{map_key}-{vehicle_key}.ddpkg"
    vehicle_name = "my-drone.ddpkg" if default_pair else f"{vehicle_key}-{map_key}.ddpkg"
    map_target = staging / "packages" / map_name
    vehicle_target = staging / "packages" / vehicle_name
    map_target.parent.mkdir(parents=True, exist_ok=True)
    bind_qualification_receipt(
        source_archive=map_source, destination_archive=map_target, receipt=receipt
    )
    bind_qualification_receipt(
        source_archive=vehicle_source, destination_archive=vehicle_target, receipt=receipt
    )
    map_source_inspected = inspect_ddpkg(map_source)
    vehicle_source_inspected = inspect_ddpkg(vehicle_source)
    map_qualified = inspect_ddpkg(map_target)
    vehicle_qualified = inspect_ddpkg(vehicle_target)
    return {
        "schema_version": "dronedream.bundled-qualified-pair.v1",
        "resource_id": MAPS[map_key],
        "vehicle_resource_id": VEHICLES[vehicle_key]["resource_id"],
        "qualification_id": receipt.qualification_id,
        "receipt_sha256": hashlib.sha256(json_bytes(receipt.model_dump(mode="json"))).hexdigest(),
        "packages": [
            {
                "kind": "map",
                "asset_id": map_qualified.manifest.asset_id,
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


def retired_pair_record(pair: dict[str, Any]) -> dict[str, str]:
    packages = {item["kind"]: item for item in pair["packages"]}
    return {
        "qualification_id": pair["qualification_id"],
        "map_content_sha256": packages["map"]["content_sha256"],
        "map_source_content_sha256": packages["map"]["source_content_sha256"],
        "vehicle_content_sha256": packages["vehicle"]["content_sha256"],
        "vehicle_source_content_sha256": packages["vehicle"]["source_content_sha256"],
    }


def main() -> None:
    args = parse_args()
    default_assets = args.default_assets.resolve()
    map_source_root = args.map_source_root.resolve()
    qualification_runs = [root.resolve() for root in args.qualification_runs]
    staging = args.staging.resolve()
    if staging.exists() and (not staging.is_dir() or any(staging.iterdir())):
        raise FileExistsError(f"staging directory is not empty: {staging}")
    staging.mkdir(parents=True, exist_ok=True)
    old_index = read_json(default_assets / "index.json")
    runs: dict[tuple[str, str], Path] = {}
    for root in qualification_runs:
        runs.update(accepted_runs(root))
    expected = {
        *(
            (vehicle, map_key)
            for vehicle in ("x500-lidar-front", "x500-gimbal")
            for map_key in MAPS
        ),
        *(("x500-depth", map_key) for map_key in MAPS),
        *(("standard-vtol", map_key) for map_key in MAPS),
        *(
            ("quadtailsitter", map_key)
            for map_key in (
                "kumpula-campus",
                "kalasatama-waterfront",
                "vuosaari-harbour",
            )
        ),
        *(
            ("tiltrotor", map_key)
            for map_key in (
                "kumpula-campus",
                "vuosaari-harbour",
            )
        ),
    }
    missing = sorted(expected - runs.keys())
    if missing:
        raise ValueError(f"accepted qualification runs are missing: {missing}")
    writer = source_package_writer()
    map_sources: dict[str, Path] = {}
    for map_key in MAPS:
        source_root = map_source_root / map_key
        semantic_path = source_root / "gazebo" / "semantic.json"
        semantic = read_json(semantic_path)
        display_name = semantic.get("display_name")
        if not isinstance(display_name, dict) or not isinstance(display_name.get("en-US"), str):
            raise ValueError(f"map has no English display name: {map_key}")
        map_sources[map_key] = writer(
            source_root=source_root,
            destination=staging / f"source-map-{map_key}.ddpkg",
            kind="map",
            asset_id=str(semantic["asset_id"]),
            name=display_name["en-US"],
            created_at=datetime.fromtimestamp(semantic_path.stat().st_mtime, tz=UTC),
        )
    vehicle_sources: dict[str, Path] = {}
    for vehicle_key in sorted({pair[0] for pair in expected}):
        profile = VEHICLES[vehicle_key]
        if "source_file" in profile:
            vehicle_sources[vehicle_key] = default_assets / profile["source_file"]
        else:
            vehicle_sources[vehicle_key] = build_vehicle_source(
                vehicle_key=vehicle_key,
                vehicle_runs=[runs[pair] for pair in sorted(expected) if pair[0] == vehicle_key],
                staging=staging,
                writer=writer,
            )
    built = []
    for vehicle_key, map_key in sorted(expected):
        try:
            built.append(
                build_pair(
                    vehicle_key=vehicle_key,
                    map_key=map_key,
                    map_source=map_sources[map_key],
                    vehicle_source=vehicle_sources[vehicle_key],
                    run_dir=runs[(vehicle_key, map_key)],
                    staging=staging,
                )
            )
        except Exception as error:
            raise RuntimeError(f"pair promotion failed: {vehicle_key}/{map_key}") from error
    pairs = built
    default_pair = next(
        pair
        for pair in pairs
        if pair["resource_id"] == "dronedream-school-map"
        and pair.get("vehicle_resource_id", "px4-x500-depth") == "px4-x500-depth"
    )
    superseded_by_id = {
        record["qualification_id"]: record
        for record in old_index.get("superseded_qualified_pairs", [])
    }
    current_ids = {pair["qualification_id"] for pair in pairs}
    for old_pair in old_index["qualified_pairs"]:
        if old_pair["qualification_id"] not in current_ids:
            record = retired_pair_record(old_pair)
            superseded_by_id[record["qualification_id"]] = record
    superseded = list(superseded_by_id.values())
    write_json(
        staging / "index.json",
        {
            "schema_version": "dronedream.bundled-assets.v3",
            "default_qualification_id": default_pair["qualification_id"],
            "qualified_pair": default_pair,
            "qualified_pairs": pairs,
            "superseded_qualified_pairs": superseded,
        },
    )
    for package in (staging / "packages").iterdir():
        shutil.copyfile(package, default_assets / package.name)
    shutil.copyfile(staging / "index.json", default_assets / "index.json")
    print(
        json.dumps(
            {
                "qualified_pair_count": len(pairs),
                "new_pair_count": len(built),
                "default_qualification_id": default_pair["qualification_id"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
