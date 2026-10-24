"""Bind an accepted named start to this mission's actual simulated launch pose.

Only runtime launch fields change. Geometry, graph, operating volumes and the
selected asset stay immutable; this is not an asset or flight qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from .assets import load_map_catalog, read_map_semantic_object, resolve_map_entity
from .collision import _clearance, planning_collision_primitives
from .contracts import MapAsset, PreparedMission, VehicleAsset
from .hashing import sha256_json
from .plugin_files import read_plugin_file
from .runtime_bindings import (
    load_map_runtime_bindings,
    resolve_grounded_vehicle_model_root,
    resolve_vehicle_collision_center_offset,
)
from .traversability import TraversableAirspace

_EVENT = "mission.launch-binding"
_LIMIT = 64 * 1024 * 1024


def default_launch_node(graph: MapAsset, semantic_path: Path) -> str:
    """The authored binding chooses the default, never the order of place names."""
    launch = load_map_runtime_bindings(
        read_map_semantic_object(semantic_path)).mission_launch_waypoint
    matches = [node.node_id for node in graph.nodes if node.semantic == "launch"
               and math.dist(tuple(node.position_m.model_dump().values()),
                             (launch.x, launch.y, launch.z)) <= .02]
    if len(matches) != 1:
        raise ValueError("MAP_BOUND_LAUNCH_ENTITY_REQUIRED")
    return matches[0]


def _derive(base: dict, graph: MapAsset, vehicle: VehicleAsset, start_node: str):
    node = next((item for item in graph.nodes if item.node_id == start_node), None)
    if node is None:
        raise ValueError("MISSION_LAUNCH_NODE_UNKNOWN")
    bindings = load_map_runtime_bindings(base)
    position = node.position_m
    if math.dist((position.x, position.y, position.z),
                 tuple(bindings.mission_launch_waypoint.model_dump().values())) <= .02:
        return None
    candidate = deepcopy(base)
    # Spawn Z initially identifies the intended storey, not a floating drone.
    # Actual support geometry below this flight waypoint determines its rest pose.
    bindings.mission_launch_waypoint = position.model_copy(deep=True)
    bindings.vehicle_spawn = position.model_copy(deep=True)
    candidate["runtime_bindings"] = bindings.model_dump(mode="json")
    root = resolve_grounded_vehicle_model_root(candidate, vehicle)
    offset = resolve_vehicle_collision_center_offset(vehicle)
    resting_center_z = root.z + offset.z
    if position.z <= resting_center_z:
        raise ValueError("MISSION_LAUNCH_WAYPOINT_BELOW_RESTING_AIRCRAFT")
    bindings.vehicle_spawn.z = resting_center_z - vehicle.body_height_m / 2
    candidate["runtime_bindings"] = bindings.model_dump(mode="json")
    # One vertical swept cylinder proves the whole climb, including its interior.
    # Tangency with the real support surface is expected at rest; no obstacle or
    # canopy footprint is removed to make a new launch possible.
    swept_half_height = (position.z - resting_center_z + vehicle.body_height_m) / 2
    center = (position.x, position.y, (position.z + resting_center_z) / 2)
    primitives = planning_collision_primitives(candidate)
    airspace = TraversableAirspace(candidate, primitives,
        radius_m=vehicle.body_radius_m, half_height_m=swept_half_height)
    minimum, nearest = airspace.clearance(center)
    for primitive in primitives:
        clearance = _clearance(center, primitive, radius_m=vehicle.body_radius_m,
                               half_height_m=swept_half_height)
        if clearance < minimum:
            minimum, nearest = clearance, str(primitive.get("name", "obstacle"))
    if not math.isfinite(minimum) or minimum < -.001:
        raise ValueError(f"MISSION_LAUNCH_CLIMB_BLOCKED:{nearest}")
    return candidate, {
        "runtime_bindings": candidate["runtime_bindings"],
        "grounded_model_root_enu_m": [root.x, root.y, root.z],
        "takeoff_minimum_clearance_m": minimum,
        "takeoff_nearest_primitive": nearest,
    }


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def prepare_mission_launch(*, semantic_path: Path, graph: MapAsset, vehicle: VehicleAsset,
                           start_node: str, intent_sha256: str, output_dir: Path):
    """Create a mission-local snapshot only when the accepted start differs."""
    result = _derive(read_map_semantic_object(semantic_path), graph, vehicle, start_node)
    if result is None:
        return semantic_path, None
    candidate, details = result
    raw = _bytes(candidate)
    digest = hashlib.sha256(raw).hexdigest()
    relative = f"mission-launch/semantic-{digest}.json"
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    receipt = {
        "schema_version": "dronedream.mission-launch-binding.v1",
        "purpose": "accepted-start-grounded-simulation-binding",
        "base_semantic_sha256": hashlib.sha256(
            read_plugin_file(semantic_path, limit=_LIMIT)).hexdigest(),
        "derived_semantic_sha256": digest, "map_sha256": sha256_json(graph),
        "vehicle_sha256": sha256_json(vehicle), "intent_sha256": intent_sha256,
        "start_node": start_node, "semantic_relative_path": relative, **details,
    }
    (path.parent / "receipt.json").write_bytes(_bytes(receipt))
    return path, receipt


def resolve_mission_execution_semantic(*, prepared: PreparedMission, output_dir: Path,
                                      base_semantic_path: Path, graph: MapAsset,
                                      vehicle: VehicleAsset, source_loader) -> Path:
    """Verify the observed-map then accepted-launch chain before runtime staging."""
    from .mission_map_refinement import resolve_execution_semantic

    records = [item.payload for item in prepared.evidence if item.event_type == _EVENT]
    if len(records) > 1:
        raise ValueError("MISSION_LAUNCH_BINDING_AMBIGUOUS")
    receipt = records[0] if records else None
    intermediate = resolve_execution_semantic(prepared=prepared, output_dir=output_dir,
        base_semantic_path=base_semantic_path, graph=graph, source_loader=source_loader,
        expected_semantic_sha256=receipt.get("base_semantic_sha256") if receipt else None)
    if receipt is None:
        return intermediate
    catalog = load_map_catalog(intermediate, qualified_graph=graph)
    start_node = resolve_map_entity(prepared.intent.start_entity, catalog, graph)
    if start_node != prepared.contract.start_node:
        raise ValueError("MISSION_LAUNCH_INTENT_MISMATCH")
    result = _derive(read_map_semantic_object(intermediate), graph, vehicle, start_node)
    if result is None:
        raise ValueError("MISSION_LAUNCH_BINDING_UNEXPECTED")
    candidate, details = result
    raw = _bytes(candidate)
    digest = hashlib.sha256(raw).hexdigest()
    expected = {
        "schema_version": "dronedream.mission-launch-binding.v1",
        "purpose": "accepted-start-grounded-simulation-binding",
        "base_semantic_sha256": hashlib.sha256(
            read_plugin_file(intermediate, limit=_LIMIT)).hexdigest(),
        "derived_semantic_sha256": digest, "map_sha256": sha256_json(graph),
        "vehicle_sha256": sha256_json(vehicle), "intent_sha256": sha256_json(prepared.intent),
        "start_node": start_node,
        "semantic_relative_path": f"mission-launch/semantic-{digest}.json", **details,
    }
    if receipt != expected or prepared.contract.map_semantic_sha256 != digest:
        raise ValueError("MISSION_LAUNCH_BINDING_MISMATCH")
    path = output_dir / expected["semantic_relative_path"]
    if read_plugin_file(path, limit=_LIMIT) != raw:
        raise ValueError("MISSION_LAUNCH_SNAPSHOT_CHANGED")
    return path
