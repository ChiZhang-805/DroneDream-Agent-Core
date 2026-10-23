"""Mission-local visual map refinement with immutable, execution-bound snapshots.

The original asset remains qualified only for its original physical world. A
derived ground layer is an observed, route-checked planning input, never a new
flight qualification. Both planner and executor verify this exact derived input.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .assets import read_map_semantic_object
from .collision import planning_collision_primitives, validate_route_clearance
from .contracts import MapAsset, ModelCallRecord, PreparedMission, RouteQuery, VehicleAsset
from .hashing import sha256_json
from .known_map_planner import KnownMapMetricPlanner, MetricPlannerPolicy
from .map_gap_refinement import (
    MapImageSource,
    MapRefinementResult,
    MapViewInterpretation,
    ObservationRequest,
    _apply_patch,
    interpretation_patch,
    prepare_map_view,
    refine_map_gaps,
)
from .model_harness.model_port import StructuredModelPort
from .plugin_files import check_plain_plugin_path, read_plugin_file
from .traversability import TraversableAirspace

_LIMIT = 64 * 1024 * 1024
_EVENT = "mission.map-refinement"
_QUALIFICATION = "observed-ground-and-route-checked-not-flight-qualified"


@dataclass(frozen=True)
class GroundMapPreparation:
    semantic_path: Path
    receipt: dict[str, Any] | None
    model_calls: list[dict[str, Any]]
    physical_calls: int


class MapObservationPending(RuntimeError):
    """The mission stays in preparation with explicit, resumable observation work."""

    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__("MAP_OBSERVATION_PENDING")
        self.details = details


def _bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _snapshot(directory: Path, semantic: dict[str, Any]) -> Path:
    raw = _bytes(semantic)
    path = directory / f"semantic-{hashlib.sha256(raw).hexdigest()}.json"
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(raw)
    elif read_plugin_file(path, limit=_LIMIT) != raw:
        raise ValueError("MAP_REFINEMENT_SNAPSHOT_CHANGED")
    return path


def validate_ground_supplement(base: dict[str, Any], candidate: dict[str, Any],
                               patches: list[dict[str, Any]]) -> None:
    """Reconstruct the only allowed change; old geometry and exclusions cannot change."""
    reconstructed = base
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {
            "areas", "corridors", "exclusions", "provenance"
        }:
            raise ValueError("MAP_REFINEMENT_PATCH_INVALID")
        reconstructed = _apply_patch(reconstructed, patch)
    if sha256_json(reconstructed) != sha256_json(candidate):
        raise ValueError("MAP_REFINEMENT_UNRELATED_MUTATION")
    before, after = base.get("traversability", {}), candidate.get("traversability", {})
    for key in ("areas", "corridors", "exclusions"):
        current = {item["id"]: item for item in after.get(key, [])}
        if any(current.get(item["id"]) != item for item in before.get(key, [])):
            raise ValueError("MAP_REFINEMENT_BASE_FEATURE_CHANGED")
    for key in ("minimum_agl_m", "maximum_agl_m", "required", "schema_version"):
        if key in before and before[key] != after.get(key):
            raise ValueError("MAP_REFINEMENT_BASE_POLICY_CHANGED")


def prepare_ground_routes(
    *, graph: MapAsset, vehicle: VehicleAsset, semantic_path: Path,
    source_loader: Callable[[], MapImageSource],
    model_ports: list[StructuredModelPort], focus_nodes: list[str], output_dir: Path,
    maximum_model_calls: int, maximum_search_seconds: float = 60.0,
) -> GroundMapPreparation:
    """Check actual routes, then inspect missing ground and replan before freezing."""
    base = read_map_semantic_object(semantic_path)
    # Indoor legacy assets continue through their own qualified 3-D geometry.
    # Outdoor maps declare this required layer during map preparation.
    if "traversability" not in base:
        return GroundMapPreparation(semantic_path, None, [], 0)
    base_hash = hashlib.sha256(read_plugin_file(semantic_path, limit=_LIMIT)).hexdigest()
    directory = output_dir / "map-refinement"
    directory.mkdir(parents=True, exist_ok=True)
    diameter, height = vehicle.body_radius_m * 2.4, vehicle.body_height_m * 1.2
    route_checks: list[dict[str, Any]] = []
    latest_issue: list[str] = []
    route_cache: dict[str, tuple[bool, list[dict[str, Any]], list[str]]] = {}

    def route_available(candidate: dict[str, Any]) -> bool:
        digest = sha256_json(candidate)
        if digest in route_cache:
            success, checks, issues = route_cache[digest]
            route_checks[:] = checks
            latest_issue[:] = issues
            return success
        path = semantic_path if candidate == base else _snapshot(directory, candidate)
        route_checks.clear()
        try:
            planner = KnownMapMetricPlanner(
                graph=graph, semantic_path=path, vehicle_diameter_m=diameter,
                vehicle_height_m=height,
                policy=MetricPlannerPolicy(resolution_m=.9,
                    required_clearance_m=max(.35, vehicle.body_radius_m * .75),
                    maximum_search_seconds=maximum_search_seconds,
                    allow_semantic_endpoint_projection=True),
            )
            for start, goal in zip(focus_nodes, focus_nodes[1:], strict=False):
                if start == goal:
                    continue
                route = planner.plan(RouteQuery(start_node=start, goal_node=goal))
                checked = validate_route_clearance(route, path,
                    vehicle_diameter_m=diameter, vehicle_height_m=height)
                if not checked.accepted:
                    latest_issue[:] = ["MAP_REFINEMENT_ROUTE_CLEARANCE_REJECTED"]
                    route_cache[digest] = (False, list(route_checks), list(latest_issue))
                    return False
                route_checks.append({"start_node": start, "goal_node": goal,
                    "route_sha256": sha256_json(route), "route": route.model_dump(mode="json"),
                    "clearance": checked.model_dump(mode="json")})
        except ValueError as error:
            latest_issue[:] = [str(error)[:240]]
            route_cache[digest] = (False, list(route_checks), list(latest_issue))
            return False
        latest_issue.clear()
        route_cache[digest] = (True, list(route_checks), [])
        return True

    if route_available(base):
        return GroundMapPreparation(semantic_path, None, [], 0)
    try:
        source = source_loader()
    except (OSError, ValueError) as error:
        details = {"state": "awaiting-observation", "next_action": "supply-registered-map-view",
                   "issue_codes": [str(error)[:240], *latest_issue], "focus_nodes": focus_nodes,
                   "base_semantic_sha256": base_hash, "actuator_authority": False}
        (directory / "pending.json").write_bytes(_bytes(details))
        raise MapObservationPending(details) from error
    vision = next((port for port in model_ports if port.supports_image_input), None)
    accepted_patches: list[dict[str, Any]] = []
    current = base
    prior_calls: list[dict[str, Any]] = []
    prior_records: list[dict[str, Any]] = [{"state": "route-gap", "issue_codes": latest_issue}]
    requests = None
    prior_attempts = 0
    # A new planning attempt in this same conversation resumes its own exact
    # map/focus observation work. Never borrow a different map or another task.
    for pending_path in sorted(output_dir.parent.glob("plan-*/map-refinement/pending.json"),
                               reverse=True):
        if pending_path.parent == directory:
            continue
        try:
            pending = json.loads(read_plugin_file(pending_path, limit=_LIMIT))
            if (pending.get("base_semantic_sha256") != base_hash
                    or pending.get("base_map_sha256") != sha256_json(graph)
                    or pending.get("source_image_sha256") != source.image_sha256
                    or pending.get("focus_nodes") != focus_nodes
                    or "candidate_semantic" not in pending):
                continue
            prior_patches = pending["patches"]
            cached_calls = [ModelCallRecord.model_validate(item).model_dump(mode="json")
                            for item in pending["model_calls"]]
            validate_ground_supplement(base, pending["candidate_semantic"], prior_patches)
            _verify_observations(prior_patches, source, base_hash, cached_calls)
            requests = [ObservationRequest(**item) for item in pending["pending_requests"]]
            prior_attempts = int(pending["physical_calls"])
            if not 0 <= prior_attempts <= 48 or not 1 <= len(requests) <= 16:
                continue
        except (KeyError, TypeError, OSError, ValueError):
            continue
        current = pending["candidate_semantic"]
        accepted_patches.extend(prior_patches)
        prior_calls = cached_calls
        prior_records = pending["observation_records"]
        break
    previous_patches = list(accepted_patches)

    def validate(candidate: dict[str, Any], patch: dict[str, Any]) -> list[str]:
        if not any(patch[key] for key in ("areas", "corridors", "exclusions")):
            return ["MAP_OBSERVATION_NO_CONFIDENT_GEOMETRY"]
        try:
            validate_ground_supplement(base, candidate, [*accepted_patches, patch])
            TraversableAirspace(candidate, planning_collision_primitives(base),
                               radius_m=diameter / 2, half_height_m=height / 2)
        except ValueError as error:
            return [str(error)[:240]]
        accepted_patches.append(patch)
        return []

    # A user-requested continuation has a fresh bounded model grant. Historical
    # observations remain evidence, but are not charged to this request twice.
    available_calls = min(maximum_model_calls, max(0, 48 - prior_attempts))
    if available_calls <= 0:
        vision = None
    result = refine_map_gaps(graph=graph, semantic=current, semantic_sha256=base_hash,
        source=source, focus_nodes=focus_nodes, model_port=vision,
        validate_candidate=validate, route_available=route_available,
        max_observations=max(1, min(8, available_calls)), requests=requests,
        previous_observation_records=prior_records,
    )
    current_physical_calls = result.physical_calls
    result.patches = [*previous_patches, *result.patches]
    result.model_calls = [*prior_calls, *result.model_calls]
    result.physical_calls += prior_attempts
    return _freeze_result(result, base=base, base_hash=base_hash, graph=graph,
        source=source, focus_nodes=focus_nodes, route_checks=route_checks,
        output_dir=output_dir, directory=directory,
        current_physical_calls=current_physical_calls)


def _freeze_result(result: MapRefinementResult, *, base: dict[str, Any], base_hash: str,
                   graph: MapAsset, source: MapImageSource, focus_nodes: list[str],
                   route_checks: list[dict[str, Any]], output_dir: Path,
                   directory: Path, current_physical_calls: int) -> GroundMapPreparation:
    details = {"state": result.state, "base_semantic_sha256": base_hash,
        "base_map_sha256": sha256_json(graph), "focus_nodes": focus_nodes,
        "source_image_sha256": source.image_sha256,
        "observation_records": result.observation_records,
        "pending_requests": [asdict(item) for item in result.pending_requests],
        "physical_calls": result.physical_calls, "model_calls": result.model_calls,
        "patches": result.patches, "actuator_authority": False,
        "next_action": "inspect-map-gaps"}
    if result.state != "route-ready":
        # Keep accepted work even when another view/provider is needed; no plan or
        # execution authority is issued for an unresolved connection.
        details["candidate_semantic"] = result.semantic
        (directory / "pending.json").write_bytes(_bytes(details))
        raise MapObservationPending({**details,
                                     "pending_artifact": str(directory / "pending.json")})
    if not result.patches:
        raise ValueError("MAP_REFINEMENT_NO_OBSERVATION")
    validate_ground_supplement(base, result.semantic, result.patches)
    path = _snapshot(directory, result.semantic)
    receipt = {"schema_version": "dronedream.mission-map-refinement.v1",
        "base_semantic_sha256": base_hash, "base_map_sha256": sha256_json(graph),
        "derived_semantic_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "semantic_relative_path": path.relative_to(output_dir).as_posix(),
        "map_id": graph.asset_id, "source_image_sha256": source.image_sha256,
        "patches": result.patches, "route_checks": route_checks,
        "physical_calls": result.physical_calls, "model_calls": result.model_calls,
        "qualification": _QUALIFICATION,
        "base_qualification_scope": "unchanged-physical-world-and-aircraft"}
    (directory / "receipt.json").write_bytes(_bytes(receipt))
    return GroundMapPreparation(path, receipt, result.model_calls, current_physical_calls)


def resolve_execution_semantic(*, prepared: PreparedMission, output_dir: Path,
                              base_semantic_path: Path, graph: MapAsset,
                              source_loader: Callable[[], MapImageSource]) -> Path:
    """Verify the prepared authority-bound revision; never trust a free path override."""
    base_raw = read_plugin_file(base_semantic_path, limit=_LIMIT)
    base_hash = hashlib.sha256(base_raw).hexdigest()
    records = [item.payload for item in prepared.evidence if item.event_type == _EVENT]
    if not records:
        if prepared.contract.map_semantic_sha256 != base_hash:
            raise ValueError("MAP_REFINEMENT_RECEIPT_REQUIRED")
        return base_semantic_path
    if len(records) != 1:
        raise ValueError("MAP_REFINEMENT_RECEIPT_AMBIGUOUS")
    receipt = records[0]
    if (receipt.get("schema_version") != "dronedream.mission-map-refinement.v1"
            or receipt.get("base_semantic_sha256") != base_hash
            or receipt.get("base_map_sha256") != sha256_json(graph)
            or receipt.get("map_id") != graph.asset_id
            or receipt.get("qualification") != _QUALIFICATION
            or receipt.get("derived_semantic_sha256") != prepared.contract.map_semantic_sha256):
        raise ValueError("MAP_REFINEMENT_BASE_BINDING_MISMATCH")
    relative = receipt.get("semantic_relative_path")
    expected = f"map-refinement/semantic-{prepared.contract.map_semantic_sha256}.json"
    if relative != expected:
        raise ValueError("MAP_REFINEMENT_PATH_INVALID")
    path = output_dir / expected
    check_plain_plugin_path(path)
    if output_dir.resolve() not in path.resolve().parents:
        raise ValueError("MAP_REFINEMENT_PATH_INVALID")
    raw = read_plugin_file(path, limit=_LIMIT)
    if hashlib.sha256(raw).hexdigest() != prepared.contract.map_semantic_sha256:
        raise ValueError("MAP_REFINEMENT_SNAPSHOT_CHANGED")
    candidate = read_map_semantic_object(path)
    patches = receipt.get("patches")
    if not isinstance(patches, list) or not 1 <= len(patches) <= 16:
        raise ValueError("MAP_REFINEMENT_PATCH_INVALID")
    source = source_loader()
    if (source.map_id != graph.asset_id or source.semantic_sha256 != base_hash
            or source.image_sha256 != receipt.get("source_image_sha256")):
        raise ValueError("MAP_REFINEMENT_IMAGE_BINDING_MISMATCH")
    calls = receipt.get("model_calls", [])
    call_hashes = {sha256_json(item) for item in calls}
    prepared_call_hashes = {sha256_json(item) for item in prepared.model_calls}
    if not call_hashes or not call_hashes <= prepared_call_hashes:
        raise ValueError("MAP_REFINEMENT_MODEL_EVIDENCE_MISSING")
    _verify_observations(patches, source, base_hash, calls)
    validate_ground_supplement(json.loads(base_raw), candidate, patches)
    # The normal executor still revalidates the final route with this exact file.
    return path


def _verify_observations(patches: list[dict[str, Any]], source: MapImageSource,
                         base_hash: str, calls: list[dict[str, Any]]) -> None:
    call_hashes = {sha256_json(item) for item in calls}
    for patch in patches:
        provenance = patch.get("provenance", {})
        if (provenance.get("source_image_sha256") != source.image_sha256
                or provenance.get("source_semantic_sha256") != base_hash
                or provenance.get("model_call_sha256") not in call_hashes):
            raise ValueError("MAP_REFINEMENT_OBSERVATION_BINDING_MISMATCH")
        output = MapViewInterpretation.model_validate(provenance.get("model_output"))
        call = next(item for item in calls
                    if sha256_json(item) == provenance["model_call_sha256"])
        if (sha256_json(output) != provenance.get("model_output_sha256")
                or call.get("output_sha256") != sha256_json(output)):
            raise ValueError("MAP_REFINEMENT_MODEL_OUTPUT_MISMATCH")
        try:
            view = prepare_map_view(source, ObservationRequest(**provenance["observation_request"]))
        except (KeyError, TypeError) as error:
            raise ValueError("MAP_REFINEMENT_OBSERVATION_REQUEST_INVALID") from error
        observed = interpretation_patch(output, view)
        if (any(observed[key] != patch[key] for key in ("areas", "corridors", "exclusions"))
                or any(provenance.get(key) != value
                       for key, value in observed["provenance"].items())):
            raise ValueError("MAP_REFINEMENT_PIXEL_GEOMETRY_MISMATCH")
