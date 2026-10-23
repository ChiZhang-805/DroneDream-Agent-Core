"""Observe missing map connections and propose grounded traversability revisions.

This module never joins graph nodes merely because they are nearby. A real vision
port reads hash-bound crops, and the caller's geometry validator and planner decide
whether the resulting *candidate* snapshot can support a route. Asset publication
and qualification remain the caller's responsibility.
"""

from __future__ import annotations

import hashlib
import io
import math
from collections import defaultdict, deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .contracts import MapAsset
from .hashing import sha256_json
from .model_harness.model_port import ModelInvocationError, StructuredModelPort
from .model_harness.progress import report_progress


class ObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PixelPoint(ObservationModel):
    x: float
    y: float


class ObservedArea(ObservationModel):
    kind: Literal["road", "path", "lawn", "parking", "open-space", "building", "tree-canopy"]
    polygon_px: list[PixelPoint] = Field(min_length=3, max_length=128)
    confidence: float = Field(ge=0, le=1)
    observation: str = Field(min_length=1, max_length=600)


class ObservedCorridor(ObservationModel):
    kind: Literal["road", "path"]
    centerline_px: list[PixelPoint] = Field(min_length=2, max_length=128)
    width_px: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    observation: str = Field(min_length=1, max_length=600)


class MapViewInterpretation(ObservationModel):
    view_id: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    areas: list[ObservedArea] = Field(max_length=64)
    corridors: list[ObservedCorridor] = Field(max_length=64)
    unresolved_observations: list[str] = Field(max_length=16)


@dataclass(frozen=True)
class MapImageSource:
    """An orthographic image registered to the selected map's ENU ground plane.

    pixel_to_map is (a,b,c,d,e,f): east=a*u+b*v+c; north=d*u+e*v+f.
    Use a separately calibrated orthographic view for an additional perspective;
    an arbitrary oblique camera image is not an affine ground map.
    """

    map_id: str
    semantic_sha256: str
    image_bytes: bytes
    image_sha256: str
    pixel_to_map: tuple[float, float, float, float, float, float]
    source_id: str
    ground_z_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.image_bytes, bytes) or not 0 < len(self.image_bytes) <= 32 * 1024**2:
            raise ValueError("MAP_OBSERVATION_IMAGE_SIZE_INVALID")
        if hashlib.sha256(self.image_bytes).hexdigest() != self.image_sha256:
            raise ValueError("MAP_OBSERVATION_IMAGE_HASH_MISMATCH")
        values = (*self.pixel_to_map, self.ground_z_m)
        if len(self.pixel_to_map) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError("MAP_OBSERVATION_TRANSFORM_INVALID")
        a, b, _, d, e, _ = self.pixel_to_map
        if abs(a * e - b * d) < 1e-12:
            raise ValueError("MAP_OBSERVATION_TRANSFORM_SINGULAR")
        with Image.open(io.BytesIO(self.image_bytes)) as image:
            if image.width * image.height > 64_000_000:
                raise ValueError("MAP_OBSERVATION_IMAGE_SIZE_INVALID")
            image.verify()

    def map_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.pixel_to_map
        determinant = a * e - b * d
        return ((e * (x - c) - b * (y - f)) / determinant,
                (-d * (x - c) + a * (y - f)) / determinant)


@dataclass(frozen=True)
class ObservationRequest:
    reason: str
    focus_node_ids: tuple[str, ...]
    bounds_xy_m: tuple[float, float, float, float]
    scale: Literal["detail", "context", "expanded", "alternate"]


@dataclass(frozen=True)
class PreparedMapView:
    view_id: str
    image_bytes: bytes
    image_sha256: str
    width: int
    height: int
    pixel_to_map: tuple[float, float, float, float, float, float]
    source_id: str
    source_image_sha256: str
    source_crop_px: tuple[int, int, int, int]
    request: ObservationRequest
    ground_z_m: float


@dataclass
class MapRefinementResult:
    state: Literal["route-ready", "awaiting-observation"]
    semantic: dict[str, Any]
    patches: list[dict[str, Any]]
    model_calls: list[dict[str, Any]]
    observation_records: list[dict[str, Any]]
    pending_requests: list[ObservationRequest]
    physical_calls: int = 0


VISION_INSTRUCTIONS = """Inspect the supplied actual map image to recover traversable ground.
Coordinates must be pixels of THIS supplied crop (origin top-left), never guessed ENU.
Trace visible road/path bends and junctions; do not draw straight shortcuts between places.
Clear lawns, open paved space and parking surfaces are also candidate traversable areas.
Trace buildings and tree canopy as exclusions at every altitude. A forest path is usable
only along the visible clear opening, never by assuming trees or shadows are open ground.
For parking, identify the surface; 3D validation must still account for vehicles/obstacles.
Report obscured/uncertain connections for another view rather than inventing a connection.
Do not infer ground elevation, flight clearance, or asset qualification from these pixels.
Use narrow conservative boundaries when the visible extent is uncertain. Preserve the
provided view_id and image_sha256 exactly. Return only the requested structured schema.
"""


def diagnose_map_gaps(graph: MapAsset, focus_nodes: list[str] | None = None) -> dict[str, Any]:
    """Identify exact missing connections without treating a nearest pair as a road."""
    nodes = {node.node_id: node for node in graph.nodes}
    focus = list(dict.fromkeys(focus_nodes or graph.named_entities.values()))
    if any(node not in nodes for node in focus):
        raise ValueError("MAP_FOCUS_NODES_INVALID")
    weak: dict[str, set[str]] = defaultdict(set)
    directed: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        weak[edge.from_node].add(edge.to_node)
        weak[edge.to_node].add(edge.from_node)
        directed[edge.from_node].add(edge.to_node)
        if edge.bidirectional:
            directed[edge.to_node].add(edge.from_node)
    components: list[list[str]] = []
    unvisited = set(nodes)
    while unvisited:
        seed = min(unvisited)
        component = _reachable(seed, weak)
        components.append(sorted(component))
        unvisited.difference_update(component)
    pairs = []
    for start, goal in zip(focus, focus[1:], strict=False):
        if goal not in _reachable(start, directed):
            a, b = nodes[start].position_m, nodes[goal].position_m
            pairs.append({"start_node": start, "goal_node": goal,
                          "bounds_xy_m": [min(a.x, b.x), min(a.y, b.y),
                                          max(a.x, b.x), max(a.y, b.y)]})
    return {"component_count": len(components), "components": components,
            "isolated_node_ids": sorted(node for node in nodes if not weak[node]),
            "disconnected_focus_pairs": pairs,
            "next_action": "inspect-map-gaps" if pairs else "use-validated-traversability",
            "policy": "reobserve-expand-refine-replan; proximity never establishes a road"}


def _reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    found = {start}
    queue = deque([start])
    while queue:
        for node in adjacency.get(queue.popleft(), ()):
            if node not in found:
                found.add(node)
                queue.append(node)
    return found


def observation_requests(graph: MapAsset, focus_nodes: list[str]) -> list[ObservationRequest]:
    """Observe endpoints closely, then the surrounding connection, then wider context."""
    nodes = {node.node_id: node for node in graph.nodes}
    if not focus_nodes or any(node not in nodes for node in focus_nodes):
        raise ValueError("MAP_FOCUS_NODES_INVALID")
    gaps = diagnose_map_gaps(graph, focus_nodes)["disconnected_focus_pairs"]
    # A connected old graph can still violate traversability (e.g. a roof shortcut).
    # The caller invokes refinement after the real planner rejects that geometry.
    pairs = gaps or [{"start_node": focus_nodes[0], "goal_node": focus_nodes[-1]}]
    requests = []
    for pair in pairs:
        start, goal = pair["start_node"], pair["goal_node"]
        a, b = nodes[start].position_m, nodes[goal].position_m
        for node_id, point in ((goal, b), (start, a)):
            requests.append(ObservationRequest("inspect-access-to-open-ground", (node_id,),
                (point.x - 24, point.y - 24, point.x + 24, point.y + 24), "detail"))
        span = max(48, abs(a.x - b.x), abs(a.y - b.y))
        for factor, scale in ((0.2, "context"), (0.6, "expanded")):
            pad = max(24, span * factor)
            requests.append(ObservationRequest("trace-missing-traversable-connection",
                (start, goal), (min(a.x, b.x) - pad, min(a.y, b.y) - pad,
                               max(a.x, b.x) + pad, max(a.y, b.y) + pad), scale))
    return requests


def prepare_map_view(source: MapImageSource, request: ObservationRequest) -> PreparedMapView:
    """Crop and resize actual pixels, composing their exact affine transform."""
    x0, y0, x1, y1 = request.bounds_xy_m
    corners = [source.map_to_pixel(x, y) for x in (x0, x1) for y in (y0, y1)]
    with Image.open(io.BytesIO(source.image_bytes)) as original:
        left = max(0, math.floor(min(point[0] for point in corners)))
        top = max(0, math.floor(min(point[1] for point in corners)))
        right = min(original.width, math.ceil(max(point[0] for point in corners)))
        bottom = min(original.height, math.ceil(max(point[1] for point in corners)))
        if right - left < 2 or bottom - top < 2:
            raise ValueError("MAP_OBSERVATION_OUTSIDE_IMAGE")
        image = original.crop((left, top, right, bottom)).convert("RGB")
        image.thumbnail((1536, 1536))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        width, height = image.size
    a, b, c, d, e, f = source.pixel_to_map
    sx, sy = (right - left) / width, (bottom - top) / height
    transform = (a * sx, b * sy, a * left + b * top + c,
                 d * sx, e * sy, d * left + e * top + f)
    digest = hashlib.sha256(content).hexdigest()
    return PreparedMapView("map-view-" + sha256_json([digest, transform])[:24], content,
        digest, width, height, transform, source.source_id, source.image_sha256,
        (left, top, right, bottom), request, source.ground_z_m)


def interpretation_patch(
    output: MapViewInterpretation, view: PreparedMapView, *, minimum_confidence: float = 0.75,
) -> dict[str, Any]:
    """Convert observed crop pixels into map geometry, keeping source provenance."""
    if output.view_id != view.view_id or output.image_sha256 != view.image_sha256:
        raise ValueError("MAP_OBSERVATION_OUTPUT_BINDING_MISMATCH")
    a, b, c, d, e, f = view.pixel_to_map

    def position(point: PixelPoint) -> list[float]:
        if not (0 <= point.x <= view.width and 0 <= point.y <= view.height):
            raise ValueError("MAP_OBSERVATION_COORDINATE_OUTSIDE_VIEW")
        return [a * point.x + b * point.y + c, d * point.x + e * point.y + f]

    areas, corridors, exclusions, observations = [], [], [], []
    # Smallest singular value prevents an anisotropic transform overstating width.
    trace = a * a + b * b + d * d + e * e
    width_scale = math.sqrt(max(0, (trace - math.sqrt(max(0, trace**2
                               - 4 * (a * e - b * d)**2))) / 2))
    for index, area in enumerate(output.areas):
        polygon = [position(point) for point in area.polygon_px]
        if area.confidence < minimum_confidence and area.kind not in {"building", "tree-canopy"}:
            continue
        identifier = f"{view.view_id}-area-{index}"
        item = {"id": identifier, "kind": area.kind, "polygon_xy_m": polygon}
        if area.kind in {"building", "tree-canopy"}:
            exclusions.append(item)
        else:
            item["ground_z_m"] = view.ground_z_m
            areas.append(item)
        observations.append({"id": identifier, "confidence": area.confidence,
                             "observation": area.observation})
    for index, corridor in enumerate(output.corridors):
        line = [position(point) for point in corridor.centerline_px]
        if corridor.width_px > min(view.width, view.height):
            raise ValueError("MAP_OBSERVATION_WIDTH_OUTSIDE_VIEW")
        if corridor.confidence < minimum_confidence:
            continue
        identifier = f"{view.view_id}-corridor-{index}"
        corridors.append({"id": identifier, "kind": corridor.kind,
                          "centerline_xy_m": line,
                          "width_m": corridor.width_px * width_scale,
                          "ground_z_m": view.ground_z_m})
        observations.append({"id": identifier, "confidence": corridor.confidence,
                             "observation": corridor.observation})
    return {"areas": areas, "corridors": corridors, "exclusions": exclusions,
            "provenance": {"view_id": view.view_id, "source_id": view.source_id,
                "source_image_sha256": view.source_image_sha256,
                "view_image_sha256": view.image_sha256,
                "source_crop_px": list(view.source_crop_px),
                "view_pixel_to_map": list(view.pixel_to_map), "coordinate_frame": "ENU",
                "observations": observations,
                "unresolved_observations": output.unresolved_observations,
                "qualification": "candidate-observation-not-flight-qualification"}}


def _apply_patch(semantic: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(semantic)
    traversal = candidate.setdefault("traversability", {
        "schema_version": "dronedream.traversability.v1", "required": True,
        "minimum_agl_m": 0.0, "maximum_agl_m": 12.0,
        "areas": [], "corridors": [], "exclusions": [],
    })
    traversal["required"] = True
    for key in ("areas", "corridors", "exclusions"):
        existing = {item["id"]: item for item in traversal.get(key, [])}
        existing.update({item["id"]: deepcopy(item) for item in patch[key]})
        traversal[key] = list(existing.values())
    candidate.setdefault("traversability_refinements", []).append(deepcopy(patch["provenance"]))
    return candidate


def refine_map_gaps(
    *, graph: MapAsset, semantic: dict[str, Any], semantic_sha256: str,
    source: MapImageSource, focus_nodes: list[str], model_port: StructuredModelPort | None,
    validate_candidate: Callable[[dict[str, Any], dict[str, Any]], list[str]],
    route_available: Callable[[dict[str, Any]], bool], max_observations: int = 4,
    additional_view: Callable[[ObservationRequest], MapImageSource | None] | None = None,
    requests: list[ObservationRequest] | None = None,
    previous_observation_records: list[dict[str, Any]] | None = None,
) -> MapRefinementResult:
    """Run bounded real observation→validation→replanning; retain unresolved work.

    validate_candidate returns geometric issue codes (empty means accepted).
    route_available must run the deterministic planner for this exact snapshot.
    Neither callback should publish/qualify a map or start an aircraft. The caller
    publishes an accepted revision and binds its new hash before execution.
    additional_view may supply a real calibrated scene view after primary crops.
    A text-only selected model needs an eligible image-capable port supplied here.
    """
    if source.map_id != graph.asset_id or source.semantic_sha256 != semantic_sha256:
        raise ValueError("MAP_OBSERVATION_SOURCE_BINDING_MISMATCH")
    if not 1 <= max_observations <= 16:
        raise ValueError("MAP_OBSERVATION_BUDGET_INVALID")
    current = deepcopy(semantic)
    result = MapRefinementResult("awaiting-observation", current, [], [],
                                deepcopy((previous_observation_records or [])[-8:]), [])
    if route_available(current):
        result.state = "route-ready"
        return result
    queue = list(requests) if requests is not None else observation_requests(graph, focus_nodes)
    if not queue:
        raise ValueError("MAP_OBSERVATION_REQUESTS_EMPTY")
    if model_port is None or not model_port.supports_image_input:
        result.pending_requests = queue
        result.observation_records.append({"state": "needs-vision-provider",
            "reason": "selected model has no declared image capability"})
        return result
    if additional_view and not any(request.scale == "alternate" for request in queue):
        queue.append(ObservationRequest("inspect-remaining-occlusion", tuple(focus_nodes),
                                        queue[-1].bounds_xy_m, "alternate"))
    seen: set[str] = set()
    attempts = 0
    while queue and attempts < max_observations:
        request = queue.pop(0)
        active_source = source
        if request.scale == "alternate":
            active_source = additional_view(request) if additional_view else None
            if active_source is None:
                result.pending_requests.append(request)
                continue
            if (active_source.map_id != source.map_id
                    or active_source.semantic_sha256 != semantic_sha256):
                raise ValueError("MAP_OBSERVATION_SOURCE_BINDING_MISMATCH")
        try:
            view = prepare_map_view(active_source, request)
        except ValueError as error:
            result.observation_records.append({"state": "needs-registered-view",
                                               "reason": str(error)})
            result.pending_requests.append(request)
            continue
        if view.view_id in seen:
            continue
        seen.add(view.view_id)
        report_progress("map_refinement", "正在放大核对道路、开阔地与障碍，补全地图连接。",
                        "Inspecting roads, open ground and obstacles to complete map connections.")
        attempts += 1
        result.physical_calls += 1
        try:
            called = model_port.call(role="context_summarizer", output_type=MapViewInterpretation,
                instructions=VISION_INSTRUCTIONS,
                input_artifact={"purpose": "map-gap-refinement", "map_id": graph.asset_id,
                    "source_semantic_sha256": semantic_sha256,
                    "view_id": view.view_id, "image_sha256": view.image_sha256,
                    "image_size_px": [view.width, view.height],
                    "focus_node_ids": list(request.focus_node_ids),
                    "observation_scale": request.scale,
                    "previous_observations": result.observation_records[-4:]},
                multimodal=[{"kind": "image-file", "content_bytes": view.image_bytes,
                             "sha256": view.image_sha256, "content_type": "image/png"}],
                maximum_physical_attempts=1)
            result.model_calls.append(called.record.model_dump(mode="json"))
            patch = interpretation_patch(called.artifact, view)
            patch["provenance"].update({"map_id": graph.asset_id,
                "source_semantic_sha256": semantic_sha256,
                "model_call_sha256": sha256_json(result.model_calls[-1]),
                "model_output_sha256": sha256_json(called.artifact),
                "model_output": called.artifact.model_dump(mode="json"),
                "observation_request": {"reason": request.reason,
                    "focus_node_ids": list(request.focus_node_ids),
                    "bounds_xy_m": list(request.bounds_xy_m), "scale": request.scale}})
            candidate = _apply_patch(current, patch)
            issues = validate_candidate(candidate, patch)
            if not any(patch[key] for key in ("areas", "corridors", "exclusions")):
                issues = [*issues, "MAP_OBSERVATION_NO_CONFIDENT_GEOMETRY"]
            if issues:
                result.observation_records.append({"view_id": view.view_id,
                    "state": "reobserve", "issue_codes": issues})
                continue
        except (ModelInvocationError, ValueError) as error:
            result.observation_records.append({"view_id": view.view_id, "state": "reobserve",
                "issue_codes": [getattr(error, "reason_code", type(error).__name__)]})
            continue
        current = candidate
        result.semantic = current
        result.patches.append(patch)
        result.observation_records.append({"view_id": view.view_id, "state": "geometry-accepted",
            "unresolved_observations": called.artifact.unresolved_observations})
        if route_available(current):
            result.state = "route-ready"
            result.pending_requests = []
            return result
    result.pending_requests.extend(queue)
    if not result.pending_requests:
        last = observation_requests(graph, focus_nodes)[-1]
        result.pending_requests.append(ObservationRequest("inspect-remaining-occlusion",
            tuple(focus_nodes), last.bounds_xy_m, "alternate"))
    return result
