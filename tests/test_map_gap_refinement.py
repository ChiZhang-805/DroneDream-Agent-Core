import hashlib
import io
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from dronedream_agent_core.contracts import MapAsset, MapEdge, MapNode, Vector3
from dronedream_agent_core.map_gap_refinement import (
    MapImageSource,
    MapViewInterpretation,
    ObservationRequest,
    diagnose_map_gaps,
    interpretation_patch,
    observation_requests,
    prepare_map_view,
    refine_map_gaps,
)
from dronedream_agent_core.model_harness.model_port import (
    ModelInvocationError,
    StructuredModelPort,
)


@pytest.fixture
def graph():
    return MapAsset(asset_id="test-map", name="Test map", nodes=[
        MapNode(node_id=name, label=name, position_m=Vector3(x=x, y=y, z=1), semantic=kind)
        for name, x, y, kind in [("start", 20, 20, "launch"), ("road", 40, 20, "corridor"),
                                 ("pickup", 100, 100, "pickup")]],
        edges=[MapEdge(edge_id="start-road", from_node="start", to_node="road",
                       distance_m=20, minimum_clearance_m=2, speed_limit_mps=2,
                       bidirectional=False)], named_entities={"start": "start", "lab": "pickup"})


@pytest.fixture
def source():
    # A synthetic calibration target, never a claimed production map observation.
    image = Image.new("RGB", (256, 256), color=(70, 130, 40))
    image.putpixel((70, 80), (210, 210, 210))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    content = buffer.getvalue()
    return MapImageSource("test-map", "a" * 64, content, hashlib.sha256(content).hexdigest(),
                          (1, 0, 0, 0, 1, 0), "test-orthophoto")


def _interpretation(call, *, kind="road", confidence=0.9):
    context = call["input_artifact"]
    return MapViewInterpretation.model_validate({
        "view_id": context["view_id"], "image_sha256": context["image_sha256"],
        "areas": [{"kind": kind, "polygon_px": [{"x": 2, "y": 2}, {"x": 8, "y": 2},
                                                  {"x": 8, "y": 8}, {"x": 2, "y": 8}],
                   "confidence": confidence, "observation": "Test-only region"}],
        "corridors": [], "unresolved_observations": []})


class RecordingPort:
    supports_image_input = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        # Exercise the actual production image byte/hash admission path.
        assert StructuredModelPort._media_data_url(kwargs["multimodal"][0]).startswith(
            "data:image/png;base64,")
        response = next(self.responses)(kwargs)
        return SimpleNamespace(artifact=response, record=SimpleNamespace(
            model_dump=lambda **_: {"attempt": 1, "provider": "test-only",
                                    "call_id": f"call-{len(self.calls)}"}))


def test_diagnosis_records_isolated_nodes_and_directed_gaps(graph):
    result = diagnose_map_gaps(graph, ["road", "start", "pickup"])
    assert result["component_count"] == 2
    assert result["isolated_node_ids"] == ["pickup"]
    assert [(item["start_node"], item["goal_node"])
            for item in result["disconnected_focus_pairs"]] == [("road", "start"),
                                                               ("start", "pickup")]
    assert result["next_action"] == "inspect-map-gaps"
    requests = observation_requests(graph, ["start", "pickup"])
    assert [request.scale for request in requests] == ["detail", "detail", "context", "expanded"]
    assert requests[-1].bounds_xy_m[0] < requests[-2].bounds_xy_m[0]
    assert len(graph.edges) == 1  # Diagnosis does not invent a connecting edge.


def test_crop_preserves_source_pixels_and_world_registration(source):
    request = ObservationRequest("test", ("pickup",), (50, 60, 110, 120), "detail")
    view = prepare_map_view(source, request)
    with Image.open(io.BytesIO(view.image_bytes)) as image:
        assert image.getpixel((20, 20)) == (210, 210, 210)
    assert view.source_crop_px == (50, 60, 110, 120)
    assert view.pixel_to_map == (1, 0, 50, 0, 1, 60)
    assert view.source_image_sha256 == source.image_sha256
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        MapImageSource(source.map_id, source.semantic_sha256, source.image_bytes, "b" * 64,
                       source.pixel_to_map, source.source_id)


def test_refinement_reobserves_rejected_geometry_then_replans(graph, source):
    semantic = {"traversability": {"schema_version": "dronedream.traversability.v1",
        "required": True, "minimum_agl_m": 0, "maximum_agl_m": 12,
        "areas": [], "corridors": [], "exclusions": [
            {"id": "known-building", "kind": "building",
             "polygon_xy_m": [[50, 50], [60, 50], [60, 60], [50, 60]]}]}}
    original = deepcopy(semantic)
    port = RecordingPort([lambda call: _interpretation(call, kind="parking"),
                          lambda call: _interpretation(call, kind="path"),
                          lambda call: _interpretation(call, kind="lawn")])
    validation_calls = []

    def validate(candidate, patch):
        validation_calls.append(patch)
        return ["INTERSECTS_BUILDING"] if len(validation_calls) == 1 else []

    planned_snapshots = []

    def replan(candidate):
        planned_snapshots.append(deepcopy(candidate))
        return len(candidate.get("traversability", {}).get("areas", [])) == 2

    result = refine_map_gaps(graph=graph, semantic=semantic, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=port,
        validate_candidate=validate, route_available=replan)
    assert result.state == "route-ready"
    assert len(port.calls) == 3
    assert len(result.patches) == 2
    assert len(planned_snapshots) == 3  # initial + two accepted revisions
    assert (result.semantic["traversability"]["exclusions"]
            == original["traversability"]["exclusions"])
    assert semantic == original
    assert result.observation_records[0]["issue_codes"] == ["INTERSECTS_BUILDING"]
    provenance = result.semantic["traversability_refinements"][0]
    assert provenance["source_image_sha256"] == source.image_sha256
    assert provenance["source_semantic_sha256"] == "a" * 64
    assert provenance["model_call_sha256"]
    assert provenance["qualification"] == "candidate-observation-not-flight-qualification"
    assert not result.pending_requests


def test_text_only_model_requests_real_vision_instead_of_faking_observation(graph, source):
    port = SimpleNamespace(supports_image_input=False)
    result = refine_map_gaps(graph=graph, semantic={}, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=port,
        validate_candidate=lambda *_: [], route_available=lambda _: False)
    assert result.state == "awaiting-observation"
    assert result.observation_records[0]["state"] == "needs-vision-provider"
    assert len(result.pending_requests) == 4
    assert not result.model_calls and not result.patches


def test_refinement_bounds_provider_calls_and_retains_remaining_work(graph, source):
    port = RecordingPort([lambda call: _interpretation(call, confidence=0.2)] * 2)
    result = refine_map_gaps(graph=graph, semantic={}, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=port,
        validate_candidate=lambda *_: [], route_available=lambda _: False, max_observations=2)
    assert result.state == "awaiting-observation"
    assert len(port.calls) == 2 and not result.patches
    assert [request.scale for request in result.pending_requests] == ["context", "expanded"]
    assert all(call["maximum_physical_attempts"] == 1 for call in port.calls)


def test_image_or_map_binding_cannot_be_substituted(graph, source):
    def wrong_view(call):
        interpretation = _interpretation(call)
        interpretation.image_sha256 = "e" * 64
        return interpretation

    port = RecordingPort([wrong_view])
    result = refine_map_gaps(graph=graph, semantic={}, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=port,
        validate_candidate=lambda *_: pytest.fail("unbound geometry must not be admitted"),
        route_available=lambda _: False, max_observations=1)
    assert not result.patches
    with pytest.raises(ValueError, match="SOURCE_BINDING_MISMATCH"):
        refine_map_gaps(graph=graph, semantic={}, semantic_sha256="b" * 64,
            source=source, focus_nodes=["start", "pickup"], model_port=port,
            validate_candidate=lambda *_: [], route_available=lambda _: False)


def test_uncertain_canopy_is_not_dropped_into_traversable_ground(source):
    view = prepare_map_view(source, ObservationRequest("test", (), (0, 0, 30, 30), "detail"))
    output = _interpretation({"input_artifact": {"view_id": view.view_id,
        "image_sha256": view.image_sha256}}, kind="tree-canopy", confidence=0.3)
    patch = interpretation_patch(output, view)
    assert patch["exclusions"][0]["kind"] == "tree-canopy"
    assert not patch["areas"]
    output.areas[0].polygon_px[0].x = 500
    with pytest.raises(ValueError, match="COORDINATE_OUTSIDE_VIEW"):
        interpretation_patch(output, view)


def test_provider_failure_advances_to_another_view(graph, source):
    def failed(_):
        raise ModelInvocationError("test", attempts_used=1)

    port = RecordingPort([failed, _interpretation])
    result = refine_map_gaps(graph=graph, semantic={}, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=port,
        validate_candidate=lambda *_: [],
        route_available=lambda sem: bool(sem.get("traversability", {}).get("areas")),
        max_observations=2)
    assert result.state == "route-ready"
    assert len(port.calls) == 2
    assert result.physical_calls == 2
    assert result.observation_records[0]["issue_codes"] == ["MODEL_INVOCATION_FAILED"]


def test_remaining_observation_work_can_resume_without_repeating_crops(graph, source):
    first_port = RecordingPort([lambda call: _interpretation(call, confidence=0.2)])
    first = refine_map_gaps(graph=graph, semantic={}, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=first_port,
        validate_candidate=lambda *_: [], route_available=lambda _: False, max_observations=1)
    second_port = RecordingPort([_interpretation])
    second = refine_map_gaps(graph=graph, semantic=first.semantic, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=second_port,
        validate_candidate=lambda *_: [],
        route_available=lambda sem: bool(sem.get("traversability", {}).get("areas")),
        max_observations=1, requests=first.pending_requests,
        previous_observation_records=first.observation_records)
    assert second.state == "route-ready"
    assert first_port.calls[0]["input_artifact"]["view_id"] != (
        second_port.calls[0]["input_artifact"]["view_id"])
    assert second_port.calls[0]["input_artifact"]["previous_observations"]


def test_real_alternate_registered_view_is_used_for_remaining_occlusion(graph, source):
    requests = [ObservationRequest("occlusion", ("pickup",), (70, 70, 130, 130), "alternate")]
    viewed = []

    def alternate(request):
        viewed.append(request)
        return source

    port = RecordingPort([_interpretation])
    result = refine_map_gaps(graph=graph, semantic={}, semantic_sha256="a" * 64,
        source=source, focus_nodes=["start", "pickup"], model_port=port,
        validate_candidate=lambda *_: [],
        route_available=lambda sem: bool(sem.get("traversability", {}).get("areas")),
        max_observations=1, requests=requests, additional_view=alternate)
    assert result.state == "route-ready" and viewed == requests
    assert port.calls[0]["input_artifact"]["observation_scale"] == "alternate"
