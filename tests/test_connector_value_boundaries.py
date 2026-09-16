"""Verify direct connector calls keep authority and data boundaries, with no external I/O."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from jsonschema import ValidationError

from dronedream_agent_core.capability_broker import BrokerHttpResponse
from dronedream_agent_plugins.connector_plugins import plugin_definitions


def _tool(name, payload, configuration=None):
    """Register the product tool against an inert provider fixture and captured call list."""
    calls = []

    def request(method, url, **kwargs):
        """Expose exact request parameters without issuing network traffic."""
        calls.append((method, url, kwargs))
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return BrokerHttpResponse(status=200, headers={}, body=body)

    environment = SimpleNamespace(
        capability_broker=SimpleNamespace(request=request), plugin_configuration=configuration
    )
    definition = next(item for item in plugin_definitions() if item.manifest.plugin_id == name)
    return definition.tool_factory(environment)[0], calls


@pytest.mark.parametrize(
    "value",
    [
        {"south": 0, "west": 0, "north": 0.1, "east": 0.1, "feature_classes": ['x"];out;']},
        {"south": True, "west": 0, "north": 0.1, "east": 0.1},
        {"south": 0, "west": 0, "north": float("nan"), "east": 0.1},
        {"south": 0, "west": 0, "north": 0.1, "east": 0.1, "feature_classes": ["building"] * 2},
    ],
)
def test_invalid_gis_arguments_never_reach_broker(value):
    """Enums and finite numeric contracts hold even when the registry is bypassed."""
    tool, calls = _tool("connector.gis.overpass", {"elements": []})
    with pytest.raises((ValueError, ValidationError)):
        tool.handler(value)
    assert calls == []


def test_presentation_schema_cannot_weaken_handler():
    """A mutable schema exposed to UI consumers is not the handler's validation source."""
    tool, calls = _tool("connector.gis.overpass", {"elements": []})
    tool.input_schema["properties"]["feature_classes"]["items"].pop("enum")
    with pytest.raises(ValidationError):
        tool.handler(
            {
                "south": 0,
                "west": 0,
                "north": 0.1,
                "east": 0.1,
                "feature_classes": ["arbitrary-query"],
            }
        )
    assert calls == []


@pytest.mark.parametrize("payload", [{}, {"elements": [1]}, {"elements": {}}])
def test_malformed_gis_is_not_empty_success(payload):
    """Count and records cannot disagree through silent filtering."""
    tool, _ = _tool("connector.gis.overpass", payload)
    with pytest.raises(RuntimeError, match="ELEMENTS_INVALID"):
        tool.handler({"south": 0, "west": 0, "north": 0.1, "east": 0.1})


@pytest.mark.parametrize("payload", [b'{"elements":[],"elements":[]}', b'{"elements":[NaN]}'])
def test_provider_json_rejects_ambiguity(payload):
    """Duplicate keys and non-standard numbers fail before provider normalization."""
    tool, _ = _tool("connector.gis.overpass", payload)
    with pytest.raises(ValueError):
        tool.handler({"south": 0, "west": 0, "north": 0.1, "east": 0.1})


def test_notion_preserves_cursor_and_frozen_configuration():
    """One page exposes continuation and cannot be redirected by subsequent config mutation."""
    config = {"database_id": "original", "credential_reference": "notion-ref"}
    tool, calls = _tool(
        "connector.erp.notion",
        {"results": [{"id": "record"}], "has_more": True, "next_cursor": "next"},
        config,
    )
    config["database_id"] = "changed"
    result = tool.handler({"page_size": 3, "cursor": "prior"})
    assert result["pagination"] == {"has_more": True, "next_cursor": "next"}
    assert "/original/query" in calls[0][1]
    assert json.loads(calls[0][2]["body"]) == {"page_size": 3, "start_cursor": "prior"}


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [], "has_more": "false", "next_cursor": None},
        {"results": [], "has_more": True, "next_cursor": None},
        {"results": [], "has_more": False, "next_cursor": "unexpected"},
    ],
)
def test_notion_rejects_ambiguous_continuation(payload):
    """A missing/invalid cursor must not cause records to vanish from future pages."""
    tool, _ = _tool(
        "connector.erp.notion", payload, {"database_id": "db", "credential_reference": "ref"}
    )
    with pytest.raises(RuntimeError, match="PAGINATION_INVALID"):
        tool.handler({})


def test_pagerduty_preserves_offset_page():
    """Pagination values are usable for an explicit next request, not automatic unbounded loops."""
    tool, calls = _tool(
        "connector.alerts.pagerduty",
        {"incidents": [], "more": True, "offset": 20, "limit": 10},
        {"credential_reference": "ref"},
    )
    result = tool.handler({"offset": 20, "limit": 10})
    assert result["pagination"]["next_offset"] == 30
    assert parse_qs(urlsplit(calls[0][1]).query)["offset"] == ["20"]


@pytest.mark.parametrize(
    "name,args,payload",
    [
        ("connector.bim.autodesk", {"identifier": "urn"}, {"data": {}}),
        ("connector.bim.autodesk", {"identifier": "urn"}, {"data": {"metadata": [1]}}),
        ("connector.logistics.aftership", {"carrier": "dhl", "tracking_number": "ABC123"}, {}),
        (
            "connector.logistics.aftership",
            {"carrier": "dhl", "tracking_number": "ABC123"},
            {"data": {"tracking": {"tag": {}, "checkpoints": []}}},
        ),
    ],
)
def test_provider_shape_errors_are_not_empty_records(name, args, payload):
    """Missing or invalid records/status must be observable failures, not successful absence."""
    tool, _ = _tool(name, payload, {"credential_reference": "ref"})
    with pytest.raises((RuntimeError, ValidationError)):
        tool.handler(args)
