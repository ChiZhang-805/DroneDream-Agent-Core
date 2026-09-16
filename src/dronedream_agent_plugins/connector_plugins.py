"""Real, opt-in external data connectors mediated by the core capability broker."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from urllib.parse import quote, urlencode

from jsonschema import Draft202012Validator

from dronedream_agent_core.capability_broker import BrokerHttpResponse
from dronedream_agent_core.plugin_api import PluginDefinition, ToolEnvironment
from dronedream_agent_core.plugin_contracts import (
    PluginCapability,
    PluginManifest,
    PluginPlacement,
    PluginRuntime,
)
from dronedream_agent_core.tools import ToolPlugin
from dronedream_plugin_sdk.protocol import copy_json, decode_json

CONNECTOR_ROUTING_METADATA: dict[str, dict[str, object]] = {
    "connector.weather.open-meteo.forecast": {
        "domains": ["weather", "wind", "visibility", "outdoor-flight"],
        "purpose": "Read model-derived weather and hourly forecasts for explicit coordinates.",
        "selection_policy": "explicit-arguments-only",
        "required_argument_sources": {
            "latitude": "explicit mission artifact or qualified geospatial asset",
            "longitude": "explicit mission artifact or qualified geospatial asset",
        },
    },
    "connector.gis.overpass.features": {
        "domains": ["gis", "buildings", "roads", "barriers", "outdoor-flight"],
        "purpose": "Read bounded OpenStreetMap geometry for an explicit bounding box.",
        "selection_policy": "explicit-arguments-only",
        "required_argument_sources": {
            "south": "explicit mission artifact or qualified geospatial asset",
            "west": "explicit mission artifact or qualified geospatial asset",
            "north": "explicit mission artifact or qualified geospatial asset",
            "east": "explicit mission artifact or qualified geospatial asset",
        },
    },
    "connector.bim.autodesk.metadata": {
        "domains": ["bim", "cad", "facility", "inspection"],
        "purpose": "Read Autodesk model-derivative metadata for an explicit design URN.",
        "selection_policy": "explicit-arguments-only",
        "required_argument_sources": {
            "identifier": "explicit mission artifact or selected external asset version"
        },
    },
    "connector.logistics.aftership.tracking": {
        "domains": ["logistics", "delivery", "pickup", "payload"],
        "purpose": "Read live carrier tracking evidence for an explicit shipment.",
        "selection_policy": "explicit-arguments-only",
        "required_argument_sources": {
            "carrier": "explicit mission artifact",
            "tracking_number": "explicit mission artifact",
        },
    },
    "connector.erp.notion.database-query": {
        "domains": ["erp", "work-order", "maintenance", "inspection"],
        "purpose": "Read configured work orders or mission records from Notion.",
        "selection_policy": "configured-resource-only",
        "required_argument_sources": {
            "database_id": "immutable plugin configuration",
        },
    },
    "connector.alerts.pagerduty.incidents": {
        "domains": ["alerts", "incident", "emergency", "operations"],
        "purpose": "Read current operational incidents from the configured PagerDuty account.",
        "selection_policy": "configured-resource-only",
        "required_argument_sources": {
            "account": "immutable plugin configuration",
        },
    },
}


def _object_schema(required: list[str], properties: dict[str, object]) -> dict[str, object]:
    """Detach reusable schema fragments so one connector cannot mutate another's contract."""
    return copy_json(
        {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }
    )


def _configuration(environment: ToolEnvironment) -> dict[str, object]:
    """Freeze JSON configuration at tool creation; no credential value is copied here."""
    value = copy_json(
        {} if environment.plugin_configuration is None else environment.plugin_configuration
    )
    if not isinstance(value, dict):
        raise ValueError("CONNECTOR_CONFIGURATION_INVALID")
    return value


def _broker(environment: ToolEnvironment):
    """Require the core's authority-limited network facade, never direct HTTP fallback."""
    if environment.capability_broker is None:
        raise RuntimeError("CONNECTOR_CAPABILITY_BROKER_REQUIRED")
    return environment.capability_broker


def _json(response: BrokerHttpResponse) -> dict[str, Any]:
    """Reject errors, excessive/ambiguous JSON and non-finite values before normalization."""
    if response.status < 200 or response.status >= 300:
        raise RuntimeError(f"CONNECTOR_HTTP_STATUS_{response.status}")
    value = decode_json(response.body)
    if not isinstance(value, dict):
        raise RuntimeError("CONNECTOR_RESPONSE_OBJECT_REQUIRED")
    return value


def _credential(configuration: dict[str, object]) -> str:
    """Return an opaque credential reference resolved only by the broker."""
    value = configuration.get("credential_reference")
    if not isinstance(value, str) or not value:
        raise RuntimeError("CONNECTOR_CREDENTIAL_REFERENCE_REQUIRED")
    return value


WEATHER_INPUT = _object_schema(
    ["latitude", "longitude"],
    {
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        "forecast_days": {"type": "integer", "minimum": 1, "maximum": 16},
    },
)
WEATHER_OUTPUT = _object_schema(
    [
        "source",
        "latitude",
        "longitude",
        "timezone",
        "current",
        "hourly",
        "current_units",
        "hourly_units",
        "utc_offset_seconds",
    ],
    {
        "source": {"const": "open-meteo"},
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
        "timezone": {"type": "string"},
        "current": {"type": "object"},
        "hourly": {"type": "object"},
        "current_units": {"type": "object"},
        "hourly_units": {"type": "object"},
        "utc_offset_seconds": {"type": "integer"},
    },
)


def _weather_tools(environment: ToolEnvironment) -> list[ToolPlugin]:
    """Create forecast reads; these grid estimates are not onboard wind-sensor measurements."""
    broker = _broker(environment)

    def current_and_forecast(value: dict[str, object]) -> dict[str, object]:
        """Request SI wind units explicitly and preserve provider time/unit metadata."""
        query = urlencode(
            {
                "latitude": value["latitude"],
                "longitude": value["longitude"],
                "forecast_days": int(value.get("forecast_days", 3)),
                "timezone": "auto",
                "wind_speed_unit": "ms",
                "current": (
                    "temperature_2m,relative_humidity_2m,precipitation,weather_code,"
                    "cloud_cover,wind_speed_10m,wind_gusts_10m"
                ),
                "hourly": (
                    "temperature_2m,precipitation_probability,precipitation,weather_code,"
                    "cloud_cover,visibility,wind_speed_10m,wind_gusts_10m"
                ),
            }
        )
        payload = _json(broker.request("GET", f"https://api.open-meteo.com/v1/forecast?{query}"))
        return {
            "source": "open-meteo",
            "latitude": payload["latitude"],
            "longitude": payload["longitude"],
            "timezone": payload["timezone"],
            "current": payload["current"],
            "hourly": payload["hourly"],
            "current_units": payload["current_units"],
            "hourly_units": payload["hourly_units"],
            "utc_offset_seconds": payload["utc_offset_seconds"],
        }

    return [
        ToolPlugin(
            tool_id="connector.weather.open-meteo.forecast",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=WEATHER_INPUT,
            output_schema=WEATHER_OUTPUT,
            handler=current_and_forecast,
            routing_metadata=CONNECTOR_ROUTING_METADATA["connector.weather.open-meteo.forecast"],
        )
    ]


GIS_INPUT = _object_schema(
    ["south", "west", "north", "east"],
    {
        "south": {"type": "number", "minimum": -90, "maximum": 90},
        "west": {"type": "number", "minimum": -180, "maximum": 180},
        "north": {"type": "number", "minimum": -90, "maximum": 90},
        "east": {"type": "number", "minimum": -180, "maximum": 180},
        "feature_classes": {
            "type": "array",
            "items": {"type": "string", "enum": ["building", "highway", "barrier"]},
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
        },
    },
)
GIS_OUTPUT = _object_schema(
    ["source", "element_count", "elements"],
    {
        "source": {"const": "openstreetmap-overpass"},
        "element_count": {"type": "integer", "minimum": 0},
        "elements": {"type": "array", "items": {"type": "object"}},
    },
)


def _gis_tools(environment: ToolEnvironment) -> list[ToolPlugin]:
    """Expose bounded OSM declarations; geometry still requires local asset qualification."""
    broker = _broker(environment)

    def query_features(value: dict[str, object]) -> dict[str, object]:
        """Build a bounded query from validated enums, never arbitrary Overpass snippets."""
        south, west, north, east = (
            float(value[name]) for name in ("south", "west", "north", "east")
        )
        if south >= north or west >= east or (north - south) * (east - west) > 0.25:
            raise ValueError("CONNECTOR_GIS_BOUNDS_INVALID_OR_TOO_LARGE")
        classes = value.get("feature_classes", ["building", "highway", "barrier"])
        assert isinstance(classes, list)
        statements = "".join(f'nwr["{item}"]({south},{west},{north},{east});' for item in classes)
        overpass_query = f"[out:json][timeout:25];({statements});out center geom;"
        response = broker.request(
            "POST",
            "https://overpass-api.de/api/interpreter",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=urlencode({"data": overpass_query}).encode("utf-8"),
        )
        payload = _json(response)
        elements = payload.get("elements")
        if not isinstance(elements, list) or any(not isinstance(item, dict) for item in elements):
            raise RuntimeError("CONNECTOR_GIS_ELEMENTS_INVALID")
        return {
            "source": "openstreetmap-overpass",
            "element_count": len(elements),
            "elements": elements,
        }

    return [
        ToolPlugin(
            tool_id="connector.gis.overpass.features",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=GIS_INPUT,
            output_schema=GIS_OUTPUT,
            handler=query_features,
            routing_metadata=CONNECTOR_ROUTING_METADATA["connector.gis.overpass.features"],
        )
    ]


IDENTIFIER_INPUT = _object_schema(
    ["identifier"],
    {"identifier": {"type": "string", "minLength": 1, "maxLength": 512}},
)
RECORDS_OUTPUT = _object_schema(
    ["source", "records"],
    {
        "source": {"type": "string"},
        "records": {"type": "array", "items": {"type": "object"}},
        "pagination": {"type": "object"},
    },
)


def _autodesk_tools(environment: ToolEnvironment) -> list[ToolPlugin]:
    """Bind BIM metadata reads to a captured configuration and broker-held credential."""
    broker = _broker(environment)
    configuration = _configuration(environment)

    def metadata(value: dict[str, object]) -> dict[str, object]:
        """Escape one explicit design URN and reject malformed responses, not empty success."""
        urn = quote(str(value["identifier"]), safe="")
        payload = _json(
            broker.request(
                "GET",
                f"https://developer.api.autodesk.com/modelderivative/v2/designdata/{urn}/metadata",
                credential_reference=_credential(configuration),
            )
        )
        data = payload.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("metadata"), list):
            raise RuntimeError("CONNECTOR_AUTODESK_RESPONSE_INVALID")
        records = data["metadata"]
        return {
            "source": "autodesk-platform-services",
            "records": records,
        }

    return [
        ToolPlugin(
            tool_id="connector.bim.autodesk.metadata",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=IDENTIFIER_INPUT,
            output_schema=RECORDS_OUTPUT,
            handler=metadata,
            routing_metadata=CONNECTOR_ROUTING_METADATA["connector.bim.autodesk.metadata"],
        )
    ]


TRACKING_INPUT = _object_schema(
    ["carrier", "tracking_number"],
    {
        "carrier": {"type": "string", "pattern": "^[a-z0-9-]{1,40}$"},
        "tracking_number": {"type": "string", "pattern": "^[A-Za-z0-9-]{3,80}$"},
    },
)
TRACKING_OUTPUT = _object_schema(
    ["source", "status", "expected_delivery", "checkpoints"],
    {
        "source": {"const": "aftership"},
        "status": {"type": "string"},
        "expected_delivery": {"type": ["string", "null"]},
        "checkpoints": {"type": "array", "items": {"type": "object"}},
    },
)


def _aftership_tools(environment: ToolEnvironment) -> list[ToolPlugin]:
    """Create carrier tracking reads; tracking status alone never actuates pickup/release."""
    broker = _broker(environment)
    configuration = _configuration(environment)

    def tracking(value: dict[str, object]) -> dict[str, object]:
        """Keep the pinned provider contract and broker credential-header mapping together."""
        carrier = quote(str(value["carrier"]), safe="")
        tracking_number = quote(str(value["tracking_number"]), safe="")
        payload = _json(
            broker.request(
                "GET",
                f"https://api.aftership.com/tracking/2024-07/trackings/{carrier}/{tracking_number}",
                credential_reference=_credential(configuration),
                credential_header="as-api-key",
                credential_prefix="",
            )
        )
        data = payload.get("data")
        track = data.get("tracking") if isinstance(data, dict) else None
        if not isinstance(track, dict):
            raise RuntimeError("CONNECTOR_TRACKING_RESPONSE_INVALID")
        checkpoints = track.get("checkpoints")
        return {
            "source": "aftership",
            "status": track.get("tag", track.get("subtag", "unknown")),
            "expected_delivery": track.get("expected_delivery"),
            "checkpoints": checkpoints,
        }

    return [
        ToolPlugin(
            tool_id="connector.logistics.aftership.tracking",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=TRACKING_INPUT,
            output_schema=TRACKING_OUTPUT,
            handler=tracking,
            routing_metadata=CONNECTOR_ROUTING_METADATA["connector.logistics.aftership.tracking"],
        )
    ]


QUERY_INPUT = _object_schema(
    [],
    {
        "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
        "cursor": {"type": "string", "minLength": 1, "maxLength": 256},
    },
)


def _notion_tools(environment: ToolEnvironment) -> list[ToolPlugin]:
    """Capture the configured database; mission arguments cannot redirect account reads."""
    broker = _broker(environment)
    configuration = _configuration(environment)
    database_id = configuration.get("database_id")
    if not isinstance(database_id, str) or not database_id:
        raise RuntimeError("CONNECTOR_NOTION_DATABASE_ID_REQUIRED")

    def database_query(value: dict[str, object]) -> dict[str, object]:
        """Return a single page with its opaque continuation token, not an implicit full dataset."""
        body: dict[str, object] = {"page_size": int(value.get("page_size", 50))}
        if value.get("cursor"):
            body["start_cursor"] = str(value["cursor"])
        payload = _json(
            broker.request(
                "POST",
                f"https://api.notion.com/v1/databases/{quote(database_id, safe='')}/query",
                headers={
                    "Content-Type": "application/json",
                    # This endpoint uses the database contract. Updating only
                    # the date would mix it with the newer data-source API.
                    "Notion-Version": "2022-06-28",
                },
                body=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                credential_reference=_credential(configuration),
            )
        )
        more, cursor = payload.get("has_more"), payload.get("next_cursor")
        if (
            type(more) is not bool
            or (more and (not isinstance(cursor, str) or not cursor or len(cursor) > 256))
            or (not more and cursor is not None)
        ):
            raise RuntimeError("CONNECTOR_NOTION_PAGINATION_INVALID")
        return {
            "source": "notion",
            "records": payload.get("results"),
            "pagination": {"has_more": more, "next_cursor": cursor},
        }

    return [
        ToolPlugin(
            tool_id="connector.erp.notion.database-query",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=QUERY_INPUT,
            output_schema=RECORDS_OUTPUT,
            handler=database_query,
            routing_metadata=CONNECTOR_ROUTING_METADATA["connector.erp.notion.database-query"],
        )
    ]


INCIDENT_INPUT = _object_schema(
    [],
    {
        "statuses": {
            "type": "array",
            "items": {"type": "string", "enum": ["triggered", "acknowledged", "resolved"]},
            "maxItems": 3,
        },
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "offset": {"type": "integer", "minimum": 0, "maximum": 9999},
    },
)


def _pagerduty_tools(environment: ToolEnvironment) -> list[ToolPlugin]:
    """Bind read-only incident pages to the current credential scope."""
    broker = _broker(environment)
    configuration = _configuration(environment)

    def incidents(value: dict[str, object]) -> dict[str, object]:
        """Preserve classic pagination; never silently present a partial page as all incidents."""
        statuses = value.get("statuses", ["triggered", "acknowledged"])
        assert isinstance(statuses, list)
        query_items: list[tuple[str, object]] = [("limit", int(value.get("limit", 50)))]
        offset = value.get("offset", 0)
        if offset + query_items[0][1] > 10000:
            raise ValueError("CONNECTOR_PAGERDUTY_PAGINATION_LIMIT")
        query_items.append(("offset", offset))
        query_items.extend(("statuses[]", status) for status in statuses)
        payload = _json(
            broker.request(
                "GET",
                "https://api.pagerduty.com/incidents?" + urlencode(query_items),
                headers={"Accept": "application/vnd.pagerduty+json;version=2"},
                credential_reference=_credential(configuration),
                credential_prefix="Token token=",
            )
        )
        more, response_offset, limit = (
            payload.get("more"),
            payload.get("offset"),
            payload.get("limit"),
        )
        if (
            type(more) is not bool
            or type(response_offset) is not int
            or response_offset != offset
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise RuntimeError("CONNECTOR_PAGERDUTY_PAGINATION_INVALID")
        return {
            "source": "pagerduty",
            "records": payload.get("incidents"),
            "pagination": {
                "has_more": more,
                "offset": offset,
                "limit": limit,
                "next_offset": offset + limit if more else None,
            },
        }

    return [
        ToolPlugin(
            tool_id="connector.alerts.pagerduty.incidents",
            version="1.0.0",
            authority="read",
            input_type=None,
            output_type=None,
            input_schema=INCIDENT_INPUT,
            output_schema=RECORDS_OUTPUT,
            handler=incidents,
            routing_metadata=CONNECTOR_ROUTING_METADATA["connector.alerts.pagerduty.incidents"],
        )
    ]


def _definition(
    *,
    plugin_id: str,
    name: str,
    description: str,
    capability_id: str,
    host: str,
    slot_id: str,
    slot_label: str,
    order: int,
    tool_factory: Callable[[ToolEnvironment], list[ToolPlugin]],
    input_schema: dict[str, object],
    output_schema: dict[str, object],
    configuration_schema: dict[str, object] | None = None,
    credential: bool = False,
) -> PluginDefinition:
    """Declare an opt-in host-scoped tool with executable validation on direct handler calls."""
    # Catalog dictionaries remain public/mutable for presentation. Handlers must
    # validate against private snapshots that UI/plugin consumers cannot weaken.
    captured_input = copy_json(input_schema)
    captured_output = copy_json(output_schema)

    def guarded_factory(environment: ToolEnvironment) -> list[ToolPlugin]:
        """Wrap each handler and detach its presentation metadata on every registration."""
        return [
            _guarded_tool(tool, captured_input, captured_output)
            for tool in tool_factory(environment)
        ]

    permissions = ["mission.read", "network.external"]
    if credential:
        permissions.append("credential.reference")
    return PluginDefinition(
        manifest=PluginManifest(
            plugin_id=plugin_id,
            name=name,
            version="1.0.0",
            description=description,
            publisher="DroneDream",
            runtime=PluginRuntime(
                kind="builtin-python", entrypoint=f"{__name__}:plugin_definitions"
            ),
            resource_policy={"allowed_network_hosts": [host]},
            capabilities=[
                PluginCapability(
                    capability_id=capability_id,
                    kind="data-service",
                    name=name,
                    description=description,
                    authority="read",
                    input_schema=input_schema,
                    output_schema=output_schema,
                    metadata={
                        "live": True,
                        "brokered": True,
                        "provider_host": host,
                        **copy_json(CONNECTOR_ROUTING_METADATA[capability_id]),
                    },
                )
            ],
            permissions=permissions,
            default_enabled=False,
            removable=False,
            placement=PluginPlacement(
                category_id="connectors",
                category_label="外部连接器",
                slot_id=slot_id,
                slot_label=slot_label,
                activation_mode="multiple",
                scope="mission",
                failure_mode="isolate",
                swap_policy="next-mission",
                category_order=80,
                slot_order=order,
                plugin_order=10,
            ),
            configuration_schema=configuration_schema or {},
        ),
        tool_factory=guarded_factory,
    )


def _guarded_tool(tool: ToolPlugin, input_schema: dict, output_schema: dict) -> ToolPlugin:
    """Validate detached finite JSON before network access and before returning provider data."""
    incoming = Draft202012Validator(copy_json(input_schema))
    outgoing = Draft202012Validator(copy_json(output_schema))

    def call(value):
        """Direct Python callers receive the same boundary checks as registry callers."""
        arguments = copy_json(value)
        incoming.validate(arguments)
        result = copy_json(tool.handler(arguments))
        outgoing.validate(result)
        return result

    return replace(
        tool,
        handler=call,
        input_schema=copy_json(input_schema),
        output_schema=copy_json(output_schema),
        routing_metadata=copy_json(tool.routing_metadata),
    )


_CREDENTIAL_SCHEMA = _object_schema(
    ["credential_reference"],
    {
        "credential_reference": {
            "type": "string",
            "title": "凭证引用",
            "format": "dronedream-credential-reference",
            "pattern": "^[a-z0-9][a-z0-9-_]{1,79}$",
        }
    },
)


def plugin_definitions() -> list[PluginDefinition]:
    """Build the six explicit external read integrations; none is enabled by default."""
    return [
        _definition(
            plugin_id="connector.weather.open-meteo",
            name="Open-Meteo 天气预报",
            description="读取任务区域的当前天气估计与逐小时预报，不替代机载传感器。",
            capability_id="connector.weather.open-meteo.forecast",
            host="api.open-meteo.com",
            slot_id="connectors.weather",
            slot_label="天气数据",
            order=10,
            tool_factory=_weather_tools,
            input_schema=WEATHER_INPUT,
            output_schema=WEATHER_OUTPUT,
        ),
        _definition(
            plugin_id="connector.gis.overpass",
            name="OpenStreetMap GIS",
            description="从 Overpass API 读取受边界约束的建筑、道路与障碍物几何。",
            capability_id="connector.gis.overpass.features",
            host="overpass-api.de",
            slot_id="connectors.gis",
            slot_label="GIS 数据",
            order=20,
            tool_factory=_gis_tools,
            input_schema=GIS_INPUT,
            output_schema=GIS_OUTPUT,
        ),
        _definition(
            plugin_id="connector.bim.autodesk",
            name="Autodesk BIM",
            description="通过 Autodesk Platform Services 读取模型派生元数据。",
            capability_id="connector.bim.autodesk.metadata",
            host="developer.api.autodesk.com",
            slot_id="connectors.bim",
            slot_label="BIM 数据",
            order=30,
            tool_factory=_autodesk_tools,
            input_schema=IDENTIFIER_INPUT,
            output_schema=RECORDS_OUTPUT,
            configuration_schema=_CREDENTIAL_SCHEMA,
            credential=True,
        ),
        _definition(
            plugin_id="connector.logistics.aftership",
            name="AfterShip 物流",
            description="读取真实承运商跟踪状态、预计送达时间与检查点。",
            capability_id="connector.logistics.aftership.tracking",
            host="api.aftership.com",
            slot_id="connectors.logistics",
            slot_label="物流数据",
            order=40,
            tool_factory=_aftership_tools,
            input_schema=TRACKING_INPUT,
            output_schema=TRACKING_OUTPUT,
            configuration_schema=_CREDENTIAL_SCHEMA,
            credential=True,
        ),
        _definition(
            plugin_id="connector.erp.notion",
            name="Notion 任务数据库",
            description="读取指定 Notion 数据库中的工单或任务记录。",
            capability_id="connector.erp.notion.database-query",
            host="api.notion.com",
            slot_id="connectors.erp",
            slot_label="ERP 与工单",
            order=50,
            tool_factory=_notion_tools,
            input_schema=QUERY_INPUT,
            output_schema=RECORDS_OUTPUT,
            configuration_schema=_object_schema(
                ["credential_reference", "database_id"],
                {
                    **_CREDENTIAL_SCHEMA["properties"],
                    "database_id": {"type": "string", "minLength": 1, "maxLength": 128},
                },
            ),
            credential=True,
        ),
        _definition(
            plugin_id="connector.alerts.pagerduty",
            name="PagerDuty 告警",
            description="读取当前触发或已确认的真实运维事件。",
            capability_id="connector.alerts.pagerduty.incidents",
            host="api.pagerduty.com",
            slot_id="connectors.alerts",
            slot_label="事件与告警",
            order=60,
            tool_factory=_pagerduty_tools,
            input_schema=INCIDENT_INPUT,
            output_schema=RECORDS_OUTPUT,
            configuration_schema=_CREDENTIAL_SCHEMA,
            credential=True,
        ),
    ]
