from __future__ import annotations

import re
import zipfile

from fastapi.testclient import TestClient

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.map_resource_catalog import get_map_resource, map_resource_catalog
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.asset_source_adapters import detect_asset_source, source_adapter_catalog


def test_reviewed_catalog_is_pinned_preparsed_and_not_flight_qualified() -> None:
    catalog = map_resource_catalog()
    resources = catalog["resources"]
    assert catalog["schema_version"] == "dronedream.map-resource-catalog.v1"
    assert len(resources) == 7
    assert len({item["resource_id"] for item in resources}) == len(resources)
    for resource in resources:
        assert resource["default_resource"] is True
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", resource["resource_id"])
        assert resource["review"]["status"] == "approved"
        assert resource["license"]["spdx_id"] == "Apache-2.0"
        assert resource["source"]["source_type"] == "git"
        assert resource["source"]["location"] == "https://github.com/open-rmf/rmf_demos.git"
        assert resource["source"]["git_ref"] == "7851a5792d19a037833292a3e2a823b0f9e0c111"
        assert resource["source"]["subpath"].startswith("rmf_demos_maps/maps/")
        assert resource["source"]["source_format"] == "rmf-building-map-package"
        assert len(resource["source"]["expected_sha256"]) == 64
        assert resource["analysis"]["status"] == "preparsed"
        assert resource["readiness"]["simulation"] == "builtin_conversion_available"
        assert resource["readiness"]["flight"] == "unqualified"


def test_catalog_returns_detached_values_and_rejects_unknown_id() -> None:
    first = get_map_resource("open-rmf-clinic")
    first["analysis"]["door_count"] = -1
    assert get_map_resource("open-rmf-clinic")["analysis"]["door_count"] == 132
    try:
        get_map_resource("unknown")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown catalog ids must fail closed")


def test_rmf_source_is_detected_without_executing_project_code(tmp_path) -> None:
    source = tmp_path / "clinic.building.yaml"
    source.write_text(
        "name: clinic\ncoordinate_system: cartesian_meters\nlevels:\n"
        "  L1:\n    elevation: 0\n    vertices:\n"
        "      - [0, 0, 0, origin]\n      - [10, 0, 0, east]\n"
        "    walls: [[0, 1, {}]]\n    floors: []\n    lanes: [[0, 1, {}]]\n",
        encoding="utf-8",
    )
    detection = detect_asset_source(source, "rmf-building-map")
    assert detection.adapter_id == "open-rmf.building-map"
    assert detection.asset_kind == "map"
    assert detection.can_normalize_locally is True
    assert detection.required_inputs == []

    archive = tmp_path / "clinic.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("clinic/clinic.building.yaml", source.read_bytes())
    packaged = detect_asset_source(archive, "rmf-building-map-package")
    assert packaged.adapter_id == "open-rmf.building-map"
    assert packaged.confidence == "exact"

    adapters = {entry["adapter_id"]: entry for entry in source_adapter_catalog()}
    assert adapters["open-rmf.building-map"]["availability"] == "builtin"


def test_rmf_compound_suffix_survives_quarantine_and_process(tmp_path) -> None:
    downloaded = tmp_path / "downloaded.yaml"
    downloaded.write_text(
        "name: clinic\ncoordinate_system: cartesian_meters\nlevels:\n"
        "  L1:\n    elevation: 0\n    vertices:\n"
        "      - [0, 0, 0, origin]\n      - [10, 0, 0, east]\n"
        "    walls: [[0, 1, {}]]\n    floors: []\n    lanes: [[0, 1, {}]]\n",
        encoding="utf-8",
    )
    service = AssetImportService(AppStore(tmp_path / "store"))
    created = service.create(
        source=downloaded,
        source_name="clinic.building.yaml",
        source_format="rmf-building-map",
        expected_kind="map",
    )
    processed = service.process(created["job_id"])
    assert processed["state"] == "needs_input"
    assert processed["required_inputs"] == ["qualification_evidence"]
    assert processed["source_adapter_id"] == "open-rmf.building-map"
    assert processed["detected_source_format"] == "rmf-building-map"


def test_map_resource_catalog_api_is_loopback_token_protected(tmp_path) -> None:
    token = "c" * 64
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))
    assert client.get("/v1/map-resource-catalog").status_code == 401
    response = client.get(
        "/v1/map-resource-catalog",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["resources"][0]["source"]["source_type"] == "git"
