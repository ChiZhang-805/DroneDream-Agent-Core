from __future__ import annotations

import re

from fastapi.testclient import TestClient

from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore
from dronedream_agent_app.vehicle_resource_catalog import (
    get_vehicle_resource,
    vehicle_resource_catalog,
)


def test_reviewed_vehicle_catalog_is_pinned_preparsed_and_unqualified() -> None:
    catalog = vehicle_resource_catalog()
    resources = catalog["resources"]
    assert catalog["schema_version"] == "dronedream.vehicle-resource-catalog.v1"
    assert len(resources) == 13
    assert len({item["resource_id"] for item in resources}) == len(resources)
    for resource in resources:
        assert resource["default_resource"] is True
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", resource["resource_id"])
        assert resource["review"]["status"] == "approved"
        assert resource["license"]["spdx_id"] == "BSD-3-Clause"
        assert resource["source"]["location"] == (
            "https://github.com/PX4/PX4-gazebo-models.git"
        )
        assert resource["source"]["git_ref"] == (
            "5577035667afb4b63fe1f966fb1a58bbb05d905b"
        )
        assert resource["source"]["subpath"].startswith("models/")
        assert resource["source"]["expected_kind"] == "vehicle"
        assert len(resource["source"]["expected_sha256"]) == 64
        assert resource["analysis"]["status"] == "preparsed"
        assert resource["readiness"]["simulation"] == "dependency_resolution_required"
        assert resource["readiness"]["flight"] == "unqualified"


def test_vehicle_catalog_returns_detached_values_and_rejects_unknown_id() -> None:
    first = get_vehicle_resource("px4-x500-depth")
    first["analysis"]["dependency_models"].clear()
    assert get_vehicle_resource("px4-x500-depth")["analysis"]["dependency_models"]
    try:
        get_vehicle_resource("unknown")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown catalog ids must fail closed")


def test_vehicle_resource_catalog_api_is_loopback_token_protected(tmp_path) -> None:
    token = "v" * 64
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))
    assert client.get("/v1/vehicle-resource-catalog").status_code == 401
    response = client.get(
        "/v1/vehicle-resource-catalog",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["resources"][0]["source"]["px4_sitl_model"] == "x500_depth"
