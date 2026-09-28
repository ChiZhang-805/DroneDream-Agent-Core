from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from dronedream_agent_app.asset_pair_catalog import (
    asset_pair_catalog,
    get_asset_pair,
)
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore


def test_catalog_covers_every_builtin_map_aircraft_combination_once() -> None:
    catalog = asset_pair_catalog()
    pairs = catalog["pairs"]
    assert catalog["schema_version"] == "dronedream.asset-pair-catalog.v1"
    assert catalog["map_count"] == 8
    assert catalog["vehicle_count"] == 13
    assert catalog["pair_count"] == 104
    assert len({item["pair_id"] for item in pairs}) == 104
    assert catalog["counts"] == {
        "qualified_builtin": 8,
        "compatible_requires_qualification": 78,
        "incompatible": 18,
    }


def test_vtol_is_only_admitted_to_the_two_outdoor_maps() -> None:
    catalog = asset_pair_catalog()
    vtol_pairs = [item for item in catalog["pairs"] if item["vehicle_class"] == "vtol"]
    compatible = [item for item in vtol_pairs if item["compatible"]]
    assert len(vtol_pairs) == 24
    assert len(compatible) == 6
    assert {item["map_resource_id"] for item in compatible} == {
        "open-rmf-battle-royale",
        "open-rmf-campus",
    }
    assert all(item["status"] == "incompatible" for item in vtol_pairs if not item["compatible"])


def test_only_exact_builtin_x500_depth_pairs_claim_existing_qualification() -> None:
    catalog = asset_pair_catalog()
    qualified = [item for item in catalog["pairs"] if item["qualification_id"]]
    assert len(qualified) == 8
    assert {item["vehicle_resource_id"] for item in qualified} == {"px4-x500-depth"}
    assert all(item["qualified_vehicle_asset_id"] == "dronedream.my-drone.v1" for item in qualified)
    assert all(
        item["qualification_scope"] == "exact_builtin_normalized_package"
        for item in qualified
    )
    assert all(not item["required_gates"] for item in qualified)


def test_qualified_pair_ids_match_the_bundled_default_asset_index() -> None:
    index_path = (
        Path(__file__).parents[1]
        / "app"
        / "desktop"
        / "src-tauri"
        / "resources"
        / "default-assets"
        / "index.json"
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = {
        item["resource_id"]: item["qualification_id"]
        for item in index["qualified_pairs"]
    }
    actual = {
        item["map_resource_id"]: item["qualification_id"]
        for item in asset_pair_catalog()["pairs"]
        if item["status"] == "qualified_builtin"
    }
    assert actual == expected


def test_catalog_values_are_detached_and_unknown_pairs_fail_closed() -> None:
    first = get_asset_pair("dronedream-school-map", "px4-x500-depth")
    first["reasons"].clear()
    assert get_asset_pair("dronedream-school-map", "px4-x500-depth")["reasons"]
    try:
        get_asset_pair("unknown", "px4-x500-depth")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown pair ids must fail closed")


def test_asset_pair_catalog_api_is_loopback_token_protected(tmp_path) -> None:
    token = "p" * 64
    client = TestClient(create_app(store=AppStore(tmp_path), token=token))
    assert client.get("/v1/asset-pair-catalog").status_code == 401
    response = client.get(
        "/v1/asset-pair-catalog",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["pair_count"] == 104
