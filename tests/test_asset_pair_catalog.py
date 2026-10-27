from __future__ import annotations

import json
from pathlib import Path

import pytest
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
    assert catalog["map_count"] == 6
    assert catalog["vehicle_count"] == 13
    assert catalog["pair_count"] == 78
    assert len({item["pair_id"] for item in pairs}) == 78
    assert catalog["counts"] == {
        "qualified_builtin": 6,
        "compatible_requires_qualification": 72,
        "incompatible": 0,
    }


def test_vtol_is_admitted_to_the_outdoor_campus_pending_qualification() -> None:
    catalog = asset_pair_catalog()
    vtol_pairs = [item for item in catalog["pairs"] if item["vehicle_class"] == "vtol"]
    compatible = [item for item in vtol_pairs if item["compatible"]]
    assert len(vtol_pairs) == 18
    assert len(compatible) == 18
    assert len({item["map_resource_id"] for item in compatible}) == 6
    assert all(item["status"] == "compatible_requires_qualification" for item in vtol_pairs)


def test_only_exact_builtin_x500_depth_pairs_claim_existing_qualification() -> None:
    catalog = asset_pair_catalog()
    qualified = [item for item in catalog["pairs"] if item["qualification_id"]]
    assert len(qualified) == 6
    assert {item["map_resource_id"]: item["qualified_map_asset_id"] for item in qualified} == {
        "dronedream-school-map": "dronedream.school-map.v1",
        "dronedream-meilahti-hospital-map": "dronedream.meilahti-hospital-map.v1",
        "dronedream-kalasatama-waterfront-map": "dronedream.kalasatama-waterfront-map.v1",
        "dronedream-vuosaari-harbour-map": "dronedream.vuosaari-harbour-map.v1",
        "dronedream-suomenlinna-map": "dronedream.suomenlinna-map.v1",
        "dronedream-uutela-coastal-forest-map": "dronedream.uutela-coastal-forest-map.v1",
    }
    assert {item["vehicle_resource_id"] for item in qualified} == {"px4-x500-depth"}
    assert all(item["qualified_vehicle_asset_id"] == "dronedream.my-drone.v1" for item in qualified)
    assert all(
        item["qualification_scope"] == "exact_builtin_normalized_package" for item in qualified
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
    expected = {item["resource_id"]: item["qualification_id"] for item in index["qualified_pairs"]}
    actual = {
        item["map_resource_id"]: item["qualification_id"]
        for item in asset_pair_catalog()["pairs"]
        if item["status"] == "qualified_builtin"
    }
    assert actual == expected


def _release_index() -> dict:
    path = (Path(__file__).parents[1] / "app" / "desktop" / "src-tauri" / "resources"
            / "default-assets" / "index.json")
    return json.loads(path.read_bytes())


def test_catalog_follows_selected_release_without_source_constants(tmp_path: Path) -> None:
    index = _release_index()
    index_path = tmp_path / "default-assets" / "index.json"
    index_path.parent.mkdir()
    for qualification_id in ("asset-qualification-" + "1" * 24, "asset-qualification-" + "2" * 24):
        index["default_qualification_id"] = qualification_id
        index["qualified_pair"]["qualification_id"] = qualification_id
        index["qualified_pairs"][0]["qualification_id"] = qualification_id
        index_path.write_text(json.dumps(index), encoding="utf-8")
        pair = get_asset_pair("dronedream-school-map", "px4-x500-depth", resource_root=tmp_path)
        assert pair["qualification_id"] == qualification_id
        assert pair["status"] == "qualified_builtin"


@pytest.mark.parametrize("damage", ["missing", "json", "default", "duplicate", "vehicle", "hash"])
def test_catalog_does_not_claim_qualification_from_damaged_index(
    tmp_path: Path, damage: str,
) -> None:
    index = _release_index()
    if damage == "default":
        index["default_qualification_id"] = "asset-qualification-" + "3" * 24
    elif damage == "duplicate":
        index["qualified_pairs"].append(index["qualified_pairs"][0])
    elif damage in {"vehicle", "hash"}:
        for pair in (index["qualified_pair"], index["qualified_pairs"][0]):
            package = next(item for item in pair["packages"] if item["kind"] == "vehicle")
            package["asset_id" if damage == "vehicle" else "content_sha256"] = "different"
    if damage != "missing":
        path = tmp_path / "default-assets" / "index.json"
        path.parent.mkdir()
        path.write_text("{" if damage == "json" else json.dumps(index), encoding="utf-8")
    pair = get_asset_pair("dronedream-school-map", "px4-x500-depth", resource_root=tmp_path)
    assert pair["qualification_id"] is None
    assert pair["status"] == "compatible_requires_qualification"
    assert pair["required_gates"]


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
    assert response.json()["pair_count"] == 78
