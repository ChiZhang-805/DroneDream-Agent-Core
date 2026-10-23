"""Live map pixels and telemetry must use the simulator's immutable coordinate frame."""

import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

import dronedream_agent_app.live_map as maps
from dronedream_agent_app.runtime_manager import ActiveRun, RuntimeBridgeError, RuntimeManager


def _map_fixture(tmp_path, monkeypatch, *, explicit=False):
    Image.new("RGB", (200, 100)).save(tmp_path / "ortho.png")
    (tmp_path / "ground.obj").write_text(
        "v -250 -250 0\nv 250 -250 0\nv 250 250 0\nv -250 250 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3 4/4\n"
    )
    (tmp_path / "world.sdf").write_text(
        "<sdf><world><model><link><visual><geometry><mesh><uri>ground.obj</uri>"
        "</mesh></geometry><material><pbr><metal><albedo_map>ortho.png</albedo_map>"
        "</metal></pbr></material></visual></link></model></world></sdf>"
    )
    semantic = {"coordinate_frame": "ENU"}
    if explicit:
        semantic["live_map"] = {
            "schema_version": "dronedream.live-map-georeference.v1",
            "coordinate_frame": "ENU",
            "image_path": "ortho.png",
            "width_px": 200,
            "height_px": 100,
            "world_to_image": [0.4, 0, 100, 0, -0.2, 50],
        }
    (tmp_path / "semantic.json").write_text(json.dumps(semantic))
    selection = SimpleNamespace(
        root=tmp_path,
        world_sdf=tmp_path / "world.sdf",
        semantic=tmp_path / "semantic.json",
        asset_id="test-map",
        content_sha256="a" * 64,
    )
    asset_ir = SimpleNamespace(
        files=[
            SimpleNamespace(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in tmp_path.iterdir()
        ]
    )
    monkeypatch.setattr(maps, "resolve_versioned_map", lambda record: selection)
    monkeypatch.setattr(maps, "AssetIR", SimpleNamespace(model_validate=lambda _: asset_ir))
    return {"asset_ir": {}}, semantic, asset_ir


@pytest.mark.parametrize("explicit", [True, False])
def test_map_registration_comes_from_declared_pixels_and_geometry(tmp_path, monkeypatch, explicit):
    record, _, _ = _map_fixture(tmp_path, monkeypatch, explicit=explicit)
    result = maps.resolve_live_map(record)
    assert result.metadata["world_to_image"] == [0.4, 0, 100, 0, -0.2, 50]
    assert result.metadata["map_content_sha256"] == "a" * 64
    assert result.metadata["width_px"] == 200
    assert result.metadata["height_px"] == 100
    assert result.image == (tmp_path / "ortho.png").read_bytes()


def test_map_image_requires_the_selected_package_hash(tmp_path, monkeypatch):
    record, _, _ = _map_fixture(tmp_path, monkeypatch)
    Image.new("RGB", (200, 100), "red").save(tmp_path / "ortho.png")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        maps.resolve_live_map(record)


def test_map_image_dimensions_cannot_drift_from_registration(tmp_path, monkeypatch):
    record, semantic, _ = _map_fixture(tmp_path, monkeypatch, explicit=True)
    semantic["live_map"]["width_px"] = 400
    (tmp_path / "semantic.json").write_text(json.dumps(semantic))
    with pytest.raises(ValueError, match="DIMENSIONS_MISMATCH"):
        maps.resolve_live_map(record)


def test_full_campus_geometry_uses_asset_budget_without_dropping_live_pixels(tmp_path, monkeypatch):
    record, semantic, _ = _map_fixture(tmp_path, monkeypatch, explicit=True)
    # Real canopy/footprint maps exceed the small plugin RPC envelope. The Live
    # reader must use the same map-file reader as planning, not reject that map.
    semantic["traversability"] = {
        "exclusions": [
            {"id": f"tree-{i}", "kind": "tree-canopy",
             "polygon_xy_m": [[float(j), float(j + 1)] for j in range(100)]}
            for i in range(1000)
        ]
    }
    (tmp_path / "semantic.json").write_text(json.dumps(semantic))
    result = maps.resolve_live_map(record)
    assert result.image == (tmp_path / "ortho.png").read_bytes()
    assert result.metadata["world_to_image"] == [0.4, 0, 100, 0, -0.2, 50]


@pytest.mark.parametrize("edit", ["rotation", "uv"])
def test_legacy_fallback_rejects_unverified_mesh_registration(tmp_path, monkeypatch, edit):
    record, _, asset_ir = _map_fixture(tmp_path, monkeypatch)
    if edit == "rotation":
        path = tmp_path / "world.sdf"
        path.write_text(path.read_text().replace("<model>", "<model><pose>0 0 0 0 0 1</pose>"))
    else:
        path = tmp_path / "ground.obj"
        path.write_text(path.read_text().replace("f 1/1 2/2 3/3 4/4", "f 1/2 2/1 3/3 4/4"))
        next(item for item in asset_ir.files if item.path == path.name).sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError, match="REGISTRATION_REQUIRED"):
        maps.resolve_live_map(record)


def test_live_map_and_telemetry_stay_bound_to_the_running_map(tmp_path):
    manager = RuntimeManager.__new__(RuntimeManager)
    manager._lock = threading.RLock()
    old_map = maps.LiveMap(
        {"map_id": "running", "map_content_sha256": "a" * 64}, b"pixels", "image/png"
    )
    manager._live_maps = {("running", "a" * 64): old_map}
    manager._active_runs = {
        "thread": ActiveRun(
            "run",
            tmp_path,
            SimpleNamespace(poll=lambda: None),
            "now",
            map_asset_id="running",
            map_content_sha256="a" * 64,
        )
    }
    manager.store = SimpleNamespace(
        get_thread=lambda _: {
            "selected_map_id": "new-selection",
            "selected_map_content_sha256": "b" * 64,
        }
    )
    assert manager.live_map("thread") is old_map
    with pytest.raises(RuntimeBridgeError, match="VERSION_MISMATCH"):
        manager.live_map("thread", expected_hash="b" * 64)
    (tmp_path / "live-telemetry.json").write_text(json.dumps({"east_m": 10, "north_m": 20}))
    assert manager.live_telemetry("thread")["map_id"] == "running"
    assert manager.live_telemetry("thread")["map_content_sha256"] == "a" * 64


@pytest.mark.usefixtures("isolated_server_credentials")
def test_live_map_endpoints_require_owner_and_bind_image_version(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dronedream_agent_app.identity import VerifiedIdentity
    from dronedream_agent_app.server import create_app
    from dronedream_agent_app.storage import AppStore
    from dronedream_agent_core.model_harness.memory import AccountMemoryStore, MemoryOwnerScope

    identity = VerifiedIdentity(
        owner_account_id="map-owner",
        tenant_id=None,
        organization_id=None,
        issuer="https://example.supabase.co/auth/v1",
        expires_at=2_000_000_000,
    )
    store = AppStore(tmp_path / "app")
    thread_id = str(store.create_thread("Map", "gpt-5.4", "en-US")["thread_id"])
    memory_path = tmp_path / "memory.sqlite3"
    AccountMemoryStore(memory_path).bind_thread(
        MemoryOwnerScope(
            owner_account_id="map-owner",
            tenant_id=None,
            organization_id=None,
            source_edition="autonomy",
        ),
        thread_id,
    )
    calls = []

    def live_map(self, thread, expected_hash=None):
        calls.append((thread, expected_hash))
        if expected_hash is not None and expected_hash != "a" * 64:
            raise RuntimeBridgeError("LIVE_MAP_VERSION_MISMATCH")
        return maps.LiveMap({"map_content_sha256": "a" * 64}, b"test-image", "image/png")

    monkeypatch.setattr(RuntimeManager, "live_map", live_map)
    client = TestClient(
        create_app(
            store=store,
            token="t" * 64,
            account_memory_path=memory_path,
            identity_verifier=lambda supplied: identity,
        )
    )
    base = f"/v1/threads/{thread_id}"
    assert client.get(base + "/live-map").status_code == 401
    headers = {"Authorization": f"Bearer {'t' * 64}", "X-DroneDream-Identity-Token": "owner"}
    response = client.get(base + "/live-map", headers=headers)
    assert response.json() == {"map_content_sha256": "a" * 64}
    response = client.get(base + "/live-map-image?map_content_sha256=" + "a" * 64, headers=headers)
    assert response.content == b"test-image"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    response = client.get(base + "/live-map-image?map_content_sha256=" + "b" * 64, headers=headers)
    assert response.status_code == 409
    assert calls[-1] == (thread_id, "b" * 64)
