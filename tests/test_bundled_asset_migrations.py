"""Release pin migration fixtures are synthetic and do not certify real flight."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_qualification_boundaries import _prepared_pair

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.server import create_app
from dronedream_agent_app.storage import AppStore, AssetImportError
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.asset_pair_qualification import bind_qualification_receipt


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _release(tmp_path):
    """Write a genuinely parsed tiny package pair plus two explicit predecessor IDs."""
    source, _, _, receipt = _prepared_pair(tmp_path)
    resources = tmp_path / "resources"
    bundle = resources / "default-assets"
    bundle.mkdir(parents=True)
    packages = []
    for kind, archive in (("map", source), ("vehicle", tmp_path / "vehicle.ddpkg")):
        target = bind_qualification_receipt(
            source_archive=archive, destination_archive=bundle / f"{kind}.ddpkg", receipt=receipt
        )
        manifest = inspect_ddpkg(target).manifest
        packages.append(
            {
                "kind": kind,
                "file": target.name,
                "asset_id": manifest.asset_id,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "content_sha256": manifest.content_sha256,
                "source_content_sha256": getattr(receipt, f"{kind}_content_sha256"),
            }
        )
    evidence = next(
        f for f in manifest.files if f.path.endswith(f"/{receipt.qualification_id}.json")
    )
    pair = {
        "schema_version": "dronedream.bundled-qualified-pair.v1",
        "qualification_id": receipt.qualification_id,
        "receipt_sha256": evidence.sha256,
        "packages": packages,
    }
    previous = [
        {
            "qualification_id": "asset-qualification-" + str(n) * 24,
            **{
                f"{kind}{suffix}_content_sha256": _digest(f"{n}:{kind}:{suffix}")
                for kind in ("map", "vehicle")
                for suffix in ("", "_source")
            },
        }
        for n in (1, 2)
    ]
    index = {
        "schema_version": "dronedream.bundled-assets.v2",
        "qualified_pair": pair,
        "superseded_qualified_pairs": previous,
    }
    (bundle / "index.json").write_text(json.dumps(index), encoding="utf-8")
    official = resources / "official-plugins"
    official.mkdir()
    (official / "index.json").write_text(
        json.dumps(
            {
                "schema_version": "dronedream.official-plugin-index.v1",
                "plugins": [],
            }
        ),
        encoding="utf-8",
    )
    return resources, index


def test_migrations_reappear_after_pruning_and_restarting_without_touching_custom_pins(tmp_path):
    resources, index = _release(tmp_path)
    bundle = resources / "default-assets"
    packages = {p["kind"]: p for p in index["qualified_pair"]["packages"]}
    store = AppStore(tmp_path / "store")
    old = index["superseded_qualified_pairs"][0]
    historical = store.create_thread("old browser task", "openai/gpt-4.1", "en-US")
    custom = store.create_thread("custom version", "openai/gpt-4.1", "en-US")
    for thread, digest in ((historical, old["map_content_sha256"]), (custom, _digest("custom"))):
        store.patch_thread(
            thread["thread_id"],
            {
                "selected_map_id": packages["map"]["asset_id"],
                "selected_map_content_sha256": digest,
            },
        )
    service = AssetImportService(store)
    service.seed_bundled_sources(bundle)
    migrations = service.bundled_asset_migrations()
    assert len(migrations) == 10  # two historical source/results per kind, plus current source.
    assert {m["index_sha256"] for m in migrations} == {
        hashlib.sha256((bundle / "index.json").read_bytes()).hexdigest()
    }
    assert {m["receipt_sha256"] for m in migrations} == {index["qualified_pair"]["receipt_sha256"]}
    for kind in ("map", "vehicle"):
        expected = {packages[kind]["source_content_sha256"]}
        for predecessor in index["superseded_qualified_pairs"]:
            expected.update(
                (
                    predecessor[f"{kind}_content_sha256"],
                    predecessor[f"{kind}_source_content_sha256"],
                )
            )
        actual = [m for m in migrations if m["kind"] == kind]
        assert {m["from_content_sha256"] for m in actual} == expected
        assert {m["to_content_sha256"] for m in actual} == {packages[kind]["content_sha256"]}
        assert all(m["asset_id"] == packages[kind]["asset_id"] for m in actual)
    assert (
        store.get_thread(historical["thread_id"])["selected_map_content_sha256"]
        == (packages["map"]["content_sha256"])
    )
    assert store.get_thread(custom["thread_id"])["selected_map_content_sha256"] == _digest("custom")
    assert all(m["from_content_sha256"] != _digest("custom") for m in migrations)
    with pytest.raises(KeyError):
        store.get_asset_version(packages["map"]["asset_id"], old["map_content_sha256"])
    store.close()
    reopened = AppStore(tmp_path / "store")
    restarted = AssetImportService(reopened)
    restarted.seed_bundled_sources(bundle)
    assert restarted.bundled_asset_migrations() == migrations
    # Callers receive copies, not mutable authority for future bootstrap responses.
    migrations[0]["to_content_sha256"] = _digest("unknown")
    assert restarted.bundled_asset_migrations()[0] != migrations[0]
    reopened.close()


@pytest.mark.parametrize("invalidated", ["missing-target", "changed-receipt", "not-qualified"])
def test_migrations_require_current_installed_qualified_pair(tmp_path, invalidated):
    resources, index = _release(tmp_path)
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    service.seed_bundled_sources(resources / "default-assets")
    assert service.bundled_asset_migrations()
    package = index["qualified_pair"]["packages"][0]
    version = store.get_asset_version(package["asset_id"], package["content_sha256"])
    if invalidated == "changed-receipt":
        path = version["manifest"]["qualification"]["evidence_paths"][0]
        (Path(version["bundle_root"]) / path).write_bytes(b"not the qualified receipt")
    else:
        query = (
            "DELETE FROM asset_versions WHERE asset_id=? AND content_sha256=?"
            if invalidated == "missing-target"
            else (
                "UPDATE asset_versions SET maturity='visual_only' "
                "WHERE asset_id=? AND content_sha256=?"
            )
        )
        store._connection.execute(query, (package["asset_id"], package["content_sha256"]))
        store._connection.commit()
    assert service.bundled_asset_migrations() == []
    store.close()


def test_invalid_index_cannot_reuse_previous_migration_authority(tmp_path):
    resources, _ = _release(tmp_path)
    bundle = resources / "default-assets"
    store = AppStore(tmp_path / "store")
    service = AssetImportService(store)
    service.seed_bundled_sources(bundle)
    assert service.bundled_asset_migrations()
    (bundle / "index.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AssetImportError, match="BUNDLED_ASSET_INDEX_INVALID"):
        service.seed_bundled_sources(bundle)
    assert service.bundled_asset_migrations() == []
    store.close()


def test_bootstrap_publishes_migrations_from_verified_release_only(tmp_path):
    resources, index = _release(tmp_path)
    store = AppStore(tmp_path / "store")
    token = "b" * 64
    client = TestClient(create_app(store=store, token=token, resource_root=resources))
    response = client.get("/v1/bootstrap", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    records = response.json()["bundled_asset_migrations"]
    assert len(records) == 10
    assert all(m["schema_version"] == "dronedream.bundled-asset-migration.v1" for m in records)
    assert all(
        m["qualification_id"] == index["qualified_pair"]["qualification_id"] for m in records
    )
    assert all(m["from_content_sha256"] != m["to_content_sha256"] for m in records)
    store.close()
