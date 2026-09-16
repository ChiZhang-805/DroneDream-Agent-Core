"""Bind one development mission input to current qualified asset identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dronedream_agent_app.asset_import_service import AssetImportService
from dronedream_agent_app.asset_runtime_resolver import (
    rebind_development_mission_input_to_map,
    rebind_development_mission_input_to_qualified_vehicle,
    resolve_versioned_map,
    resolve_versioned_vehicle,
)
from dronedream_agent_app.storage import AppStore


def _current_map_record(
    store: AppStore,
    default_assets_root: Path,
) -> dict[str, object]:
    try:
        index = json.loads((default_assets_root / "index.json").read_text(encoding="utf-8"))
        pair = index["qualified_pair"]
        packages = pair["packages"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("CURRENT_BUNDLED_ASSET_INDEX_INVALID") from error
    if index.get("schema_version") != "dronedream.bundled-assets.v2" or not isinstance(
        packages, list
    ):
        raise ValueError("CURRENT_BUNDLED_ASSET_INDEX_INVALID")
    maps = [entry for entry in packages if isinstance(entry, dict) and entry.get("kind") == "map"]
    if len(maps) != 1:
        raise ValueError("CURRENT_BUNDLED_MAP_ENTRY_INVALID")
    asset_id = maps[0].get("asset_id")
    content_sha256 = maps[0].get("content_sha256")
    if not isinstance(asset_id, str) or not isinstance(content_sha256, str):
        raise ValueError("CURRENT_BUNDLED_MAP_ENTRY_INVALID")
    return store.get_asset_version(asset_id, content_sha256)


def _current_vehicle_record(
    store: AppStore,
    default_assets_root: Path,
) -> dict[str, object]:
    try:
        index = json.loads((default_assets_root / "index.json").read_text(encoding="utf-8"))
        pair = index["qualified_pair"]
        packages = pair["packages"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("CURRENT_BUNDLED_ASSET_INDEX_INVALID") from error
    if index.get("schema_version") != "dronedream.bundled-assets.v2" or not isinstance(
        packages, list
    ):
        raise ValueError("CURRENT_BUNDLED_ASSET_INDEX_INVALID")
    vehicles = [
        entry
        for entry in packages
        if isinstance(entry, dict) and entry.get("kind") == "vehicle"
    ]
    if len(vehicles) != 1:
        raise ValueError("CURRENT_BUNDLED_VEHICLE_ENTRY_INVALID")
    asset_id = vehicles[0].get("asset_id")
    content_sha256 = vehicles[0].get("content_sha256")
    if not isinstance(asset_id, str) or not isinstance(content_sha256, str):
        raise ValueError("CURRENT_BUNDLED_VEHICLE_ENTRY_INVALID")
    return store.get_asset_version(asset_id, content_sha256)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-development-input", type=Path, required=True)
    parser.add_argument("--default-assets-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--bind-qualified-vehicle",
        action="store_true",
        help=(
            "Replace the development vehicle identity with the selected qualified "
            "vehicle only when SDF and controller bytes are identical."
        ),
    )
    args = parser.parse_args()

    output_parent = args.output_root.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="current-asset-resolution-", dir=output_parent) as temporary:
        store = AppStore(Path(temporary) / "asset-store.sqlite3")
        try:
            AssetImportService(store).seed_bundled_sources(args.default_assets_root)
            if args.bind_qualified_vehicle:
                selected_vehicle = resolve_versioned_vehicle(
                    _current_vehicle_record(store, args.default_assets_root)
                )
                result = rebind_development_mission_input_to_qualified_vehicle(
                    args.source_development_input,
                    selected_vehicle,
                    args.output_root,
                )
            else:
                selected_map = resolve_versioned_map(
                    _current_map_record(store, args.default_assets_root)
                )
                result = rebind_development_mission_input_to_map(
                    args.source_development_input,
                    selected_map,
                    args.output_root,
                )
        finally:
            store.close()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
