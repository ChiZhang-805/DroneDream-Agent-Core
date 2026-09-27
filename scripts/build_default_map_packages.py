"""Acquire, verify and convert every reviewed default map into an unqualified DDPkg."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from dronedream_agent_app.asset_remote_sources import RemoteAssetSourceService
from dronedream_agent_app.map_resource_catalog import map_resource_catalog
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.asset_source_adapters import detect_asset_source, normalize_asset_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source-package-root",
        type=Path,
        help=(
            "Optional previously verified DDPkg directory. Its content-pinned "
            "source/source.zip is reused when the remote is temporarily unavailable."
        ),
    )
    return parser


def _normalize_source(path: Path, source: dict[str, object], destination: Path) -> None:
    detection = detect_asset_source(path, str(source["source_format"]))
    normalize_asset_source(
        path,
        detection,
        destination,
        expected_kind="map",
        embed_source_snapshot=False,
    )


def _verified_cached_source(
    *, package: Path, expected_sha256: str, destination: Path
) -> Path:
    inspected = inspect_ddpkg(package)
    if inspected.manifest.asset_kind != "map":
        raise ValueError("cached source package must be a map")
    source_entries = [
        entry
        for entry in inspected.manifest.files
        if entry.role == "source" and entry.path.casefold().endswith(".zip")
    ]
    if len(source_entries) != 1:
        raise ValueError("cached package must contain one embedded source archive")
    with ZipFile(package) as bundle:
        source = bundle.read(source_entries[0].path)
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("cached source archive does not match the reviewed catalog")
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(source)
    return destination


def main() -> int:
    args = _parser().parse_args()
    args.staging_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    service = RemoteAssetSourceService(args.staging_root)
    records: list[dict[str, object]] = []
    for resource in map_resource_catalog()["resources"]:
        source = resource["source"]
        destination = args.output_root / f"{resource['resource_id']}.ddpkg"
        if destination.exists():
            raise FileExistsError(destination)
        if args.source_package_root is not None:
            path = _verified_cached_source(
                package=args.source_package_root / destination.name,
                expected_sha256=source["expected_sha256"],
                destination=args.staging_root / f"{resource['resource_id']}.source.zip",
            )
            _normalize_source(path, source, destination)
        else:
            with service.acquire(
                source_type=source["source_type"],
                location=source["location"],
                expected_sha256=source["expected_sha256"],
                git_ref=source.get("git_ref"),
                subpath=source.get("subpath"),
            ) as (path, _display_name):
                _normalize_source(path, source, destination)
        inspected = inspect_ddpkg(destination)
        records.append(
            {
                "resource_id": resource["resource_id"],
                "filename": destination.name,
                "package_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "content_sha256": inspected.manifest.content_sha256,
                "asset_id": inspected.asset_ir.asset_id,
                "world_entrypoint": inspected.asset_ir.simulation_targets[0].entrypoint,
                "flight_qualified": False,
                "required_next_gate": "map-aircraft-pair-qualification",
            }
        )
    index = {
        "schema_version": "dronedream.default-map-package-index.v1",
        "source_catalog_revision": map_resource_catalog()["catalog_revision"],
        "package_count": len(records),
        "packages": records,
    }
    (args.output_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
