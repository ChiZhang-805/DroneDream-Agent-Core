"""Verify every reviewed default-map source through the production downloader.

This audit intentionally stops at source acquisition and format detection.  It
does not convert a map, launch Gazebo, or promote a resource to flight-ready.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dronedream_agent_app.asset_remote_sources import RemoteAssetSourceService
from dronedream_agent_app.map_resource_catalog import map_resource_catalog
from dronedream_agent_core.asset_source_adapters import detect_asset_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-root",
        type=Path,
        required=True,
        help="Existing project-owned directory used for temporary verified downloads.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    service = RemoteAssetSourceService(args.staging_root)
    results: list[dict[str, object]] = []

    for resource in map_resource_catalog()["resources"]:
        source = resource["source"]
        with service.acquire(
            source_type=source["source_type"],
            location=source["location"],
            expected_sha256=source["expected_sha256"],
            git_ref=source.get("git_ref"),
            subpath=source.get("subpath"),
        ) as (path, _display_name):
            detection = detect_asset_source(path, source["source_format"])
            if detection.adapter_id != "open-rmf.building-map":
                raise RuntimeError(
                    f"{resource['resource_id']}: unexpected adapter {detection.adapter_id}"
                )
            if detection.asset_kind != "map" or not detection.can_normalize_locally:
                raise RuntimeError(f"{resource['resource_id']}: unsafe readiness classification")
            results.append(
                {
                    "resource_id": resource["resource_id"],
                    "source_format": detection.source_format,
                    "adapter_id": detection.adapter_id,
                    "confidence": detection.confidence,
                    "simulation_readiness": resource["readiness"]["simulation"],
                    "flight_readiness": resource["readiness"]["flight"],
                }
            )

    print(
        json.dumps(
            {
                "schema_version": "dronedream.map-resource-verification.v1",
                "verified_count": len(results),
                "resources": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
