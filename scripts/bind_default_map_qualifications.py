"""Bind verified PX4/Gazebo runs into a multi-pair default asset release.

The existing bundled school map remains the default pair.  Every reviewed map is
added as an independently qualified map/vehicle pair because qualification
receipts bind both exact package contents; one vehicle package cannot honestly
claim qualification against several maps with a single receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dronedream_agent_core.asset_package_storage import (
    export_stored_asset,
    read_stored_manifest,
)
from dronedream_agent_core.asset_packages import inspect_ddpkg
from dronedream_agent_core.asset_pair_qualification import (
    AssetPairQualificationPlan,
    bind_qualification_receipt,
    build_pair_qualification_receipt,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment(path: Path) -> dict[str, str]:
    raw = _json(path)
    required = {"ros_distribution", "gazebo_sim", "px4_commit", "runtime_manifest_sha256"}
    if set(raw) != required or not all(isinstance(raw[key], str) and raw[key] for key in required):
        raise ValueError("qualification environment must contain the four pinned Runtime fields")
    return {key: str(raw[key]) for key in sorted(required)}


def _pair_record(
    *,
    resource_id: str,
    output_root: Path,
    map_package: Path,
    vehicle_package: Path,
    source_map_sha256: str,
    source_vehicle_sha256: str,
    qualification_id: str,
) -> dict[str, Any]:
    packages: list[dict[str, str]] = []
    for kind, package, source_sha256 in (
        ("map", map_package, source_map_sha256),
        ("vehicle", vehicle_package, source_vehicle_sha256),
    ):
        absolute_package = output_root / package
        inspected = inspect_ddpkg(absolute_package)
        binding = inspected.manifest.qualification
        if binding is None or len(binding.evidence_paths) != 1:
            raise ValueError(f"qualified package has no unique receipt: {absolute_package}")
        evidence_path = binding.evidence_paths[0]
        evidence = next(entry for entry in inspected.manifest.files if entry.path == evidence_path)
        packages.append(
            {
                "kind": kind,
                "asset_id": inspected.manifest.asset_id,
                "file": package.as_posix(),
                "sha256": _sha256(absolute_package),
                "source_content_sha256": source_sha256,
                "content_sha256": inspected.manifest.content_sha256,
                "receipt_sha256": evidence.sha256,
            }
        )
    receipt_hashes = {entry["receipt_sha256"] for entry in packages}
    if len(receipt_hashes) != 1:
        raise ValueError(f"map and vehicle receipts differ for {resource_id}")
    for entry in packages:
        entry.pop("receipt_sha256")
    return {
        "schema_version": "dronedream.bundled-qualified-pair.v1",
        "resource_id": resource_id,
        "qualification_id": qualification_id,
        "receipt_sha256": receipt_hashes.pop(),
        "packages": packages,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--existing-default-assets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    runs = _json(args.runs)
    environment = _environment(args.environment)
    existing_index = _json(args.existing_default_assets / "index.json")
    default_pair = existing_index.get("qualified_pair")
    if (
        existing_index.get("schema_version")
        not in {"dronedream.bundled-assets.v2", "dronedream.bundled-assets.v3"}
        or not isinstance(default_pair, dict)
    ):
        raise ValueError("existing default assets have no qualified default pair")

    args.output_root.mkdir(parents=True)
    copied_default = json.loads(json.dumps(default_pair))
    for package in copied_default.get("packages", []):
        if not isinstance(package, dict) or not isinstance(package.get("file"), str):
            raise ValueError("existing default pair package entry is invalid")
        source = args.existing_default_assets / package["file"]
        destination = args.output_root / package["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if _sha256(destination) != package.get("sha256"):
            raise ValueError("copied default pair hash mismatch")
    copied_default["resource_id"] = "dronedream-school-map"

    pairs: list[dict[str, Any]] = [copied_default]
    for resource_id in sorted(runs):
        run_value = runs[resource_id]
        if not isinstance(run_value, str):
            raise ValueError(f"run path must be a string: {resource_id}")
        work_root = args.matrix_root / resource_id
        run_dir = Path(run_value)
        plan = AssetPairQualificationPlan.model_validate(_json(work_root / "qualification-plan.json"))
        evidence_path = run_dir / "mission_evidence.json"
        runtime_evidence = _json(evidence_path)
        qualified_at = datetime.fromtimestamp(evidence_path.stat().st_mtime, tz=UTC)
        receipt = build_pair_qualification_receipt(
            plan=plan,
            work_root=work_root,
            runtime_evidence=runtime_evidence,
            environment_versions=environment,
            qualified_at=qualified_at,
        )

        source_map = args.source_map_root / f"{resource_id}.ddpkg"
        source_map_inspected = inspect_ddpkg(source_map)
        vehicle_manifest, _ = read_stored_manifest(work_root / "vehicle")
        pair_root = args.output_root / "pairs" / resource_id
        pair_root.mkdir(parents=True)
        source_vehicle = pair_root / "vehicle-source.ddpkg"
        export_stored_asset(
            work_root / "vehicle",
            source_vehicle,
            expected_asset_id=vehicle_manifest.asset_id,
            expected_content_sha256=vehicle_manifest.content_sha256,
        )
        map_package = pair_root / "map.ddpkg"
        vehicle_package = pair_root / "vehicle.ddpkg"
        bind_qualification_receipt(
            source_archive=source_map,
            destination_archive=map_package,
            receipt=receipt,
        )
        bind_qualification_receipt(
            source_archive=source_vehicle,
            destination_archive=vehicle_package,
            receipt=receipt,
        )
        source_vehicle.unlink()
        pairs.append(
            _pair_record(
                resource_id=resource_id,
                output_root=args.output_root,
                map_package=map_package.relative_to(args.output_root),
                vehicle_package=vehicle_package.relative_to(args.output_root),
                source_map_sha256=source_map_inspected.manifest.content_sha256,
                source_vehicle_sha256=vehicle_manifest.content_sha256,
                qualification_id=receipt.qualification_id,
            )
        )

    index = {
        "schema_version": "dronedream.bundled-assets.v3",
        "default_qualification_id": copied_default["qualification_id"],
        "qualified_pair": copied_default,
        "qualified_pairs": pairs,
        "superseded_qualified_pairs": existing_index.get("superseded_qualified_pairs", []),
    }
    (args.output_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"pair_count": len(pairs), "output": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
