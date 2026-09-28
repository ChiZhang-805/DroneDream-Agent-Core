#!/usr/bin/env python3
"""Verify the reviewed PX4 vehicle catalog against an immutable Git checkout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from dronedream_agent_app.vehicle_resource_catalog import vehicle_resource_catalog


def _git(checkout: Path, *arguments: str, text: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def _source_archive(checkout: Path, subpath: str) -> tuple[str, int, int, dict[str, bytes]]:
    """Reproduce RemoteAssetSourceService's fixed ZIP without checkout line-ending changes."""
    prefix = f"{subpath.rstrip('/')}/"
    listing = str(_git(checkout, "ls-tree", "-r", "--name-only", "HEAD", "--", prefix, text=True))
    names = [name for name in listing.splitlines() if name.startswith(prefix)]
    if not names:
        raise ValueError(f"catalog source path is absent: {subpath}")
    payloads: dict[str, bytes] = {}
    total_size = 0
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for name in sorted(names, key=lambda item: item[len(prefix) :].encode("utf-8")):
            relative = name[len(prefix) :]
            payload = bytes(_git(checkout, "cat-file", "blob", f"HEAD:{name}"))
            payloads[relative] = payload
            total_size += len(payload)
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.file_size = len(payload)
            with bundle.open(info, "w", force_zip64=True) as output:
                output.write(payload)
    return hashlib.sha256(archive.getvalue()).hexdigest(), total_size, len(names), payloads


def verify_catalog(checkout: Path) -> dict[str, object]:
    catalog = vehicle_resource_catalog()
    expected_commit = str(catalog["resources"][0]["source"]["git_ref"])
    actual_commit = str(_git(checkout, "rev-parse", "HEAD", text=True)).strip()
    if actual_commit != expected_commit:
        raise ValueError(f"checkout commit mismatch: {actual_commit} != {expected_commit}")
    results: list[dict[str, object]] = []
    all_model_paths = {
        path.name
        for path in (checkout / "models").iterdir()
        if path.is_dir() and (path / "model.sdf").is_file()
    }
    for resource in catalog["resources"]:
        source = resource["source"]
        analysis = resource["analysis"]
        digest, source_bytes, file_count, payloads = _source_archive(
            checkout, str(source["subpath"])
        )
        if digest != source["expected_sha256"]:
            raise ValueError(f"source digest mismatch for {resource['resource_id']}")
        if source_bytes != analysis["source_size_bytes"]:
            raise ValueError(f"source byte count mismatch for {resource['resource_id']}")
        if file_count != analysis["source_file_count"]:
            raise ValueError(f"source file count mismatch for {resource['resource_id']}")
        if "model.sdf" not in payloads or "model.config" not in payloads:
            raise ValueError(f"Gazebo entrypoint metadata missing for {resource['resource_id']}")
        ElementTree.fromstring(payloads["model.sdf"])
        ElementTree.fromstring(payloads["model.config"])
        missing_dependencies = sorted(
            dependency
            for dependency in analysis["dependency_models"]
            if dependency not in all_model_paths
        )
        if missing_dependencies:
            raise ValueError(
                f"catalog dependencies missing for {resource['resource_id']}: "
                + ", ".join(missing_dependencies)
            )
        results.append(
            {
                "resource_id": resource["resource_id"],
                "source_sha256": digest,
                "source_bytes": source_bytes,
                "source_file_count": file_count,
                "dependencies_present": True,
                "xml_valid": True,
            }
        )
    return {
        "schema_version": "dronedream.vehicle-resource-verification.v1",
        "catalog_revision": catalog["catalog_revision"],
        "source_commit": actual_commit,
        "resource_count": len(results),
        "passed": True,
        "resources": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    report = verify_catalog(arguments.checkout.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
