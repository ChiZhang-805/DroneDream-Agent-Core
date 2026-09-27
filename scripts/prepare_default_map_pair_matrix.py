"""Prepare a bounded map/aircraft qualification plan for every packaged default map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dronedream_agent_core.asset_pair_qualification import (
    AssetPairQualificationError,
    prepare_asset_pair_qualification,
)
from dronedream_agent_core.asset_packages import inspect_ddpkg


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--vehicle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for map_archive in sorted(args.map_root.glob("*.ddpkg")):
        if inspect_ddpkg(map_archive).manifest.asset_kind not in {"map", "world"}:
            continue
        work_root = args.output_root / map_archive.stem
        try:
            plan = prepare_asset_pair_qualification(
                map_archive=map_archive,
                vehicle_archive=args.vehicle,
                work_root=work_root,
            )
        except (AssetPairQualificationError, ValueError) as error:
            results.append(
                {
                    "map": map_archive.name,
                    "prepared": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        results.append(
            {
                "map": map_archive.name,
                "prepared": True,
                "qualification_id": plan.qualification_id,
                "route_length_m": plan.route.route_length_m,
                "node_count": len(plan.route.node_ids),
                "all_edges_flight_verified": plan.route.all_edges_flight_verified,
                "required_runtime_gates": plan.required_runtime_gates,
            }
        )
    report = {
        "schema_version": "dronedream.default-map-pair-preparation.v1",
        "map_count": len(results),
        "prepared_count": sum(1 for item in results if item["prepared"]),
        "results": results,
    }
    (args.output_root / "matrix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["prepared_count"] == report["map_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
