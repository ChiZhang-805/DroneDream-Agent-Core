"""Freeze real blocked-passage materials without claiming successful replanning."""

import argparse
from pathlib import Path

from dronedream_agent_core.contracts import GraphRoute
from dronedream_agent_core.decision_dataset import decode_object, evidence_path, file_digest
from dronedream_agent_core.decision_detour_scene import detour_documents, verify_detour_geometry
from dronedream_agent_core.decision_scene_topology import verified_scene_family
from dronedream_agent_core.training.evidence_publication import (
    publish_evidence_bytes,
    write_evidence_object,
)
from scripts.prepare_decision_native_scene import make_world


# 功能：将已固定划分的房间树变成有真实阻塞与绕行的环路材料；不生成正式执行标签。
# 输入：原套件路径、机体半径/高度/净空米、独占输出目录；输出：按原布局保留集合的清单。
# 先验证所有来源及几何再写入；相同环路图若跨集合冲突则拒绝，不能静默重划测试集。
def prepare(suite_path, output, *, radius_m, height_m, clearance_m):
    if suite_path.stat().st_size > 1024**2:
        raise ValueError("DECISION_DETOUR_SUITE_TOO_LARGE")
    suite = decode_object(suite_path.read_text("utf-8"))
    entries = suite.get("layouts")
    if (
        suite.get("schema_version") != "dronedream.decision-topology-suite.v1"
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= 20
    ):
        raise ValueError("DECISION_DETOUR_SUITE_INVALID")
    prepared, families = [], {}
    for entry in entries:
        receipt_path = evidence_path(suite_path.parent, entry["receipt"])
        if receipt_path.stat().st_size > 65536:
            raise ValueError("DECISION_DETOUR_RECEIPT_TOO_LARGE")
        receipt = decode_object(receipt_path.read_text("utf-8"))
        family = verified_scene_family(receipt, receipt["semantic_sha256"])
        if (
            family != entry["family"]
            or entry["split"] != receipt.get("split")
            or entry["split"] not in {"train", "development", "calibration", "test", "stress"}
        ):
            raise ValueError("DECISION_DETOUR_PARENT_BINDING_INVALID")
        recipe = receipt["recipe"]
        size = recipe["size"]
        existing = {tuple(e) for e in recipe["edges"]}
        extra = next(
            [a, b]
            for a in range(size**2)
            for b in range(a + 1, size**2)
            if abs(a % size - b % size) + abs(a // size - b // size) == 1 and (a, b) not in existing
        )
        doc = detour_documents(recipe, [extra], receipt["vehicle_clearance"])
        checks = verify_detour_geometry(
            doc, radius_m=radius_m, height_m=height_m, clearance_m=clearance_m
        )
        for name in ("initial_route", "replacement_route"):
            GraphRoute.model_validate(doc[name])
        prior = families.setdefault(doc["family"], entry["split"])
        if prior != entry["split"]:
            raise ValueError("DECISION_DETOUR_LAYOUT_SPLIT_LEAKAGE")
        prepared.append((entry, receipt, extra, doc, checks))
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for index, (entry, receipt, extra, doc, checks) in enumerate(prepared):
        folder = output / f"layout-{index:02d}"
        folder.mkdir()
        hashes = {}
        for key in ("before", "blocked", "initial_route", "replacement_route"):
            filename = key + ".json"
            write_evidence_object(folder / filename, doc[key])
            hashes[filename] = file_digest(folder / filename)
        for key in ("before", "blocked"):
            filename = key + ".sdf"
            publish_evidence_bytes(
                folder / filename,
                make_world(doc[key]["collision_primitives"]).encode("utf-8"),
                limit=4 * 1024**2,
            )
            hashes[filename] = file_digest(folder / filename)
        provenance = dict(
            schema_version="dronedream.decision-detour-material.v1",
            scene_family="room-loop-detour-generator-v1",
            recipe=receipt["recipe"],
            vehicle_clearance=receipt["vehicle_clearance"],
            extra_edges=[extra],
            parent_family=doc["parent_family"],
            family=doc["family"],
            split=entry["split"],
            blocked_edge=doc["blocked_edge"],
            assets=hashes,
            checks=checks,
            simulation_only=True,
            qualification_granted=False,
            formal_training_additions=0,
        )
        write_evidence_object(folder / "scene-receipt.json", provenance)
        manifest.append(
            dict(
                receipt=f"layout-{index:02d}/scene-receipt.json",
                receipt_sha256=file_digest(folder / "scene-receipt.json"),
                split=entry["split"],
                parent_family=doc["parent_family"],
            )
        )
    result = dict(
        schema_version="dronedream.decision-detour-suite.v1",
        layouts=manifest,
        source_suite_sha256=file_digest(suite_path),
        materials=len(manifest),
        physically_verified_layouts=0,
        formal_training_additions=0,
        gpu_data_ready=False,
    )
    write_evidence_object(output / "suite.json", result)
    return result


# 功能：显式创建仿真材料；输入：原套件、米制机体参数及新目录；输出：真实准备数量。
def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, required=True)
    parser.add_argument("--height-m", type=float, required=True)
    parser.add_argument("--clearance-m", type=float, required=True)
    args = parser.parse_args()
    result = prepare(
        args.suite,
        args.output,
        radius_m=args.radius_m,
        height_m=args.height_m,
        clearance_m=args.clearance_m,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "layouts"}))


if __name__ == "__main__":
    main()
