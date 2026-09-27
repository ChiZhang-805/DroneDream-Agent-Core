"""Prepare collision-consistent independent layouts and fix splits before any flights."""

import argparse
import copy
import json
from pathlib import Path

from dronedream_agent_core.contracts import GraphRoute
from dronedream_agent_core.decision_dataset import decode_object, file_digest
from dronedream_agent_core.decision_scene_topology import generate_edges, scene_documents
from dronedream_agent_core.runtime_bindings import load_map_runtime_bindings
from dronedream_agent_core.training.evidence_publication import (
    publish_evidence_bytes,
    write_evidence_object,
)
from dronedream_agent_core.training.swept_geometry import SweptMapGeometry
from scripts.prepare_decision_native_scene import make_world


# 功能：生成20种非同形房间拓扑，先划分集合再采集；每种地图的宽度/动态变体沿用同集合。
# 输入：既有原生配置、净空配置、机体半径/高度米和新目录；输出：静态验证清单，正式新增0。
# 只写新实验目录，不能直接安装到用户Runtime，也不把静态无碰撞等同于实际飞行通过。
def prepare_suite(base, vehicle_clearance, output, *, radius_m, height_m):
    families, materials = set(), []
    for seed in range(10000):
        recipe = {"size": 4, "edges": generate_edges(4, seed), "cell_m": 4.0, "flight_z_m": 1.6}
        semantic, route, family = scene_documents(recipe, vehicle_clearance)
        if family in families:
            continue
        geometry = SweptMapGeometry(
            semantic["collision_primitives"], radius_m=radius_m, half_height_m=height_m / 2
        )
        clearance = geometry.clearance([tuple(p[k] for k in "xyz") for p in route["positions_m"]])
        if clearance < base["required_clearance_m"]:
            raise ValueError("DECISION_TOPOLOGY_ROUTE_CLEARANCE_INSUFFICIENT")
        GraphRoute.model_validate(route)
        load_map_runtime_bindings(semantic)
        families.add(family)
        materials.append((recipe, semantic, route, family, clearance))
        if len(materials) == 20:
            break
    if len(materials) != 20:
        raise ValueError("DECISION_TOPOLOGY_INDEPENDENT_LAYOUTS_INSUFFICIENT")
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    splits = (
        ["train"] * 12 + ["development"] * 2 + ["calibration"] * 2 + ["test"] * 2 + ["stress"] * 2
    )
    for index, ((recipe, semantic, route, family, clearance), split) in enumerate(
        zip(materials, splits, strict=True)
    ):
        directory = output / f"layout-{index:02d}"
        directory.mkdir()
        write_evidence_object(directory / "semantic.json", semantic)
        write_evidence_object(directory / "route.json", route)
        publish_evidence_bytes(
            directory / "world.sdf",
            make_world(semantic["collision_primitives"]).encode("utf-8"),
            limit=4 * 1024**2,
        )
        config = copy.deepcopy(base)
        for key, filename in (
            ("semantic", "semantic.json"),
            ("route", "route.json"),
            ("world_sdf", "world.sdf"),
        ):
            config[key] = str(directory / filename)
            config["asset_sha256"][key] = file_digest(directory / filename)
        config.update(
            output_root=str(directory / "native-run"),
            mission_id=f"decision-room-tree-{index:02d}",
            minimum_enu_m=[-2.0, -2.0, 0.0],
            maximum_enu_m=[14.0, 14.0, 3.6],
            episode_steps=480,
            bounded_hybrid_control=True,
            batch_static_world_visuals=False,
            collection_expert_roles=[
                "local-navigation-policy",
                "precision-maneuver-policy",
                "recovery-policy",
            ],
        )
        write_evidence_object(directory / "config.json", config)
        receipt = {
            "scene_family": "room-tree-generator-v1",
            "recipe": recipe,
            "vehicle_clearance": vehicle_clearance,
            "distinct_layout_claim": True,
            "semantic_sha256": config["asset_sha256"]["semantic"],
            "world_sha256": config["asset_sha256"]["world_sdf"],
            "split": split,
            "simulation_only": True,
            "qualification_granted": False,
            "formal_training_additions": 0,
        }
        write_evidence_object(directory / "scene-receipt.json", receipt)
        entries.append(
            {
                "config": f"layout-{index:02d}/config.json",
                "config_sha256": file_digest(directory / "config.json"),
                "receipt": f"layout-{index:02d}/scene-receipt.json",
                "family": family,
                "split": split,
                "static_route_clearance_m": clearance,
                "route_length_m": route["route_length_m"],
            }
        )
    report = {
        "schema_version": "dronedream.decision-topology-suite.v1",
        "layouts": entries,
        "independent_topologies_prepared": len(families),
        "physically_verified_layouts": 0,
        "formal_training_additions": 0,
        "gpu_data_ready": False,
    }
    write_evidence_object(output / "suite.json", report)
    return report


# 功能：在当前操作系统可访问的源配置上构建新实验套件；输入：显式路径；输出：可审计清单。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = decode_object(args.base_config.read_text("utf-8"))
    semantic = decode_object(Path(base["semantic"]).read_text("utf-8"))
    vehicle = decode_object(Path(base["vehicle"]).read_text("utf-8"))
    report = prepare_suite(
        base,
        semantic["vehicle_clearance"],
        args.output,
        radius_m=vehicle["body_radius_m"],
        height_m=vehicle["body_height_m"],
    )
    print(json.dumps({k: v for k, v in report.items() if k != "layouts"}))


if __name__ == "__main__":
    main()
