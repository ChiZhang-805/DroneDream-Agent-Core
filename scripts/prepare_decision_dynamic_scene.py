"""Build visible/collidable crossing scenes without giving their motion to the teacher."""

import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from dronedream_agent_core.decision_dataset import decode_object, file_digest
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from scripts.build_dynamic_obstacle_world import SCHEMA_VERSION, build_world


# 功能：把动态障碍加入独立实验世界，保持部署静态地图不包含未来运动信息。
# 输入：已准备通道配置、独立新目录、横穿方向和轨道半径；米制 ENU，速度米每秒。
# 输出：真实 SDF、来源回执和新配置；仅场景准备，不表示已采集或行为正确。
def prepare_dynamic(base, output, *, clockwise=True, radius_m=0.6, forward_m=2.0):
    if type(clockwise) is not bool:
        raise ValueError("DECISION_DYNAMIC_DIRECTION_INVALID")
    if (
        type(radius_m) not in (int, float)
        or not math.isfinite(radius_m)
        or not 0.2 <= radius_m <= 1.5
    ):
        raise ValueError("DECISION_DYNAMIC_RADIUS_INVALID")
    if (
        type(forward_m) not in (int, float)
        or not math.isfinite(forward_m)
        or not 2 <= forward_m <= 5
    ):
        raise ValueError("DECISION_DYNAMIC_FORWARD_INVALID")
    for name in ("route", "semantic", "world_sdf"):
        if file_digest(Path(base[name])) != base["asset_sha256"][name]:
            raise ValueError("DECISION_DYNAMIC_SOURCE_CHANGED:" + name)
    semantic = decode_object(Path(base["semantic"]).read_text("utf-8"))
    if semantic.get("name") != "Independent decision corridor":
        raise ValueError("DECISION_DYNAMIC_EXPLICIT_CORRIDOR_REQUIRED")
    # 障碍自身也会旋转，必须用方盒半对角线而非半边长核对整圈包络。
    # 起点y=±radius，机体向前速度和偏航角速度使圆心位于y=0。
    padding = math.hypot(0.2, 0.2) + 0.05
    lower, upper = base["minimum_enu_m"], base["maximum_enu_m"]
    if (
        lower[1] > -radius_m - padding
        or upper[1] < radius_m + padding
        or lower[0] > forward_m - radius_m - padding
        or upper[0] < forward_m + radius_m + padding
    ):
        raise ValueError("DECISION_DYNAMIC_ORBIT_DOES_NOT_FIT")
    # 通道由纯基元构成；隔离输入文件，禁止通用世界构建器复制旁边整个 native-run。
    # 若未来加入纹理/网格，必须另做资源依赖清单，不能静默漏拷贝或遍历用户数据。
    world_bytes = Path(base["world_sdf"]).read_bytes()
    root = ET.fromstring(world_bytes)
    forbidden = {"uri", "mesh", "heightmap", "image", "pbr", "script", "plugin", "include"}
    if any(node.tag in forbidden for node in root.iter()):
        raise ValueError("DECISION_DYNAMIC_PRIMITIVE_WORLD_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    source_root = output / "frozen-source"
    source_root.mkdir()
    frozen_world = source_root / "world.sdf"
    with frozen_world.open("xb") as stream:
        stream.write(world_bytes)
    if file_digest(frozen_world) != base["asset_sha256"]["world_sdf"]:
        raise ValueError("DECISION_DYNAMIC_SOURCE_CHANGED_DURING_FREEZE")
    config = copy.deepcopy(base)
    # 参数仅规定物理挑战，不给教师“应当等待/恢复”的标签；镜像和半径变体仍属同一地图组。
    specification = {
        "schema_version": SCHEMA_VERSION,
        "challenge_id": "decision-crossing-box",
        "required_encounter_distance_m": 3.0,
        "obstacles": [
            {
                "obstacle_id": "decision-crossing-box",
                "center_enu_m": [forward_m, radius_m if clockwise else -radius_m, 1.6],
                "size_m": [0.4, 0.4, 0.6],
                "motion": {
                    "kind": "circle",
                    "radius_m": radius_m,
                    "speed_mps": 0.3,
                    "clockwise": clockwise,
                },
            }
        ],
    }
    write_evidence_object(output / "physical-challenge.json", specification)
    receipt = build_world(
        base_world=frozen_world,
        challenge_spec=output / "physical-challenge.json",
        output_world=output / "world.sdf",
        receipt_output=output / "physical-world-receipt.json",
    )
    config["world_sdf"] = str(output / "world.sdf")
    config["asset_sha256"]["world_sdf"] = receipt["output_world_sha256"]
    config["output_root"] = str(output / "native-run")
    config["mission_id"] = "decision-crossing-" + ("clockwise" if clockwise else "counterclockwise")
    config["collection_expert_roles"] = [
        "local-navigation-policy",
        "precision-maneuver-policy",
        "recovery-policy",
    ]
    # 规划输入仍只有原静态地图；场景运动参数不进入配置或教师的状态接口。
    write_evidence_object(output / "config.json", config)
    write_evidence_object(
        output / "scene-receipt.json",
        {
            "simulation_only": True,
            "formal_training_additions": 0,
            "qualification_granted": False,
            "distinct_layout_claim": False,
            "scenario_not_a_behavior_label": True,
            "static_map_sha256": base["asset_sha256"]["semantic"],
            "world_sha256": config["asset_sha256"]["world_sdf"],
            "physical_receipt_sha256": file_digest(output / "physical-world-receipt.json"),
            "scene_family": "straight-corridor-crossing-variants-must-stay-in-same-split",
            "warning": (
                "Visible moving box, not a human recognition benchmark; "
                "actual motion must be witnessed."
            ),
        },
    )
    return config


# 功能：只在明确新目录准备场景；输入：命令行配置与方向；输出：配置路径和零正式新增数。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterclockwise", action="store_true")
    parser.add_argument("--radius-m", type=float, default=0.6)
    parser.add_argument("--forward-m", type=float, default=2.0)
    parser.add_argument("--maximum-decisions", type=int, default=200)
    args = parser.parse_args()
    if not 2 <= args.maximum_decisions <= 480:
        raise ValueError("DECISION_COLLECTION_BUDGET_INVALID")
    base = decode_object(args.base_config.read_text("utf-8"))
    base["episode_steps"] = args.maximum_decisions
    prepare_dynamic(
        base,
        args.output,
        clockwise=not args.counterclockwise,
        radius_m=args.radius_m,
        forward_m=args.forward_m,
    )
    print(json.dumps({"config": str(args.output / "config.json"), "formal_training_additions": 0}))


if __name__ == "__main__":
    main()
