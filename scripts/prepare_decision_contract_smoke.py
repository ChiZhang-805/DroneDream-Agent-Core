"""Build explicitly synthetic contract examples; NEVER formal UAV training data."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.decision_dataset import SPLITS, audit_records
from dronedream_agent_core.decision_shadow import ACTIONS
from dronedream_agent_core.decision_state_adapter import DecisionStateV2, decision_digest


# 功能：生成含完整 v2 身份与来源形状的测试夹具，不声称来源是实测。
# 输入：确定性编号及场景动作；输出：仅合成接口测试使用的 JSON 状态。
def synthetic_state(index: int, action: str) -> dict:
    stamp = {"source_id": "synthetic-contract", "evidence_sha256": "a" * 64,
             "observed_at_ms": 10000, "maximum_age_ms": 350}
    state = {"mission_id": f"smoke-{index}", "goal_id": "goal-1",
             "route_sha256": "b" * 64, "map_sha256": decision_digest(index),
             "vehicle_sha256": "c" * 64, "calibration_sha256": "d" * 64,
             "clock_domain": "synthetic-only", "sequence": index,
             "task": "navigate", "scene": "corridor", "braking_acceleration_mps2": .8,
             "body_radius_m": .25, "preferred_height_above_floor_m": 1.8,
             "frame": {"observed_at_ms": 10000,
                       "position_world_enu_m": {"x": 0.0, "y": 0.0, "z": 1.5},
                       "velocity_world_enu_mps": {"x": .1 + index * .0001, "y": 0.0, "z": 0.0},
                       "velocity_body_frd_mps": {"x": .1 + index * .0001, "y": 0.0, "z": 0.0},
                       "goal_relative_body_frd_m": {"x": 2.0, "y": 0.0, "z": 0.0},
                       "position_uncertainty_m": .05, "pose_source": stamp,
                       "geometry_source": stamp, "route_source": stamp,
                       "clearance_front_m": 4.0, "clearance_back_m": 2.0,
                       "clearance_left_m": 1.0, "clearance_right_m": 1.0,
                       "clearance_up_m": 1.0, "clearance_down_m": 1.2,
                       "local_route_verified": True,
                       "crossing_obstacle": action == "wait",
                       "persistent_blockage": action == "replan",
                       "caution_required": action == "slow_down",
                       "can_hold_position": True, "can_brake": True,
                       "payload_attached": False}}
    if action == "request_observation":
        state["frame"]["geometry_source"] = dict(stamp, observed_at_ms=9000)
    return DecisionStateV2.model_validate_json(json.dumps(state)).model_dump(mode="json")


# 功能：生成五类、五集合的独立烟测记录，所有正式资格明确为 false。
# 输入：每类数量；输出：记录列表，不混入用户任务或地图数据。
def smoke_rows(per_class: int = 2) -> list[dict]:
    rows = []
    for split in SPLITS:
        for action in ACTIONS:
            for _ in range(per_class):
                index = len(rows)
                rows.append({"sample_id": f"synthetic-{index}", "parent_group": f"{split}-group",
                             "episode_id": f"episode-{index}", "split": split,
                             "state": synthetic_state(index, action),
                             "acceptable_actions": [action], "license": "MIT",
                             "label_source": "synthetic-contract", "evidence": [],
                             "formal_training_eligible": False})
    return rows


# 功能：写出新烟测目录及审计，不允许覆写既有数据或冒充正式样本。
# 输入：新输出目录；输出：JSONL 与明确不可租 GPU 的数据准入报告。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = smoke_rows()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "contract-smoke.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    report = audit_records(rows, args.output, smoke=True)
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
