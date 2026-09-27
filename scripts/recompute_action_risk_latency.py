"""Recompute existing risk evidence under the decision-time latency envelope."""

import argparse
import json
import time
from pathlib import Path

from dronedream_agent_core.training.evidence_files import read_evidence_object
from dronedream_agent_core.training.risk_latency_reannotation import reannotate_decision_latency


# 功能：从明确源目录及原地图重算风险监督，发布到新目录并显示真实处理进度。
# 输入：CLI 的 source、semantic、output 路径。输出：原始证据不变的新离线数据；不训练或部署。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    semantic, _ = read_evidence_object(args.semantic, limit=64 * 1024 * 1024)
    started = time.monotonic()

    # 功能：按已完成动作数报告重算进度，不把预计完成或重标注计作新增采集。
    # 输入：done、total：已完成及总动作数。输出：结构化标准输出。
    def progress(done, total):
        if done % (161 * 8) == 0 or done == total:
            print(
                json.dumps(
                    dict(
                        completed_predictions=done,
                        total_predictions=total,
                        elapsed_seconds=round(time.monotonic() - started, 2),
                    )
                ),
                flush=True,
            )

    # 与原始 native/stream episode 加载器使用同一份运行碰撞几何；展示层几何不等价。
    primitives = semantic.get("runtime_collision_primitives", semantic.get("collision_primitives"))
    receipt = reannotate_decision_latency(args.source, args.output, primitives, progress=progress)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                class_counts=receipt["class_counts"],
                changed_labels=receipt["reannotation"]["changed_label_count"],
                new_observations=0,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
