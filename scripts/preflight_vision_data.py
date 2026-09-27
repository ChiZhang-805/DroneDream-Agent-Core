"""Read-only rent readiness audit; cannot grant model or flight qualification."""

import argparse
import json
from pathlib import Path

from dronedream_agent_core.plugin_files import hash_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_manifest import (
    inspect_vision_split,
    read_vision_manifest,
    require_disjoint_vision_splits,
)
from dronedream_agent_core.training.vision_resolution import inspect_model_resolution


# 功能：
#   租卡前实际解码三份数据并查重、核查标签和类别覆盖，缺口不得被当作可训练。
# 输入：
#   roots：训练、验证和最终测试各自包含 samples.jsonl 的目录。
#   width、height：本次拟训练模型的实际输入尺寸。
# 输出：
#   report：实际数据摘要、覆盖统计和缺口；就绪只代表输入结构满足最低要求。
def preflight(roots, width=224, height=128):
    if set(roots) != {"training", "validation", "test"}:
        raise ValueError("VISION_PREFLIGHT_THREE_SPLITS_REQUIRED")
    datasets, digests = {}, {}
    for split, root in roots.items():
        datasets[split], digests[split] = read_vision_manifest(root / "samples.jsonl")
    require_disjoint_vision_splits(datasets)
    if sum(map(len, datasets.values())) > 100_000:
        raise ValueError("VISION_PREFLIGHT_COMBINED_SAMPLE_BUDGET")
    reports, issues = {}, []
    for split, samples in datasets.items():
        result = inspect_vision_split(roots[split], samples)
        result["model_resolution"] = inspect_model_resolution(roots[split], samples, width, height)
        groups = {sample.scene_group_id or sample.flight_id
                  for sample in samples}
        result["groups"] = len(groups)
        result["manifest_sha256"] = digests[split]
        result["supervised_positive_counts"] = [sum(
            [*sample.scene_targets, *sample.quality_targets][index] >= 0.5
            and [*sample.scene_target_weights, *sample.quality_target_weights][index] > 0
            and (index >= 6 or sample.perception_supervision_enabled)
            for sample in samples) for index in range(10)]
        if len(samples) < 20 or len(groups) < 2:
            issues.append(f"{split}:INSUFFICIENT_SAMPLES_OR_GROUPS")
        if result["labelled_samples"] != len(samples):
            issues.append(f"{split}:SEMANTIC_LABELS_MISSING")
        if any(count < 100 for count in result["semantic_pixels"]):
            issues.append(f"{split}:SEMANTIC_CLASS_COVERAGE_INCOMPLETE")
        visibility = result["model_resolution"]
        if (any(count < 20 for count in visibility["class_visible_images"])
                or any(count < 2 for count in visibility["class_visible_groups"])):
            issues.append(f"{split}:MODEL_RESOLUTION_CLASS_COVERAGE_INCOMPLETE")
        if any(count < 20 for count in result["auxiliary_label_counts"]):
            issues.append(f"{split}:AUXILIARY_LABEL_COVERAGE_INCOMPLETE")
        for index, positive in enumerate(result["supervised_positive_counts"]):
            negative = result["auxiliary_label_counts"][index] - positive
            if positive < 2 or negative < 2:
                issues.append(f"{split}:AUXILIARY_{index}_CLASS_BALANCE_INCOMPLETE")
        reports[split] = result
    # 完成多集合解码后再次检查最初读取的清单，不给已变更的数据留下旧的就绪回执。
    for split, root in roots.items():
        if hash_plugin_file(root / "samples.jsonl", limit=256 * 1024**2) != digests[split]:
            raise ValueError("VISION_PREFLIGHT_MANIFEST_CHANGED")
    report = {"schema": "dronedream.vision-data-preflight.v1", "splits": reports,
              "ready_for_training": not issues, "issues": issues,
              "statistical_sufficiency_proven": False, "flight_qualification_granted": False}
    return report


# 功能：
#   运行无 GPU 的数据预检，独占保存回执，不修改数据或启动计费资源。
# 输入：
#   命令行参数：三个划分根目录与回执输出路径。
# 输出：
#   exit_code：结构就绪为零，有数据缺口为一。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("training", "validation", "test"):
        parser.add_argument("--" + split + "-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=128)
    args = parser.parse_args()
    roots = {split: getattr(args, split + "_root") for split in ("training", "validation", "test")}
    report = preflight(roots, args.width, args.height)
    publish_runtime_json(args.receipt, report, replace_existing=False)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ready_for_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
