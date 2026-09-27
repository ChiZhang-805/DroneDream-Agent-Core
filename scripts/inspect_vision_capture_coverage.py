"""Check completed native batches and their actual model-resolution signal before assembly."""

import argparse
import json
from collections import Counter
from pathlib import Path

from assemble_vision_dataset import inspect_capture

from dronedream_agent_core.plugin_files import hash_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_curation import CURATION_POLICY, curate_render_sample
from dronedream_agent_core.training.vision_resolution import inspect_model_resolution


# 功能：
#   核验完整原生来源并预演固定筛图规则，输出真实有效数量和模型尺寸下的类别覆盖。
# 输入：
#   captures、width、height：显式指定的完整采集批次及预定模型输入尺寸。
# 输出：
#   report：逐批、逐集合的客观覆盖；不修改图片或赋予训练及飞行资格。
def inspect_batches(captures, width=224, height=128):
    if not 1 <= len(captures) <= 100 or len(set(captures)) != len(captures):
        raise ValueError("VISION_COVERAGE_CAPTURE_BUDGET_INVALID")
    seen, records, summary, identities, digests = set(), [], {}, {}, []
    for root in captures:
        datasets, _receipt, _sources, receipt_digest = inspect_capture(root, allow_duplicates=True)
        digests.append((root / "capture-receipt.json", receipt_digest))
        for split, samples in datasets.items():
            kept, reasons = [], Counter()
            for sample in samples:
                # 预演去重不允许把同一布局偷偷改名分入另一集合。
                for kind, identity in (("scene", sample.scene_group_id),
                                       ("flight", sample.flight_id)):
                    if (identity is not None
                            and identities.setdefault((kind, identity), split) != split):
                        raise ValueError("VISION_COVERAGE_ORIGINAL_GROUP_LEAKAGE")
                decision = curate_render_sample(root / split, sample, seen)
                reasons[decision["reason"]] += 1
                if decision["retained"]:
                    kept.append(sample)
            visibility = inspect_model_resolution(root / split, kept, width, height)
            record = {"batch": root.name, "split": split, "captured": len(samples),
                      "retained": len(kept), "reasons": dict(reasons), "visibility": visibility}
            records.append(record)
            total = summary.setdefault(split, {"captured": 0, "retained": 0, "quality_only": 0,
                "class_visible_images": [0] * 8, "class_pixels": [0] * 8,
                "class_visible_group_ids": [set() for _ in range(8)], "reasons": Counter()})
            total["captured"] += len(samples)
            total["retained"] += len(kept)
            total["quality_only"] += sum(not s.perception_supervision_enabled for s in kept)
            total["reasons"].update(reasons)
            for index in range(8):
                total["class_visible_images"][index] += visibility["class_visible_images"][index]
                total["class_pixels"][index] += visibility["class_pixels"][index]
                total["class_visible_group_ids"][index].update(
                    visibility["class_visible_group_ids"][index])
        print(json.dumps({"checked": root.name, "batches": len(digests)}), flush=True)
    for total in summary.values():
        total["class_visible_group_ids"] = [sorted(ids) for ids in total["class_visible_group_ids"]]
        total["class_visible_groups"] = [len(ids) for ids in total["class_visible_group_ids"]]
        total["reasons"] = dict(total["reasons"])
    for path, digest in digests:
        if hash_plugin_file(path, limit=8 * 1024**2) != digest:
            raise ValueError("VISION_COVERAGE_RECEIPT_CHANGED")
    report = {"schema": "dronedream.native-capture-coverage.v1", "batches": records,
              "splits": summary, "curation_policy": CURATION_POLICY,
              "capture_receipt_sha256": {str(path): digest for path, digest in digests},
              "input_width": width, "input_height": height,
              "ready_for_training": False, "flight_qualification_granted": False,
              "model_improvement_measured": False}
    return report


# 功能：
#   对明确指定的采集目录生成新的只读核验报告；未完成批次不能混入实际数量。
# 输入：
#   命令行参数：采集目录、模型输入尺寸及新报告路径。
# 输出：
#   exit_code：全部来源检查与覆盖统计完成为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_batches(args.capture, args.width, args.height)
    publish_runtime_json(args.report, report, replace_existing=False)
    print(json.dumps(report["splits"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
