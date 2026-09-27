"""Prepare grouped native-render inputs; never rent a GPU or fabricate camera images."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    portable_plugin_path,
    read_plugin_file,
)
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_scenario_suite import PERSON_SHA256, build_scenario
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   独占保存生成制品并记录摘要，拒绝覆盖以前采集的数据。
# 输入：
#   path、payload：新文件路径与已固定的字节。
# 输出：
#   digest：实际写入内容的 SHA-256。
def write_new(path, payload):
    check_plain_plugin_path(path)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    return digest


# 功能：
#   预先固定整布局的训练、验证和测试划分，保留人形资产原始归属及全部输入摘要。
# 输入：
#   命令行参数：源人形目录、新场景目录、划分布局数和每区域视角数。
# 输出：
#   exit_code：完整发布计划为零；任何身份、预算或输出冲突均报错。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-layouts", type=int, default=20)
    parser.add_argument("--validation-layouts", type=int, default=8)
    parser.add_argument("--test-layouts", type=int, default=8)
    parser.add_argument("--views-per-anchor", type=int, default=16)
    args = parser.parse_args()
    counts = [args.training_layouts, args.validation_layouts, args.test_layouts]
    if (any(not 1 <= value <= 30 for value in counts)
            or not 2 <= args.views_per_anchor <= 128):
        raise ValueError("VISION_SCENARIO_SUITE_BUDGET_INVALID")
    attribution = read_plugin_file(args.person_source / "source.json", limit=64 * 1024)
    source = decode_json(attribution)
    files = source.get("files_sha256") if type(source) is dict else None
    if (type(files) is not dict or not 1 <= len(files) <= 32
            or files.get("meshes/standing.dae") != PERSON_SHA256
            or source.get("license_name") != "Creative Commons Zero v1.0 Universal"):
        raise ValueError("VISION_SCENARIO_PERSON_SOURCE_CHANGED")
    person_files, total_bytes = {}, 0
    for relative, digest in files.items():
        path = portable_plugin_path(relative)
        raw = read_plugin_file(args.person_source / path, limit=32 * 1024**2)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("VISION_SCENARIO_PERSON_FILE_CHANGED")
        total_bytes += len(raw)
        if total_bytes > 64 * 1024**2:
            raise ValueError("VISION_SCENARIO_PERSON_TOTAL_BUDGET")
        person_files[relative] = raw
    output = args.output.absolute()
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    assignments = zip(("training", "validation", "test"), counts, (1000, 2000, 3000), strict=True)
    for split, count, offset in assignments:
        for index in range(1, count + 1):
            seed = offset + index
            world, labels, plan, receipt = build_scenario(seed, split, args.views_per_anchor)
            root = output / f"layout-{seed}"
            root.mkdir()
            write_new(root / "world.sdf", world)
            for relative, raw in person_files.items():
                target = root / "person" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                write_new(target, raw)
            write_new(root / "person-source.json", attribution)
            publish_runtime_json(root / "labels.json", labels, replace_existing=False)
            publish_runtime_json(root / "views.json", plan, replace_existing=False)
            hashes = {path.relative_to(root).as_posix():
                      hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob("*") if path.is_file()}
            receipt["files_sha256"] = hashes
            publish_runtime_json(root / "layout-receipt.json", receipt, replace_existing=False)
            entries.append({"directory": root.name, **receipt})
    result = {"schema": "dronedream.vision-suite-plan.v1", "complete": True,
              "planned_native_frames": sum(entry["views"] for entry in entries),
              "layouts": entries, "rendered_frames": 0, "ready_for_training": False}
    publish_runtime_json(output / "suite-plan.json", result, replace_existing=False)
    print(json.dumps({"layouts": len(entries),
                      "planned_native_frames": result["planned_native_frames"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
