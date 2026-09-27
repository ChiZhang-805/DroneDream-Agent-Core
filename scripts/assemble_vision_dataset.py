"""Assemble completed render batches without weakening provenance or spatial splits."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dronedream_agent_core.asset_package_storage import publish_asset_file
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.runtime_control_io import publish_runtime_json
from dronedream_agent_core.training.vision_balance import cap_training_group_weights
from dronedream_agent_core.training.vision_curation import CURATION_POLICY, curate_render_sample
from dronedream_agent_core.training.vision_manifest import (
    inspect_vision_split,
    read_vision_manifest,
    require_disjoint_vision_splits,
)
from dronedream_plugin_sdk.protocol import decode_json

SPLITS = ("training", "validation", "test")


# 功能：
#   从同一有界字节快照解析来源回执并计算摘要，避免先检查旧内容再复制新内容。
# 输入：
#   path、limit：普通来源文件和允许字节数。
# 输出：
#   payload、digest：已解析对象及其同一份来源字节的摘要。
def read_source_snapshot(path, limit):
    raw = read_plugin_file(path, limit=limit)
    payload = decode_json(raw, limit=limit)
    if type(payload) is not dict:
        raise ValueError("VISION_ASSEMBLY_SOURCE_NOT_OBJECT")
    digest = hashlib.sha256(raw).hexdigest()
    return payload, digest


# 功能：
#   验证完整采集回执、清单和每张图的原始来源，失败批次不能经汇总变成成功数据。
# 输入：
#   root：一个已结束的原生相机采集目录。
#   allow_duplicates：是否仅核查原始完整性，随后仍须正式去重。
# 输出：
#   datasets、receipt、sources、receipt_digest：样本、完整回执、逐样本来源及回执摘要。
def inspect_capture(root, *, allow_duplicates=False):
    receipt, receipt_digest = read_source_snapshot(root / "capture-receipt.json", 8 * 1024**2)
    hashes = receipt.get("manifest_sha256")
    if (receipt.get("complete") is not True or receipt.get("source_kind") != "rendered-view"
            or receipt.get("physical_flight_evidence") is not False
            or type(hashes) is not dict or not hashes or set(hashes) - set(SPLITS)):
        raise ValueError("VISION_ASSEMBLY_CAPTURE_INCOMPLETE")
    datasets, sources = {}, {}
    for split, expected in hashes.items():
        values, digest = read_vision_manifest(root / split / "samples.jsonl")
        if digest != expected:
            raise ValueError("VISION_ASSEMBLY_MANIFEST_CHANGED")
        inspect_vision_split(root / split, values, allow_duplicates=allow_duplicates)
        datasets[split] = values
        for sample in values:
            view_id = Path(sample.image_relative_path).stem
            source_path = root / "views" / (view_id + ".json")
            record, source_digest = read_source_snapshot(source_path, 2 * 1024**2)
            source = record.get("source")
            view = source.get("view") if type(source) is dict else None
            stored_sample = LocalVisionTrainingSample.model_validate(record.get("sample"))
            if (stored_sample != sample or sample.source_kind != "rendered-view"
                    or type(source) is not dict or type(view) is not dict
                    or sha256_json(source) != sample.source_record_sha256
                    or view.get("view_id") != view_id
                    or view.get("split") != split
                    or view.get("scene_group_id") != sample.scene_group_id
                    or source.get("world_sha256") != receipt.get("world_sha256")
                    or source.get("world_sha256") != sample.map_sha256
                    or source.get("camera_sha256") != receipt.get("camera_sha256")
                    or source.get("rgb_sha256") != sample.image_sha256
                    or source.get("semantic_sha256") != sample.semantic_mask_sha256):
                raise ValueError("VISION_ASSEMBLY_SOURCE_BINDING_INVALID")
            if view_id in sources:
                raise ValueError("VISION_ASSEMBLY_DUPLICATE_SOURCE_ID")
            raw_paths = {}
            for kind in ("rgb", "semantic"):
                path = root / "raw" / (view_id + "-" + kind + ".png")
                digest = hash_plugin_file(path, limit=16 * 1024**2)
                if digest != source.get("raw_" + kind + "_sha256"):
                    raise ValueError("VISION_ASSEMBLY_RAW_IMAGE_CHANGED")
                raw_paths[path] = digest
            sources[view_id] = {source_path: source_digest, **raw_paths}
    if type(receipt.get("view_count")) is not int or receipt["view_count"] != len(sources):
        raise ValueError("VISION_ASSEMBLY_SAMPLE_COUNT_MISMATCH")
    require_disjoint_vision_splits(datasets)
    return datasets, receipt, sources, receipt_digest


# 功能：
#   1. 汇总多个明确指定的完成批次，保持原始图、逐视角来源及整区域划分不变。
#   2. 新目录独占发布；预算、重复图或来源检查失败时保留现场，不发布完成标记。
# 输入：
#   captures、destination、max_bytes：来源目录列表、新汇总目录及总制品字节上限。
#   curate：是否应用预先固定的去重及无信息图剔除规则并保存逐图决定。
#   group_share_cap：可选训练空间组累计损失权重上限，不改变留出集合。
# 输出：
#   receipt：样本与逐文件摘要，汇总成功不授予训练充分性或飞行资格。
def assemble(captures, destination, max_bytes=10 * 1024**3, *, curate=False, group_share_cap=None):
    if (type(curate) is not bool or not 1 <= len(captures) <= 100 or type(max_bytes) is not int
            or not 1024**2 <= max_bytes <= 40 * 1024**3):
        raise ValueError("VISION_ASSEMBLY_INPUT_BUDGET_INVALID")
    captures = [path.absolute() for path in captures]
    if len(set(captures)) != len(captures):
        raise ValueError("VISION_ASSEMBLY_DUPLICATE_CAPTURE")
    combined, transfers, lineage = {split: [] for split in SPLITS}, {}, []
    decisions, seen, source_groups, source_flights = [], set(), {}, {}
    for index, root in enumerate(captures):
        datasets, capture_receipt, sources, receipt_digest = inspect_capture(
            root, allow_duplicates=curate)
        prefix = f"batch-{index:03d}"
        for split, values in datasets.items():
            for sample in values:
                # 筛图不能掩盖原始空间分组泄漏，即使泄漏样本本来会因重复而被剔除。
                for registry, identity in ((source_groups, sample.scene_group_id),
                                            (source_flights, sample.flight_id)):
                    if identity is not None and registry.setdefault(identity, split) != split:
                        raise ValueError("VISION_ASSEMBLY_ORIGINAL_SPLIT_LEAKAGE")
                if curate:
                    decision = curate_render_sample(root / split, sample, seen)
                    decisions.append({"batch": prefix, "split": split, **decision})
                    if not decision["retained"]:
                        continue
                payload = sample.model_dump(mode="json")
                for field, digest_field in (("image_relative_path", "image_sha256"),
                                           ("semantic_mask_relative_path", "semantic_mask_sha256")):
                    relative = payload[field]
                    if relative is None:
                        raise ValueError("VISION_ASSEMBLY_SEMANTIC_REQUIRED")
                    rewritten = f"{prefix}/{relative}"
                    transfers[f"{split}/{rewritten}"] = (root / split / relative,
                                                           payload[digest_field])
                    payload[field] = rewritten
                combined[split].append(LocalVisionTrainingSample.model_validate(payload))
        for artifacts in sources.values():
            for source, digest in artifacts.items():
                relative = f"sources/{prefix}/{source.relative_to(root).as_posix()}"
                transfers[relative] = source, digest
        capture_path = root / "capture-receipt.json"
        transfers[f"sources/{prefix}/capture-receipt.json"] = (
            capture_path, receipt_digest)
        lineage.append({"batch": prefix, "capture": capture_receipt})
    if (any(not values for values in combined.values())
            or sum(map(len, combined.values())) > 100_000):
        raise ValueError("VISION_ASSEMBLY_THREE_NONEMPTY_SPLITS_REQUIRED")
    require_disjoint_vision_splits(combined)
    for values in combined.values():
        if len({sample.image_sha256 for sample in values}) != len(values):
            raise ValueError("VISION_ASSEMBLY_DUPLICATE_IMAGE")
    balance = None
    if group_share_cap is not None:
        combined["training"], balance = cap_training_group_weights(
            combined["training"], group_share_cap)
    destination = destination.absolute()
    check_plain_plugin_path(destination)
    if any(destination == root or root in destination.parents for root in captures):
        raise ValueError("VISION_ASSEMBLY_OUTPUT_INSIDE_SOURCE")
    destination.mkdir(parents=True, exist_ok=False)
    used = 0
    index_path = destination / "files.jsonl"
    # 索引按行写入，数据扩大时不会触发单个 RPC JSON 的节点上限。
    with index_path.open("xb") as stream:
        for relative, (source, digest) in transfers.items():
            check_plain_plugin_path(source)
            size = source.stat().st_size
            if used + size + 8 * 1024**2 > max_bytes:
                raise ValueError("VISION_ASSEMBLY_STORAGE_BUDGET_REACHED")
            path = publish_asset_file(source, destination / relative, expected_sha256=digest,
                                      limit=16 * 1024**2)
            entry = {"path": relative, "sha256": digest, "size_bytes": path.stat().st_size}
            line = (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8")
            stream.write(line)
            used += entry["size_bytes"] + len(line)
        stream.flush()
        os.fsync(stream.fileno())
    summaries = {}
    for split, values in combined.items():
        summary = inspect_vision_split(destination / split, values)
        content = "".join(value.model_dump_json() + "\n" for value in values).encode("utf-8")
        used += len(content)
        if used + 8 * 1024**2 > max_bytes:
            raise ValueError("VISION_ASSEMBLY_STORAGE_BUDGET_REACHED")
        path = destination / split / "samples.jsonl"
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        summary["manifest_sha256"] = hash_plugin_file(path, limit=256 * 1024**2)
        summaries[split] = summary
    receipt = {"schema": "dronedream.assembled-vision-dataset.v1", "complete": True,
               "splits": summaries, "lineage": lineage,
               "files_index_sha256": hash_plugin_file(index_path, limit=256 * 1024**2),
               "indexed_files": len(transfers),
               "requires_training_preflight": True, "statistical_sufficiency_proven": False,
               "flight_qualification_granted": False}
    if balance is not None:
        receipt["training_group_loss_balance"] = balance
    if curate:
        curation_path = destination / "curation.jsonl"
        with curation_path.open("xb") as stream:
            for decision in decisions:
                line = (json.dumps(decision, sort_keys=True) + "\n").encode("utf-8")
                if used + len(line) + 8 * 1024**2 > max_bytes:
                    raise ValueError("VISION_ASSEMBLY_STORAGE_BUDGET_REACHED")
                stream.write(line)
                used += len(line)
            stream.flush()
            os.fsync(stream.fileno())
        receipt["curation"] = {"policy": CURATION_POLICY,
            "index_sha256": hash_plugin_file(curation_path, limit=256 * 1024**2),
            "input_samples": len(decisions), "retained_samples": sum(map(len, combined.values())),
            "rejected_samples": sum(not d["retained"] for d in decisions),
            "original_sources_preserved": True}
    receipt["data_bytes"] = used
    publish_runtime_json(destination / "assembly-receipt.json", receipt, replace_existing=False)
    return receipt


# 功能：
#   从显式白名单批次生成可移机训练数据，既不扫描用户文件夹也不自动改动划分。
# 输入：
#   命令行参数：一个或多个完整采集目录、新输出目录和字节预算。
# 输出：
#   exit_code：完整汇总返回零，后续仍须单独通过数据覆盖预检。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=10 * 1024**3)
    parser.add_argument("--curate", action="store_true")
    parser.add_argument("--cap-training-group-share", type=float)
    args = parser.parse_args()
    result = assemble(args.capture, args.output, args.max_bytes, curate=args.curate,
                      group_share_cap=args.cap_training_group_share)
    print(json.dumps({"complete": result["complete"], "splits": result["splits"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
