"""Assembly keeps exact rendered provenance and never upgrades data to flight evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample
from dronedream_agent_core.training.vision_manifest import read_vision_manifest


# 功能：
#   构造三组可解码、逐视角来源可追溯的小夹具，不代表飞行或数据覆盖合格。
# 输入：
#   root、pattern：隔离采集目录及是否生成有空间信息的图案。
# 输出：
#   root：包含原图、语义、来源及完整采集回执的目录。
def capture_fixture(root, pattern=False):
    hashes = {}
    for index, split in enumerate(("training", "validation", "test")):
        base = root / split
        for name in ("rgb", "semantic"):
            (base / name).mkdir(parents=True)
        for name in ("raw", "views"):
            (root / name).mkdir(exist_ok=True)
        image_path, mask_path = base / "rgb" / f"{split}.png", base / "semantic" / f"{split}.png"
        with Image.new("RGB", (8, 8), (80, index * 30, 180)) as rgb:
            if pattern:
                rgb.paste((30, 180, index * 30), (0, 0, 4, 8))
            rgb.save(image_path)
        Image.new("L", (8, 8), 1).save(mask_path)
        rgb_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        mask_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        (root / "raw" / f"{split}-rgb.png").write_bytes(image_path.read_bytes())
        (root / "raw" / f"{split}-semantic.png").write_bytes(mask_path.read_bytes())
        source = {"view": {"view_id": split, "split": split, "scene_group_id": split},
                  "world_sha256": "1" * 64, "camera_sha256": "2" * 64,
                  "rgb_sha256": rgb_hash, "semantic_sha256": mask_hash,
                  "raw_rgb_sha256": rgb_hash, "raw_semantic_sha256": mask_hash}
        sample = LocalVisionTrainingSample(flight_id=split, source_kind="rendered-view",
            scene_group_id=split, map_sha256="1" * 64,
            image_relative_path=f"rgb/{split}.png", image_sha256=rgb_hash,
            semantic_mask_relative_path=f"semantic/{split}.png", semantic_mask_sha256=mask_hash,
            source_record_sha256=sha256_json(source), traversability_target=0.0,
            scene_targets=[0.0] * 6, quality_targets=[0.0] * 4,
            rgb_semantic_time_offset_seconds=0.0)
        (root / "views" / f"{split}.json").write_text(
            json.dumps({"sample": sample.model_dump(mode="json"), "source": source}),
            encoding="utf-8")
        manifest = base / "samples.jsonl"
        manifest.write_text(sample.model_dump_json() + "\n", encoding="utf-8")
        hashes[split] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt = {"complete": True, "source_kind": "rendered-view", "physical_flight_evidence": False,
               "view_count": 3, "world_sha256": "1" * 64, "camera_sha256": "2" * 64,
               "manifest_sha256": hashes}
    (root / "capture-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root


# 功能：
#   加载当前汇总脚本，不调用 CLI 主入口或额外进程。
# 输入：
#   无。
# 输出：
#   module：待测试工具模块。
def tool():
    path = Path(__file__).parents[1] / "scripts/assemble_vision_dataset.py"
    spec = importlib.util.spec_from_file_location("assemble_vision_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 功能：
#   对实际小规模组装产物核验完整传输，再注入漏传、额外文件和内容篡改。
# 输入：
#   tmp_path、monkeypatch：独立样本目录与脚本导入范围。
#   fault：要注入的传输故障，正常分支同时核对记录的文件数量。
# 输出：
#   None：正常产物通过，损坏或混入的旧文件均被拒绝。
@pytest.mark.parametrize("fault", [None, "extra", "missing", "modified", "wrong-receipt"])
def test_full_dataset_transfer_verification(tmp_path, monkeypatch, fault):
    source = capture_fixture(tmp_path / "capture", pattern=True)
    destination = tmp_path / "assembled"
    assembled = tool().assemble([source], destination, curate=True)
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("vision_transfer_test",
                                                scripts / "verify_vision_dataset.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    digest = hashlib.sha256((destination / "assembly-receipt.json").read_bytes()).hexdigest()
    if fault is None:
        result = verifier.verify(destination, digest)
        assert result["verified"] and not result["training_started"]
        assert result["files"] == assembled["indexed_files"] + 6
        return
    if fault == "extra":
        (destination / "old-model.py").write_text("not allowed", encoding="utf-8")
    elif fault == "missing":
        (destination / "sources/batch-000/raw/training-rgb.png").unlink()
    elif fault == "modified":
        (destination / "training/batch-000/rgb/training.png").write_bytes(b"corrupt")
    else:
        digest = "0" * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        verifier.verify(destination, digest)


# 功能：
#   汇总保持来源摘要且不覆盖输入；再次使用输出目录必须失败。
# 输入：
#   tmp_path：测试专属目录。
# 输出：
#   None：不返回业务数据。
def test_assembly_preserves_raw_origin_and_is_exclusive(tmp_path):
    source = capture_fixture(tmp_path / "capture")
    destination = tmp_path / "assembled"
    result = tool().assemble([source], destination, group_share_cap=1.0)
    assert result["complete"] and result["requires_training_preflight"]
    assert set(result["training_group_loss_balance"]["groups"]) == {"training"}
    assert not result["flight_qualification_granted"]
    for split in ("training", "validation", "test"):
        before, _ = read_vision_manifest(source / split / "samples.jsonl")
        after, _ = read_vision_manifest(destination / split / "samples.jsonl")
        assert before[0].source_record_sha256 == after[0].source_record_sha256
        assert after[0].image_relative_path == f"batch-000/rgb/{split}.png"
        assert (destination / "sources/batch-000/raw" / f"{split}-rgb.png").is_file()
    with pytest.raises(FileExistsError):
        tool().assemble([source], destination)


# 功能：
#   原始帧、完成回执、来源身份任意一项失真，均在汇总目录创建前拒绝。
# 输入：
#   tmp_path、mutation：隔离目录和需要注入的损坏类型。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mutation", ["raw", "complete", "source"])
def test_assembly_rejects_incomplete_or_changed_capture(tmp_path, mutation):
    source = capture_fixture(tmp_path / "capture")
    if mutation == "raw":
        (source / "raw/training-rgb.png").write_bytes(b"changed")
    elif mutation == "complete":
        path = source / "capture-receipt.json"
        payload = json.loads(path.read_text())
        payload["complete"] = False
        path.write_text(json.dumps(payload))
    else:
        path = source / "views/training.json"
        payload = json.loads(path.read_text())
        payload["source"]["camera_sha256"] = "3" * 64
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        tool().assemble([source], tmp_path / "output")
    assert not (tmp_path / "output").exists()


# 功能：
#   采集回执在读取后被替换时，不能以旧内容检查而把新内容复制进完整汇总。
# 输入：
#   tmp_path、monkeypatch：隔离采集和确定性的文件替换注入点。
# 输出：
#   None：来源变化须拒绝，且不发布完成回执。
def test_assembly_binds_receipt_to_bytes_actually_parsed(tmp_path, monkeypatch):
    source = capture_fixture(tmp_path / "capture")
    module = tool()
    original = module.inspect_vision_split
    changed = [False]

    # 功能：
    #   在回执解析完成、图片检查开始时替换完成标记，复现检查与复制间的变化。
    # 输入：
    #   root、samples：当前集合和样本。
    # 输出：
    #   result：图像本身的真实校验结果。
    def inspect_and_replace(root, samples, **options):
        result = original(root, samples, **options)
        if not changed[0]:
            path = source / "capture-receipt.json"
            payload = json.loads(path.read_text())
            payload["complete"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            changed[0] = True
        return result

    monkeypatch.setattr(module, "inspect_vision_split", inspect_and_replace)
    destination = tmp_path / "assembled"
    with pytest.raises(ValueError, match="(CHANGED|HASH|[Hh]ash|[Dd]igest)"):
        module.assemble([source], destination)
    assert not (destination / "assembly-receipt.json").exists()


# 功能：
#   完整演练固定筛图和组装，核对原始数据、逐图决定及实际字节预算一起保留。
# 输入：
#   tmp_path：三集合有图案的合成来源目录。
# 输出：
#   None：组装回执不得遗漏筛图索引字节，保留来源不代表正式覆盖合格。
def test_curated_assembly_preserves_decisions_and_counts_every_payload_byte(tmp_path):
    source = capture_fixture(tmp_path / "capture", pattern=True)
    destination = tmp_path / "assembled"
    receipt = tool().assemble([source], destination, curate=True)
    assert receipt["curation"]["retained_samples"] == 3
    assert receipt["curation"]["rejected_samples"] == 0
    assert receipt["curation"]["original_sources_preserved"]
    lines = (destination / "curation.jsonl").read_text().splitlines()
    decisions = [json.loads(line) for line in lines]
    assert len(decisions) == 3 and all(record["retained"] for record in decisions)
    payload_bytes = sum(path.stat().st_size for path in destination.rglob("*")
                        if path.is_file() and path.name != "assembly-receipt.json")
    assert receipt["data_bytes"] == payload_bytes


# 功能：
#   即使下一批所有同图都会被去重，也不得把原始空间组的跨集合泄漏藏起来。
# 输入：
#   tmp_path、monkeypatch：有效采集夹具和第二批原始分组的故障注入。
# 输出：
#   None：泄漏在输出目录创建之前被拒绝。
def test_curation_cannot_hide_original_group_leakage(tmp_path, monkeypatch):
    source = capture_fixture(tmp_path / "capture", pattern=True)
    module = tool()
    inspected = module.inspect_capture(source)
    # 第一批真实检查完整；第二批只在测试中注入与已读取组冲突的划分。
    second = inspected[0]["training"][0]
    snapshots = iter([inspected, ({"test": [second]}, *inspected[1:])])
    monkeypatch.setattr(module, "inspect_capture", lambda *_args, **_kwargs: next(snapshots))
    with pytest.raises(ValueError, match="ORIGINAL_SPLIT_LEAKAGE"):
        module.assemble([source, tmp_path / "second"], tmp_path / "assembled", curate=True)
    assert not (tmp_path / "assembled").exists()
