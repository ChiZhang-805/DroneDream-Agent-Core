"""Bounded local vision dataset admission and decoding, using synthetic PNGs."""

import hashlib
import json

import numpy as np
import pytest
from test_local_vision_dataset_builder import _flight, _label_map, _png

from scripts import build_local_vision_dataset as builder


# 功能：
#   生成一份完整的本地合成飞行数据，供训练准入故障注入使用。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   root：含 RGB、语义掩码、摘要和任务回执的数据集目录。
def _dataset(tmp_path):
    digest = _label_map(tmp_path / "labels.json")
    root = tmp_path / "flight" / "dataset"
    _flight(root, "test-flight", digest, 40)
    return root


# 功能：
#   持久化或后台写入失败的记录链不能被训练构建器接纳。
# 输入：
#   tmp_path：隔离目录。
#   updates：待注入的失败标记。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("updates", [
    {"issue_code": "MULTIMODAL_DATASET_WRITE_FAILED:OSError"},
    {"writer_complete": False},
    {"writer_complete": 1},
    {"writer_issue": "DATASET_WRITER_DRAIN_TIMEOUT"},
    {"bytes_accounting_complete": False},
])
def test_failed_recording_not_admitted(tmp_path, updates):
    root = _dataset(tmp_path)
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(updates)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="incomplete|failed|invalid"):
        builder._flight_records(root, allow_failed_source=True)


# 功能：
#   重复 JSON 键和以布尔值冒充记录条数不能绕过摘要绑定。
# 输入：
#   tmp_path：隔离目录。
#   duplicate：是否使用重复键而非布尔条数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("duplicate", [False, True])
def test_summary_rejects_ambiguous_values(tmp_path, duplicate):
    root = _dataset(tmp_path)
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    if duplicate:
        text = json.dumps(summary).replace(
            '"record_count": 1', '"record_count": 9, "record_count": 1'
        )
    else:
        summary["record_count"] = True
        text = json.dumps(summary)
    path.write_text(text)
    with pytest.raises(ValueError):
        builder._flight_records(root, allow_failed_source=False)


# 功能：
#   图像软链接即使解析后仍处于数据集内部，也不能通过普通制品检查。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：不返回业务数据。
def test_artifact_symlink_rejected_before_resolution(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    target = root / "image.png"
    target.write_bytes(b"opaque test artifact")
    linked = root / "linked.png"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links unavailable for this test account")
    with pytest.raises(ValueError, match="LINK|escapes"):
        builder._resolved_artifact(
            root, linked.name, hashlib.sha256(target.read_bytes()).hexdigest()
        )


# 功能：
#   RGB 与标签的纵横比不同或标签是 RGB 色彩图时，不能生成虚假的监督目标。
# 输入：
#   tmp_path：隔离目录。
#   kind：不匹配的几何或标签模式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["aspect", "color"])
def test_targets_reject_incompatible_semantics(tmp_path, kind):
    rgb = tmp_path / "rgb.png"
    mask = tmp_path / "mask.png"
    rgb.write_bytes(_png(np.zeros((4, 8, 3), dtype=np.uint8)))
    shape = (8, 4) if kind == "aspect" else (4, 8, 3)
    mask.write_bytes(_png(np.ones(shape, dtype=np.uint8)))
    with pytest.raises(ValueError, match="aspect|mode"):
        builder._targets(rgb, mask)


# 功能：
#   派生监督必须核对实际解码字节与样本摘要，不能先验一次后解码被替换的图像。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：不返回业务数据。
def test_targets_bind_decoded_bytes(tmp_path):
    rgb, mask = tmp_path / "rgb.png", tmp_path / "mask.png"
    rgb.write_bytes(_png(np.zeros((4, 8, 3), dtype=np.uint8)))
    mask.write_bytes(_png(np.ones((4, 8), dtype=np.uint8)))
    with pytest.raises(ValueError, match="hash"):
        builder._targets(rgb, mask, image_sha256="0" * 64, mask_sha256="1" * 64)


# 功能：
#   无效比例、隐式布尔值和重复飞行列表必须明确拒绝，不能生成含糊的数据集划分。
# 输入：
#   ids：输入飞行标识。
#   fraction：划分比例。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("ids,fraction", [
    (["a", "b"], float("nan")), (["a", "b"], True),
    (["a", "b"], 2), (["a", "a"], 0.2),
])
def test_split_rejects_ambiguous_inputs(ids, fraction):
    with pytest.raises(ValueError):
        builder._validation_flights(ids, [], fraction)


# 功能：
#   成功追加后缺少最后换行表示未完成记录，不得当作完整飞行数据。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：不返回业务数据。
def test_incomplete_final_record_rejected(tmp_path):
    root = _dataset(tmp_path)
    path = root / "records.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="incomplete|bind"):
        builder._flight_records(root, allow_failed_source=False)
