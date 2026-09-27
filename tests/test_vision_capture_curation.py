"""Native source, quality-only training and curation boundaries."""

import hashlib
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample
from dronedream_agent_core.training.vision_curation import curate_render_sample
from dronedream_agent_core.training.vision_manifest import inspect_vision_split
from dronedream_agent_core.training.vision_render_world import collada_texture_hashes
from dronedream_agent_core.training.vision_resolution import (
    inspect_model_resolution,
    mask_resolution_counts,
)


# 功能：
#   构造有实际像素和哈希的小样本，用于核验去重与失效曝光监督边界。
# 输入：
#   root、quality_only、varied：输出目录、是否仅训练质量、是否具有空间图案。
# 输出：
#   sample：可解析的测试样本，不代表原生仿真数据。
def image_sample(root, quality_only=False, varied=False):
    pixels = np.full((16, 16, 3), 5 if quality_only else 80, dtype=np.uint8)
    if varied:
        pixels[:8] += 70
    Image.fromarray(pixels).save(root / "rgb.png")
    Image.fromarray(np.ones((16, 16), dtype=np.uint8)).save(root / "mask.png")
    sample = LocalVisionTrainingSample(flight_id="fixture", map_sha256="a" * 64,
        image_relative_path="rgb.png",
        image_sha256=hashlib.sha256((root / "rgb.png").read_bytes()).hexdigest(),
        semantic_mask_relative_path="mask.png",
        semantic_mask_sha256=hashlib.sha256((root / "mask.png").read_bytes()).hexdigest(),
        traversability_target=1.0, scene_targets=[0.0] * 6,
        quality_targets=[float(quality_only), 0.0, 0.0, 0.0],
        perception_supervision_enabled=not quality_only)
    return sample


# 功能：
#   剔除无信息纯色墙面，保留可辨别画面；重复图只出现一次，原始文件仍存在。
# 输入：
#   tmp_path：仅含测试图像的隔离目录。
# 输出：
#   None：固定剔除规则和非破坏性行为均须成立。
def test_curation_removes_duplicates_and_blank_perception_without_deleting(tmp_path):
    sample = image_sample(tmp_path)
    seen = set()
    decision = curate_render_sample(tmp_path, sample, seen)
    assert decision["reason"] == "NO_SPATIAL_INFORMATION_SINGLE_CLASS"
    sample = image_sample(tmp_path, varied=True)
    assert curate_render_sample(tmp_path, sample, seen)["retained"] is True
    assert curate_render_sample(tmp_path, sample, seen)["reason"] == "EXACT_DUPLICATE_RGB"
    assert (tmp_path / "rgb.png").is_file() and (tmp_path / "mask.png").is_file()


# 功能：
#   严重欠曝图只贡献质量标签，不贡献语义或场景覆盖；正式预检仍拒绝重复。
# 输入：
#   tmp_path：真实解码边界的临时制品目录。
# 输出：
#   None：无法观测的隐藏标签不能被算作有效监督。
def test_quality_only_is_not_counted_as_perception_coverage(tmp_path):
    sample = image_sample(tmp_path, quality_only=True)
    assert curate_render_sample(tmp_path, sample, set())["retained"] is True
    report = inspect_vision_split(tmp_path, [sample])
    assert report["labelled_samples"] == 1
    assert report["perception_supervised_samples"] == 0
    assert report["semantic_pixels"] == [0] * 8
    assert report["auxiliary_label_counts"] == [0] * 6 + [1] * 4
    with pytest.raises(ValueError, match="DUPLICATE_IMAGE"):
        inspect_vision_split(tmp_path, [sample, sample])
    report = inspect_vision_split(tmp_path, [sample, sample], allow_duplicates=True)
    assert report["duplicate_images"] == 1
    payload = sample.model_dump()
    payload["quality_targets"] = [0.0] * 4
    with pytest.raises(ValueError, match="exposure failure"):
        LocalVisionTrainingSample.model_validate(payload)


# 功能：
#   外部纹理也必须被摘要绑定，且 COLLADA 不得逃出明确资源根目录。
# 输入：
#   tmp_path：网格相对引用使用的隔离目录。
# 输出：
#   None：完整纹理摘要和目录逃逸检查均须成立。
def test_collada_texture_dependencies_are_bound_and_confined(tmp_path):
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures/body.png").write_bytes(b"image")
    raw = (b'<COLLADA><library_images><image><init_from>../textures/body.png</init_from>'
           b'</image></library_images></COLLADA>')
    hashes = collada_texture_hashes("meshes/person.dae", raw, tmp_path)
    assert hashes == {"textures/body.png": hashlib.sha256(b"image").hexdigest()}
    with pytest.raises(ValueError):
        collada_texture_hashes("person.dae", raw, tmp_path)
    (tmp_path / "textures/body.png").write_bytes(b"changed")
    assert collada_texture_hashes("meshes/person.dae", raw, tmp_path) != hashes


# 功能：
#   加载一个待测命令行模块，不启动它的采集主函数。
# 输入：
#   name、monkeypatch：脚本名称和独立模块注册范围。
# 输出：
#   module：可直接测试纯检查函数的模块。
def command_module(name, monkeypatch):
    source = Path(__file__).parents[1] / "scripts" / (name + ".py")
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location(name + "_curation_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


# 功能：
#   计划摘要必须使用实际采集器的默认字段规范化，来源变化或未知布局直接拒绝。
# 输入：
#   tmp_path、monkeypatch：源计划目录及独立命令模块。
# 输出：
#   None：哈希与显式白名单检查均须成立。
def test_suite_plan_hash_uses_capture_normalization(tmp_path, monkeypatch):
    from dronedream_agent_core.hashing import sha256_json

    module = command_module("capture_vision_suite", monkeypatch)
    from capture_labelled_vision_views import checked_views

    root = tmp_path / "layout-1"
    root.mkdir()
    view = {"view_id": "one", "scene_group_id": "group", "split": "training",
            "position_m": [0.0, 0.0, 1.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    plan = {"schema": "dronedream.vision-view-plan.v1", "views": [view]}
    files = {"world.sdf": b"world", "labels.json": b"{}", "views.json": json.dumps(plan).encode()}
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    suite = {"schema": "dronedream.vision-suite-plan.v1", "complete": True, "layouts": [
        {"directory": root.name, "views": 1,
         "files_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}]}
    (tmp_path / "suite-plan.json").write_text(json.dumps(suite))
    job = module.checked_jobs([tmp_path], ["layout-1"])[0]
    expected = sha256_json([v.model_dump(mode="json") for v in checked_views(plan)])
    assert job["plan_sha256"] == expected
    with pytest.raises(ValueError, match="UNKNOWN_LAYOUT"):
        module.checked_jobs([tmp_path], ["missing"])
    (root / "world.sdf").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        module.checked_jobs([tmp_path], [])


# 功能：
#   环形网格用保守外包络保护相机，并拒绝未支持的网格格式，不能直接忽略未知对象。
# 输入：
#   monkeypatch：独立加载源计划模块。
# 输出：
#   None：网格几何的边界转换必须保守且显式。
def test_school_mesh_camera_envelope_does_not_drop_geometry(monkeypatch):
    module = command_module("prepare_school_route_views", monkeypatch)
    primitive = {"name": "ring", "uri": "ring.obj", "scale_x": 2.0, "scale_y": 1.0,
                 "scale_z": 1.0, "center_x": 5.0, "center_y": 2.0, "center_z": 1.0}
    semantics = {"collision_primitives": [], "visual_only_primitives": [primitive]}
    result = module.checked_camera_geometry(semantics, {"ring.obj": b"v -1 -2 -3\nv 1 2 3\n"})
    assert result[0]["size_x"] == 4 and result[0]["size_z"] == 6
    primitive["uri"] = "unknown.mesh"
    with pytest.raises(ValueError, match="MESH_UNSUPPORTED"):
        module.checked_camera_geometry(semantics, {})


# 功能：
#   原图中的单像素目标缩小后可能消失，覆盖检查必须明确记录，不能沿用原图计数。
# 输入：
#   无。
# 输出：
#   None：原图与模型尺寸的统计确实不同，非法像素仍在缩小前拒绝。
def test_resolution_check_detects_disappearing_targets_and_invalid_original():
    pixels = np.ones((360, 640), dtype=np.uint8)
    pixels[0, 0] = 5
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    original, resized = mask_resolution_counts(stream.getvalue(), 224, 128)
    assert original[5] == 1 and resized[5] == 0
    assert resized.sum() == 224 * 128
    pixels[0, 0] = 8
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    with pytest.raises(ValueError, match="CLASS_INVALID"):
        mask_resolution_counts(stream.getvalue(), 224, 128)


# 功能：
#   模型尺寸覆盖按空间组去重，并且失效曝光样本不能贡献可见场景数量。
# 输入：
#   tmp_path：隔离测试图像目录。
# 输出：
#   None：同组多图与质量专用图不会被统计成新的独立感知证据。
def test_resolution_coverage_counts_groups_and_excludes_quality_only(tmp_path):
    sample = image_sample(tmp_path, varied=True)
    report = inspect_model_resolution(tmp_path, [sample, sample])
    assert report["class_visible_images"][1] == 2
    assert report["class_visible_groups"][1] == 1
    quality = image_sample(tmp_path, quality_only=True)
    report = inspect_model_resolution(tmp_path, [quality])
    assert report["class_pixels"] == [0] * 8 and report["excluded_samples"] == 1
    with pytest.raises(ValueError, match="POLICY_INVALID"):
        inspect_model_resolution(tmp_path, [], width=True)


# 功能：
#   续传同时绑定标签分配与采集实现，不能把同世界的旧协议数据充作本次完成批次。
# 输入：
#   tmp_path、monkeypatch：隔离模块及完整回执的受控替身。
# 输出：
#   None：实现或标签任一变化必须拒绝复用。
def test_capture_resume_binds_implementation_and_semantic_assignment(tmp_path, monkeypatch):
    module = command_module("capture_vision_suite", monkeypatch)
    import assemble_vision_dataset as assembly

    receipt = {"world_sha256": "a" * 64, "camera_sha256": "b" * 64,
               "plan_sha256": "c" * 64, "labels_sha256": "d" * 64, "view_count": 2,
               "capture_implementation_sha256": {"capture.py": "e" * 64}}
    job = {"hashes": {"world.sdf": "a" * 64}, "plan_sha256": "c" * 64,
           "labels_sha256": "d" * 64, "views": 2}
    monkeypatch.setattr(assembly, "inspect_capture",
                        lambda *_args, **_kwargs: ({}, receipt, {}, ""))
    monkeypatch.setattr(module, "current_capture_implementation", lambda: {"capture.py": "e" * 64})
    assert module.verify_finished(tmp_path, job, "b" * 64) == receipt
    receipt["labels_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SOURCE_MISMATCH"):
        module.verify_finished(tmp_path, job, "b" * 64)
    receipt["labels_sha256"] = "d" * 64
    receipt["capture_implementation_sha256"] = {"capture.py": "f" * 64}
    with pytest.raises(ValueError, match="IMPLEMENTATION_MISMATCH"):
        module.verify_finished(tmp_path, job, "b" * 64)
