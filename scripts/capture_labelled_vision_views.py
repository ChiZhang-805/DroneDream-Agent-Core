"""Generate grouped static Gazebo RGB/labels, without PX4 or aircraft commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from dronedream_agent_core.contracts import StrictModel
from dronedream_agent_core.gazebo_adapter import _gazebo_image_png, _gazebo_semantic_label_png
from dronedream_agent_core.geometry_fixture_transport import (
    FixturePoseRequests,
    validate_fixture_command,
)
from dronedream_agent_core.geometry_motion_fixture import POSE_TOPIC
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_agent_core.runtime_control_io import publish_runtime_json, read_runtime_object
from dronedream_agent_core.simulation_graphics_lifetime import prepare_render_process_environment
from dronedream_agent_core.training.vision_corruptions import (
    RenderCorruption,
    corrupt_rendered_pair,
)
from dronedream_agent_core.training.vision_render_capture import RenderPairBuffer
from dronedream_agent_core.training.vision_render_world import (
    RGB_TOPIC,
    SEMANTIC_TOPIC,
    bind_render_resources,
    build_labelled_render_world,
)


class RenderView(StrictModel):
    view_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,39}$")
    scene_group_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,99}$")
    split: Literal["training", "validation", "test"]
    position_m: list[float] = Field(min_length=3, max_length=3)
    orientation_wxyz: list[float] = Field(min_length=4, max_length=4)
    setting: Literal["indoor", "outdoor", "unknown"] = "unknown"
    corruption: RenderCorruption = Field(default_factory=RenderCorruption)

    # 功能：
    #   校验有限米制坐标和单位四元数，拒绝无法作为光学姿态的扫描点。
    # 输入：
    #   self：解析后的视角配置。
    # 输出：
    #   self：经过几何约束检查的配置。
    @model_validator(mode="after")
    def validate_pose(self):
        validate_fixture_command({"position_m": self.position_m,
                                  "orientation_wxyz": self.orientation_wxyz})
        return self


# 功能：
#   检查视角唯一性和预先划定的空间组，禁止同一区域换视角后进入另一个数据划分。
# 输入：
#   payload：包含 views 列表的扫描计划。
# 输出：
#   views：已验证的视角序列，不包含任意命令或真实飞行权限。
def checked_views(payload):
    if (type(payload) is not dict or set(payload) != {"schema", "views"}
            or payload["schema"] != "dronedream.vision-view-plan.v1"
            or type(payload["views"]) is not list or not 1 <= len(payload["views"]) <= 10_000):
        raise ValueError("VISION_RENDER_PLAN_INVALID")
    views = [RenderView.model_validate(item) for item in payload["views"]]
    if len({view.view_id for view in views}) != len(views):
        raise ValueError("VISION_RENDER_DUPLICATE_VIEW_ID")
    groups = {}
    for view in views:
        if view.scene_group_id in groups and groups[view.scene_group_id] != view.split:
            raise ValueError("VISION_RENDER_SPATIAL_GROUP_LEAKAGE")
        groups[view.scene_group_id] = view.split
    return views


# 功能：
#   独占写出有限采集制品并刷盘；已有同名用户资料不覆盖。
# 输入：
#   path、payload：输出路径及本次生成的字节。
# 输出：
#   digest：实际写入内容的完整摘要。
def write_new(path, payload):
    check_plain_plugin_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    return digest


# 功能：
#   将原生同刻图像及实测姿态写入相应划分，标签来自渲染而非模型猜测或界面截图。
# 输入：
#   output、view、pair、metadata、run_id：新输出目录、扫描配置、观测及源身份。
# 输出：
#   sample、receipt、size：训练样本、原始渲染回执及本次图像总字节数。
def save_view(output, view, pair, metadata, run_id):
    from build_local_vision_dataset import _targets

    original_rgb = _gazebo_image_png(pair["rgb"])
    original_labels = _gazebo_semantic_label_png(pair["semantic"])
    rgb, labels, quality = corrupt_rendered_pair(original_rgb, original_labels, view.corruption)
    rgb_hash, mask_hash = hashlib.sha256(rgb).hexdigest(), hashlib.sha256(labels).hexdigest()
    rgb_relative, mask_relative = f"rgb/{view.view_id}.png", f"semantic/{view.view_id}.png"
    root = output / view.split
    raw_rgb_hash = write_new(output / "raw" / (view.view_id + "-rgb.png"), original_rgb)
    raw_mask_hash = write_new(output / "raw" / (view.view_id + "-semantic.png"), original_labels)
    write_new(root / rgb_relative, rgb)
    write_new(root / mask_relative, labels)
    receipt = {**metadata, "view": view.model_dump(mode="json"),
               "simulation_ns": pair["simulation_ns"], "measured_pose": pair["measured_pose"],
               "rgb_sha256": rgb_hash, "semantic_sha256": mask_hash,
               "raw_rgb_sha256": raw_rgb_hash, "raw_semantic_sha256": raw_mask_hash,
               "quality_label_source": "controlled-static-render-no-lens-effects",
               "source_kind": "rendered-view", "flight_qualification_granted": False}
    targets = _targets(root / rgb_relative, root / mask_relative,
                       image_sha256=rgb_hash, mask_sha256=mask_hash)
    targets.pop("class_pixel_ratios")
    targets["quality_targets"][2:] = [quality["blurred"], quality["occluded"]]
    targets["quality_target_weights"][2:] = [1.0, 1.0]
    if view.setting != "unknown":
        targets["scene_targets"][:2] = [float(view.setting == "indoor"),
                                         float(view.setting == "outdoor")]
        targets["scene_target_weights"][:2] = [1.0, 1.0]
    sample = LocalVisionTrainingSample(
        flight_id=f"render-{run_id}-{view.view_id}", source_kind="rendered-view",
        scene_group_id=view.scene_group_id, map_sha256=metadata["world_sha256"],
        image_relative_path=rgb_relative, image_sha256=rgb_hash,
        semantic_mask_relative_path=mask_relative, semantic_mask_sha256=mask_hash,
        source_record_sha256=sha256_json(receipt), rgb_semantic_time_offset_seconds=0.0,
        perception_supervision_enabled=not any(targets["quality_targets"][:2]), **targets)
    size = len(rgb) + len(labels) + len(original_rgb) + len(original_labels)
    publish_runtime_json(output / "views" / (view.view_id + ".json"),
                         {"sample": sample.model_dump(mode="json"), "source": receipt},
                         replace_existing=False)
    return sample, receipt, size


# 功能：
#   1. 在独立仿真分区生成有来源的视觉监督，有限等待并核对实际位姿。
#   2. 异常时回收自有进程，保留采集现场；只有完整关闭后才写成功回执。
# 输入：
#   命令行参数：静态地图、标签分配、真实 RGB 相机、分组视角及新输出目录。
# 输出：
#   exit_code：采集完成为零；缺帧、错位或超限均抛出错误。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("world", "labels", "camera", "views", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resource-path", type=Path, action="append", default=[])
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if sys.platform != "linux" or not 1024**2 <= args.max_bytes <= 20 * 1024**3:
        raise ValueError("VISION_RENDER_REQUIRES_LINUX_AND_BOUNDED_STORAGE")
    views = checked_views(read_runtime_object(args.views, maximum_bytes=8 * 1024**2))
    world_bytes = read_plugin_file(args.world, limit=32 * 1024**2)
    labels = read_runtime_object(args.labels, maximum_bytes=8 * 1024**2)
    camera_bytes = read_plugin_file(args.camera, limit=1024**2)
    derived, metadata = build_labelled_render_world(world_bytes, labels, camera_bytes)
    derived, resource_hashes = bind_render_resources(derived, args.world.parent)
    metadata["derived_sha256"] = hashlib.sha256(derived).hexdigest()
    metadata["render_resource_sha256"] = resource_hashes
    metadata["labels_sha256"] = sha256_json(labels)
    metadata["plan_sha256"] = sha256_json([view.model_dump(mode="json") for view in views])
    implementation = [Path(__file__), Path(__file__).with_name("build_local_vision_dataset.py")]
    for component in (RenderPairBuffer, build_labelled_render_world, corrupt_rendered_pair,
                      LocalVisionTrainingSample, FixturePoseRequests):
        implementation.append(Path(sys.modules[component.__module__].__file__))
    metadata["capture_implementation_sha256"] = {
        path.name: hashlib.sha256(read_plugin_file(path, limit=2 * 1024**2)).hexdigest()
        for path in implementation}
    output = args.output.absolute()
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "render-world.sdf", derived)
    run_id = uuid.uuid4().hex
    partition = "dronedream-geometry-fixture-" + run_id
    os.environ["GZ_PARTITION"] = partition
    os.environ["GZ_IP"] = "127.0.0.1"
    sys.path.append("/usr/lib/python3/dist-packages")
    from gz.msgs10.image_pb2 import Image
    from gz.msgs10.pose_pb2 import Pose
    from gz.transport13 import Node

    node, buffer = Node(), RenderPairBuffer()
    topics = []
    process = commands = None
    samples = {"training": [], "validation": [], "test": []}
    records, used = [], 0
    try:
        for topic, kind, callback in (
            (RGB_TOPIC, Image, lambda msg: buffer.image("rgb", msg)),
            (SEMANTIC_TOPIC + "/labels_map", Image, lambda msg: buffer.image("semantic", msg)),
            (POSE_TOPIC, Pose, buffer.measured_pose),
        ):
            if not node.subscribe(kind, topic, callback):
                raise RuntimeError("VISION_RENDER_SUBSCRIPTION_FAILED")
            topics.append(topic)
        env = dict(os.environ)
        env["GZ_SIM_RESOURCE_PATH"] = ":".join(str(path.absolute()) for path in
                                                 [args.world.parent, *args.resource_path])
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        env["GALLIUM_DRIVER"] = "llvmpipe"
        env, graphics = prepare_render_process_environment(env)
        metadata["graphics"] = graphics
        # 先完成轻量通信进程握手，再启动高负载渲染，避免两个冷启动互相争抢资源。
        # 初始化最长 60 秒；下方的逐帧对齐、连续稳定和命令期限保持原值。
        commands = FixturePoseRequests(partition, "vision_dataset_world",
                                       startup_timeout_seconds=60.0)
        metadata["fixture_startup_seconds"] = commands.startup_elapsed_seconds
        print(f"fixture ready after {commands.startup_elapsed_seconds:.3f}s", flush=True)
        with (output / "gazebo.log").open("xb") as log:
            process = subprocess.Popen(["gz", "sim", "-r", "-s", "--headless-rendering",
                                        str(output / "render-world.sdf")], env=env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, cwd=output)
            for view in views:
                wanted = {"position_m": view.position_m, "orientation_wxyz": view.orientation_wxyz}
                # 有限重试只用于未启动完成的无动力相机架服务，不影响任何飞机。
                acknowledgement = None
                service_deadline = time.monotonic() + 90.0
                while time.monotonic() < service_deadline:
                    if process.poll() is not None:
                        raise RuntimeError("VISION_RENDER_SIMULATOR_EXITED")
                    acknowledgement = commands.request(wanted)
                    if acknowledgement["acknowledged"]:
                        break
                    time.sleep(0.2)
                if not acknowledgement or not acknowledgement["acknowledged"]:
                    raise RuntimeError("VISION_RENDER_RIG_POSE_NOT_ACCEPTED")
                pair = buffer.wait_pair(wanted, buffer.clear())
                if used + 20 * 1024**2 > args.max_bytes:
                    raise ValueError("VISION_RENDER_STORAGE_BUDGET_REACHED")
                sample, receipt, size = save_view(output, view, pair, metadata, run_id)
                samples[view.split].append(sample)
                records.append(receipt)
                used += size
                print(f"render views={len(records)}/{len(views)} bytes={used}", flush=True)
    finally:
        for topic in topics:
            with suppress(Exception):
                node.unsubscribe(topic)
        if commands is not None:
            with suppress(Exception):
                commands.close()
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    log_bytes = read_plugin_file(output / "gazebo.log", limit=8 * 1024**2)
    if b"[Err]" in log_bytes:
        raise RuntimeError("VISION_RENDER_ENGINE_ERROR_SEE_PRESERVED_LOG")
    # 帧已保存也不能忽略资源加载失败或采集中途资源变化。
    _, current_resource_hashes = bind_render_resources(
        build_labelled_render_world(world_bytes, labels, camera_bytes)[0], args.world.parent)
    if current_resource_hashes != resource_hashes:
        raise ValueError("VISION_RENDER_RESOURCE_CHANGED")
    hashes = {}
    for split, values in samples.items():
        if values:
            content = "".join(value.model_dump_json() + "\n" for value in values).encode("utf-8")
            hashes[split] = write_new(output / split / "samples.jsonl", content)
    write_new(output / "render-records.jsonl", "".join(
        json.dumps(record, allow_nan=False) + "\n" for record in records).encode("utf-8"))
    publish_runtime_json(output / "capture-receipt.json", {**metadata, "complete": True,
        "view_count": len(records), "manifest_sha256": hashes, "image_bytes": used,
        "flight_qualification_granted": False}, replace_existing=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
