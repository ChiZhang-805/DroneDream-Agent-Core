"""Preparation boundaries; synthetic fixtures never qualify production models."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from clock_fixtures import isolate_monotonic
from PIL import Image

from dronedream_agent_core.local_vision_training import LocalVisionTrainingSample
from dronedream_agent_core.training.vision_manifest import (
    inspect_vision_split,
    require_disjoint_vision_splits,
)
from dronedream_agent_core.training.vision_render_capture import RenderPairBuffer


# 功能：
#   构造带真实时间布局的原生消息替身，供配对和倒退时钟边界测试。
# 输入：
#   stamp：纳秒来源时间。
# 输出：
#   message：两像素 RGB 消息，不代表实际飞行。
def frame(stamp):
    message = NS(header=NS(stamp=NS(sec=stamp // 10**9, nsec=stamp % 10**9)),
                 width=2, height=1, step=6, pixel_format_type=3, data=b"\x10" * 6)
    return message


# 功能：
#   为相机架写入同一来源时钟的独立实测位置。
# 输入：
#   buffer、stamp、x：缓存、纳秒时间和实际横坐标。
# 输出：
#   None：不返回业务数据。
def pose(buffer, stamp, x=0.0):
    from dronedream_agent_core.geometry_motion_fixture import RIG_NAME

    message = frame(stamp)
    message.name = RIG_NAME
    message.position = NS(x=x, y=0.0, z=1.0)
    message.orientation = NS(w=1.0, x=0.0, y=0.0, z=0.0)
    buffer.measured_pose(message)


# 功能：
#   只有稳定实测姿态之后同刻 RGB/标签可以返回，旧帧或偏移标签均不得配对。
# 输入：
#   monkeypatch：控制等待期间的新传感器事件，不使用真实睡眠。
#   mismatch：是否注入一纳秒的标签错位。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mismatch", [False, True])
def test_render_pair_uses_stable_measured_pose_and_exact_clock(monkeypatch, mismatch):
    import dronedream_agent_core.training.vision_render_capture as capture

    buffer = RenderPairBuffer()
    wanted = {"position_m": [0.0, 0.0, 1.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    pose(buffer, 1_000_000_000)
    clock = [0.0]
    isolate_monotonic(monkeypatch, capture, lambda: clock[0])

    # 功能：
    #   在第二次轮询送达稳定姿态及新帧，旧的姿态值不会冒充新观测。
    # 输入：
    #   seconds：等待器请求的时间步长。
    # 输出：
    #   None：不返回业务数据。
    def advance(seconds):
        clock[0] += seconds
        for stamp in range(1_010_000_000, 1_200_000_001, 10_000_000):
            if stamp > buffer.last_time["pose"]:
                pose(buffer, stamp)
        buffer.image("rgb", frame(1_200_000_000))
        buffer.image("semantic", frame(1_200_000_000 + int(mismatch)))

    monkeypatch.setattr(capture.time, "sleep", advance)
    if mismatch:
        with pytest.raises(TimeoutError, match="ALIGNED_PAIR"):
            buffer.wait_pair(wanted, -1, timeout=0.1)
    else:
        result = buffer.wait_pair(wanted, -1, timeout=0.1)
        assert result["simulation_ns"] == 1_200_000_000
        assert result["measured_pose"] == wanted


# 功能：
#   缓存有界、重复帧不累积、时钟倒退立即失败，服务成功不替代错误实测姿态。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_render_buffer_rejects_regression_and_command_only_pose():
    buffer = RenderPairBuffer()
    for stamp in range(20):
        buffer.image("rgb", frame(stamp))
    buffer.image("rgb", frame(19))
    assert len(buffer.frames["rgb"]) == 8
    assert buffer.clear() == 19 and not buffer.frames["rgb"]
    buffer.image("rgb", frame(18))
    wanted = {"position_m": [0.0, 0.0, 1.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="CLOCK_REGRESSED"):
        buffer.wait_pair(wanted, -1)
    buffer = RenderPairBuffer()
    pose(buffer, 20, x=1.0)
    with pytest.raises(TimeoutError):
        buffer.wait_pair(wanted, -1, timeout=0.02)


# 功能：
#   原生渲染图可晚于物理线程送达，只要拍摄时刻仍被连续稳定实测位姿夹住就能接纳。
# 输入：
#   无；使用已复现的约 156 ms 发布延迟，图像与标签仍为同一采样时刻。
# 输出：
#   None：不会把静态世界的渲染延迟误判为位姿不稳定。
def test_static_render_uses_capture_time_not_delivery_delay():
    buffer = RenderPairBuffer()
    for stamp in range(1_000_000_000, 1_361_000_000, 12_000_000):
        pose(buffer, stamp)
    buffer.image("rgb", frame(1_200_000_000))
    buffer.image("semantic", frame(1_200_000_000))
    wanted = {"position_m": [0.0, 0.0, 1.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    result = buffer.wait_pair(wanted, -1, timeout=0.1)
    assert result["simulation_ns"] == 1_200_000_000
    assert result["measured_pose"] == wanted


# 功能：
#   写入可解码且绑定摘要的极小合成样本，保留未知监督为缺口。
# 输入：
#   root、split：测试目录和来源划分名。
# 输出：
#   sample：只供结构预检的合成视觉样本。
def sample_fixture(root, split):
    root.mkdir()
    image = root / "image.png"
    Image.new("RGB", (8, 8), (len(split), ord(split[0]), 30)).save(image)
    sample = LocalVisionTrainingSample(flight_id=split, map_sha256="1" * 64,
        image_relative_path="image.png",
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        traversability_target=0.0, scene_targets=[0.0] * 6, quality_targets=[0.0] * 4,
        scene_target_weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        quality_target_weights=[1.0, 0.0, 0.0, 0.0])
    (root / "samples.jsonl").write_text(sample.model_dump_json() + "\n", encoding="utf-8")
    return sample


# 功能：
#   真正解码输入后报告缺失语义及质量监督，不能因文件存在就允许付费训练。
# 输入：
#   tmp_path、monkeypatch：测试目录与独立脚本命名空间。
# 输出：
#   None：不返回业务数据。
def test_preflight_reports_real_data_gaps(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/preflight_vision_data.py"
    spec = importlib.util.spec_from_file_location("preflight_data_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    roots = {split: tmp_path / split for split in ("training", "validation", "test")}
    for split, root in roots.items():
        sample_fixture(root, split)
    report = module.preflight(roots)
    assert report["ready_for_training"] is False
    assert "training:SEMANTIC_LABELS_MISSING" in report["issues"]
    assert "test:AUXILIARY_LABEL_COVERAGE_INCOMPLETE" in report["issues"]
    assert report["flight_qualification_granted"] is False


# 功能：
#   改名不能隐藏同图或同区域跨集合泄漏，实际制品修改也必须在预检中发现。
# 输入：
#   tmp_path：输入制品隔离目录。
# 输出：
#   None：不返回业务数据。
def test_manifest_detects_content_and_spatial_leakage(tmp_path):
    root = tmp_path / "data"
    sample = sample_fixture(root, "training")
    second = sample.model_copy(update={"flight_id": "validation"})
    with pytest.raises(ValueError, match="SPLIT_LEAKAGE"):
        require_disjoint_vision_splits({"training": [sample], "test": [second]})
    sample.scene_group_id = "corridor"
    second = sample.model_copy(update={"flight_id": "other", "image_sha256": "2" * 64,
                                       "map_sha256": "3" * 64})
    with pytest.raises(ValueError, match="SPLIT_LEAKAGE"):
        require_disjoint_vision_splits({"training": [sample], "test": [second]})
    (root / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError):
        inspect_vision_split(root, [sample])


# 功能：
#   姿态在两次轮询之间离开目标后返回、长时间断流或仅有旧帧时不能冒充稳定配对。
# 输入：
#   monkeypatch、fault：独立时钟及要注入的姿态故障。
# 输出：
#   None：所有不满足连续实测条件的情况均须超时。
@pytest.mark.parametrize("fault", ["excursion", "gap", "stale-image"])
def test_render_pair_cannot_hide_between_poll_pose_changes(monkeypatch, fault):
    import dronedream_agent_core.training.vision_render_capture as capture

    buffer = RenderPairBuffer()
    wanted = {"position_m": [0.0, 0.0, 1.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    pose(buffer, 1_000_000_000)
    clock = [0.0]
    isolate_monotonic(monkeypatch, capture, lambda: clock[0])

    # 功能：
    #   一次性送达多个姿态，模拟回调快于主循环；之后只推进等待时钟。
    # 输入：
    #   seconds：等待器的轮询间隔。
    # 输出：
    #   None：不执行真实仿真或睡眠。
    def advance(seconds):
        clock[0] += seconds
        if buffer.last_time["pose"] != 1_000_000_000:
            return
        if fault == "gap":
            pose(buffer, 1_200_000_000)
        else:
            for stamp in range(1_010_000_000, 1_200_000_001, 10_000_000):
                pose(buffer, stamp, x=1.0 if fault == "excursion" and stamp == 1_190_000_000
                     else 0.0)
        frame_stamp = 1_100_000_000 if fault == "stale-image" else 1_200_000_000
        buffer.image("rgb", frame(frame_stamp))
        buffer.image("semantic", frame(frame_stamp))
        if fault == "stale-image":
            # 帧早于有界位姿历史，无法证明拍摄时刻稳定；不是只按发布延迟判旧。
            for stamp in range(1_210_000_000, 4_600_000_001, 10_000_000):
                pose(buffer, stamp)

    monkeypatch.setattr(capture.time, "sleep", advance)
    with pytest.raises(TimeoutError, match="ALIGNED_PAIR"):
        buffer.wait_pair(wanted, -1, timeout=0.1)


# 功能：
#   预检期间清单改变时拒绝发布原来样本的就绪报告。
# 输入：
#   tmp_path、monkeypatch：隔离数据和在检查阶段注入的源文件变化。
# 输出：
#   None：来源变化必须显式报错。
def test_preflight_rechecks_manifest_after_inspection(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/preflight_vision_data.py"
    spec = importlib.util.spec_from_file_location("preflight_drift_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    roots = {split: tmp_path / split for split in ("training", "validation", "test")}
    for split, root in roots.items():
        sample_fixture(root, split)
    original = module.inspect_vision_split

    # 功能：
    #   在最后一个集合检查完成时修改第一份清单，复现长预检窗口中的文件漂移。
    # 输入：
    #   root、samples：实际被检查的集合和样本。
    # 输出：
    #   report：修改之前的真实检查结果。
    def inspect_then_change(root, samples):
        report = original(root, samples)
        if root == roots["test"]:
            with (roots["training"] / "samples.jsonl").open("ab") as stream:
                stream.write(b"\n")
        return report

    monkeypatch.setattr(module, "inspect_vision_split", inspect_then_change)
    with pytest.raises(ValueError, match="MANIFEST_CHANGED"):
        module.preflight(roots)
