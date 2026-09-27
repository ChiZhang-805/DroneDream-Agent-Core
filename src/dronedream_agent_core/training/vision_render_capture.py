"""Synchronize static rig RGB, semantic labels and independent measured pose."""

from __future__ import annotations

import math
import time
from collections import deque
from threading import Lock

from dronedream_agent_core.geometry_fixture_transport import validate_fixture_command
from dronedream_agent_core.geometry_motion_fixture import RIG_NAME
from dronedream_agent_core.rgb_input_quality import freeze_gazebo_rgb
from dronedream_agent_core.simulation_sensor_frames import simulation_pose_time_ns


class RenderPairBuffer:
    """Latest-only bounded buffers; commanded pose is never substituted for observation."""

    # 功能：
    #   建立有界 RGB、标签及姿态缓存，回调只复制数据，不执行训练或磁盘写入。
    # 输入：
    #   无。
    # 输出：
    #   None：初始化缓存与错误状态。
    def __init__(self):
        self.lock = Lock()
        self.frames = {"rgb": deque(maxlen=8), "semantic": deque(maxlen=8)}
        self.pose = None
        self.poses = deque(maxlen=256)
        self.error = None
        self.last_time = {"rgb": -1, "semantic": -1, "pose": -1}

    # 功能：
    #   接收原生 RGB 或标签消息，冻结字节并检查来源时间单调性。
    # 输入：
    #   kind、message：已固定订阅类型和原生图像消息。
    # 输出：
    #   None：写入有限缓存；错误供主循环读取并停止本次采集。
    def image(self, kind, message):
        try:
            if kind not in self.frames:
                raise ValueError("VISION_RENDER_FRAME_KIND_INVALID")
            stamp = simulation_pose_time_ns(message)
            frozen = freeze_gazebo_rgb(message)
            with self.lock:
                if stamp < self.last_time[kind]:
                    raise ValueError("VISION_RENDER_CLOCK_REGRESSED")
                if stamp > self.last_time[kind]:
                    self.frames[kind].append((stamp, frozen))
                    self.last_time[kind] = stamp
        except Exception as error:
            with self.lock:
                self.error = type(error).__name__ + ":" + str(error)[:256]

    # 功能：
    #   接收指定无动力夹具的独立实测姿态，拒绝非法数值和倒退的仿真时钟。
    # 输入：
    #   message：Gazebo 原生 Pose 消息。
    # 输出：
    #   None：更新实测姿态或保存错误，不接受其他实体的姿态。
    def measured_pose(self, message):
        try:
            if message.name != RIG_NAME:
                return
            stamp = simulation_pose_time_ns(message)
            p, q = message.position, message.orientation
            pose = {"position_m": [p.x, p.y, p.z], "orientation_wxyz": [q.w, q.x, q.y, q.z]}
            validate_fixture_command(pose)
            with self.lock:
                if stamp < self.last_time["pose"]:
                    raise ValueError("VISION_RENDER_POSE_CLOCK_REGRESSED")
                if stamp == self.last_time["pose"]:
                    if self.pose is not None and self.pose[1] != pose:
                        raise ValueError("VISION_RENDER_POSE_TIME_CONFLICT")
                    return
                self.pose = (stamp, pose)
                self.poses.append(self.pose)
                self.last_time["pose"] = stamp
        except Exception as error:
            with self.lock:
                self.error = type(error).__name__ + ":" + str(error)[:256]

    # 功能：
    #   清空前一视角帧，返回新的来源时间屏障，防止服务返回后仍读取旧位置图片。
    # 输入：
    #   无。
    # 输出：
    #   barrier：清空前已经观察到的最大仿真纳秒时刻。
    def clear(self):
        with self.lock:
            barrier = max(self.last_time.values())
            for frames in self.frames.values():
                frames.clear()
            self.poses.clear()
            return barrier

    # 功能：
    #   等待同一源时间的 RGB/标签，采样时刻须落在连续稳定实测位姿内；超时不猜测配对。
    # 输入：
    #   wanted、barrier：目标光学姿态及命令后的来源时间屏障。
    #   timeout：最大等待秒数，不以服务确认代替实际观测。
    # 输出：
    #   result：配对图像、采样时刻及实测姿态；超时抛出错误。
    def wait_pair(self, wanted, barrier, timeout=20.0):
        validate_fixture_command(wanted)
        if type(timeout) not in (int, float) or not 0 < timeout <= 60:
            raise ValueError("VISION_RENDER_WAIT_BUDGET_INVALID")
        if type(barrier) is not int or barrier < -1:
            raise ValueError("VISION_RENDER_BARRIER_INVALID")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.error:
                    raise ValueError(self.error)
                poses = list(self.poses)
                rgb, semantic = list(self.frames["rgb"]), dict(self.frames["semantic"])
            stable_since, previous_stamp = None, None
            # 检查每个回调，而非每次轮询的最后姿态；丢失超过五个发布周期就重新计稳。
            for stamp, observed in poses:
                position_error = math.dist(wanted["position_m"], observed["position_m"])
                dot = abs(sum(a * b for a, b in zip(wanted["orientation_wxyz"],
                                                   observed["orientation_wxyz"], strict=True)))
                if stamp > barrier and position_error <= 0.002 and dot >= 1 - 1e-7:
                    if (stable_since is None or previous_stamp is None
                            or stamp - previous_stamp > 50_000_000):
                        stable_since = stamp
                else:
                    stable_since = None
                previous_stamp = stamp
            if stable_since is not None:
                latest_stamp = poses[-1][0]
                for frame_stamp, image in reversed(rgb):
                    if (stable_since + 100_000_000 <= frame_stamp <= latest_stamp
                            and frame_stamp in semantic):
                        # 此处只采集无动态物体的静态训练世界。渲染线程可晚于物理线程
                        # 发布图像，必须核对拍摄时刻两侧的实测位姿，不能将发布时间差
                        # 当作相机移动；真实飞行图像仍使用运行端独立的新鲜度规则。
                        before_stamp, measured = next((stamp, value)
                            for stamp, value in reversed(poses) if stamp <= frame_stamp)
                        after_stamp = next(stamp for stamp, _ in poses if stamp >= frame_stamp)
                        if (frame_stamp - before_stamp > 50_000_000
                                or after_stamp - frame_stamp > 50_000_000):
                            continue
                        labels = semantic[frame_stamp]
                        if (image.width, image.height) != (labels.width, labels.height):
                            raise ValueError("VISION_RENDER_LABEL_DIMENSIONS_DIFFER")
                        result = {"rgb": image, "semantic": labels,
                                  "simulation_ns": frame_stamp, "measured_pose": measured}
                        return result
            time.sleep(0.01)
        raise TimeoutError("VISION_RENDER_ALIGNED_PAIR_NOT_READY")
