"""Small simulation-only truth witness; no files, planner, inference or actuator.

Runs ahead of the adapter's diagnostic worker. The same canonical-link resolver
is injected so a faster channel cannot quietly switch to a different vehicle.
"""

from __future__ import annotations

import threading
import time
from itertools import islice

from ..contracts import RuntimeLocalSafetyObservation, Vector3
from ..simulation_dynamic_obstacles import (
    dynamic_observation,
    dynamic_positions,
    simulation_velocity,
    witness_age,
)
from .geometry_inputs import finite_metric, metric_vector

# Keep a <=50 Hz label-only witness, independent of the policy request rate.
# A 40 ms throttle aliased alternating native arrivals into >100 ms gaps even
# when the source's largest interval was <80 ms. The outcome window's original
# 100 ms endpoint / 250 ms internal-gap limits remain unchanged.
MINIMUM_WITNESS_INTERVAL_SECONDS = 0.02


class GazeboOutcomeWitness:
    # 功能：
    #   固定当前机体控制点、目标和已声明障碍包络，建立独立于规划器的仿真见证状态。
    # 输入：
    #   publisher：发送见证的独立通道。
    #   resolve_pose：当前机体规范链接解析器。
    #   collision_offset：控制点相对解析位置的三维偏移，米。
    #   target：本轮固定任务目标。
    #   dynamic_geometry：当前世界实体的模型根包围球半径映射；缺失实体不猜尺寸。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, publisher, resolve_pose, collision_offset, target, dynamic_geometry=None):
        if not callable(resolve_pose) or not callable(getattr(publisher, "send", None)):
            raise ValueError("OUTCOME_WITNESS_CALLBACK_INVALID")
        self.publisher, self.resolve_pose = publisher, resolve_pose
        self.collision_offset = metric_vector(collision_offset)
        self.target = Vector3.model_validate(target.model_dump(), strict=True)
        if dynamic_geometry is not None and (
            type(dynamic_geometry) is not dict or len(dynamic_geometry) > 128
        ):
            raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_MAPPING_INVALID")
        self.dynamic_geometry = dict(dynamic_geometry or {})
        for radius in self.dynamic_geometry.values():
            if not finite_metric(radius) or not 0 < radius <= 20:
                raise ValueError("OUTCOME_DYNAMIC_GEOMETRY_BOUND_INVALID")
        self._previous = None
        self._dynamic = {}
        self._sequence = 0
        self._last_received = -1.0
        self._last_seen_received = -1.0
        self._last_seen_unix_ms = -1
        self._last_seen_scene_ns = -1
        self.skipped_repeated_scene = 0
        self._lock = threading.Lock()
        self.skipped_busy = 0
        self.skipped_cadence = 0
        self.error = None
        self.maximum_processing_ms = 0.0
        self._closed = False

    # 功能：
    #   1. 在有界非阻塞回调中冻结动态位置，核对源时间及仿真时间后发布独立真值。
    #   2. 首帧速度未就绪不标健康；年龄依赖主机时钟，物理速度只依赖仿真时钟。
    #   3. 任意异常锁存为失败并释放锁，异常退出统计也不能阻止释放。
    # 输入：
    #   poses：当前原生位姿帧，最多 65,536 个实体。
    #   received_at：原始接收单调时刻，秒。
    #   received_at_unix_ms：原始接收 Unix 毫秒时间。
    #   simulation_time_ns：发布方场景纳秒时间。
    # 输出：
    #   None：观测通过 publisher 发送，不返回业务数据。
    def receive(
        self, poses, received_at: float, received_at_unix_ms: int, *, simulation_time_ns: int
    ):
        if self.error is not None or self._closed:
            return
        if not self._lock.acquire(blocking=False):
            self.skipped_busy += 1
            return  # Never hold up another native callback.
        processing_started = None
        try:
            if self.error is not None or self._closed:
                return
            processing_started = time.monotonic()
            if not finite_metric(processing_started) or processing_started < 0:
                raise ValueError("OUTCOME_PROCESSING_CLOCK_INVALID")
            if (
                not finite_metric(received_at)
                or received_at < 0
                or received_at > processing_started
                or type(received_at_unix_ms) is not int
                or not 0 <= received_at_unix_ms <= 2**63 - 1
            ):
                raise ValueError("OUTCOME_SOURCE_CLOCK_INVALID")
            # Windows monotonic clocks can return the same tick for adjacent
            # callbacks. Equality adds no new elapsed time; cadence filtering
            # below cannot turn it into a second published witness.
            if (
                received_at < self._last_seen_received
                or received_at_unix_ms < self._last_seen_unix_ms
            ):
                raise ValueError("OUTCOME_SOURCE_CLOCK_REGRESSED")
            if type(simulation_time_ns) is not int or not 0 <= simulation_time_ns <= 2**63 - 1:
                raise ValueError("OUTCOME_SIMULATION_CLOCK_INVALID")
            if simulation_time_ns < self._last_seen_scene_ns:
                raise ValueError("OUTCOME_SIMULATION_CLOCK_REGRESSED")
            repeated_scene = simulation_time_ns == self._last_seen_scene_ns
            self._last_seen_received = received_at
            self._last_seen_unix_ms = received_at_unix_ms
            self._last_seen_scene_ns = simulation_time_ns
            if repeated_scene:
                self.skipped_repeated_scene += 1
                return
            if received_at - self._last_received < MINIMUM_WITNESS_INTERVAL_SECONDS:
                self.skipped_cadence += 1
                return
            # 一次性迭代器只能消费一次；先固定动态数值，解析回调不得改写这些证据。
            poses = tuple(islice(iter(poses), 65_537))
            if len(poses) > 65_536:
                raise ValueError("OUTCOME_POSE_ENTITY_CAPACITY")
            dynamic = dynamic_positions(poses)
            resolved = self.resolve_pose(poses)
            if resolved is None:
                return
            position = metric_vector(
                tuple(
                    p + o
                    for p, o in zip(metric_vector(resolved[0]), self.collision_offset, strict=True)
                )
            )
            # Physical metres per SIMULATION second. Arrival spacing changes
            # with real-time factor and callback jitter; it is only for age.
            velocity = simulation_velocity(position, self._previous, simulation_time_ns)
            obstacles = []
            for name, root in sorted(dynamic.items()):
                history = self._dynamic.get(name)
                speed = simulation_velocity(root, history, simulation_time_ns)
                obstacles.append(
                    dynamic_observation(
                        name,
                        root,
                        speed,
                        self.dynamic_geometry,
                        velocity_known=history is not None,
                    )
                )
            completed = time.monotonic()
            if not finite_metric(completed) or completed < processing_started:
                raise ValueError("OUTCOME_PROCESSING_CLOCK_INVALID")
            age = witness_age(received_at, completed)
            observation = RuntimeLocalSafetyObservation(
                sequence=self._sequence + 1,
                observed_at_unix_ms=received_at_unix_ms,
                source="simulation-ground-truth",
                stream_healthy=(
                    age <= 0.1
                    and self._previous is not None
                    and all(item.confidence == 1 for item in obstacles)
                ),
                stream_age_seconds=age,
                localization_covariance_m2=0.0,
                current_position_m=Vector3(x=position[0], y=position[1], z=position[2]),
                current_velocity_mps=Vector3(x=velocity[0], y=velocity[1], z=velocity[2]),
                target_position_m=self.target.model_copy(deep=True),
                dynamic_obstacles=obstacles,
            )
            if self.publisher.send(observation) is False:
                raise ValueError("OUTCOME_PUBLISH_REJECTED")
            self._previous = simulation_time_ns, position
            self._dynamic = {name: (simulation_time_ns, root) for name, root in dynamic.items()}
            self._last_received = received_at
            self._sequence += 1
        except Exception as error:
            self.error = type(error).__name__ + ":" + str(error)
        finally:
            try:
                if processing_started is not None:
                    finished = time.monotonic()
                    if not finite_metric(finished) or finished < processing_started:
                        raise ValueError("OUTCOME_PROCESSING_CLOCK_INVALID")
                    duration = (finished - processing_started) * 1000
                    if not finite_metric(duration):
                        raise ValueError("OUTCOME_PROCESSING_CLOCK_INVALID")
                    self.maximum_processing_ms = max(self.maximum_processing_ms, duration)
            except Exception as error:
                if self.error is None:
                    self.error = type(error).__name__ + ":" + str(error)
            finally:
                self._lock.release()

    # 功能：
    #   在调用方排空原生订阅后关闭发布器，保留遗漏帧和首个异常的真实诊断。
    # 输入：
    #   self：当前独立采集器。
    # 输出：
    #   receipt：发送、丢帧、时基和处理时间记录，不授予训练或飞行资格。
    def close(self):
        # Caller must first drain native subscriptions; close never races a callback.
        with self._lock:
            self._closed = True
            self.publisher.close()
        receipt = {
            "sent": self.publisher.sent,
            "dropped": self.publisher.dropped,
            "skipped_busy": self.skipped_busy,
            "error": self.error,
            "skipped_cadence": self.skipped_cadence,
            "skipped_repeated_scene": self.skipped_repeated_scene,
            "velocity_time_basis": "publisher-simulation-time",
            "freshness_time_basis": "original-host-receipt",
            "minimum_publish_interval_seconds": MINIMUM_WITNESS_INTERVAL_SECONDS,
            "maximum_processing_ms": self.maximum_processing_ms,
            "source": "independent-gazebo; label-only; never actor input",
            "qualification_granted": False,
        }
        return receipt
