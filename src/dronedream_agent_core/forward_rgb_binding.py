"""Bind selected camera pixels and registry quality to the same original source."""

import hashlib

from .rgb_input_quality import (
    PreparedRgbMeasurement,
    decode_gazebo_rgb,
    freeze_gazebo_rgb,
    measure_rgb_input_quality,
)
from .runtime_sensor_contracts import (
    RuntimeSensorEnvelope,
    RuntimeSensorRegistry,
    oakd_lite_forward_rgb_runtime_contract,
)
from .sensor_frame_clock import SensorFrameTime, require_model_frame_time


class ForwardRgbBinding:
    """Control-thread-owned binding; late selection never dates old pixels anew."""

    # 功能：
    #   为单个飞行器建立前视 RGB 来源绑定，清空序号、尺寸与内容基线。
    # 输入：
    #   self：控制线程独占使用的绑定实例。
    #   registry：接收已校验传感器契约与观测的注册表。
    #   vehicle_id：当前飞行器身份。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, registry: RuntimeSensorRegistry, *, vehicle_id: str):
        self._registry, self._vehicle_id = registry, vehicle_id
        self._contract = self._dimensions = self._last_clock = self._last_digest = None
        self._sequence = 0
        self.last_quality = None

    # 功能：
    #   1. 将同一像素快照的布局、质量、摘要和原始时钟绑定为一条传感器观测。
    #   2. 同一来源与内容重复登记时跳过解码；同钟换图、尺寸变化或时间回退时拒绝。
    #   3. 注册表接受后才推进本地序号；RGB 引导可以降级，登记本身不授予运动权限。
    # 输入：
    #   self：当前控制线程持有的绑定实例。
    #   message：待登记的前视相机消息。
    #   frame_time：接入时保存的原始时钟记录。
    #   source_monotonic：原始来源的单调钟秒数。
    #   source_unix_ms：原始来源的 UNIX 毫秒数。
    #   now_monotonic：进行本次登记校验的当前单调钟秒数。
    #   prepared_measurement：同源图像工作器的原始像素质量结果，可为空。
    # 输出：
    #   registered：新增观测成功时为 True，相同观测未重复登记时为 False。
    def register(
        self, message, *, frame_time: SensorFrameTime | None, source_monotonic: float,
        source_unix_ms: int, now_monotonic: float,
        prepared_measurement: PreparedRgbMeasurement | None = None,
    ) -> bool:
        if frame_time is None:
            raise ValueError("RGB_ORIGINAL_SOURCE_CLOCK_UNAVAILABLE")
        require_model_frame_time(message, frame_time, source_monotonic, source_unix_ms)
        snapshot = freeze_gazebo_rgb(message)
        dimensions = (snapshot.width, snapshot.height, snapshot.pixel_format_type, snapshot.step)
        if self._dimensions is not None and dimensions != self._dimensions:
            raise ValueError("FORWARD_RGB_DIMENSIONS_CHANGED")
        digest = hashlib.sha256(snapshot.data).hexdigest()
        if prepared_measurement is not None and (
            not isinstance(prepared_measurement, PreparedRgbMeasurement)
            or prepared_measurement.snapshot != snapshot
            or prepared_measurement.sha256 != digest
        ):
            raise ValueError("FORWARD_RGB_PREPARATION_SOURCE_MISMATCH")
        if self._last_clock is not None:
            previous = self._last_clock.sample_monotonic_seconds
            if source_monotonic < previous:
                raise ValueError("FORWARD_RGB_SOURCE_REGRESSED")
            if source_monotonic == previous:
                if frame_time != self._last_clock or digest != self._last_digest:
                    raise ValueError("FORWARD_RGB_SOURCE_CHANGED")
                registered = False
                return registered
        contract = self._contract or oakd_lite_forward_rgb_runtime_contract(
            vehicle_id=self._vehicle_id, width=dimensions[0], height=dimensions[1],
            # 白墙可以缺少视觉纹理；独立几何覆盖、状态和安全约束仍决定各运动轴是否可用。
            required_for_motion=False,
        )
        if prepared_measurement is not None:
            quality = prepared_measurement.quality
        else:
            with decode_gazebo_rgb(snapshot) as rgb:
                quality = measure_rgb_input_quality(rgb)
        envelope = RuntimeSensorEnvelope(
            sensor_id=contract.sensor_id, vehicle_id=contract.vehicle_id,
            modality=contract.modality, sequence=self._sequence + 1,
            sample_monotonic_seconds=source_monotonic,
            received_monotonic_seconds=frame_time.received_monotonic_seconds,
            coordinate_frame=contract.coordinate_frame, unit=contract.unit,
            calibration_sha256=contract.calibration_sha256,
            intrinsics_sha256=contract.intrinsics_sha256,
            extrinsics_sha256=contract.extrinsics_sha256,
            quality=quality.score, coverage=1., payload_sha256=digest,
        )
        self._registry.register_contract(contract)
        self._registry.ingest(envelope, now_monotonic_seconds=now_monotonic)
        # 注册表校验通过后才提交基线，拒绝样本不能消耗本地序号或变成下一次重复判断依据。
        self._contract, self._dimensions = contract, dimensions
        self._last_clock, self._last_digest = frame_time, digest
        self._sequence += 1
        self.last_quality = quality
        registered = True
        return registered
