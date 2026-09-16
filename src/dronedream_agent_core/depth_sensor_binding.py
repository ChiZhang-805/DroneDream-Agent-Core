"""Bind a validated simulation stream to its actual projection, never a default size."""

from .depth_projection import (
    DepthProjectionCalibration,
    ProjectedDepthFrame,
    metric_depth_sample_stride,
    project_metric_depth_frame,
)
from .runtime_sensor_contracts import RuntimeSensorRegistry, oakd_lite_depth_runtime_contract


class DepthSensorBinding:
    """One control-thread owner; failed frames cannot replace calibration."""

    # 功能：
    #   初始化单控制线程拥有的深度来源绑定，首个成功解码帧决定实际像素布局。
    # 输入：
    #   self：当前深度绑定。
    #   registry：运行传感器注册表。
    #   vehicle_id：归属机体标识。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, registry: RuntimeSensorRegistry, *, vehicle_id: str):
        if (not isinstance(registry, RuntimeSensorRegistry) or type(vehicle_id) is not str
                or not 1 <= len(vehicle_id) <= 160):
            raise ValueError("DEPTH_BINDING_CONFIGURATION_INVALID")
        self._registry, self._vehicle_id = registry, vehicle_id
        self._layout: tuple[int, ...] | None = None
        self._calibration: DepthProjectionCalibration | None = None

    # 功能：
    #   1. 按实际 Gazebo 帧尺寸投影米制深度，仅成功解码并登记后锁定校准布局。
    #   2. 本入口只适用已由上游核对光学配置的仿真流；真机需要实测内外参和无回波语义。
    #   3. 来源身份和新鲜度仍由上游核对，不把校准绑定当作运动许可。
    # 输入：
    #   self：当前绑定及已成功接入的布局。
    #   message：包含宽高、行跨度、R_FLOAT32 格式和原始像素缓冲的仿真消息。
    # 输出：
    #   result：实际源像素覆盖率和校准后的米制射线。
    def project(self, message) -> ProjectedDepthFrame:
        # gz-msgs10 PixelFormatType R_FLOAT32 = 13. The caller separately verifies
        # source identity/profile/time; a message cannot choose its own optics.
        layout = tuple(getattr(message, name, None)
                       for name in ("width", "height", "step", "pixel_format_type"))
        if any(type(value) is not int for value in layout) or layout[3] != 13:
            raise ValueError("DEPTH_PIXEL_LAYOUT_UNSUPPORTED")
        if self._layout is not None and layout != self._layout:
            raise ValueError("DEPTH_SOURCE_LAYOUT_CHANGED")
        calibration = self._calibration or DepthProjectionCalibration(
            width=layout[0], height=layout[1], horizontal_fov_rad=1.274,
            minimum_depth_m=0.2, maximum_depth_m=19.1,
            sample_stride_pixels=metric_depth_sample_stride(width=layout[0], height=layout[1]),
            no_return_mode="gazebo-far-clip",
        )
        result = project_metric_depth_frame(
            # 共享投影入口先校验缓冲类型与 nbytes，再冻结；禁止 bytes(整数) 隐式分配。
            data=getattr(message, "data", None), row_step_bytes=layout[2], calibration=calibration,
        )
        if self._calibration is None:
            self._registry.register_contract(oakd_lite_depth_runtime_contract(
                vehicle_id=self._vehicle_id, calibration=calibration,
            ))
            self._calibration, self._layout = calibration, layout
        return result
