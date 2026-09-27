"""Portable wiring for bounded, candidate-model Gazebo map-fusion sessions."""

from pathlib import Path

from .map_fusion_experiment import prepare_map_fusion_experiment
from .plugin_files import hash_plugin_file
from .simulation_camera_clock import validate_native_camera_clock


# 功能：把已确认的仿真计划接到连续定位执行器，不依赖开发目录，不授予真机或验收资格。
# 输入：本次运行、实际地图、产品执行器和 PX4 安装路径；上层先验证模型试用契约。
# 输出：旁置执行器与实际传感器参数；缺失资源在创建仿真进程前明确失败。
def prepare_simulation_fusion(*, run_dir, world_sdf, semantic_path, executor_path, px4_root):
    runtime = Path(executor_path).parent
    wrapper = runtime / "px4_map_fusion_experiment_executor.py"
    for required in (wrapper, runtime / "px4_offboard_track_executor.py"):
        hash_plugin_file(required, limit=4 * 1024**2)
    validate_native_camera_clock(runtime / "camera-clock")
    camera = Path(px4_root) / "Tools/simulation/gz/models/OakD-Lite/model.sdf"
    camera_hash = hash_plugin_file(camera, limit=4 * 1024**2)
    path, inputs = prepare_map_fusion_experiment(run_dir, world_sdf, semantic_path)
    return wrapper, {
        "simulation_fusion_inputs": path,
        "native_source_clock_domain": "px4-gz-sitl:" + inputs["run_name"],
        "localization_source_channel": Path(run_dir) / "runtime-state/localization-source.json",
        "local_reference_prearm": True,
        "batch_static_world_visuals": True,
        "simulation_camera_profile": "responsive-control",
        "camera_source_model_sha256": camera_hash,
    }
