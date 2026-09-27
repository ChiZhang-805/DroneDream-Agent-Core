"""Exercise real launcher validation without starting a simulator or qualifying a flight."""

import pytest

from dronedream_agent_core import gazebo_adapter, model_image_cache


# 功能：
#   为真实启动器提供独立路径和绑定的派生相机配置，不创建任何仿真进程。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   arguments：实际启动器的最小关键字参数。
def camera_arguments(tmp_path):
    arguments = {
        name: tmp_path / name for name in (
            "run_dir", "world_sdf", "semantic_path", "vehicle_sdf", "route_path",
            "track_path", "clearance_path", "controller_params_path", "px4_root",
            "executor_path", "ros_workspace", "vehicle_metadata_path",
        )
    }
    arguments.update(
        simulation_camera_profile="responsive-control",
        camera_source_model_sha256="a" * 64,
        heading_policy="route-tangent-relative",
        local_navigation_visual_enabled=True,
        record_learning_observations=True,
        simulation_teacher_control=True,
    )
    return arguments


# 功能：
#   确认真实启动器分别接受教师采集和独占学习器权限，并在后续图像依赖检查处停止。
# 输入：
#   tmp_path：独占路径。
#   monkeypatch：仅在本测试替换尚未执行的图像依赖检查。
#   mode：显式教师采集或在线学习器。
#   replica：是否请求独立渲染，后续仍需完整制品与场景校验。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["teacher", "learner"])
@pytest.mark.parametrize("replica", [False, True])
def test_explicit_visual_training_reaches_next_dependency_check(tmp_path, monkeypatch, mode, replica):
    arguments = camera_arguments(tmp_path)
    if replica:
        arguments["render_replica_runtime"] = tmp_path / "replica"
        arguments["live_camera_enabled"] = False
    if mode == "learner":
        arguments.update(
            simulation_teacher_control=False,
            simulation_training_channel=tmp_path / "learner.json",
            local_navigation_provider="simulation-training",
            local_navigation_control_authority_required=True,
        )

    # 功能：
    #   在所有相关权限校验之后中断，不加载图像运行时，不启动飞行进程。
    # 输入：
    #   无。
    # 输出：
    #   无。
    def stop_at_image_dependency():
        raise RuntimeError("visual-runtime-boundary")

    monkeypatch.setattr(model_image_cache, "require_model_image_runtime", stop_at_image_dependency)
    with pytest.raises(RuntimeError, match="^visual-runtime-boundary$"):
        gazebo_adapter.run_px4_gazebo_track(**arguments)
    assert not (tmp_path / "run_dir").exists()


# 功能：
#   独立渲染只适用于显式且记录观测的仿真教师或学习器，拒绝普通运行及混合渲染源。
# 输入：
#   tmp_path：隔离目录。
#   changes：破坏合法教师渲染请求的参数。
# 输出：
#   None：非法组合未被拒绝时测试失败。
@pytest.mark.parametrize("changes", [
    {"simulation_teacher_control": False},
    {"record_learning_observations": False},
    {"local_navigation_visual_enabled": False},
    {"simulation_camera_profile": "native", "camera_source_model_sha256": None},
    {"live_camera_enabled": True},
    {"preflight_render_warmup": True},
    {"preflight_depth_warmup": True},
    {"render_preparation_runtime": "other"},
    {"render_cache_bundle": "other"},
    {"semantic_label_topic": "/labels"},
])
def test_replica_still_rejects_missing_authority_and_mixed_sources(tmp_path, changes):
    arguments = {**camera_arguments(tmp_path), "live_camera_enabled": False,
                 "render_replica_runtime": tmp_path / "replica", **changes}
    with pytest.raises(ValueError, match="isolated rendering requires exclusive"):
        gazebo_adapter.run_px4_gazebo_track(**arguments)
    assert not (tmp_path / "run_dir").exists()


# 功能：
#   拒绝不完整权限、非布尔开关、缺少视觉或来源绑定以及教师与模型权限混用。
# 输入：
#   tmp_path：独占路径。
#   changes：向合法教师请求注入的缺陷。
#   message：该缺陷必须触发的实际启动错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("changes,message", [
    ({"simulation_teacher_control": False}, "restricted to explicit visual"),
    ({"record_learning_observations": False}, "restricted to explicit visual"),
    ({"simulation_teacher_control": False, "record_learning_observations": False},
     "restricted to explicit visual"),
    ({"local_navigation_visual_enabled": False}, "restricted to explicit visual"),
    ({"simulation_teacher_control": "true"}, "must be boolean"),
    ({"record_learning_observations": 1}, "must be boolean"),
    ({"camera_source_model_sha256": None}, "SIMULATION_CAMERA_PROFILE_"),
    ({"local_navigation_provider": "local-policy"}, "teacher requires observation recording"),
    ({"local_navigation_control_authority_required": True},
     "teacher requires observation recording"),
    ({"vehicle_metadata_path": None}, "requires the actual vehicle metadata"),
])
def test_incomplete_training_never_authorizes_camera_changes(tmp_path, changes, message):
    arguments = {**camera_arguments(tmp_path), **changes}
    with pytest.raises(ValueError, match=message):
        gazebo_adapter.run_px4_gazebo_track(**arguments)
    assert not (tmp_path / "run_dir").exists()
