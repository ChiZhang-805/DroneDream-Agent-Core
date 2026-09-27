"""Portable installed-session wiring, independent of development paths."""

from pathlib import Path

import pytest

from dronedream_agent_core import simulation_fusion_runtime as fusion
from dronedream_agent_core.map_fusion_experiment import read_map_fusion_experiment


# 功能：构造带空格和中文的安装/运行位置；输入：临时根；输出：独立路径，不启动设备。
def inputs(tmp_path, monkeypatch):
    runtime = tmp_path / "another user/软件/runtime"
    runtime.mkdir(parents=True)
    for name in ("px4_offboard_track_executor.py", "px4_map_fusion_experiment_executor.py"):
        (runtime / name).write_text("# packaged executor", encoding="utf-8")
    px4 = tmp_path / "custom px4"
    camera = px4 / "Tools/simulation/gz/models/OakD-Lite/model.sdf"
    camera.parent.mkdir(parents=True)
    camera.write_text("<sdf/>", encoding="utf-8")
    world, semantic = tmp_path / "world.sdf", tmp_path / "semantic.json"
    world.write_text("<sdf/>", encoding="utf-8")
    semantic.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fusion, "validate_native_camera_clock", lambda path: None)
    return dict(run_dir=tmp_path / "新任务/simulation", world_sdf=world,
                semantic_path=semantic, executor_path=runtime / "px4_offboard_track_executor.py",
                px4_root=px4)


# 功能：验证每次绑定真实资产/源钟，且路径可移植；输出不得包含路线作为测量或授予资格。
def test_portable_owned_runtime(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    wrapper, options = fusion.prepare_simulation_fusion(**args)
    config = read_map_fusion_experiment(options["simulation_fusion_inputs"])
    assert wrapper.parent == args["executor_path"].parent
    assert config["qualification_granted"] is False
    assert options["native_source_clock_domain"] == "px4-gz-sitl:" + config["run_name"]
    assert options["localization_source_channel"] == args["run_dir"] / "runtime-state/localization-source.json"
    assert options["simulation_camera_profile"] == "responsive-control"
    assert not any("route" in key for key in config)
    with pytest.raises(FileExistsError):
        fusion.prepare_simulation_fusion(**args)


# 功能：缺少任一执行器不得发布貌似可运行的输入；输出：在启动前报错。
@pytest.mark.parametrize("name", ["px4_offboard_track_executor.py", "px4_map_fusion_experiment_executor.py"])
def test_missing_executor_does_not_publish(tmp_path, monkeypatch, name):
    args = inputs(tmp_path, monkeypatch)
    (args["executor_path"].parent / name).unlink()
    with pytest.raises((OSError, ValueError)):
        fusion.prepare_simulation_fusion(**args)
    assert not (args["run_dir"].parent / "fusion-inputs.json").exists()


# 功能：原生钟未通过校验时不得进入准备阶段；输出：保留原异常，不回退假时间。
def test_clock_failure_is_fatal(tmp_path, monkeypatch):
    args = inputs(tmp_path, monkeypatch)
    def invalid(path):
        raise ValueError("NATIVE_CAMERA_CLOCK_RUNTIME_INVALID")
    monkeypatch.setattr(fusion, "validate_native_camera_clock", invalid)
    with pytest.raises(ValueError, match="CLOCK_RUNTIME_INVALID"):
        fusion.prepare_simulation_fusion(**args)
    assert not (args["run_dir"].parent / "fusion-inputs.json").exists()
