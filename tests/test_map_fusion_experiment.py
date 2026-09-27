"""Synthetic ownership/lifecycle checks; no flight qualification or aircraft."""

import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from dronedream_agent_core.map_fusion_experiment import (
    MapFusionExperimentMixin,
    prepare_map_fusion_experiment,
    read_map_fusion_experiment,
)


# 功能：创建便携的合成资产和独占实验；输入：测试目录；输出：输入文件及配置。
def fixture(tmp_path):
    world, semantic = tmp_path / "world.sdf", tmp_path / "semantic.json"
    world.write_text("<sdf/>")
    semantic.write_text("{}")
    return prepare_map_fusion_experiment(tmp_path / "owned" / "simulation", world, semantic)


# 功能：核对来源绑定与无资格声明；输入：合成目录；输出：断言结果。
def test_owned_manifest_and_no_overwrite(tmp_path):
    path, expected = fixture(tmp_path)
    assert read_map_fusion_experiment(path) == expected
    assert expected["qualification_granted"] is False
    with pytest.raises(FileExistsError):
        prepare_map_fusion_experiment(
            expected["run_dir"], expected["world_sdf"], expected["semantic_path"]
        )


# 功能：拒绝错运行、错类型、相对资产及伪造资格；输入：篡改字段；输出：明确失败。
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_name", "old-run"),
        ("run_dir", None),
        ("run_dir", "relative/simulation"),
        ("world_sdf", "world.sdf"),
        ("qualification_granted", True),
        ("experimental_only", False),
    ],
)
def test_manifest_rejects_mutations(tmp_path, field, value):
    path, inputs = fixture(tmp_path)
    inputs[field] = value
    path.write_text(json.dumps(inputs))
    with pytest.raises(ValueError, match="RUN_BINDING_INVALID"):
        read_map_fusion_experiment(path)


# 功能：资产变化不继承旧解析；输入：合成目录；输出：摘要不匹配。
def test_asset_change_is_detected(tmp_path):
    path, inputs = fixture(tmp_path)
    Path(inputs["world_sdf"]).write_text("<sdf>changed</sdf>")
    with pytest.raises(ValueError, match="MAP_ASSET_CHANGED"):
        read_map_fusion_experiment(path)


class NativeStub:
    connect = AsyncMock()
    close = AsyncMock()
    wait_until_local_position_ready = AsyncMock(return_value=True)


class Client(MapFusionExperimentMixin, NativeStub):
    pass


# 功能：缺失配置必须在连接飞控前失败；输入：隔离环境；输出：不调用连接。
def test_missing_config_fails_before_connect(monkeypatch):
    monkeypatch.delenv("DRONEDREAM_MAP_FUSION_INPUTS", raising=False)
    NativeStub.connect.reset_mock()
    with pytest.raises(ValueError, match="INPUTS_MISSING"):
        asyncio.run(Client().connect("unused"))
    NativeStub.connect.assert_not_awaited()


# 功能：后台早退或失败不得误报预检就绪；输入：任务结束类型；输出：原始失败。
@pytest.mark.parametrize("failure", [False, True])
def test_early_task_exit_prevents_ready(failure):
    async def check():
        client = Client()
        client._fusion_ready = asyncio.Event()
        client._fusion_ready.set()

        async def ended():
            if failure:
                raise RuntimeError("source-failed")

        client._fusion_task = asyncio.create_task(ended())
        await asyncio.sleep(0)
        NativeStub.wait_until_local_position_ready.reset_mock()
        with pytest.raises(RuntimeError, match="source-failed|STOPPED_BEFORE_PREFLIGHT"):
            await client.wait_until_local_position_ready(1)
        NativeStub.wait_until_local_position_ready.assert_not_awaited()
        await client.close()

    asyncio.run(check())


# 功能：真实包装执行器可加载并保留原生实现；输入：源码路径；输出：类型继承证明。
def test_executor_wrapper_retains_native_client():
    path = Path(__file__).resolve().parents[1] / "runtime/px4_map_fusion_experiment_executor.py"
    spec = importlib.util.spec_from_file_location("fusion_wrapper_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert issubclass(module.MavsdkOffboardClient, MapFusionExperimentMixin)
    assert module.MavsdkOffboardClient.close is MapFusionExperimentMixin.close
    assert callable(module.main)


# 功能：实验采集必须有真实源钟、原生传感器与原始媒体；输入：失配配置；输出：拒绝。
@pytest.mark.parametrize(
    "missing",
    ["native_sensor_runtime", "record_multimodal_training_dataset", "simulation_camera_profile"],
)
def test_experiment_requires_source_matched_recording(tmp_path, missing):
    from test_native_action_risk_artifacts import native_episode

    from dronedream_agent_core.training.px4_environment import Px4TrainingConfig

    episode, _ = native_episode(tmp_path)
    config = json.loads((episode / "reset.json").read_bytes())["config"]
    config.update(
        experimental_map_fusion=True,
        native_sensor_runtime=str(tmp_path),
        native_camera_clock_runtime=str(tmp_path),
        record_multimodal_training_dataset=True,
        simulation_camera_profile="responsive-control",
        camera_source_model_sha256="a" * 64,
        visual_package=str(tmp_path),
    )
    assert Px4TrainingConfig.model_validate(config).experimental_map_fusion
    config[missing] = {
        "native_sensor_runtime": None,
        "record_multimodal_training_dataset": False,
        "simulation_camera_profile": "native",
    }[missing]
    with pytest.raises(ValueError, match="map fusion collection requires|exclusive native sensor"):
        Px4TrainingConfig.model_validate(config)


# 功能：拒绝同时启用两套相机源钟；输入：合成配置；输出：明确配置错误。
def test_competing_camera_clocks_rejected(tmp_path):
    from test_native_action_risk_artifacts import native_episode

    from dronedream_agent_core.training.px4_environment import Px4TrainingConfig

    episode, _ = native_episode(tmp_path)
    config = json.loads((episode / "reset.json").read_bytes())["config"]
    config.update(
        native_sensor_runtime=str(tmp_path),
        native_camera_clock_runtime=str(tmp_path),
        render_replica_runtime=str(tmp_path),
    )
    with pytest.raises(ValueError, match="exclusive native sensor runtime"):
        Px4TrainingConfig.model_validate(config)
