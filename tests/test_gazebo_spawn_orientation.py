import math
from types import SimpleNamespace

import pytest

from dronedream_agent_core import gazebo_adapter


# 功能：显式偏航必须进入 Gazebo 生成请求并留下回执，不只修改后续参考航向。
# 输入：tmp_path：隔离资产；monkeypatch：记录协议请求；yaw：选定偏航。
# 输出：无；验证位置保持不变，姿态为对应单位四元数。
@pytest.mark.parametrize("yaw", [None, 0., math.pi / 4, -math.pi])
def test_spawn_orientation_is_applied_at_entity_creation(tmp_path, monkeypatch, yaw):
    asset = tmp_path / "vehicle.sdf"
    asset.write_text("<sdf/>")
    requests = []

    # 功能：仅截取生成请求，测试不启动 Gazebo 或接触执行机构。
    # 输入：command：待发送参数；kwargs：标准执行配置。
    # 输出：成功形状的模拟传输回执。
    def capture(command, **kwargs):
        requests.append(command[command.index("--req") + 1])
        return SimpleNamespace(returncode=0, stdout="data: true", stderr="")

    monkeypatch.setattr(gazebo_adapter, "_run", capture)
    result = gazebo_adapter._spawn_entity("gz", world_name="world", entity_name="drone",
        sdf_path=asset, pose=(1., 2., 3.), env={}, yaw_rad=yaw)
    assert "position { x: 1 y: 2 z: 3 }" in requests[0]
    assert result["pose_enu_m"] == (1., 2., 3.)
    if yaw is None:
        assert "orientation {" not in requests[0]
        assert "initial_yaw_rad" not in result
    else:
        assert f"w: {math.cos(yaw / 2):.12g}" in requests[0]
        assert f"z: {math.sin(yaw / 2):.12g}" in requests[0]
        assert result["initial_yaw_rad"] == yaw


# 功能：畸形偏航不能到达外部服务。
# 输入：tmp_path、monkeypatch：隔离测试工具；yaw：非法输入。
# 输出：无；在调用外部命令前明确拒绝。
@pytest.mark.parametrize("yaw", [True, "0", float("nan"), float("inf"), math.pi + .1])
def test_invalid_spawn_orientation_never_launches(tmp_path, monkeypatch, yaw):
    # 功能：禁止测试非法输入时启动任何外部进程。
    # 输入：args、kwargs：原命令入口参数。
    # 输出：无；被调用即表示输入校验失败。
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid heading must not reach the simulator")

    monkeypatch.setattr(gazebo_adapter, "_run", forbidden)
    with pytest.raises((ValueError, gazebo_adapter.SimulationRuntimeError)):
        gazebo_adapter._spawn_entity("gz", world_name="world", entity_name="drone",
            sdf_path=tmp_path / "absent.sdf", pose=(1., 2., 3.), env={}, yaw_rad=yaw)
