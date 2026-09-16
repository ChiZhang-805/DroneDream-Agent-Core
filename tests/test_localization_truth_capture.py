"""Independent simulator-truth fixtures must never authorize model control."""

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.localization_truth_capture import LocalizationTruthCapture
from dronedream_agent_core.simulation_sensor_frames import collision_center_from_canonical


# 功能：
#   构造用于坐标计算对照的独立合成位姿字典。
# 输入：
#   position：三维位置，单位米。
#   quaternion：按 w、x、y、z 排列的四元数。
# 输出：
#   value：使用列表保存位置和方向的位姿对象。
def pose(position, quaternion=(1., 0., 0., 0.)):
    value = {"position_m": list(position), "orientation_wxyz": list(quaternion)}
    return value


# 功能：
#   构造自洽的安装契约夹具，只用于验证代码边界，不冒充实机标定。
# 输入：
#   无。
# 输出：
#   frames：带自身摘要及明确真值隔离标志的合成契约。
def contract():
    value = {"vehicle_model_name": "drone", "canonical_at_rest": pose([0, 0, .24]),
             "canonical_link_name": "base_link",
             "collision_center_model_m": [0, 0, .228], "depth_mount_verified": True,
             "truth_used_as_control_input": False}
    frames = {**value, "record_sha256": sha256_json(value)}
    return frames


# 功能：
#   模拟观测器消费的两个 Gazebo 实体及来源时钟，可注入重复机体实体。
# 输入：
#   t：仿真来源秒数。
#   duplicate：是否加入第二个同名机体。
# 输出：
#   value：包含实体序列和来源时间戳的消息替身。
def message(t=1, duplicate=False):
    # 功能：
    #   生成观测器接口所需的命名实体及单位方向。
    # 输入：
    #   name：模型或链接名称。
    #   position：三维位置米数。
    # 输出：
    #   value：具有位置和四元数属性的实体替身。
    def entity(name, position):
        value = NS(name=name, position=NS(**dict(zip("xyz", position, strict=True))),
                   orientation=NS(w=1., x=0., y=0., z=0.))
        return value
    poses = [entity("drone", [-42.25, 15.3, 7.487]), entity("base_link", [0, 0, .24])]
    if duplicate:
        poses.append(entity("base_link", [9, 9, 9]))
    value = NS(pose=poses, header=NS(stamp=NS(sec=t, nsec=0)))
    return value


# 功能：
#   只在测试临时目录写入独立见证回执。
# 输入：
#   path：回执输出路径。
#   value：待写入对象。
# 输出：
#   None：不返回业务数据。
def publish(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


# 功能：
#   验证对照位置使用碰撞中心，不错误使用高出十二毫米的链接原点。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_truth_uses_geometric_center_not_the_12mm_higher_link_origin():
    result = collision_center_from_canonical(model_world=pose([-42.25, 15.3, 7.487]),
        canonical_in_model=pose([0, 0, .24]), canonical_at_rest=pose([0, 0, .24]),
        collision_center_model_m=[0, 0, .228])
    np.testing.assert_allclose(result, [-42.25, 15.3, 7.715], atol=1e-12)


# 功能：
#   验证模型外层平移与机体中心偏移均按各自旋转合成，不仅叠加世界坐标偏移。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_truth_rotates_both_wrapper_translation_and_moving_body_center_offset():
    h = math.sqrt(.5)
    result = collision_center_from_canonical(model_world=pose([1, 2, 3], [h, 0, 0, h]),
        canonical_in_model=pose([2, 0, .7], [h, 0, h, 0]),
        canonical_at_rest=pose([0, 0, .24]), collision_center_model_m=[0, 0, .228])
    np.testing.assert_allclose(result, [1, 3.988, 3.7], atol=1e-12)


# 功能：
#   拒绝零、未归一和非有限方向，避免错误方向被包装成位置精度证据。
# 输入：
#   q：非法四元数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("q", [[0, 0, 0, 0], [1, 1, 0, 0], [float("nan"), 0, 0, 0]])
def test_truth_refuses_invalid_orientation(q):
    with pytest.raises(ValueError, match="SIMULATION_FRAME"):
        collision_center_from_canonical(model_world=pose([1, 2, 3], q),
            canonical_in_model=pose([0, 0, .24]), canonical_at_rest=pose([0, 0, .24]),
            collision_center_model_m=[0, 0, .228])


# 功能：
#   验证契约和消息隔离、来源时钟、摘要、采样限制及不授予控制权的真实写盘结果。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_independent_truth_owns_frames_preserves_clocks_and_never_grants_control(tmp_path):
    frames = contract()
    capture = LocalizationTruthCapture(tmp_path, frames=frames, summary_publisher=publish)
    frames["collision_center_model_m"][2] = 10
    sample = message()
    assert capture.record(sample, received_monotonic=10., received_unix_ms=10000)
    sample.pose[0].position.z = 100
    assert not capture.record(message(), received_monotonic=10.1, received_unix_ms=10100)
    assert capture.record(message(2), received_monotonic=10.2, received_unix_ms=10200)
    summary = capture.close()
    assert summary["complete"] and summary["accepted_count"] == 2
    rows = [json.loads(line) for line in capture.path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["collision_center_world_enu_m"][2] == pytest.approx(7.715)
    assert rows[0]["received_monotonic_seconds"] == 10.
    assert rows[0]["received_at_unix_ms"] == 10000
    assert rows[0]["publisher_simulation_time_ns"] == 10**9
    assert all(not r["truth_used_as_control_input"] and not r["qualification_granted"]
               for r in rows)
    for row in rows:
        digest = row.pop("record_sha256")
        assert digest == sha256_json(row)
    assert not capture.record(message(3), received_monotonic=11., received_unix_ms=11000)
    with pytest.raises(FileExistsError):
        LocalizationTruthCapture(tmp_path, frames=contract(), summary_publisher=publish)


# 功能：
#   重复实体或回退的来源时钟必须使见证不完整，不能作为精度通过依据。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_duplicate_entity_and_backwards_clock_are_not_precision_evidence(tmp_path):
    for name, bad in (("entity", message(2, duplicate=True)), ("clock", message(0))):
        directory = tmp_path / name
        directory.mkdir()
        capture = LocalizationTruthCapture(directory, frames=contract(), summary_publisher=publish)
        assert capture.record(message(), received_monotonic=10., received_unix_ms=10000)
        assert not capture.record(bad, received_monotonic=10.2, received_unix_ms=10200)
        summary = capture.close()
        assert not summary["complete"] and summary["capture_issue"]


# 功能：
#   检查安装契约变化而未重绑摘要时，初始化不创建任何采集文件。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_modified_frame_contract_does_not_start_a_capture(tmp_path):
    frames = contract()
    frames["canonical_at_rest"]["position_m"][2] = .228
    with pytest.raises(ValueError, match="FRAME_CONTRACT_INVALID"):
        LocalizationTruthCapture(tmp_path, frames=frames, summary_publisher=publish)
    assert not list(tmp_path.iterdir())


# 功能：
#   在存在性预检过期时仍独占领取真值流，保留第一所有者的文件和关闭结果。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：临时替换存在性检查。
# 输出：
#   None：不返回业务数据。
def test_truth_capture_claim_is_exclusive_even_with_stale_precheck(tmp_path, monkeypatch):
    first = LocalizationTruthCapture(tmp_path, frames=contract(), summary_publisher=publish)
    second = None
    try:
        monkeypatch.setattr(Path, "exists", lambda self: False)
        with pytest.raises(FileExistsError):
            second = LocalizationTruthCapture(tmp_path, frames=contract(),
                                               summary_publisher=publish)
    finally:
        if second is not None:
            second.close()
        first.close()
