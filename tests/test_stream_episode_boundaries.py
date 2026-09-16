"""Grounded stream reader tests using synthetic runs and deliberate file mutations."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_stream_collection import _write_new, stream_fixture

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.stream_capture import finalize_stream_captures
from dronedream_agent_core.training.stream_episode import (
    load_grounded_stream_episode,
    validate_stream_visual,
)


# 功能：
#   建立并加载一份有完整合成账本的控制流回合，不产生真实飞行资格。
# 输入：
#   tmp_path：测试独占根目录。
# 输出：
#   path：回合目录。
#   config：离线教师配置。
#   loaded：已加载的训练回合。
def grounded_episode(tmp_path):
    path, config, _, packed = stream_fixture(tmp_path)
    finalize_stream_captures(path, [packed], write_new=_write_new)
    loaded = load_grounded_stream_episode(path, config)
    return path, config, loaded


# 功能：
#   验证调用方改写返回的教师速度不会污染下一次缓存结果。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_stream_teacher_cache_is_not_shared_with_caller(tmp_path):
    _, _, loaded = grounded_episode(tmp_path)
    observation = loaded.visits[0].observation
    first = loaded.oracle._context(observation)
    expected = first.velocity.x
    first.velocity.x += 1
    assert loaded.oracle._context(observation).velocity.x == expected


# 功能：
#   验证编码器不能改写被核对的原视觉特征来制造两边相等。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_visual_encoder_cannot_change_expected_features(tmp_path):
    _, _, capture, _ = stream_fixture(tmp_path)
    before = sha256_json(capture.observation)

    # 功能：
    #   注入会改写传入样本的故障编码器，模拟原地替换特征后返回同一对象。
    # 输入：
    #   sample：验证器传入的样本。
    #   media：测试不使用的媒体载荷。
    # 输出：
    #   sample：被故意改变特征的对象。
    def attach(sample, media):
        sample.visual_features = [.1] * 8
        sample.source_visual_sha256 = "a" * 64
        return sample

    with pytest.raises(ValueError, match="VISUAL"):
        validate_stream_visual(capture, SimpleNamespace(attach=attach))
    assert sha256_json(capture.observation) == before


# 功能：
#   验证车辆资产不能通过重复 JSON 字段被静默重解释，即使文件摘要被一起重绑。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_stream_asset_rejects_duplicate_json_keys(tmp_path):
    path, config, _, packed = stream_fixture(tmp_path)
    reset_path = path / "reset.json"
    reset = json.loads(reset_path.read_bytes())
    vehicle = Path(reset["config"]["vehicle"])
    original = vehicle.read_text(encoding="utf-8")
    key, value = next(iter(json.loads(original).items()))
    vehicle.write_text("{" + json.dumps(key) + ":" + json.dumps(value) + "," + original[1:],
                       encoding="utf-8")
    reset["config"]["asset_sha256"]["vehicle"] = hashlib.sha256(vehicle.read_bytes()).hexdigest()
    reset_path.write_text(json.dumps(reset), encoding="utf-8")
    finalize_stream_captures(path, [packed], write_new=_write_new)
    with pytest.raises(ValueError):
        load_grounded_stream_episode(path, config)
