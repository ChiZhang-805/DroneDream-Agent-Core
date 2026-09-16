"""Admission of synthetic native episodes; these tests do not qualify flight."""

import json

import pytest
from test_native_action_risk_artifacts import native_episode

from dronedream_agent_core.training.native_episode import load_grounded_episode


# 功能：
#   验证停止标记、初始化和步骤记录都拒绝重复 JSON 字段，不接受静默覆盖。
# 输入：
#   tmp_path：合成回合根目录。
#   name：需要注入重复字段的文件。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("name", ["reset.json", "transition-000001.json",
                                 "flight/simulation/native-terminal-lifecycle.json"])
def test_native_episode_rejects_duplicate_fields(tmp_path, name):
    episode, config = native_episode(tmp_path)
    path = episode / name
    content = path.read_text(encoding="utf-8")
    key, value = next(iter(json.loads(content).items()))
    path.write_text("{" + json.dumps(key) + ":" + json.dumps(value) + "," + content[1:],
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_grounded_episode(episode, config)


# 功能：
#   验证入口在返回可训练访问之前就检查真实执行绑定，不能只验证各字段的类型。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_native_episode_rejects_forged_applied_action_at_load(tmp_path):
    episode, config = native_episode(tmp_path)
    path = episode / "transition-000001.json"
    row = json.loads(path.read_bytes())
    row["step"]["applied_action"]["axes"][0] += .1
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="APPLIED_ACTION"):
        load_grounded_episode(episode, config)


# 功能：
#   验证同回合残留控制流记录时拒绝按奖励步骤加载，即使流式汇总文件尚未生成。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_native_episode_cannot_mix_unfinished_stream_records(tmp_path):
    episode, config = native_episode(tmp_path)
    (episode / "stream-proposal-000000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="MODES_CANNOT_BE_MIXED"):
        load_grounded_episode(episode, config)


# 功能：
#   验证超出单转移读取预算时立即拒绝，不把数兆空白当成可无限读取的普通记录。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_native_episode_bounds_transition_size(tmp_path):
    episode, config = native_episode(tmp_path)
    with (episode / "transition-000001.json").open("ab") as stream:
        stream.write(b" " * (4 * 1024 * 1024))
    with pytest.raises(ValueError, match="LARGE|SIZE"):
        load_grounded_episode(episode, config)
