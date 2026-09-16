"""Replay file ownership and strict decoding; no physical rollout is started."""

import json

import pytest
from test_causal_policy import samples
from test_mission_groups import group_fixture

from dronedream_agent_core.training import causal_replay as replay


# 功能：
#   导出一组小型合成回放，用实际生产写入器生成完整内容绑定。
# 输入：
#   tmp_path：隔离的回放目录。
# 输出：
#   paths：五个固定回放文件的路径映射。
#   receipt：匹配实际写出字节的回放摘要和分组契约。
def bundle(tmp_path):
    hashes = replay.export_replay_bundle(tmp_path, training=samples(),
        validation=samples(100, "val"), training_history=[], validation_history=[],
        manifest=group_fixture())
    paths = {name: tmp_path / filename for name, filename in replay.REPLAY_FILES.items()}
    receipt = {"split_contract": replay.CAUSAL_SPLIT_CONTRACT, "replay_artifact_sha256": hashes}
    return paths, receipt


# 功能：
#   读取回放必须执行实际文件字节预算，不能因摘要正确就接受无界文件。
# 输入：
#   tmp_path：隔离回放目录。
#   monkeypatch：缩小预算以避免创建大文件。
# 输出：
#   None：不返回业务数据。
def test_bound_replay_enforces_actual_file_budget(tmp_path, monkeypatch):
    paths, receipt = bundle(tmp_path)
    monkeypatch.setattr(replay, "MAX_REPLAY_FILE_BYTES", 16, raising=False)
    with pytest.raises(ValueError):
        replay.read_bound_replay(paths, receipt)


# 功能：
#   每一行回放都必须是严格对象，不能忽略空白行或用重复键覆盖历史输入。
# 输入：
#   tmp_path：隔离回放目录。
#   fault：空白记录或重复字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fault", ["blank", "duplicate"])
def test_replay_rejects_blank_records_and_duplicate_keys(tmp_path, fault):
    paths, receipt = bundle(tmp_path)
    contents = replay.read_bound_replay(paths, receipt)
    if fault == "blank":
        contents["training_replay"] += b"\n"
    else:
        original, *rest = contents["training_replay"].splitlines()
        encoded = b'{"sample_weight":0,' + original[1:]
        contents["training_replay"] = b"\n".join([encoded, *rest]) + b"\n"
    with pytest.raises(ValueError):
        replay.decode_replay(contents)


# 功能：
#   后面一个目标已存在时必须在创建任何新文件前拒绝，不能留下半组覆盖冲突。
# 输入：
#   tmp_path：隔离回放目录。
# 输出：
#   None：不返回业务数据。
def test_existing_last_target_is_rejected_before_creating_first_file(tmp_path):
    target = tmp_path / replay.REPLAY_FILES["stream_groups"]
    target.write_text("preserve-existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        replay.export_replay_bundle(tmp_path, training=samples(), validation=samples(100, "val"),
            training_history=[], validation_history=[], manifest=group_fixture())
    assert target.read_text(encoding="utf-8") == "preserve-existing"
    assert not (tmp_path / replay.REPLAY_FILES["training_replay"]).exists()


# 功能：
#   补充来源分组后的统计应独立拥有原有列表，调用方修改原统计不能改写回执内容。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_bound_metrics_do_not_alias_original_lists():
    original = {"training_groups": ["a"], "validation_groups": ["b"]}
    result = replay.bind_source_group_metrics(original, ({"a"}, {"b"}))
    original["training_groups"].append("mutated")
    assert result["training_window_groups"] == ["a"]


# 功能：
#   缺失或额外的文件集合不能被解码器默默接受，也不能以无关键错误终止。
# 输入：
#   tmp_path：隔离回放目录。
# 输出：
#   None：不返回业务数据。
def test_decoder_requires_exact_bundle_keys(tmp_path):
    paths, receipt = bundle(tmp_path)
    contents = replay.read_bound_replay(paths, receipt)
    contents["unbound-extra"] = json.dumps({"sample": "ignored"}).encode()
    with pytest.raises(ValueError):
        replay.decode_replay(contents)
