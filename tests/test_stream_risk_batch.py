import pytest
from test_stream_collection import stream_fixture

from dronedream_agent_core.training.px4_environment import _write_new
from dronedream_agent_core.training.stream_capture import finalize_stream_captures
from dronedream_agent_core.training.stream_episode import load_grounded_stream_episode


# 功能：
#   建立绑定原始文件的合成离线回合，验证批次机制，不连接飞机。
# 输入：
#   tmp_path：隔离回合目录。
# 输出：
#   loaded：完整离线回合及其教师。
def episode_fixture(tmp_path):
    episode, config, _, packed = stream_fixture(tmp_path)
    finalize_stream_captures(episode, [packed], write_new=_write_new)
    loaded = load_grounded_stream_episode(episode, config)
    return loaded


# 功能：
#   批次复用与逐次核验产生相同风险；关闭后的回调以及错误观测不能继续使用。
# 输入：
#   tmp_path：隔离回合目录。
# 输出：
#   None：等价性、来源约束及回调生命周期通过断言。
def test_fixed_risk_batch_preserves_labels_and_scope(tmp_path):
    loaded = episode_fixture(tmp_path)
    visit = loaded.visits[0]
    oracle = loaded.oracle
    assert oracle.risk_context_issue(visit.observation) is None
    assert oracle.receipts == []
    expected = oracle.risk(visit.observation, visit.proposal)
    with oracle.fixed_risk_context(visit.observation) as assess:
        assert assess(visit.observation, visit.proposal) == expected
        changed = visit.observation.model_copy(update={"sequence": 99})
        with pytest.raises(ValueError, match="OBSERVATION_MISMATCH"):
            assess(changed, visit.proposal)
    with pytest.raises(ValueError, match="OBSERVATION_MISMATCH"):
        assess(visit.observation, visit.proposal)


# 功能：
#   标注过程中源文件被修改时，退出检查必须失败且撤回本批回执，不能发布部分成功标签。
# 输入：
#   tmp_path：隔离回合目录。
# 输出：
#   None：篡改和证据回滚通过断言。
def test_fixed_risk_batch_rechecks_sources_on_exit(tmp_path):
    loaded = episode_fixture(tmp_path)
    oracle, visit = loaded.oracle, loaded.visits[0]
    with (pytest.raises(ValueError, match="SOURCE_CHANGED"),
          oracle.fixed_risk_context(visit.observation) as assess):
        assess(visit.observation, visit.proposal)
        (oracle.episode / 'stream-action-000000.json').write_text('{}')
    assert oracle.receipts == []
    assert oracle._key is None


# 功能：
#   时序预检只报告指定延迟错误，任何来源绑定错误仍直接拒绝，不能自动过滤。
# 输入：
#   tmp_path、monkeypatch：隔离回合与指定的上下文异常。
# 输出：
#   None：错误分类与证据无副作用通过断言。
def test_risk_context_precheck_never_suppresses_source_errors(tmp_path, monkeypatch):
    loaded = episode_fixture(tmp_path)
    oracle, visit = loaded.oracle, loaded.visits[0]

    # 功能：
    #   向上下文检查注入明确异常，验证只有延迟类型可被分类。
    # 输入：
    #   observation、record_context：原观测与禁止证据写入标记。
    # 输出：
    #   None：始终抛出当前测试异常。
    def invalid(observation, *, record_context=True):
        assert record_context is False
        raise ValueError(issue)

    monkeypatch.setattr(oracle, '_context', invalid)
    issue = 'DAGGER_NATIVE_ACTION_LATENCY_OUTSIDE_MODEL'
    assert oracle.risk_context_issue(visit.observation) == issue
    issue = 'STREAM_ANNOTATION_SOURCE_CHANGED'
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        oracle.risk_context_issue(visit.observation)
    assert oracle.receipts == []
