import pytest

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.temporal_evidence import ObservationHistory, TemporalEvidence


# 功能：
#   构造独立观测身份夹具，不用模型调用次数或当前墙钟伪造传感器样本。
# 输入：
#   sequence：用于生成不同内容摘要的样本编号。
#   stream：所属任务来源标识。
#   timestamp：可选的显式来源 UNIX 毫秒时刻。
#   reset：是否显式重置历史。
# 输出：
#   value：带来源标识、时间和内容摘要的证据模型。
def evidence(sequence, stream="mission-a", timestamp=None, reset=False):
    value = TemporalEvidence(
        stream_id=stream,
        sample_sha256=sha256_json({"sample": sequence}),
        observed_at_unix_ms=1000 + sequence * 50 if timestamp is None else timestamp,
        reset_history=reset,
    )
    return value


# 功能：
#   同一来源的重复调用不能填满窗口，只有八个独立样本才能完成八行历史。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_history_requires_physical_samples_not_invocation_count():
    history = ObservationHistory(8)
    for _ in range(30):
        history.append(evidence(0), (1.0,), (2.0,))
    assert len(history.rows) == 1 and not history.ready
    for sequence in range(1, 8):
        history.append(evidence(sequence), (1.0,), (2.0,))
    assert history.ready
    assert len(history.rows) == 8


# 功能：
#   同一内容更换时刻或同一时刻更换内容均应拒绝，不能冒充新的物理样本。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_redating_or_mutating_same_timestamp_is_rejected():
    history = ObservationHistory(8)
    history.append(evidence(0), (1.0,), (2.0,))
    with pytest.raises(ValueError, match="REDATED"):
        history.append(evidence(0, timestamp=1020), (1.0,), (2.0,))
    with pytest.raises(ValueError, match="SAME_TIME_CONFLICT"):
        history.append(evidence(1, timestamp=1000), (1.0,), (2.0,))


# 功能：
#   来源变化、较长间隔或显式重启使旧历史失效，新来源从一条样本重新积累。
# 输入：
#   changed：需要重建窗口的新来源证据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "changed",
    [
        evidence(2, stream="mission-b"),
        evidence(2, timestamp=2000),
        evidence(2, timestamp=500, reset=True),
    ],
)
def test_stream_gap_and_explicit_restart_reset_history(changed):
    history = ObservationHistory(2)
    history.append(evidence(0), (1.0,), (2.0,))
    history.append(evidence(1), (1.0,), (2.0,))
    assert history.ready
    history.append(changed, (1.0,), (2.0,))
    assert len(history.rows) == 1 and not history.ready


# 功能：
#   未授权的来源时钟回退清空历史并报错，不复用回退前的满窗口。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_clock_regression_drops_old_history_and_raises():
    history = ObservationHistory(2)
    history.append(evidence(2), (1.0,), (2.0,))
    with pytest.raises(ValueError, match="CLOCK_REGRESSED"):
        history.append(evidence(1), (1.0,), (2.0,))
    assert not history.rows


# 功能：
#   历史长度和间隔必须为真实整数，布尔值、小数和缺失值不能进入 deque 构造。
# 输入：
#   length：候选历史长度。
#   gap：候选最大间隔毫秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("length,gap", [(True, 250), (2, True), (2.5, 250), (2, 2.5), (None, 250)])
def test_history_bounds_are_strict_integers(length, gap):
    with pytest.raises(ValueError, match="BOUNDS_INVALID"):
        ObservationHistory(length, gap)


# 功能：
#   绕过模型校验的字符串重置标志不能允许回退来源重新进入控制历史。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_mutated_reset_flag_does_not_authorize_clock_regression():
    history = ObservationHistory(2)
    history.append(evidence(2), (1.,), (2.,))
    changed = evidence(1).model_copy(update={"reset_history": "false"})
    with pytest.raises(ValueError):
        history.append(changed, (1.,), (2.,))


# 功能：
#   输入容器后续修改不能重写已经记录的历史行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_history_owns_mutable_input_rows():
    history = ObservationHistory(2)
    state, payload = [1.], [2.]
    history.append(evidence(1), state, payload)
    state[0], payload[0] = 100., 200.
    assert history.rows[0] == ((1.,), (2.,))


# 功能：
#   历史拒绝非有限或非数值行，失败提交不能替换最后的合法来源身份。
# 输入：
#   state：非法状态行。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("state", [(float("nan"),), (float("inf"),), ("one",), (True,)])
def test_history_rejects_invalid_rows_before_advancing_identity(state):
    history = ObservationHistory(2)
    first = evidence(1)
    history.append(first, (1.,), ())
    with pytest.raises(ValueError):
        history.append(evidence(2), state, ())
    assert history.latest == first
    assert list(history.rows) == [((1.,), ())]
