import json
import time
from collections import deque
from threading import Lock

import pytest

from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation, Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training.outcome_verifier import (
    OutcomeEnvelope,
    SimulationOutcomeVerifier,
    swept_clearance_bound,
)
from dronedream_agent_core.training.runtime_evidence import (
    ControlReceiptMonitor,
    JsonlCursor,
    read_object,
)


# 功能：
#   构造源时间每帧递增 50 毫秒的独立仿真真值，用于窗口和训练评分测试。
# 输入：
#   sequence：从一开始的观测序号。
#   x：无人机在测试世界坐标中的横向位置，米。
# 输出：
#   observation：固定健康、速度和目标的类型化仿真观测。
def pose(sequence, x):
    observation = RuntimeLocalSafetyObservation(
        sequence=sequence, observed_at_unix_ms=1000 + sequence * 50,
        source="simulation-ground-truth", stream_healthy=True, stream_age_seconds=0,
        localization_covariance_m2=0, current_position_m=Vector3(x=x, y=0, z=1),
        current_velocity_mps=Vector3(x=1, y=0, z=0), target_position_m=Vector3(x=4, y=0, z=1),
    )
    return observation


WALL = {"center_x": 0, "center_y": 0, "center_z": 1,
        "size_x": .001, "size_y": 3, "size_z": 3}


# 功能：
#   验证扫掠体检查能发现两个无碰撞端点之间的薄墙，不能仅按端点奖励安全。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_swept_reward_verification_catches_wall_between_collision_free_endpoints():
    value = swept_clearance_bound((-1, 0, 1), (1, 0, 1), [WALL],
                                   radius_m=.1, half_height_m=.1)
    assert value < 0


# 功能：
#   验证奖励保留真实碰撞、未知能耗和未完成任务，并拒绝机载估计替代独立评分真值。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_independent_outcome_never_fabricates_mission_success_or_energy():
    verifier = SimulationOutcomeVerifier([WALL], OutcomeEnvelope(.1, .2, (-3, -3, 0), (3, 3, 3)))
    args = dict(observations=[pose(1, -1), pose(2, 1)], start_ms=1050, end_ms=1100,
                stage_id="office", goal_revision="a", goal_position_m=(2, 0, 1),
                safety_intervened=False, commanded_change_squared=0., power_joules=None)
    reward, receipt = verifier.evaluate(**args)
    assert reward.collision
    assert reward.power_joules is None
    assert not reward.verified_mission_complete
    assert reward.verifier_receipt_sha256 == sha256_json(receipt)
    args["observations"][0] = pose(1, -1).model_copy(update={"source": "onboard"})
    with pytest.raises(ValueError, match="INDEPENDENT"):
        verifier.evaluate(**args)


# 功能：
#   验证增量游标保留半行、完整行仅返回一次，并拒绝文件缩短后重用旧偏移。
# 输入：
#   tmp_path：测试 JSONL 的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_evidence_cursor_preserves_split_lines_without_replaying(tmp_path):
    path = tmp_path / "receipt.jsonl"
    cursor = JsonlCursor(path)
    assert cursor.read() == []
    path.write_bytes(b'{"x":')
    assert cursor.read() == []
    with path.open("ab") as stream:
        stream.write(b'1}\n{"x":2}\n')
    assert cursor.read() == [{"x": 1}, {"x": 2}]
    assert cursor.read() == []
    path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="TRUNCATED"):
        cursor.read()


# 功能：
#   核对独立评分监视器没有动作执行接口，防止真值读取通道承担控制职责。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_observer_only_environment_never_has_actuator_methods():
    from dronedream_agent_core.training.runtime_evidence import IndependentPoseMonitor

    assert not any(hasattr(IndependentPoseMonitor, method)
                   for method in ("step", "arm", "setpoint"))


# 功能：
#   验证起始基线必须不晚于输入且足够新鲜，最新基线不健康或通道失败时不能继续。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_first_step_waits_for_independent_baseline_not_future_truth():
    from dronedream_agent_core.training.runtime_evidence import IndependentPoseMonitor

    monitor = object.__new__(IndependentPoseMonitor)
    monitor._lock, monitor._rows, monitor._error = Lock(), deque(maxlen=512), None
    assert not monitor.has_start_baseline(1100)
    monitor._rows.append(pose(1, 0))  # First witness is at 1050 ms.
    assert not monitor.has_start_baseline(1049)
    assert monitor.has_start_baseline(1050)
    assert monitor.has_start_baseline(1150)
    assert not monitor.has_start_baseline(1151)
    monitor._rows.append(pose(2, 1))
    assert monitor.has_start_baseline(1151)
    monitor._rows[-1] = monitor._rows[-1].model_copy(update={"stream_healthy": False})
    assert not monitor.has_start_baseline(1151)
    monitor._error = "lost observer"
    with pytest.raises(ValueError, match="OUTCOME_MONITOR_FAILED"):
        monitor.has_start_baseline(1151)


# 功能：
#   先写接受回执再写命令，验证仅在两端摘要匹配后返回证据，且调用方收到独立副本。
# 输入：
#   tmp_path：两份独立写入流的测试目录。
# 输出：
#   None：不返回业务数据。
def test_receipt_monitor_joins_independently_flushed_streams(tmp_path):
    from test_control_execution_evidence import evidence

    application, data = evidence()
    commands = data["command_records"]
    command_path, application_path = tmp_path / "commands.jsonl", tmp_path / "accepted.jsonl"
    monitor = ControlReceiptMonitor(command_path, application_path)
    call_id = commands[0]["command"]["model_call_id"]
    try:
        application_path.write_text(json.dumps(application) + "\n", encoding="utf-8")
        assert monitor.poll(call_id) is None  # No forged join from acceptance alone.
        command_path.write_text(json.dumps(commands[0]) + "\n", encoding="utf-8")
        deadline = time.monotonic() + 2
        result = None
        while result is None and time.monotonic() < deadline:
            result = monitor.poll(call_id)
            time.sleep(.005)
        assert result is not None
        command, record = result
        assert record.command_sha256 == sha256_json(command)
        assert record.accepted_at_unix_ms == application["accepted_at_unix_ms"]
        command.decision.issue_codes.append("test-only-mutation")
        assert monitor.poll(call_id)[0].decision.issue_codes == []  # Detached caller copy.
        assert monitor.poll("model-" + "c" * 24) is None
    finally:
        monitor.close()


# 功能：
#   验证接受回执必须序号连续、时间不倒退，缺号不能补造或重新编号。
# 输入：
#   tmp_path：监视器测试目录。
#   change：选择缺号或接受时间倒退的损坏方式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("change", ["missing-sequence", "clock-regression"])
def test_receipt_monitor_rejects_broken_execution_order(tmp_path, change):
    from test_control_execution_evidence import evidence

    application, data = evidence()
    monitor = ControlReceiptMonitor(tmp_path / "commands", tmp_path / "receipts")
    try:
        monitor._ingest(data["command_records"], [application])
        following = dict(application)
        following["sequence"] = 3 if change == "missing-sequence" else 2
        if change == "clock-regression":
            following["accepted_at_unix_ms"] -= 1
        with pytest.raises(ValueError, match="REORDERED_OR_MISSING"):
            monitor._ingest([], [following])
    finally:
        monitor.close()


# 功能：
#   验证证据对象和逐行回执都拒绝歧义键、非有限数值及过深结构，不让后写字段覆盖前值。
# 输入：
#   tmp_path：本次证据文件的隔离目录。
#   raw：需要拒绝的原始 JSON 字节。
#   as_jsonl：选择逐行回执读取器或单对象读取器。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("as_jsonl", [False, True], ids=["object", "jsonl"])
@pytest.mark.parametrize("raw", [
    b'{"accepted":false,"accepted":true}',
    b'{"latency":NaN}',
    b'{"latency":Infinity}',
    b'{"latency":1e999}',
    b'{"nested":' + b'[' * 65 + b'0' + b']' * 65 + b'}',
])
def test_evidence_readers_reject_ambiguous_or_unbounded_values(tmp_path, raw, as_jsonl):
    path = tmp_path / "evidence.json"
    path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError):
        if as_jsonl:
            JsonlCursor(path).read()
        else:
            read_object(path)


# 功能：
#   验证有限嵌套 JSON 和中文内容仍能被两类读取器原样读取，校验增强不改变合法值。
# 输入：
#   tmp_path：本次证据文件目录。
# 输出：
#   None：不返回业务数据。
def test_evidence_readers_preserve_valid_nested_values(tmp_path):
    value = {"位置": [1, -2.5, None], "gates": {"accepted": False}, "sequence": 2}
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_object(path) == value
    cursor = JsonlCursor(path)
    assert cursor.read() == [value]
    assert cursor.read() == []
