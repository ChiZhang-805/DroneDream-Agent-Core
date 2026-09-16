"""Real loopback transport tests; not flight qualification evidence."""

import hashlib
import hmac
import json
import socket
from types import SimpleNamespace

import pytest
from test_executed_control_training import teacher_evidence
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation, Vector3
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_safety_channel import (
    _FAMILY,
    LocalSafetyPublisher,
    LocalSafetyReceiver,
    _address,
)


# 功能：
#   将固定教师测试指令绑定到合成机载观测，保留原始到期时刻，不制造真实飞行证据。
# 输入：
#   sequence：本次观测序号。
#   generated：本次指令生成时刻，单位 Unix 毫秒。
# 输出：
#   result：具有相同观测摘要和序号的观测、指令二元组。
def pair(sequence=1, generated=1020):
    _, command, _, _ = teacher_evidence()
    observation = RuntimeLocalSafetyObservation(
        sequence=sequence,
        observed_at_unix_ms=1000,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.0,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0, y=0, z=1),
        current_velocity_mps=Vector3(x=0, y=0, z=0),
        target_position_m=Vector3(x=1, y=2, z=3),
    )
    result = (
        observation,
        command.model_copy(
            update={
                "observation_sha256": sha256_json(observation),
                "observation_sequence": sequence,
                "generated_at_unix_ms": generated,
            }
        ),
    )
    return result


# 功能：
#   验证真实回环通信成对交付、返回深副本、拒绝重放且不续租，关闭后释放自身端点。
# 输入：
#   tmp_path：测试通信目录。
# 输出：
#   None：不返回业务数据。
def test_atomic_pair_detached_bounded_and_original_lease_preserved(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver, publisher = LocalSafetyReceiver(path), LocalSafetyPublisher(path)
    try:
        assert receiver.read_latest() is None
        obs, command = pair()
        assert publisher.send(obs, command)
        actual = receiver.read_latest()
        assert actual == (obs, command)
        actual[0].current_position_m.x = 42
        assert receiver.read_latest()[0].current_position_m.x == 0
        assert publisher.send(obs, command)
        assert receiver.read_latest()[1].valid_until_unix_ms == 1200
        assert receiver.rejected == 1  # Repetition is not a new source or lease.
        assert publisher.send(*pair(2, 1030))
        assert receiver.read_latest()[0].sequence == 2
        publisher.send(obs, command)
        assert receiver.read_latest()[0].sequence == 2
        assert receiver.accepted == 2 and receiver.rejected == 2
    finally:
        publisher.close()
        receiver.close()
    assert not path.exists()
    assert receiver.read_latest() is None
    with pytest.raises(RuntimeError, match="CLOSED"):
        publisher.send(*pair())


# 功能：
#   认证失败、观测摘要错配和超长数据报均不能替换已经接受的指令。
# 输入：
#   tmp_path：测试端点目录。
# 输出：
#   None：不返回业务数据。
def test_untrusted_malformed_and_mismatched_datagrams_cannot_replace_pair(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver, publisher = LocalSafetyReceiver(path), LocalSafetyPublisher(path)
    try:
        publisher.send(*pair())
        receiver.read_latest()
        descriptor = json.loads(path.read_text())
        with socket.socket(_FAMILY, socket.SOCK_DGRAM) as sender:
            endpoint = _address(descriptor)
            sender.sendto(b"x" * 100, endpoint)
            obs, command = pair(2, 1030)
            body = json.dumps(
                {
                    "observation": obs.model_dump(mode="json"),
                    "command": command.model_copy(
                        update={
                            "observation_sha256": "0" * 64,
                        }
                    ).model_dump(mode="json"),
                }
            ).encode()
            signature = (
                hmac.new(bytes.fromhex(descriptor["secret"]), body, hashlib.sha256)
                .hexdigest()
                .encode()
            )
            sender.sendto(signature + body, endpoint)
            sender.sendto(b"x" * 60_001, endpoint)
        assert receiver.read_latest()[0].sequence == 1
        assert receiver.rejected == 3
        with pytest.raises(ValueError, match="PAIR_MISMATCH"):
            publisher.send(obs, command.model_copy(update={"observation_sha256": "f" * 64}))
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   接收端未就绪时不发送，已有端点不得被新运行覆盖，关闭旧接收端不删未知替换内容。
# 输入：
#   tmp_path：测试端点目录。
# 输出：
#   None：不返回业务数据。
def test_new_session_never_reuses_or_deletes_unknown_endpoint(tmp_path):
    path = tmp_path / "endpoint.json"
    publisher = LocalSafetyPublisher(path)
    assert publisher.send(*pair()) is False
    receiver = LocalSafetyReceiver(path)
    try:
        with pytest.raises(FileExistsError):
            LocalSafetyReceiver(path)
        assert publisher.send(*pair())
        assert receiver.read_latest()
        path.write_text("replacement")
    finally:
        publisher.close()
        receiver.close()
    assert path.read_text() == "replacement"


# 功能：
#   实际执行器读取通道时仍执行原租期及派发余量，过期不会回退读取文件取得旧指令。
# 输入：
#   tmp_path：仅供测试回环通信的目录。
# 输出：
#   None：不返回业务数据。
def test_executor_channel_has_same_deadline_and_never_falls_back_to_files(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver, publisher = LocalSafetyReceiver(path), LocalSafetyPublisher(path)
    executor = _load_executor()
    try:
        args = SimpleNamespace(
            _local_safety_receiver=receiver,
            simulation_teacher_control=True,
            local_safety_command=tmp_path / "does-not-exist.json",
        )
        publisher.send(*pair())
        executor.time = SimpleNamespace(time=lambda: 1.019)
        assert executor._read_local_safety_command(args) is None
        executor.time = SimpleNamespace(time=lambda: 1.130)
        assert executor._read_local_safety_command(args) is not None
        executor.time = SimpleNamespace(time=lambda: 1.180)
        assert executor._read_local_safety_command(args) is not None
        executor.time = SimpleNamespace(time=lambda: 1.181)
        assert executor._read_local_safety_command(args) is None
        executor.time = SimpleNamespace(time=lambda: 1.3)
        assert executor._read_local_safety_command(args) is None
    finally:
        publisher.close()
        receiver.close()


# 功能：
#   发送端拒绝模拟器真值来源，即使其他内容与机载测试指令相似也不混用来源。
# 输入：
#   tmp_path：尚不存在端点的测试目录。
# 输出：
#   None：不返回业务数据。
def test_simulator_truth_is_not_an_onboard_control_pair(tmp_path):
    publisher = LocalSafetyPublisher(tmp_path / "absent-endpoint.json")
    try:
        observation, command = pair()
        with pytest.raises(ValueError, match="ONBOARD_EVIDENCE"):
            publisher.send(
                observation.model_copy(update={"source": "simulation-ground-truth"}), command
            )
    finally:
        publisher.close()


# 功能：
#   首次绑定的执行器关闭后，旧发布器不能自动把指令改发给同路径的新运行。
# 输入：
#   tmp_path：旧、新接收器依次使用的测试目录。
# 输出：
#   None：不返回业务数据。
def test_closed_executor_never_redirects_command_to_a_replacement_session(tmp_path):
    path = tmp_path / "endpoint.json"
    receiver, publisher = LocalSafetyReceiver(path), LocalSafetyPublisher(path)
    try:
        assert publisher.send(*pair())
        assert receiver.read_latest() is not None
        receiver.close()
        result = publisher.send(*pair(2, 1030))
        if _FAMILY != socket.AF_INET:
            assert result is False
        # UDP may accept a datagram without a listener; in either transport
        # the publisher's old run binding must not target a new executor.
        replacement = LocalSafetyReceiver(path)
        try:
            publisher.send(*pair(3, 1040))
            assert replacement.read_latest() is None
        finally:
            replacement.close()
    finally:
        publisher.close()
        receiver.close()
