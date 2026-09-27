"""Live source framing does not change source time or depend on sampled logs."""

import base64
import hashlib
import zlib
from pathlib import Path

import pytest

from dronedream_agent_core.local_packet_channel import LatestPacketPublisher, LatestPacketReceiver
from dronedream_agent_core.localization_source_channel import (
    LOCALIZATION_SOURCE_CONTRACT,
    MAX_RAW_BYTES,
    decode_localization_source,
    encode_localization_source,
)


# 功能：创建不带运动权限且保留源时刻的有界协议夹具。
# 输入：stamp：观测的原始毫秒时刻。
# 输出：独立测量对象。
def record(stamp=10):
    return {"schema_version": "dronedream.live-geometry-observation.v1",
            "motion_permission_granted": False, "truth_correction_applied": False,
            "scan": {"observed_at_unix_ms": stamp, "source_stamp_ns": 123456789,
                     "samples": [{"x": 1.2, "y": 3.4}] * 512},
            "map_sha256": "a" * 64}


# 功能：检查无损传输不会改写时刻或原始对象。
# 输入：协议夹具。
# 输出：逐字段相等断言。
def test_round_trip_does_not_retime_or_mutate():
    original = record()
    payload = encode_localization_source(original)
    assert decode_localization_source(payload) == original
    assert payload["updated_at_unix_ms"] == 10
    assert len(payload["data"]) < 56_000


@pytest.mark.parametrize("key,value", [
    ("data", "!"), ("data", "x" * 56_001), ("data", None),
    ("sha256", "0" * 64), ("updated_at_unix_ms", True),
    ("updated_at_unix_ms", 11), ("schema_version", "wrong"),
], ids=["base64", "oversize", "missing", "digest", "bool-time", "retimed", "schema"])
# 功能：逐项拒绝非法传输字段。
# 输入：被篡改的字段和值。
# 输出：明确异常断言。
def test_invalid_packet_rejected(key, value):
    payload = encode_localization_source(record())
    payload[key] = value
    with pytest.raises(ValueError):
        decode_localization_source(payload)


@pytest.mark.parametrize("suffix", [b"trailing", zlib.compress(b"second stream")])
# 功能：阻止附加压缩流或尾随字节绕过单包验证。
# 输入：尾随内容。
# 输出：拒绝断言。
def test_extra_compressed_data_rejected(suffix):
    payload = encode_localization_source(record())
    payload["data"] = base64.b64encode(base64.b64decode(payload["data"]) + suffix).decode()
    with pytest.raises(ValueError):
        decode_localization_source(payload)


# 功能：验证解压预算及截断检测，不分配无界缓冲区。
# 输入：超预算或截断报文。
# 输出：拒绝断言。
def test_decompression_budget_and_truncation():
    payload = encode_localization_source(record())
    for raw, compressed in [(b"x" * (MAX_RAW_BYTES + 1), None), (b"x", b"x")]:
        payload["data"] = base64.b64encode(compressed or zlib.compress(raw)).decode()
        payload["sha256"] = hashlib.sha256(raw).hexdigest()
        with pytest.raises(ValueError):
            decode_localization_source(payload)


@pytest.mark.parametrize("field", ["motion_permission_granted", "truth_correction_applied"])
# 功能：防止测量通道混入控制或真值修正许可。
# 输入：伪造的权限字段。
# 输出：拒绝断言。
def test_channel_does_not_carry_authority(field):
    value = record()
    value[field] = True
    with pytest.raises(ValueError):
        encode_localization_source(value)


# 功能：实际验证运行独占通道、最新观测选择、旧时刻拒绝及关闭清理。
# 输入：pytest 独立临时目录。
# 输出：端到端来源与清理断言。
def test_latest_run_scoped_channel(tmp_path):
    path = tmp_path / "localization-source.json"
    publisher = LatestPacketPublisher(path, contract=LOCALIZATION_SOURCE_CONTRACT)
    assert publisher.send(encode_localization_source(record())) is False
    receiver = LatestPacketReceiver(path, contract=LOCALIZATION_SOURCE_CONTRACT)
    try:
        for stamp in (10, 11, 12):
            assert publisher.send(encode_localization_source(record(stamp)))
        assert decode_localization_source(receiver.read_latest()) == record(12)
        assert receiver.read_latest() is None
        assert publisher.send(encode_localization_source(record(9)))
        assert receiver.read_latest() is None  # Original source time may not regress.
    finally:
        publisher.close()
        receiver.close()
    assert not path.exists()


# 功能：出站不再解压不等于信任报文；接收端拒绝伪造摘要和权限，并能接收下一帧。
@pytest.mark.parametrize('corruption', ['digest', 'authority', 'trailing'])
def test_receiver_still_validates_full_measurement(tmp_path, corruption):
    path = tmp_path / 'source.json'
    receiver = LatestPacketReceiver(path, contract=LOCALIZATION_SOURCE_CONTRACT)
    sender = LatestPacketPublisher(path, contract=LOCALIZATION_SOURCE_CONTRACT)
    try:
        value = record()
        payload = encode_localization_source(value)
        if corruption == 'digest':
            payload['sha256'] = '0'*64
        elif corruption == 'authority':
            import json
            value['motion_permission_granted'] = True
            raw = json.dumps(value).encode()
            payload.update(data=base64.b64encode(zlib.compress(raw)).decode(),
                           sha256=hashlib.sha256(raw).hexdigest())
        else:
            payload['data'] = base64.b64encode(base64.b64decode(payload['data'])+b'trailing').decode()
        assert sender.send(payload)
        assert receiver.read_latest() is None and receiver.rejected == 1
        assert sender.send(encode_localization_source(record(11)))
        assert decode_localization_source(receiver.read_latest()) == record(11)
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("value", [None, [], {}, {"scan": None}, {"scan": {}},
                                  {"scan": {"observed_at_unix_ms": True}}])
# 功能：统一拒绝缺失或非法来源，不泄漏偶然 KeyError 或隐式转换。
# 输入：各类无效测量结构。
# 输出：值错误断言。
def test_encode_invalid_source(value):
    with pytest.raises(ValueError):
        encode_localization_source(value)


# 功能：锁定定位来源的独立调度，防止后续避障工作再次阻塞连续定位。
# 输入：产品深度工作器源代码。
# 输出：独立生产者在主循环前启动，避障仍完整组装，且先于原生缓冲关闭。
def test_measurement_publication_has_independent_owner():
    code = (Path(__file__).parents[1] / 'scripts/runtime_depth_safety_worker.py').read_text(
        encoding='utf-8')
    start = code.index('localization_producer.start()')
    assemble = code.index('frame = bridge.assemble(scan)', start)
    track = code.index('prepared_tracks = tracker.prepare(', assemble)
    close = code.index('close_localization_producer())', track)
    assert start < assemble < track < close < code.index('close_native_sampler()', close)
    assert 'localization_publisher.send(' not in code
    assert 'independent localization and development depth-drop injection ' in code
