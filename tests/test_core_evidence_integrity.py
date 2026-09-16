"""Core boundary fixtures; no recorded event substitutes for physical execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from test_px4_track import _semantic, _vehicle

from dronedream_agent_core.evidence import EvidenceChain
from dronedream_agent_core.observation_validity import ObservationValidity, image_control_deadline
from dronedream_agent_core.runtime_bindings import (
    MapRuntimeBindingsError,
    load_map_runtime_bindings,
    resolve_vehicle_collision_center_offset,
)


# 功能：
#   验证地图坐标不能通过布尔值或数字字符串隐式转换成可执行位置。
# 输入：
#   coordinate：错误类型的坐标。
# 输出：
#   None：解析器明确拒绝时通过。
@pytest.mark.parametrize("coordinate", [True, "1", None, float("nan")])
def test_runtime_binding_requires_actual_numbers(coordinate):
    semantic = _semantic()
    semantic["runtime_bindings"]["vehicle_spawn"]["x"] = coordinate
    with pytest.raises(MapRuntimeBindingsError, match="MAP_RUNTIME_BINDINGS_INVALID"):
        load_map_runtime_bindings(semantic)


# 功能：
#   验证返回的碰撞中心不能反向修改所选飞机资产，后续被破坏的几何也必须重新拒绝。
# 输入：
#   无；使用独立飞机夹具。
# 输出：
#   None：副本隔离及几何重校验成立时通过。
def test_vehicle_offset_is_detached_and_revalidated():
    vehicle = _vehicle()
    offset = resolve_vehicle_collision_center_offset(vehicle)
    offset.x = 99
    assert vehicle.collision_center_offset_model_m.x != 99
    vehicle.collision_center_offset_model_m.x = 99
    with pytest.raises(MapRuntimeBindingsError, match="OFFSET_INVALID"):
        resolve_vehicle_collision_center_offset(vehicle)


# 功能：
#   确认直接构造的非法时效记录不能因 Python 布尔数值兼容而获得控制时效。
# 输入：
#   无；创建边界夹具而非真实帧。
# 输出：
#   None：非法记录及非映射图像被拒绝时通过。
def test_invalid_deadline_receipt_cannot_authorize_dispatch():
    valid = ObservationValidity("control-eligible", 0, 250, 250, "fixture")
    assert valid.usable_at(10)
    assert not replace(valid, source_observed_at_unix_ms=True).usable_at(10)
    assert not replace(valid, control_deadline_unix_ms="250").usable_at(10)
    assert image_control_deadline(None, now_unix_ms=10) == 0


# 功能：
#   验证证据返回值和已保存记录不共享调用者的可变载荷。
# 输入：
#   tmp_path：测试专用目录。
# 输出：
#   None：修改原载荷不能改变新记录或落盘内容时通过。
def test_evidence_freezes_payload_before_hashing(tmp_path):
    chain = EvidenceChain(tmp_path / "events.jsonl")
    payload = {"position": {"x": 1}}
    record = chain.append("observation", payload)
    payload["position"]["x"] = 9
    assert record.payload["position"]["x"] == 1
    assert chain.read() == [record]


# 功能：
#   拒绝会导致摘要歧义或非标准 JSON 的载荷，不能在证据边界自动字符串化键。
# 输入：
#   tmp_path：测试专用目录。
#   payload：包含歧义键或非有限值的载荷。
# 输出：
#   None：错误输入未创建证据文件时通过。
@pytest.mark.parametrize("payload", [{1: "a", "1": "b"}, {"x": float("nan")}, {"x": (1, 2)}])
def test_ambiguous_payload_is_rejected_before_write(tmp_path, payload):
    path = tmp_path / "events.jsonl"
    chain = EvidenceChain(path)
    with pytest.raises(ValueError):
        chain.append("fixture", payload)
    assert not path.exists()


# 功能：
#   同值重复字段仍属于含义歧义，不能仅因为摘要相同就接受。
# 输入：
#   tmp_path：测试专用目录。
# 输出：
#   None：带重复键的证据被拒绝时通过。
def test_duplicate_evidence_keys_are_not_hidden_by_hash(tmp_path):
    path = tmp_path / "events.jsonl"
    chain = EvidenceChain(path)
    chain.append("fixture", {})
    path.write_bytes(
        path.read_bytes().replace(
            b'{"schema_version"', b'{"event_type":"fixture","schema_version"', 1
        )
    )
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        chain.read()


# 功能：
#   半行可能是中断写入，追加不能把新事件直接接到其后造成链损坏。
# 输入：
#   tmp_path：测试专用目录。
# 输出：
#   None：半行被拒绝且原现场保留时通过。
def test_unterminated_evidence_is_preserved_and_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    chain = EvidenceChain(path)
    chain.append("fixture", {})
    incomplete = path.read_bytes().rstrip(b"\n")
    path.write_bytes(incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        chain.append("next", {})
    assert path.read_bytes() == incomplete


# 功能：
#   验证同实例并发追加保持唯一序号和完整前驱链接。
# 输入：
#   tmp_path：测试专用目录。
# 输出：
#   None：全部真实文件写入后仍是连续有效链时通过。
def test_same_chain_serializes_threaded_appends(tmp_path):
    chain = EvidenceChain(tmp_path / "events.jsonl")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(chain.append, "fixture", {"index": i}) for i in range(12)]
        results = [future.result() for future in futures]
    records = chain.read()
    assert sorted(record.sequence for record in results) == list(range(1, 13))
    assert len(records) == 12


# 功能：
#   在读完前缀后模拟文件被改写，确认追加前还会对照实际打开文件的确切内容。
# 输入：
#   tmp_path：测试专用目录。
#   monkeypatch：只替换测试实例的读取回调。
# 输出：
#   None：被替换的文件不再被追加新事件时通过。
def test_prefix_replacement_between_read_and_write_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    chain = EvidenceChain(path)
    chain.append("fixture", {})
    original = chain._read_snapshot

    # 功能：
    #   保存旧快照后替换底层文件，产生确定性的读写间隔反例。
    # 输入：
    #   无；捕获原读取器和测试文件。
    # 输出：
    #   snapshot：替换前的快照。
    def replace_after_read():
        snapshot = original()
        path.write_bytes(b"replacement\n")
        return snapshot

    monkeypatch.setattr(chain, "_read_snapshot", replace_after_read)
    with pytest.raises(ValueError, match="prefix changed"):
        chain.append("next", {})
    assert path.read_bytes() == b"replacement\n"
