"""Local file ownership and bounded values; no vehicle or network execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from dronedream_agent_core import runtime_control_io, runtime_interrupt
from dronedream_agent_core.contracts import Vector3
from dronedream_agent_core.plugin_values import plugin_json_value


# 功能：
#   验证运行期改令的旧写入入口也保留同名未归属暂存，不能在独占创建失败后删除它。
# 输入：
#   tmp_path：隔离输出目录。
#   monkeypatch：固定暂存随机标识以触发冲突。
# 输出：
#   None：不返回业务数据。
def test_interruption_atomic_write_preserves_existing_temporary(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_control_io, "uuid4", lambda: SimpleNamespace(hex="d" * 32))
    path = tmp_path / "command.json"
    temporary = tmp_path / f".command.json.{'d' * 32}.tmp"
    temporary.write_text("other writer", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runtime_interrupt._atomic_json(path, {"ready": True})
    assert temporary.read_text(encoding="utf-8") == "other writer"
    assert not path.exists()


# 功能：
#   验证改令证据写入前拒绝被破坏模型中的非有限值，不能转换成 null 后落盘。
# 输入：
#   tmp_path：应保持未创建的输出目录。
# 输出：
#   None：不返回业务数据。
def test_interruption_atomic_write_rejects_nonfinite_model_before_io(tmp_path):
    path = tmp_path / "new" / "command.json"
    payload = Vector3(x=1, y=0, z=0).model_copy(update={"x": float("nan")})
    with pytest.raises(ValueError):
        runtime_interrupt._atomic_json(path, payload)
    assert not path.parent.exists()


# 功能：
#   验证默认转换仍限制普通消息，但宿主能明确选择较大的有界制品预算。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_explicit_asset_budget_does_not_expand_default_message_budget():
    value = {"trace": "x" * (2 * 1024 * 1024)}
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        plugin_json_value(value)
    assert plugin_json_value(value, limit=4 * 1024 * 1024) == value


# 功能：
#   验证错误类型、非正数或超过宿主总上限的转换预算在处理内容前被拒绝。
# 输入：
#   limit：错误字节预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, 65 * 1024 * 1024])
def test_conversion_rejects_invalid_explicit_budget(limit):
    with pytest.raises(ValueError, match="BUDGET_INVALID"):
        plugin_json_value({}, limit=limit)


# 功能：
#   验证一个字节的合法标量仍可使用一个字节预算，预算预检不能额外占用正文长度。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_conversion_supports_minimum_positive_byte_budget():
    assert plugin_json_value(0, limit=1) == 0


# 功能：
#   验证对象读取在访问磁盘前拒绝无效宿主预算，不能先读取文件再发现配置错误。
# 输入：
#   tmp_path：不存在的测试文件目录。
#   limit：错误字节预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, 65 * 1024 * 1024])
def test_runtime_read_validates_budget_before_open(tmp_path, limit):
    with pytest.raises(ValueError, match="BUDGET_INVALID"):
        runtime_control_io.read_runtime_object(tmp_path / "missing", maximum_bytes=limit)


# 功能：
#   验证大型类型化替换制品保持独立读写预算，同时相同大小的普通消息仍被拒绝。
# 输入：
#   tmp_path：隔离的制品输出目录。
# 输出：
#   None：不返回业务数据。
def test_large_typed_replacement_uses_separate_budget(tmp_path):
    from test_runtime_revision import _replacement

    # 大字段只验证文件预算，不声称这是业务上可执行的改令参数。
    replacement = _replacement()
    replacement.amendment_parameters = {"trace": "x" * (2 * 1024 * 1024)}
    path = tmp_path / "replacement.json"
    runtime_interrupt._atomic_json(path, replacement)
    value = runtime_control_io.read_runtime_object(
        path,
        maximum_bytes=runtime_control_io.MAX_RUNTIME_REPLACEMENT_BYTES,
    )
    assert value == replacement.model_dump(mode="json")
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        runtime_interrupt._atomic_json(tmp_path / "ordinary.json", value)
    assert not (tmp_path / "ordinary.json").exists()


# 功能：
#   验证覆盖选项必须是真正布尔值，不能把字符串 false 当成允许覆盖已有证据。
# 输入：
#   tmp_path：已有证据目录。
#   mode：非法覆盖模式。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("mode", ["false", 1, 0, None])
def test_runtime_publication_rejects_untyped_replace_option(tmp_path, mode):
    path = tmp_path / "evidence.json"
    path.write_bytes(b'{"original":true}\n')
    with pytest.raises(ValueError, match="REPLACE_MODE_INVALID"):
        runtime_control_io.publish_runtime_json(path, {"other": True}, replace_existing=mode)
    assert path.read_bytes() == b'{"original":true}\n'


# 功能：
#   验证读锁重试不会超出剩余等待预算，耗尽预算后拒绝发布并清理自有暂存。
# 输入：
#   tmp_path：证据输出目录。
#   monkeypatch：模拟替换读锁与可控单调钟。
# 输出：
#   None：不返回业务数据。
def test_publication_retry_caps_sleep_to_remaining_budget(tmp_path, monkeypatch):
    clock, sleeps, attempts = [0.0], [], []
    original = Path.replace

    # 功能：
    #   前两次替换模拟 Windows 读锁，第三次恢复真实原子替换。
    # 输入：
    #   self：本次暂存路径。
    #   target：目标证据路径。
    # 输出：
    #   result：实际替换成功后的目标路径。
    def transient_lock(self, target):
        attempts.append(self)
        if len(attempts) < 3:
            clock[0] += 0.004
            raise PermissionError("reader lock")
        result = original(self, target)
        return result

    # 功能：
    #   推进隔离测试时钟，记录发布器实际请求等待的秒数。
    # 输入：
    #   seconds：重试等待预算。
    # 输出：
    #   None：不返回业务数据。
    def advance(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(Path, "replace", transient_lock)
    monkeypatch.setattr(runtime_control_io.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime_control_io.time, "sleep", advance)
    path = tmp_path / "evidence.json"
    with pytest.raises(PermissionError):
        runtime_control_io.publish_runtime_json(
            path,
            {"ready": True},
            replace_timeout_seconds=0.01,
            replace_retry_seconds=1.0,
        )
    assert sleeps == pytest.approx([0.006])
    assert not path.exists() and list(tmp_path.iterdir()) == []


# 功能：
#   验证非法重试参数在创建任何目录或暂存文件前被拒绝。
# 输入：
#   tmp_path：应保持未创建的输出目录。
#   field：重试配置字段。
#   value：非法秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["replace_timeout_seconds", "replace_retry_seconds"])
@pytest.mark.parametrize("value", [True, "1", -1, float("nan"), float("inf"), 10**400])
def test_publication_rejects_invalid_retry_before_io(tmp_path, field, value):
    path = tmp_path / "new" / "record.json"
    with pytest.raises(ValueError, match="RETRY"):
        runtime_control_io.publish_runtime_json(path, {}, **{field: value})
    assert not path.parent.exists()


# 功能：
#   验证控制文件转移不覆盖已处理证据，失败后源消息和原有目标均可恢复。
# 输入：
#   tmp_path：隔离的控制消息目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_transfer_never_overwrites_existing_receipt(tmp_path):
    source, destination = tmp_path / "inbox.json", tmp_path / "processed.json"
    source.write_bytes(b"new message")
    destination.write_bytes(b"prior message")
    with pytest.raises(FileExistsError):
        runtime_control_io.transfer_runtime_file(source, destination)
    assert source.read_bytes() == b"new message"
    assert destination.read_bytes() == b"prior message"


# 功能：
#   验证正常领取只移除原目录项，保留原始字节与文件身份。
# 输入：
#   tmp_path：隔离消息目录。
# 输出：
#   None：不返回业务数据。
def test_runtime_transfer_preserves_original_content(tmp_path):
    source, destination = tmp_path / "inbox.json", tmp_path / "claimed.json"
    source.write_bytes(b'{"exact":true}\n')
    original = source.stat()
    runtime_control_io.transfer_runtime_file(source, destination)
    assert not source.exists()
    assert runtime_control_io.os.path.samestat(original, destination.stat())
    assert destination.read_bytes() == b'{"exact":true}\n'


# 功能：
#   验证领取中源文件被替换时不误删后来者，同时保留已领取的旧证据。
# 输入：
#   tmp_path：隔离消息目录。
#   monkeypatch：仅在本次硬链接完成后替换源路径。
# 输出：
#   None：不返回业务数据。
def test_runtime_transfer_keeps_both_sides_when_source_changes(tmp_path, monkeypatch):
    source, destination = tmp_path / "inbox.json", tmp_path / "claimed.json"
    replacement = tmp_path / "replacement.json"
    source.write_bytes(b"old message")
    replacement.write_bytes(b"new message")
    original_link = runtime_control_io.os.link

    # 功能：
    #   在真实领取完成后模拟另一个发布者原子替换收件箱目录项。
    # 输入：
    #   old：领取来源。
    #   new：独占领取目标。
    # 输出：
    #   None：不返回业务数据。
    def replace_after_link(old, new):
        original_link(old, new)
        replacement.replace(source)

    monkeypatch.setattr(runtime_control_io.os, "link", replace_after_link)
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        runtime_control_io.transfer_runtime_file(source, destination)
    assert source.read_bytes() == b"new message"
    assert destination.read_bytes() == b"old message"
