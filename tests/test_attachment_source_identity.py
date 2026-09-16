"""Local preview reads must stay attached to the file that passed admission."""

from pathlib import Path

import pytest

import dronedream_agent_plugins.attachment_decoders as decoders


# 功能：
#   模拟文件检查之后、实际打开之前另一个写入者替换文件的竞争。
# 输入：
#   monkeypatch：替换 Path.open 的测试工具。
#   source：本测试拥有的输入文件。
#   replacement：本测试拥有的替换文件。
# 输出：
#   None：不返回业务数据。
def _replace_on_open(monkeypatch, source, replacement):
    original = Path.open
    replaced = False

    # 功能：
    #   在首次打开目标文件时执行测试替换，其他打开行为保持原样。
    # 输入：
    #   path：待打开路径。
    #   args：原始位置参数。
    #   kwargs：原始关键字参数。
    # 输出：
    #   stream：原始打开方法返回的文件流。
    def open_replaced(path, *args, **kwargs):
        nonlocal replaced
        if path == source and not replaced:
            replaced = True
            replacement.replace(source)
        stream = original(path, *args, **kwargs)
        return stream

    monkeypatch.setattr(Path, "open", open_replaced)


# 功能：
#   验证头部和文本预览不会把检查之后替换的内容作为原文件输出。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：注入竞争的测试工具。
#   kind：待验证的预览入口。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["prefix", "text"])
def test_preview_detects_replacement_between_stat_and_open(tmp_path, monkeypatch, kind):
    source = tmp_path / "source.txt"
    source.write_bytes(b"original")
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(b"different")
    _replace_on_open(monkeypatch, source, replacement)
    with pytest.raises(ValueError, match="FILE_CHANGED"):
        if kind == "prefix":
            decoders._prefix(source, 4)
        else:
            decoders._decode_text(path=str(source), content_type="text/plain")


# 功能：
#   验证头部读取上限不能是负数、布尔值、小数或超出固定上限。
# 输入：
#   tmp_path：本测试独占目录。
#   count：待拒绝的字节上限。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("count", [-1, True, 1.5, 128 * 1024 + 1])
def test_prefix_rejects_unbounded_or_coerced_count(tmp_path, count):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="PREFIX_LIMIT_INVALID"):
        decoders._prefix(source, count)


# 功能：
#   验证有界文本读取保留 UTF-8 字符和统一换行语义，并释放所有句柄。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：设置较小字符预算的测试工具。
# 输出：
#   None：不返回业务数据。
def test_text_preview_preserves_unicode_and_universal_newlines(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_bytes("飞行\r\n任务\r完成".encode())
    monkeypatch.setattr(decoders, "MAX_TEXT_CHARACTERS", 5)
    result = decoders._decode_text(path=str(source), content_type="text/plain")
    assert result["text"] == "飞行\n任务"
    assert result["structured_data"]["truncated"] is True
    assert result["structured_data"]["line_count"] == 2
    source.unlink()
    assert not source.exists()
