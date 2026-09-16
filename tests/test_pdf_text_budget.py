"""Validate text accounting with inert page doubles, without OCR or user documents."""

from pathlib import Path
from types import SimpleNamespace

import pypdf
import pytest

from dronedream_agent_plugins import _attachment_pdf_worker as worker


# 功能：
#   将固定页面文本交给真实文本预算逻辑，替换 PDF 解析与文件读取以隔离测试。
# 输入：
#   monkeypatch：替换读取器的测试工具。
#   texts：各页应返回的完整文本。
#   budget：本次允许返回的最大字符数。
# 输出：
#   result：工作者提取的文本和实际截断元数据。
def _extract(monkeypatch, texts, budget):
    pages = [SimpleNamespace(extract_text=lambda text=text: text) for text in texts]
    reader = SimpleNamespace(pages=pages, metadata={}, is_encrypted=False)
    monkeypatch.setattr(pypdf, "PdfReader", lambda source: reader)
    monkeypatch.setattr(worker, "read_plugin_file", lambda *args, **kwargs: b"")
    monkeypatch.setattr(worker, "MAX_TEXT_CHARACTERS", budget)
    result = worker.extract(Path("inert-fixture.pdf"))
    return result


# 功能：
#   验证页间换行也占预算，截断判断与实际交付文本一致，包括刚好装满和空白页。
# 输入：
#   monkeypatch：隔离解析与文件读取的测试工具。
#   texts：各页原始文本。
#   budget：字符预算。
#   expected：应交付的文本。
#   truncated：应报告的截断状态。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "texts,budget,expected,truncated",
    [
        (["abc", "def"], 6, "abc\n\nd", True),
        (["abc", "def"], 8, "abc\n\ndef", False),
        (["abcdef"], 6, "abcdef", False),
        (["abcdefg"], 6, "abcdef", True),
        (["abc", ""], 4, "abc", True),
        (["", ""], 2, "\n\n", False),
        (["", "abc"], 5, "\n\nabc", False),
    ],
)
def test_pdf_text_and_truncation_share_one_budget(monkeypatch, texts, budget, expected, truncated):
    result = _extract(monkeypatch, texts, budget)
    assert result["text"] == expected
    assert len(result["text"]) <= budget
    assert result["metadata"]["truncated"] is truncated


# 功能：
#   复现多页正文未超限但加上分隔符后超限的情况，要求真实报告不完整。
# 输入：
#   monkeypatch：隔离读取器的测试工具。
# 输出：
#   None：不返回业务数据。
def test_pdf_many_pages_report_separator_truncation(monkeypatch):
    result = _extract(monkeypatch, ["x" * 399] * 500, 200_000)
    assert len(result["text"]) == 200_000
    assert result["metadata"]["truncated"] is True


# 功能：
#   验证页面数量上限不会被空文本绕过，未提取页面仍使结果标为截断。
# 输入：
#   monkeypatch：隔离读取器的测试工具。
# 输出：
#   None：不返回业务数据。
def test_pdf_page_limit_is_reported_even_for_empty_pages(monkeypatch):
    result = _extract(monkeypatch, [""] * 501, 200_000)
    assert result["metadata"]["page_count"] == 501
    assert result["metadata"]["extracted_pages"] == 500
    assert result["metadata"]["truncated"] is True
