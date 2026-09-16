"""Preview budgets include headings and separators, not just extracted text nodes."""

import zipfile

import pytest

import dronedream_agent_plugins.attachment_decoders as decoders


# 功能：
#   构造少量真实 Office XML 部件，不使用用户文档。
# 输入：
#   path：测试独占目标路径。
#   members：ZIP 成员名与 XML 文本的映射。
# 输出：
#   path：写入完成的测试文档路径。
def _document(path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return path


# 功能：
#   验证工作表／幻灯片之间的分隔符也计入输出预算，并正确报告截断。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：缩小输出预算的测试工具。
#   kind：工作表或幻灯片预览类型。
#   budget：本次字符上限。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("kind", ["xlsx", "pptx"])
@pytest.mark.parametrize("budget", [10, 12, 21, 22, 24, 25, 26])
def test_office_rendered_length_controls_truncation(tmp_path, monkeypatch, kind, budget):
    if kind == "xlsx":
        members = {
            f"xl/worksheets/sheet{index}.xml": "<worksheet><c><v>ab</v></c></worksheet>"
            for index in (1, 2)
        }
        expected = "[sheet1]\nab\n\n[sheet2]\nab"
        decoder = decoders._decode_xlsx
    else:
        members = {f"ppt/slides/slide{index}.xml": "<sld><t>ab</t></sld>" for index in (1, 2)}
        expected = "[slide 1]\nab\n\n[slide 2]\nab"
        decoder = decoders._decode_pptx
    path = _document(tmp_path / ("budget." + kind), members)
    monkeypatch.setattr(decoders, "MAX_TEXT_CHARACTERS", budget)
    text, metadata = decoder(path)
    assert text == expected[:budget]
    assert metadata["truncated"] is (len(expected) > budget)


# 功能：
#   验证长共享字符串超出预览预算后不再展开后续单元格。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：缩小输出预算的测试工具。
# 输出：
#   None：不返回业务数据。
def test_repeated_shared_strings_stop_at_preview_budget(tmp_path, monkeypatch):
    path = _document(
        tmp_path / "shared.xlsx",
        {
            "xl/sharedStrings.xml": "<sst><si><t>" + "a" * 20_000 + "</t></si></sst>",
            "xl/worksheets/sheet1.xml": "<worksheet>"
            + '<c t="s"><v>0</v></c>' * 100
            + "</worksheet>",
        },
    )
    monkeypatch.setattr(decoders, "MAX_TEXT_CHARACTERS", 100)
    text, metadata = decoders._decode_xlsx(path)
    assert text == "[sheet1]\n" + "a" * 91
    assert metadata["truncated"] is True
    assert metadata["cell_value_count"] == 1


# 功能：
#   验证完整单页内容刚好填满预算时，不误报还有内容被截断。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：设置精确字符预算的测试工具。
# 输出：
#   None：不返回业务数据。
def test_slide_exact_budget_is_not_truncated(tmp_path, monkeypatch):
    path = _document(tmp_path / "exact.pptx", {"ppt/slides/slide1.xml": "<sld><t>ab</t></sld>"})
    monkeypatch.setattr(decoders, "MAX_TEXT_CHARACTERS", len("[slide 1]\nab"))
    text, metadata = decoders._decode_pptx(path)
    assert text == "[slide 1]\nab"
    assert metadata["truncated"] is False
