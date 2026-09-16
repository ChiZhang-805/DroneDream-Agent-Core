"""Real local fixture parses for bounded Office/XML/header previews, without user files or tools."""

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from dronedream_agent_plugins._attachment_xml import OfficeArchive, parse_xml
from dronedream_agent_plugins.attachment_decoders import (
    _decode_bim_cad,
    _decode_binary,
    _decode_document,
    _decode_docx,
    _decode_geospatial,
    _decode_point_cloud,
    _decode_pptx,
    _decode_rosbag,
    _decode_text,
    _decode_xlsx,
)


# 功能：
#   创建测试独占的压缩文档，不把成员解压到文件系统。
# 输入：
#   path：测试文档路径。
#   members：成员名与内容的映射。
# 输出：
#   path：已写入的测试文档路径。
def _archive(path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


# 功能：
#   验证不同编码下均在实体展开之前拒绝 DTD。
# 输入：
#   encoding：XML 文档声明及实际采用的编码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_xml_rejects_doctype_before_entity_expansion(encoding):
    content = (
        '<?xml version="1.0" encoding="' + encoding + '"?>'
        '<!DOCTYPE a [<!ENTITY payload "expanded">]><a>&payload;</a>'
    ).encode(encoding)
    with pytest.raises(ValueError, match="DOCTYPE_FORBIDDEN"):
        parse_xml(content)


# 功能：
#   验证过深的 XML 在建树阶段被拒绝，而非完整分配之后才检查。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_xml_depth_limit_applies_while_building_the_tree():
    with pytest.raises(ValueError, match="STRUCTURE_LIMIT"):
        parse_xml(b"<a>" * 129 + b"</a>" * 129)


# 功能：
#   验证很小的压缩文件仍受真实解压字节数限制。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_office_xml_limit_uses_expanded_not_compressed_bytes(tmp_path):
    path = _archive(
        tmp_path / "compressed.docx",
        {"word/document.xml": b"<p>" + b" " * (8 * 1024 * 1024) + b"</p>"},
    )
    assert path.stat().st_size < 20_000
    with pytest.raises(ValueError, match="XML_SIZE_LIMIT"):
        _decode_docx(path)


# 功能：
#   验证重复的正文成员不会按最后一项静默选择。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_office_duplicate_member_is_ambiguous_not_last_write_wins(tmp_path):
    path = tmp_path / "duplicate.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", "<p/>")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("word/document.xml", "<p/>")
    with pytest.raises(ValueError, match="ENTRIES_INVALID"):
        OfficeArchive(path)


# 功能：
#   验证富文本片段按段落拼接，文字样式拆分不引入额外换行。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_docx_rich_text_runs_stay_in_the_same_paragraph(tmp_path):
    path = _archive(
        tmp_path / "paragraph.docx",
        {
            "word/document.xml": "<document><p><r><t>Flight</t></r><r><t> control</t></r></p>"
            "<p><r><t>Next paragraph</t></r></p></document>"
        },
    )
    text, metadata = _decode_docx(path)
    assert text == "Flight control\nNext paragraph"
    assert metadata["paragraph_count"] == 2


# 功能：
#   验证共享字符串索引包含空项，合并富文本并排除注音内容。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_xlsx_shared_index_counts_entries_not_rich_text_runs(tmp_path):
    path = _archive(
        tmp_path / "rich.xlsx",
        {
            "xl/sharedStrings.xml": "<sst><si><r><t>Office</t></r><r><t> roof</t></r>"
            "</si><si/><si><t>Gate</t><rPh><t>phonetic only</t></rPh></si></sst>",
            "xl/worksheets/sheet1.xml": '<worksheet><c t="s"><v>0</v></c>'
            '<c t="s"><v>1</v></c><c t="s"><v>2</v></c></worksheet>',
        },
    )
    text, metadata = _decode_xlsx(path)
    assert "Office roof\tGate" in text
    assert "phonetic only" not in text
    assert metadata["shared_string_count"] == 3


# 功能：
#   验证无效共享字符串索引被拒绝，外层文档入口保留明确失败状态。
# 输入：
#   tmp_path：本测试独占目录。
#   index：待拒绝的单元格索引文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("index", ["-1", "2", "0.5", "True"])
def test_xlsx_invalid_index_cannot_select_another_cell(tmp_path, index):
    path = _archive(
        tmp_path / "bad.xlsx",
        {
            "xl/sharedStrings.xml": "<sst><si><t>Gate</t></si></sst>",
            "xl/worksheets/sheet1.xml": f'<worksheet><c t="s"><v>{index}</v></c></worksheet>',
        },
    )
    with pytest.raises(ValueError, match="INDEX_INVALID"):
        _decode_xlsx(path)
    preview = _decode_document(path=str(path))
    assert preview["text"] is None
    assert preview["issue_codes"] == ["DOCUMENT_DECODE_FAILED:ValueError"]


# 功能：
#   验证演示关系决定幻灯片顺序，不按归档文件名排序。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_presentation_relationship_order_precedes_filename_order(tmp_path):
    path = _archive(
        tmp_path / "reordered.pptx",
        {
            "ppt/presentation.xml": '<presentation xmlns:r="urn:r">'
            '<sldId r:id="last"/><sldId r:id="first"/></presentation>',
            "ppt/_rels/presentation.xml.rels": "<Relationships>"
            '<Relationship Id="first" Target="slides/slide1.xml"/>'
            '<Relationship Id="last" Target="slides/slide10.xml"/></Relationships>',
            "ppt/slides/slide1.xml": "<sld><t>first-file</t></sld>",
            "ppt/slides/slide10.xml": "<sld><t>last-file</t></sld>",
        },
    )
    text, metadata = _decode_pptx(path)
    assert text.index("last-file") < text.index("first-file")
    assert metadata["slide_count"] == 2


# 功能：
#   验证无索引文档按数字顺序预览，有索引文档不跟随外部关系地址。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_index_free_preview_uses_numeric_order_and_never_follows_external_relationships(tmp_path):
    path = _archive(
        tmp_path / "numeric.pptx",
        {f"ppt/slides/slide{index}.xml": f"<sld><t>file-{index}</t></sld>" for index in (10, 2, 1)},
    )
    text, _ = _decode_pptx(path)
    assert text.index("file-1") < text.index("file-2") < text.index("file-10")
    path = _archive(
        tmp_path / "external.pptx",
        {
            "ppt/presentation.xml": '<presentation xmlns:r="urn:r">'
            '<sldId r:id="a"/></presentation>',
            "ppt/_rels/presentation.xml.rels": '<Relationships><Relationship Id="a" '
            'TargetMode="External" Target="https://example.invalid/private"/></Relationships>',
        },
    )
    with pytest.raises(ValueError, match="RELATIONSHIP_INVALID"):
        _decode_pptx(path)


# 功能：
#   验证二进制兜底只返回文件头，不调用无界的整文件读取方法。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：阻止整文件读取的测试工具。
# 输出：
#   None：不返回业务数据。
def test_header_preview_does_not_read_the_whole_attachment(tmp_path, monkeypatch):
    path = tmp_path / "unknown.bin"
    path.write_bytes(b"h" * 10000)

    # 功能：
    #   在解码器退回整文件读取时立即使测试失败。
    # 输入：
    #   _：本应只读文件头的路径对象。
    # 输出：
    #   None：不返回业务数据。
    def forbidden_read(_):
        raise AssertionError("read_bytes would allocate the entire attachment")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    result = _decode_binary(path=str(path))
    assert result["structured_data"]["header_hex"] == (b"h" * 64).hex()


# 功能：
#   验证不完整或伪造的点云头部不能被报告为有效预览。
# 输入：
#   tmp_path：本测试独占目录。
#   suffix：点云文件扩展名。
#   content：不合法的头部字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "suffix,content",
    [
        (".pcd", b"# DATA is only a comment\nPOINTS 10\n"),
        (".ply", b"bad magic\nend_header\n"),
        (".las", b"LASF"),
    ],
)
def test_point_cloud_preview_cannot_invent_missing_headers(tmp_path, suffix, content):
    path = tmp_path / ("bad" + suffix)
    path.write_bytes(content)
    with pytest.raises(ValueError):
        _decode_point_cloud(path=str(path))


# 功能：
#   验证文本输出有字符上限，截断标志和行数统计范围均明确。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_text_preview_reports_only_preview_line_count_and_truncation(tmp_path):
    path = tmp_path / "mission.txt"
    path.write_text("a" * 200_001, encoding="utf-8")
    result = _decode_text(path=str(path), content_type="text/plain")
    assert len(result["text"]) == 200_000
    assert result["structured_data"]["truncated"] is True
    assert result["structured_data"]["line_count_scope"] == "preview"


# 功能：
#   验证地理空间 JSON 中的 NaN 不进入后续几何统计。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_geospatial_preview_rejects_nonfinite_json(tmp_path):
    path = tmp_path / "nan.geojson"
    path.write_text('{"type":"FeatureCollection","features":[],"x":NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        _decode_geospatial(path=str(path))


# 功能：
#   验证 Unicode 与井号文件名按 SQLite URI 正确转义，读取真实测试索引。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_rosbag_uri_escapes_filename_and_queries_read_only(tmp_path):
    path = tmp_path / "mission #测试.db3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,"
            "serialization_format TEXT); CREATE TABLE messages(id INTEGER);"
            "INSERT INTO topics VALUES(1,'/pose','Pose','cdr');"
            "INSERT INTO messages VALUES(1);"
        )
    result = _decode_rosbag(path=str(path))
    assert result["issue_codes"] == []
    assert result["structured_data"]["message_count"] == 1
    assert result["structured_data"]["topics"][0]["name"] == "/pose"


# 功能：
#   验证错误要素、坐标维度及集合成员不能冒充合法空结果。
# 输入：
#   tmp_path：本测试独占目录。
#   payload：待拒绝的 GeoJSON 数据。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "payload",
    [
        {"type": "FeatureCollection", "features": {}},
        {"type": "FeatureCollection", "features": [1]},
        {"type": "not-a-geometry"},
        {"type": []},
        {"type": {}},
        {"type": "Point", "coordinates": [1]},
        {"type": "Point", "coordinates": [True, 2]},
        {"type": "GeometryCollection", "geometries": [None]},
    ],
)
def test_geospatial_invalid_structure_cannot_look_like_zero_features(tmp_path, payload):
    path = tmp_path / "invalid.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        _decode_geospatial(path=str(path))


# 功能：
#   验证单个要素及嵌套几何集合都按真实声明计数。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_single_feature_and_geometry_collection_are_counted(tmp_path):
    path = tmp_path / "single.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "geometry": {
                    "type": "GeometryCollection",
                    "geometries": [{"type": "Point", "coordinates": [1, 2]}],
                },
            }
        ),
        encoding="utf-8",
    )
    result = _decode_geospatial(path=str(path))["structured_data"]
    assert result["feature_count"] == 1
    assert result["geometry_types"] == {"GeometryCollection": 1, "Point": 1}


# 功能：
#   验证使用命名空间的模型元素不会被错误统计成零。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_model_preview_counts_namespaced_elements(tmp_path):
    path = tmp_path / "vehicle.urdf"
    path.write_text(
        '<robot xmlns="urn:fixture"><link><collision/><visual/></link><joint/></robot>',
        encoding="utf-8",
    )
    metadata = _decode_bim_cad(path=str(path))["structured_data"]
    assert {key: metadata[key] for key in ("links", "joints", "collisions", "visuals")} == {
        "links": 1,
        "joints": 1,
        "collisions": 1,
        "visuals": 1,
    }
