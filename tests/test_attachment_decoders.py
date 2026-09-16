"""Discovery/parser integration fixtures, not user attachments or qualified flight assets."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

from PIL import Image

from dronedream_agent_core.plugin_api import build_discovered_extension_registry


# 功能：
#   通过实际插件注册器调用附件解码器，选择已识别且优先级最高的测试预览。
# 输入：
#   path：本测试创建的附件路径。
#   content_type：传入解码器的媒体类型。
# 输出：
#   result：优先级最高的已识别预览。
def _decode(path: Path, content_type: str = "application/octet-stream") -> dict[str, object]:
    registry = build_discovered_extension_registry()
    outputs, receipts = registry.invoke_multiple(
        "input.attachment-decoders",
        "decode_attachment",
        path=str(path),
        attachment_id="attachment-" + "a" * 32,
        display_name=path.name,
        content_type=content_type,
        size_bytes=path.stat().st_size,
        source_sha256="b" * 64,
    )
    assert all(receipt.outcome == "accepted" for receipt in receipts)
    candidates = [
        output for output in outputs if isinstance(output, dict) and output.get("accepted") is True
    ]
    result = max(candidates, key=lambda item: int(item.get("priority", 0)))
    return result


# 功能：
#   验证文本、地理空间、图像和 CAD 预览来自实际文件字节，不是固定默认值。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_text_geojson_image_and_cad_decoders_use_real_file_content(tmp_path: Path) -> None:
    text = tmp_path / "mission.md"
    text.write_text("# Mission\ninspect the west facade\n", encoding="utf-8")
    decoded_text = _decode(text, "text/markdown")
    assert decoded_text["decoded_kind"] == "text"
    assert "west facade" in str(decoded_text["text"])

    geojson = tmp_path / "area.geojson"
    geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": []},
                        "properties": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    decoded_geo = _decode(geojson, "application/geo+json")
    assert decoded_geo["decoded_kind"] == "geospatial"
    assert decoded_geo["structured_data"]["geometry_types"] == {"Polygon": 1}

    image = tmp_path / "scene.png"
    Image.new("RGB", (320, 180), (220, 20, 60)).save(image)
    decoded_image = _decode(image, "image/png")
    assert decoded_image["decoded_kind"] == "image"
    assert decoded_image["structured_data"]["width"] == 320
    assert decoded_image["model_input"]["requires_multimodal_preprocessor"] is True

    urdf = tmp_path / "vehicle.urdf"
    urdf.write_text(
        "<robot name='uav'><link name='base'/><joint name='fixed' type='fixed'>"
        "<parent link='base'/><child link='base'/></joint></robot>",
        encoding="utf-8",
    )
    decoded_urdf = _decode(urdf, "application/xml")
    assert decoded_urdf["decoded_kind"] == "cad"
    assert decoded_urdf["structured_data"]["links"] == 1


# 功能：
#   验证真实点云头部和 SQLite 索引的预览，不把它们冒充完整消息或点云解码。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_point_cloud_and_rosbag_decoder_read_headers_and_database(tmp_path: Path) -> None:
    pcd = tmp_path / "scan.pcd"
    pcd.write_bytes(
        b"# .PCD v0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nWIDTH 2\nHEIGHT 1\n"
        b"POINTS 2\nDATA ascii\n0 0 0\n1 1 1\n"
    )
    decoded_cloud = _decode(pcd)
    assert decoded_cloud["decoded_kind"] == "point-cloud"
    assert "POINTS 2" in decoded_cloud["structured_data"]["declarations"]

    bag = tmp_path / "mission.db3"
    connection = sqlite3.connect(bag)
    connection.executescript(
        "CREATE TABLE topics(id INTEGER PRIMARY KEY,name TEXT,type TEXT,"
        "serialization_format TEXT,offered_qos_profiles TEXT);"
        "CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,"
        "timestamp INTEGER,data BLOB);"
        "INSERT INTO topics VALUES(1,'/pose','geometry_msgs/msg/PoseStamped','cdr','');"
        "INSERT INTO messages VALUES(1,1,1,x'00');"
    )
    connection.commit()
    connection.close()
    decoded_bag = _decode(bag)
    assert decoded_bag["decoded_kind"] == "rosbag"
    assert decoded_bag["structured_data"]["message_count"] == 1
    assert decoded_bag["structured_data"]["topics"][0]["name"] == "/pose"


# 功能：
#   验证最小 Office 文档通过插件链路提取工作表共享字符串与幻灯片正文。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_office_open_xml_decoders_extract_spreadsheet_and_slides(tmp_path: Path) -> None:
    workbook = tmp_path / "inspection.xlsx"
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            "<sst xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>"
            "<si><t>West facade</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            "<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>"
            "<sheetData><row><c t='s'><v>0</v></c><c><v>42</v></c></row></sheetData>"
            "</worksheet>",
        )
    decoded_workbook = _decode(
        workbook, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert decoded_workbook["decoded_kind"] == "document"
    assert "West facade" in decoded_workbook["text"]
    assert decoded_workbook["structured_data"]["cell_value_count"] == 2

    slides = tmp_path / "briefing.pptx"
    with zipfile.ZipFile(slides, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            "<p:sld xmlns:p='urn:p' xmlns:a='urn:a'><a:t>Inspect roof</a:t></p:sld>",
        )
    decoded_slides = _decode(
        slides, "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
    assert decoded_slides["decoded_kind"] == "document"
    assert "Inspect roof" in decoded_slides["text"]
    assert decoded_slides["structured_data"]["slide_count"] == 1
