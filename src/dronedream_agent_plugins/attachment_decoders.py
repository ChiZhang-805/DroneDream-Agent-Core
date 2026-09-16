"""Bounded attachment previews and references, not learned perception or qualified flight assets."""

from __future__ import annotations

import io
import shutil
import sqlite3
import struct
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

from dronedream_agent_core.plugin_api import PluginDefinition
from dronedream_agent_core.plugin_files import check_plain_plugin_path, read_plugin_file
from dronedream_plugin_sdk.protocol import copy_json, decode_json, encode_json

from ._attachment_process import preview_process
from ._attachment_source import MAX_ATTACHMENT_BYTES, open_preview_source
from ._attachment_text import PreviewText
from ._attachment_xml import MAX_XML_BYTES, OfficeArchive, local_tag, read_xml
from ._helpers import hook_plugin

MAX_TEXT_CHARACTERS = 200_000


# 功能：
#   建立与输入对象隔离的附件预览；识别格式不代表成功解码或可以用于飞行。
# 输入：
#   accepted：是否识别该格式。
#   kind：解码类别。
#   priority：多个解码结果之间的选择优先级。
#   text：可选的文本预览。
#   structured_data：可选的格式或内容统计。
#   model_input：交给后续模型处理器的输入引用。
#   issue_codes：解码问题代码，使用者必须检查。
# 输出：
#   result：经过 JSON 边界验证和复制的预览对象。
def _base(
    *,
    accepted: bool,
    kind: str,
    priority: int,
    text: str | None = None,
    structured_data: dict[str, object] | None = None,
    model_input: dict[str, object] | None = None,
    issue_codes: list[str] | None = None,
) -> dict[str, object]:
    result = copy_json(
        {
            "accepted": accepted,
            "decoded_kind": kind,
            "priority": priority,
            "text": text,
            "structured_data": {} if structured_data is None else structured_data,
            "model_input": {} if model_input is None else model_input,
            "issue_codes": [] if issue_codes is None else issue_codes,
        }
    )
    return result


# 功能：
#   校验附件路径、静态链接和声明大小；具体读取器仍须复核打开后的内容。
# 输入：
#   value：调用方传入的路径字符串。
# 输出：
#   path：已解析的本地普通文件路径。
def _path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("ATTACHMENT_SOURCE_INVALID")
    path = Path(value)
    check_plain_plugin_path(path)
    path = path.resolve()
    if not path.is_file() or path.stat().st_size > MAX_ATTACHMENT_BYTES:
        raise ValueError("ATTACHMENT_SOURCE_INVALID")
    return path


# 功能：
#   只读取有界文件头并复核文件身份，不因附件允许较大体积就读取整个文件。
# 输入：
#   source：附件路径。
#   count：允许读取的字节数，取 0 至 128 KiB 的整数。
# 输出：
#   prefix：不超过 count 字节的文件头。
def _prefix(source: Path, count: int) -> bytes:
    if type(count) is not int or not 0 <= count <= 128 * 1024:
        raise ValueError("ATTACHMENT_PREFIX_LIMIT_INVALID")
    with open_preview_source(source) as stream:
        prefix = stream.read(count)
    return prefix


# 功能：
#   提取有界 UTF-8 文本预览，替换无效字节并统一换行；行数只代表预览部分。
# 输入：
#   path：附件路径。
#   content_type：调用方声明的媒体类型，结合扩展名判断是否接收。
#   _：统一插件协议中本解码器不使用的参数。
# 输出：
#   result：文本预览、截断标志及字符编码信息，或未识别结果。
def _decode_text(*, path: str, content_type: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    suffixes = {
        ".txt",
        ".md",
        ".csv",
        ".tsv",
        ".yaml",
        ".yml",
        ".toml",
        ".py",
        ".log",
        ".jsonl",
    }
    if source.suffix.lower() not in suffixes and not content_type.startswith("text/"):
        result = _base(accepted=False, kind="text", priority=10)
        return result
    with open_preview_source(source) as stream:
        reader = io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
        try:
            preview = reader.read(MAX_TEXT_CHARACTERS + 1)
        finally:
            # 外层读取器拥有原始句柄，还需通过该句柄做读取后校验。
            reader.detach()
    text = preview[:MAX_TEXT_CHARACTERS]
    result = _base(
        accepted=True,
        kind="text",
        priority=70,
        text=text,
        structured_data={
            "line_count": text.count("\n") + 1,
            "encoding": "utf-8",
            "truncated": len(preview) > MAX_TEXT_CHARACTERS,
            "line_count_scope": "preview",
        },
    )
    return result


# 功能：
#   在有资源及时限约束的子进程中解析 PDF，验证结构化返回内容后交回预览。
# 输入：
#   source：PDF 路径。
# 输出：
#   result：工作者返回的文本预览和元数据。
def _decode_pdf(source: Path) -> tuple[str, dict[str, object]]:
    command = (
        [sys.executable, "--attachment-pdf-worker"]
        if getattr(sys, "frozen", False)
        else [
            sys.executable,
            "-m",
            "dronedream_agent_plugins._attachment_pdf_worker",
        ]
    )
    payload = decode_json(
        preview_process(
            command,
            request=encode_json({"path": str(source)}, limit=4096).encode("utf-8"),
        )
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("text"), str)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise ValueError("ATTACHMENT_PDF_RESPONSE_INVALID")
    result = (payload["text"], payload["metadata"])
    return result


# 功能：
#   按段落和富文本顺序提取有界正文，不把每个文字片段误当成独立段落。
# 输入：
#   source：DOCX 文件路径。
# 输出：
#   result：正文预览与段落、文字节点、媒体数量及截断标志。
def _decode_docx(source: Path) -> tuple[str, dict[str, object]]:
    preview = PreviewText(MAX_TEXT_CHARACTERS)
    with OfficeArchive(source) as archive:
        root = archive.xml("word/document.xml")
        for paragraph in (item for item in root.iter() if local_tag(item) == "p"):
            if not preview.append("", separator="\n"):
                break
            for item in paragraph.iter():
                if local_tag(item) == "t" and not preview.append(item.text or ""):
                    break
            if preview.truncated:
                break
        media = [name for name in archive.names if name.startswith("word/media/")]
    result = (
        preview.render(),
        {
            "paragraph_count": sum(local_tag(item) == "p" for item in root.iter()),
            "paragraph_text_nodes": sum(local_tag(item) == "t" for item in root.iter()),
            "embedded_media_count": len(media),
            "truncated": preview.truncated,
        },
    )
    return result


# 功能：
#   1. 按工作簿顺序读取单元格，按 si 解析共享字符串并保留富文本及空项的索引。
#   2. 边读取边限制标题、单元格和分隔符的总长度，不展开预算外的重复长字符串。
# 输入：
#   source：XLSX 文件路径。
# 输出：
#   result：预览文字、部件总数、实际处理数量及截断标志。
def _decode_xlsx(source: Path) -> tuple[str, dict[str, object]]:
    preview = PreviewText(MAX_TEXT_CHARACTERS)
    with OfficeArchive(source) as archive:
        worksheets, worksheet_count = archive.ordered_parts(
            index="xl/workbook.xml", child_tag="sheet", prefix="xl/worksheets/sheet", limit=256
        )
        shared = []
        if "xl/sharedStrings.xml" in archive.names:
            root = archive.xml("xl/sharedStrings.xml")
            # si, not each t, owns an index. Preserve empty entries and exclude
            # phonetic annotation runs, which are not the displayed cell text.
            for entry in root:
                if local_tag(entry) != "si":
                    continue
                shared.append(
                    "".join(
                        (part.text or "")
                        if local_tag(part) == "t"
                        else "".join(child.text or "" for child in part if local_tag(child) == "t")
                        for part in entry
                        if local_tag(part) in {"t", "r"}
                    )
                )
        cell_count = 0
        extracted_worksheets = 0
        for name in worksheets:
            root = archive.xml(name)
            extracted_worksheets += 1
            if not preview.append(f"[{Path(name).stem}]\n", separator="\n\n"):
                break
            first_value = True
            for cell in (item for item in root.iter() if local_tag(item) == "c"):
                cell_type = cell.attrib.get("t")
                value = next((item.text for item in cell if local_tag(item) == "v"), None)
                if value is None:
                    inline = [
                        item.text for item in cell.iter() if local_tag(item) == "t" and item.text
                    ]
                    value = "".join(inline) if inline else None
                if value is None:
                    continue
                if cell_type == "s":
                    if not value.isascii() or not value.isdecimal() or len(value) > 10:
                        raise ValueError("XLSX_SHARED_STRING_INDEX_INVALID")
                    index = int(value)
                    if index >= len(shared):
                        raise ValueError("XLSX_SHARED_STRING_INDEX_INVALID")
                    value = shared[index]
                if value:
                    cell_count += 1
                    if not preview.append(value, separator="" if first_value else "\t"):
                        break
                    first_value = False
            if preview.truncated:
                break
    result = (
        preview.render(),
        {
            "worksheet_count": worksheet_count,
            "extracted_worksheets": extracted_worksheets,
            "truncated": preview.truncated or extracted_worksheets < worksheet_count,
            "cell_value_count": cell_count,
            "shared_string_count": len(shared),
        },
    )
    return result


# 功能：
#   按演示顺序提取幻灯片文字，将标题、节点及页间分隔符都纳入同一预览预算。
# 输入：
#   source：PPTX 文件路径。
# 输出：
#   result：预览文字、幻灯片总数、实际处理数量及截断标志。
def _decode_pptx(source: Path) -> tuple[str, dict[str, object]]:
    preview = PreviewText(MAX_TEXT_CHARACTERS)
    with OfficeArchive(source) as archive:
        slides, slide_count = archive.ordered_parts(
            index="ppt/presentation.xml", child_tag="sldId", prefix="ppt/slides/slide", limit=500
        )
        text_nodes = 0
        extracted_slides = 0
        for index, name in enumerate(slides, start=1):
            root = archive.xml(name)
            extracted_slides += 1
            if not preview.append(f"[slide {index}]\n", separator="\n\n"):
                break
            first_value = True
            for item in root.iter():
                if local_tag(item) in {"t", "v"} and item.text and item.text.strip():
                    text_nodes += 1
                    if not preview.append(item.text.strip(), separator="" if first_value else "\n"):
                        break
                    first_value = False
            if preview.truncated:
                break
    result = (
        preview.render(),
        {
            "slide_count": slide_count,
            "extracted_slides": extracted_slides,
            "truncated": preview.truncated or extracted_slides < slide_count,
            "text_node_count": text_nodes,
        },
    )
    return result


# 功能：
#   分派 PDF 和 Office 文档预览；失败时保留问题代码，不回传可能含路径的异常正文。
# 输入：
#   path：文档路径。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：文档预览、未识别结果或带错误代码的识别结果。
def _decode_document(*, path: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    try:
        if source.suffix.lower() == ".pdf":
            text, metadata = _decode_pdf(source)
        elif source.suffix.lower() == ".docx":
            text, metadata = _decode_docx(source)
        elif source.suffix.lower() == ".xlsx":
            text, metadata = _decode_xlsx(source)
        elif source.suffix.lower() == ".pptx":
            text, metadata = _decode_pptx(source)
        else:
            result = _base(accepted=False, kind="document", priority=10)
            return result
    except Exception as error:
        result = _base(
            accepted=True,
            kind="document",
            priority=80,
            issue_codes=[f"DOCUMENT_DECODE_FAILED:{type(error).__name__}"],
            structured_data={"suffix": source.suffix.lower()},
        )
        return result
    result = _base(
        accepted=True,
        kind="document",
        priority=90,
        text=text,
        structured_data=metadata,
    )
    return result


# 功能：
#   读取图像尺寸、格式、帧数和有限 EXIF 信息；不将元数据预览冒充视觉模型理解。
# 输入：
#   path：图像附件路径。
#   content_type：用于识别的媒体类型。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：图像元数据与后续多模态处理引用，或未识别结果。
def _decode_image(*, path: str, content_type: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    if source.suffix.lower() not in suffixes and not content_type.startswith("image/"):
        result = _base(accepted=False, kind="image", priority=10)
        return result
    from PIL import Image

    with open_preview_source(source) as stream, Image.open(stream) as image:
        metadata = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "frames": int(getattr(image, "n_frames", 1)),
            "exif": {
                str(key): str(value)[:500] for key, value in islice(image.getexif().items(), 128)
            },
        }
    result = _base(
        accepted=True,
        kind="image",
        priority=90,
        structured_data=metadata,
        model_input={
            "type": "input_image_reference",
            "source_path": str(source),
            "requires_multimodal_preprocessor": True,
        },
    )
    return result


# 功能：
#   获取视频流元数据与待处理引用；缺少探测器或探测失败均明确记录。
# 输入：
#   path：视频附件路径。
#   content_type：用于识别的媒体类型。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：视频预览与问题代码，或未识别结果。
def _decode_video(*, path: str, content_type: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    suffixes = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    if source.suffix.lower() not in suffixes and not content_type.startswith("video/"):
        result = _base(accepted=False, kind="video", priority=10)
        return result
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        result = _base(
            accepted=True,
            kind="video",
            priority=80,
            structured_data={"suffix": source.suffix.lower()},
            model_input={
                "type": "input_video_reference",
                "source_path": str(source),
                "requires_multimodal_preprocessor": True,
            },
            issue_codes=["FFPROBE_NOT_AVAILABLE"],
        )
        return result
    try:
        metadata = _probe_media(ffprobe, source, video=True)
        issues = []
    except Exception as error:
        metadata = {}
        issues = [f"FFPROBE_FAILED:{type(error).__name__}"]
    result = _base(
        accepted=True,
        kind="video",
        priority=90,
        structured_data=metadata,
        model_input={
            "type": "input_video_reference",
            "source_path": str(source),
            "requires_multimodal_preprocessor": True,
        },
        issue_codes=issues,
    )
    return result


# 功能：
#   限制探测进程的输出、时间、协议和解复用格式；只读取元数据，不构成 OS 沙箱。
# 输入：
#   ffprobe：已找到的探测器可执行路径。
#   source：本地媒体路径，交由外部探测器打开。
#   video：是否选择视频字段和解复用器，否则使用音频配置。
# 输出：
#   value：通过 JSON 边界和对象类型验证的探测结果。
def _probe_media(ffprobe: str, source: Path, *, video: bool) -> dict[str, object]:
    formats = "mov,matroska,avi" if video else "wav,mp3,mov,flac,ogg,aac"
    entries = (
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate"
        if video
        else "format=duration,format_name:stream=codec_name,sample_rate,channels"
    )
    value = decode_json(
        preview_process(
            [
                ffprobe,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-format_whitelist",
                formats,
                "-probesize",
                "8388608",
                "-analyzeduration",
                "5000000",
                "-show_entries",
                entries,
                "-of",
                "json",
                str(source),
            ]
        )
    )
    if not isinstance(value, dict):
        raise ValueError("FFPROBE_RESPONSE_OBJECT_REQUIRED")
    return value


# 功能：
#   预览音频编解码及格式信息，记录探测错误；不执行语音识别。
# 输入：
#   path：音频附件路径。
#   content_type：用于识别的媒体类型。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：音频元数据与后续处理引用，或未识别结果。
def _decode_audio(*, path: str, content_type: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    suffixes = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
    if source.suffix.lower() not in suffixes and not content_type.startswith("audio/"):
        result = _base(accepted=False, kind="audio", priority=10)
        return result
    metadata: dict[str, object] = {"format": source.suffix.lower().lstrip(".")}
    issues: list[str] = []
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        try:
            metadata.update(_probe_media(ffprobe, source, video=False))
        except Exception as error:
            issues.append(f"AUDIO_FFPROBE_FAILED:{type(error).__name__}")
    else:
        issues.append("AUDIO_FFPROBE_UNAVAILABLE")
    result = _base(
        accepted=True,
        kind="audio",
        priority=85,
        structured_data=metadata,
        model_input={"type": "audio_reference", "source_path": str(source)},
        issue_codes=issues,
    )
    return result


# 功能：
#   检查有界 GeoJSON 的要素及坐标嵌套结构，统计几何类别；不验证拓扑或飞行适用性。
# 输入：
#   source：GeoJSON 文件路径。
# 输出：
#   result：空文本与要素数、几何类型计数和坐标参考声明。
def _decode_geojson(source: Path) -> tuple[str | None, dict[str, object]]:
    payload = decode_json(read_plugin_file(source, limit=MAX_XML_BYTES), limit=MAX_XML_BYTES)
    if not isinstance(payload, dict):
        raise ValueError("GEOSPATIAL_ROOT_INVALID")
    root_type = payload.get("type")
    if not isinstance(root_type, str):
        raise ValueError("GEOSPATIAL_ROOT_TYPE_INVALID")
    if root_type == "FeatureCollection":
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError("GEOSPATIAL_FEATURES_INVALID")
    elif root_type == "Feature":
        features = [payload]
    else:
        features = []
    geometry_types: Counter[str] = Counter()
    geometries = [] if root_type in {"FeatureCollection", "Feature"} else [payload]
    for feature in features:
        if (
            not isinstance(feature, dict)
            or feature.get("type") != "Feature"
            or "geometry" not in feature
        ):
            raise ValueError("GEOSPATIAL_FEATURE_INVALID")
        if feature["geometry"] is not None:
            geometries.append(feature["geometry"])
    # Structural preview, not topology validation or a source of qualified
    # coordinates. Finite JSON and global node/depth caps were checked above.
    depths = {
        "Point": 0,
        "MultiPoint": 1,
        "LineString": 1,
        "MultiLineString": 2,
        "Polygon": 2,
        "MultiPolygon": 3,
    }
    while geometries:
        geometry = geometries.pop()
        if not isinstance(geometry, dict):
            raise ValueError("GEOSPATIAL_GEOMETRY_INVALID")
        kind = geometry.get("type")
        if kind == "GeometryCollection":
            children = geometry.get("geometries")
            if not isinstance(children, list):
                raise ValueError("GEOSPATIAL_GEOMETRY_INVALID")
            geometries.extend(children)
        elif isinstance(kind, str) and kind in depths:
            pending = [(geometry.get("coordinates"), depths[kind])]
            while pending:
                coordinates, depth = pending.pop()
                if not isinstance(coordinates, list):
                    raise ValueError("GEOSPATIAL_COORDINATES_INVALID")
                if depth:
                    pending.extend((item, depth - 1) for item in coordinates)
                elif coordinates and (
                    len(coordinates) < 2
                    or any(type(number) not in (int, float) for number in coordinates)
                ):
                    raise ValueError("GEOSPATIAL_COORDINATES_INVALID")
        else:
            raise ValueError("GEOSPATIAL_GEOMETRY_TYPE_INVALID")
        geometry_types[kind] += 1
    result = (
        None,
        {
            "root_type": root_type,
            "feature_count": len(features),
            "geometry_types": dict(geometry_types),
            "coordinate_reference": payload.get("crs"),
        },
    )
    return result


# 功能：
#   识别 GeoJSON／KML／GPX 并给出结构预览；坐标系、单位及导航资格由资产导入另行校验。
# 输入：
#   path：地理空间附件路径。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：结构统计预览，或未识别结果。
def _decode_geospatial(*, path: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    if source.suffix.lower() not in {".geojson", ".kml", ".gpx"}:
        result = _base(accepted=False, kind="geospatial", priority=10)
        return result
    if source.suffix.lower() == ".geojson":
        text, metadata = _decode_geojson(source)
    else:
        root = read_xml(source)
        tags = Counter(item.tag.rsplit("}", 1)[-1] for item in root.iter())
        text = None
        metadata = {"root_tag": root.tag, "element_counts": dict(tags.most_common(64))}
    result = _base(
        accepted=True,
        kind="geospatial",
        priority=95,
        text=text,
        structured_data=metadata,
    )
    return result


# 功能：
#   验证并预览 PCD／PLY／LAS 有界头部，不把头部声明当作每个点均有效或无碰撞的证据。
# 输入：
#   path：点云附件路径。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：头部声明与点云引用，或未识别结果。
def _decode_point_cloud(*, path: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    suffix = source.suffix.lower()
    if suffix not in {".pcd", ".ply", ".las"}:
        result = _base(accepted=False, kind="point-cloud", priority=10)
        return result
    metadata: dict[str, object] = {"format": suffix[1:]}
    if suffix in {".pcd", ".ply"}:
        prefix = _prefix(source, 128 * 1024)
        lines = []
        terminated = False
        for raw_line in prefix.splitlines():
            line = raw_line.decode("ascii", errors="replace")
            lines.append(line)
            parts = line.strip().split()
            if (
                suffix == ".pcd"
                and len(parts) == 2
                and parts[0] == "DATA"
                and parts[1] in {"ascii", "binary", "binary_compressed"}
            ) or (suffix == ".ply" and parts == ["end_header"]):
                terminated = True
                break
        if not terminated or (suffix == ".ply" and (not lines or lines[0].strip() != "ply")):
            raise ValueError("POINT_CLOUD_HEADER_INVALID_OR_TOO_LARGE")
        header = "\n".join(lines)
        metadata["header"] = header[:20_000]
        for line in header.splitlines():
            parts = line.strip().split()
            if parts and parts[0].upper() in {"POINTS", "WIDTH", "HEIGHT", "ELEMENT"}:
                metadata.setdefault("declarations", []).append(line.strip())
    else:
        header = _prefix(source, 227)
        if len(header) < 227 or header[:4] != b"LASF":
            raise ValueError("LAS_HEADER_INVALID")
        header_size = struct.unpack_from("<H", header, 94)[0]
        point_offset = struct.unpack_from("<I", header, 96)[0]
        if not 227 <= header_size <= point_offset <= source.stat().st_size:
            raise ValueError("LAS_HEADER_INVALID")
        metadata.update(
            {
                "version": f"{header[24]}.{header[25]}",
                "header_size": header_size,
                "point_data_offset": point_offset,
                "legacy_point_count": struct.unpack_from("<I", header, 107)[0],
            }
        )
    result = _base(
        accepted=True,
        kind="point-cloud",
        priority=95,
        structured_data=metadata,
        model_input={"type": "point_cloud_reference", "source_path": str(source)},
    )
    return result


# 功能：
#   1. 只读查询 ROS 2 SQLite 索引，限制字段长度、主题数量和查询时间。
#   2. MCAP／ROS1 只检查文件头；消息负载解码交给对应 Runtime 适配器。
# 输入：
#   path：记录文件路径。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：索引预览、待处理引用及问题代码，或未识别结果。
def _decode_rosbag(*, path: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    suffix = source.suffix.lower()
    if suffix not in {".db3", ".mcap", ".bag"}:
        result = _base(accepted=False, kind="rosbag", priority=10)
        return result
    metadata: dict[str, object] = {"format": suffix[1:]}
    issues: list[str] = []
    if suffix == ".db3":
        try:
            # as_uri escapes #/? and Unicode filenames instead of treating them
            # as SQLite URI query/fragment syntax. Never create a missing bag.
            connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=2)
            try:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA trusted_schema=OFF")
                connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024 * 1024)
                deadline = time.monotonic() + 2
                connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                schema = dict(
                    connection.execute(
                        "SELECT name,type FROM sqlite_schema WHERE name IN ('topics','messages')"
                    )
                )
                if schema != {"topics": "table", "messages": "table"}:
                    raise sqlite3.DataError("bag index must use ordinary tables")
                topics = connection.execute(
                    "SELECT id,substr(name,1,513),substr(type,1,513),"
                    "substr(serialization_format,1,513) FROM topics ORDER BY id LIMIT 1025"
                ).fetchall()
                if len(topics) > 1024 or any(
                    type(row[0]) is not int
                    or row[0] < 0
                    or any(not isinstance(item, str) or len(item) > 512 for item in row[1:])
                    for row in topics
                ):
                    raise sqlite3.DataError("topic preview limit exceeded")
                message_count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            finally:
                connection.close()
            metadata.update(
                {
                    "topics": [
                        {
                            "id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "serialization_format": row[3],
                        }
                        for row in topics
                    ],
                    "message_count": message_count,
                }
            )
        except sqlite3.Error as error:
            issues.append(f"ROSBAG2_SQLITE_INVALID:{type(error).__name__}")
    else:
        header = _prefix(source, 16)
        if (suffix == ".mcap" and not header.startswith(b"\x89MCAP0\r\n")) or (
            suffix == ".bag" and not header.startswith(b"#ROSBAG V2.0\n")
        ):
            raise ValueError("ROSBAG_HEADER_INVALID")
        metadata["header_magic"] = header.hex()
        issues.append("ROSBAG_DEEP_INDEX_REQUIRES_RUNTIME_ADAPTER")
    result = _base(
        accepted=True,
        kind="rosbag",
        priority=95,
        structured_data=metadata,
        model_input={"type": "rosbag_reference", "source_path": str(source)},
        issue_codes=issues,
    )
    return result


# 功能：
#   预览 IFC 行内实体声明、XML 模型结构或 STL 头部提示，不执行重建或飞行资格认证。
# 输入：
#   path：BIM／CAD／飞行器模型附件路径。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：模型声明统计与原文件引用，或未识别结果。
def _decode_bim_cad(*, path: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    suffix = source.suffix.lower()
    if suffix not in {".ifc", ".urdf", ".sdf", ".dae", ".stl"}:
        result = _base(accepted=False, kind="cad", priority=10)
        return result
    metadata: dict[str, object] = {"format": suffix[1:]}
    kind = "bim" if suffix == ".ifc" else "cad"
    if suffix == ".ifc":
        text = read_plugin_file(source, limit=MAX_XML_BYTES).decode("utf-8", errors="replace")
        entity_types = Counter()
        for line in text.splitlines():
            if "=IFC" in line.upper():
                entity = line.upper().split("=", 1)[1].split("(", 1)[0]
                entity_types[entity] += 1
        metadata.update(
            {
                "entity_count": sum(entity_types.values()),
                "entity_types": dict(entity_types.most_common(100)),
            }
        )
    elif suffix in {".urdf", ".sdf", ".dae"}:
        root = read_xml(source)
        # ElementTree 的展开标签带命名空间；按裸路径查找会把存在的模型部件误报为零。
        tags = Counter(local_tag(item) for item in root.iter())
        metadata.update(
            {
                "root_tag": root.tag,
                "links": tags["link"],
                "joints": tags["joint"],
                "collisions": tags["collision"],
                "visuals": tags["visual"],
            }
        )
    else:
        prefix = _prefix(source, 84)
        if len(prefix) >= 84:
            metadata["triangle_count_binary_hint"] = struct.unpack_from("<I", prefix, 80)[0]
    result = _base(
        accepted=True,
        kind=kind,
        priority=95,
        structured_data=metadata,
        model_input={"type": f"{kind}_reference", "source_path": str(source)},
    )
    return result


# 功能：
#   为未获专门解码的二进制保留短头部证据，并明确标记能力缺失。
# 输入：
#   path：附件路径。
#   _：统一协议中本解码器不使用的参数。
# 输出：
#   result：低优先级的头部预览与问题代码。
def _decode_binary(*, path: str, **_: Any) -> dict[str, object]:
    source = _path(path)
    prefix = _prefix(source, 64)
    result = _base(
        accepted=True,
        kind="binary-metadata",
        priority=1,
        structured_data={
            "suffix": source.suffix.lower(),
            "header_hex": prefix.hex(),
        },
        issue_codes=["NO_SPECIALIZED_DECODER"],
    )
    return result


# 功能：
#   注册具有只读附件权限、失败隔离策略及大小声明的预览插件，不授予执行控制权限。
# 输入：
#   plugin_id：插件唯一标识。
#   name：界面名称。
#   description：解码能力说明。
#   order：插件展示排序。
#   decoder：接收统一附件输入的解码函数。
#   suffixes：用于发现支持格式的扩展名列表。
# 输出：
#   definition：可供插件注册器发现的定义。
def _definition(
    plugin_id: str,
    name: str,
    description: str,
    order: int,
    decoder,
    suffixes: list[str],
) -> PluginDefinition:
    definition = hook_plugin(
        module_name=__name__,
        plugin_id=plugin_id,
        name=name,
        description=description,
        capability_id=f"{plugin_id}.decode",
        capability_kind="attachment-decoder",
        capability_name=name,
        capability_description=description,
        category_id="input",
        category_label="输入与理解",
        slot_id="input.attachment-decoders",
        slot_label="附件解码器",
        activation_mode="multiple",
        category_order=20,
        slot_order=30,
        plugin_order=order,
        hooks={"decode_attachment": decoder},
        default_enabled=True,
        failure_mode="isolate",
        swap_policy="anytime",
        permissions=["mission.read", "attachment.read"],
        metadata={"suffixes": suffixes, "maximum_bytes": MAX_ATTACHMENT_BYTES},
    )
    return definition


# 功能：
#   汇总各格式预览插件与低优先级兜底插件，不用占位结果冒充未实现的感知能力。
# 输入：
#   无。
# 输出：
#   definitions：全部内置附件预览插件定义。
def plugin_definitions() -> list[PluginDefinition]:
    definitions = [
        _definition(
            "attachment.text",
            "文本附件",
            "读取文本、Markdown、表格文本、日志和配置文件。",
            10,
            _decode_text,
            ["txt", "md", "csv", "tsv", "yaml", "toml", "py", "log", "jsonl"],
        ),
        _definition(
            "attachment.document",
            "PDF 与 Word 文档",
            "提取 PDF 页面文本、元数据以及 DOCX 正文和媒体统计。",
            20,
            _decode_document,
            ["pdf", "docx", "xlsx", "pptx"],
        ),
        _definition(
            "attachment.image",
            "图像附件",
            "读取图像尺寸、格式、帧数和 EXIF，并交给多模态预处理器。",
            30,
            _decode_image,
            ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        ),
        _definition(
            "attachment.video",
            "视频附件",
            "使用 ffprobe 读取真实编解码、时长、帧率和分辨率。",
            40,
            _decode_video,
            ["mp4", "mov", "mkv", "avi", "webm", "m4v"],
        ),
        _definition(
            "attachment.audio",
            "音频附件",
            "使用 ffprobe 读取真实编码、采样率、声道和时长。",
            45,
            _decode_audio,
            ["wav", "mp3", "m4a", "flac", "ogg", "opus", "aac"],
        ),
        _definition(
            "attachment.geospatial",
            "地理空间附件",
            "解析 GeoJSON、KML 和 GPX 的结构、要素和几何类型。",
            50,
            _decode_geospatial,
            ["geojson", "kml", "gpx"],
        ),
        _definition(
            "attachment.point-cloud",
            "点云附件",
            "读取 PCD、PLY 和 LAS 的真实头部、点数和字段结构。",
            60,
            _decode_point_cloud,
            ["pcd", "ply", "las"],
        ),
        _definition(
            "attachment.rosbag",
            "ROS Bag 附件",
            "读取 ROS 2 SQLite bag 的 Topic 和消息统计，并识别 MCAP/ROS1 bag。",
            70,
            _decode_rosbag,
            ["db3", "mcap", "bag"],
        ),
        _definition(
            "attachment.bim-cad",
            "BIM 与无人机模型附件",
            "解析 IFC、URDF、SDF、DAE 和 STL 的模型结构与几何元数据。",
            80,
            _decode_bim_cad,
            ["ifc", "urdf", "sdf", "dae", "stl"],
        ),
        _definition(
            "attachment.binary-metadata",
            "二进制附件兜底",
            "保留文件类型和头部证据，不把未知二进制误当成文本。",
            999,
            _decode_binary,
            ["*"],
        ),
    ]
    return definitions
