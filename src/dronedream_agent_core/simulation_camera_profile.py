"""Explicit, source-bound simulated camera stream profiles in a run-local overlay.

Installed Runtime assets remain untouched. A lower-bandwidth stream is a new
sensor configuration requiring validation, not an equivalent-image or flight
qualification claim. Model input tensor dimensions are not camera dimensions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .plugin_files import check_plain_plugin_path, portable_plugin_path, read_plugin_file
from .xml_values import parse_xml

CONTROL_STREAMS = {
    "IMX214": ("camera", 640, 360, 20),
    "StereoOV7251": ("depth_camera", 320, 240, 20),
}
CONTROL_PROFILES = {
    "low-latency": CONTROL_STREAMS,
    "compact-control": {
        "IMX214": ("camera", 320, 180, 20),
        "StereoOV7251": ("depth_camera", 160, 120, 20),
    },
    # Rates are in simulation time. On a slower-than-real-time host, explicit
    # faster sampling can avoid skipping otherwise fresh source snapshots.
    # Actual arrival/source cadence remains measured, never assumed to be 60 Hz.
    "responsive-control": {
        "IMX214": ("camera", 320, 180, 60),
        "StereoOV7251": ("depth_camera", 160, 120, 60),
    },
}
SIMULATION_CAMERA_CHOICES = ("native", *CONTROL_PROFILES)
_SUPPORTED_OPTICS = {"IMX214": (1.204, .1, 100.), "StereoOV7251": (1.274, .2, 19.1)}
_MAX_MODEL_BYTES = 1024 * 1024
_MAX_RECEIPT_BYTES = 131_072
_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_MAX_BUNDLE_ENTRIES = 1024
_MAX_BUNDLE_FILES = 256
_FLOAT_TEXT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


# 功能：
#   校验用户明确选择的相机配置和来源摘要，原生模式不能夹带派生配置的摘要。
# 输入：
#   profile：原生模式或当前支持的相机配置名称。
#   source_sha256：派生配置必须绑定的完整来源摘要；原生模式为 None。
# 输出：
#   None：不返回业务数据。
def validate_camera_profile_choice(profile: str, source_sha256: str | None) -> None:
    if not isinstance(profile, str) or profile not in SIMULATION_CAMERA_CHOICES:
        raise ValueError("SIMULATION_CAMERA_PROFILE_UNSUPPORTED")
    if ((profile == "native" and source_sha256 is not None)
            or (profile != "native" and (not isinstance(source_sha256, str)
                or re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None))):
        raise ValueError("SIMULATION_CAMERA_PROFILE_REQUIRES_EXPLICIT_SOURCE_BINDING")


# 功能：
#   为实际读取或写入的相机文件字节计算内容摘要，不据此授予飞行资格。
# 输入：
#   content：需要绑定身份的原始字节。
# 输出：
#   digest：完整小写 SHA-256 摘要。
def _sha(content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return digest


# 功能：
#   在模型字节和 XML 结构预算内解析相机来源，拒绝 DTD 和畸形内容。
# 输入：
#   content：相机模型的原始 bytes。
# 输出：
#   root：未执行任何插件的 XML 根节点。
def _camera_tree(content: bytes) -> ET.Element:
    if type(content) is not bytes:
        raise ValueError("SIMULATION_CAMERA_MODEL_INVALID")
    if len(content) > _MAX_MODEL_BYTES:
        raise ValueError("SIMULATION_CAMERA_MODEL_TOO_LARGE")
    try:
        root = parse_xml(content, maximum_bytes=_MAX_MODEL_BYTES)
    except (ValueError, ET.ParseError) as error:
        raise ValueError("SIMULATION_CAMERA_MODEL_INVALID") from error
    return root


# 功能：
#   获取唯一的直接子节点，防止 find 只读第一个节点而忽略重复配置。
# 输入：
#   parent：当前 XML 父节点。
#   tag：要求恰好出现一次的直接子标签。
# 输出：
#   element：唯一匹配的子节点。
def _one(parent: ET.Element, tag: str) -> ET.Element:
    matches = parent.findall(tag)
    if len(matches) != 1:
        raise ValueError("SIMULATION_CAMERA_CONFIGURATION_AMBIGUOUS")
    element = matches[0]
    return element


# 功能：
#   解析有界的十进制整数或浮点标量，拒绝嵌套节点、下划线和非标准数值文本。
# 输入：
#   parent：标量所属的 XML 父节点。
#   tag：唯一标量标签。
#   integer：尺寸字段为 True，其余物理数值为 False。
# 输出：
#   value：解析出的整数或浮点数，范围由配置校验继续确认。
def _number(parent: ET.Element, tag: str, *, integer: bool = False) -> int | float:
    element = _one(parent, tag)
    text = (element.text or "").strip()
    if len(element) or not 1 <= len(text) <= 64 or (
        re.fullmatch(r"[0-9]+", text) is None if integer else _FLOAT_TEXT.fullmatch(text) is None
    ):
        raise ValueError("SIMULATION_CAMERA_INTRINSICS_INVALID")
    value = int(text) if integer else float(text)
    return value


# 功能：
#   复用同一规则校验模型及回执的尺寸、频率和已标定光学参数，先比较范围避免整数溢出。
# 输入：
#   name：当前支持的 RGB 或深度传感器名称。
#   fields：六项相机配置值。
# 输出：
#   None：不返回业务数据。
def _validate_fields(name: str, fields: dict) -> None:
    if type(fields) is not dict or set(fields) != {
        "width", "height", "update_rate_hz", "horizontal_fov_rad", "near_m", "far_m"
    }:
        raise ValueError("SIMULATION_CAMERA_INTRINSICS_INVALID")
    if (any(type(fields[key]) is not int or not 0 < fields[key] <= 8192
            for key in ("width", "height"))
            or any(type(fields[key]) not in (int, float)
                   or not 0 < fields[key] <= sys.float_info.max
                   for key in ("update_rate_hz", "horizontal_fov_rad", "near_m", "far_m"))
            or fields["horizontal_fov_rad"] >= math.pi or fields["near_m"] >= fields["far_m"]
            or fields["update_rate_hz"] > 240):
        raise ValueError("SIMULATION_CAMERA_INTRINSICS_INVALID")
    optics = tuple(fields[key] for key in ("horizontal_fov_rad", "near_m", "far_m"))
    if optics != _SUPPORTED_OPTICS[name]:
        raise ValueError("SIMULATION_CAMERA_OPTICS_REQUIRE_ENCODER_CALIBRATION")


# 功能：
#   提取唯一 OakD 双相机配置并复核数值和已支持的标定，不把新摘要当作新光学标定证明。
# 输入：
#   content：原生或派生相机 SDF 字节。
# 输出：
#   result：RGB 与深度传感器各自的尺寸、频率和光学字段。
def camera_configuration(content: bytes) -> dict:
    root = _camera_tree(content)
    model = _one(root, "model")
    if root.tag != "sdf" or model.get("name") != "OakD-Lite" or model.findall(".//include"):
        raise ValueError("SIMULATION_CAMERA_MODEL_UNSUPPORTED")
    sensors = model.findall("link/sensor")
    if (len(sensors) != 2 or {s.get("name") for s in sensors} != set(CONTROL_STREAMS)
            or root.findall(".//sensor") != sensors):
        raise ValueError("SIMULATION_CAMERA_SENSOR_SET_UNSUPPORTED")
    result = {}
    for sensor in sensors:
        name = sensor.get("name")
        camera = _one(sensor, "camera")
        if sensor.get("type") != CONTROL_STREAMS[name][0]:
            raise ValueError("SIMULATION_CAMERA_TYPE_UNSUPPORTED")
        image, clip = _one(camera, "image"), _one(camera, "clip")
        fields = {"width": _number(image, "width", integer=True),
                  "height": _number(image, "height", integer=True),
                  "update_rate_hz": _number(sensor, "update_rate"),
                  "horizontal_fov_rad": _number(camera, "horizontal_fov"),
                  "near_m": _number(clip, "near"), "far_m": _number(clip, "far")}
        _validate_fields(name, fields)
        result[name] = fields
    return result


# 功能：
#   按明确名称选择当前相机配置，并返回独立表，避免调用方改写共享预设。
# 输入：
#   profile：三种派生相机配置之一。
# 输出：
#   streams：传感器名称对应类型、宽高和配置频率的独立字典。
def _profile_streams(profile: str) -> dict:
    if not isinstance(profile, str) or profile not in CONTROL_PROFILES:
        raise ValueError("SIMULATION_CAMERA_PROFILE_UNSUPPORTED")
    streams = dict(CONTROL_PROFILES[profile])
    return streams


# 功能：
#   只调整已绑定相机的宽高及配置频率，恢复白名单字段后逐节点核对其余 XML 未变。
# 输入：
#   content：已选定来源模型的实际字节。
#   profile：明确选择的派生配置名称。
# 输出：
#   result：派生模型字节与前后参数、摘要及限制说明组成的元组。
def control_stream_profile(content: bytes, *, profile: str = "low-latency") -> tuple[bytes, dict]:
    streams = _profile_streams(profile)
    before = camera_configuration(content)
    root = _camera_tree(content)
    for sensor in root.findall("model/link/sensor"):
        name = sensor.get("name")
        _, width, height, rate = streams[name]
        if width * before[name]["height"] != height * before[name]["width"]:
            raise ValueError("SIMULATION_CAMERA_PROFILE_ASPECT_RATIO_CHANGED")
        sensor.find("camera/image/width").text = str(width)
        sensor.find("camera/image/height").text = str(height)
        sensor.find("update_rate").text = str(rate)
    profiled = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    # 恢复白名单中的三项字段后与同一来源树比较，位姿、碰撞和其他配置不得悄悄改变。
    source = _camera_tree(content)
    source_sensors = {s.get("name"): s for s in source.findall("model/link/sensor")}
    for sensor in root.findall("model/link/sensor"):
        for field in ("camera/image/width", "camera/image/height", "update_rate"):
            sensor.find(field).text = source_sensors[sensor.get("name")].find(field).text
    if ET.tostring(root) != ET.tostring(source):
        raise ValueError("SIMULATION_CAMERA_PROFILE_CHANGED_UNRELATED_MODEL_DATA")
    result = profiled, {"before": before, "after": camera_configuration(profiled),
        "source_model_sha256": _sha(content), "profiled_model_sha256": _sha(profiled),
        "only_stream_dimensions_and_rates_changed": True,
        "same_field_of_view_and_aspect_ratio": True,
        "image_equivalence_claimed": False, "flight_qualification_granted": False}
    return result


# 功能：
#   逐项有界枚举并读取相机包，目录也占预算；拒绝链接、特殊文件和跨平台路径冲突。
# 输入：
#   bundle：相机模型及附属网格、纹理所在目录。
# 输出：
#   files：相对路径对应独立字节的有界快照，包含模型以外的全部文件。
def _read_camera_bundle(bundle: Path) -> dict[str, bytes]:
    check_plain_plugin_path(bundle)
    if not bundle.is_dir():
        raise ValueError("SIMULATION_CAMERA_SOURCE_BUNDLE_INVALID")
    files, total, entries = {}, 0, 0
    names: set[str] = set()
    pending = [bundle]
    while pending:
        directory = pending.pop()
        check_plain_plugin_path(directory)
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > _MAX_BUNDLE_ENTRIES:
                    raise ValueError("SIMULATION_CAMERA_BUNDLE_CAPACITY")
                path = Path(child.path)
                name = portable_plugin_path(path.relative_to(bundle).as_posix())
                if name.casefold() in names:
                    raise ValueError("SIMULATION_CAMERA_BUNDLE_PATH_COLLISION")
                names.add(name.casefold())
                metadata = child.stat(follow_symlinks=False)
                if (stat.S_ISLNK(metadata.st_mode)
                        or getattr(metadata, "st_file_attributes", 0) & 0x400):
                    raise ValueError("SIMULATION_CAMERA_SOURCE_LINK_FORBIDDEN")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                    continue
                if not stat.S_ISREG(metadata.st_mode) or len(files) >= _MAX_BUNDLE_FILES:
                    raise ValueError("SIMULATION_CAMERA_BUNDLE_CAPACITY")
                if name == "model.sdf" and metadata.st_size > _MAX_MODEL_BYTES:
                    raise ValueError("SIMULATION_CAMERA_MODEL_TOO_LARGE")
                remaining = _MAX_BUNDLE_BYTES - total
                if metadata.st_size > remaining:
                    raise ValueError("SIMULATION_CAMERA_BUNDLE_CAPACITY")
                content = read_plugin_file(path, limit=min(
                    remaining, _MAX_MODEL_BYTES if name == "model.sdf" else _MAX_BUNDLE_BYTES))
                total += len(content)
                files[name] = content
    return files


# 功能：
#   1. 冻结并绑定整个来源包，仅替换模型中的允许字段，在独占目录生成相机配置。
#   2. 创建输出前复核来源及回执预算，复制后验证摘要，不修改或覆盖已安装模型。
# 输入：
#   source_models：包含 OakD-Lite 包的来源模型目录。
#   output：必须尚不存在且位于来源包之外的运行输出目录。
#   expected_source_sha256：调用方已选定原始 model.sdf 的摘要。
#   profile：明确选择的派生相机配置。
# 输出：
#   result：运行模型资源根及完整来源、派生配置和文件摘要回执。
def prepare_camera_profile(*, source_models: Path, output: Path,
                           expected_source_sha256: str, profile: str = "low-latency") -> dict:
    _profile_streams(profile)
    if (type(expected_source_sha256) is not str
            or re.fullmatch(r"[a-f0-9]{64}", expected_source_sha256) is None):
        raise ValueError("SIMULATION_CAMERA_SOURCE_HASH_REQUIRED")
    check_plain_plugin_path(output)
    if output.exists():
        raise FileExistsError(output)
    bundle = source_models / "OakD-Lite"
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("SIMULATION_CAMERA_OUTPUT_INSIDE_SOURCE")
    files = _read_camera_bundle(bundle)
    source = files.get("model.sdf")
    if source is None or _sha(source) != expected_source_sha256:
        raise ValueError("SIMULATION_CAMERA_SOURCE_HASH_MISMATCH")
    profiled, receipt = control_stream_profile(source, profile=profile)
    overlay = {name: profiled if name == "model.sdf" else content
               for name, content in files.items()}
    source_hashes = {name: _sha(content) for name, content in files.items()}
    copied = {name: _sha(content) for name, content in overlay.items()}
    receipt.update(profile=profile, source_bundle_bytes=sum(len(data) for data in files.values()),
                   source_files=source_hashes, overlay_files=copied,
                   installed_assets_modified=False)
    # 生成端使用回读端相同的预算；不能成功写出一份回读时必定因过大而拒绝的回执。
    encode_json(receipt, limit=_MAX_RECEIPT_BYTES)
    rendered = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    if len(rendered.encode("utf-8")) > _MAX_RECEIPT_BYTES:
        raise ValueError("SIMULATION_CAMERA_RECEIPT_TOO_LARGE")
    if source_hashes != {name: _sha(data) for name, data in _read_camera_bundle(bundle).items()}:
        raise ValueError("SIMULATION_CAMERA_SOURCE_CHANGED_DURING_PREPARATION")
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_root = output / "models" / "OakD-Lite"
    model_root.mkdir(parents=True, exist_ok=False)
    for relative, data in overlay.items():
        target = model_root / relative
        check_plain_plugin_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        if _sha(read_plugin_file(target, limit=len(data))) != copied[relative]:
            raise ValueError("SIMULATION_CAMERA_COPY_VERIFICATION_FAILED")
    receipt_path = output / "camera-profile.json"
    check_plain_plugin_path(receipt_path)
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    result = {"model_resource_root": str(output / "models"), "receipt": receipt}
    return result


class CameraProfileReadback:
    """Check actual messages under the caller's sensor lock, without callback I/O."""

    # 功能：
    #   1. 严格读取回执并核对全部派生资源、配置及来源摘要关系，建立双流回读基线。
    #   2. 回执仅证明当前包与其描述相符，不认证回执作者、不授予飞行资格。
    # 输入：
    #   self：待初始化的相机回读器。
    #   receipt_path：运行专属相机回执路径。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, receipt_path: Path):
        raw = read_plugin_file(receipt_path, limit=_MAX_RECEIPT_BYTES)
        receipt = decode_json(raw, limit=_MAX_RECEIPT_BYTES)
        if type(receipt) is not dict:
            raise ValueError("SIMULATION_CAMERA_RECEIPT_INVALID")
        streams = _profile_streams(receipt.get("profile"))
        if (receipt.get("installed_assets_modified") is not False
                or receipt.get("flight_qualification_granted") is not False
                or receipt.get("image_equivalence_claimed") is not False
                or receipt.get("same_field_of_view_and_aspect_ratio") is not True
                or receipt.get("only_stream_dimensions_and_rates_changed") is not True
                or type(receipt.get("source_bundle_bytes")) is not int
                or not 0 < receipt["source_bundle_bytes"] <= _MAX_BUNDLE_BYTES):
            raise ValueError("SIMULATION_CAMERA_RECEIPT_INVALID")
        source_files, overlay_files = receipt.get("source_files"), receipt.get("overlay_files")
        if (type(source_files) is not dict or type(overlay_files) is not dict
                or not 1 <= len(source_files) <= _MAX_BUNDLE_FILES
                or source_files.keys() != overlay_files.keys() or "model.sdf" not in source_files):
            raise ValueError("SIMULATION_CAMERA_RECEIPT_INVALID")
        for name in source_files:
            portable_plugin_path(name)
            if any(type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                   for digest in (source_files[name], overlay_files[name])):
                raise ValueError("SIMULATION_CAMERA_RECEIPT_INVALID")
            if name != "model.sdf" and source_files[name] != overlay_files[name]:
                raise ValueError("SIMULATION_CAMERA_RECEIPT_INVALID")
        if (source_files["model.sdf"] != receipt.get("source_model_sha256")
                or overlay_files["model.sdf"] != receipt.get("profiled_model_sha256")):
            raise ValueError("SIMULATION_CAMERA_RECEIPT_INVALID")
        bundle = _read_camera_bundle(receipt_path.parent / "models/OakD-Lite")
        if {name: _sha(data) for name, data in bundle.items()} != overlay_files:
            raise ValueError("SIMULATION_CAMERA_OVERLAY_CHANGED")
        content = bundle["model.sdf"]
        config = camera_configuration(content)
        if config != receipt.get("after"):
            raise ValueError("SIMULATION_CAMERA_OVERLAY_CHANGED")
        before = receipt.get("before")
        if type(before) is not dict or before.keys() != config.keys():
            raise ValueError("SIMULATION_CAMERA_RECEIPT_INVALID")
        expected = {}
        for name, (_, width, height, rate) in streams.items():
            _validate_fields(name, before[name])
            fields = config[name]
            if width * before[name]["height"] != height * before[name]["width"]:
                raise ValueError("SIMULATION_CAMERA_PROFILE_ASPECT_RATIO_CHANGED")
            actual = fields["width"], fields["height"], fields["update_rate_hz"]
            if actual != (width, height, rate):
                raise ValueError("SIMULATION_CAMERA_PROFILE_STREAMS_CHANGED")
            expected["rgb" if name == "IMX214" else "depth"] = (width, height)
        if read_plugin_file(receipt_path, limit=_MAX_RECEIPT_BYTES) != raw:
            raise ValueError("SIMULATION_CAMERA_RECEIPT_CHANGED")
        self._expected = expected
        self.receipt_sha256 = _sha(raw)
        self.seen = set()
        self.error = None

    # 功能：
    #   提供预期尺寸的独立说明副本，调用方修改说明不会改变回读判定基线。
    # 输入：
    #   self：已绑定配置的回读器。
    # 输出：
    #   expected：RGB、深度流的预期宽高字典。
    @property
    def expected(self) -> dict[str, tuple[int, int]]:
        expected = dict(self._expected)
        return expected

    # 功能：
    #   在调用方传感器锁内核对消息宽高并登记流类别，任何配置错误保持失败状态。
    # 输入：
    #   self：已初始化的相机回读器。
    #   kind：rgb 或 depth 流名称。
    #   width：实际消息宽度。
    #   height：实际消息高度。
    # 输出：
    #   accepted：消息类型及尺寸正确且之前未失败时为 True。
    def observe(self, kind: str, width: int, height: int) -> bool:
        if self.error is not None:
            accepted = False
            return accepted
        if (type(kind) is not str or kind not in self._expected
                or type(width) is not int or type(height) is not int
                or (width, height) != self._expected[kind]):
            self.error = "SIMULATION_CAMERA_ACTUAL_STREAM_PROFILE_MISMATCH"
            accepted = False
            return accepted
        self.seen.add(kind)
        accepted = True
        return accepted

    # 功能：
    #   1. 要求两路流都实际出现且无配置错误；实际频率仍需独立测量，不声明飞行资格。
    #   2. 输出独立的 JSON 数组，避免内部元组导致跨进程发布失败或外部修改回读基线。
    # 输入：
    #   self：由传感器锁保护的回读器。
    # 输出：
    #   result：已验证宽高、回执摘要及频率校验状态组成的回读记录。
    def require_ready(self) -> dict:
        if self.error is not None:
            raise ValueError(self.error)
        if self.seen != set(self._expected):
            raise ValueError("SIMULATION_CAMERA_PROFILE_READBACK_PENDING")
        dimensions = {kind: list(size) for kind, size in self._expected.items()}
        result = {"verified_dimensions": dimensions,
                "receipt_sha256": self.receipt_sha256,
                "field_of_view_source": "unchanged source-bound camera SDF",
                "rate_verification": "configured; actual arrival timing is measured separately",
                "flight_qualification_granted": False}
        return result
