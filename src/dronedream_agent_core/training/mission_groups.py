"""One spatial-route holdout identity for demonstrations, DAgger and PPO.

Task names, seeds, vehicle variants, return traversals and render-only changes
cannot turn the same route into an independent validation group. This grouping
does not claim independence of nearby or partially overlapping different routes.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from dronedream_plugin_sdk.protocol import decode_json

from ..contracts import StrictModel
from ..hashing import sha256_json
from ..plugin_files import check_plain_plugin_path, read_plugin_file

SPATIAL_SPLIT_CONTRACT = "map-spatial-route-micrometre-coverage"
MAX_ROUTE_EVIDENCE_BYTES = 4 * 1024 * 1024


# 功能：
#   限定内容身份为小写 SHA-256，拒绝任务名、标签或非摘要值。
# 输入：
#   value：待检查的内容摘要。
# 输出：
#   value：通过检查的原摘要。
def _hash(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("MISSION_GROUP_CONTENT_HASH_INVALID")
    return value


# 功能：
#   将同地图下的无向空间线段覆盖规范化，使改名、反向、重走或共线细分保持同一分组。
#   不据此认定相邻或部分重叠的不同路线彼此独立。
# 输入：
#   route：含 positions_m 米制三维点列的路线；仅空间覆盖参与分组。
#   semantic_sha256：语义地图内容摘要。
# 输出：
#   group：微米量化后的空间覆盖摘要。
def spatial_mission_group(route: dict, semantic_sha256: str) -> str:
    _hash(semantic_sha256)
    raw = route.get("positions_m") if isinstance(route, dict) else None
    if not isinstance(raw, list) or not 2 <= len(raw) <= 10000:
        raise ValueError("MISSION_GROUP_METRIC_ROUTE_REQUIRED")
    # Micrometre resolution is far below sensor/route accuracy. Integer line
    # identities avoid floating-point cross-product overflow and give the same
    # coverage for a loop started mid-edge or a partially retraced segment.
    points = []
    for point in raw:
        if not isinstance(point, dict) or set(point) != {"x", "y", "z"}:
            raise ValueError("MISSION_GROUP_METRIC_POINT_INVALID")
        values = tuple(point[k] for k in ("x", "y", "z"))
        if any(isinstance(v, bool) or not isinstance(v, int | float)
               or abs(v) > 10_000_000 or not math.isfinite(v) for v in values):
            raise ValueError("MISSION_GROUP_METRIC_POINT_INVALID")
        values = tuple(round(v * 1_000_000) for v in values)
        if points and values == points[-1]:
            continue
        points.append(values)
    lines = {}
    for a, b in zip(points, points[1:], strict=False):
        delta = tuple(b[k] - a[k] for k in range(3))
        divisor = math.gcd(*delta)
        direction = tuple(v // divisor for v in delta)
        if next(v for v in direction if v) < 0:
            direction = tuple(-v for v in direction)
        moment = (a[1] * direction[2] - a[2] * direction[1],
                  a[2] * direction[0] - a[0] * direction[2],
                  a[0] * direction[1] - a[1] * direction[0])
        start, end = sorted((a, b))
        lines.setdefault((direction, moment), []).append((start, end))
    segments = []
    for intervals in lines.values():
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        segments.extend(merged)
    if not segments:
        raise ValueError("MISSION_GROUP_NONZERO_ROUTE_REQUIRED")
    group = sha256_json({"semantic_sha256": semantic_sha256,
                        "undirected_segments_um": sorted(segments)})
    return group


class MissionGroupEvidence(StrictModel):
    """Retain exact route bytes alongside derived map-bound spatial identity."""
    route_source_utf8: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(
        min_length=1, max_length=2 * 1024 * 1024)
    route_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    # 功能：
    #   从有界严格路线原文重算字节摘要和空间分组，拒绝重绑摘要的歧义或损坏记录。
    # 输入：
    #   self：保留路线原文及地图身份的候选证据。
    # 输出：
    #   self：内容及空间分组验证通过的证据。
    @model_validator(mode="after")
    def verify(self):
        content = self.route_source_utf8.encode("utf-8")
        route = decode_json(content, limit=MAX_ROUTE_EVIDENCE_BYTES, node_limit=1_000_000)
        if hashlib.sha256(content).hexdigest() != self.route_sha256:
            raise ValueError("MISSION_GROUP_ROUTE_CONTENT_CHANGED")
        if spatial_mission_group(route, self.semantic_sha256) != self.group_sha256:
            raise ValueError("MISSION_GROUP_SPATIAL_IDENTITY_CHANGED")
        return self


class MissionGroupManifest(StrictModel):
    """Map every sensory stream to a route group backed by retained route evidence."""
    method: Literal["map-spatial-route"] = "map-spatial-route"
    groups: dict[str, str]
    evidence: list[MissionGroupEvidence] = Field(min_length=1, max_length=10000)

    # 功能：
    #   重验嵌套证据，并要求每条传感器流映射到可由路线原文重建的分组。
    # 输入：
    #   self：流分组及路线证据清单。
    # 输出：
    #   self：所有流均具有有效路线证据的清单。
    @model_validator(mode="after")
    def verify(self):
        available = {MissionGroupEvidence.model_validate(item.model_dump()).group_sha256
                     for item in self.evidence}
        if not self.groups or any(not stream.strip() or group not in available
                                  for stream, group in self.groups.items()):
            raise ValueError("MISSION_GROUP_STREAM_EVIDENCE_MISSING")
        return self


# 功能：
#   保留路线原始空白与 Unicode，同时验证有界 JSON 和预期摘要并派生空间分组。
# 输入：
#   route_content：完整不可变 UTF-8 路线字节。
#   semantic_sha256：语义地图摘要。
#   expected_route_sha256：可选的独立预期路线摘要。
# 输出：
#   evidence：绑定原文、地图和空间覆盖的路线证据。
def mission_group_evidence(route_content: bytes, semantic_sha256: str, *,
                           expected_route_sha256: str | None = None) -> MissionGroupEvidence:
    if type(route_content) is not bytes:
        raise ValueError("MISSION_GROUP_ROUTE_BYTES_REQUIRED")
    route = decode_json(route_content, limit=MAX_ROUTE_EVIDENCE_BYTES, node_limit=1_000_000)
    digest = hashlib.sha256(route_content).hexdigest()
    if expected_route_sha256 is not None and digest != _hash(expected_route_sha256):
        raise ValueError("MISSION_GROUP_ROUTE_SOURCE_MISMATCH")
    evidence = MissionGroupEvidence(route_source_utf8=route_content.decode("utf-8"),
        route_sha256=digest, semantic_sha256=semantic_sha256,
        group_sha256=spatial_mission_group(route, semantic_sha256))
    return evidence


# 功能：
#   只读取运行自身路线或明确的原始兄弟计划，有界检查普通文件及摘要，不搜索同名新旧地图。
# 输入：
#   run_root：历史运行目录。
#   artifacts：该运行独立记录的语义地图和路线摘要。
#   retained_route：明确指定的原始路线文件，仅在运行和兄弟计划均缺失时使用。
# 输出：
#   evidence：与历史运行一致的路线分组证据。
def recorded_mission_group(run_root: Path, artifacts: dict, *, retained_route=None):
    run_root = Path(run_root).absolute()
    path = run_root / "mission-route.json"
    check_plain_plugin_path(path)
    if not path.exists():
        path = run_root.parent / "plan" / "route.json"
    if retained_route is not None:
        if path.exists():
            raise ValueError("MISSION_GROUP_RETAINED_ROUTE_CANNOT_REPLACE_EXISTING_SOURCE")
        path = Path(retained_route).absolute()
        check_plain_plugin_path(path)
    # 仅“原路线不存在”允许明确的布局兼容；存在但损坏时必须失败，不能偷换为兄弟计划。
    content = read_plugin_file(path, limit=MAX_ROUTE_EVIDENCE_BYTES)
    evidence = mission_group_evidence(content, _hash(artifacts.get("semantic_sha256")),
                                      expected_route_sha256=_hash(artifacts.get("route_sha256")))
    return evidence


# 功能：
#   合并重验后的路线清单并按完整证据去重，禁止为已有传感器流改派另一条路线。
# 输入：
#   manifests：待合并的路线证据清单。
# 输出：
#   merged：拥有独立验证证据的合并清单。
def merge_mission_group_manifests(*manifests: MissionGroupManifest) -> MissionGroupManifest:
    groups, evidence = {}, {}
    for manifest in manifests:
        manifest = MissionGroupManifest.model_validate(manifest.model_dump())
        for stream, group in manifest.groups.items():
            if stream in groups and groups[stream] != group:
                raise ValueError("MISSION_GROUP_STREAM_CONFLICT")
            groups[stream] = group
        for item in manifest.evidence:
            evidence[sha256_json(item)] = item
    merged = MissionGroupManifest(groups=groups, evidence=list(evidence.values()))
    return merged
