"""Immutable local expert inputs and spatial split evidence, without optimization."""

from dataclasses import dataclass
from pathlib import Path

from ..plugin_files import read_plugin_file
from .mission_groups import MissionGroupEvidence

MAX_RECORDED_SOURCE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class RecordedSource:
    """Freeze the bytes read once; later file changes cannot change this training source."""
    path: Path
    content: bytes

    # 功能：
    #   固定来源绝对路径并要求有界不可变字节，不把可变缓冲包装成冻结训练记录。
    # 输入：
    #   self：包含路径和已读取字节的候选来源。
    # 输出：
    #   None：不返回业务数据。
    def __post_init__(self):
        if type(self.content) is not bytes or len(self.content) > MAX_RECORDED_SOURCE_BYTES:
            raise ValueError("ADVISOR_SOURCE_BYTES_INVALID")
        object.__setattr__(self, "path", Path(self.path).absolute())

    # 功能：
    #   通过共享普通文件边界读取一次有界字节，拒绝链接、身份替换或读取期间增长。
    # 输入：
    #   cls：冻结来源类型。
    #   path：明确选中的训练来源文件。
    # 输出：
    #   source：固定路径及原始字节的来源快照。
    @classmethod
    def read(cls, path: Path):
        path = Path(path).absolute()
        source = cls(path, read_plugin_file(path, limit=MAX_RECORDED_SOURCE_BYTES))
        return source

    # 功能：
    #   返回已经固定的不可变字节，不重新读取原路径。
    # 输入：
    #   self：已冻结的训练来源。
    # 输出：
    #   content：最初读取的原始字节。
    def read_bytes(self) -> bytes:
        content = self.content
        return content

    # 功能：
    #   按指定字符编码严格解码冻结字节，不读取磁盘或替换非法字符。
    # 输入：
    #   self：冻结来源。
    #   encoding：用于解码的字符编码，默认 UTF-8。
    # 输出：
    #   text：完整解码后的来源文本。
    def read_text(self, *, encoding="utf-8") -> str:
        text = self.content.decode(encoding)
        return text

    # 功能：
    #   提供稳定的来源路径文本，仅作为名称而不触发读取。
    # 输入：
    #   self：固定绝对路径的来源。
    # 输出：
    #   text：来源绝对路径字符串。
    def __str__(self):
        text = str(self.path)
        return text


# 功能：
#   从有界来源列表提取可重算的地图路线分组，拒绝名称划分、损坏原文或地图身份错绑。
# 输入：
#   sources：一至一万条带路线证据和地图摘要的来源记录。
# 输出：
#   groups：通过独立路线证据校验的空间组集合。
def advisor_spatial_groups(sources: list[dict]) -> set[str]:
    if type(sources) is not list or not 1 <= len(sources) <= 10_000:
        raise ValueError("ADVISOR_BOUND_SPATIAL_SOURCES_REQUIRED")
    groups = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("ADVISOR_BOUND_SPATIAL_SOURCE_INVALID")
        candidate = source.get("mission_split")
        if isinstance(candidate, MissionGroupEvidence):
            candidate = candidate.model_dump()
        split = MissionGroupEvidence.model_validate(candidate)
        if split.semantic_sha256 != source.get("semantic_sha256"):
            raise ValueError("ADVISOR_SPATIAL_MAP_IDENTITY_MISMATCH")
        groups.add(split.group_sha256)
    return groups


# 功能：
#   训练开始前拒绝训练与验证路线的空间组重叠，不因文件名或回合名不同就当作独立数据。
# 输入：
#   training_sources：训练侧的来源记录列表。
#   validation_sources：验证侧的来源记录列表。
# 输出：
#   training：验证通过的训练空间组集合。
#   validation：验证通过且与训练集合不相交的验证空间组集合。
def validate_advisor_spatial_splits(training_sources, validation_sources):
    training, validation = (advisor_spatial_groups(sources)
                            for sources in (training_sources, validation_sources))
    if training & validation:
        raise ValueError("ADVISOR_TRAINING_VALIDATION_SPATIAL_ROUTE_OVERLAP")
    return training, validation
