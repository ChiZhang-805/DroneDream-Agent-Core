"""Bounded cleanup for replaceable desktop runtime artifacts."""

from __future__ import annotations

import os
import shutil
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_MISSION_RETENTION_DAYS = 30
RECENT_EXECUTIONS_PER_THREAD = 5
STAGING_RETENTION_HOURS = 24
QUARANTINE_RETENTION_DAYS = 7


@dataclass(frozen=True)
class _Candidate:
    path: Path
    category: str
    bytes: int


# 功能：判断路径是否为符号链接或 Windows 重解析点，防止清理越过应用目录。
# 输入：path：待检查路径。
# 输出：is_link：路径是否具有链接或重解析属性。
def _is_reparse_point(path: Path) -> bool:
    metadata = path.lstat()
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return path.is_symlink() or bool(attributes & reparse_flag)


# 功能：在不跟随链接的前提下计算普通目录树大小。
# 输入：path：已经限定在应用根目录内的候选路径。
# 输出：total：候选目录树的总字节数。
def _plain_tree_bytes(path: Path) -> int:
    """Measure one ordinary tree without following links or reparse points."""

    total = 0
    pending = [path]
    while pending:
        current = pending.pop()
        if _is_reparse_point(current):
            raise ValueError(f"STORAGE_CLEANUP_LINK_REJECTED:{current.name}")
        if current.is_file():
            total += current.stat().st_size
            continue
        if not current.is_dir():
            raise ValueError(f"STORAGE_CLEANUP_SPECIAL_FILE_REJECTED:{current.name}")
        with os.scandir(current) as entries:
            pending.extend(Path(entry.path) for entry in entries)
    return total


# 功能：删除已经复核的普通目录树，并仅为删除只读文件临时恢复写权限。
# 输入：path：已经通过所有权及内容复核的候选路径。
# 输出：None：删除成功后不返回业务数据。
def _remove_plain_tree(path: Path) -> None:
    """Remove an already validated ordinary tree, including read-only files."""

    # 功能：仅在删除只读文件失败后恢复目标写权限并重试同一删除操作。
    # 输入：function：原删除函数；target：精确失败路径；错误上下文不读取。
    # 输出：None：重试结果由原删除函数决定。
    def make_writable_and_retry(function: object, target: str, _: object) -> None:
        os.chmod(target, stat.S_IWRITE)
        function(target)  # type: ignore[operator]

    if path.is_dir():
        shutil.rmtree(path, onerror=make_writable_and_retry)
    else:
        path.chmod(stat.S_IWRITE)
        path.unlink()


# 功能：解析并确认候选路径严格位于应用数据根目录之下。
# 输入：root：应用数据根；candidate：待清理路径。
# 输出：candidate：规范化后的受管子路径。
def _owned_child(root: Path, candidate: Path) -> Path:
    root = root.resolve(strict=True)
    candidate = candidate.resolve(strict=True)
    if candidate == root or root not in candidate.parents:
        raise ValueError("STORAGE_CLEANUP_PATH_ESCAPED")
    return candidate


# 功能：比较候选最后修改时间与明确的 UTC 截止时间。
# 输入：path：候选路径；cutoff：保留期截止时间。
# 输出：expired：候选是否已经过期。
def _older_than(path: Path, cutoff: datetime) -> bool:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return modified < cutoff


# 功能：枚举一个受管暂存目录中超过保留期的直接子项。
# 输入：root：应用根；relative：暂存目录；cutoff：截止时间；category：报告分类。
# 输出：candidates：候选路径及分类列表。
def _children_older_than(
    root: Path,
    relative: str,
    cutoff: datetime,
    category: str,
) -> list[tuple[Path, str]]:
    directory = root / relative
    if not directory.is_dir() or _is_reparse_point(directory):
        return []
    return [
        (child, category)
        for child in directory.iterdir()
        if _older_than(child, cutoff)
    ]


# 功能：选择超过任务保留期且不属于每个会话最近五次运行的执行目录。
# 输入：root：应用根；cutoff：任务执行保留期截止时间。
# 输出：candidates：可再生成的过期执行目录列表。
def _mission_candidates(root: Path, cutoff: datetime) -> list[tuple[Path, str]]:
    missions = root / "missions"
    if not missions.is_dir() or _is_reparse_point(missions):
        return []
    candidates: list[tuple[Path, str]] = []
    for thread in missions.iterdir():
        if not thread.is_dir() or _is_reparse_point(thread):
            continue
        executions = sorted(
            (entry for entry in thread.iterdir() if entry.is_dir()),
            key=lambda entry: entry.stat().st_mtime,
            reverse=True,
        )
        for execution in executions[RECENT_EXECUTIONS_PER_THREAD:]:
            if _older_than(execution, cutoff):
                candidates.append((execution, "expired-mission-execution"))
    return candidates


# 功能：汇总应用拥有的过期暂存、隔离、导出和任务执行目录并测量大小。
# 输入：root：应用根；now：具有时区的当前时刻。
# 输出：candidates、issues：安全候选及拒绝原因。
def _collect(root: Path, now: datetime) -> tuple[list[_Candidate], list[str]]:
    root = root.resolve(strict=True)
    staging_cutoff = now - timedelta(hours=STAGING_RETENTION_HOURS)
    quarantine_cutoff = now - timedelta(days=QUARANTINE_RETENTION_DAYS)
    mission_cutoff = now - timedelta(days=DEFAULT_MISSION_RETENTION_DAYS)
    raw: list[tuple[Path, str]] = []
    raw.extend(_children_older_than(root, "plugin-staging", staging_cutoff, "plugin-staging"))
    raw.extend(
        _children_older_than(
            root, "remote-asset-staging", staging_cutoff, "remote-asset-staging"
        )
    )
    raw.extend(
        _children_older_than(
            root, "asset-quarantine", quarantine_cutoff, "asset-quarantine"
        )
    )
    raw.extend(_mission_candidates(root, mission_cutoff))
    raw.extend(
        (entry, "abandoned-export")
        for entry in root.glob("ddpkg-export-*")
        if _older_than(entry, staging_cutoff)
    )

    candidates: list[_Candidate] = []
    issues: list[str] = []
    for path, category in raw:
        try:
            owned = _owned_child(root, path)
            candidates.append(
                _Candidate(path=owned, category=category, bytes=_plain_tree_bytes(owned))
            )
        except (OSError, ValueError) as error:
            issues.append(str(error)[:240])
    candidates.sort(key=lambda item: str(item.path).casefold())
    return candidates, issues


class StorageMaintenanceService:
    """Inspect and remove only expired, replaceable data below one app root."""

    # 功能：绑定唯一应用数据根并建立进程内串行锁。
    # 输入：root：由 AppStore 管理的应用数据根。
    # 输出：None：初始化服务状态。
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self._lock = threading.Lock()

    # 功能：只读预览当前可安全清理的数据。
    # 输入：now：可选测试时钟。
    # 输出：report：候选数量、字节、分类和问题报告。
    def inspect(self, *, now: datetime | None = None) -> dict[str, object]:
        with self._lock:
            return self._run(apply=False, now=now)

    # 功能：清除经过所有权、年龄和链接检查的过期数据。
    # 输入：now：可选测试时钟。
    # 输出：report：实际删除数量、字节及未删除问题。
    def clean(self, *, now: datetime | None = None) -> dict[str, object]:
        with self._lock:
            return self._run(apply=True, now=now)

    # 功能：在同一候选算法上执行预览或清理，避免界面与实际删除规则分叉。
    # 输入：apply：是否实际删除；now：可选测试时钟。
    # 输出：report：结构化维护报告。
    def _run(self, *, apply: bool, now: datetime | None) -> dict[str, object]:
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            raise ValueError("STORAGE_CLEANUP_TIMEZONE_REQUIRED")
        candidates, issues = _collect(self.root, current.astimezone(UTC))
        removed_bytes = 0
        removed_count = 0
        if apply:
            for item in candidates:
                try:
                    owned = _owned_child(self.root, item.path)
                    if _plain_tree_bytes(owned) != item.bytes:
                        raise ValueError("STORAGE_CLEANUP_TREE_CHANGED")
                    _remove_plain_tree(owned)
                    removed_bytes += item.bytes
                    removed_count += 1
                except (OSError, ValueError) as error:
                    issues.append(str(error)[:240])
        by_category: dict[str, int] = {}
        for item in candidates:
            by_category[item.category] = by_category.get(item.category, 0) + item.bytes
        return {
            "schema_version": "dronedream.storage-maintenance.v1",
            "generated_at": current.astimezone(UTC).isoformat(),
            "automatic_retention_days": DEFAULT_MISSION_RETENTION_DAYS,
            "protected_recent_executions_per_thread": RECENT_EXECUTIONS_PER_THREAD,
            "candidate_count": 0 if apply and not issues else len(candidates) - removed_count,
            "reclaimable_bytes": (
                0
                if apply and not issues
                else sum(item.bytes for item in candidates) - removed_bytes
            ),
            "removed_count": removed_count,
            "removed_bytes": removed_bytes,
            "categories": by_category,
            "issues": issues,
        }
