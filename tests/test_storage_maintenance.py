"""Storage maintenance keeps user state while pruning expired generated data."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dronedream_agent_app.storage_maintenance import StorageMaintenanceService


# 功能：把测试候选修改时间调整到指定天数之前。
# 输入：path：测试路径；days：回溯天数。
# 输出：None：只修改测试文件时间。
def _age(path: Path, *, days: int) -> None:
    timestamp = (datetime.now(UTC) - timedelta(days=days)).timestamp()
    os.utime(path, (timestamp, timestamp))


# 功能：验证清理始终保留每个会话最近执行和用户数据库。
# 输入：tmp_path：隔离测试根。
# 输出：None：通过断言验证结果。
def test_cleanup_preserves_recent_executions_and_user_state(tmp_path: Path) -> None:
    root = tmp_path / "core"
    missions = root / "missions" / "thread-1"
    missions.mkdir(parents=True)
    (root / "autonomy.sqlite3").write_bytes(b"user-state")
    for index in range(7):
        execution = missions / f"execution-{index}"
        execution.mkdir()
        (execution / "evidence.bin").write_bytes(b"x" * (index + 1))
        _age(execution, days=40 + index)

    report = StorageMaintenanceService(root).clean()

    assert report["removed_count"] == 2
    assert len(list(missions.iterdir())) == 5
    assert (root / "autonomy.sqlite3").read_bytes() == b"user-state"


# 功能：验证过期暂存会删除而新鲜暂存继续保留。
# 输入：tmp_path：隔离测试根。
# 输出：None：通过断言验证结果。
def test_cleanup_removes_expired_staging_but_keeps_fresh_entries(tmp_path: Path) -> None:
    root = tmp_path / "core"
    staging = root / "plugin-staging"
    staging.mkdir(parents=True)
    expired = staging / "expired"
    expired.mkdir()
    (expired / "bundle.bin").write_bytes(b"replaceable")
    _age(expired, days=2)
    fresh = staging / "fresh"
    fresh.mkdir()
    (fresh / "bundle.bin").write_bytes(b"active")

    service = StorageMaintenanceService(root)
    preview = service.inspect()
    result = service.clean()

    assert preview["reclaimable_bytes"] == len(b"replaceable")
    assert result["removed_bytes"] == len(b"replaceable")
    assert result["candidate_count"] == 0
    assert result["reclaimable_bytes"] == 0
    assert not expired.exists()
    assert fresh.exists()


# 功能：验证链接候选不能把清理范围扩展到应用根目录之外。
# 输入：tmp_path：隔离测试根。
# 输出：None：通过断言验证外部文件仍存在。
def test_cleanup_rejects_linked_candidates(tmp_path: Path) -> None:
    root = tmp_path / "core"
    staging = root / "remote-asset-staging"
    staging.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    link = staging / "expired-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        return
    _age(link, days=2)

    result = StorageMaintenanceService(root).clean()

    assert result["removed_count"] == 0
    assert result["issues"]
    assert (outside / "keep.txt").exists()
