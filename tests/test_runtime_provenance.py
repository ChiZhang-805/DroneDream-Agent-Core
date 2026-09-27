from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dronedream_agent_core.source_identity import source_repository_identity
from scripts.write_runtime_provenance import RUNTIME_SOURCES, build_runtime_provenance


def _stage_runtime_sources(repository: Path, runtime_root: Path) -> None:
    runtime_root.mkdir(parents=True)
    for repository_relative, staged_relative in RUNTIME_SOURCES.values():
        shutil.copy2(repository / repository_relative, runtime_root / staged_relative)


# 功能：
#   从当前执行器源码副本生成 Runtime 身份，拒绝依赖用户仓库的跨系统 Git 元数据。
# 输入：
#   tmp_path、isolated_source_repository：隔离运行目录与当前源码副本。
# 输出：
#   None：无返回值。
def test_runtime_provenance_binds_exact_core_sources(tmp_path, isolated_source_repository):
    repository = isolated_source_repository
    runtime_root = tmp_path / "runtime"
    _stage_runtime_sources(repository, runtime_root)

    provenance = build_runtime_provenance(repository, runtime_root)

    assert provenance["source_repository"] == source_repository_identity(repository)
    assert set(provenance["sources"]) == set(RUNTIME_SOURCES)
    assert all(
        not str(source["source_path"]).startswith("scripts/simulators/")
        for source in provenance["sources"].values()
    )


def test_runtime_provenance_rejects_a_staged_old_executor(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    runtime_root = tmp_path / "runtime"
    _stage_runtime_sources(repository, runtime_root)
    (runtime_root / "px4_offboard_track_executor.py").write_text(
        "# stale executor\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match current Agent Core source"):
        build_runtime_provenance(repository, runtime_root)
