from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from dronedream_agent_core.source_identity import source_repository_identity

RUNTIME_SOURCES = {
    "px4_offboard_track_executor": (
        "runtime/px4_offboard_track_executor.py",
        "px4_offboard_track_executor.py",
    ),
    "px4_checkpoint_executor": (
        "scripts/px4_checkpoint_executor.py",
        "px4_checkpoint_executor.py",
    ),
    "runtime_depth_safety_worker": (
        "scripts/runtime_depth_safety_worker.py",
        "runtime_depth_safety_worker.py",
    ),
    "runtime_requirements": (
        "runtime/requirements-linux.txt",
        "requirements-linux.txt",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    return completed.stdout.strip()


def build_runtime_provenance(repository: Path, runtime_root: Path) -> dict[str, Any]:
    repository = repository.resolve()
    runtime_root = runtime_root.resolve()
    sources: dict[str, dict[str, object]] = {}
    for source_id, (repository_relative, staged_relative) in RUNTIME_SOURCES.items():
        source = repository / repository_relative
        staged = runtime_root / staged_relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if not staged.is_file():
            raise FileNotFoundError(staged)
        source_sha256 = _sha256(source)
        staged_sha256 = _sha256(staged)
        if source_sha256 != staged_sha256:
            raise ValueError(
                f"staged Runtime source does not match current Agent Core source: {source_id}"
            )
        sources[source_id] = {
            "source_path": repository_relative,
            "staged_path": staged_relative,
            "sha256": source_sha256,
            "bytes": source.stat().st_size,
        }

    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "schema_version": "dronedream.runtime-source-provenance.v1",
        "source_repository": source_repository_identity(repository),
        "source_commit": _git(repository, "rev-parse", "--verify", "HEAD"),
        "source_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "working_tree_dirty": bool(status),
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    provenance = build_runtime_provenance(args.repository, runtime_root)
    (runtime_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
