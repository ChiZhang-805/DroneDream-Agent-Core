"""Content-bound provenance for development ROS workspaces."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from .source_identity import source_repository_identity

ROS_WORKSPACE_PROVENANCE_SCHEMA = "dronedream.ros-workspace-provenance.v1"
ROS_WORKSPACE_PROVENANCE_FILENAME = "ros-workspace-provenance.json"


def _file_snapshot(path: Path) -> tuple[int, str]:
    """Bind size and hash to the same streamed bytes, not a stat/read race."""
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[str, object]:
    """Hash sorted relative names and contents; symbolic links cannot borrow another build."""
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    rows: list[dict[str, object]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"ROS provenance tree contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(resolved).as_posix()
        size, digest = _file_snapshot(path)
        rows.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": digest,
            }
        )
    if not rows:
        raise ValueError(f"ROS provenance tree is empty: {resolved}")
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "file_count": len(rows),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _git(repository: Path, *arguments: str) -> str:
    """Read Git provenance with literal arguments and a bounded process wait."""
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    return completed.stdout.strip()


def build_ros_workspace_provenance(
    repository: Path,
    workspace: Path,
) -> dict[str, Any]:
    """Bind one merged install tree to the repository's exact current ROS sources."""

    repository = repository.resolve()
    workspace = workspace.resolve()
    source_root = repository / "ros_ws" / "src"
    install_root = workspace / "install"
    setup = install_root / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "schema_version": ROS_WORKSPACE_PROVENANCE_SCHEMA,
        "source_repository": source_repository_identity(repository),
        "source_commit": _git(repository, "rev-parse", "--verify", "HEAD"),
        "source_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "working_tree_dirty": bool(status),
        "source_root": "ros_ws/src",
        "source_snapshot": _tree_snapshot(source_root),
        "install_root": "install",
        "install_snapshot": _tree_snapshot(install_root),
    }


def write_ros_workspace_provenance(
    repository: Path,
    workspace: Path,
) -> Path:
    """Publish a complete replacement receipt atomically and remove failed staging files."""
    workspace = workspace.resolve()
    payload = build_ros_workspace_provenance(repository, workspace)
    destination = workspace / ROS_WORKSPACE_PROVENANCE_FILENAME
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def verify_ros_workspace_provenance(
    repository: Path,
    workspace: Path,
) -> dict[str, Any]:
    """Fail closed if sources or any installed ROS artifact differ from the receipt."""

    repository = repository.resolve()
    workspace = workspace.resolve()
    receipt_path = workspace / ROS_WORKSPACE_PROVENANCE_FILENAME
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"ROS_WORKSPACE_PROVENANCE_MISSING: {receipt_path}"
        )
    try:
        with receipt_path.open("rb") as stream:
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError("receipt exceeds provenance bound")
        payload = json.loads(content)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("ROS_WORKSPACE_PROVENANCE_INVALID") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        ROS_WORKSPACE_PROVENANCE_SCHEMA
    ):
        raise ValueError("ROS_WORKSPACE_PROVENANCE_INVALID")
    expected_source = _tree_snapshot(repository / "ros_ws" / "src")
    # Git commit/dirty metadata describes the build; exact source/install bytes
    # decide reuse, so unrelated repository edits need not invalidate ROS artifacts.
    if payload.get("source_snapshot") != expected_source:
        raise ValueError("ROS_WORKSPACE_SOURCE_MISMATCH")
    expected_install = _tree_snapshot(workspace / "install")
    if payload.get("install_snapshot") != expected_install:
        raise ValueError("ROS_WORKSPACE_INSTALL_MISMATCH")
    return payload


__all__ = [
    "ROS_WORKSPACE_PROVENANCE_FILENAME",
    "ROS_WORKSPACE_PROVENANCE_SCHEMA",
    "build_ros_workspace_provenance",
    "verify_ros_workspace_provenance",
    "write_ros_workspace_provenance",
]
