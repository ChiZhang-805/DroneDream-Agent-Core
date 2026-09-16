from __future__ import annotations

import json
from pathlib import Path

import pytest

from dronedream_agent_core.ros_workspace_provenance import (
    ROS_WORKSPACE_PROVENANCE_FILENAME,
    verify_ros_workspace_provenance,
    write_ros_workspace_provenance,
)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "ros-workspace"
    install = workspace / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# merged ROS workspace\n", encoding="utf-8")
    (install / "capability_host").write_bytes(b"current-host")
    return workspace


def test_ros_workspace_receipt_binds_sources_and_install_tree(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    workspace = _workspace(tmp_path)

    receipt = write_ros_workspace_provenance(repository, workspace)
    verified = verify_ros_workspace_provenance(repository, workspace)

    assert receipt == workspace / ROS_WORKSPACE_PROVENANCE_FILENAME
    assert verified["source_snapshot"]["file_count"] > 0
    assert verified["install_snapshot"]["file_count"] == 2


def test_ros_workspace_receipt_rejects_replaced_installed_binary(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    workspace = _workspace(tmp_path)
    write_ros_workspace_provenance(repository, workspace)
    (workspace / "install" / "capability_host").write_bytes(b"stale-host")

    with pytest.raises(ValueError, match="ROS_WORKSPACE_INSTALL_MISMATCH"):
        verify_ros_workspace_provenance(repository, workspace)


def test_ros_workspace_receipt_rejects_different_source_snapshot(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    workspace = _workspace(tmp_path)
    receipt_path = write_ros_workspace_provenance(repository, workspace)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_snapshot"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="ROS_WORKSPACE_SOURCE_MISMATCH"):
        verify_ros_workspace_provenance(repository, workspace)


@pytest.mark.parametrize("content", [b" " * (1024 * 1024 + 1), b"[" * 2000 + b"]" * 2000],
                         ids=["oversize-receipt", "excessive-json-depth"])
def test_bad_receipt_is_rejected_before_tree_reads(tmp_path, monkeypatch, content):
    from dronedream_agent_core import ros_workspace_provenance as module

    workspace = _workspace(tmp_path)
    (workspace / ROS_WORKSPACE_PROVENANCE_FILENAME).write_bytes(content)
    monkeypatch.setattr(module, "_tree_snapshot",
                        lambda *_: pytest.fail("invalid receipt reached source tree"))
    with pytest.raises(ValueError, match="PROVENANCE_INVALID"):
        verify_ros_workspace_provenance(tmp_path, workspace)


def test_failed_publication_preserves_receipt_and_removes_staging(tmp_path, monkeypatch):
    from dronedream_agent_core import ros_workspace_provenance as module

    workspace = _workspace(tmp_path)
    receipt = workspace / ROS_WORKSPACE_PROVENANCE_FILENAME
    receipt.write_bytes(b"previous-receipt")
    monkeypatch.setattr(module, "build_ros_workspace_provenance", lambda *a: {"source": "new"})
    def fail_replace(*args):
        raise OSError("publication failed")
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failed"):
        write_ros_workspace_provenance(tmp_path, workspace)
    assert receipt.read_bytes() == b"previous-receipt"
    assert not list(workspace.glob(".*.tmp"))
