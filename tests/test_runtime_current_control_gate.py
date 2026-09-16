"""A structurally readable legacy ensemble must not enter a current installer."""

import sys

import pytest
from test_local_policy_runtime_staging import _admission, _package

from scripts import stage_local_policy_runtime as staging


# 功能：
#   验证十角色旧候选排序包即使持有旧仿真准入回执，也不能被打入当前连续控制 Runtime。
# 输入：
#   tmp_path：隔离测试包与发布路径。
#   monkeypatch：设置本测试命令行的工具。
# 输出：
#   None：不返回业务数据。
def test_legacy_ensemble_cannot_be_published_as_current_runtime(tmp_path, monkeypatch):
    package = _package(tmp_path / "legacy")
    receipt = tmp_path / "admission.json"
    _admission(receipt, package)
    output = tmp_path / "runtime"
    monkeypatch.setattr(sys, "argv", [
        "stage", "--package", str(package.root),
        "--simulation-admission", str(receipt), "--output", str(output),
    ])
    with pytest.raises(ValueError, match="LOCAL_POLICY_RUNTIME_CURRENT_CONTROL_REQUIRED"):
        staging.main()
    assert not output.exists()
