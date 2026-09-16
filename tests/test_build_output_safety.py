"""Validate build ownership in disposable fixtures, without invoking build tools."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows build paths")


def _run(root: Path, target: Path, operation: str):
    """Load helpers or extract only the old reset function; never run the build script."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    helper = scripts / "build-output-safety.ps1"
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell
    setup = "$ErrorActionPreference='Stop'; $repoRoot=$env:DD_TEST_ROOT; "
    if helper.exists():
        setup += ". $env:DD_TEST_HELPER; "
    else:
        setup += (
            "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
            "$env:DD_TEST_BUILD,[ref]$null,[ref]$null); "
            "$func=$ast.Find({param($n) $n -is "
            "[System.Management.Automation.Language.FunctionDefinitionAst] "
            "-and $n.Name -eq 'Reset-GeneratedDirectory'},$true); "
            ". ([scriptblock]::Create($func.Extent.Text)); "
        )
    expressions = {
        "reset": "Reset-GeneratedDirectory $env:DD_TEST_TARGET",
        "official": (
            "Resolve-OfficialBuildOutput -RepositoryRoot $repoRoot "
            "-OutputRoot $env:DD_TEST_TARGET"
        ),
        "tree": "Assert-PlainBuildTree -RepositoryRoot $repoRoot -Path $env:DD_TEST_TARGET",
        "publish": (
            "Publish-GeneratedBuildFile -RepositoryRoot $repoRoot "
            "-Source (Join-Path $repoRoot 'new.json') -Destination $env:DD_TEST_TARGET"
        ),
        "workspace": "(New-OfficialBuildWorkspace -RepositoryRoot $repoRoot).Path",
    }
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", setup + expressions[operation]],
        env=dict(
            os.environ,
            DD_TEST_ROOT=str(root),
            DD_TEST_TARGET=str(target),
            DD_TEST_HELPER=str(helper),
            DD_TEST_BUILD=str(scripts / "build-autonomy-windows.ps1"),
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )


@pytest.mark.parametrize("relative", ["src", "artifacts/test-runs", "runtime"])
def test_reset_never_accepts_source_or_test_evidence(tmp_path: Path, relative: str) -> None:
    target = tmp_path / relative
    target.mkdir(parents=True)
    sentinel = target / "preserved.txt"
    sentinel.write_text("keep", encoding="utf-8")
    result = _run(tmp_path, target, "reset")
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_only_named_generated_runtime_can_be_reset(tmp_path: Path) -> None:
    target = tmp_path / "app/desktop/src-tauri/resources/runtime"
    target.mkdir(parents=True)
    (target / "old-generated.py").write_text("old", encoding="utf-8")
    result = _run(tmp_path, target, "reset")
    assert result.returncode == 0, result.stderr
    assert list(target.iterdir()) == []


def test_nested_junction_is_rejected_before_any_removal(tmp_path: Path) -> None:
    target = tmp_path / "app/desktop/src-tauri/resources/runtime"
    target.mkdir(parents=True)
    sentinel = target / "generated.py"
    sentinel.write_text("keep", encoding="utf-8")
    outside = tmp_path / "unrelated"
    outside.mkdir()
    link = target / "linked"
    created = subprocess.run(
        ["cmd", "/C", "mklink", "/J", str(link), str(outside)], capture_output=True, timeout=10
    )
    assert created.returncode == 0
    try:
        assert _run(tmp_path, target, "reset").returncode != 0
        assert sentinel.read_text(encoding="utf-8") == "keep"
    finally:
        if link.exists():
            os.rmdir(link)  # Only our fixture junction, never its target tree.


@pytest.mark.parametrize(
    "relative,allowed",
    [
        ("artifacts/official-plugins", True),
        ("app/desktop/src-tauri/resources/official-plugins", True),
        ("artifacts/test-runs", False),
        ("app/desktop/src-tauri/resources/runtime", False),
    ],
)
def test_official_output_is_not_an_arbitrary_generated_folder(
    tmp_path: Path, relative: str, allowed: bool
) -> None:
    target = tmp_path / relative
    assert (_run(tmp_path, target, "official").returncode == 0) is allowed
    assert not target.exists()


def test_publish_replaces_only_the_named_file(tmp_path: Path) -> None:
    target = tmp_path / "index.json"
    target.write_text("old", encoding="utf-8")
    (tmp_path / "new.json").write_text("new", encoding="utf-8")
    sentinel = tmp_path / "other.json"
    sentinel.write_text("keep", encoding="utf-8")
    result = _run(tmp_path, target, "publish")
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == "new"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_publish_rejects_directory_before_consuming_source(tmp_path: Path) -> None:
    target = tmp_path / "index.json"
    target.mkdir()
    source = tmp_path / "new.json"
    source.write_text("new", encoding="utf-8")
    assert _run(tmp_path, target, "publish").returncode != 0
    assert source.read_text(encoding="utf-8") == "new"


def test_missing_source_does_not_destroy_published_index(tmp_path: Path) -> None:
    target = tmp_path / "index.json"
    target.write_text("old", encoding="utf-8")
    assert _run(tmp_path, target, "publish").returncode != 0
    assert target.read_text(encoding="utf-8") == "old"


def test_build_work_is_unique_and_outside_packaged_resources(tmp_path: Path) -> None:
    root = tmp_path / "DroneDream-Workspace/Agent-Core/example"
    root.mkdir(parents=True)
    first = _run(root, root, "workspace")
    second = _run(root, root, "workspace")
    assert first.returncode == second.returncode == 0
    first_path, second_path = Path(first.stdout.strip()), Path(second.stdout.strip())
    assert first_path != second_path
    expected = tmp_path / "DroneDream-Workspace/Build/Official-Plugins/example"
    assert first_path.parent == second_path.parent == expected
    assert first_path.is_dir() and second_path.is_dir()


def test_build_scripts_parse_and_use_shared_reset(tmp_path: Path) -> None:
    """Parse actual scripts under both Windows PowerShell and pwsh where available."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for shell in dict.fromkeys(filter(None, (shutil.which("powershell"), shutil.which("pwsh")))):
        _prepare_publication_fixture(tmp_path)
        result = subprocess.run(
            [
                shell, "-NoProfile", "-NonInteractive", "-Command",
                "$ErrorActionPreference='Stop'; "
                "foreach($name in @('build-output-safety.ps1','build-official-plugins.ps1',"
                "'build-autonomy-windows.ps1')) { "
                "$tokens=$null; $parseErrors=$null; "
                "$ast=[System.Management.Automation.Language.Parser]::ParseFile("
                "(Join-Path $env:DD_TEST_SCRIPTS $name),[ref]$tokens,[ref]$parseErrors); "
                "if($parseErrors.Count) { throw ($parseErrors | Out-String) }; "
                "if($name -eq 'build-autonomy-windows.ps1' -and $ast.Find({param($n) "
                "$n -is [System.Management.Automation.Language.FunctionDefinitionAst] "
                "-and $n.Name -eq 'Reset-GeneratedDirectory'},$true)) { throw 'Stale reset' } "
                "}; . (Join-Path $env:DD_TEST_SCRIPTS 'build-output-safety.ps1'); "
                "Publish-GeneratedBuildFile -RepositoryRoot $env:DD_TEST_ROOT "
                "-Source (Join-Path $env:DD_TEST_ROOT 'new.json') "
                "-Destination (Join-Path $env:DD_TEST_ROOT 'index.json')",
            ],
            env=dict(os.environ, DD_TEST_SCRIPTS=str(scripts), DD_TEST_ROOT=str(tmp_path)),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "index.json").read_text(encoding="utf-8") == "new"


def _prepare_publication_fixture(root: Path) -> None:
    """Recreate only the two files each shell is about to atomically publish."""
    (root / "index.json").write_text("old", encoding="utf-8")
    (root / "new.json").write_text("new", encoding="utf-8")
