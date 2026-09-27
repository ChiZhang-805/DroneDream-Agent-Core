"""Exercise source/byte binding in disposable repositories, not real flight qualification."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/core_build_receipt.py"
spec = importlib.util.spec_from_file_location("core_build_receipt", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# 功能：
#   在一次性目录中创建文件绑定测试夹具，不生成可运行模型或真实 Windows 程序。
# 输入：
#   root：测试仓库根目录。
# 输出：
#   commit：已提交夹具源码的固定提交号。
def make_build(root: Path) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin",
                    "https://github.com/ChiZhang-805/DroneDream-Agent-Core.git"], check=True)
    (root / ".gitignore").write_text("artifacts/\napp/\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture"],
                   check=True)
    commit = module.git_value(root, "rev-parse", "HEAD")
    resources = root / module.RESOURCE_ROOT
    binaries = root / module.BINARY_ROOT
    binaries.mkdir(parents=True)
    for name in module.BINARIES:
        (binaries / name).write_bytes(b"MZ-fixture-not-executable")
    runtime = resources / "runtime"
    runtime.mkdir(parents=True)
    provenance = {"source_commit": commit, "source_tree": module.git_value(root, "rev-parse",
                 "HEAD^{tree}"), "working_tree_dirty": False,
                  "source_repository": "ChiZhang-805/DroneDream-Agent-Core"}
    payloads = {"local-policy/catalog.json": b"{}",
                "camera-clock/camera-clock-runtime.json": b"{}",
                "camera-clock/libdronedream-camera-clock.so": b"unit-test-only",
                "payload-placement/payload-placement-runtime.json": b"{}",
                "payload-placement/libdronedream-payload-placement.so": b"unit-test-only",
                "provenance.json": json.dumps(provenance).encode(),
                "native-sensors/native-sensor-runtime.json": b"{}", "licenses/LICENSE": b"MIT",
                "licenses/default-assets-licenses.json": b"{}", "local-policy/licenses.json": b"{}"}
    entries = []
    for name, content in payloads.items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append({"path": name, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    (runtime / "runtime-manifest.json").write_text(json.dumps({"files": entries}))
    plugins = resources / "official-plugins"
    plugins.mkdir()
    (plugins / "plugin.zip").write_bytes(b"test-plugin")
    (plugins / "index.json").write_text(json.dumps({"plugins": [{"file": "plugin.zip",
        "sha256": hashlib.sha256(b"test-plugin").hexdigest()}]}))
    assets = resources / "default-assets"
    assets.mkdir()
    packages = []
    for kind in ("map", "vehicle"):
        (assets / f"{kind}.ddpkg").write_bytes(kind.encode())
        packages.append({"kind": kind, "file": f"{kind}.ddpkg",
                         "sha256": hashlib.sha256(kind.encode()).hexdigest()})
    qualification_id = "asset-qualification-fixture"
    qualified_pair = {
        "schema_version": "dronedream.bundled-qualified-pair.v1",
        "qualification_id": qualification_id,
        "packages": packages,
    }
    (assets / "index.json").write_text(json.dumps({
        "schema_version": "dronedream.bundled-assets.v3",
        "default_qualification_id": qualification_id,
        "qualified_pair": qualified_pair,
        "qualified_pairs": [qualified_pair],
    }))
    return commit


# 功能：
#   验证新回执能读回，而替换任一二进制后原回执不能继续通过。
# 输入：
#   tmp_path：隔离测试根目录。
# 输出：
#   None：断言失败时测试报错。
def test_build_receipt_binds_actual_binary_bytes(tmp_path: Path) -> None:
    root = tmp_path / "core"
    commit = make_build(root)
    receipt = module.build_receipt(root, commit, verify=False)
    assert module.build_receipt(root, commit, verify=True) == receipt
    binary = root / module.BINARY_ROOT / module.BINARIES[0]
    binary.write_bytes(b"MZ-different-previous-build")
    with pytest.raises(ValueError, match="RECEIPT_MISMATCH"):
        module.build_receipt(root, commit, verify=True)


# 功能：
#   即使替换后的 JSON 含义相同，暂存方冻结的字节摘要不同也必须拒绝。
# 输入：
#   tmp_path：独立构建夹具目录。
# 输出：
#   None：断言同一字节回执通过，改写字节后失败。
def test_build_receipt_verification_binds_callers_frozen_bytes(tmp_path: Path) -> None:
    root = tmp_path / "core"
    commit = make_build(root)
    receipt = module.build_receipt(root, commit, verify=False)
    path = root / module.RECEIPT
    frozen = path.read_bytes()
    digest = hashlib.sha256(frozen).hexdigest()
    assert module.build_receipt(root, commit, verify=True,
                                expected_receipt_sha256=digest) == receipt
    path.write_bytes(frozen + b"\n")
    with pytest.raises(ValueError, match="RECEIPT_BYTES_CHANGED"):
        module.build_receipt(root, commit, verify=True, expected_receipt_sha256=digest)
    assert module.build_receipt(root, commit, verify=True) == receipt


# 功能：
#   错误摘要或生成模式携带核验摘要时，在任何构建路径读取之前拒绝调用。
# 输入：
#   tmp_path：不存在的测试构建目录。
#   verify：是否请求只读核验。
#   digest：被测试的摘要参数。
# 输出：
#   None：断言参数检查先于文件读取执行。
@pytest.mark.parametrize("verify,digest", [(False, "a" * 64), (True, "latest"), (True, "A" * 64)])
def test_frozen_receipt_hash_requires_verify_mode(tmp_path, verify, digest):
    with pytest.raises(ValueError, match="RECEIPT_HASH_ARGUMENT_INVALID"):
        module.build_receipt(tmp_path / "absent", "a" * 40, verify=verify,
                             expected_receipt_sha256=digest)


# 功能：
#   验证源码变更、提交错配和资源损坏都拒绝复用构建结果。
# 输入：
#   tmp_path：隔离测试根目录。
#   problem：本用例注入的变更类型。
# 输出：
#   None：断言失败时测试报错。
@pytest.mark.parametrize("problem", ["dirty", "commit", "runtime"])
def test_build_receipt_rejects_drift(tmp_path: Path, problem: str) -> None:
    root = tmp_path / "core"
    commit = make_build(root)
    module.build_receipt(root, commit, verify=False)
    if problem == "dirty":
        (root / ".gitignore").write_text("app/\nartifacts/\n# changed\n")
    elif problem == "commit":
        commit = "b" * 40
    else:
        (root / module.RESOURCE_ROOT / "runtime/local-policy/catalog.json").write_text("changed")
    with pytest.raises(ValueError):
        module.build_receipt(root, commit, verify=True)


# 功能：
#   验证危险或模糊清单路径在读取文件前被拒绝。
# 输入：
#   tmp_path：隔离测试根目录。
#   relative：待拒绝的相对路径。
# 输出：
#   None：断言失败时测试报错。
@pytest.mark.parametrize("relative", ["../outside", "/absolute", "x/../outside", "x:stream",
                                       "x\\outside", "x./file", "x//file", "NUL.json", "x/COM1"])
def test_build_manifest_rejects_unsafe_paths(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="PATH_INVALID"):
        module.bound_file(tmp_path, relative)


# 功能：
#   验证构建脚本在清理输出前预检模型，且组件回执生成在独立安装包步骤之前。
# 输入：
#   无：只读当前构建脚本。
# 输出：
#   None：断言失败时测试报错。
def test_builder_preflights_before_reset_and_supports_components_only() -> None:
    source = (SCRIPT.parent / "build-autonomy-windows.ps1").read_text(encoding="utf-8")
    assert source.index("--check-only --verify-onnx") < source.index("Reset-GeneratedDirectory")
    assert source.index("--stage-from $PayloadPlacementRuntime --check-only") < source.index("Reset-GeneratedDirectory")
    assert source.index("--source $CameraClockRuntime --check-only") < source.index("Reset-GeneratedDirectory")
    assert source.index("core_build_receipt.py") < source.index("if ($StageOnly)")
    assert source.index("if ($StageOnly)") < source.index('run build\n')
    assert "ReuseOfficialPluginBundle" not in source
    assert source.index("build-official-plugins.ps1") < source.index("core_build_receipt.py")
