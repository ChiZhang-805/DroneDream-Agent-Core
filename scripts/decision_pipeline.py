"""Reproducible pre-GPU checks and allowlisted training bundle, not a flight release."""

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from importlib.metadata import version
from importlib.util import resolve_name
from pathlib import Path

from dronedream_agent_core.decision_dataset import audit_records, file_digest, read_records
from dronedream_agent_core.decision_state_adapter import decision_digest
from dronedream_agent_core.laya_uav_training import verify_package

CODE = (
    "src/dronedream_agent_core/decision_shadow.py",
    "src/dronedream_agent_core/decision_state_adapter.py",
    "src/dronedream_agent_core/decision_dataset.py",
    "src/dronedream_agent_core/decision_label_evidence.py",
    "src/dronedream_agent_core/laya_uav_training.py",
    "src/dronedream_agent_core/local_stage_decision_port.py",
    "src/dronedream_agent_core/decision_runtime_adapter.py",
    "src/dronedream_agent_core/decision_native_capture.py",
    "src/dronedream_agent_core/uav_decision_student.py",
    "scripts/train_laya_uav.py",
    "scripts/prepare_decision_contract_smoke.py",
    "scripts/collect_decision_replay.py",
    "scripts/decision_pipeline.py",
    "scripts/verify_laya_roundtrip.py",
    "scripts/benchmark_laya_cpu.py",
    "scripts/train_decision_student.py",
    "scripts/assemble_decision_corpus.py",
    "scripts/merge_decision_corpora.py",
    "scripts/audit_decision_candidates.py",
    "scripts/benchmark_decision_student.py",
    "tests/test_decision_v2.py",
    "tests/test_stage_decision_port.py",
    "tests/test_decision_runtime_adapter.py",
    "tests/test_decision_label_evidence.py",
    "tests/test_decision_student.py",
    "tests/test_decision_assembly.py",
    "tests/test_decision_input_evidence.py",
    "tests/test_decision_collection_teacher.py",
    "tests/test_decision_dynamic_context.py",
    "tests/test_decision_native_capture.py",
    "tests/test_decision_option_order.py",
    "tests/test_decision_distillation.py",
    "tests/decision_goal_fixture.py",
    "tests/test_decision_stage_control.py",
    "tests/test_decision_stage_label.py",
    "tests/test_decision_stage_wait.py",
    "tests/test_decision_pipeline.py",
    "tests/test_decision_layout_lineage.py",
    "tests/test_decision_replan_evidence.py",
    "tests/test_decision_observation_recovery.py",
    "tests/test_decision_stage_recovery.py",
    "tests/test_decision_route_teacher.py",
    "tests/test_decision_scene_topology.py",
    "scripts/prepare_decision_native_scene.py",
    "scripts/prepare_decision_topology_suite.py",
    "docs/LAYA_UAV_COMPLETE_IMPLEMENTATION_PLAN.md",
    "training/laya/bootstrap.sh",
    "training/laya/requirements.txt",
    "training/laya/run.sh",
    "training/laya/README.md",
)


# 功能：只递归收集训练/验证所需的本地 Python 模块，不扫描或上传用户数据库和密钥。
# 输入：仓库根和明确入口；输出：源文件清单；包初始化另用无副作用的工具包入口。
def dependency_files(root: Path) -> list[str]:
    seen, pending = set(), list(CODE)
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        seen.add(relative)
        path = root / relative
        if path.suffix != ".py":
            continue
        package = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        package = package.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text("utf-8-sig"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                base = resolve_name(name, package) if node.level else name
                names = [base, *(base + "." + a.name for a in node.names)]
            for name in names:
                if not name.startswith("dronedream_"):
                    continue
                candidate = "src/" + name.replace(".", "/") + ".py"
                if (root / candidate).is_file():
                    pending.append(candidate)
    return sorted(seen)


# 功能：从已经冻结的测试文件清单生成隔离回归入口，避免打包了新测试却仍只跑旧名单。
# 输入：已审核的相对文件列表；输出：按名称排序的pytest入口，不包含夹具或主仓库conftest。
def bundled_tests(files):
    return sorted(name for name in files if name.startswith("tests/test_") and name.endswith(".py"))


# 功能：审计所有数据和白名单源文件；未通过时生成明确失败报告，不诱导用户租卡。
# 输入：新输出目录、正式语料/可选 smoke 模式、已测训练探针；输出：可验证准备包。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--training-probe", type=Path, required=True)
    parser.add_argument("--roundtrip", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = read_records(args.corpus)
    audit = audit_records(rows, args.corpus.parent, smoke=args.smoke)
    probe = json.loads(args.training_probe.read_text(encoding="utf-8"))
    roundtrip = json.loads(args.roundtrip.read_text(encoding="utf-8"))
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    training_root = args.training_report.parent
    identity = json.loads((training_root / "training-identity.json").read_text("utf-8"))
    parity = json.loads((training_root / "export-parity.json").read_text("utf-8"))
    package_receipt = verify_package(training_root / "model")
    if (
        probe.get("actual_laya_backward") is not True
        or probe.get("optimizer_steps", 0) < 1
        or probe.get("encoder_trainable") is not True
    ):
        raise ValueError("DECISION_REAL_BACKWARD_PROBE_REQUIRED")
    args.output.mkdir(parents=True, exist_ok=False)
    # 独立包不携带主仓库自动 fixture；仅运行本决策模块测试，不读取用户会话/密钥。
    files = dependency_files(root)
    manifest = {name: file_digest(root / name) for name in files}
    target = args.output / "laya-uav-preparation.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        packages = {str(Path(f).parent).replace("\\", "/") for f in files if f.startswith("src/")}
        for package in packages:
            while package != "src":
                name = package + "/__init__.py"
                if name not in manifest:
                    payload = b'"""Isolated decision training toolkit namespace."""\n'
                    archive.writestr(name, payload)
                    manifest[name] = hashlib.sha256(payload).hexdigest()
                package = package.rsplit("/", 1)[0]
        for relative in files:
            payload = (root / relative).read_bytes()
            if relative.endswith(".sh"):
                payload = payload.replace(b"\r\n", b"\n")
            archive.writestr(relative, payload)
            manifest[relative] = hashlib.sha256(payload).hexdigest()
        archive.writestr("bundle-files.json", json.dumps(manifest, indent=2))
    isolated = args.output / "isolated-kit"
    isolated.mkdir()
    with zipfile.ZipFile(target) as archive:
        archive.extractall(isolated)
    for name, digest in manifest.items():
        if file_digest(isolated / name) != digest:
            raise ValueError("DECISION_BUNDLE_EXTRACTION_MISMATCH")
    packages = {
        name: version(name)
        for name in ("torch", "transformers", "safetensors", "laya", "numpy", "pydantic")
    }
    # 只冻结代码/依赖身份。原始日志、数据库、地图和凭证不自动纳入云端包。
    report = {
        "schema_version": "dronedream.laya-preparation.v1",
        "data_audit": audit,
        "code_sha256": manifest,
        "archive_sha256": file_digest(target),
        "corpus_sha256": file_digest(args.corpus),
        "python": platform.python_version(),
        "environment": packages,
        "probe_sha256": file_digest(args.training_probe),
        "roundtrip_sha256": file_digest(args.roundtrip),
        "complete_training_report_sha256": file_digest(args.training_report),
        "gpu_ready": False,
        "installed_software_changed": False,
        "remaining": list(audit["readiness_issues"]),
        "deployment_warnings": [],
    }
    if (
        training.get("run_sha256") != identity.get("run_sha256")
        or identity.get("run_sha256")
        != decision_digest({key: value for key, value in identity.items() if key != "run_sha256"})
        or package_receipt.get("run_sha256") != training.get("run_sha256")
        or training.get("optimizer_steps", 0) < 2
        or not parity.get("passed")
        or identity.get("config", {}).get("epochs", 0) < 2
    ):
        report["remaining"].append("COMPLETE_TRAINING_SMOKE_NOT_VERIFIED")
    expected_names = {p.name for p in (root / "src/dronedream_agent_core").glob("decision_*.py")}
    expected_names.update(("train_laya_uav.py", "laya_uav_training.py"))
    if set(identity.get("code_sha256", {})) != expected_names:
        report["remaining"].append("TRAINING_SMOKE_SOURCE_MANIFEST_INCOMPLETE")
    for name, expected in identity.get("code_sha256", {}).items():
        if name not in expected_names:
            continue
        source = (
            root / "scripts" / name
            if name == "train_laya_uav.py"
            else (root / "src/dronedream_agent_core" / name)
        )
        if file_digest(source) != expected:
            report["remaining"].append("TRAINING_SMOKE_CODE_CHANGED:" + name)
    environment = dict(
        os.environ, PYTHONPATH=os.pathsep.join((str(isolated / "src"), str(isolated)))
    )
    report["regression_files"] = bundled_tests(files)
    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            *report["regression_files"],
            "--junitxml=" + str(args.output.resolve() / "regression.xml"),
        ],
        cwd=isolated,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    with (args.output / "regression.log").open("x", encoding="utf-8") as stream:
        stream.write(test.stdout + test.stderr)
    report["regression_exit_code"] = test.returncode
    if test.returncode:
        report["remaining"].append("DECISION_REGRESSION_FAILED")
    # 数据/单元测试之外还需要采集器、端到端导出回载和同分布对照的真实证据。
    if roundtrip.get("parity", {}).get("passed") is not True:
        report["remaining"].append("EXPORT_RELOAD_EVIDENCE_FAILED")
    if not roundtrip.get("channel_results") or not all(
        r.get("advice") for r in roundtrip["channel_results"]
    ):
        # CPU 推理不达标是部署缺口，不与微调/学生训练是否可以开始混为一谈。
        report["deployment_warnings"].append("LOCAL_DECISION_LATENCY_NOT_READY")
    if not audit["formal_windows"]:
        report["remaining"].append("PHYSICAL_DECISION_LABEL_COVERAGE_REQUIRED")
    if args.smoke:
        report["remaining"].append("SYNTHETIC_CORPUS_NOT_FORMAL_TRAINING")
    report["gpu_ready"] = audit["gpu_data_ready"] and not report["remaining"]
    with (args.output / "readiness.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "gpu_ready": report["gpu_ready"],
                "remaining": report["remaining"],
                "formal_windows": audit["formal_windows"],
            }
        )
    )
    return 0 if report["gpu_ready"] or args.smoke and test.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
