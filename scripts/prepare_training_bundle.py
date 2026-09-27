"""Stage an explicit training-only upload; never copy accounts, secrets or histories."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from dronedream_agent_core.asset_package_storage import publish_asset_file
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.runtime_control_io import publish_runtime_json

SCRIPTS = (
    "verify_training_bundle.py", "verify_vision_dataset.py", "preflight_vision_data.py",
    "probe_training_device.py",
    "inspect_preferred_airspace.py",
    "train_local_vision.py", "evaluate_frozen_vision.py", "build_local_vision_dataset.py",
    "capture_labelled_vision_views.py", "prepare_school_vision_source.py",
    "assemble_vision_dataset.py",
    "capture_vision_suite.py", "prepare_vision_scenario_suite.py",
    "prepare_school_route_views.py", "prepare_vision_identity_holdout.py",
    "inspect_vision_capture_coverage.py",
    "encode_local_policy_visual_features.py", "evaluate_local_policy_offline.py",
    "train_causal_control_role.py", "train_causal_control_ensemble.py",
    "assemble_causal_control_data.py", "validate_causal_control_role.py", "verify_causal_control_data.py",
    "evaluate_frozen_causal_role.py",
    "train_local_advisors.py", "train_native_action_risk.py", "refine_local_policy_ppo.py",
    "build_local_advisor_dataset.py",
    "build_native_action_risk_dataset.py", "build_executed_policy_dataset.py",
    "assemble_complete_control_ensemble.py",
)
WEIGHT_SHA256 = "d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280"
SUPPORT_FILES = (
    "training/cloud/bootstrap.sh", "training/cloud/run_vision.sh", "training/cloud/run_control.sh",
    "training/cloud/requirements.txt", "training/control/baseline.json",
    "training/control/regularization.json", "training/control/smoke.json",
    "training/control/seed-806.json", "training/control/seed-807.json", "docs/PRE_GPU_TRAINING_RUNBOOK.md",
    "docs/UAV_PREFERRED_AIRSPACE.md", "docs/VERTICAL_MOTION_CONTRACT.md", "LICENSE",
)


# 功能：
#   逐源模块比对 wheel 中的实际字节，拒绝把旧构建与新训练脚本混装。
# 输入：
#   repository、wheel：当前 Core 源码根及本次构建的 wheel。
# 输出：
#   module_count：已验证内容完全相同的 Python 模块数量。
def verify_wheel_sources(repository, wheel):
    check_plain_plugin_path(wheel)
    module_count = 0
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 10_000:
            raise ValueError("TRAINING_BUNDLE_WHEEL_STRUCTURE_INVALID")
        expected = set()
        for package in (repository / "src").iterdir():
            if not package.is_dir() or not package.name.startswith("dronedream_"):
                continue
            for path in package.rglob("*.py"):
                relative = path.relative_to(repository / "src").as_posix()
                raw = read_plugin_file(path, limit=2 * 1024**2)
                if relative not in names or archive.getinfo(relative).file_size != len(raw):
                    raise ValueError("TRAINING_BUNDLE_WHEEL_SOURCE_MISMATCH:" + relative)
                if archive.read(relative) != raw:
                    raise ValueError("TRAINING_BUNDLE_WHEEL_SOURCE_MISMATCH:" + relative)
                expected.add(relative)
                module_count += 1
        if {name for name in names if name.endswith(".py")} != expected or not module_count:
            raise ValueError("TRAINING_BUNDLE_WHEEL_MODULE_SET_MISMATCH")
    return module_count


# 功能：
#   只复制预定训练入口、当前 Core wheel 和指定公开权重；不复制工作树或用户资料。
# 输入：
#   repository、wheel、weights、destination：当前源码、已构建 wheel、权重和新暂存目录。
# 输出：
#   receipt：每文件摘要、清单摘要及明确的未训练状态。
def prepare(repository, wheel, weights, destination):
    if wheel.suffix != ".whl" or not wheel.name.startswith("dronedream_flight_agent_core-"):
        raise ValueError("TRAINING_BUNDLE_WHEEL_INVALID")
    if hash_plugin_file(weights, limit=256 * 1024**2) != WEIGHT_SHA256:
        raise ValueError("TRAINING_BUNDLE_PRETRAINED_HASH_MISMATCH")
    module_count = verify_wheel_sources(repository, wheel)
    sources = {"scripts/" + name: repository / "scripts" / name for name in SCRIPTS}
    for relative in SUPPORT_FILES:
        sources[relative] = repository / relative
    sources[wheel.name] = wheel
    sources["weights/lraspp_mobilenet_v3_large-d234d4ea.pth"] = weights
    hashes = {relative: hash_plugin_file(path, limit=256 * 1024**2)
              for relative, path in sources.items()}
    destination = destination.absolute()
    check_plain_plugin_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    for relative, source in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_asset_file(source, target, expected_sha256=hashes[relative], limit=256 * 1024**2)
        entries.append({"path": relative, "sha256": hashes[relative],
                        "size_bytes": target.stat().st_size})
    payload = {"schema": "dronedream.training-transfer-bundle.v1", "files": entries,
               "weights_redistribution_authorized_by_this_manifest": False,
               "contains_user_accounts_or_sessions": False, "dataset_included": False,
               "gpu_verified": False, "flight_qualification_granted": False,
               "wheel_source_modules_verified": module_count}
    manifest = destination / "bundle-manifest.json"
    publish_runtime_json(manifest, payload, replace_existing=False)
    receipt = {"bundle": str(destination), "file_count": len(entries),
               "bytes": sum(entry["size_bytes"] for entry in entries),
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    return receipt


# 功能：
#   将当前准备版本发布到新目录，输出用于 SSH 上传后复核的清单摘要。
# 输入：
#   命令行参数：本地 wheel、权重及暂存目录。
# 输出：
#   exit_code：所有制品完整发布后返回零，不上传或租卡。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wheel", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(Path(__file__).parents[1], args.wheel, args.weights, args.output)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
