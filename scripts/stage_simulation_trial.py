"""Stage an explicitly unqualified, local-only Gazebo candidate experiment."""

import argparse
import hashlib
import time
from pathlib import Path
from uuid import uuid4

from dronedream_agent_core.contracts import VehicleAsset
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_policy_port import OnnxLocalPolicyBackend
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    read_plugin_file,
)
from dronedream_agent_core.runtime_sensor_contracts import oakd_lite_depth_sensor_contract
from dronedream_agent_core.simulation_trial import SimulationTrialPermit, select_simulation_trial
from dronedream_agent_core.training.evidence_publication import (
    publish_evidence_bytes,
    write_evidence_object,
)


# 功能：
#   1. 将完整十专家候选按内容摘要复制到全新的本地试飞资源目录。
#   2. 实际计算 ONNX 接口，签发七天限速仿真许可，保留未通过的质量限制。
# 输入：
#   args：候选包、地图语义、机型、新输出目录及已知限制。
# 输出：
#   report：资源位置、候选身份和实际接口探针结果，不代表飞行验收通过。
def stage(args):
    check_plain_plugin_path(args.output)
    if args.output.exists() or args.output.resolve().is_relative_to(args.package.resolve()):
        raise ValueError("TRIAL_OUTPUT_MUST_BE_NEW_AND_OUTSIDE_PACKAGE")
    package = load_local_policy_package(args.package)
    vehicle = VehicleAsset.model_validate_json(read_plugin_file(args.vehicle, limit=1024**2))
    now = int(time.time() * 1000)
    permit = SimulationTrialPermit(
        permit_id=f"simulation-trial-{uuid4().hex}",
        package_sha256=package.package_sha256,
        map_semantic_sha256=hash_plugin_file(args.semantic, limit=64 * 1024**2),
        vehicle_sha256=sha256_json(vehicle),
        sensor_contract_sha256=sha256_json(oakd_lite_depth_sensor_contract()),
        issued_at_unix_ms=now,
        expires_at_unix_ms=now + 7 * 86400 * 1000,
        maximum_speed_mps=0.6,
        known_limitations=args.limitation,
    )
    select_simulation_trial(
        permit,
        [package],
        map_sha256=permit.map_semantic_sha256,
        vehicle_sha256=permit.vehicle_sha256,
        sensor_sha256=permit.sensor_contract_sha256,
    )
    backend = OnnxLocalPolicyBackend(package, execution_providers=["CPUExecutionProvider"])
    try:
        io_report = backend.verify_runtime_io()
    finally:
        backend.close()
    target = args.output / "packages" / "candidate"
    target.mkdir(parents=True)
    for artifact in package.manifest.artifacts:
        destination = target / artifact.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hash_plugin_file(
            package.artifact_paths[artifact.role], limit=1024**3, destination=destination
        )
        if digest != artifact.sha256:
            raise ValueError("TRIAL_COPY_HASH_MISMATCH")
    publish_evidence_bytes(
        target / "manifest.json", read_plugin_file(package.manifest_path, limit=2 * 1024**2),
        limit=2 * 1024**2,
    )
    if load_local_policy_package(target).package_sha256 != package.package_sha256:
        raise ValueError("TRIAL_PACKAGE_IDENTITY_CHANGED")
    permit_bytes = permit.model_dump_json(indent=2).encode()
    publish_evidence_bytes(args.output / "trial.json", permit_bytes, limit=64 * 1024)
    report = {
        "candidate_only": True,
        "hardware_allowed": False,
        "package_sha256": package.package_sha256,
        "io_probe": io_report,
    }
    write_evidence_object(args.output / "io-probe.json", report)
    write_evidence_object(
        args.output / "catalog.json",
        {
            "schema_version": "dronedream.local-policy-runtime-catalog.v2",
            "deployment_scope": "simulation-trial",
            "entries": [
                {
                    "package": "packages/candidate",
                    "package_sha256": package.package_sha256,
                    "receipt": {
                        "kind": "simulation-trial",
                        "path": "trial.json",
                        "sha256": hashlib.sha256(permit_bytes).hexdigest(),
                    },
                }
            ],
        },
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("package", "semantic", "vehicle", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--limitation", action="append", required=True)
    print(stage(parser.parse_args()))
