"""Export deterministic JSON Schemas for every cross-process contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from dronedream_agent_core.asset_packages import (
    AssetImportJob,
    AssetIR,
    DDPkgManifest,
    QualificationBinding,
)
from dronedream_agent_core.asset_pair_qualification import (
    AssetPairQualificationJob,
    AssetPairQualificationPlan,
    AssetPairQualificationReceipt,
    AssetQualificationInputs,
    AssetQualificationPluginCheck,
)
from dronedream_agent_core.contracts import (
    BodyFrameControlIntent,
    CoveragePattern,
    CoveragePlanRequest,
    FlightPlan,
    GraphRoute,
    IntentArtifact,
    MapAsset,
    MapRuntimeBindings,
    MissionAssetPairQualificationBinding,
    MissionContract,
    MissionLifecycleBinding,
    MissionRequest,
    MissionVerificationPlan,
    MissionVerificationRequirement,
    ModelCallRecord,
    NormalizedPilotControl,
    PlannerContribution,
    PlannerValidation,
    PlanRevisionRecord,
    PluginInvocationPlan,
    PreparedMission,
    Px4GazeboRunEvidence,
    Px4Track,
    Px4TrackRequest,
    RouteAlternativeCandidate,
    RouteAlternativeDecision,
    RouteAlternativeSet,
    RouteClearanceReport,
    RouteQuery,
    RuntimeAmendmentDirective,
    RuntimeAuthorizedCommand,
    RuntimeCheckpointDecision,
    RuntimeCheckpointRequest,
    RuntimeCommandAdoption,
    RuntimeControlSession,
    RuntimeHoldAcknowledgement,
    RuntimeInterruptionDecision,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    RuntimeMessageClassification,
    RuntimeOperatorControlCommand,
    RuntimeOperatorTakeoverAdoption,
    RuntimeOperatorTakeoverGrant,
    RuntimeReplacementTrack,
    RuntimeUserMessage,
    SemanticPlan,
    SimulationWorkflowResult,
    TaskGraphArtifact,
    TaskThread,
    ToolReceipt,
    VehicleAsset,
)
from dronedream_agent_core.control_execution_evidence import ControlApplicationRecord
from dronedream_agent_core.local_policy_packages import (
    LocalPolicyQualificationReceipt,
    LocalPolicySimulationAdmissionReceipt,
)
from dronedream_agent_core.plugin_contracts import (
    CapabilityBrokerReceipt,
    PluginLifecycleReceipt,
    PluginManifest,
    PluginSnapshot,
)
from dronedream_agent_core.realtime_feature_encoders import RealtimeFeatureSnapshot

CONTRACTS = (
    ControlApplicationRecord,
    BodyFrameControlIntent,
    NormalizedPilotControl,
    RuntimeLocalSafetyCommand,
    RuntimeLocalSafetyObservation,
    RealtimeFeatureSnapshot,
    LocalPolicyQualificationReceipt,
    LocalPolicySimulationAdmissionReceipt,
    AssetIR,
    DDPkgManifest,
    AssetImportJob,
    QualificationBinding,
    AssetQualificationInputs,
    AssetPairQualificationPlan,
    AssetQualificationPluginCheck,
    AssetPairQualificationReceipt,
    AssetPairQualificationJob,
    MapAsset,
    MapRuntimeBindings,
    VehicleAsset,
    MissionRequest,
    MissionAssetPairQualificationBinding,
    IntentArtifact,
    MissionContract,
    TaskGraphArtifact,
    SemanticPlan,
    PlannerContribution,
    PlannerValidation,
    FlightPlan,
    RouteQuery,
    GraphRoute,
    RouteClearanceReport,
    RouteAlternativeCandidate,
    RouteAlternativeSet,
    RouteAlternativeDecision,
    CoveragePlanRequest,
    CoveragePattern,
    Px4TrackRequest,
    Px4Track,
    RuntimeCheckpointRequest,
    RuntimeCheckpointDecision,
    TaskThread,
    PlanRevisionRecord,
    MissionLifecycleBinding,
    MissionVerificationRequirement,
    MissionVerificationPlan,
    PluginManifest,
    PluginSnapshot,
    PluginLifecycleReceipt,
    CapabilityBrokerReceipt,
    PluginInvocationPlan,
    RuntimeControlSession,
    RuntimeUserMessage,
    RuntimeHoldAcknowledgement,
    RuntimeMessageClassification,
    RuntimeAmendmentDirective,
    RuntimeInterruptionDecision,
    RuntimeAuthorizedCommand,
    RuntimeCommandAdoption,
    RuntimeOperatorTakeoverGrant,
    RuntimeOperatorControlCommand,
    RuntimeOperatorTakeoverAdoption,
    RuntimeReplacementTrack,
    ModelCallRecord,
    ToolReceipt,
    PreparedMission,
    Px4GazeboRunEvidence,
    SimulationWorkflowResult,
)


# 功能：
#   按固定排序、缩进和末尾换行编码有限 JSON，使输出文件与索引摘要使用同一份字节。
# 输入：
#   value：待编码的 Schema 或索引对象。
# 输出：
#   payload：UTF-8 JSON 字节。
def _json_bytes(value: Any) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return payload


# 功能：
#   为显式登记的跨进程契约生成 Schema 及摘要索引，不把任意导入的内部模型公开为契约。
# 输入：
#   无。
# 输出：
#   files：目标文件名到完整输出字节的映射。
def expected_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    index_entries: list[dict[str, str]] = []
    for contract in CONTRACTS:
        filename = f"{contract.__name__}.schema.json"
        payload = _json_bytes(contract.model_json_schema())
        files[filename] = payload
        index_entries.append(
            {
                "contract": contract.__name__,
                "file": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    files["index.json"] = _json_bytes(
        {
            "schema_version": "dronedream.schema-index.v1",
            "contracts": index_entries,
        }
    )
    return files


# 功能：
#   1. 检查模式只比较文件内容和文件集合，不修改磁盘。
#   2. 导出模式先拒绝未分类 JSON，再更新登记文件；不自行删除未知文件。
# 输入：
#   output_dir：Schema 输出目录。
#   check：是否只检查现有导出结果。
# 输出：
#   exit_code：成功为 0，内容陈旧或存在未分类文件为 1。
def export(output_dir: Path, *, check: bool) -> int:
    expected = expected_files()
    mismatches: list[str] = []
    exit_code = 0
    if check:
        for filename, payload in expected.items():
            path = output_dir / filename
            if not path.is_file() or path.read_bytes() != payload:
                mismatches.append(filename)
        if output_dir.is_dir():
            expected_names = set(expected)
            for path in output_dir.glob("*.json"):
                if path.name not in expected_names:
                    mismatches.append(f"unexpected:{path.name}")
        if mismatches:
            print("Schema export is stale: " + ", ".join(sorted(mismatches)))
            exit_code = 1
            return exit_code
        print(f"SCHEMAS_CURRENT count={len(CONTRACTS)} directory={output_dir}")
        return exit_code

    # 未知 JSON 不等于废弃契约，必须在任何写入前退出，留给维护者确认其用途。
    unexpected = sorted(
        path.name for path in output_dir.glob("*.json") if path.name not in expected
    )
    if unexpected:
        print("Schema export refused unclassified files: " + ", ".join(unexpected))
        exit_code = 1
        return exit_code
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in expected.items():
        (output_dir / filename).write_bytes(payload)
    print(f"SCHEMAS_EXPORTED count={len(CONTRACTS)} directory={output_dir}")
    return exit_code


# 功能：
#   解析导出目录及只读检查选项，执行 Schema 工具；该工具不执行 Runtime 构建资格验收。
# 输入：
#   无。
# 输出：
#   exit_code：Schema 检查或导出的退出码。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "schemas",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    exit_code = export(args.output_dir, check=args.check)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
