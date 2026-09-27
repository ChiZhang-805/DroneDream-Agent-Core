"""Pinned-runtime entrypoint for one prepared asset-pair qualification run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dronedream_plugin_sdk.protocol import decode_json

from .asset_packages import normalized_member_path
from .asset_pair_qualification import (
    AssetPairQualificationError,
    AssetPairQualificationPlan,
    verify_prepared_asset_inputs,
)
from .gazebo_adapter import run_px4_gazebo_track
from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file


class AssetQualificationCliError(ValueError):
    """A prepared qualification workspace failed its immutable boundary checks."""


# 功能：
#   有界计算准备材料的摘要，并检查普通文件身份与读取期间变化。
# 输入：
#   path：需要复核的材料路径。
# 输出：
#   digest：实际读取字节的 SHA-256 十六进制摘要。
def _sha256(path: Path) -> str:
    try:
        digest = hash_plugin_file(path, limit=256 * 1024 * 1024)
    except (OSError, ValueError) as error:
        raise AssetQualificationCliError("ASSET_QUALIFICATION_ARTIFACT_INVALID") from error
    return digest


# 功能：
#   查找准备目录中的普通文件，拒绝歧义路径和链接，不先解析链接后再放行。
# 输入：
#   root：准备目录根路径。
#   relative：计划声明的规范相对路径。
# 输出：
#   candidate：位于准备目录内的实际输入路径。
def _bound_path(root: Path, relative: str) -> Path:
    try:
        normalized_member_path(relative)
        check_plain_plugin_path(root / relative)
        resolved_root = root.resolve()
        candidate = (resolved_root / relative).resolve()
        if resolved_root not in candidate.parents or not candidate.is_file():
            raise ValueError("prepared input is not a regular child file")
    except (OSError, ValueError) as error:
        raise AssetQualificationCliError("ASSET_QUALIFICATION_INPUT_PATH_INVALID") from error
    return candidate


# 功能：
#   定义准备目录、证据目录与执行环境的命令行参数，不在解析阶段启动仿真。
# 输入：
#   无。
# 输出：
#   parser：配置完成的命令行解析器。
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute a hash-bound DroneDream map/vehicle qualification plan"
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--px4-root", type=Path, default=Path("/opt/PX4-Autopilot"))
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--base-executor", type=Path, required=True)
    parser.add_argument("--ros-workspace", type=Path, required=True)
    parser.add_argument(
        "--simulation-ground-truth-control",
        action="store_true",
        help=(
            "Use simulator truth only for bounded map/aircraft geometry qualification; "
            "this does not qualify onboard localization or model control"
        ),
    )
    return parser


# 功能：
#   1. 复核计划、路线、轨迹、净空与原始资产绑定，拒绝执行准备后被替换的材料。
#   2. 调用真实 PX4/Gazebo 适配器；此函数会执行仿真，上层仍需验证全部资格门禁。
# 输入：
#   work_root：已生成并隔离的准备目录。
#   run_dir：本次独立的运行证据目录。
#   px4_root：固定版本 PX4 的安装路径。
#   executor：产品分发的飞控执行器脚本。
#   ros_workspace：固定版本 ROS 工作空间。
# 输出：
#   evidence：适配器产生的运行结果。
def run_prepared_asset_qualification(
    *,
    work_root: Path,
    run_dir: Path,
    px4_root: Path,
    executor: Path,
    base_executor: Path | None = None,
    ros_workspace: Path,
    simulation_ground_truth_control: bool = False,
) -> dict[str, Any]:
    plan_path = _bound_path(work_root, "qualification-plan.json")
    work_root = work_root.resolve()
    try:
        payload = read_plugin_file(plan_path, limit=2 * 1024 * 1024)
        plan = AssetPairQualificationPlan.model_validate(
            decode_json(payload, limit=2 * 1024 * 1024, node_limit=2_000_000)
        )
    except (OSError, ValueError) as error:
        raise AssetQualificationCliError("ASSET_QUALIFICATION_PLAN_INVALID") from error
    try:
        verify_prepared_asset_inputs(plan, work_root)
    except AssetPairQualificationError as error:
        raise AssetQualificationCliError("ASSET_QUALIFICATION_PREPARED_ASSET_INVALID") from error
    route = _bound_path(work_root, "route.json")
    clearance = _bound_path(work_root, "clearance.json")
    track = _bound_path(work_root, "track.json")
    expected = {
        route: plan.route_sha256,
        clearance: plan.clearance_sha256,
        track: plan.track_sha256,
    }
    mismatched = [path.name for path, digest in expected.items() if _sha256(path) != digest]
    if mismatched:
        raise AssetQualificationCliError(
            f"ASSET_QUALIFICATION_PLAN_ARTIFACT_MISMATCH:{','.join(mismatched)}"
        )

    evidence = run_px4_gazebo_track(
        run_dir=run_dir.resolve(),
        world_sdf=_bound_path(work_root, plan.inputs.world_sdf),
        semantic_path=_bound_path(work_root, plan.inputs.semantic),
        vehicle_sdf=_bound_path(work_root, plan.inputs.vehicle_sdf),
        vehicle_metadata_path=_bound_path(work_root, plan.inputs.vehicle_metadata),
        route_path=route,
        track_path=track,
        clearance_path=clearance,
        controller_params_path=_bound_path(work_root, plan.inputs.controller_params),
        px4_root=px4_root.resolve(),
        executor_path=executor.resolve(),
        executor_extra_args=(
            ["--base-executor", str(base_executor.resolve())] if base_executor is not None else None
        ),
        ros_workspace=ros_workspace.resolve(),
        contract_id=plan.qualification_id,
        world_name=plan.inputs.world_name,
        vehicle_name=plan.inputs.vehicle_name,
        px4_sitl_model=plan.inputs.px4_sitl_model,
        simulation_ground_truth_control=simulation_ground_truth_control,
    )
    return evidence


# 功能：
#   执行命令行指定的验收并输出有限 JSON；运行未报告通过时返回非零退出码。
# 输入：
#   argv：显式参数序列，缺省时使用当前进程的命令行参数。
# 输出：
#   exit_code：通过为 0，未通过为 2。
def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = run_prepared_asset_qualification(
        work_root=args.work_root,
        run_dir=args.run_dir,
        px4_root=args.px4_root,
        executor=args.executor,
        base_executor=args.base_executor,
        ros_workspace=args.ros_workspace,
        simulation_ground_truth_control=args.simulation_ground_truth_control,
    )
    print(json.dumps(evidence, ensure_ascii=False, allow_nan=False, sort_keys=True))
    exit_code = 0 if evidence.get("status") == "verified" else 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
