"""Explicit, bounded candidate experiments; never a model qualification receipt."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import StrictModel
from .local_policy_packages import LoadedLocalPolicyPackage, LocalPolicySelection
from .plugin_files import read_plugin_file

TRIAL_EXPERT_ROLES = {
    "local-navigation-policy",
    "precision-maneuver-policy",
    "recovery-policy",
    "risk-critic",
    "perception-encoder",
    "perception-health-critic",
    "settle-stability-critic",
    "payload-dynamics-adapter",
    "state-anomaly-detector",
    "cross-modal-consistency-critic",
}


class SimulationTrialPermit(StrictModel):
    schema_version: Literal["dronedream.simulation-trial.v1"] = "dronedream.simulation-trial.v1"
    permit_id: str = Field(pattern=r"^simulation-trial-[0-9a-f]{32}$")
    domain: Literal["gazebo-px4-sitl"] = "gazebo-px4-sitl"
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vehicle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sensor_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at_unix_ms: int = Field(ge=1, strict=True)
    expires_at_unix_ms: int = Field(ge=1, strict=True)
    maximum_speed_mps: float = Field(gt=0, le=0.6, allow_inf_nan=False)
    hardware_allowed: Literal[False] = False
    qualification_granted: Literal[False] = False
    known_limitations: list[str] = Field(min_length=1, max_length=32)

    # 功能：
    #   限制候选试验的有效期，避免临时测试授权被永久使用。
    # 输入：
    #   self：带签发和到期时刻的仿真试验许可。
    # 输出：
    #   self：有效期不超过七天的许可。
    @model_validator(mode="after")
    def bounded_duration(self):
        if not 0 < self.expires_at_unix_ms - self.issued_at_unix_ms <= 7 * 86400 * 1000:
            raise ValueError("SIMULATION_TRIAL_DURATION_INVALID")
        if any(not value.strip() or len(value) > 1000 for value in self.known_limitations):
            raise ValueError("SIMULATION_TRIAL_LIMITATIONS_INVALID")
        return self


# 功能：
#   有界读取显式候选试验许可，拒绝未生效、过期和被修改为正式资格的文件。
# 输入：
#   path：本机人工测试版明确指定的许可文件。
# 输出：
#   permit：当前有效的仿真试验许可，不代表离线或飞行验收通过。
def read_simulation_trial(path: Path) -> SimulationTrialPermit:
    permit = SimulationTrialPermit.model_validate_json(read_plugin_file(path, limit=64 * 1024))
    now = int(time.time() * 1000)
    if not permit.issued_at_unix_ms <= now < permit.expires_at_unix_ms:
        raise ValueError("SIMULATION_TRIAL_EXPIRED_OR_NOT_YET_VALID")
    return permit


# 功能：
#   绑定一个当前连续控制候选与本次仿真资产，保留模型安全层，不签发正式准入。
# 输入：
#   permit：显式试验许可；packages：当前实际加载的模型包。
#   map_sha256、vehicle_sha256、sensor_sha256：运行时实际资产与传感器摘要。
# 输出：
#   selection：仅用于候选仿真的模型选择记录。
def select_simulation_trial(permit, packages, *, map_sha256, vehicle_sha256, sensor_sha256):
    from .control_feature_contract import CURRENT_POLICY_FEATURE_CONTRACT_SHA256

    permit = SimulationTrialPermit.model_validate(permit.model_dump())
    if not permit.issued_at_unix_ms <= int(time.time() * 1000) < permit.expires_at_unix_ms:
        raise ValueError("SIMULATION_TRIAL_EXPIRED_OR_NOT_YET_VALID")
    if len(packages) != 1 or not isinstance(packages[0], LoadedLocalPolicyPackage):
        raise ValueError("SIMULATION_TRIAL_REQUIRES_ONE_PACKAGE")
    package = packages[0]
    if (
        permit.package_sha256 != package.package_sha256
        or permit.map_semantic_sha256 != map_sha256
        or permit.vehicle_sha256 != vehicle_sha256
        or permit.sensor_contract_sha256 != sensor_sha256
        or package.manifest.vehicle_sha256 != vehicle_sha256
        or package.manifest.sensor_contract_sha256 != sensor_sha256
    ):
        raise ValueError("SIMULATION_TRIAL_BINDING_MISMATCH")
    if package.manifest.pilot_control_mode != "normalized-body-velocity":
        raise ValueError("SIMULATION_TRIAL_CONTINUOUS_CONTROL_REQUIRED")
    if (
        set(package.artifact_paths) != TRIAL_EXPERT_ROLES
        or package.manifest.control_feature_contract_sha256
        != CURRENT_POLICY_FEATURE_CONTRACT_SHA256
    ):
        raise ValueError("SIMULATION_TRIAL_CURRENT_ENSEMBLE_REQUIRED")
    selection = LocalPolicySelection(
        package_id=package.manifest.package_id,
        package_sha256=package.package_sha256,
        scope=package.manifest.scope,
        qualification_receipt_id=permit.permit_id,
        selection_reason="explicit-candidate-simulation-trial",
        simulation_only=True,
    )
    return selection


# 功能：
#   1. 在启动仿真前检查候选、地图、机型和传感器的内容绑定。
#   2. 禁止混用正式资格、教师控制和云端低层接管，试验不改变任何资格分数。
# 输入：
#   path、packages、semantic、vehicle：许可路径、模型目录、地图语义文件和机型。
#   qualifications、admissions、fallback、incompatible：不允许混用的运行模式。
# 输出：
#   permit：经过内容核验的限时、限速仿真许可。
def validate_trial_configuration(
    path,
    packages,
    semantic,
    vehicle,
    *,
    qualifications=(),
    admissions=(),
    fallback=None,
    incompatible=False,
):
    from .hashing import sha256_json
    from .local_policy_packages import load_local_policy_package
    from .plugin_files import hash_plugin_file
    from .runtime_sensor_contracts import oakd_lite_depth_sensor_contract

    if qualifications or admissions or fallback is not None or incompatible or vehicle is None:
        raise ValueError("SIMULATION_TRIAL_MODE_CONFLICT")
    permit = read_simulation_trial(path)
    select_simulation_trial(
        permit,
        [load_local_policy_package(package) for package in packages],
        map_sha256=hash_plugin_file(semantic, limit=64 * 1024**2),
        vehicle_sha256=sha256_json(vehicle),
        sensor_sha256=sha256_json(oakd_lite_depth_sensor_contract()),
    )
    return permit
