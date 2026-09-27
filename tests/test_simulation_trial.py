import hashlib
import json
import time
from types import SimpleNamespace

import pytest
from test_local_policy_runtime_staging import SENSOR_SHA256, VEHICLE_SHA256, _package

from dronedream_agent_app.runtime_manager import RuntimeBridgeError, RuntimeManager
from dronedream_agent_core.local_policy_packages import select_local_policy_for_simulation
from dronedream_agent_core.simulation_trial import (
    SimulationTrialPermit,
    read_simulation_trial,
    select_simulation_trial,
    validate_trial_configuration,
)


# 功能：
#   构建明确标为仿真候选的测试许可，不提供任何生产准入凭据。
# 输入：
#   package：测试模型包；updates：需要验证的边界字段。
# 输出：
#   permit：候选仿真许可。
def _permit(package, **updates):
    now = int(time.time() * 1000)
    values = dict(
        permit_id="simulation-trial-" + "a" * 32,
        package_sha256=package.package_sha256,
        map_semantic_sha256="c" * 64,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
        issued_at_unix_ms=now - 1000,
        expires_at_unix_ms=now + 3600000,
        maximum_speed_mps=0.6,
        known_limitations=["Not flight qualified"],
    )
    return SimulationTrialPermit(**(values | updates))


# 功能：
#   证明候选可被显式试飞选中，但不能凭试飞许可进入标准仿真准入。
# 输入：
#   tmp_path：隔离测试目录。
# 输出：
#   无：通过断言或抛出测试失败。
def test_trial_is_not_an_admission(tmp_path):
    package = _package(tmp_path / "package", continuous_control=True)
    permit = _permit(package)
    selection = select_simulation_trial(
        permit,
        [package],
        map_sha256="c" * 64,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_sha256=SENSOR_SHA256,
    )
    assert selection.simulation_only
    assert selection.selection_reason == "explicit-candidate-simulation-trial"
    with pytest.raises(ValueError):
        select_local_policy_for_simulation(
            packages=[package],
            admissions=[],
            qualification_receipts=[],
            map_sha256="c" * 64,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )


# 功能：
#   检查过期、错绑和放宽硬限制均被拒绝。
# 输入：
#   tmp_path：隔离目录；updates：待拒绝的许可改动。
# 输出：
#   无：通过拒绝断言或抛出测试失败。
@pytest.mark.parametrize(
    "updates",
    [
        {"hardware_allowed": True},
        {"qualification_granted": True},
        {"maximum_speed_mps": 0.61},
        {"maximum_speed_mps": float("nan")},
        {"package_sha256": "d" * 64},
        {"map_semantic_sha256": "d" * 64},
        {"expires_at_unix_ms": int(time.time() * 1000) - 500},
    ],
)
def test_trial_rejects_unsafe_configuration(tmp_path, updates):
    package = _package(tmp_path / "package", continuous_control=True)
    with pytest.raises(ValueError):
        permit = _permit(package, **updates)
        path = tmp_path / "permit.json"
        path.write_text(permit.model_dump_json())
        select_simulation_trial(
            read_simulation_trial(path),
            [package],
            map_sha256="c" * 64,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_sha256=SENSOR_SHA256,
        )


# 功能：
#   检查不允许与试飞混用的运行模式在读取文件前即被拒绝。
# 输入：
#   settings：冲突运行模式。
# 输出：
#   无：通过拒绝断言或抛出测试失败。
@pytest.mark.parametrize(
    "settings",
    [{"fallback": "kimi"}, {"qualifications": [1]}, {"admissions": [1]}, {"incompatible": True}],
)
def test_trial_rejects_mixed_modes(settings):
    with pytest.raises(ValueError, match="MODE_CONFLICT"):
        validate_trial_configuration(None, (), None, object(), **settings)


# 功能：
#   核验桌面资源目录明确分离候选许可，损坏的摘要不能静默回退。
# 输入：
#   tmp_path：隔离资源目录。
# 输出：
#   无：通过资源读取和拒绝断言或抛出测试失败。
def test_runtime_trial_catalog(tmp_path):
    root = tmp_path / "runtime" / "local-policy"
    root.mkdir(parents=True)
    package = _package(root / "candidate", continuous_control=True)
    content = _permit(package).model_dump_json().encode()
    (root / "trial.json").write_bytes(content)
    catalog = {
        "schema_version": "dronedream.local-policy-runtime-catalog.v2",
        "deployment_scope": "simulation-trial",
        "entries": [
            {
                "package": "candidate",
                "package_sha256": package.package_sha256,
                "receipt": {
                    "kind": "simulation-trial",
                    "path": "trial.json",
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
            }
        ],
    }
    (root / "catalog.json").write_text(json.dumps(catalog))
    manager = object.__new__(RuntimeManager)
    manager.resource_root = tmp_path
    packages, qualifications, admissions = manager._local_policy_runtime_resources()
    assert packages == (root / "candidate",)
    assert qualifications == admissions == ()
    assert (
        manager._local_policy_trial_resource(packages, qualifications, admissions)
        == root / "trial.json"
    )
    (root / "trial.json").write_bytes(content + b" ")
    with pytest.raises(RuntimeBridgeError, match="HASH_MISMATCH"):
        manager._local_policy_runtime_resources()


# 功能：
#   检查试飞环境只放行资源版本差异，不允许更换基础仿真引擎或资产身份。
# 输入：
#   tmp_path：隔离测试资产目录。
# 输出：
#   无：通过边界断言或抛出测试失败。
def test_trial_environment_keeps_engine_and_asset_identity(tmp_path):
    from test_asset_resolver_boundaries import _qualified_map

    from dronedream_agent_app.asset_runtime_resolver import trial_asset_environment_matches

    record = _qualified_map(tmp_path)
    previous = {"gazebo_sim": "8.0", "ros_distribution": "jazzy", "px4_commit": "a" * 40,
                "runtime_manifest_sha256": "b" * 64}
    record["manifest"]["qualification"]["environment_versions"] = previous
    current = previous | {"runtime_manifest_sha256": "c" * 64}
    assert trial_asset_environment_matches(record, current)
    assert not trial_asset_environment_matches(record, current | {"gazebo_sim": "9.0"})
    assert not trial_asset_environment_matches(record, {})
    assert not trial_asset_environment_matches(record | {"maturity": "raw"}, current)


# 功能：
#   检查两个角度分别来自不同的实际文件，未知来源不能被当成文件路径读取。
# 输入：
#   tmp_path：隔离的活动运行目录。
# 输出：
#   无：通过相机选择断言或抛出测试失败。
def test_live_camera_selects_actual_source(tmp_path):
    import threading

    manager = object.__new__(RuntimeManager)
    manager._lock = threading.RLock()
    manager._active_runs = {"trial": SimpleNamespace(
        run_dir=tmp_path, process=SimpleNamespace(poll=lambda: None))}
    (tmp_path / "live-frame.png").write_bytes(b"overview")
    (tmp_path / "live-onboard.png").write_bytes(b"onboard")
    assert manager.live_frame("trial").read_bytes() == b"overview"
    assert manager.live_frame("trial", "gazebo-onboard").read_bytes() == b"onboard"
    with pytest.raises(RuntimeBridgeError, match="SOURCE_INVALID"):
        manager.live_frame("trial", "../secret")
