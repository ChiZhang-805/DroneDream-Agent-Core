"""Package identity and eligibility boundaries; receipts below are explicit test fixtures."""

import json
import math

import pytest
from test_local_policy_packages import (
    MAP_SHA256,
    SENSOR_SHA256,
    VEHICLE_SHA256,
    _admission,
    _qualification,
    _write_package,
)

from dronedream_agent_core import local_policy_packages as packages


# 功能：
#   基座清单出现重复字段时拒绝加载，不把最后一个值当作唯一发布内容。
# 输入：
#   tmp_path：有效包及损坏清单的测试目录。
# 输出：
#   None：不返回业务数据。
def test_package_rejects_duplicate_manifest_fields(tmp_path):
    package = _write_package(tmp_path / "package", package_id="test.package")
    manifest = package.manifest_path
    manifest.write_bytes(b'{"scope":"map-specialist",' + manifest.read_bytes()[1:])
    with pytest.raises(ValueError):
        packages.load_local_policy_package(package.root)


# 功能：
#   模型相对路径须跨 Windows/Linux 保持同一含义，点路径、设备名及数据流等非法形式拒绝。
# 输入：
#   value：候选非法相对路径。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "value",
    [
        ".",
        "folder/model.onnx:stream",
        "con.onnx",
        "model.onnx.",
        "folder/../model.onnx",
        "model\x00.onnx",
    ],
)
def test_package_rejects_nonportable_artifact_paths(value):
    with pytest.raises(ValueError):
        packages.LocalPolicyArtifact(
            role="local-navigation-policy",
            relative_path=value,
            sha256="a" * 64,
            input_names=["state_features"],
            output_names=["risk_score"],
        )


# 功能：
#   两个角色不能声明大小写不同但在 Windows 指向同一文件的路径。
# 输入：
#   tmp_path：含导航与风险角色的普通包。
# 输出：
#   None：不返回业务数据。
def test_package_rejects_casefold_path_collision(tmp_path):
    package = _write_package(tmp_path / "package", package_id="test.package")
    payload = package.manifest.model_dump(mode="json")
    payload["artifacts"][1]["relative_path"] = payload["artifacts"][0]["relative_path"].upper()
    with pytest.raises(ValueError):
        packages.LocalPolicyPackageManifest.model_validate(payload)


# 功能：
#   包内模型改为指向同一包内其他文件的链接后也必须拒绝，不能 resolve 后丢失链接证据。
# 输入：
#   tmp_path：普通模型包目录。
# 输出：
#   None：不返回业务数据。
def test_package_rejects_artifact_symlink_before_resolve(tmp_path):
    package = _write_package(tmp_path / "package", package_id="test.package")
    artifact = package.artifact_paths["local-navigation-policy"]
    target = artifact.with_name("actual.onnx")
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Host cannot create test symlink: {error}")
    with pytest.raises(ValueError):
        packages.load_local_policy_package(package.root)


# 功能：
#   地图专家不能选用另一机型或传感器的通用包作回退，即使该包对自己的绑定有合格回执。
# 输入：
#   tmp_path：基座及地图专家目录。
#   simulation：检查仿真选择或正式资格选择。
#   field：故意不兼容的设备绑定字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("simulation", [False, True])
@pytest.mark.parametrize("field", ["vehicle_sha256", "sensor_contract_sha256"])
def test_specialist_rejects_incompatible_rollback_binding(tmp_path, simulation, field):
    base = _write_package(tmp_path / "base", package_id="test.base")
    payload = json.loads(base.manifest_path.read_bytes())
    payload[field] = "f" * 64
    base.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    base = packages.load_local_policy_package(base.root)
    specialist = _write_package(
        tmp_path / "specialist",
        package_id="test.specialist",
        scope="map-specialist",
        base_package_sha256=base.package_sha256,
        map_sha256=MAP_SHA256,
    )
    options = dict(
        packages=[base, specialist],
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )
    if simulation:
        receipts = [
            _admission(base, suffix="a", map_sha256=None).model_copy(update={field: "f" * 64}),
            _admission(specialist, suffix="b", map_sha256=MAP_SHA256),
        ]
        with pytest.raises(ValueError, match="NO_SIMULATION_ADMITTED"):
            packages.select_local_policy_for_simulation(
                **options, admissions=receipts, qualification_receipts=[]
            )
    else:
        receipts = [
            _qualification(base, suffix="a", map_sha256=None).model_copy(update={field: "f" * 64}),
            _qualification(specialist, suffix="b", map_sha256=MAP_SHA256),
        ]
        with pytest.raises(ValueError, match="NO_QUALIFIED"):
            packages.select_local_policy(**options, receipts=receipts)


# 功能：
#   绕过模型校验改坏的资格回执不能被选择函数继续信任，碰撞或非完整成功证据必须拒绝。
# 输入：
#   tmp_path：普通基座目录。
#   mutation：不经校验注入的错误资格字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "mutation",
    [
        {"collision_count": 1},
        {"successful_trial_count": 0},
        {"issue_codes": ["failed"]},
        {"trial_count": 0},
    ],
)
def test_selection_revalidates_mutated_qualification(tmp_path, mutation):
    base = _write_package(tmp_path / "base", package_id="test.base")
    receipt = _qualification(base, suffix="a", map_sha256=None).model_copy(update=mutation)
    with pytest.raises(ValueError):
        packages.select_local_policy(
            packages=[base],
            receipts=[receipt],
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )


# 功能：
#   仿真准入必须使用已经报告的全链路延迟，不能仅检查快速导航头而漏掉其他专家开销。
# 输入：
#   tmp_path：带 500 毫秒预算的基座及超时准入回执。
# 输出：
#   None：不返回业务数据。
def test_simulation_selection_checks_combined_latency(tmp_path):
    base = _write_package(tmp_path / "base", package_id="test.base")
    receipt = _admission(base, suffix="a", map_sha256=None).model_copy(
        update={"combined_p99_inference_latency_ms": 501.0}
    )
    with pytest.raises(ValueError, match="NO_SIMULATION_ADMITTED"):
        packages.select_local_policy_for_simulation(
            packages=[base],
            admissions=[receipt],
            qualification_receipts=[],
            map_sha256=MAP_SHA256,
            vehicle_sha256=VEHICLE_SHA256,
            sensor_contract_sha256=SENSOR_SHA256,
        )


# 功能：
#   无意义的布尔或超大整数不能进入延迟排序，避免把无效测量当一毫秒或触发浮点溢出。
# 输入：
#   value：非法延迟标量。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [True, math.nan, math.inf, -1.0, 10**400])
def test_latency_percentile_rejects_invalid_numeric_values(value):
    with pytest.raises(ValueError):
        packages.latency_percentile([value], 99.0)


# 功能：
#   加载后内存清单、角色路径或磁盘模型变化都必须阻止选择，不能仅凭保留的旧包摘要放行。
# 输入：
#   tmp_path：普通包目录。
#   mutation：被更改的加载状态部位。
#   simulation：检查正式或仿真选择。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("simulation", [False, True])
@pytest.mark.parametrize("mutation", ["manifest", "paths", "model"])
def test_selection_rechecks_loaded_package_identity(tmp_path, mutation, simulation):
    base = _write_package(tmp_path / "base", package_id="test.base")
    qualification = _qualification(base, suffix="a", map_sha256=None)
    admission = _admission(base, suffix="b", map_sha256=None)
    if mutation == "manifest":
        base.manifest.maximum_inference_latency_ms = 1_000
    elif mutation == "paths":
        base.artifact_paths["local-navigation-policy"] = tmp_path / "another.onnx"
    else:
        base.artifact_paths["local-navigation-policy"].write_bytes(b"replaced")
    options = dict(
        packages=[base],
        map_sha256=MAP_SHA256,
        vehicle_sha256=VEHICLE_SHA256,
        sensor_contract_sha256=SENSOR_SHA256,
    )
    with pytest.raises(ValueError):
        if simulation:
            packages.select_local_policy_for_simulation(
                **options, admissions=[admission], qualification_receipts=[]
            )
        else:
            packages.select_local_policy(**options, receipts=[qualification])


# 功能：
#   模型散列期间清单变化时拒绝返回混合身份，保留变化后的原文件供调用方处理。
# 输入：
#   tmp_path：有效包目录。
#   monkeypatch：在实际散列后修改清单的工具。
# 输出：
#   None：不返回业务数据。
def test_loading_detects_manifest_change_during_model_hash(tmp_path, monkeypatch):
    base = _write_package(tmp_path / "base", package_id="test.base")
    hash_file = packages._file_sha256

    # 功能：
    #   在真实模型散列后改变发布清单，模拟两个读取阶段之间的来源变化。
    # 输入：
    #   path：模型文件路径。
    # 输出：
    #   digest：真实模型摘要。
    def change_manifest(path):
        digest = hash_file(path)
        payload = json.loads(base.manifest_path.read_bytes())
        payload["display_name"] = "Changed during load"
        base.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return digest

    monkeypatch.setattr(packages, "_file_sha256", change_manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        packages.load_local_policy_package(base.root)
