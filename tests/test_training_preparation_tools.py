"""Upload integrity, authored semantics and controlled image supervision tests."""

import hashlib
import importlib.util
import json
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dronedream_agent_core.training.vision_corruptions import (
    RenderCorruption,
    corrupt_rendered_pair,
)
from dronedream_agent_core.training.vision_render_world import bind_render_resources


# 功能：
#   独立载入无自动主入口执行的准备工具。
# 输入：
#   name：当前仓库明确的脚本文件名。
# 输出：
#   module：用于直接调用纯逻辑的测试模块。
def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location("tool_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 功能：
#   上传内容或清单改变、绝对路径和越界路径都不能通过完整性验证。
# 输入：
#   tmp_path：上传夹具专属目录。
# 输出：
#   None：不返回业务数据。
def test_bundle_checks_every_byte_and_portable_path(tmp_path):
    module = script("verify_training_bundle.py")
    file = tmp_path / "file.txt"
    file.write_bytes(b"good")
    entry = {"path": file.name, "size_bytes": 4,
             "sha256": hashlib.sha256(b"good").hexdigest()}
    payload = {"schema": "dronedream.training-transfer-bundle.v1", "files": [entry]}
    manifest = tmp_path / "bundle-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert module.verify(tmp_path, digest)["verified"] is True
    file.write_bytes(b"evil")
    with pytest.raises(ValueError, match="FILE_MISMATCH"):
        module.verify(tmp_path, digest)
    file.write_bytes(b"good")
    for path in ("../outside.txt", "/root/file", "C:/file", "a\\b", "./file.txt"):
        entry["path"] = path
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="PATH_INVALID"):
            module.verify(tmp_path, digest)


# 功能：
#   作者语义必须显式且能对应每个可见面；未知类别不能被默认为可通行。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_authored_semantics_require_known_exact_visuals():
    module = script("prepare_school_vision_source.py")
    world = b'''<sdf><world name="test"><model name="m"><static>true</static>
    <link name="l"><visual name="pane-visual"/></link></model></world></sdf>'''
    semantics = {"collision_primitives": [{"name": "pane", "semantic": "window-glazing"}],
                 "visual_only_primitives": []}
    labels, counts = module.authored_labels(world, semantics)
    assert labels["visuals"] == {"m::l::pane-visual": 6} and counts[6] == 1
    semantics["collision_primitives"][0]["semantic"] = "unreviewed-material"
    with pytest.raises(ValueError, match="NOT_REVIEWED"):
        module.authored_labels(world, semantics)


# 功能：
#   场景移到采集目录后仍使用正确贴图，缺文件及目录逃逸均在启动渲染前失败。
# 输入：
#   tmp_path：源纹理隔离目录。
# 输出：
#   None：不返回业务数据。
def test_render_resources_bound_before_gazebo_launch(tmp_path):
    world = b'<sdf><world><albedo_map>floor.ppm</albedo_map></world></sdf>'
    with pytest.raises(FileNotFoundError):
        bind_render_resources(world, tmp_path)
    (tmp_path / "floor.ppm").write_bytes(b"texture")
    derived, hashes = bind_render_resources(world, tmp_path)
    assert (tmp_path / "floor.ppm").as_posix().encode() in derived
    assert hashes == {"floor.ppm": hashlib.sha256(b"texture").hexdigest()}
    with pytest.raises(ValueError):
        bind_render_resources(world.replace(b"floor.ppm", b"../floor.ppm"), tmp_path)


# 功能：
#   原始渲染不变，合成镜头遮挡不能留下不可见物体的语义标签，质量标签来自实际变换。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_corruption_preserves_raw_and_updates_visible_supervision():
    rgb_stream, mask_stream = BytesIO(), BytesIO()
    with Image.new("RGB", (16, 16), (200, 100, 50)) as image:
        image.save(rgb_stream, format="PNG")
    with Image.new("L", (16, 16), 5) as mask:
        mask.save(mask_stream, format="PNG")
    raw, raw_mask = rgb_stream.getvalue(), mask_stream.getvalue()
    config = RenderCorruption(exposure_multiplier=0.5, blur_sigma_px=2.0,
                              occlusion_rect=[0.0, 0.0, 0.5, 1.0])
    changed, labels, quality = corrupt_rendered_pair(raw, raw_mask, config)
    assert raw == rgb_stream.getvalue() and raw_mask == mask_stream.getvalue()
    with Image.open(BytesIO(changed)) as image, Image.open(BytesIO(labels)) as mask:
        pixels, classes = np.array(image), np.array(mask)
    assert not pixels[:, :8].any() and not classes[:, :8].any()
    assert (classes[:, 8:] == 5).all() and (pixels[:, 8:, 0] == 100).all()
    assert quality == {"blurred": 1.0, "occluded": 1.0}
    with pytest.raises(ValueError):
        RenderCorruption(occlusion_rect=[0.9, 0.0, 0.1, 1.0])


# 功能：
#   上传目录不能混入未索引的旧脚本或同名模块，即使所有索引文件的摘要都正确。
# 输入：
#   tmp_path、extra：隔离上传根和清单外文件位置。
# 输出：
#   None：混入额外文件时必须拒绝整个上传包。
@pytest.mark.parametrize("extra", ["torch.py", "scripts/json.py", "old/entry.py"])
def test_bundle_rejects_unlisted_code(tmp_path, extra):
    module = script("verify_training_bundle.py")
    (tmp_path / "good.txt").write_bytes(b"good")
    payload = {"schema": "dronedream.training-transfer-bundle.v1", "files": [
        {"path": "good.txt", "sha256": hashlib.sha256(b"good").hexdigest(), "size_bytes": 4}]}
    manifest = tmp_path / "bundle-manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    additional = tmp_path / extra
    additional.parent.mkdir(parents=True, exist_ok=True)
    additional.write_text("# obsolete code", encoding="utf-8")
    with pytest.raises(ValueError, match="UNLISTED"):
        module.verify(tmp_path, digest)


# 功能：
#   重复配置键必须在读取大数据或建立输出前被拒绝，避免教师与训练器采用歧义配置。
# 输入：
#   tmp_path、monkeypatch、name：隔离目录、入口参数替身和待测训练脚本。
# 输出：
#   None：两种入口均不能接受重复键。
@pytest.mark.parametrize("name", [
    "train_native_action_risk.py", "build_native_action_risk_dataset.py"])
def test_action_risk_cli_rejects_ambiguous_config(tmp_path, monkeypatch, name):
    module = script(name)
    config = tmp_path / "config.json"
    config.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")
    output = tmp_path / "output"
    arguments = (
        ["--config", str(config), "--training-dataset", "missing-train",
         "--validation-dataset", "missing-validation"]
        if name.startswith("train_") else
        ["--teacher-config", str(config), "--episode", "missing-episode"]
    )
    monkeypatch.setattr("sys.argv", [name, *arguments, "--output", str(output)])
    with pytest.raises(ValueError, match="[Dd][Uu][Pp][Ll][Ii][Cc][Aa][Tt][Ee]"):
        module.main()
    assert not output.exists()


# 功能：
#   对照上传 wheel 核验已安装模块，旧版本或额外的旧模块都不能通过环境检查。
# 输入：
#   tmp_path、monkeypatch：隔离目录及安装位置替身。
# 输出：
#   None：原始模块通过，替换内容或增加模块后拒绝。
def test_training_environment_requires_exact_installed_core(tmp_path, monkeypatch):
    module = script("verify_training_bundle.py")
    bundle, site = tmp_path / "bundle", tmp_path / "site"
    bundle.mkdir()
    package = site / "dronedream_agent_core"
    package.mkdir(parents=True)
    target = package / "__init__.py"
    target.write_bytes(b"# current\n")
    wheel_path = bundle / "dronedream_flight_agent_core-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr("dronedream_agent_core/__init__.py", target.read_bytes())
    monkeypatch.setattr("importlib.metadata.distribution",
                        lambda name: SimpleNamespace(locate_file=lambda path: site / path))
    assert module.verify_installed_core(bundle) == 1
    target.write_bytes(b"# outdated\n")
    with pytest.raises(ValueError, match="CORE_MISMATCH"):
        module.verify_installed_core(bundle)
    target.write_bytes(b"# current\n")
    (package / "obsolete.py").write_bytes(b"# untracked legacy module\n")
    with pytest.raises(ValueError, match="EXTRA_OR_MISSING"):
        module.verify_installed_core(bundle)


# 功能：
#   辅助专家不能从含重复字段的证据产生标签，也不能覆盖另一作业已发布的数据。
# 输入：
#   tmp_path：隔离输入和输出目录。
# 输出：
#   None：歧义输入及已存在输出均被拒绝。
def test_advisor_builder_strict_evidence_and_exclusive_output(tmp_path):
    module = script("build_local_advisor_dataset.py")
    record = tmp_path / "record.jsonl"
    record.write_bytes(b'{"status":"verified","status":"failed"}\n')
    for loader in (module._json, module._jsonl):
        with pytest.raises(ValueError, match="(?i)duplicate"):
            loader(record)
    output = tmp_path / "output.jsonl"
    output.write_bytes(b"keep this evidence\n")
    with pytest.raises(FileExistsError):
        module._write_jsonl(output, {})
    assert output.read_bytes() == b"keep this evidence\n"


# 功能：
#   保留当前因果数据组装和真实部署验证入口，排除不含完整时序历史的旧数据入口。
# 输入：
#   无。
# 输出：
#   None：当前数据入口在包中，旧入口必须排除。
def test_bundle_uses_executed_control_dataset_entrypoint():
    module = script("prepare_training_bundle.py")
    assert "build_executed_policy_dataset.py" in module.SCRIPTS
    assert "assemble_causal_control_data.py" in module.SCRIPTS
    assert "validate_causal_control_role.py" in module.SCRIPTS
    assert "verify_causal_control_data.py" in module.SCRIPTS
    assert "evaluate_frozen_causal_role.py" in module.SCRIPTS
    assert len(module.SCRIPTS) == len(set(module.SCRIPTS))
    assert "build_local_policy_dataset.py" not in module.SCRIPTS


# 功能：
#   检查可搬运包包含实际训练器能够读取的固定控制配方，不依赖开发机私有路径。
# 输入：
#   无：读取仓库内公开的控制配置及上传白名单。
# 输出：
#   None：配置与训练接口不一致或遗漏白名单时失败。
def test_bundle_includes_valid_fixed_control_recipes():
    from dronedream_agent_core.training.causal_policy import CausalPolicyConfig
    from dronedream_agent_core.training.causal_regularization import (
        CausalRegularization,
        validate_regularization,
    )

    module = script("prepare_training_bundle.py")
    root = Path(__file__).parents[1]
    config_path = "training/control/baseline.json"
    regularization_path = "training/control/regularization.json"
    assert {config_path, regularization_path} <= set(module.SUPPORT_FILES)
    assert len(module.SUPPORT_FILES) == len(set(module.SUPPORT_FILES))
    config = CausalPolicyConfig.model_validate_json((root / config_path).read_bytes(), strict=True)
    regularization = CausalRegularization.model_validate_json((root / regularization_path).read_bytes(), strict=True)
    validate_regularization(regularization, config.visual_feature_count)
    assert (config.history_length, config.visual_feature_count, config.seed) == (16, 139, 805)
    assert regularization.balance_mission_groups and regularization.visual_block_dropout == .35


# 功能：
#   云控制配方保持相同结构和优化参数，仅允许预声明的种子及单轮启动检查差异。
# 输入：
#   无：读取实际上传白名单、配方及云端执行脚本。
# 输出：
#   None：设备、保护路径、预算或配方与实际 CLI 不匹配时失败。
def test_cloud_control_recipes_are_bounded_and_packaged():
    from dronedream_agent_core.training.causal_policy import CausalPolicyConfig

    root = Path(__file__).parents[1]
    module = script("prepare_training_bundle.py")
    baseline = CausalPolicyConfig.model_validate_json((root / "training/control/baseline.json").read_bytes())
    for recipe, seed, epochs in (("smoke", 805, 1), ("seed-806", 806, 120), ("seed-807", 807, 120)):
        path = f"training/control/{recipe}.json"
        assert path in module.SUPPORT_FILES
        config = CausalPolicyConfig.model_validate_json((root / path).read_bytes(), strict=True)
        assert config.model_dump() == {**baseline.model_dump(), "seed": seed, "epochs": epochs}
    path = "training/cloud/run_control.sh"
    assert path in module.SUPPORT_FILES
    recipe = (root / path).read_text(encoding="utf-8")
    assert recipe.index("unset PYTHONPATH PYTHONHOME") < recipe.index('"$py"')
    assert "--device cuda" in recipe and "--check-installed" in recipe
    assert "verify_causal_control_data.py" in recipe
    assert "validate_causal_control_role.py" in recipe
    assert "timeout --signal=TERM --kill-after=30s 3600" in recipe
    assert "test-replay.jsonl" not in recipe
    assert "--base-policy" not in recipe
    assert '--encoder-training-receipt "$encoder_root/training-receipt.json"' in recipe


# 功能：
#   检查单卡正式训练入口的恢复点间隔及依赖声明，防止默认 30 轮耗尽作业检查点预算。
# 输入：
#   无：读取本仓库的正式云训练脚本及包元数据。
# 输出：
#   None：验证固定配方且环境清理先于任何解释器执行。
def test_cloud_training_storage_and_runtime_dependencies():
    import tomllib

    root = Path(__file__).parents[1]
    recipe = (root / "training/cloud/run_vision.sh").read_text(encoding="utf-8")
    assert "--checkpoint-batches 500" in recipe
    assert recipe.index("unset PYTHONPATH PYTHONHOME") < recipe.index('"$py"')
    bootstrap = (root / "training/cloud/bootstrap.sh").read_text(encoding="utf-8")
    assert bootstrap.index("unset PYTHONPATH PYTHONHOME") < bootstrap.index("python3")
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["local-vision-training"]
    assert "onnxruntime>=1.29,<2" in dependencies
