"""Host contract fixtures only; actual Ogre loading is tested in Gazebo separately."""

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from dronedream_agent_core import simulation_render_cache as cache


# 功能：
#   验证缺失当前 Ogre Next、缺少动态渲染模块或混入旧 ABI 时拒绝启动准备。
# 输入：
#   libraries：故意不完整或混合的依赖路径列表。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "libraries",
    [
        [],
        ["/a/libOgreMain.so.1.9.0"],
        [
            "/a/libOgreNextMain.so.2.3.1",
            "/a/libOgreMain.so.1.9.0",
            "/a/RenderSystem_GL3Plus.so.2.3.1",
        ],
        ["/a/libOgreNextMain.so.2.3.1"],
        ["/a/libOgreNextMain.so.3.0", "/a/RenderSystem_GL3Plus.so.2.3.1"],
    ],
)
def test_mixed_ogre_generations_rejected_before_simulator(libraries):
    with pytest.raises(ValueError, match="OGRE_ABI"):
        cache.validate_ogre_dependencies(libraries)


# 功能：
#   验证当前明确适配的 Ogre 与渲染模块组合仍能通过。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_exact_ogre_next_and_dynamic_renderer_are_bound():
    cache.validate_ogre_dependencies(
        ["/a/libOgreNextMain.so.2.3.1", "/a/RenderSystem_GL3Plus.so.2.3.1"]
    )


# 功能：
#   建立独立配置、资源与假原生库，只替换原生构建校验而使用真实宿主准备逻辑。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换原生验证入口的测试工具。
# 输出：
#   arguments：准备渲染缓存所需的参数字典。
def fixture(tmp_path, monkeypatch):
    runtime, resources = tmp_path / "runtime", tmp_path / "resources"
    runtime.mkdir()
    resources.mkdir()
    (runtime / cache.LIBRARY).write_bytes(b"not-a-native-library-test-only")
    (resources / "material").write_bytes(b"test-material")
    config = tmp_path / "server.config"
    config.write_text('<server_config><plugins><plugin name="physics"/></plugins></server_config>')
    monkeypatch.setattr(
        cache, "validate_runtime", lambda p: {"library_sha256": cache.file_sha(p / cache.LIBRARY)}
    )
    arguments = dict(
        runtime_root=runtime,
        input_bundle=None,
        output=tmp_path / "capture",
        server_config=config,
        resources={"assets": resources},
        assets={"camera": "bound-test"},
        environment={"GALLIUM_DRIVER": "fixture-only"},
    )
    return arguments


# 功能：
#   写入与当前部署身份一致的假原生 XML 回执，用于宿主校验而非证明实际渲染。
# 输入：
#   args：部署参数，包含本次输出目录。
#   files：回执声明的缓存文件记录。
#   renderer：夹具中的显卡及渲染器身份摘要。
#   status：ready 或 complete 回执状态。
# 输出：
#   root：写入文件对应的 XML 根节点。
def ready(args, *, files=(), renderer="a" * 64, status="ready"):
    deployment = json.loads((args["output"] / "deployment.json").read_text())
    root = ET.Element(
        "receipt",
        {
            "status": status,
            "identity_sha256": deployment["identity_sha256"],
            "renderer_sha256": renderer,
            "qualification_granted": "false",
            "error": "",
            "prepare_ms": "1.2",
            "microcode_supported": "true",
            "mode": deployment["mode"],
        },
    )
    for row in files:
        ET.SubElement(root, "file", {name: str(value) for name, value in row.items()})
    path = args["output"] / ("ready.xml" if status == "ready" else "finished.xml")
    path.write_bytes(ET.tostring(root))
    return root


# 功能：
#   为未生成飞机的夹具完成准备、写入和封存，得到可用于加载测试的真实文件包。
# 输入：
#   args：隔离部署参数。
# 输出：
#   result：文件包目录、文件数、字节数及 NOT_STARTED 终态。
def sealed(args):
    cache.prepare_render_cache(**args)
    ready(args)
    rows = []
    for kind, name in [(0, "microcode.bin"), (1, "hlms-1.bin")]:
        path = args["output"] / "output" / name
        path.write_bytes(b"opaque cache test fixture")
        rows.append(
            {
                "type": kind,
                "name": name,
                "bytes": path.stat().st_size,
                "sha256": cache.file_sha(path),
            }
        )
    ready(args, files=rows, status="complete")
    result = cache.finalize_render_cache(
        args["output"],
        all_owned_processes_exited=True,
        flight_vehicle_spawn_attempted=False,
        timing_path=Path("unused"),
    )
    return result


# 功能：
#   验证准备只创建运行专属配置、保留原插件及源配置，并拒绝覆盖已有输出目录。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：替换原生验证入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_prepare_does_not_change_installed_config_or_resources(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    before = args["server_config"].read_bytes()
    result = cache.prepare_render_cache(**args)
    assert args["server_config"].read_bytes() == before
    assert result["qualification_granted"] is False
    plugins = ET.parse(args["output"] / "server.config").findall("./plugins/plugin")
    assert [item.get("name") for item in plugins] == ["physics", "dronedream::RenderPreparation"]
    assert plugins[1].findtext("output") == str(args["output"].resolve())
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        cache.verify_render_preparation(args["output"])
    ready(args)
    assert cache.verify_render_preparation(args["output"])["loaded_files"] == 0
    with pytest.raises(FileExistsError):
        cache.prepare_render_cache(**args)


# 功能：
#   验证已封存缓存通过实际文件复制及回读使用，显卡身份变化仍被拒绝。
# 输入：
#   tmp_path：隔离捕获及加载目录。
#   monkeypatch：替换原生验证入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_stopped_bundle_is_copied_and_all_actual_files_read_back(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    result = sealed(args)
    assert result["terminal_state"] == "NOT_STARTED"
    args.update(output=tmp_path / "load", input_bundle=Path(result["bundle"]))
    deployment = cache.prepare_render_cache(**args)
    assert deployment["mode"] == "load"
    assert len(deployment["input_files"]) == 2
    ready(args, files=deployment["input_files"])
    assert cache.verify_render_preparation(args["output"])["loaded_files"] == 2
    # 环境变量仅是请求，不能把另一块显卡的原生报告当作当前渲染器身份。
    ready(args, files=deployment["input_files"], renderer="b" * 64)
    with pytest.raises(ValueError, match="READBACK_MISMATCH"):
        cache.verify_render_preparation(args["output"])


# 功能：
#   逐项更换当前资源或篡改缓存身份，验证在创建加载目录前拒绝复用旧缓存。
# 输入：
#   tmp_path：隔离捕获及加载目录。
#   monkeypatch：替换原生验证入口的测试工具。
#   change：本次修改的资源或身份字段。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change", ["material", "camera", "runtime", "environment", "digest", "qualification"]
)
def test_old_or_relabelled_cache_is_not_reused(tmp_path, monkeypatch, change):
    args = fixture(tmp_path, monkeypatch)
    result = sealed(args)
    args.update(output=tmp_path / "load", input_bundle=Path(result["bundle"]))
    if change == "material":
        (args["resources"]["assets"] / "material").write_bytes(b"new")
    elif change == "camera":
        args["assets"] = {"camera": "different"}
    elif change == "runtime":
        (args["runtime_root"] / cache.LIBRARY).write_bytes(b"changed")
    elif change == "environment":
        args["environment"] = {"GALLIUM_DRIVER": "different"}
    else:
        path = args["input_bundle"] / "cache-bundle.json"
        data = json.loads(path.read_text())
        data["identity_sha256" if change == "digest" else "qualification_granted"] = (
            "c" * 64 if change == "digest" else True
        )
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        cache.prepare_render_cache(**args)
    assert not args["output"].exists()


# 功能：
#   验证文件清单中的路径穿越、重复类型、伪整数、缺失字段和内容变化不会进入加载阶段。
# 输入：
#   tmp_path：隔离捕获及加载目录。
#   monkeypatch：替换原生验证入口的测试工具。
#   change：本次清单或文件缺陷。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "change", ["traversal", "duplicate", "bool", "zero", "missing", "tamper", "no_hlms"]
)
def test_malformed_cache_files_rejected_before_output_creation(tmp_path, monkeypatch, change):
    args = fixture(tmp_path, monkeypatch)
    result = sealed(args)
    args.update(output=tmp_path / "load", input_bundle=Path(result["bundle"]))
    path = args["input_bundle"] / "cache-bundle.json"
    data = json.loads(path.read_text())
    rows = data["files"]
    if change == "traversal":
        rows[0]["name"] = "../outside"
    elif change == "duplicate":
        rows[1] = rows[0].copy()
    elif change == "bool":
        rows[0]["type"] = False
    elif change == "zero":
        rows[0]["bytes"] = 0
    elif change == "missing":
        rows[0].pop("name")
    elif change == "tamper":
        (args["input_bundle"] / "microcode.bin").write_bytes(b"changed")
    else:
        data["files"] = rows[:1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="RENDER_CACHE_"):
        cache.prepare_render_cache(**args)
    assert not args["output"].exists()


# 功能：
#   验证仿真进程退出与实际落地是独立条件，文字标签不能单独触发缓存封存。
# 输入：
#   tmp_path：隔离部署目录。
#   monkeypatch：替换原生验证入口的测试工具。
#   stopped：自有进程是否已退出。
#   spawned：是否尝试过生成飞机。
#   cleanup：用于反例的原生清理记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "stopped,spawned,cleanup",
    [
        (False, False, {}),
        (True, True, {}),
        (True, True, {"land": "requested", "landing_observation": {"state": "ON_GROUND"}}),
        (True, True, {"land": "confirmed_on_ground", "landing_observation": {"state": "IN_AIR"}}),
    ],
)
def test_process_exit_or_landing_label_alone_cannot_promote(
    tmp_path, monkeypatch, stopped, spawned, cleanup
):
    args = fixture(tmp_path, monkeypatch)
    cache.prepare_render_cache(**args)
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"cleanup": cleanup}))
    with pytest.raises(ValueError, match="NOT_STOPPED|LANDING_NOT_CONFIRMED"):
        cache.finalize_render_cache(
            args["output"],
            all_owned_processes_exited=stopped,
            flight_vehicle_spawn_attempted=spawned,
            timing_path=timing,
        )
    assert not (args["output"] / "bundle").exists()


# 功能：
#   验证引擎支持的仅 HLMS 缓存无需伪造 microcode 文件也能正常加载。
# 输入：
#   tmp_path：隔离捕获及加载目录。
#   monkeypatch：替换原生验证入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_hlms_cache_without_microcode_is_explicitly_valid(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    result = sealed(args)
    args.update(output=tmp_path / "load", input_bundle=Path(result["bundle"]))
    path = args["input_bundle"] / "cache-bundle.json"
    data = json.loads(path.read_text())
    data["files"] = data["files"][1:]
    path.write_text(json.dumps(data))
    deployment = cache.prepare_render_cache(**args)
    assert [row["name"] for row in deployment["input_files"]] == ["hlms-1.bin"]
    ready(args, files=deployment["input_files"])
    assert cache.verify_render_preparation(args["output"])["loaded_files"] == 1
    assert not (args["output"] / "input/microcode.bin").exists()


# 功能：
#   验证缺少原生结束回执时不创建可复用缓存包，即使已有 ready 回执。
# 输入：
#   tmp_path：隔离部署目录。
#   monkeypatch：替换原生验证入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_failed_or_missing_finalization_does_not_promote_partial_files(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    cache.prepare_render_cache(**args)
    ready(args)
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        cache.finalize_render_cache(
            args["output"],
            all_owned_processes_exited=True,
            flight_vehicle_spawn_attempted=False,
            timing_path=Path("unused"),
        )
    assert not (args["output"] / "bundle").exists()
