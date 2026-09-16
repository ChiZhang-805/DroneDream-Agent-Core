"""Temporary rendering receipts and fake binary bytes, not real renderer qualification."""

import json
from pathlib import Path

import pytest
from test_render_replica_runtime import fixture as replica_fixture
from test_simulation_render_cache import fixture as cache_fixture
from test_simulation_render_cache import ready

from dronedream_agent_core import render_artifact_io as artifact_io
from dronedream_agent_core import render_replica_runtime as replica
from dronedream_agent_core import simulation_render_cache as cache


# 功能：
#   验证构建回执不能用重复键隐藏相反的资格或原生测试标志。
# 输入：
#   tmp_path：隔离构建目录。
#   monkeypatch：替换实际源码根的测试工具。
# 输出：
#   None：不返回业务数据。
def test_replica_receipt_rejects_duplicate_keys(tmp_path, monkeypatch):
    root, _, _ = replica_fixture(tmp_path, monkeypatch)
    path = root / replica.RECEIPT
    content = path.read_text()
    path.write_text('{"qualification_granted": true,' + content[1:])
    with pytest.raises(ValueError):
        replica.validate_replica_runtime(root)


# 功能：
#   验证源码子目录也参与构建绑定，修改其中头文件后不能沿用旧构建回执。
# 输入：
#   tmp_path：隔离源码和构建目录。
#   monkeypatch：替换源码根的测试工具。
# 输出：
#   None：不返回业务数据。
def test_replica_binds_nested_source_files(tmp_path, monkeypatch):
    root, receipt, source = replica_fixture(tmp_path, monkeypatch)
    directory = source / "nested"
    directory.mkdir()
    header = directory / "geometry.hpp"
    header.write_bytes(b"current geometry")
    receipt["sources"] = replica.source_files()
    assert "nested/geometry.hpp" in receipt["sources"]
    (root / replica.RECEIPT).write_text(json.dumps(receipt))
    assert replica.validate_replica_runtime(root)["native_tests_passed"] is True
    header.write_bytes(b"different geometry")
    with pytest.raises(ValueError):
        replica.validate_replica_runtime(root)


# 功能：
#   验证空源码目录不能和空来源回执互相证明构建有效。
# 输入：
#   tmp_path：隔离源码及构建目录。
#   monkeypatch：替换源码根的测试工具。
# 输出：
#   None：不返回业务数据。
def test_empty_native_source_tree_is_not_a_valid_binding(tmp_path, monkeypatch):
    root, receipt, source = replica_fixture(tmp_path, monkeypatch)
    # 仅删除本测试创建的单个源文件，故障夹具不访问项目源码。
    (source / "source.cpp").unlink()
    receipt["sources"] = {}
    (root / replica.RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        replica.validate_replica_runtime(root)


# 功能：
#   生成尚未封存的仿真缓存夹具，提供完整 ready/finished 回执及一个 HLMS 文件。
# 输入：
#   tmp_path：测试专属目录。
#   monkeypatch：替换原生运行包校验的测试工具。
# 输出：
#   arguments：可用于测试封存阶段的部署参数。
def pending_capture(tmp_path, monkeypatch):
    arguments = cache_fixture(tmp_path, monkeypatch)
    cache.prepare_render_cache(**arguments)
    ready(arguments)
    path = arguments["output"] / "output" / "hlms-1.bin"
    path.write_bytes(b"isolated cache bytes")
    rows = [{"type": 1, "name": path.name, "bytes": path.stat().st_size,
             "sha256": cache.file_sha(path)}]
    ready(arguments, files=rows, status="complete")
    return arguments


# 功能：
#   验证飞行后封存缓存必须有字面 True 的原生落地确认，不能只相信状态文字。
# 输入：
#   tmp_path：本轮测试目录。
#   monkeypatch：替换运行包校验的测试工具。
#   confirmation：缺失、假值或错误类型的确认标志。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("confirmation", [None, False, 1, "true"])
def test_cache_promotion_requires_literal_native_confirmation(tmp_path, monkeypatch, confirmation):
    arguments = pending_capture(tmp_path, monkeypatch)
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"cleanup": {"land": "confirmed_on_ground",
        "landing_observation": {"state": "ON_GROUND", "confirmed": confirmation}}}))
    with pytest.raises(ValueError, match="LANDING_NOT_CONFIRMED"):
        cache.finalize_render_cache(arguments["output"], all_owned_processes_exited=True,
                                   flight_vehicle_spawn_attempted=True, timing_path=timing)
    assert not (arguments["output"] / "bundle").exists()


# 功能：
#   验证 ready XML 不接受文档类型声明，避免原生回执使用另一套宽松 XML 解析规则。
# 输入：
#   tmp_path：测试专属部署目录。
#   monkeypatch：替换运行包校验的测试工具。
# 输出：
#   None：不返回业务数据。
def test_cache_receipt_rejects_doctype(tmp_path, monkeypatch):
    arguments = pending_capture(tmp_path, monkeypatch)
    path = arguments["output"] / "ready.xml"
    path.write_bytes(b'<!DOCTYPE receipt [<!ENTITY unused "value">]>' + path.read_bytes())
    with pytest.raises(ValueError):
        cache.verify_render_preparation(arguments["output"])


# 功能：
#   验证部署内容与身份摘要不一致时，不接受引用旧摘要的 ready 回执。
# 输入：
#   tmp_path：隔离部署目录。
#   monkeypatch：替换运行包校验的测试工具。
# 输出：
#   None：不返回业务数据。
def test_cache_deployment_identity_is_recomputed_before_acceptance(tmp_path, monkeypatch):
    arguments = pending_capture(tmp_path, monkeypatch)
    path = arguments["output"] / "deployment.json"
    deployment = json.loads(path.read_text())
    deployment["identity"]["assets"] = {"camera": "replacement"}
    path.write_text(json.dumps(deployment))
    with pytest.raises(ValueError):
        cache.verify_render_preparation(arguments["output"])


# 功能：
#   验证准备后的服务器配置被更换时，旧 ready 文件不能继续宣称当前部署通过。
# 输入：
#   tmp_path：隔离部署目录。
#   monkeypatch：替换运行包校验的测试工具。
# 输出：
#   None：不返回业务数据。
def test_cache_configuration_replacement_invalidates_ready_receipt(tmp_path, monkeypatch):
    arguments = pending_capture(tmp_path, monkeypatch)
    config = arguments["output"] / "server.config"
    config.write_bytes(config.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        cache.verify_render_preparation(arguments["output"])


# 功能：
#   验证公共制品校验实现变化后不能沿用只绑定旧入口文件的宿主身份。
# 输入：
#   tmp_path：隔离部署目录。
#   monkeypatch：仅改变公共制品模块的摘要返回值，不修改真实源码文件。
# 输出：
#   None：不返回业务数据。
def test_shared_artifact_implementation_is_part_of_cache_identity(tmp_path, monkeypatch):
    arguments = pending_capture(tmp_path, monkeypatch)
    original = cache.file_sha

    # 功能：
    #   仅为公共制品实现模拟新的内容摘要，其他文件仍用真实哈希。
    # 输入：
    #   path：请求摘要的源码或制品路径。
    # 输出：
    #   digest：模拟更新或真实计算的摘要。
    def changed_shared_source(path):
        digest = "f" * 64 if path.name == "render_artifact_io.py" else original(path)
        return digest

    monkeypatch.setattr(cache, "file_sha", changed_shared_source)
    with pytest.raises(ValueError):
        cache.verify_render_preparation(arguments["output"])


# 功能：
#   验证源码目录条目和字节预算分别生效，不先无限遍历或读取后才检查上限。
# 输入：
#   tmp_path：隔离源码目录。
#   monkeypatch：临时收紧其中一个源码预算。
#   budget：待验证的预算名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", ["MAX_NATIVE_SOURCE_ENTRIES", "MAX_NATIVE_SOURCE_BYTES"])
def test_native_source_inventory_enforces_both_budgets(tmp_path, monkeypatch, budget):
    (tmp_path / "a.cpp").write_bytes(b"source bytes")
    monkeypatch.setattr(artifact_io, budget, 0)
    with pytest.raises(ValueError, match="SOURCE_LIMIT"):
        artifact_io.source_inventory(tmp_path)


# 功能：
#   验证来源在复制前增长时不能把超声明字节写入目标后才发现不匹配。
# 输入：
#   tmp_path：隔离缓存源和复制目录。
#   monkeypatch：仅在首次读取源文件时注入增长。
# 输出：
#   None：不返回业务数据。
def test_cache_copy_detects_growth_before_writing_extra_bytes(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    file = source / "hlms-1.bin"
    file.write_bytes(b"ab")
    rows = [{"type": 1, "name": file.name, "bytes": 2, "sha256": cache.file_sha(file)}]
    original_open = Path.open

    # 功能：
    #   在首次源文件 stat 与实际打开之间增大测试文件，保留真实文件描述符检查。
    # 输入：
    #   path：被打开的路径。
    #   args：原始打开位置参数。
    #   kwargs：原始打开关键字参数。
    # 输出：
    #   stream：真实文件流。
    def grow_before_open(path, *args, **kwargs):
        if path == file and args == ("rb",):
            with original_open(path, "ab") as changed:
                changed.write(b"extra bytes")
        stream = original_open(path, *args, **kwargs)
        return stream

    monkeypatch.setattr(Path, "open", grow_before_open)
    with pytest.raises(ValueError):
        cache._copy_cache_files(rows, source, target)
    copied = target / file.name
    assert not copied.exists() or copied.stat().st_size <= 2


# 功能：
#   验证封存记录中的时序回执摘要来自已经验证的原文，不在验证后另读一份替换内容。
# 输入：
#   tmp_path：隔离捕获目录。
#   monkeypatch：在第一次读取时序回执后改变磁盘文件。
# 输出：
#   None：不返回业务数据。
def test_terminal_evidence_hash_uses_same_bytes_as_landing_decision(tmp_path, monkeypatch):
    import hashlib

    arguments = pending_capture(tmp_path, monkeypatch)
    timing = tmp_path / "timing.json"
    content = json.dumps({"cleanup": {"landing_observation": {
        "state": "ON_GROUND", "confirmed": True}}}).encode()
    timing.write_bytes(content)
    original = cache.read_object_snapshot

    # 功能：
    #   返回实际读取的旧快照，然后模拟另一个写入者替换磁盘上的时序文件。
    # 输入：
    #   path：被读取的 JSON 路径。
    #   kwargs：字节预算和错误码。
    # 输出：
    #   snapshot：实际读取并验证的字典与原文字节。
    def replace_after_read(path, **kwargs):
        snapshot = original(path, **kwargs)
        if path == timing:
            timing.write_bytes(b'{"cleanup":{}}')
        return snapshot

    monkeypatch.setattr(cache, "read_object_snapshot", replace_after_read)
    result = cache.finalize_render_cache(arguments["output"], all_owned_processes_exited=True,
                                        flight_vehicle_spawn_attempted=True, timing_path=timing)
    manifest = json.loads((Path(result["bundle"]) / "cache-bundle.json").read_text())
    assert manifest["terminal_timing_sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["terminal_timing_sha256"] != hashlib.sha256(timing.read_bytes()).hexdigest()


# 功能：
#   验证两个原生构建入口都拒绝复用已有构建目录，避免继承之前的 CMake 缓存选项。
# 输入：
#   tmp_path：测试专属的已存在输出目录。
#   monkeypatch：替换命令行及禁止启动编译器的测试工具。
#   script：待验证的构建入口文件名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("script", ["build_render_preparation.py", "build_render_replica.py"])
def test_native_builders_require_fresh_output_directory(tmp_path, monkeypatch, script):
    import runpy
    import subprocess
    import sys

    main = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / script))["main"]
    monkeypatch.setattr(sys, "argv", [script, "--output", str(tmp_path)])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("compiler invoked"))
    with pytest.raises(FileExistsError):
        main()


# 功能：
#   验证原生构建输出不能置于实际源码树中，避免写入生成文件后才发现源码绑定已变化。
# 输入：
#   tmp_path：隔离源码及脚本位置。
#   monkeypatch：替换源码位置、命令行与编译器入口。
#   script：待验证的构建入口文件名。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("script", ["build_render_preparation.py", "build_render_replica.py"])
def test_native_builders_do_not_write_build_output_inside_sources(tmp_path, monkeypatch, script):
    import runpy
    import subprocess
    import sys

    main = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / script))["main"]
    native_name = "render_preparation" if "preparation" in script else "render_replica"
    source = tmp_path / "native" / native_name
    source.mkdir(parents=True)
    (source / "a.cpp").write_bytes(b"fixture source")
    main.__globals__["__file__"] = str(tmp_path / "scripts" / script)
    main.__globals__["SOURCE_ROOT"] = source
    output = source / "build"
    monkeypatch.setattr(sys, "argv", [script, "--output", str(output)])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("compiler invoked"))
    with pytest.raises(ValueError, match="BUILD_OUTPUT_INSIDE_SOURCE"):
        main()
    assert not output.exists()
