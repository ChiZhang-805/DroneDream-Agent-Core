"""Host-owned temporary fixtures for package parsing, provenance and publication boundaries."""

import hashlib
import io
import json
import struct
import zipfile
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from test_asset_packages import _write_package

import dronedream_agent_core.asset_packages as packages
import dronedream_agent_core.asset_source_adapters as adapters
from dronedream_agent_core.xml_values import parse_xml
from dronedream_agent_core.zip_index import validate_zip_index
from dronedream_plugin_sdk.protocol import decode_json, encode_json


# 功能：
#   构造少量真实 ZIP 成员供目录预检使用，不生成压缩炸弹或大量测试数据。
# 输入：
#   count：成员数量。
#   comment：归档尾部注释字节。
# 输出：
#   archive_bytes：完整的测试归档字节。
def _zip_bytes(*, count=2, comment=b""):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as bundle:
        for index in range(count):
            bundle.writestr(f"file-{index}", b"test payload" * 8)
        bundle.comment = comment
    archive_bytes = target.getvalue()
    return archive_bytes


# 功能：
#   验证普通 ZIP 和 ZIP64 均通过目录预检；降低测试阈值触发 ZIP64，不分配巨型文件。
# 输入：
#   monkeypatch：自动恢复 ZIP64 阈值的测试替换器。
#   force_zip64：是否要求生成 ZIP64 记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("force_zip64", [False, True])
def test_index_preflight_accepts_plain_and_zip64(monkeypatch, force_zip64):
    if force_zip64:
        monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 50)
    raw = _zip_bytes(comment=b"a normal archive comment")
    if force_zip64:
        assert b"PK\x06\x06" in raw
    assert validate_zip_index(io.BytesIO(raw), maximum_members=2) == 2


# 功能：
#   验证目录尾记录少报成员数时仍会被实际目录头计数发现，不能借此绕过分配预算。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_index_counts_actual_headers_not_only_declared_total():
    raw = bytearray(_zip_bytes())
    end = raw.rfind(b"PK\x05\x06")
    struct.pack_into("<HH", raw, end + 8, 1, 1)
    with pytest.raises(ValueError, match="COUNT_MISMATCH"):
        validate_zip_index(io.BytesIO(raw), maximum_members=2)


# 功能：
#   验证归档尾部追加普通数据或伪结束标记都会被拒绝，避免不同解析器识别不同边界。
# 输入：
#   suffix：追加在合法归档之后的字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("suffix", [b"unexpected payload", b"PK\x05\x06"])
def test_index_rejects_appended_data(suffix):
    with pytest.raises(ValueError):
        validate_zip_index(io.BytesIO(_zip_bytes() + suffix), maximum_members=2)


# 功能：
#   验证伪造的超大目录在 ZipFile 构建完整成员表之前被拒绝。
# 输入：
#   tmp_path：测试独占目录。
#   monkeypatch：拦截归档构造函数的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_oversized_index_is_rejected_before_zipfile_allocates(tmp_path, monkeypatch):
    raw = bytearray(_zip_bytes())
    struct.pack_into("<I", raw, raw.rfind(b"PK\x05\x06") + 12, 33 * 1024 * 1024)
    path = tmp_path / "forged.ddpkg"
    path.write_bytes(raw)

    # 功能：
    #   若无界构造函数被调用则立即使测试失败，证明拒绝发生在分配之前。
    # 输入：
    #   _：被拦截构造函数的位置参数。
    #   __：被拦截构造函数的关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*_, **__):
        raise AssertionError("unbounded ZipFile constructor")

    monkeypatch.setattr(packages.zipfile, "ZipFile", forbidden)
    with pytest.raises(packages.AssetPackageError, match="INDEX_INVALID"):
        packages.inspect_ddpkg(path)


# 功能：
#   验证保留名称、流名称、尾随空白及歧义分隔符不能进入跨平台包成员路径。
# 输入：
#   name：待拒绝的成员名称。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    "name", ["CON", "a/NUL.txt", "a/file:stream", "a/last.", "a/last ", ".", "a//b", "a\u0001b"]
)
def test_member_paths_have_one_portable_meaning(name):
    with pytest.raises(packages.AssetPackageError):
        packages.normalized_member_path(name)


# 功能：
#   验证目录也参与大小写碰撞和文件／父目录冲突检查，而非仅检查普通文件。
# 输入：
#   names：具有冲突关系的成员名称列表。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("names", [["root", "root/child"], ["root/", "ROOT/"]])
def test_member_directories_cannot_alias_files_or_each_other(names):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as bundle:
        for name in names:
            bundle.writestr(name, b"" if name.endswith("/") else b"data")
    with (
        zipfile.ZipFile(io.BytesIO(target.getvalue())) as bundle,
        pytest.raises(packages.AssetPackageError),
    ):
        packages.inspect_zip_members(bundle)


# 功能：
#   修改测试包的语义索引并重算字节摘要，确保后续失败来自语义不一致而非过期摘要。
# 输入：
#   path：测试独占的资产包路径。
#   mutate：原地修改 manifest 与 ir 的测试回调。
# 输出：
#   None：不返回业务数据。
def _rewrite_package(path, mutate):
    with zipfile.ZipFile(path) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(payloads["manifest.json"])
    ir_path = manifest["asset_ir_path"]
    ir = json.loads(payloads[ir_path])
    mutate(manifest, ir)
    payloads[ir_path] = json.dumps(ir).encode()
    for entry in manifest["files"]:
        if entry["path"] == ir_path:
            entry["size_bytes"] = len(payloads[ir_path])
            entry["sha256"] = hashlib.sha256(payloads[ir_path]).hexdigest()
    manifest["content_sha256"] = packages.package_content_sha256(
        [packages.AssetFile.model_validate(entry) for entry in manifest["files"]]
    )
    payloads["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)


# 功能：
#   验证字节摘要正确也不能掩盖内外索引中大小或权限角色不一致。
# 输入：
#   tmp_path：测试独占目录。
#   field：要篡改的 IR 文件索引字段。
#   bad：与外部清单不一致的字段值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field,bad", [("size_bytes", 999), ("role", "metadata")])
def test_ir_and_manifest_must_agree_on_size_and_authority_role(tmp_path, field, bad):
    path = tmp_path / "contradictory.ddpkg"
    _write_package(path)

    # 功能：
    #   仅改写 IR 中 SDF 条目的指定字段，让外层摘要由测试重建器重新计算。
    # 输入：
    #   _manifest：保留不变的外部清单。
    #   ir：将被原地修改的内部索引。
    # 输出：
    #   None：不返回业务数据。
    def mutate(_manifest, ir):
        next(entry for entry in ir["files"] if entry["path"].endswith(".sdf"))[field] = bad

    _rewrite_package(path, mutate)
    with pytest.raises(packages.AssetPackageError, match="IR_FILE_INDEX_MISMATCH"):
        packages.inspect_ddpkg(path)


# 功能：
#   验证内外索引同时将 SDF 伪装成元数据时，仍会检查其中未声明的可执行扩展。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_wrong_role_cannot_hide_sdf_plugins(tmp_path):
    path = tmp_path / "hidden.ddpkg"
    _write_package(path, world=b'<sdf><world name="w"><plugin name="hidden"/></world></sdf>')

    # 功能：
    #   同时改写两份索引中的 SDF 角色，构造索引一致但语义误导的测试包。
    # 输入：
    #   manifest：将被原地修改的外部清单。
    #   ir：将被原地修改的内部索引。
    # 输出：
    #   None：不返回业务数据。
    def mutate(manifest, ir):
        for index in (manifest, ir):
            next(entry for entry in index["files"] if entry["path"].endswith(".sdf"))["role"] = (
                "metadata"
            )

    _rewrite_package(path, mutate)
    with pytest.raises(packages.AssetPackageError, match="EXTENSION_UNDECLARED"):
        packages.inspect_ddpkg(path)


# 功能：
#   验证不同字符编码中的 DTD 均在实体展开前被识别并拒绝。
# 输入：
#   encoding：XML 声明与原始字节采用的编码。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_asset_xml_rejects_dtd_and_entities(encoding):
    raw = (
        f'<?xml version="1.0" encoding="{encoding}"?>'
        '<!DOCTYPE sdf [<!ENTITY x "payload">]><sdf>&x;</sdf>'
    ).encode(encoding)
    with pytest.raises(ValueError, match="DOCTYPE_FORBIDDEN"):
        parse_xml(raw, maximum_bytes=10000)


# 功能：
#   验证 JSON 编解码均拒绝无效节点预算，不能靠类型转换或超限配置放大解析范围。
# 输入：
#   budget：待拒绝的节点预算值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("budget", [None, True, 0, -1, 2_000_001])
def test_json_node_budgets_are_validated_before_encoding_or_parsing(budget):
    for action, value in ((encode_json, []), (decode_json, b"[]")):
        with pytest.raises(ValueError, match="BUDGET_INVALID"):
            action(value, node_limit=budget)


# 功能：
#   验证单次资产解析显式增加预算后，后续 RPC 默认预算不会随之放宽。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_explicit_asset_json_budget_does_not_change_rpc_default():
    value = [0] * 100_001
    wire = encode_json(value, node_limit=200_000)
    assert decode_json(wire, node_limit=200_000) == value
    with pytest.raises(ValueError, match="COMPLEXITY_LIMIT"):
        decode_json(wire)


# 功能：
#   写入测试用声明式地图源文件，不启动转换器或仿真进程。
# 输入：
#   tmp_path：测试独占目录。
#   text：源文件的 XML 文本。
# 输出：
#   source：已写入的测试源文件路径。
def _source(tmp_path, text='<sdf><world name="Fixture"/></sdf>'):
    source = tmp_path / "input.world"
    source.write_text(text, encoding="utf-8")
    return source


# 功能：
#   验证规范化输出已存在时直接拒绝，并保持原有文件内容不变。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_normalization_never_overwrites_existing_output(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "existing.ddpkg"
    target.write_bytes(b"preserve")
    with pytest.raises(adapters.AssetSourceAdapterError, match="DESTINATION_EXISTS"):
        adapters.normalize_asset_source(source, adapters.detect_asset_source(source), target)
    assert target.read_bytes() == b"preserve"


# 功能：
#   验证临时文件独占创建失败时不误删占用同名路径的既有文件。
# 输入：
#   tmp_path：测试独占目录。
#   monkeypatch：固定临时名称的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_normalization_collision_never_deletes_an_unowned_temporary(tmp_path, monkeypatch):
    source = _source(tmp_path)
    target = tmp_path / "result.ddpkg"
    collision = tmp_path / ".result.ddpkg-fixed.tmp"
    collision.write_bytes(b"preserve")
    monkeypatch.setattr(adapters, "uuid4", lambda: SimpleNamespace(hex="fixed"))
    with pytest.raises(FileExistsError):
        adapters.normalize_asset_source(source, adapters.detect_asset_source(source), target)
    assert collision.read_bytes() == b"preserve"


# 功能：
#   验证发布前出现竞争写入时，失败方仅清理自己的暂存文件而保留竞争方结果。
# 输入：
#   tmp_path：测试独占目录。
#   monkeypatch：在发布瞬间插入竞争写入的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_normalization_publish_race_preserves_the_other_writer(tmp_path, monkeypatch):
    source = _source(tmp_path)
    target = tmp_path / "result.ddpkg"
    original_link = adapters.os.link

    # 功能：
    #   先模拟另一写入方占用目标，再执行真实独占发布以触发文件已存在异常。
    # 输入：
    #   temporary：当前写入方拥有的暂存文件。
    #   destination：竞争写入方抢先创建的目标路径。
    # 输出：
    #   result：底层硬链接调用的返回值。
    def concurrent_publish(temporary, destination):
        destination.write_bytes(b"other writer")
        result = original_link(temporary, destination)
        return result

    monkeypatch.setattr(adapters.os, "link", concurrent_publish)
    with pytest.raises(FileExistsError):
        adapters.normalize_asset_source(source, adapters.detect_asset_source(source), target)
    assert target.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".result.ddpkg-*.tmp"))


# 功能：
#   验证嵌入源文件后发生内容变化会撤销规范化发布，并回收自己的暂存包。
# 输入：
#   tmp_path：测试独占目录。
#   monkeypatch：在快照写入完成后修改源文件的测试替换器。
# 输出：
#   None：不返回业务数据。
def test_source_mutation_during_normalization_cannot_publish(tmp_path, monkeypatch):
    source = _source(tmp_path)
    target = tmp_path / "result.ddpkg"
    store_snapshot = adapters._store_source_snapshot

    # 功能：
    #   完成原有快照复制后替换测试源内容，模拟复制与发布之间的源文件变化。
    # 输入：
    #   args：传递给原快照函数的位置参数。
    #   kwargs：传递给原快照函数的关键字参数。
    # 输出：
    #   result：原快照函数的返回值。
    def changing_source(*args, **kwargs):
        result = store_snapshot(*args, **kwargs)
        source.write_text('<sdf><world name="Changed"/></sdf>', encoding="utf-8")
        return result

    monkeypatch.setattr(adapters, "_store_source_snapshot", changing_source)
    with pytest.raises(adapters.AssetSourceAdapterError, match="SOURCE_CHANGED"):
        adapters.normalize_asset_source(source, adapters.detect_asset_source(source), target)
    assert not target.exists()
    assert not list(tmp_path.glob(".result.ddpkg-*.tmp"))


# 功能：
#   验证源文件改变后不能继续用过期识别结果选择解析器或发布资产包。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_stale_detection_is_not_an_authoritative_parser_selection(tmp_path):
    source = _source(tmp_path)
    detected = adapters.detect_asset_source(source)
    source.write_text('<robot name="Changed"/>', encoding="utf-8")
    with pytest.raises(adapters.AssetSourceAdapterError, match="DETECTION_STALE"):
        adapters.normalize_asset_source(source, detected, tmp_path / "result.ddpkg")
    assert not (tmp_path / "result.ddpkg").exists()


# 功能：
#   验证包含两个候选世界的归档必须消除入口歧义，不能按文件名字母顺序猜测。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_ambiguous_bundle_entrypoint_is_not_chosen_by_filename(tmp_path):
    source = tmp_path / "worlds.zip"
    with zipfile.ZipFile(source, "w") as bundle:
        bundle.writestr("first.world", '<sdf><world name="First"/></sdf>')
        bundle.writestr("second.world", '<sdf><world name="Second"/></sdf>')
    detected = adapters.detect_asset_source(source)
    with pytest.raises(adapters.AssetSourceAdapterError, match="ENTRYPOINT_AMBIGUOUS"):
        adapters.normalize_asset_source(source, detected, tmp_path / "result.ddpkg")
    assert not (tmp_path / "result.ddpkg").exists()


# 功能：
#   验证超过嵌入预算的源文件以明确的 source_reference 角色保存，摘要仍关联实际源字节。
# 输入：
#   tmp_path：测试独占目录。
#   monkeypatch：仅为测试降低嵌入容量的替换器。
# 输出：
#   None：不返回业务数据。
def test_large_source_is_explicitly_referenced_not_mislabeled(tmp_path, monkeypatch):
    source = _source(tmp_path)
    monkeypatch.setattr(adapters, "_MAX_EMBEDDED_SOURCE_BYTES", 8)
    result = adapters.normalize_asset_source(
        source, adapters.detect_asset_source(source), tmp_path / "result.ddpkg"
    )
    inspected = packages.inspect_ddpkg(result)
    reference = next(
        entry for entry in inspected.asset_ir.files if entry.role == "source_reference"
    )
    with zipfile.ZipFile(result) as archive:
        value = json.loads(archive.read(reference.path))
    assert value["embedded"] is False
    assert value["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


# 功能：
#   验证 FreeCAD 的 ZIP 容器仍识别为需要专用转换器的工程，而非未知仿真归档。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_native_archive_can_request_its_converter(tmp_path):
    source = tmp_path / "airframe.fcstd"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Document.xml", "<Document/>")
    detected = adapters.detect_asset_source(source)
    assert detected.adapter_id == "freecad.robotics"
    assert not detected.can_normalize_locally


# 功能：
#   验证将 xacro 改名为 urdf 不能绕过宏执行边界；内容识别仍要求伴随转换器。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_renamed_xacro_still_requires_a_companion(tmp_path):
    source = tmp_path / "renamed.urdf"
    source.write_text(
        '<robot name="R" xmlns:xacro="http://www.ros.org/wiki/xacro">'
        '<xacro:include filename="other"/></robot>'
    )
    detected = adapters.detect_asset_source(source)
    assert detected.adapter_id == "ros2.xacro"
    assert not detected.can_normalize_locally


# 功能：
#   验证已声明但无效的传感器频率会报错，不能被当作未提供频率而静默接受。
# 输入：
#   tmp_path：测试独占目录。
#   rate：待拒绝的频率文本。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("rate", ["NaN", "-1", "0", "Infinity", "fast", "1000001"])
def test_malformed_sensor_rate_is_not_silently_unknown(tmp_path, rate):
    source = _source(
        tmp_path,
        '<sdf><model name="R"><link name="base"><sensor name="imu" type="imu">'
        f'<update_rate>{rate}</update_rate></sensor></link></model></sdf>',
    )
    with pytest.raises(adapters.AssetSourceAdapterError, match="SENSOR_RATE_INVALID"):
        adapters.normalize_asset_source(
            source, adapters.detect_asset_source(source), tmp_path / "result.ddpkg"
        )
    assert not (tmp_path / "result.ddpkg").exists()


# 功能：
#   验证一个 link 的重复动力学字段不能冒充另一个 link 的完整惯性、质量及碰撞声明。
# 输入：
#   tmp_path：测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_extra_fields_on_one_link_cannot_complete_another_links_physics(tmp_path):
    source = _source(
        tmp_path,
        '<sdf><model name="R"><link name="one"><collision/><collision/>'
        '<inertial><mass>1</mass><inertia/></inertial>'
        '<inertial><mass>1</mass><inertia/></inertial></link>'
        '<link name="two"/></model></sdf>',
    )
    result = adapters.normalize_asset_source(
        source, adapters.detect_asset_source(source), tmp_path / "result.ddpkg"
    )
    physics = packages.inspect_ddpkg(result).asset_ir.physics
    assert physics.link_count == physics.mass_entry_count == 2
    assert not physics.collision_complete
    assert not physics.mass_complete
    assert not physics.inertia_complete


# 功能：
#   验证任务状态修订不能将更新时间回退到先前版本之前。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_import_time_cannot_move_backwards():
    now = datetime(2026, 9, 1, tzinfo=UTC)
    job = packages.AssetImportJob(
        job_id="asset-job-" + "a" * 24,
        owner_id="fixture",
        source_name="x",
        source_format="ddpkg",
        progress_percent=0,
        revision=0,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(packages.AssetPackageError, match="TIME_REGRESSION"):
        packages.transition_import_job(
            job, "quarantining", progress_percent=1, now=now - timedelta(seconds=1)
        )


# 功能：
#   验证不同中文名称得到不同的可移植标识，过长回退前缀也不能挤掉区分名称的摘要。
# 输入：
#   fallback：非 ASCII 或过长的候选回退前缀。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("fallback", ["地图", "a" * 200])
def test_unicode_names_get_distinct_portable_fallback_ids(fallback):
    first = adapters._slug("实验楼", fallback)
    second = adapters._slug("图书馆", fallback)
    assert first != second
    assert first.isascii() and second.isascii()
    assert max(len(first), len(second)) <= 160
