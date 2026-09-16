"""打包 SDK 的读取与暂存身份测试，仅操作本次测试拥有的文件。"""

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

import dronedream_agent_core.plugin_sdk as sdk
from dronedream_agent_core.plugin_sdk import _read_source_file, build_plugin_bundle, scaffold_plugin


# 功能：
#   验证小源文件只申请实际大小加探测字节，不按二百五十六 MiB 的预算预分配读取。
# 输入：
#   tmp_path：源文件所在的独立测试目录。
#   monkeypatch：记录并限制当前源文件的实际读取请求。
# 输出：
#   None：不返回业务数据。
def test_sdk_small_source_read_does_not_allocate_its_whole_budget(tmp_path, monkeypatch):
    source = tmp_path / "tiny.txt"
    source.write_bytes(b"x")
    original_open = Path.open

    class BoundedReader:
        # 功能：
        #   包装真实文件流，但不改变描述符或文件身份。
        # 输入：
        #   stream：本测试源文件的真实读取流。
        # 输出：
        #   None：不返回业务数据。
        def __init__(self, stream):
            self.stream = stream

        # 功能：
        #   将调用方限制为实际大小加一个探测字节，防止测试自身分配过大缓冲。
        # 输入：
        #   size：被测读取器申请的字节数。
        # 输出：
        #   payload：底层流读取的字节串。
        def read(self, size):
            assert 0 < size <= 2
            payload = self.stream.read(size)
            return payload

        # 功能：
        #   提供原始描述符，让正式身份检查不被夹具绕过。
        # 输入：
        #   无。
        # 输出：
        #   descriptor：底层流的文件描述符。
        def fileno(self):
            descriptor = self.stream.fileno()
            return descriptor

    # 功能：
    #   只包装当前测试源文件，其他文件仍按原入口读取。
    # 输入：
    #   path：待打开路径。
    #   args：原打开操作的位置参数。
    #   kwargs：原打开操作的关键字参数。
    # 输出：
    #   stream：原始流或当前源文件的受控读取包装器。
    @contextmanager
    def open_owned_source(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as original:
            stream = BoundedReader(original) if path == source else original
            yield stream

    monkeypatch.setattr(Path, "open", open_owned_source)
    assert _read_source_file(source, limit=256 * 1024 * 1024) == b"x"


# 功能：
#   验证打包源码读取拒绝非法预算，而不把负数、布尔值或浮点数交给文件流。
# 输入：
#   tmp_path：源文件所在目录。
#   limit：非法读取预算。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("limit", [-2, -1, True, 1.5, None])
def test_sdk_source_reader_rejects_invalid_budget(tmp_path, limit):
    source = tmp_path / "data"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="LIMIT_INVALID"):
        _read_source_file(source, limit=limit)


# 功能：
#   验证 ZIP 写完后暂存路径被替换时，不发布替换内容、不伪报旧摘要，也不删除替换方文件。
# 输入：
#   tmp_path：源码、暂存 ZIP 与输出所在的独立测试目录。
#   monkeypatch：在暂存流关闭后制造身份替换。
# 输出：
#   None：不返回业务数据。
def test_sdk_replaced_staging_is_neither_published_nor_deleted(tmp_path, monkeypatch):
    root = scaffold_plugin(
        tmp_path / "source",
        plugin_id="fixture.panel",
        name="Fixture",
        publisher="Fixture",
        kind="ui",
    )
    original_mkstemp = sdk.tempfile.mkstemp
    original_fdopen = sdk.os.fdopen
    captured = {}

    # 功能：
    #   记录 SDK 自己创建的暂存描述符与路径，不接管其他临时文件。
    # 输入：
    #   kwargs：SDK 指定的目录、前缀及后缀参数。
    # 输出：
    #   temporary：原工厂返回的描述符与路径元组。
    def record_staging(**kwargs):
        temporary = original_mkstemp(**kwargs)
        captured.update(descriptor=temporary[0], path=Path(temporary[1]))
        return temporary

    # 功能：
    #   在原 ZIP 流关闭后保留原文件并替换其路径，模拟发布前另一写入者占用暂存名称。
    # 输入：
    #   descriptor：SDK 已独占打开的暂存描述符。
    #   args：原 fdopen 的位置参数。
    #   kwargs：原 fdopen 的关键字参数。
    # 输出：
    #   stream：原始文件流。
    @contextmanager
    def replace_after_close(descriptor, *args, **kwargs):
        with original_fdopen(descriptor, *args, **kwargs) as stream:
            yield stream
        if descriptor == captured.get("descriptor"):
            path = captured["path"]
            path.rename(tmp_path / "retained-original.zip")
            path.write_bytes(b"replacement owned by another writer")

    monkeypatch.setattr(sdk.tempfile, "mkstemp", record_staging)
    monkeypatch.setattr(sdk.os, "fdopen", replace_after_close)
    output = tmp_path / "output.zip"
    with pytest.raises(ValueError, match="STAGING_CHANGED"):
        build_plugin_bundle(root, output)
    assert not output.exists()
    assert captured["path"].read_bytes() == b"replacement owned by another writer"
    assert (tmp_path / "retained-original.zip").is_file()


# 功能：
#   验证暂存描述符转换为 Python 流失败时，仍关闭本次描述符并回收属于本次的暂存文件。
# 输入：
#   tmp_path：隔离的源码和输出目录。
#   monkeypatch：在 fdopen 入口注入失败，不操作其他进程或用户文件。
# 输出：
#   None：不返回业务数据。
def test_sdk_fdopen_failure_releases_owned_descriptor_and_staging(tmp_path, monkeypatch):
    root = scaffold_plugin(
        tmp_path / "source",
        plugin_id="fixture.panel",
        name="Fixture",
        publisher="Fixture",
        kind="ui",
    )
    original_mkstemp = sdk.tempfile.mkstemp
    captured = {}

    # 功能：
    #   保存测试实际创建的描述符及身份，失败夹具也只能关闭这一份资源。
    # 输入：
    #   kwargs：临时文件创建参数。
    # 输出：
    #   temporary：新描述符和路径的元组。
    def record_staging(**kwargs):
        temporary = original_mkstemp(**kwargs)
        captured.update(
            descriptor=temporary[0], path=Path(temporary[1]), identity=os.fstat(temporary[0])
        )
        return temporary

    # 功能：
    #   模拟独占文件创建成功后流包装失败，用于检查交接窗口的资源回收。
    # 输入：
    #   descriptor：本次暂存文件描述符。
    #   args：流包装的位置参数。
    #   kwargs：流包装的关键字参数。
    # 输出：
    #   None：不返回业务数据。
    def reject_stream(descriptor, *args, **kwargs):
        raise OSError("fixture fdopen failed")

    monkeypatch.setattr(sdk.tempfile, "mkstemp", record_staging)
    monkeypatch.setattr(sdk.os, "fdopen", reject_stream)
    try:
        with pytest.raises(OSError, match="fixture fdopen failed"):
            build_plugin_bundle(root, tmp_path / "output.zip")
        with pytest.raises(OSError):
            os.fstat(captured["descriptor"])
        assert not captured["path"].exists()
    finally:
        # 旧代码复现泄漏时由测试回收它自己创建的描述符，不能把失败夹具变成真实泄漏。
        try:
            current = os.fstat(captured["descriptor"])
        except OSError:
            pass
        else:
            if os.path.samestat(current, captured["identity"]):
                os.close(captured["descriptor"])
