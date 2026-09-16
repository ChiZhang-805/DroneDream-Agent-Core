"""Local file budget and mutation probes; no plugin code is executed."""

import hashlib
from pathlib import Path

import pytest

from dronedream_agent_core.plugin_files import hash_plugin_file, read_plugin_file


# 功能：
#   验证字节预算必须是非负整数，不能退化为无界读取或类型强制转换。
# 输入：
#   tmp_path：隔离测试目录。
#   reader：实际读取或摘要函数。
#   limit：待拒绝的预算值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reader", [read_plugin_file, hash_plugin_file])
@pytest.mark.parametrize("limit", [-2, -1, True, 1.5, "4", None])
def test_invalid_file_budget_is_rejected(tmp_path, reader, limit):
    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="LIMIT_INVALID"):
        reader(path, limit=limit)


class _ReadProbe:
    # 功能：
    #   包装真实只读流，记录请求量并可在首次读取前模拟文件增长。
    # 输入：
    #   self：读取探针。
    #   source：本测试已打开的真实文件流。
    #   reads：收集读取请求大小的列表。
    #   grow：可选的受控增长动作。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, source, reads, grow):
        self.source = source
        self.reads = reads
        self.grow = grow

    # 功能：
    #   返回探针，由上下文退出负责关闭其持有的真实流。
    # 输入：
    #   self：读取探针。
    # 输出：
    #   self：当前探针。
    def __enter__(self):
        return self

    # 功能：
    #   关闭本测试流，不吞掉调用方异常。
    # 输入：
    #   self：读取探针。
    #   error_type：异常类型或空值。
    #   error：异常对象或空值。
    #   traceback：异常追踪信息或空值。
    # 输出：
    #   None：不返回业务数据。
    def __exit__(self, error_type, error, traceback):
        self.source.close()

    # 功能：
    #   提供真实描述符，让被测函数执行真实 fstat 检查。
    # 输入：
    #   self：读取探针。
    # 输出：
    #   descriptor：真实文件描述符。
    def fileno(self):
        descriptor = self.source.fileno()
        return descriptor

    # 功能：
    #   记录实际请求量，按用例触发增长后读取真实文件内容。
    # 输入：
    #   self：读取探针。
    #   size：被测函数请求的字节数。
    # 输出：
    #   payload：真实文件流返回的字节。
    def read(self, size):
        if self.grow is not None:
            grow = self.grow
            self.grow = None
            grow()
        self.reads.append(size)
        payload = self.source.read(size)
        return payload


# 功能：
#   验证小文件即使有大预算，也只按已知大小读取；读取中增长应立即拒绝。
# 输入：
#   tmp_path：隔离测试目录。
#   monkeypatch：安装文件读取探针的测试工具。
#   reader：实际读取或摘要函数。
#   growing：是否在 fstat 之后追加测试内容。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("reader", [read_plugin_file, hash_plugin_file])
@pytest.mark.parametrize("growing", [False, True])
def test_read_request_tracks_file_size_not_large_budget(tmp_path, monkeypatch, reader, growing):
    path = tmp_path / "input.bin"
    path.write_bytes(b"abc")
    original_open = Path.open
    reads = []

    # 功能：
    #   用原始文件打开接口追加内容，模拟文件在身份检查之后增长。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def grow():
        with original_open(path, "ab") as target:
            target.write(b"x" * 8192)

    # 功能：
    #   只拦截当前测试输入的二进制读取，不改变其他路径的打开行为。
    # 输入：
    #   candidate：待打开路径。
    #   args：原文件打开的位置参数。
    #   kwargs：原文件打开的关键字参数。
    # 输出：
    #   opened：受控读取探针或原始文件流。
    def probe_open(candidate, *args, **kwargs):
        opened = original_open(candidate, *args, **kwargs)
        if candidate == path and args == ("rb",):
            opened = _ReadProbe(opened, reads, grow if growing else None)
        return opened

    monkeypatch.setattr(Path, "open", probe_open)
    if growing:
        with pytest.raises(ValueError):
            reader(path, limit=64 * 1024 * 1024)
    else:
        result = reader(path, limit=64 * 1024 * 1024)
        expected = b"abc" if reader is read_plugin_file else hashlib.sha256(b"abc").hexdigest()
        assert result == expected
    assert reads and max(reads) <= 4
