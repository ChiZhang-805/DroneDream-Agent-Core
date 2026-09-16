"""Office snapshot ownership and preallocation limits, using only test-owned inputs."""

import io
import zipfile
from types import SimpleNamespace

import pytest

import dronedream_agent_plugins._attachment_xml as office


# 功能：
#   创建只含工作表 XML 的内存文档，不读写用户资料。
# 输入：
#   count：生成的 ZIP 成员数量。
# 输出：
#   payload：完整 ZIP 字节。
def _payload(count):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for number in range(count):
            archive.writestr(f"xl/worksheets/sheet{number}.xml", "<worksheet/>")
    payload = buffer.getvalue()
    return payload


# 功能：
#   验证超量索引在标准 ZIP 解析器分配成员对象前被拒绝。
# 输入：
#   tmp_path：本测试独占目录。
#   monkeypatch：替换分配入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_office_rejects_member_count_before_zipfile_allocation(tmp_path, monkeypatch):
    path = tmp_path / "oversized.xlsx"
    path.write_bytes(_payload(4097))

    # 功能：
    #   让越过预算的标准解析调用立即暴露，而非真正分配成员列表。
    # 输入：
    #   source：传入的 ZIP 快照。
    # 输出：
    #   None：不返回业务数据。
    def forbidden_open(source):
        raise AssertionError("ZipFile allocated before the directory budget was checked")

    monkeypatch.setattr(office.zipfile, "ZipFile", forbidden_open)
    with pytest.raises(ValueError, match="ZIP_INDEX_LIMIT_EXCEEDED"):
        office.OfficeArchive(path)


# 功能：
#   验证 ZIP 关闭出错也会释放当前实例拥有的输入快照。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_office_close_releases_buffer_even_when_archive_close_fails():
    archive = object.__new__(office.OfficeArchive)
    archive._buffer = io.BytesIO(b"owned snapshot")

    # 功能：
    #   模拟 ZIP 关闭错误，不访问真实句柄。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def fail_close():
        raise OSError("close failed")

    archive.archive = SimpleNamespace(close=fail_close)
    with pytest.raises(OSError, match="close failed"):
        archive.close()
    assert archive._buffer.closed


# 功能：
#   验证清理错误不会覆盖已有解析错误，但会留在该错误的附注中。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_office_context_keeps_parse_failure_visible():
    archive = object.__new__(office.OfficeArchive)
    archive._buffer = io.BytesIO(b"owned snapshot")

    # 功能：
    #   注入与正文解析错误不同的关闭异常。
    # 输入：
    #   无。
    # 输出：
    #   None：不返回业务数据。
    def fail_close():
        raise OSError("close failed")

    archive.archive = SimpleNamespace(close=fail_close)
    with pytest.raises(ValueError, match="parse failed") as caught, archive:
        raise ValueError("parse failed")
    assert archive._buffer.closed
    assert any("OFFICE_CLOSE_FAILED" in note for note in caught.value.__notes__)


# 功能：
#   验证布尔、小数、负数和超大上限不能借助 Python 切片语义蒙混过关。
# 输入：
#   tmp_path：本测试独占目录。
#   limit：待拒绝的部件数量上限。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("limit", [True, -1, 1.5, "1", 4097])
def test_office_part_limit_requires_bounded_integer(tmp_path, limit):
    path = tmp_path / "minimal.xlsx"
    path.write_bytes(_payload(2))
    with (
        office.OfficeArchive(path) as archive,
        pytest.raises(ValueError, match="OFFICE_PART_LIMIT_INVALID"),
    ):
        archive.ordered_parts(
            index="xl/workbook.xml", child_tag="sheet", prefix="xl/worksheets/sheet", limit=limit
        )


# 功能：
#   验证零预览预算仍返回真实总数，且快照不受原文件替换影响。
# 输入：
#   tmp_path：本测试独占目录。
# 输出：
#   None：不返回业务数据。
def test_office_zero_budget_and_snapshot_identity(tmp_path):
    path = tmp_path / "minimal.xlsx"
    path.write_bytes(_payload(2))
    with office.OfficeArchive(path) as archive:
        path.write_bytes(b"replacement is not a ZIP")
        result = archive.ordered_parts(
            index="xl/workbook.xml", child_tag="sheet", prefix="xl/worksheets/sheet", limit=0
        )
        assert result == ([], 2)
        assert archive.xml("xl/worksheets/sheet0.xml").tag == "worksheet"
    assert archive._buffer.closed
