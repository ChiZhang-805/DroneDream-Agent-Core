from pathlib import Path

import pytest

from dronedream_agent_core.runtime_receipt_reader import RuntimeReceiptReader


# 功能：
#   验证目录晚创建、回执新增和原子替换都立即可见，句柄复用不缓存旧内容。
# 输入：
#   tmp_path：隔离运行目录。
# 输出：
#   None：通过断言验证实时读取行为。
def test_receipt_reader_observes_new_and_replaced_bytes(tmp_path: Path):
    directory = tmp_path / "receipts"
    reader = RuntimeReceiptReader(directory)
    try:
        assert reader.read() == []
        directory.mkdir()
        first = directory / "001.receipt.json"
        first.write_text('{"state":"detached"}', encoding="utf-8")
        assert reader.read() == [(first, {"state": "detached"})]
        temporary = directory / "new.tmp"
        temporary.write_text('{"state":"attached"}', encoding="utf-8")
        temporary.replace(first)
        assert reader.read() == [(first, {"state": "attached"})]
        second = directory / "002.receipt.json"
        second.write_text('{}', encoding="utf-8")
        assert [path for path, _ in reader.read()] == [first, second]
        first.unlink()
        with pytest.raises(ValueError, match="RUNTIME_RECEIPT_REMOVED"):
            reader.read()
    finally:
        reader.close()
    reader.close()
    with pytest.raises(ValueError, match="CLOSED"):
        reader.read()


# 功能：
#   目录被移走或替换后拒绝读取，不能悄悄回退到无载荷状态。
# 输入：
#   tmp_path：隔离运行目录。
#   replace：是否在原路径新建另一个目录。
# 输出：
#   None：目录身份约束通过断言。
@pytest.mark.parametrize("replace", [False, True])
def test_receipt_reader_rejects_directory_change(tmp_path: Path, replace: bool):
    directory = tmp_path / "receipts"
    directory.mkdir()
    reader = RuntimeReceiptReader(directory)
    try:
        assert reader.read() == []
        directory.rename(tmp_path / "preserved")
        if replace:
            directory.mkdir()
        with pytest.raises(ValueError, match="DIRECTORY_(CHANGED|REMOVED)"):
            reader.read()
    finally:
        reader.close()


# 功能：
#   每次都检查新字节，已读取的有效回执随后损坏也不能沿用旧值。
# 输入：
#   tmp_path：隔离运行目录。
# 输出：
#   None：损坏回执被拒绝。
def test_receipt_reader_revalidates_corruption(tmp_path: Path):
    path = tmp_path / "001.receipt.json"
    path.write_text('{}', encoding="utf-8")
    reader = RuntimeReceiptReader(tmp_path)
    try:
        assert reader.read() == [(path, {})]
        path.write_text('{broken', encoding="utf-8")
        with pytest.raises(ValueError):
            reader.read()
    finally:
        reader.close()


# 功能：
#   限制单次目录扫描的回执数量，防止异常目录拖垮实时控制循环。
# 输入：
#   tmp_path：独立目录，测试内创建超过预算的空回执。
# 输出：
#   None：超限读取在打开回执前被拒绝。
def test_receipt_reader_bounds_receipt_count(tmp_path: Path):
    for index in range(257):
        (tmp_path / f"{index:04}.receipt.json").write_text('{}', encoding="utf-8")
    reader = RuntimeReceiptReader(tmp_path)
    try:
        with pytest.raises(ValueError, match="COUNT_LIMIT"):
            reader.read()
    finally:
        reader.close()
