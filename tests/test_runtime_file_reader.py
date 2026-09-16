import os

import pytest

from dronedream_agent_core.runtime_file_reader import PinnedRuntimeObjectReader


# 功能：
#   验证固定运行文件可以读取原子替换后的新快照，关闭后不能再次使用。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
def test_pinned_reader_accepts_new_complete_snapshot_and_closes(tmp_path):
    path = tmp_path / "phase.json"
    reader = PinnedRuntimeObjectReader(path, maximum_bytes=1024)
    try:
        with pytest.raises(FileNotFoundError):
            reader.read()
        path.write_text('{"phase":"PREFLIGHT"}')
        assert reader.read() == {"phase": "PREFLIGHT"}
        replacement = tmp_path / "new.json"
        replacement.write_text('{"phase":"TRACK"}')
        replacement.replace(path)
        assert reader.read() == {"phase": "TRACK"}
    finally:
        reader.close()
    reader.close()
    with pytest.raises(ValueError, match="CLOSED"):
        reader.read()


# 功能：
#   验证快速路径仍拒绝超预算、重复键、非有限数及非对象输入。
# 输入：
#   tmp_path：测试私有目录。
#   content：不应接纳的原始字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("content", [b" " * 1025, b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'])
def test_pinned_reader_keeps_strict_json_boundaries(tmp_path, content):
    path = tmp_path / "phase.json"
    path.write_bytes(content)
    reader = PinnedRuntimeObjectReader(path, maximum_bytes=1024)
    try:
        with pytest.raises(ValueError):
            reader.read()
    finally:
        reader.close()


# 功能：
#   固定目录被同名新目录替换后必须拒绝，不悄悄读取旧句柄或新运行的内容。
# 输入：
#   tmp_path：测试私有目录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="POSIX directory handles")
def test_pinned_reader_detects_replaced_run_directory(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    path = run / "phase.json"
    path.write_text('{"phase":"TRACK"}')
    reader = PinnedRuntimeObjectReader(path, maximum_bytes=1024)
    try:
        run.rename(tmp_path / "preserved-run")
        run.mkdir()
        path.write_text('{"phase":"COMPLETE"}')
        with pytest.raises(ValueError, match="DIRECTORY_CHANGED"):
            reader.read()
    finally:
        reader.close()


# 功能：
#   文件变成链接或 FIFO 后立即拒绝；不能跟随链接或阻塞等待管道生产者。
# 输入：
#   tmp_path：测试私有目录。
#   special：要替换成的特殊文件种类。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="POSIX special files")
@pytest.mark.parametrize("special", ["link", "fifo"])
def test_pinned_reader_rejects_special_file_replacements(tmp_path, special):
    path = tmp_path / "phase.json"
    reader = PinnedRuntimeObjectReader(path, maximum_bytes=1024)
    try:
        if special == "fifo":
            os.mkfifo(path)
        else:
            target = tmp_path / "target.json"
            target.write_text('{"phase":"TRACK"}')
            path.symlink_to(target)
        with pytest.raises(ValueError, match="FILE_INVALID"):
            reader.read()
    finally:
        reader.close()


# 功能：
#   核对读取后目录项时注入原子替换，验证不能接纳已经被另一版本取代的内容。
# 输入：
#   tmp_path：测试私有目录。
#   monkeypatch：仅替换本测试中的 stat 调用。
# 输出：
#   None：不返回业务数据。
@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="POSIX directory-relative stat")
def test_pinned_reader_detects_replacement_during_read(tmp_path, monkeypatch):
    path = tmp_path / "phase.json"
    path.write_text('{"phase":"TRACK"}')
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"phase":"TRACK"}')
    reader = PinnedRuntimeObjectReader(path, maximum_bytes=1024)
    original_stat = os.stat
    checks = []

    # 功能：
    #   在最后一次按目录句柄查文件身份时替换文件，模拟正常发布器的竞争。
    # 输入：
    #   name、args、kwargs：原始 stat 参数。
    # 输出：
    #   metadata：替换完成后的实际文件元数据。
    def racing_stat(name, *args, **kwargs):
        if kwargs.get("dir_fd") == reader._directory:
            checks.append(name)
            if len(checks) == 2:
                replacement.replace(path)
        metadata = original_stat(name, *args, **kwargs)
        return metadata

    monkeypatch.setattr(os, "stat", racing_stat)
    try:
        with pytest.raises(ValueError, match="FILE_CHANGED"):
            reader.read()
    finally:
        reader.close()
