"""Preparation boundaries, independent from cloud services and flight runtimes."""
import importlib.util
from pathlib import Path
import stat
import zipfile
import pytest


# 功能：装载工具函数而不执行下载/训练入口；输入：脚本名；输出：模块。
def tool(name):
    path = Path(__file__).parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["../escape.png", "/absolute.png", "C:/drive.png", "x\\escape.png", "script.py", "file.exe", "file.txt:stream"])
def test_archive_paths_rejected(tmp_path, name):
    """功能：拒绝路径穿越和执行代码；输入：不安全成员；输出：异常。"""
    with pytest.raises(ValueError):
        tool("prepare_tartanair_sample").member_target(zipfile.ZipInfo(name), tmp_path)


def test_archive_links_and_large_members_rejected(tmp_path):
    """功能：阻止软链接和大成员绕过；输入：伪造成员；输出：异常。"""
    module = tool("prepare_tartanair_sample")
    member = zipfile.ZipInfo("frame.png")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError):
        module.member_target(member, tmp_path)
    member.external_attr = 0
    member.file_size = 257 * 1024**2
    with pytest.raises(ValueError):
        module.member_target(member, tmp_path)


def test_native_depth_bytes_decode_without_channel_reorder():
    """功能：检验四通道深度字节次序；输入：已知浮点深度；输出：原值。"""
    import numpy as np
    values = np.array([[1., 2.5], [10., .3]], dtype="<f4")
    result = tool("audit_tartanair_sample").decode_depth(values.view(np.uint8).reshape(2, 2, 4))
    np.testing.assert_array_equal(result, values)


def test_manifest_paths_portable_without_escape():
    """功能：历史 Windows 路径可迁移但不能越界；输入：跨平台路径；输出：统一路径或拒绝。"""
    parser = tool("audit_tartanair_sample").portable_path
    assert parser("P000\\camera\\image.png").as_posix() == "P000/camera/image.png"
    for path in ("../x.png", "C:/x.png", "/x.png", "\\\\server\\file.png"):
        with pytest.raises(ValueError):
            parser(path)


def test_quality_report_cannot_hide_modified_actual_data(tmp_path):
    """功能：旧回执不能掩盖数据篡改；输入：合法数据后修改/重复；输出：拒绝。"""
    audit = tool("audit_tartanair_sample")
    path = tmp_path / "frame.txt"
    path.write_bytes(b"original")
    row = {"path": "frame.txt", "bytes": 8, "sha256": audit.digest(path)}
    assert audit.verify_files(tmp_path, {"files": [row]}) == {"frame.txt"}
    with pytest.raises(ValueError, match="INTEGRITY"):
        audit.verify_files(tmp_path, {"files": [row, row]})
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="INTEGRITY"):
        audit.verify_files(tmp_path, {"files": [row]})


def test_synthetic_smoke_cannot_enter_formal_training():
    """功能：明确合成接口数据不成为飞行示范；输入：烟测语料；输出：正式模式拒绝。"""
    rows = tool("evaluate_stage_decisions").contract_cases()
    with pytest.raises(ValueError, match="LABEL_EVIDENCE"):
        tool("train_stage_classifier").validate_corpus(rows)
    assert tool("train_stage_classifier").validate_corpus(rows, smoke=True) == rows


def test_task_group_split_leakage_rejected():
    """功能：同任务不同帧不能跨集合；输入：被篡改划分；输出：拒绝。"""
    rows = tool("evaluate_stage_decisions").contract_cases()
    rows[1]["split"] = "test" if rows[0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="GROUP_LEAKAGE"):
        tool("train_stage_classifier").validate_corpus(rows, smoke=True)


def test_oversize_log_line_does_not_corrupt_next_record(tmp_path):
    """功能：大行有界跳过且保留下一条；输入：超限行加正常行；输出：一个失败一个成功。"""
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"x" * (5 * 1024**2) + b'\n{"ok": true}\n')
    result = list(tool("evaluate_stage_decisions").rows(path))
    assert result == [(1, None, "row-too-large"), (2, {"ok": True}, None)]
