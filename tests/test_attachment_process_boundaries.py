"""Exercise real owned child lifecycles using fixtures, never user media or remote APIs."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter

from dronedream_agent_plugins import _attachment_pdf_worker, attachment_decoders
from dronedream_agent_plugins._attachment_process import preview_process


# 功能：
#   验证真实小型 PDF 经独立工作者及产品使用的管道链路后返回页数和元数据。
# 输入：
#   tmp_path：生成无用户内容测试 PDF 的隔离目录。
# 输出：
#   None：不返回业务数据。
def test_pdf_worker_extracts_real_file_outside_parent(tmp_path):
    source = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Title": "preview fixture"})
    writer.write(source)
    text, metadata = attachment_decoders._decode_pdf(source)
    assert text == ""
    assert metadata["page_count"] == metadata["extracted_pages"] == 1
    assert metadata["metadata"]["/Title"] == "preview fixture"


# 功能：
#   验证大量标准输出在捕获过程中触发上限，不等完整缓冲后才拒绝。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_owned_preview_rejects_excess_stdout():
    with pytest.raises(ValueError, match="OUTPUT_LIMIT"):
        preview_process(
            [sys.executable, "-c", "import sys; sys.stdout.write('x'*100000)"], output_limit=1000
        )


# 功能：
#   验证不会交给调用方的错误输出仍消耗合计预算，不能无限占用内存。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_owned_preview_also_bounds_discarded_diagnostics():
    with pytest.raises(ValueError, match="OUTPUT_LIMIT"):
        preview_process(
            [sys.executable, "-c", "import sys; sys.stderr.write('x'*100000)"],
            output_limit=1000,
        )


# 功能：
#   验证请求类型和字节预算在进程、Job 或管道创建前完成检查。
# 输入：
#   monkeypatch：拦截共享启动器的测试工具。
#   request_value：非法类型或超过预算的请求。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("request_value", ["not bytes", bytearray(b"data"), b"x" * 4097])
def test_owned_preview_validates_request_before_launch(monkeypatch, request_value):
    import dronedream_agent_plugins._attachment_process as module

    # 功能：
    #   使非法请求意外到达进程创建入口时立即失败。
    # 输入：
    #   args：启动器位置参数。
    #   kwargs：启动器通信与资源配置。
    # 输出：
    #   None：不返回业务数据。
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected launch")

    monkeypatch.setattr(module, "capture_process", forbidden)
    with pytest.raises(ValueError, match="REQUEST_LIMIT"):
        preview_process([sys.executable, "-c", "pass"], request=request_value)


# 功能：
#   验证超时解码器经过终止和回收流程，再向调用方报告超时。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_owned_preview_rejects_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        preview_process([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.15)


# 功能：
#   验证解码器非零退出不能变成成功的空预览。
# 输入：
#   无输入参数。
# 输出：
#   None：不返回业务数据。
def test_owned_preview_handles_nonzero_exit():
    with pytest.raises(RuntimeError, match="PROCESS_FAILED"):
        preview_process([sys.executable, "-c", "raise SystemExit(3)"])


# 功能：
#   验证打包入口识别 PDF 工作者参数后只调用工作者，不进入 HTTP 服务初始化。
# 输入：
#   monkeypatch：替换参数与工作者入口的测试工具。
# 输出：
#   None：不返回业务数据。
def test_packaged_dispatch_does_not_start_server(monkeypatch):
    called = []
    monkeypatch.setattr(_attachment_pdf_worker, "main", lambda: called.append("worker"))
    monkeypatch.setattr(sys, "argv", ["bundled-core.exe", "--attachment-pdf-worker"])
    runpy.run_path(str(Path(__file__).parents[1] / "app/backend_entry.py"), run_name="__main__")
    assert called == ["worker"]


# 功能：
#   验证媒体探测仅允许本地文件协议和指定容器格式，不启用网络或播放列表解复用。
# 输入：
#   monkeypatch：记录探测命令的测试工具。
#   tmp_path：无需实际读取的测试媒体路径所属目录。
# 输出：
#   None：不返回业务数据。
def test_media_probe_disallows_network_and_playlist_inputs(monkeypatch, tmp_path):
    commands = []

    # 功能：
    #   记录参数并返回空媒体流结果，不执行程序或读取媒体文件。
    # 输入：
    #   command：计划调用的媒体探测命令。
    # 输出：
    #   response：包含空流列表的 JSON 字节。
    def capture(command):
        commands.append(command)
        response = b'{"streams":[]}'
        return response

    monkeypatch.setattr(attachment_decoders, "preview_process", capture)
    assert attachment_decoders._probe_media("ffprobe", tmp_path / "x.mp4", video=True) == {
        "streams": []
    }
    command = commands[0]
    assert command[command.index("-protocol_whitelist") + 1] == "file"
    assert "hls" not in command[command.index("-format_whitelist") + 1]


# 功能：
#   验证非对象、非有限数和重复字段的元数据不会传递给后续模型。
# 输入：
#   monkeypatch：替换探测结果的测试工具。
#   tmp_path：隔离媒体路径所属目录。
#   response：有缺陷的 JSON 响应字节。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("response", [b"[]", b'{"duration":NaN}', b'{"a":1,"a":2}'])
def test_media_probe_rejects_invalid_json(monkeypatch, tmp_path, response):
    monkeypatch.setattr(attachment_decoders, "preview_process", lambda *_: response)
    with pytest.raises(ValueError):
        attachment_decoders._probe_media("ffprobe", tmp_path / "x.mp4", video=True)
