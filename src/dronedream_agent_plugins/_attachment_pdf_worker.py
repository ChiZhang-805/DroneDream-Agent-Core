"""Single-document PDF text worker; spawned by the host, never a flight-control loop."""

from __future__ import annotations

import io
import os
import sys
from itertools import islice
from pathlib import Path

from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_plugin_sdk.protocol import decode_json, encode_json

MAX_TEXT_CHARACTERS = 200_000


# 功能：
#   在受限工作者中提取前 500 页内的有限文本与元数据，并准确记录是否截断。
#   页间分隔符也占字符预算；此处不做 OCR，也不保证保留原版面的阅读顺序。
# 输入：
#   source：本地普通 PDF 文件路径，原始内容上限为 16 MiB。
# 输出：
#   result：提取文本以及页数、截断、加密与文档元数据。
def extract(source: Path) -> dict[str, object]:
    from pypdf import PdfReader

    # Resource limits protect all supported pypdf versions without changing
    # process-global decompression settings shared by unrelated readers.
    reader = PdfReader(io.BytesIO(read_plugin_file(source, limit=16 * 1024 * 1024)))
    texts: list[str] = []
    characters = 0
    text_truncated = False
    for page in reader.pages[:500]:
        separator = 2 if texts else 0
        remaining = MAX_TEXT_CHARACTERS - characters - separator
        if remaining < 0:
            text_truncated = True
            break
        original = page.extract_text() or ""
        text = original[:remaining]
        text_truncated = text_truncated or len(original) > remaining
        texts.append(text)
        characters += separator + len(text)
        if characters >= MAX_TEXT_CHARACTERS:
            break
    result = {
        "text": "\n\n".join(texts),
        "metadata": {
            "page_count": len(reader.pages),
            "extracted_pages": len(texts),
            "truncated": len(texts) < len(reader.pages) or text_truncated,
            "encrypted": bool(reader.is_encrypted),
            "metadata": {
                str(key): str(value)[:500]
                for key, value in islice((reader.metadata or {}).items(), 128)
            },
        },
    }
    return result


# 功能：
#   在解析前设置 POSIX 资源限制，读取有界单文档请求并向标准输出写入提取结果。
#   Windows 的资源 Job 由宿主在请求送入管道前分配，工作者不会自行启动服务。
# 输入：
#   无函数参数；标准输入接收仅含 path 字段的 JSON 请求。
# 输出：
#   None：不返回业务数据。
def main() -> None:
    if os.name != "nt":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024, 384 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    request = decode_json(sys.stdin.buffer.read(4097), limit=4096)
    if not isinstance(request, dict) or set(request) != {"path"}:
        raise ValueError("ATTACHMENT_PDF_REQUEST_INVALID")
    path = request["path"]
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("ATTACHMENT_PDF_PATH_INVALID")
    sys.stdout.buffer.write(encode_json(extract(Path(path))).encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
