"""Verify a transferred assembled dataset against its separately saved receipt hash."""

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

from verify_training_bundle import file_digest


# 功能：
#   规范跨平台数据索引路径，拒绝越界、大小写别名和保留的元数据文件名。
# 输入：
#   value、seen：索引路径及已经出现的大小写归一化名称。
# 输出：
#   relative：可安全附加到数据根的相对路径。
def checked_path(value, seen):
    if (type(value) is not str or not value or len(value) > 512 or "\\" in value
            or ":" in value or value.startswith("/")
            or any(part in ("", ".", "..") or part.endswith((".", " "))
                   for part in value.split("/"))
            or PurePosixPath(value).as_posix() != value or value.casefold() in seen):
        raise ValueError("VISION_TRANSFER_PATH_INVALID")
    seen.add(value.casefold())
    relative = value
    return relative


# 功能：
#   1. 逐字节验证组装索引、原图、标签、逐图来源、筛选记录和三个训练清单。
#   2. 拒绝漏传、旧文件、链接及更换后的清单，不执行任何训练代码。
# 输入：
#   root、expected：数据根及上传前另外保存的 assembly-receipt.json 摘要。
# 输出：
#   report：实际核验文件数和字节数，不授予训练充分性或模型资格。
def verify(root, expected):
    if type(expected) is not str or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise ValueError("VISION_TRANSFER_RECEIPT_DIGEST_REQUIRED")
    receipt_path = root / "assembly-receipt.json"
    if file_digest(receipt_path, 8 * 1024**2)[0] != expected:
        raise ValueError("VISION_TRANSFER_RECEIPT_CHANGED")
    raw = receipt_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("VISION_TRANSFER_RECEIPT_CHANGED")
    receipt = json.loads(raw)
    if (type(receipt) is not dict or receipt.get("complete") is not True
            or receipt.get("schema") != "dronedream.assembled-vision-dataset.v1"
            or type(receipt.get("indexed_files")) is not int
            or not 1 <= receipt["indexed_files"] <= 600_000
            or type(receipt.get("splits")) is not dict
            or set(receipt["splits"]) != {"training", "validation", "test"}):
        raise ValueError("VISION_TRANSFER_RECEIPT_INVALID")
    controls = {"assembly-receipt.json": expected,
                "files.jsonl": receipt.get("files_index_sha256")}
    for split, summary in receipt["splits"].items():
        controls[split + "/samples.jsonl"] = summary.get("manifest_sha256")
    if "curation" in receipt:
        controls["curation.jsonl"] = receipt["curation"].get("index_sha256")
    allowed, seen, total, count = set(controls), {name.casefold() for name in controls}, 0, 0
    for name, digest in controls.items():
        if (type(digest) is not str or re.fullmatch(r"[a-f0-9]{64}", digest) is None
                or file_digest(root / name, 256 * 1024**2)[0] != digest):
            raise ValueError("VISION_TRANSFER_CONTROL_CHANGED:" + name)
    with (root / "files.jsonl").open("rb") as stream:
        while line := stream.readline(4097):
            count += 1
            if len(line) > 4096 or count > receipt["indexed_files"]:
                raise ValueError("VISION_TRANSFER_INDEX_BUDGET")
            entry = json.loads(line)
            if type(entry) is not dict or set(entry) != {"path", "sha256", "size_bytes"}:
                raise ValueError("VISION_TRANSFER_INDEX_ENTRY_INVALID")
            relative = checked_path(entry["path"], seen)
            digest, size = file_digest(root / relative, 16 * 1024**2)
            if (digest != entry["sha256"] or type(entry["size_bytes"]) is not int
                    or size != entry["size_bytes"]):
                raise ValueError("VISION_TRANSFER_FILE_CHANGED:" + relative)
            allowed.add(relative)
            total += size
            if total > 40 * 1024**3:
                raise ValueError("VISION_TRANSFER_TOTAL_BUDGET")
    if count != receipt["indexed_files"]:
        raise ValueError("VISION_TRANSFER_INDEX_COUNT_MISMATCH")
    directories = {parent.as_posix() for name in allowed for parent in PurePosixPath(name).parents
                   if parent.as_posix() != "."}
    pending, actual, visited = [root], set(), 0
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            visited += 1
            if visited > 800_000:
                raise ValueError("VISION_TRANSFER_INVENTORY_BUDGET")
            relative, info = path.relative_to(root).as_posix(), path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("VISION_TRANSFER_LINK_NOT_ALLOWED")
            if stat.S_ISDIR(info.st_mode) and relative in directories:
                pending.append(path)
            elif stat.S_ISREG(info.st_mode) and relative in allowed:
                actual.add(relative)
            else:
                raise ValueError("VISION_TRANSFER_UNLISTED_PATH:" + relative)
    if actual != allowed:
        raise ValueError("VISION_TRANSFER_INVENTORY_MISMATCH")
    for name, digest in controls.items():
        actual_digest, size = file_digest(root / name, 256 * 1024**2)
        if actual_digest != digest:
            raise ValueError("VISION_TRANSFER_CONTROL_CHANGED:" + name)
        total += size
    report = {"schema": "dronedream.vision-transfer-verification.v1", "verified": True,
              "receipt_sha256": expected, "files": len(allowed), "bytes": total,
              "training_started": False, "model_capability_proven": False}
    return report


# 功能：
#   在 SSH 上传后验证数据完整性；结果写标准输出，避免把校验日志混入只读数据集。
# 输入：
#   命令行参数：明确的数据目录和另外保存的回执摘要。
# 输出：
#   exit_code：全部索引文件及目录核验通过为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset.absolute(), args.sha256)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
