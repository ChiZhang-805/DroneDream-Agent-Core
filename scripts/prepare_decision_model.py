#!/usr/bin/env python3
"""Download a pinned Laya checkpoint without touching the installed Runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import re


# 功能：按不可变版本下载指定官方权重及本地推理配置，拒绝远程可执行文件。
# 输入：固定版本 SHA 与新输出目录；下载可重入，但完成清单不可覆盖。
# 输出：逐文件大小、SHA256、许可及来源清单；不导入或执行模型。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model", choices=("laya", "laya-multilingual"), default="laya-multilingual")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be an immutable 40-character commit")
    from huggingface_hub import HfApi, snapshot_download
    model = "convaiinnovations/" + args.model
    files = ("README.md", "model.safetensors", "rl_agent_config.json", "encoder/config.json",
             "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")
    info = HfApi().model_info(model, revision=args.revision, files_metadata=True, token=False)
    if info.sha != args.revision or info.card_data.get("license") != "apache-2.0":
        raise ValueError("MODEL_REVISION_OR_LICENSE_MISMATCH")
    sizes = {item.rfilename: item.size for item in info.siblings if item.rfilename in files}
    if set(sizes) != set(files) or any(v is None for v in sizes.values()) or sum(sizes.values()) > 2 * 1024**3:
        raise ValueError("MODEL_FILE_INVENTORY_OR_SIZE_INVALID")
    if (args.output / "download-receipt.json").exists():
        raise FileExistsError("completed download receipt already exists")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(model, revision=args.revision, allow_patterns=list(files),
                      local_dir=args.output, max_workers=2, token=False)
    receipt = {"repository": model, "revision": args.revision, "license": "Apache-2.0",
               "source": "https://huggingface.co/" + model, "files": {}}
    for name in files:
        path = args.output / name
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        if path.stat().st_size != sizes[name]:
            raise ValueError("MODEL_DOWNLOAD_SIZE_MISMATCH:" + name)
        source = next(item for item in info.siblings if item.rfilename == name)
        if source.lfs and source.lfs.sha256 != result.hexdigest():
            raise ValueError("MODEL_DOWNLOAD_HASH_MISMATCH:" + name)
        receipt["files"][name] = {"sha256": result.hexdigest(), "bytes": sizes[name]}
    with (args.output / "download-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"downloaded_files": len(files), "bytes": sum(sizes.values()),
                      "revision": args.revision}), flush=True)


if __name__ == "__main__":
    main()
