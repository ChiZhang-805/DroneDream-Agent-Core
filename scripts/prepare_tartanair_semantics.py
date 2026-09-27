"""Fetch official pinned semantic names; never infer names from arbitrary IDs."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

BLOB = "63b086b83d70787106c0feff8e54965a2fe8403d"
URL = "https://api.github.com/repos/castacks/tartanairpy/git/blobs/" + BLOB


# 功能：读取官方固定 Git blob 并核对内容，提取选定环境的语义类别名。
# 输入：新输出目录；输出：原始包、选定映射及来源回执，不自动成为人员检测标签。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    request = urllib.request.Request(URL, headers={"User-Agent": "DroneDream-Data-Audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(1024**2 + 1)
    if len(payload) > 1024**2:
        raise ValueError("SEMANTIC_DOWNLOAD_TOO_LARGE")
    metadata = json.loads(payload)
    if metadata["encoding"] != "base64" or metadata["sha"] != BLOB:
        raise ValueError("SEMANTIC_BLOB_IDENTITY_INVALID")
    raw = base64.b64decode(metadata["content"])
    git_digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()
    if git_digest != BLOB or len(raw) != 169705:
        raise ValueError("SEMANTIC_BLOB_HASH_INVALID")
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        matches = [name for name in bundle.namelist() if name.endswith("ArchVizTinyHouseDay/seg_label_map.json")]
        if len(matches) != 1 or bundle.getinfo(matches[0]).file_size > 256 * 1024:
            raise ValueError("SEMANTIC_ENVIRONMENT_MAPPING_MISSING")
        mapping_raw = bundle.read(matches[0])
        mapping = json.loads(mapping_raw)
    args.output.mkdir(parents=True)
    with (args.output / "segfiles.zip").open("xb") as stream:
        stream.write(raw)
    with (args.output / "seg_label_map.json").open("xb") as stream:
        stream.write(mapping_raw)
    receipt = {"source": URL, "git_blob_sha1": BLOB, "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(), "archive_member": matches[0],
        "attribution": "TartanAir V2, The AirLab, Carnegie Mellon University", "dataset_license": "CC-BY-4.0",
        "formal_training_additions": 0}
    with (args.output / "semantic-receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"mapping_keys": list(mapping), "mapping_file": str(args.output / "seg_label_map.json")}))


if __name__ == "__main__":
    main()
