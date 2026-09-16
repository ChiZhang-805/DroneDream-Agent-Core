from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "shared" / "runtime-base.contract.json"


def _git(*arguments: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    submodule_relative = str(contract["submodule_path"])
    submodule = ROOT / submodule_relative
    expected_commit = str(contract["source_commit"])

    if not submodule.is_dir():
        raise RuntimeError("Shared Runtime Base submodule is not initialized")
    index_entry = _git("ls-files", "--stage", "--", submodule_relative)
    fields = index_entry.split(maxsplit=3)
    if len(fields) != 4 or fields[0] != "160000":
        raise RuntimeError("Shared Runtime Base source is not a pinned Git submodule")
    if fields[1] != expected_commit:
        raise RuntimeError("Runtime Base gitlink does not match the frozen source commit")
    if _git("rev-parse", "HEAD", cwd=submodule) != expected_commit:
        raise RuntimeError("Checked-out Runtime Base source does not match its gitlink")

    verified_bytes = 0
    for entry in contract["source_files"]:
        path = submodule / str(entry["path"])
        payload = path.read_bytes()
        if len(payload) != int(entry["bytes"]):
            raise RuntimeError(f"Runtime Base source size drifted: {entry['path']}")
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Runtime Base source hash drifted: {entry['path']}")
        verified_bytes += len(payload)

    keyring_path = submodule / "runtime" / "release-public-keys.json"
    keyring = json.loads(keyring_path.read_text(encoding="utf-8"))
    matching_keys = [
        key
        for key in keyring.get("keys", [])
        if key.get("keyId") == contract["trusted_key_id"]
        and key.get("algorithm") == "Ed25519"
        and key.get("status") == "active"
        and key.get("usage") == "runtime-release"
    ]
    if len(matching_keys) != 1:
        raise RuntimeError("Runtime Base trusted Ed25519 key is missing or ambiguous")

    desktop_library = (
        ROOT / "app" / "desktop" / "src-tauri" / "src" / "lib.rs"
    ).read_text(encoding="utf-8")
    for entry in contract["source_files"]:
        if not str(entry["path"]).endswith(".rs"):
            continue
        if str(entry["path"]) not in desktop_library:
            raise RuntimeError(
                f"Runtime Base source is not wired into the desktop: {entry['path']}"
            )

    build_script = (ROOT / "app" / "desktop" / "src-tauri" / "build.rs").read_text(
        encoding="utf-8"
    )
    if str(contract["release_manifest_url"]) not in build_script:
        raise RuntimeError("Desktop Runtime Base release manifest URL drifted")

    print(
        json.dumps(
            {
                "status": "ok",
                "source_commit": expected_commit,
                "verified_files": len(contract["source_files"]),
                "verified_bytes": verified_bytes,
                "trusted_key_id": contract["trusted_key_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
