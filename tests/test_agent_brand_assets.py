from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BRAND_ROOT = ROOT / "app" / "frontend" / "public" / "brand"
MARK_PATH = BRAND_ROOT / "dronedream-agent-mark.png"
LOCKUP_PATH = BRAND_ROOT / "dronedream-agent-lockup.png"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visible_height(alpha: Image.Image, left: int, right: int) -> int:
    bbox = alpha.crop((left, 0, right, alpha.height)).getbbox()
    assert bbox is not None
    return bbox[3] - bbox[1]


def test_agent_brand_directory_contains_only_the_two_approved_assets() -> None:
    assert sorted(path.name for path in BRAND_ROOT.iterdir() if path.is_file()) == [
        "dronedream-agent-lockup.png",
        "dronedream-agent-mark.png",
    ]


def test_agent_brand_assets_are_the_approved_canonical_bytes() -> None:
    assert _sha256(MARK_PATH) == (
        "675188a785e56e2e1d81fb42627de18cca460018f83ee12baa389ef7b4e1d72e"
    )
    assert _sha256(LOCKUP_PATH) == (
        "6d86d5a1a7d29f861a87acf51937f7c33eb2a4c6681479e2b121ccf403373ffb"
    )
    with Image.open(MARK_PATH) as mark:
        assert mark.size == (1024, 1024)
        assert mark.mode == "RGBA"
    with Image.open(LOCKUP_PATH) as lockup:
        assert lockup.size == (2718, 218)
        assert lockup.mode == "RGBA"


def test_agent_wordmark_and_label_keep_the_approved_equal_cap_height() -> None:
    """Prevent the old small-DroneDream/oversized-AGENT lockup from returning."""

    with Image.open(LOCKUP_PATH) as lockup:
        alpha = lockup.getchannel("A")

        # Exact approved glyph spans: the two DroneDream capital Ds and AGENT.
        first_d_height = _visible_height(alpha, 249, 402)
        second_d_height = _visible_height(alpha, 982, 1138)
        agent_height = _visible_height(alpha, 1917, lockup.width)

        assert first_d_height == second_d_height == 186
        assert agent_height == 190
        assert abs(agent_height - first_d_height) <= 4

        # The centered separator has equal transparent gaps on both sides.
        assert alpha.crop((1749, 0, 1807, lockup.height)).getbbox() is None
        assert alpha.crop((1859, 0, 1917, lockup.height)).getbbox() is None
        assert 1807 - 1749 == 1917 - 1859 == 58


def test_agent_installer_bundles_the_qualified_default_asset_repository() -> None:
    config = json.loads(
        (ROOT / "app" / "desktop" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    )
    resources = config["bundle"]["resources"]
    assert resources["resources/default-assets"] == "default-assets"

    bundled = json.loads(
        (
            ROOT / "app" / "desktop" / "src-tauri" / "resources" / "default-assets" / "index.json"
        ).read_text(encoding="utf-8")
    )
    packages = bundled["qualified_pair"]["packages"]
    assert [(entry["kind"], entry["asset_id"]) for entry in packages] == [
        ("map", "dronedream.school-map.v1"),
        ("vehicle", "dronedream.my-drone.v1"),
    ]
    assert "bundles" not in bundled
    asset_root = ROOT / "app" / "desktop" / "src-tauri" / "resources" / "default-assets"
    assert sorted(
        path.suffix
        for path in asset_root.iterdir()
        if path.is_file() and path.name != "index.json"
    ) == [".ddpkg", ".ddpkg"]

    verifier = (ROOT / "scripts" / "verify-packaged-sidecar.ps1").read_text(encoding="utf-8-sig")
    assert "A fresh profile must contain exactly one qualified default map and aircraft" in verifier
    assert "dronedream.school-map.v1" in verifier
    assert "dronedream.my-drone.v1" in verifier
