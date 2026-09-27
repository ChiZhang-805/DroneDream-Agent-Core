"""Keep generated variants in one layout family without rewriting old certificates."""

from .decision_dataset import decode_object, evidence_path, file_digest

# These are generator contracts, not arbitrary dataset-provided aliases.
CORRIDOR_FAMILIES = {
    "straight-corridor-width-variants-must-stay-in-same-split",
    "straight-corridor-crossing-variants-must-stay-in-same-split",
}


# 功能：恢复采集前固定的集合；旧无上下文回合仅视为train，不自动升为独立测试集。
# 输入：已结束回合目录；输出：集合名，存在但损坏的上下文不得回退为默认值。
def collection_split(episode):
    path = episode / "decision-collection-context.json"
    if not path.exists():
        return "train"
    if path.stat().st_size > 65536:
        raise ValueError("DECISION_COLLECTION_CONTEXT_TOO_LARGE")
    doc = decode_object(path.read_text("utf-8"))
    if doc.get("schema_version") != "dronedream.decision-collection-context.v1" or doc.get(
        "split"
    ) not in {"train", "development", "calibration", "test", "stress"}:
        raise ValueError("DECISION_COLLECTION_CONTEXT_INVALID")
    return doc["split"]


# 功能：按生成器收据归并布局变体，不把不同地图哈希自动视为独立场景。
# 输入：语料根及绑定地图/原始父组的侧车清单；输出：父组到布局族，缺清单返回空映射。
# 原样保留旧证书的parent_group；未知生成器拒绝，新增场景族需明确实现其来源校验。
def load_layout_families(root, *, assignments=None):
    manifest = root / "layout-lineage.json"
    if not manifest.exists():
        return {}, {}
    if manifest.stat().st_size > 4 * 1024**2:
        raise ValueError("DECISION_LAYOUT_LINEAGE_TOO_LARGE")
    document = decode_object(manifest.read_text("utf-8"))
    if (
        set(document) != {"schema_version", "entries"}
        or document["schema_version"] != "dronedream.decision-layout-lineage.v1"
        or type(document["entries"]) is not list
        or not 1 <= len(document["entries"]) <= 10000
    ):
        raise ValueError("DECISION_LAYOUT_LINEAGE_INVALID")
    groups, maps = {}, {}
    for entry in document["entries"]:
        if type(entry) is not dict or set(entry) != {"parent_group", "map_sha256", "receipt"}:
            raise ValueError("DECISION_LAYOUT_LINEAGE_ENTRY_INVALID")
        group, map_sha = entry["parent_group"], entry["map_sha256"]
        if type(group) is not str or not group or type(map_sha) is not str or len(map_sha) != 64:
            raise ValueError("DECISION_LAYOUT_LINEAGE_ID_INVALID")
        reference = entry["receipt"]
        if type(reference) is not dict or set(reference) != {"path", "sha256"}:
            raise ValueError("DECISION_LAYOUT_RECEIPT_INVALID")
        path = evidence_path(root, reference["path"])
        if path.stat().st_size > 65536 or file_digest(path) != reference["sha256"]:
            raise ValueError("DECISION_LAYOUT_RECEIPT_INVALID")
        receipt = decode_object(path.read_text("utf-8"))
        if receipt.get("scene_family") == "room-tree-generator-v1":
            from .decision_scene_topology import verified_scene_family

            family = verified_scene_family(receipt, map_sha)
            split = receipt.get("split")
            if split not in {"train", "development", "calibration", "test", "stress"}:
                raise ValueError("DECISION_LAYOUT_SPLIT_REQUIRED")
            if assignments is not None:
                assignments[group] = split
        elif (
            receipt.get("scene_family") in CORRIDOR_FAMILIES
            and receipt.get("distinct_layout_claim") is False
            and receipt.get("semantic_sha256", receipt.get("static_map_sha256")) == map_sha
        ):
            family = "native-straight-corridor-v1"
        else:
            raise ValueError("DECISION_LAYOUT_GENERATOR_NOT_VERIFIED")
        if group in groups:
            raise ValueError("DECISION_LAYOUT_LINEAGE_DUPLICATE")
        groups[group] = (map_sha, family)
        maps[map_sha] = family
    return groups, maps
