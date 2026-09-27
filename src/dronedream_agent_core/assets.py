"""Read bounded semantic context; route-graph availability is not flight qualification.

The asset-pair admission layer owns provenance and environment qualification.
This module neither reconstructs a route from an image nor fills missing 3-D
coordinates or aliases from a built-in map.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from dronedream_plugin_sdk.protocol import decode_json, encode_json

from .contracts import CatalogEntity, MapAsset, MapCatalog, PlanarCatalogLandmark, Vector3
from .plugin_files import read_plugin_file

SUPPORTED_MAP_SEMANTIC_SCHEMAS = {"dronedream.map-semantic.v1"}
MAX_SEMANTIC_BYTES = 16 * 1024 * 1024
MAX_SEMANTIC_NODES = 1_000_000

UNQUALIFIED_TOPOLOGY_LIMIT = (
    "This asset exposes road polylines and facility anchors, not a qualified 3-D route graph."
)


class AssetQualificationError(ValueError):
    """The supplied asset cannot support the requested planning stage."""


# 功能：
#   从同一份有界字节解析语义，拒绝重复键、非有限数和过深结构，避免摘要与散列读取不同文件。
# 输入：
#   path：当前选中资产的语义路径。
# 输出：
#   snapshot：解析对象和对应原始字节二元组；文件检查不替代资产存储的不可变性约束。
def _load_object(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = read_plugin_file(path, limit=MAX_SEMANTIC_BYTES)
        value = decode_json(raw, limit=MAX_SEMANTIC_BYTES, node_limit=MAX_SEMANTIC_NODES)
    except (OSError, ValueError) as error:
        raise AssetQualificationError("MAP_SEMANTIC_READ_INVALID") from error
    if not isinstance(value, dict):
        raise AssetQualificationError("MAP_SEMANTIC_OBJECT_REQUIRED")
    snapshot = value, raw
    return snapshot


# 功能：
#   按地图专用的字节及结构预算读取完整语义，不误用小型实时控制消息的节点限制。
# 输入：
#   path：当前资产的明确语义文件路径。
# 输出：
#   semantic：有界、有限、无重复键的对象；具体地图及执行契约由调用方继续校验。
def read_map_semantic_object(path: Path) -> dict[str, Any]:
    semantic, _raw = _load_object(path)
    return semantic


# 功能：
#   接收明确的三维 ENU 米制坐标，禁止给二维位置补出未经测量的高度。
# 输入：
#   value：三个数的列表，或仅含 x、y、z 的对象。
# 输出：
#   position：有限三维位置，布尔及字符串坐标均拒绝。
def _vector(value: object) -> Vector3:
    if isinstance(value, dict) and value.keys() == {"x", "y", "z"}:
        components = [value[axis] for axis in ("x", "y", "z")]
    elif isinstance(value, list) and len(value) == 3:
        components = value
    else:
        raise AssetQualificationError("MAP_SEMANTIC_ENU_ANCHOR_INVALID")
    try:
        if any(type(item) not in (int, float) or not math.isfinite(item) for item in components):
            raise ValueError("invalid coordinate")
        position = Vector3(x=components[0], y=components[1], z=components[2])
        return position
    except (ValueError, OverflowError) as error:
        raise AssetQualificationError("MAP_SEMANTIC_ENU_ANCHOR_INVALID") from error


# 功能：
#   校验资产标识和说明，不把空值或布尔值转换成看似真实的名称。
# 输入：
#   value：原始文本值。
#   field：用于定位错误的字段名。
#   maximum：允许的字符上限。
# 输出：
#   text：去掉首尾空白的非空文本。
def _text(value: object, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(char) < 32 for char in value)
    ):
        raise AssetQualificationError(f"MAP_SEMANTIC_{field}_INVALID")
    text = value.strip()
    return text


# 功能：
#   逐项验证文本并按首次出现去重，不静默丢弃无效字段。
# 输入：
#   value：原始列表。
#   field：错误定位字段名。
#   maximum：去重前允许的条目数。
# 输出：
#   texts：保持原顺序的独立文本列表。
def _text_list(value: object, *, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise AssetQualificationError(f"MAP_SEMANTIC_{field}_INVALID")
    texts = list(dict.fromkeys(_text(item, field=field, maximum=240) for item in value))
    return texts


# 功能：
#   保留地图道路数据明确声明的二维地标，不给它补高度，也不把它变成飞行终点。
# 输入：
#   raw、semantic：同一文件内的实体和完整语义；entity_id、aliases：已验证的身份与别名。
# 输出：
#   landmark：与原始道路锚点逐值绑定的平面地标，不合格输入抛出错误。
def _planar_landmark(raw, semantic, entity_id: str, aliases: list[str]) -> PlanarCatalogLandmark:
    position = raw.get("position_m")
    pointer = f"/roads/facility_anchors/{entity_id}"
    roads = semantic.get("roads")
    anchors = roads.get("facility_anchors") if isinstance(roads, dict) else None
    source = anchors.get(entity_id) if isinstance(anchors, dict) else None
    if (raw.get("source_pointer") != pointer
            or raw.get("semantic") not in {"landmark", "facility-anchor"}
            or not isinstance(position, list) or len(position) != 2
            or not isinstance(source, list) or len(source) != 2):
        raise AssetQualificationError("MAP_SEMANTIC_ENU_ANCHOR_INVALID")
    # 必须同时检查引用处和实体处；True == 1 不能冒充数值绑定。
    try:
        valid = all(type(v) in (int, float) and math.isfinite(v) for v in [*position, *source])
    except OverflowError:
        valid = False
    if not valid or position != source:
        raise AssetQualificationError("MAP_SEMANTIC_PLANAR_SOURCE_INVALID")
    landmark = PlanarCatalogLandmark(entity_id=entity_id, aliases=aliases,
        east_m=position[0], north_m=position[1], semantic=raw["semantic"], source_pointer=pointer)
    return landmark


# 功能：
#   合并当前已准入图的命名节点，保持语义锚点与实际接近节点的区别。
#   这里只核验结构；图、语义及环境资格是否来自同一资产由资产对准入层负责。
# 输入：
#   catalog：从当前语义文件提取的目录。
#   graph：经过资产对准入的图，未提供时不能声称具备路线拓扑。
# 输出：
#   bound：包含图实体且重新校验容量的目录。
def _bind_qualified_topology(catalog: MapCatalog, graph: MapAsset | None) -> MapCatalog:
    if graph is None:
        return catalog
    graph = MapAsset.model_validate_json(
        encode_json(graph.model_dump(mode="json"), limit=MAX_SEMANTIC_BYTES, node_limit=1_000_000),
        strict=True,
    )
    if not graph.nodes or not graph.edges or not graph.named_entities:
        raise AssetQualificationError("qualified map graph is incomplete")
    nodes = {node.node_id: node for node in graph.nodes}
    existing_entity_ids = {entity.entity_id for entity in catalog.entities}
    graph_entities: list[CatalogEntity] = []
    for entity_id, node_id in graph.named_entities.items():
        if entity_id in existing_entity_ids:
            continue
        node = nodes.get(node_id)
        if node is None:
            raise AssetQualificationError(
                f"qualified map graph named entity references an unknown node: {entity_id}"
            )
        graph_entities.append(
            CatalogEntity(
                entity_id=entity_id,
                aliases=list(
                    dict.fromkeys(
                        [
                            entity_id,
                            entity_id.replace("-", " "),
                        ]
                    )
                ),
                position_m=node.position_m,
                semantic=node.semantic,
                source_pointer=f"qualified-graph/named_entities/{entity_id}",
            )
        )
    # Revalidate the merged size/identities too; model_copy(update=...) alone
    # bypasses validation and could exceed the catalog's entity limit.
    bound = MapCatalog.model_validate_json(
        encode_json(
            {
                **catalog.model_dump(mode="json"),
                "entities": [
                    item.model_dump(mode="json") for item in [*catalog.entities, *graph_entities]
                ],
                "topology_available": True,
                "known_limits": [
                    item for item in catalog.known_limits if item != UNQUALIFIED_TOPOLOGY_LIMIT
                ],
            },
            limit=MAX_SEMANTIC_BYTES,
            node_limit=1_000_000,
        ),
        strict=True,
    )
    return bound


# 功能：
#   读取当前语义文件的真实实体与别名，并绑定同一选择的准入图；不导入内置地图的隐式目标。
# 输入：
#   semantic_path：当前语义文件路径。
#   qualified_graph：可选的已准入路线图。
# 输出：
#   catalog：带原始字节散列的语义目录；仅有语义时明确标记拓扑未核验。
def load_map_catalog(semantic_path: Path, *, qualified_graph: MapAsset | None = None) -> MapCatalog:
    semantic, raw_bytes = _load_object(semantic_path)
    schema_version = semantic.get("schema_version")
    if not isinstance(schema_version, str) or schema_version not in SUPPORTED_MAP_SEMANTIC_SCHEMAS:
        raise AssetQualificationError("unsupported map semantic schema")
    if semantic.get("coordinate_frame") != "ENU":
        raise AssetQualificationError("map semantics must use ENU coordinates")

    if schema_version == "dronedream.map-semantic.v1":
        raw_entities = semantic.get("entities")
        if not isinstance(raw_entities, list) or not 1 <= len(raw_entities) <= 2000:
            raise AssetQualificationError("map semantic entities are missing")
        entities: list[CatalogEntity] = []
        planar_landmarks: list[PlanarCatalogLandmark] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_entities):
            if not isinstance(raw, dict):
                raise AssetQualificationError("map semantic entity is invalid")
            entity_id = _text(raw.get("entity_id"), field="ENTITY_ID", maximum=160)
            if not entity_id or entity_id in seen:
                raise AssetQualificationError("map semantic entity identity is invalid")
            seen.add(entity_id)
            raw_aliases = _text_list(raw.get("aliases", []), field="ALIASES", maximum=24)
            aliases = list(
                dict.fromkeys(
                    [
                        entity_id,
                        entity_id.replace("-", " "),
                        *[alias.strip() for alias in raw_aliases],
                    ]
                )
            )
            if len(aliases) > 24:
                raise AssetQualificationError("MAP_SEMANTIC_ALIAS_LIMIT_EXCEEDED")
            position = raw.get("position_m")
            if isinstance(position, list) and len(position) == 2:
                planar_landmarks.append(_planar_landmark(raw, semantic, entity_id, aliases))
                continue
            entities.append(
                CatalogEntity(
                    entity_id=entity_id,
                    aliases=aliases,
                    position_m=_vector(raw.get("position_m")),
                    semantic=_text(raw.get("semantic", "landmark"), field="KIND", maximum=96),
                    source_pointer=_text(
                        raw.get("source_pointer", f"/entities/{index}"),
                        field="SOURCE_POINTER",
                        maximum=240,
                    ),
                )
            )
        navigation = semantic.get("navigation", {})
        if not isinstance(navigation, dict):
            raise AssetQualificationError("MAP_SEMANTIC_NAVIGATION_INVALID")
        segment_ids = _text_list(navigation.get("segment_ids", []), field="SEGMENTS", maximum=2000)
        known_limits = _text_list(semantic.get("known_limits", []), field="LIMITS", maximum=64)
        if planar_landmarks:
            limitation = "Planar landmarks have no measured altitude and are not flight targets."
            if limitation not in known_limits:
                if len(known_limits) >= 64:
                    raise AssetQualificationError("MAP_SEMANTIC_LIMITS_CAPACITY_EXCEEDED")
                known_limits.append(limitation)
        if qualified_graph is None and UNQUALIFIED_TOPOLOGY_LIMIT not in known_limits:
            if len(known_limits) >= 64:
                raise AssetQualificationError("MAP_SEMANTIC_LIMITS_CAPACITY_EXCEEDED")
            known_limits.append(UNQUALIFIED_TOPOLOGY_LIMIT)
        catalog = _bind_qualified_topology(
            MapCatalog(
                scene_id=_text(
                    semantic.get("scene_id", semantic.get("compiler_scene_id")),
                    field="SCENE_ID",
                    maximum=160,
                ),
                semantic_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                entities=entities,
                planar_landmarks=planar_landmarks,
                road_segment_ids=segment_ids,
                topology_available=False,
                known_limits=known_limits,
            ),
            qualified_graph,
        )
        return catalog
    # 支持集合扩充时必须同时增加明确解析器，不能落入隐式 None。
    raise AssetQualificationError("unsupported map semantic parser")


# 功能：
#   仅通过当前图和目录解析目的地，同名对应不同节点时要求澄清，不按列表顺序选位置。
# 输入：
#   entity：用户或计划引用的实体名称。
#   catalog：当前语义目录。
#   graph：当前路线图。
# 输出：
#   node_id：唯一解析出的节点标识；未知或歧义名称抛出异常。
def resolve_map_entity(entity: str, catalog: MapCatalog, graph: MapAsset) -> str:
    entity = _text(entity, field="ENTITY_REFERENCE", maximum=240)
    graph = MapAsset.model_validate_json(
        encode_json(graph.model_dump(mode="json"), limit=MAX_SEMANTIC_BYTES, node_limit=1_000_000),
        strict=True,
    )
    catalog = MapCatalog.model_validate_json(
        encode_json(
            catalog.model_dump(mode="json"), limit=MAX_SEMANTIC_BYTES, node_limit=1_000_000
        ),
        strict=True,
    )
    node_ids = {node.node_id for node in graph.nodes}
    if entity in node_ids:
        node_id = entity
        return node_id
    direct = graph.named_entities.get(entity)
    if direct is not None:
        node_id = direct
        return node_id
    normalized = entity.casefold().strip()
    matches: set[str] = set()
    for item in catalog.entities:
        candidates = {item.entity_id.casefold(), *(alias.casefold() for alias in item.aliases)}
        if normalized not in candidates:
            continue
        resolved = graph.named_entities.get(item.entity_id)
        if resolved is not None:
            matches.add(resolved)
    # Shared names are acceptable only when every match means the same node.
    # Otherwise the caller must clarify; list order must not choose a destination.
    if len(matches) == 1:
        node_id = next(iter(matches))
        return node_id
    if matches:
        raise AssetQualificationError("AMBIGUOUS_MAP_ENTITY")
    raise AssetQualificationError(f"UNRESOLVED_MAP_ENTITY:{entity}")


# 功能：
#   检测当前目录实体的文字提及，不推断否定、操作意图或动作权限。
# 输入：
#   message：用户原文，最多十万字符。
#   entity：待检查的当前实体名称。
#   catalog：当前语义目录。
# 输出：
#   mentioned：至少一个有效别名被提及的布尔值。
def map_entity_is_mentioned(message: str, entity: str, catalog: MapCatalog) -> bool:
    if not isinstance(message, str) or len(message) > 100_000:
        raise AssetQualificationError("MAP_SEMANTIC_MESSAGE_INVALID")
    entity = _text(entity, field="ENTITY_REFERENCE", maximum=240)
    catalog = MapCatalog.model_validate_json(
        encode_json(
            catalog.model_dump(mode="json"), limit=MAX_SEMANTIC_BYTES, node_limit=1_000_000
        ),
        strict=True,
    )
    normalized_message = message.casefold()
    normalized_entity = entity.casefold().strip()
    candidates: set[str] = set()
    for item in catalog.entities:
        item_candidates = {
            item.entity_id.casefold(),
            *(alias.casefold() for alias in item.aliases),
        }
        if normalized_entity in item_candidates:
            candidates.update(item_candidates)
    mentioned = any(
        len(candidate) >= 2 and candidate in normalized_message for candidate in candidates
    )
    return mentioned
