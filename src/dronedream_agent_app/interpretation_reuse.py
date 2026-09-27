"""Conservative dependency fingerprints and trusted bundled map understanding."""
import json
from pathlib import Path

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_files import read_plugin_file


# 功能：
#   为节点语义、别名和相邻边建立依赖摘要；坐标系或空间几何变化使全图解释失效。
# 输入：
#   source：已核验的当前资产事实。
# 输出：
#   lineage：全局和逐项依赖摘要，不包含用户身份或模型密钥。
def interpretation_lineage(source: dict) -> dict:
    facts = source["facts"]
    required = source["required_source_ids"]
    if source["kind"] != "map":
        return {"asset_id": source["asset_id"], "kind": source["kind"], "global": sha256_json(facts), "items": {item: sha256_json(facts) for item in required}}
    nodes = {node["node_id"]: node for node in facts.get("nodes", [])}
    edges = facts.get("edges", [])
    names = facts.get("named_entities", {})
    global_facts = {key: value for key, value in facts.items() if key not in {"nodes", "edges", "named_entities"}}
    topology = sorted((edge.get("edge_id"), edge.get("from_node"), edge.get("to_node")) for edge in edges)
    # 连通性变更可能影响远端路线，不把它当作只影响两个端点的局部修改。
    global_facts["topology"] = topology
    global_facts["node_inventory"] = sorted(nodes)
    global_facts["required_source_ids"] = sorted(required)
    covered_nodes = set(required)
    for edge in edges:
        if edge.get("from_node") in required or edge.get("to_node") in required:
            covered_nodes.update((edge.get("from_node"), edge.get("to_node")))
    # 未命名的远端楼梯或通路也可能出现在总体说明中，不能因未单独解释就忽略其变化。
    global_facts["unscoped_nodes"] = {key: node for key, node in nodes.items() if key not in covered_nodes}
    global_facts["unscoped_edges"] = [edge for edge in edges if edge.get("from_node") not in required and edge.get("to_node") not in required]
    item_hashes = {}
    for item in required:
        incident = [edge for edge in edges if item in (edge.get("from_node"), edge.get("to_node"))]
        neighbours = {edge.get(key) for edge in incident for key in ("from_node", "to_node")}
        item_hashes[item] = sha256_json({"node": nodes.get(item), "aliases": sorted(name for name, value in names.items() if value == item), "edges": incident, "neighbours": {node: nodes.get(node) for node in sorted(neighbours)}})
    return {"asset_id": source["asset_id"], "kind": source["kind"], "global": sha256_json(global_facts), "items": item_hashes}


# 功能：
#   选择仍可复用的条目，只有依赖全部未变的解释才保留。
# 输入：
#   old：旧解析记录；lineage：新依赖摘要。
# 输出：
#   retained：不受本次变化影响的源条目标识。
def retained_interpretation_ids(old: dict, lineage: dict) -> set[str]:
    previous = old.get("lineage", {})
    if any(previous.get(key) != lineage.get(key) for key in ("asset_id", "kind", "global")):
        return set()
    return {key for key, value in lineage["items"].items() if previous.get("items", {}).get(key) == value}


# 功能：
#   读取随程序发布的解析包，只接受完全匹配的原始资产摘要、提示词版本和语言。
# 输入：
#   source：已验证资产；locale：语言；prompt_sha256：解析规则版本。
# 输出：
#   record：可信内置解析记录，缺失或不匹配时为 None。
def bundled_interpretation(source: dict, locale: str, prompt_sha256: str) -> dict | None:
    # 路径只来自程序包，绝不从用户地图中寻找所谓“已批准”的解析。
    root = Path(__file__).with_name("bundled_interpretations")
    key = sha256_json({"source": sha256_json(source), "locale": locale, "prompt": prompt_sha256})
    path = root / f"{key}.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(read_plugin_file(path, limit=262144))
        if record["source_sha256"] != sha256_json(source) or record["locale"] != locale or record["prompt_sha256"] != prompt_sha256:
            return None
        if record["output_sha256"] != sha256_json(record["understanding"]):
            return None
        return record
    except (OSError, KeyError, TypeError, ValueError):
        return None
