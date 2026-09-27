import copy
from types import SimpleNamespace

import pytest

from dronedream_agent_app.asset_interpretation import AssetInterpretationService, AssetUnderstanding
from dronedream_agent_app.custom_models import ModelConnection
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_app import interpretation_reuse
from dronedream_agent_app.asset_interpretation import INTERPRETATION_PROMPT
import hashlib
import json


@pytest.fixture
def case(tmp_path):
    store = AppStore(tmp_path)
    calls = []
    class Port:
        def call(self, **kwargs):
            calls.append(kwargs)
            ids = kwargs["input_artifact"]["required_source_ids"]
            return SimpleNamespace(artifact=AssetUnderstanding(summary="地图解析", items=[{"source_id": key, "explanation": f"说明{key}"} for key in ids], limitations=["动态障碍需实时感知"]), record=SimpleNamespace(model_dump=lambda **_: {"call_id": f"call-{len(calls)}"}))
    source = {"kind": "map", "asset_id": "test-map", "content_sha256": "a" * 64, "required_source_ids": ["a", "b", "c"], "facts": {"coordinate_frame": "map_enu", "nodes": [{"node_id": key, "semantic": "room"} for key in ["a", "b", "c"]], "edges": [{"edge_id": "ab", "from_node": "a", "to_node": "b"}], "named_entities": {"room-a": "a", "room-b": "b", "room-c": "c"}}}
    options = {"source": source, "scope": {"owner_account_id": "test"}, "locale": "zh-CN", "connection": ModelConnection(selection_id="kimi", provider="kimi", model_id="kimi-k2.6", api_key="not-persisted", base_url="https://example.test", api_style="chat-completions", capability_id="model.kimi", source="default"), "port": Port()}
    yield AssetInterpretationService(store), options, calls
    store.close()


def test_local_change_reinterprets_node_and_neighbour_but_retains_independent_room(case):
    service, options, calls = case
    service.interpret(**options)
    options["source"]["facts"]["nodes"][0]["semantic"] = "pickup"
    options["source"]["content_sha256"] = "b" * 64
    result = service.interpret(**options)
    assert result["reuse_mode"] == "incremental" and result["retained_items"] == 1
    assert calls[-1]["input_artifact"]["required_source_ids"] == ["a", "b"]
    assert {item["source_id"] for item in result["understanding"]["items"]} == {"a", "b", "c"}
    assert service.interpret(**options)["cached"]


@pytest.mark.parametrize("change", ["coordinate", "topology", "owner", "force"])
def test_global_change_or_account_change_does_not_reuse_local_explanations(case, change):
    service, options, calls = case
    service.interpret(**options)
    options["source"]["content_sha256"] = "b" * 64
    if change == "coordinate": options["source"]["facts"]["coordinate_frame"] = "changed"
    if change == "topology": options["source"]["facts"]["edges"].append({"edge_id": "bc", "from_node": "b", "to_node": "c"})
    if change == "owner": options["scope"] = {"owner_account_id": "another"}
    if change == "force": options["force"] = True
    result = service.interpret(**options)
    assert result["reuse_mode"] == "full"
    assert calls[-1]["input_artifact"]["required_source_ids"] == ["a", "b", "c"]


def test_removed_node_never_survives_in_merged_interpretation(case):
    service, options, calls = case
    service.interpret(**options)
    changed = copy.deepcopy(options["source"])
    changed["required_source_ids"] = ["a", "b"]
    changed["facts"]["nodes"] = changed["facts"]["nodes"][:2]
    del changed["facts"]["named_entities"]["room-c"]
    changed["content_sha256"] = "b" * 64
    result = service.interpret(**{**options, "source": changed})
    assert {item["source_id"] for item in result["understanding"]["items"]} == {"a", "b"}


# 功能：
#   未命名远端楼梯变化也使总体解释失效，避免摘要继续引用旧楼层或通路。
# 输入：
#   case：具有独立区域的地图夹具。
# 输出：
#   无：远端变化要求完整解析。
def test_unnamed_remote_node_change_invalidates_global_summary(case):
    service, options, calls = case
    options["source"]["facts"]["nodes"].append({"node_id": "remote-stair", "semantic": "stair"})
    service.interpret(**options)
    options["source"]["facts"]["nodes"][-1]["semantic"] = "closed-stair"
    result = service.interpret(**options)
    assert result["reuse_mode"] == "full" and len(calls) == 2


# 功能：
#   验证发布包被精确复用、写入账户缓存，且显式重解析不会被发布包拦截。
# 输入：
#   case：隔离资产；tmp_path：隔离程序目录；monkeypatch：测试替换器。
# 输出：
#   无：调用计数和缓存来源符合约定。
def test_exact_bundle_reuse_requires_no_model_budget(case, tmp_path, monkeypatch):
    service, options, calls = case
    monkeypatch.setattr(interpretation_reuse, "__file__", str(tmp_path / "interpretation_reuse.py"))
    source = options["source"]
    prompt = hashlib.sha256(INTERPRETATION_PROMPT.encode()).hexdigest()
    key = sha256_json({"source": sha256_json(source), "locale": "zh-CN", "prompt": prompt})
    root = tmp_path / "bundled_interpretations"
    root.mkdir()
    understanding = {"summary": "发布地图知识", "items": [{"source_id": item, "explanation": item} for item in source["required_source_ids"]], "limitations": []}
    (root / f"{key}.json").write_text(json.dumps({"source_sha256": sha256_json(source), "locale": "zh-CN", "prompt_sha256": prompt, "understanding": understanding, "output_sha256": sha256_json(understanding)}), encoding="utf-8")
    result = service.interpret(**options, remaining_calls=0)
    assert result["reuse_mode"] == "bundled" and calls == []
    assert service.interpret(**options, remaining_calls=0)["reuse_mode"] == "exact"
    path = root / f"{key}.json"
    revised = json.loads(path.read_text(encoding="utf-8"))
    revised["understanding"]["summary"] = "发布地图知识的纠错版本"
    revised["output_sha256"] = sha256_json(revised["understanding"])
    path.write_text(json.dumps(revised), encoding="utf-8")
    upgraded = service.interpret(**options, remaining_calls=0)
    assert upgraded["reuse_mode"] == "bundled"
    assert upgraded["understanding"]["summary"] == "发布地图知识的纠错版本"
    assert service.interpret(**options, force=True)["reuse_mode"] == "full"
    assert len(calls) == 1
    changed = copy.deepcopy(source)
    changed["facts"]["coordinate_frame"] = "changed"
    assert interpretation_reuse.bundled_interpretation(changed, "zh-CN", prompt) is None


# 功能：
#   畸形发布包按缓存缺失处理，不变成规划接口的服务器错误。
# 输入：
#   case：隔离资产；tmp_path：隔离目录；monkeypatch：测试替换器。
# 输出：
#   无：畸形数据被拒绝。
def test_malformed_bundle_is_not_loaded(case, tmp_path, monkeypatch):
    _, options, _ = case
    monkeypatch.setattr(interpretation_reuse, "__file__", str(tmp_path / "interpretation_reuse.py"))
    source = options["source"]
    key = sha256_json({"source": sha256_json(source), "locale": "zh-CN", "prompt": "prompt"})
    root = tmp_path / "bundled_interpretations"
    root.mkdir()
    (root / f"{key}.json").write_text("[]", encoding="utf-8")
    assert interpretation_reuse.bundled_interpretation(source, "zh-CN", "prompt") is None
