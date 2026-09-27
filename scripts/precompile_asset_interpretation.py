"""Build release-owned, hash-bound reusable understanding with real model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dronedream_agent_app.asset_interpretation import AssetInterpretationService, INTERPRETATION_PROMPT, interpretation_source, validate_understanding, AssetUnderstanding
from dronedream_agent_app.custom_models import ModelConnection
from dronedream_agent_app.storage import AppStore
from dronedream_agent_core.hashing import sha256_json


# 功能：
#   从明确的资产版本真实生成发布解析包；失败不写包，已有相同输出不重复调用模型。
# 输入：
#   命令行参数：隔离资产库、资产标识、内容摘要、类型、输出目录和语言。
# 输出：
#   JSON：与资产字节、提示词和语言绑定的解析及模型调用证据。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--kind", choices=["map", "vehicle"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--locale", choices=["zh-CN", "en-US"], required=True)
    args = parser.parse_args()
    store = AppStore(args.data_root)
    try:
        source = interpretation_source(store, args.kind, args.asset_id, args.sha256)
        prompt_sha = hashlib.sha256(INTERPRETATION_PROMPT.encode()).hexdigest()
        source_sha = sha256_json(source)
        key = sha256_json({"source": source_sha, "locale": args.locale, "prompt": prompt_sha})
        output = args.output_dir / f"{key}.json"
        if output.exists():
            record = json.loads(output.read_text(encoding="utf-8"))
            checked = validate_understanding(AssetUnderstanding.model_validate(record["understanding"]), source)
            if record["source_sha256"] != source_sha or record["locale"] != args.locale or record["prompt_sha256"] != prompt_sha or record["output_sha256"] != sha256_json(checked):
                raise ValueError("PRECOMPILED_INTERPRETATION_INVALID")
            print(json.dumps({"reused": True, "file": str(output)}))
            return
        connection = ModelConnection(selection_id="kimi-k2.6", provider="kimi", model_id="kimi-k2.6", api_key=os.environ["KIMI_API_KEY"], base_url=os.environ.get("KIMI_BASE_URL", "https://api.moonshot.ai/v1"), api_style="chat-completions", capability_id="model.kimi", source="default")
        result = AssetInterpretationService(store).interpret(scope={"owner_account_id": "release-precompiler", "source_edition": "autonomy"}, source=source, connection=connection, locale=args.locale)
        if not result["model_calls"]:
            raise ValueError("RELEASE_PRECOMPILE_REQUIRES_MODEL_EVIDENCE")
        record = {"schema_version": "dronedream.bundled-interpretation.v1", "asset_id": args.asset_id, "content_sha256": args.sha256, "source_sha256": source_sha, "prompt_sha256": prompt_sha, "locale": args.locale, "understanding": result["understanding"], "output_sha256": sha256_json(result["understanding"]), "model_call": result["model_calls"][0], "authority": "advisory-only"}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # 仅在结果结构和引用通过校验后原子发布，不留下半份解析供软件读取。
        temporary = output.with_suffix(".pending")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(json.dumps({"asset_id": args.asset_id, "locale": args.locale, "items": len(result["understanding"]["items"]), "file": str(output)}))
    finally:
        store.close()


if __name__ == "__main__":
    main()
