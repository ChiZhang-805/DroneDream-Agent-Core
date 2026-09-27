"""Freeze portable experiment tools and provenance, excluding user data/secrets."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import zipfile


# 功能：逐块计算文件摘要，不一次加载大型模型。
# 输入：文件；输出：SHA256。
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


# 功能：冻结白名单实验代码和环境，记录上游 tokenizer 兼容改写的前后差异。
# 输入：源码根、模型缓存根、新输出目录；输出：代码 ZIP、摘要及依赖清单，绝不打包账户/任务。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    packages = ("torch", "numpy", "scikit-learn", "transformers", "huggingface_hub", "safetensors", "opencv-python-headless", "laya", "Pillow")
    report = {"environment": {package: version(package) for package in packages},
        "laya_source_commit": "970dc8c5f63d7b886a68409493f37d569424f933",
        "laya_source": "https://github.com/NandhaKishorM/laya",
        "source_files": {}, "models": {}, "flight_authority": False, "installed_desktop_changed": False}
    for name in ("laya-english", "laya-multilingual"):
        root = args.models / name
        receipt = json.loads((root / "download-receipt.json").read_text())
        check = {"repository": receipt["repository"], "revision": receipt["revision"],
            "download_receipt_sha256": digest(root / "download-receipt.json"), "files": {}}
        for relative, original in receipt["files"].items():
            actual = digest(root / relative)
            info = {"source_sha256": original["sha256"], "runtime_sha256": actual, "unchanged": actual == original["sha256"]}
            if not info["unchanged"]:
                if relative != "tokenizer/tokenizer_config.json":
                    raise ValueError("MODEL_UNEXPECTED_MUTATION:" + relative)
                from huggingface_hub import hf_hub_download
                source = Path(hf_hub_download(receipt["repository"], relative, revision=receipt["revision"],
                    cache_dir=args.models / "source-verification", token=False))
                if digest(source) != original["sha256"]:
                    raise ValueError("SOURCE_CONFIG_HASH_MISMATCH")
                before = json.loads(source.read_text(encoding="utf-8"))
                expected = dict(before)
                if expected.get("tokenizer_class") in (None, "TokenizersBackend"):
                    expected["tokenizer_class"] = "PreTrainedTokenizerFast"
                    expected.pop("backend", None)
                    expected.pop("is_local", None)
                if isinstance(expected.get("extra_special_tokens"), list):
                    expected["extra_special_tokens"] = {f"extra_{i}": token for i, token in enumerate(expected["extra_special_tokens"])}
                if json.loads((root/relative).read_text(encoding="utf-8")) != expected:
                    raise ValueError("TOKENIZER_REPAIR_MISMATCH")
                info["explained_by"] = "Pinned upstream laya.agent._fix_tokenizer_config compatibility conversion; no weight changes."
            check["files"][relative] = info
        report["models"][name] = check
    scripts = ("evaluate_stage_decisions", "prepare_decision_model", "prepare_tartanair_sample", "prepare_tartanair_semantics", "audit_tartanair_sample", "evaluate_rgbd_odometry", "train_stage_classifier", "check_laya_adapter", "freeze_decision_experiment")
    # Historical run-by-run notes live under TestRuns and are evidence, not
    # executable training inputs.  Freeze the implementation and portable
    # operator plan without coupling a release bundle to one workstation's logs.
    selected = [
        "src/dronedream_agent_core/decision_shadow.py",
        "docs/LAYA_UAV_COMPLETE_IMPLEMENTATION_PLAN.md",
    ] + ["scripts/" + name + ".py" for name in scripts]
    target = args.output / "decision-experiment-tools.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        # 最小命名空间包无副作用，不把整套产品依赖或用户数据库装进研究包。
        bundle.writestr("src/dronedream_agent_core/__init__.py", '"""Offline experiment namespace."""\n')
        for relative in selected:
            source = args.core / relative
            report["source_files"][relative] = digest(source)
            bundle.write(source, relative)
    report["archive_sha256"] = digest(target)
    with (args.output / "experiment-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"archive": str(target), "source_files": len(selected), "flight_authority": False}))


if __name__ == "__main__":
    main()
