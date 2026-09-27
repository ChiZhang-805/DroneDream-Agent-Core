#!/usr/bin/env python3
"""Reproducible offline-only replay and model evaluation; no flight channel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import time

from dronedream_agent_core.decision_shadow import (
    ACTIONS, StageState, group_split, laya_request, numeric_features, rule_decision,
    shadow_suggestion,
)


# 功能：摘要文件内容，绑定完整原始记录而非文件名或可变修改时间。
# 输入：待审计文件路径。
# 输出：SHA256 字符串；只读原文件。
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


# 功能：以独占创建方式保存报告，禁止覆盖前次实验与源数据。
# 输入：新文件路径及可序列化数据。
# 输出：无；路径已存在或数据含非法数值时抛错。
def save(path, data):
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


# 功能：逐行读取有限大小的 JSONL，损坏记录计入诊断，不以默认安全数据替代。
# 输入：运行日志路径。
# 输出：行号、对象、错误代码；不丢弃错误计数。
def rows(path):
    with path.open("rb") as stream:
        number, limit = 0, 4 * 1024**2
        while True:
            line = stream.readline(limit + 1)
            if not line:
                break
            number += 1
            if len(line) > limit:
                while not line.endswith(b"\n"):
                    line = stream.readline(limit + 1)
                    if not line:
                        break
                yield number, None, "row-too-large"
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not an object")
                yield number, row, None
            except (ValueError, UnicodeError):
                yield number, None, "invalid-json-row"


# 功能：提取真实运行的因果状态；预测路径的净空不冒充整条目标路线的可执行性。
# 输入：一条本地安全历史；模型授权造成的 stream_healthy=False 不冒充传感器故障。
# 输出：保守影子状态，未知路线和障碍持久性明确为 None。
def replay_state(row):
    observation = row.get("observation") or {}
    command = row.get("command") or {}
    encodings = (row.get("realtime_feature_snapshot") or {}).get("encodings", [])
    now = command.get("generated_at_unix_ms", row.get("recorded_at_unix_ms"))
    if type(now) is not int or now < 0:
        raise ValueError("REPLAY_TIMESTAMP_INVALID")
    freshness = {}
    for encoding in encodings:
        age = now - encoding["observed_at_unix_ms"]
        freshness[encoding["encoder_role"]] = (
            0 <= age <= encoding["maximum_age_milliseconds"]
            and not encoding.get("issue_codes"))
    velocity = observation.get("current_velocity_mps")
    speed = None if velocity is None else math.sqrt(sum(velocity[k] ** 2 for k in "xyz"))
    age = (now - observation["observed_at_unix_ms"]
           if "observed_at_unix_ms" in observation else None)
    # 动态障碍的存在不意味着正在横穿；空列表也可能只是缺少检测证据。
    return StageState(pose_fresh=freshness.get("flight-state-encoder"),
        geometry_fresh=freshness.get("metric-geometry-encoder"), route_verified=None,
        dynamic_crossing=None, persistent_blockage=None, caution_required=None,
        speed_mps=speed, observation_age_ms=age)


# 功能：覆盖所有三值状态组合，专用于接口/决策规则实验，不伪装成采集照片。
# 输入：无。
# 输出：带显式 synthetic 标记、任务组划分和规则参考标签的测试样本。
def contract_cases():
    result = []
    for values in itertools.product((False, True, None), repeat=6):
        group = "contract-v1:" + json.dumps(values)
        for media in (False, True):
            state = StageState(*values, optional_media_available=media)
            result.append({"id": f"{group}:media={media}", "group": group,
                "split": group_split(group), "state": asdict(state),
                "reference": rule_decision(state), "label_source": "synthetic-rule-contract",
                "formal_training_eligible": False})
    return result


# 功能：单独覆盖五类边界，避免三值组合中“未知”多数类掩盖继续/等待的错误。
# 输入：无。
# 输出：人工定义的接口案例；不加入小分类器训练，不声称独立飞行验证。
def boundary_cases():
    clear = StageState(True, True, True, False, False, False)
    scenarios = [("clear", clear, "follow_route"),
        ("caution", replace(clear, caution_required=True), "slow_down"),
        ("crossing", replace(clear, dynamic_crossing=True), "wait"),
        ("blocked", replace(clear, persistent_blockage=True), "replan"),
        ("stale-pose", replace(clear, pose_fresh=False), "request_observation"),
        ("unknown-pose", replace(clear, pose_fresh=None), "request_observation"),
        ("stale-depth", replace(clear, geometry_fresh=False), "request_observation"),
        ("unknown-route", replace(clear, route_verified=None), "request_observation")]
    return [{"id": f"boundary:{name}:{media}", "group": f"boundary:{name}",
        "split": "boundary-audit", "state": asdict(replace(state, optional_media_available=media)),
        "reference": label, "label_source": "curated-interface-boundary",
        "formal_training_eligible": False}
        for name, state, label in scenarios for media in (False, True)]


# 功能：对全部日志统计故障类别及延迟，真实回放不自动产生训练答案。
# 输入：一个或多个运行目录；输出目录须由调用方独占创建。
# 输出：完整未标注影子样本、分阶段诊断及内容哈希。
def audit_runs(runs, output):
    corpus, reports = [], []
    for run in runs:
        report = {"run": str(run), "files": {}, "errors": Counter(),
                  "authority_reasons": Counter(), "model_failures": Counter()}
        phases = defaultdict(list)
        history = run / "depth-local-safety-history.jsonl"
        route = run / "mission-route.json"
        if not history.is_file() or not route.is_file():
            raise ValueError(f"REPLAY_REQUIRED_FILES_MISSING:{run}")
        group = "route:" + digest(route)
        report["files"][history.name] = digest(history)
        report["files"][route.name] = digest(route)
        for number, row, error in rows(history):
            if error:
                report["errors"][error] += 1
                continue
            command = row.get("command") or {}
            if not isinstance(command, dict):
                report["errors"]["command-not-object"] += 1
                command = {}
            report["authority_reasons"][str(command.get("model_authority_reason", "no-command"))] += 1
            for key, value in (row.get("pipeline_phase_ms") or {}).items():
                if type(value) in (float, int) and math.isfinite(value) and value >= 0:
                    phases[key].append(value)
            try:
                state = replay_state(row)
            except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
                report["errors"]["state-incomplete-or-invalid"] += 1
                continue
            corpus.append({"id": f"{report['files'][history.name]}:{number}",
                "group": group, "split": group_split(group), "state": asdict(state),
                "source_file_sha256": report["files"][history.name], "source_line": number,
                "reference": None, "label_source": "unlabeled-runtime-replay",
                "formal_training_eligible": False})
        cycle_path = run / "model-navigation-cycles.jsonl"
        if cycle_path.is_file():
            report["files"][cycle_path.name] = digest(cycle_path)
            for _, row, error in rows(cycle_path):
                if error:
                    report["errors"][error] += 1
                else:
                    report["model_failures"][row.get("failure_reason_code") or row.get("hold_reason") or "none"] += 1
        report["phase_ms"] = {key: {"count": len(values), "p50": statistics.median(values),
            "p95": sorted(values)[math.ceil(len(values) * .95) - 1], "max": max(values)}
            for key, values in phases.items()}
        report["phase_note"] = "PhaseTimings.mark stores consecutive phase durations, not cumulative latency."
        reports.append(report)
    # 同内容任务日志重复传入不是独立样本。
    if len({r["id"] for r in corpus}) != len(corpus):
        raise ValueError("REPLAY_DUPLICATE_ROWS")
    save(output / "runtime-replay.json", corpus)
    save(output / "runtime-audit.json", reports)
    return corpus


# 功能：统一计算有标签的接口测试成绩与延迟；无标签回放不报告正确率。
# 输入：逐条候选结果。
# 输出：覆盖率、平均/尾部延迟、准确率及类别计数；不是飞行验收。
def metrics(results):
    latencies = [r["latency_ms"] for r in results]
    labeled = [r for r in results if r["reference"] is not None]
    return {"count": len(results), "labeled_count": len(labeled),
        "invalid_output_count": sum(r.get("choice") not in ACTIONS for r in results),
        "non_executable_suggestion_count": sum(r.get("shadow", {}).get("execution_authority") is False for r in results),
        "contract_accuracy": (sum(r["choice"] == r["reference"] for r in labeled) / len(labeled)
                              if labeled else None),
        "p50_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": sorted(latencies)[math.ceil(len(latencies) * .95) - 1] if latencies else None,
        "choices": dict(Counter(str(r["choice"]) for r in results)),
        "per_class": {action: {"count": sum(r["reference"] == action for r in labeled),
            "correct": sum(r["reference"] == action and r["choice"] == action for r in labeled)}
            for action in ACTIONS},
        "flight_acceptance": False}


# 功能：离线运行规则、小型分类器和可选 Laya，同样本比较且不修改 Runtime。
# 输入：日志目录、独占输出目录、可选已下载的本地模型。
# 输出：语义明确的实验报告；模型错误逐条保存，不静默用规则替换模型成绩。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--laya-model", type=Path)
    parser.add_argument("--maximum-cases", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.maximum_cases <= 2000:
        parser.error("maximum-cases must be 1..2000")
    args.output.mkdir(parents=True, exist_ok=False)
    replay = audit_runs(args.run, args.output)
    cases = contract_cases()
    save(args.output / "contract-cases.json", cases)
    train = [r for r in cases if r["split"] == "train"]
    test = [r for r in cases if r["split"] == "test"]
    # 均匀选择测试样本，与模型输出无关；可选媒体变体保持同一任务组。
    chosen = test[::max(1, math.ceil(len(test) / args.maximum_cases))][:args.maximum_cases]
    chosen += boundary_cases()
    chosen += replay[::max(1, math.ceil(len(replay) / 32))][:32]
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(max_depth=8, random_state=24)
    tree.fit([numeric_features(StageState(**r["state"])) for r in train],
             [r["reference"] for r in train])
    from sklearn import __version__ as sklearn_version
    save(args.output / "tiny-tree.json", {"classes": tree.classes_.tolist(),
        "feature": tree.tree_.feature.tolist(), "threshold": tree.tree_.threshold.tolist(),
        "children_left": tree.tree_.children_left.tolist(),
        "children_right": tree.tree_.children_right.tolist(), "value": tree.tree_.value.tolist(),
        "training_scope": "synthetic-contract-only", "sklearn_version": sklearn_version})
    predictors = {
        "rules": lambda state: {"choice": rule_decision(state)},
        "tiny-tree": lambda state: {"choice": str(tree.predict([numeric_features(state)])[0])},
    }
    metadata = {"platform": platform.platform(), "formal_training_additions": 0,
        "source_sha256": digest(Path(__file__)), "models": {}, "output_authority": "shadow-only"}
    if args.laya_model:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        import laya
        torch.set_num_threads(4)
        started = time.perf_counter()
        agent = laya.load(str(args.laya_model.resolve()), device="cpu")
        metadata["models"]["laya"] = {"package": laya.__version__, "torch": torch.__version__,
            "device": str(agent.device), "load_ms": (time.perf_counter() - started) * 1000,
            "weights_sha256": digest(args.laya_model / "model.safetensors"),
            "config_sha256": digest(args.laya_model / "rl_agent_config.json")}

        # 功能：解析真实 Laya 概率接口，不把 confidence 字段误当选择概率。
        # 输入：类型约束后的状态。
        # 输出：原始概率及明确无执行权限的影子建议。
        def predict_laya(state):
            text, questions = laya_request(state)
            answer = agent.predict(text, questions)["answers"]["stage"]
            result = shadow_suggestion(answer["probabilities"])
            return {"choice": result["raw_choice"], "shadow": result}

        predictors["laya"] = predict_laya
    all_results, summary = {}, {}
    for name, predict in predictors.items():
        results = []
        for index, case in enumerate(chosen):
            started = time.perf_counter()
            try:
                prediction = predict(StageState(**case["state"]))
            except Exception as exc:
                prediction = {"choice": None, "error_type": type(exc).__name__, "error": str(exc)[:300]}
            results.append({"id": case["id"], "reference": case["reference"],
                "label_source": case["label_source"], **prediction,
                "latency_ms": (time.perf_counter() - started) * 1000})
            if name == "laya" and index % 16 == 0:
                print(json.dumps({"model": name, "completed": index + 1, "total": len(chosen)}), flush=True)
        all_results[name] = results
        summary[name] = {kind: metrics([r for r in results if r["label_source"] == kind])
            for kind in {r["label_source"] for r in results}}
    save(args.output / "predictions.json", all_results)
    save(args.output / "evaluated-cases.json", chosen)
    save(args.output / "summary.json", {**metadata, "results": summary})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
