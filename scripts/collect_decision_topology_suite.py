"""Sequential, bounded physical collection with immediate audit and explicit failure retention."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from dronedream_agent_core.decision_dataset import decode_object, evidence_path, file_digest
from dronedream_agent_core.decision_scene_topology import verified_scene_family
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.px4_environment import Px4TrainingConfig
from scripts.assemble_decision_corpus import assemble
from scripts.audit_decision_stage_windows import audit
from scripts.collect_native_decisions import collect
from scripts.prepare_decision_native_scene import make_world


# 功能：逐项复核预先划分的地图/路线/配置，禁止按运行结果选择测试集或悄悄换文件。
# 输入：套件清单路径；输出：经过摘要与配方复核的配置和集合，不启动仿真。
def preflight_suite(path):
    if path.stat().st_size > 1024**2:
        raise ValueError("DECISION_SUITE_MANIFEST_TOO_LARGE")
    suite = decode_object(path.read_text("utf-8"))
    if (
        suite.get("schema_version") != "dronedream.decision-topology-suite.v1"
        or type(suite.get("layouts")) is not list
        or not 1 <= len(suite["layouts"]) <= 20
    ):
        raise ValueError("DECISION_SUITE_MANIFEST_INVALID")
    families, configs = set(), []
    for entry in suite["layouts"]:
        config_path = evidence_path(path.parent, entry["config"])
        receipt_path = evidence_path(path.parent, entry["receipt"])
        if file_digest(config_path) != entry["config_sha256"]:
            raise ValueError("DECISION_SUITE_CONFIG_CHANGED")
        config = Px4TrainingConfig.model_validate(decode_object(config_path.read_text("utf-8")))
        receipt = decode_object(receipt_path.read_text("utf-8"))
        family = verified_scene_family(receipt, config.asset_sha256["semantic"])
        if (
            family != entry["family"]
            or family in families
            or entry["split"] != receipt["split"]
            or entry["split"] not in {"train", "development", "calibration", "test", "stress"}
        ):
            raise ValueError("DECISION_SUITE_FAMILY_OR_SPLIT_INVALID")
        for key in ("semantic", "route", "world_sdf"):
            if file_digest(getattr(config, key)) != config.asset_sha256[key]:
                raise ValueError("DECISION_SUITE_ASSET_CHANGED:" + key)
        semantic = decode_object(config.semantic.read_text("utf-8"))
        expected_world = hashlib.sha256(
            make_world(semantic["collision_primitives"]).encode("utf-8")
        ).hexdigest()
        if (
            receipt.get("world_sha256") != expected_world
            or config.asset_sha256["world_sdf"] != expected_world
        ):
            raise ValueError("DECISION_SUITE_RENDER_COLLISION_MISMATCH")
        families.add(family)
        configs.append((entry, config))
    return configs


# 功能：按清单顺序串行执行，落地后立即导出/组装；不绕过拒收，也不把提案计作正式样本。
# 输入：预检套件、新输出目录、最多20个布局和每轮3～120秒；输出：逐轮保留的结果清单。
# 无安全关闭、两轮连续采集异常或剩余空间不足10GiB立即结束批次，避免无意义反复飞行。
def run_suite(suite_path, output, *, maximum_layouts=20, seconds=120.0):
    if (
        type(maximum_layouts) is not int
        or not 1 <= maximum_layouts <= 20
        or type(seconds) not in (int, float)
        or not 3 <= seconds <= 120
    ):
        raise ValueError("DECISION_SUITE_BUDGET_INVALID")
    configs = preflight_suite(suite_path)
    output.mkdir(parents=True, exist_ok=False)
    reports, failures, empty_attempts, stop_reason = [], 0, 0, None
    for index, (entry, config) in enumerate(configs[:maximum_layouts]):
        if shutil.disk_usage(output).free < 10 * 1024**3:
            stop_reason = "INSUFFICIENT_EVIDENCE_DISK_SPACE"
            break
        attempt = output / f"layout-{index:02d}"
        attempt.mkdir()
        runtime_root = attempt / "native-run"
        runtime_root.mkdir()
        config = Px4TrainingConfig.model_validate(
            {**config.model_dump(mode="python"), "output_root": runtime_root}
        )
        row = {
            "layout": entry["config"],
            "split": entry["split"],
            "family": entry["family"],
            "formal_decision_windows": 0,
            "collection": None,
            "audit_error": None,
        }
        try:
            report = collect(config, seconds, "corridor", split=entry["split"], route_guidance=True)
            row["collection"] = report
        except Exception as error:
            row["collection"] = {
                "error": type(error).__name__ + ":" + str(error)[:500],
                "safely_closed": False,
            }
        report = row["collection"]
        failed = False
        if report.get("safely_closed") is not True:
            stop_reason = "SAFE_TERMINAL_STATE_NOT_CONFIRMED"
        elif report.get("error") or report.get("archive_errors"):
            failed = True
        else:
            try:
                episode = Path(report["episode_path"])
                # 回合必须归属于本轮输出，不能误用另一轮较好的运行结果。
                if not episode.resolve().is_relative_to(runtime_root.resolve()):
                    raise ValueError("DECISION_SUITE_EPISODE_OUTSIDE_ATTEMPT")
                audit(
                    episode,
                    attempt / "stage-audit.json",
                    export_root=attempt / "candidates",
                    window_ms=6000,
                )
                assembled = assemble(attempt / "candidates/manifest.json", attempt / "corpus")
                row["formal_decision_windows"] = assembled["formal_decision_windows"]
                row["assembly"] = assembled
            except Exception as error:
                row["audit_error"] = type(error).__name__ + ":" + str(error)[:500]
                failed = True
        failures = failures + 1 if failed else 0
        empty_attempts = empty_attempts + 1 if row["formal_decision_windows"] == 0 else 0
        write_evidence_object(attempt / "attempt-report.json", row)
        reports.append(row)
        print(
            json.dumps(
                {
                    "layout": index,
                    "selections": report.get("selections", 0),
                    "submitted": report.get("submitted", 0),
                    "formal_windows": row["formal_decision_windows"],
                    "safely_closed": report.get("safely_closed"),
                    "error": report.get("error") or row["audit_error"],
                }
            ),
            flush=True,
        )
        if failures >= 2:
            stop_reason = "CONSECUTIVE_COLLECTION_OR_AUDIT_ERRORS"
        elif empty_attempts >= 3 and stop_reason is None:
            stop_reason = "CONSECUTIVE_COLLECTION_WITHOUT_VERIFIED_DATA"
        if stop_reason:
            break
    summary = {
        "schema_version": "dronedream.decision-suite-collection.v1",
        "suite_sha256": file_digest(suite_path),
        "attempts": len(reports),
        "formal_decision_windows": sum(r["formal_decision_windows"] for r in reports),
        "stop_reason": stop_reason,
        "gpu_ready": False,
        "flight_authority": False,
        "reports": [f"layout-{i:02d}/attempt-report.json" for i in range(len(reports))],
    }
    write_evidence_object(output / "collection-summary.json", summary)
    return summary


# 功能：启动显式实验套件并输出累计真实数量；不允许覆盖或借此发布生产飞行模型。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-layouts", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = run_suite(
        args.suite, args.output, maximum_layouts=args.maximum_layouts, seconds=args.seconds
    )
    print(json.dumps(result))
    return int(result["stop_reason"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
