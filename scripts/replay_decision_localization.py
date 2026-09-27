"""Read-only replay of native geometry inputs, never producing control or training labels."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dronedream_agent_core.decision_dataset import decode_object
from dronedream_agent_core.live_map_measurement import LiveMapMeasurementSource, MapNoiseBounds
from dronedream_agent_core.optical_map import compile_optical_map
from dronedream_agent_core.training.evidence_publication import write_evidence_object


# 功能：离线重放原几何输入并保存具体配准失败原因；不把回放时钟用到在线飞控。
# 输入：本次运行的 flight 目录及新报告路径；输出：逐帧求解诊断，训练新增始终为零。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = decode_object((args.flight / "fusion-inputs.json").read_text("utf-8"))
    report = decode_object((args.flight / "fusion-stream.json").read_text("utf-8"))
    identity = decode_object(
        (args.flight / "simulation/runtime-state/px4-identity-telemetry.json").read_text("utf-8")
    )
    index, _, _ = compile_optical_map(
        Path(inputs["world_sdf"]), expected_world_sha256=inputs["world_sha256"]
    )
    # 起步成功帧不能代替后来失败现场；按原源钟重放，不按文件名前缀倒置时间。
    paths = [
        *args.flight.glob("geometry-input-*.json"),
        *args.flight.glob("geometry-failure-*.json"),
    ]
    paths.sort(
        key=lambda p: decode_object(p.read_text("utf-8"))["native_odometry_snapshot"][
            "source_alignment"
        ]["image_timestamp_ns"]
    )
    if not paths:
        raise ValueError("DECISION_LOCALIZATION_REPLAY_INPUTS_MISSING")
    first = decode_object(paths[0].read_text("utf-8"))
    source = LiveMapMeasurementSource(
        index=index,
        map_sha256=inputs["semantic_sha256"],
        binding=identity["map_frame_binding"],
        binding_sha256=identity["map_frame_binding_sha256"],
        clock_domain=first["native_odometry_snapshot"]["source_alignment"]["clock_domain"],
        noise=MapNoiseBounds(**report["noise"]),
    )
    original = source.tracker.update_native
    fits = []

    # 功能：旁路记录求解器原返回值，不篡改输入或结果；输入：原参数；输出：原返回对象。
    def capture(**kwargs):
        result = original(**kwargs)
        fits.append(asdict(result))
        return result

    source.tracker.update_native = capture
    rows = []
    seen = set()
    for path in paths:
        record = decode_object(path.read_text("utf-8"))
        stamp = record["native_odometry_snapshot"]["source_alignment"]["image_timestamp_ns"]
        if stamp in seen:
            continue
        seen.add(stamp)
        # 离线时钟固定为该帧实际两个接收时刻的较大值，绝不能使用重启后主机单调钟。
        received = max(
            record["native_odometry_snapshot"]["source_alignment"]["received_monotonic_seconds"],
            record["scan"]["observed_at_monotonic_seconds"],
        )
        count = len(fits)
        try:
            source.measure(
                record,
                source_now_ns=lambda stamp=stamp: stamp,
                monotonic_now=lambda received=received: received,
            )
            status = "measured-offline"
        except ValueError as error:
            status = str(error)
        rows.append(
            {"input": path.name, "result": status, "fit": fits[-1] if len(fits) > count else None}
        )
    result = {"offline_only": True, "formal_training_additions": 0, "rows": rows}
    write_evidence_object(args.output, result)
    print(
        json.dumps(
            [
                {
                    "input": r["input"],
                    "result": r["result"],
                    "issue": r["fit"]["fit"]["issue"] if r["fit"] else None,
                }
                for r in rows
            ]
        )
    )


if __name__ == "__main__":
    main()
