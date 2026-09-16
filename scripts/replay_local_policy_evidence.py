#!/usr/bin/env python3
"""Replay every completed local-policy invocation from one runtime evidence set.

This diagnostic verifies the recorded JSONL hashes, resolves each forward RGB
frame by content hash, and executes the same package in original cycle order.
It is intentionally allowed to consume a failed mission because its purpose is
to distinguish deterministic model/input failures from simulator-load latency.
The output never grants simulation or production qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dronedream_agent_core.contracts import TextNavigationDecision
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.local_policy_packages import load_local_policy_package
from dronedream_agent_core.local_policy_port import LocalPolicyPort


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        records.append(value)
    return records


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _failure_record(error: BaseException) -> dict[str, str]:
    root = error
    seen: set[int] = set()
    while root.__cause__ is not None and id(root) not in seen:
        seen.add(id(root))
        root = root.__cause__
    message = str(error)
    root_message = str(root)
    record = {
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "root_exception_type": f"{type(root).__module__}.{type(root).__qualname__}",
        "diagnostic": message[:500],
        "root_diagnostic": root_message[:500],
        "fingerprint": hashlib.sha256(
            "|".join(
                (
                    f"{type(error).__module__}.{type(error).__qualname__}",
                    f"{type(root).__module__}.{type(root).__qualname__}",
                    message,
                    root_message,
                )
            ).encode("utf-8", errors="replace")
        ).hexdigest(),
    }
    diagnostic_metrics = getattr(error, "diagnostic_metrics", None)
    if isinstance(diagnostic_metrics, dict):
        record["diagnostic_metrics"] = json.dumps(
            diagnostic_metrics,
            sort_keys=True,
        )
    return record


def replay(run_dir: Path, package_dir: Path) -> dict[str, Any]:
    evidence_path = run_dir / "mission_evidence.json"
    cycles_path = run_dir / "model-navigation-cycles.jsonl"
    snapshots_path = run_dir / "model-navigation-snapshots.jsonl"
    calls_path = run_dir / "model-navigation-model-calls.jsonl"
    frames_dir = run_dir / "model-navigation-frames"
    for path in (evidence_path, cycles_path, snapshots_path, calls_path, frames_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    evidence = _json(evidence_path)
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("mission evidence has no artifact bindings")
    expected_hashes = {
        cycles_path: artifacts.get("model_navigation_cycles_sha256"),
        snapshots_path: artifacts.get("model_navigation_snapshots_sha256"),
        calls_path: artifacts.get("model_navigation_calls_sha256"),
    }
    for path, expected in expected_hashes.items():
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise RuntimeError(f"recorded artifact hash mismatch: {path.name}")
    source_calls = {
        str(record.get("call_id")): record
        for record in _jsonl(calls_path)
        if isinstance(record.get("call_id"), str)
    }

    snapshots: dict[str, dict[str, Any]] = {}
    for record in _jsonl(snapshots_path):
        snapshot = record.get("snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("navigation snapshot record is missing its snapshot")
        observed = snapshot.get("snapshot_sha256")
        material = dict(snapshot)
        material.pop("snapshot_sha256", None)
        if not isinstance(observed, str) or observed != sha256_json(material):
            raise RuntimeError("navigation snapshot content hash mismatch")
        snapshots[observed] = snapshot

    required_frame_hashes = {
        str(visual[0]["sha256"])
        for snapshot in snapshots.values()
        if isinstance((visual := snapshot.get("visual_evidence")), list)
        and visual
        and isinstance(visual[0], dict)
        and isinstance(visual[0].get("sha256"), str)
    }
    frames: dict[str, Path] = {}
    for path in sorted(frames_dir.glob("*.png")):
        digest = _sha256(path)
        if digest in required_frame_hashes:
            # Several sensor ticks may legitimately persist byte-identical
            # frames. The content hash, not the timestamped filename, is the
            # replay identity, so one deterministic representative is enough.
            frames.setdefault(digest, path)
    missing_frames = required_frame_hashes - frames.keys()
    if missing_frames:
        raise RuntimeError(f"recorded visual frames are missing: {len(missing_frames)}")

    replay_cycles = [
        cycle
        for cycle in _jsonl(cycles_path)
        if cycle.get("model_call_id") is not None
        or cycle.get("failure_stage") == "model-invocation"
    ]
    replay_cycles.sort(
        key=lambda item: (int(item.get("sequence", 0)), int(item.get("recorded_at_unix_ms", 0)))
    )
    package = load_local_policy_package(package_dir)
    port = LocalPolicyPort(package)
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    selected_roles: Counter[str] = Counter()
    reproduced_output_count = 0
    nonlocal_source_output_count = 0
    width = package.manifest.visual_width
    height = package.manifest.visual_height
    if width is None or height is None:
        raise RuntimeError("replayed visual package has no model frame dimensions")
    frame_payload_cache: dict[str, tuple[bytes, bytes]] = {}
    try:
        for cycle in replay_cycles:
            snapshot_sha256 = cycle.get("snapshot_sha256")
            snapshot = snapshots.get(str(snapshot_sha256))
            if snapshot is None:
                raise RuntimeError("invocation cycle refers to a missing snapshot")
            visual = snapshot.get("visual_evidence")
            if (
                not isinstance(visual, list)
                or len(visual) != 1
                or not isinstance(visual[0], dict)
            ):
                raise RuntimeError("local visual invocation lacks one bound frame")
            frame_sha256 = str(visual[0].get("sha256", ""))
            frame_path = frames.get(frame_sha256)
            if frame_path is None:
                raise RuntimeError("local visual invocation frame is unavailable")
            payloads = frame_payload_cache.get(frame_sha256)
            if payloads is None:
                import io

                from PIL import Image

                content = frame_path.read_bytes()
                with Image.open(io.BytesIO(content)) as image:
                    model_rgb = image.convert("RGB").resize((width, height)).tobytes()
                payloads = content, model_rgb
                frame_payload_cache[frame_sha256] = payloads
            content, model_rgb = payloads
            media = [
                {
                    "kind": "image-file",
                    "path": str(frame_path),
                    "content_bytes": content,
                    "content_sha256": frame_sha256,
                    "model_rgb_bytes": model_rgb,
                    "model_rgb_sha256": hashlib.sha256(model_rgb).hexdigest(),
                    "model_rgb_width": width,
                    "model_rgb_height": height,
                }
            ]
            port.prime_multimodal(media)
            started = time.perf_counter()
            try:
                result = port.call(
                    role="local_navigation_advisor",
                    output_type=TextNavigationDecision,
                    instructions="",
                    input_artifact={"text_navigation_snapshot": snapshot},
                    multimodal=media,
                    maximum_physical_attempts=1,
                )
                trace = result.record.local_expert_trace
                if trace is not None:
                    selected_roles[trace.selected_navigation_role] += 1
                source_call_id = cycle.get("model_call_id")
                if source_call_id is not None:
                    source_call = source_calls.get(str(source_call_id))
                    if source_call is None:
                        raise RuntimeError("source model call record is missing")
                    if source_call.get("provider") == "local-policy":
                        if result.record.output_sha256 != source_call.get("output_sha256"):
                            raise RuntimeError("replayed local policy output changed")
                        reproduced_output_count += 1
                    else:
                        # The evidence stream may contain a bounded cloud
                        # fallback response. Replaying the local package is
                        # still useful for proving that the same snapshot was
                        # locally executable, but its output hash must never be
                        # compared with a different model/provider.
                        nonlocal_source_output_count += 1
            except Exception as error:
                failures.append(
                    {
                        "sequence": int(cycle.get("sequence", 0)),
                        "snapshot_sha256": snapshot_sha256,
                        "source_failed": (
                            cycle.get("failure_stage") == "model-invocation"
                        ),
                        **_failure_record(error),
                    }
                )
            latencies.append((time.perf_counter() - started) * 1_000.0)
    finally:
        port.close()

    source_failure_count = sum(
        cycle.get("failure_stage") == "model-invocation" for cycle in replay_cycles
    )
    return {
        "schema_version": "dronedream.local-policy-evidence-replay.v1",
        "status": "replayed",
        "qualification_granted": False,
        "source_status": evidence.get("status"),
        "source_mission_evidence_sha256": _sha256(evidence_path),
        "source_cycles_sha256": _sha256(cycles_path),
        "source_snapshots_sha256": _sha256(snapshots_path),
        "source_calls_sha256": _sha256(calls_path),
        "package_sha256": package.package_sha256,
        "invocation_count": len(replay_cycles),
        "source_invocation_failure_count": source_failure_count,
        "source_success_output_count": len(source_calls),
        "source_local_success_output_count": sum(
            item.get("provider") == "local-policy" for item in source_calls.values()
        ),
        "nonlocal_source_output_count": nonlocal_source_output_count,
        "exactly_reproduced_source_output_count": reproduced_output_count,
        "replay_invocation_failure_count": len(failures),
        "source_failures_reproduced": sum(item["source_failed"] for item in failures),
        "selected_navigation_role_counts": dict(sorted(selected_roles.items())),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "maximum": max(latencies, default=None),
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = replay(args.run_dir.resolve(), args.package_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["replay_invocation_failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
