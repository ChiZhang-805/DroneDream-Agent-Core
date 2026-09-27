"""Bounded native capture jobs with immutable plans and verified restart receipts."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_files import (
    check_plain_plugin_path,
    hash_plugin_file,
    portable_plugin_path,
    read_plugin_file,
)
from dronedream_agent_core.runtime_control_io import publish_runtime_json, read_runtime_object


# 功能：
#   核对每个场景源文件，冻结目录、视角和来源哈希，拒绝改名覆盖或未经核对的新文件。
# 输入：
#   suites、only：明确套件目录和可选布局白名单。
# 输出：
#   jobs：可执行的固定场景列表。
def checked_jobs(suites, only):
    from capture_labelled_vision_views import checked_views

    jobs, names = [], set()
    for suite in suites:
        plan = read_runtime_object(suite / "suite-plan.json", maximum_bytes=8 * 1024**2)
        if (plan.get("schema") != "dronedream.vision-suite-plan.v1"
                or plan.get("complete") is not True or type(plan.get("layouts")) is not list
                or not 1 <= len(plan["layouts"]) <= 100):
            raise ValueError("VISION_CAPTURE_SUITE_INVALID")
        for layout in plan["layouts"]:
            name = layout["directory"]
            if str(portable_plugin_path(name)) != name or "/" in name or "\\" in name:
                raise ValueError("VISION_CAPTURE_LAYOUT_NAME_INVALID")
            if name in names:
                raise ValueError("VISION_CAPTURE_DUPLICATE_LAYOUT")
            names.add(name)
            if only and name not in only:
                continue
            root = suite / name
            hashes = layout.get("files_sha256")
            if type(hashes) is not dict or not 3 <= len(hashes) <= 64:
                raise ValueError("VISION_CAPTURE_LAYOUT_HASHES_INVALID")
            for relative, digest in hashes.items():
                path = root / portable_plugin_path(relative)
                if hash_plugin_file(path, limit=32 * 1024**2) != digest:
                    raise ValueError("VISION_CAPTURE_LAYOUT_SOURCE_CHANGED")
            if not {"world.sdf", "labels.json", "views.json"} <= hashes.keys():
                raise ValueError("VISION_CAPTURE_LAYOUT_FILES_MISSING")
            views = read_runtime_object(root / "views.json", maximum_bytes=8 * 1024**2)
            normalized = [view.model_dump(mode="json") for view in checked_views(views)]
            if layout["views"] != len(normalized):
                raise ValueError("VISION_CAPTURE_PLAN_COUNT_MISMATCH")
            jobs.append({"root": root, "name": name, "hashes": hashes,
                         "labels_sha256": sha256_json(read_runtime_object(root / "labels.json",
                            maximum_bytes=8 * 1024**2)),
                         "plan_sha256": sha256_json(normalized), "views": layout["views"]})
    if only and set(only) - names:
        raise ValueError("VISION_CAPTURE_UNKNOWN_LAYOUT")
    if not jobs or len(jobs) > 100:
        raise ValueError("VISION_CAPTURE_JOB_BUDGET_INVALID")
    return jobs


# 功能：
#   读取当前采集器实际记录的实现摘要，旧采集不能因世界和相机相同就混入新协议。
# 输入：
#   无；文件范围严格等同于当前采集器写入回执的七个实现文件。
# 输出：
#   hashes：逐实现文件的实际内容摘要。
def current_capture_implementation():
    import capture_labelled_vision_views as capture

    paths = [Path(capture.__file__),
             Path(capture.__file__).with_name("build_local_vision_dataset.py")]
    for component in (capture.RenderPairBuffer, capture.build_labelled_render_world,
                      capture.corrupt_rendered_pair, capture.LocalVisionTrainingSample,
                      capture.FixturePoseRequests):
        paths.append(Path(sys.modules[component.__module__].__file__))
    hashes = {path.name: hash_plugin_file(path, limit=2 * 1024**2) for path in paths}
    return hashes


# 功能：
#   只复用完整且来源未改变的采集结果，进程异常遗留目录不得作为完成结果。
# 输入：
#   root、job、camera_hash：既有采集目录、冻结输入和相机摘要。
# 输出：
#   receipt：校验后的采集回执。
def verify_finished(root, job, camera_hash):
    from assemble_vision_dataset import inspect_capture

    # 原生画面可能重复，采集回执不宣称已可训练；正式汇总仍必须逐图去重。
    _, receipt, _, _ = inspect_capture(root, allow_duplicates=True)
    if (receipt["world_sha256"] != job["hashes"]["world.sdf"]
            or receipt["plan_sha256"] != job["plan_sha256"]
            or receipt.get("labels_sha256") != job["labels_sha256"]
            or receipt["camera_sha256"] != camera_hash
            or receipt["view_count"] != job["views"]):
        raise ValueError("VISION_CAPTURE_RESUME_SOURCE_MISMATCH")
    if receipt.get("capture_implementation_sha256") != current_capture_implementation():
        raise ValueError("VISION_CAPTURE_RESUME_IMPLEMENTATION_MISMATCH")
    for relative, digest in receipt.get("render_resource_sha256", {}).items():
        if job["hashes"].get(relative) != digest:
            raise ValueError("VISION_CAPTURE_RESUME_RESOURCE_MISMATCH")
    return receipt


# 功能：
#   执行一个独立分区的原生采集；取消和超时先给采集器机会回收其 Gazebo 子进程。
# 输入：
#   job、output、camera、camera_hash、resume、stop：冻结输入、输出及取消信号。
# 输出：
#   record：通过完整性检查的采集目录和实测帧数。
def capture_job(job, output, camera, camera_hash, resume, stop):
    for relative, digest in job["hashes"].items():
        if hash_plugin_file(job["root"] / relative, limit=32 * 1024**2) != digest:
            raise ValueError("VISION_CAPTURE_PENDING_SOURCE_CHANGED")
    root = output / job["name"]
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        receipt = verify_finished(root, job, camera_hash)
        return {"capture": str(root), "frames": receipt["view_count"], "reused": True}
    if stop.is_set():
        raise RuntimeError("VISION_CAPTURE_CANCELLED")
    args = [sys.executable, str(Path(__file__).with_name("capture_labelled_vision_views.py"))]
    for key in ("world", "labels", "views"):
        suffix = ".sdf" if key == "world" else ".json"
        args.extend(["--" + key, str(job["root"] / (key + suffix))])
    args.extend(["--camera", str(camera), "--output", str(root)])
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = env["LP_NUM_THREADS"] = "2"
    started = time.monotonic()
    with (output / (job["name"] + ".log")).open("xb") as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            while process.poll() is None:
                if stop.wait(0.25) or time.monotonic() - started > 1200:
                    raise TimeoutError("VISION_CAPTURE_CANCELLED_OR_JOB_DEADLINE")
            if process.returncode:
                raise RuntimeError("VISION_CAPTURE_CHILD_FAILED:" + job["name"])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    receipt = verify_finished(root, job, camera_hash)
    return {"capture": str(root), "frames": receipt["view_count"], "reused": False,
            "wall_seconds": round(time.monotonic() - started, 3)}


# 功能：
#   在有界并行度下采集预先固定的训练场景，逐批落盘，并在全部完成后发布汇总回执。
# 输入：
#   命令行参数：一个或多个源套件、相机、输出、并发数、可选白名单与续传标记。
# 输出：
#   exit_code：全部指定布局采集校验成功为零，失败时保留已采制品和错误现场。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise ValueError("VISION_CAPTURE_REQUIRES_LINUX")
    jobs = checked_jobs(args.suite, args.only)
    camera_hash = hashlib.sha256(read_plugin_file(args.camera, limit=1024**2)).hexdigest()
    output = args.output.absolute()
    check_plain_plugin_path(output)
    output.mkdir(parents=True, exist_ok=args.resume)
    stop, records, pending = Event(), [], {}
    run_id = uuid.uuid4().hex[:12]
    report_path = output / ("suite-run-" + run_id + ".json")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            iterator = iter(jobs)
            for _ in range(args.workers):
                job = next(iterator, None)
                if job is not None:
                    pending[executor.submit(capture_job, job, output, args.camera,
                                            camera_hash, args.resume, stop)] = job
            while pending:
                try:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=1)
                    for future in finished:
                        job = pending.pop(future)
                        record = future.result()
                        records.append(record)
                        print(json.dumps({"completed": len(records), "total": len(jobs),
                                          "name": job["name"], **record}), flush=True)
                        next_job = next(iterator, None)
                        if next_job is not None:
                            pending[executor.submit(capture_job, next_job, output, args.camera,
                                                    camera_hash, args.resume, stop)] = next_job
                except BaseException:
                    stop.set()
                    raise
    except BaseException as error:
        stop.set()
        publish_runtime_json(report_path, {"complete": False, "captures": records,
            "error": type(error).__name__ + ":" + str(error)[:400]}, replace_existing=False)
        raise
    publish_runtime_json(report_path, {"schema": "dronedream.native-vision-suite-capture.v1",
        "complete": True, "captures": records, "frames": sum(r["frames"] for r in records),
        "physical_flight_evidence": False, "ready_for_training": False}, replace_existing=False)
    print(str(report_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
