"""Owned map-fusion experiments for source-matched SITL training, not production qualification."""

import asyncio
import json
import os
import re
import time
import traceback
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from dronedream_plugin_sdk.protocol import decode_json

from .plugin_files import check_plain_plugin_path, hash_plugin_file, read_plugin_file
from .runtime_control_io import publish_runtime_json


# 功能：为独占的仿真实验绑定实际地图、源钟和输出位置，不使用开发机硬编码路径。
# 输入：run_dir：新实验 simulation 目录；world_sdf、semantic：原始地图资产。
# 输出：新的输入文件和契约，明确不授予定位或飞行资格。
def prepare_map_fusion_experiment(run_dir, world_sdf, semantic):
    run = Path(run_dir).absolute()
    check_plain_plugin_path(run)
    if run.name != "simulation":
        raise ValueError("FUSION_EXPERIMENT_DIRECTORY_INVALID")
    world_sdf, semantic = Path(world_sdf), Path(semantic)
    inputs = dict(
        run_name="map-fusion-" + uuid4().hex,
        run_dir=str(run),
        world_sdf=str(Path(world_sdf).resolve(strict=True)),
        semantic_path=str(Path(semantic).resolve(strict=True)),
        world_sha256=hash_plugin_file(world_sdf, limit=64 * 1024**2),
        semantic_sha256=hash_plugin_file(semantic, limit=64 * 1024**2),
        experimental_only=True,
        qualification_granted=False,
    )
    path = run.parent / "fusion-inputs.json"
    publish_runtime_json(path, inputs, replace_existing=False)
    return path, inputs


# 功能：重新核对来源、运行所有权和地图身份，拒绝跨运行使用旧输入及被修改的地图。
# 输入：path：本次实验生成的普通输入文件。输出：已验证配置，不读取仿真真值。
def read_map_fusion_experiment(path):
    path = Path(path).absolute()
    inputs = decode_json(read_plugin_file(path, limit=65536), limit=65536)
    if (
        path.name != "fusion-inputs.json"
        or type(inputs) is not dict
        or set(inputs)
        != {
            "run_name",
            "run_dir",
            "world_sdf",
            "semantic_path",
            "world_sha256",
            "semantic_sha256",
            "experimental_only",
            "qualification_granted",
        }
        or type(inputs["run_name"]) is not str
        or re.fullmatch(r"map-fusion-[0-9a-f]{32}", inputs["run_name"]) is None
        or inputs["experimental_only"] is not True
        or inputs["qualification_granted"] is not False
        or any(
            type(inputs[field]) is not str or not inputs[field]
            for field in (
                "run_dir",
                "world_sdf",
                "semantic_path",
                "world_sha256",
                "semantic_sha256",
            )
        )
        or any(
            not Path(inputs[field]).is_absolute()
            for field in ("run_dir", "world_sdf", "semantic_path")
        )
        or Path(inputs["run_dir"]).resolve() != (path.parent / "simulation").resolve()
    ):
        raise ValueError("FUSION_INPUTS_RUN_BINDING_INVALID")
    check_plain_plugin_path(Path(inputs["run_dir"]))
    for field, digest in [("world_sdf", "world_sha256"), ("semantic_path", "semantic_sha256")]:
        if hash_plugin_file(Path(inputs[field]), limit=64 * 1024**2) != inputs[digest]:
            raise ValueError("FUSION_MAP_ASSET_CHANGED")
    return inputs


class MapFusionExperimentMixin:
    # 功能：在独占物理仿真中启动持续地图观测，飞行仍经过原有预检、净空和小模型权限约束。
    # 输入：当前隔离飞控连接；输出：已连接客户端及后台测量监督任务，不直接解锁。
    async def connect(self, connection):
        # 先验证输入，避免后台任务在建立就绪事件前失败而静默等待整个超时。
        configured = os.environ.get("DRONEDREAM_MAP_FUSION_INPUTS")
        if not configured:
            raise ValueError("FUSION_EXPERIMENT_INPUTS_MISSING")
        self._fusion_inputs_path = Path(configured).absolute()
        read_map_fusion_experiment(self._fusion_inputs_path)
        # 独占实验执行器也有定位计算与遥测回调，不沿用默认五毫秒线程量子。
        from dronedream_agent_core.runtime_scheduling import configure_sensor_thread_handoff

        configure_sensor_thread_handoff()
        await super().connect(connection)
        self._fusion_stop, self._fusion_ready = asyncio.Event(), asyncio.Event()
        self._fusion_task = asyncio.create_task(self._fusion_loop())

    # 功能：不因原生本地位置就绪就提前进入模式，先等待真实地图回读。
    # 输入：等待上限；输出：真实本地健康，仍不冒充 armable。
    async def wait_until_local_position_ready(self, timeout_seconds):
        async with asyncio.timeout(timeout_seconds):
            await self._fusion_ready.wait()
            if self._fusion_task.done():
                await self._fusion_task
                raise RuntimeError("FUSION_EXPERIMENT_STOPPED_BEFORE_PREFLIGHT")
            return await super().wait_until_local_position_ready(timeout_seconds)

    # 功能：在预先分组的采集场景运行地图融合，原始源时间、噪声和失败全部保留。
    # 输入：本次明确绑定的地图与实验参数；输出：逐帧记录和失败时运行停止请求。
    async def _fusion_loop(self):
        from dronedream_agent_core.isolated_map_measurement import IsolatedMapMeasurementSource
        from dronedream_agent_core.live_map_measurement import MapNoiseBounds
        from dronedream_agent_core.local_packet_channel import LatestPacketReceiver
        from dronedream_agent_core.localization_source_channel import (
            LOCALIZATION_SOURCE_CONTRACT,
            decode_localization_source,
        )
        from dronedream_agent_core.optical_map import compile_optical_map
        from dronedream_agent_core.runtime_control_io import publish_runtime_json
        from dronedream_agent_core.sitl_map_vision_transport import run_sitl_map_vision_transport

        inputs_path = self._fusion_inputs_path
        root = inputs_path.absolute().parent
        run = root / "simulation"
        output = root / "fusion-stream.json"
        report = {
            "qualification_granted": False,
            "formal_training_additions": 0,
            "experimental_moving_sitl_only": True,
            "transport": {},
        }
        receiver = source = None
        try:
            inputs = read_map_fusion_experiment(inputs_path)
            run_name = inputs["run_name"]
            world_sdf = Path(inputs["world_sdf"])
            map_hash = inputs["semantic_sha256"]
            index, _, optical = await asyncio.to_thread(
                compile_optical_map, world_sdf, expected_world_sha256=inputs["world_sha256"]
            )
            report["optical"] = optical
            identity_path = run / "runtime-state/px4-identity-telemetry.json"
            async with asyncio.timeout(10.0):
                # 原子替换的启动快照可能在 exists 与 open 之间变化；
                # 只重试尚不可读，损坏 JSON 仍立即失败，不用旧快照代替。
                while True:
                    try:
                        with identity_path.open("rb") as identity_stream:
                            identity_bytes = identity_stream.read(262145)
                        if len(identity_bytes) > 262144:
                            raise ValueError("FUSION_IDENTITY_EXCEEDS_BOUND")
                        from dronedream_plugin_sdk.protocol import decode_json

                        identity = decode_json(identity_bytes, limit=262144)
                        break
                    except FileNotFoundError:
                        await asyncio.sleep(0.01)
            # Deliberately conservative experiment bounds, not fitted against
            # this run and not a qualified covariance for the user's narrow route.
            # Two 4-ms SITL ticks cover bounded quantization/transport skew.
            # Every measurement includes the resulting motion uncertainty in
            # covariance; readback outside this bound is still rejected.
            noise = MapNoiseBounds(0.0015, 0.03, 0.01, 0.6, 2.0, 1.0, 2.0,
                                   transport_clock_uncertainty_seconds=.008)
            report["noise"] = noise.__dict__
            report["measure_timing"] = []
            sample_count = 0
            failed_sample_count = 0
            failed_sample_types = {}

            class TimedSource(IsolatedMapMeasurementSource):
                # 功能：只读记录真实源年龄及计算阶段耗时，不能改写输入钟或放宽有效期。
                # 输入：原始观测和真实时钟；输出：原结果或原异常，统计固定容量。
                def measure(self, record, *, source_now_ns, monotonic_now):
                    nonlocal sample_count, failed_sample_count
                    stamp = record["native_odometry_snapshot"]["source_alignment"][
                        "image_timestamp_ns"
                    ]
                    start_clock, started = source_now_ns(), time.monotonic()
                    entry = {
                        "source_ns": stamp,
                        "input_age_ms": (start_clock - stamp) / 1e6
                        if start_clock is not None
                        else None,
                        "producer_timing": record.get("source_clock"),
                    }
                    try:
                        result = super().measure(
                            record, source_now_ns=source_now_ns, monotonic_now=monotonic_now
                        )
                        entry["result"] = "measured"
                        return result
                    except Exception as error:
                        entry["result"] = str(error)
                        if ((str(error).startswith("LIVE_MAP_NO_USABLE_GEOMETRY_")
                                or str(error) == "LIVE_MAP_PARTIAL_CONSTRAINT_ONLY")
                                and failed_sample_count < 32
                                and failed_sample_types.get(str(error), 0) < 8):
                            # 起步前八帧通常成功，不能代替后来失败帧的诊断现场。
                            # 按失败类别最多八帧、总计三十二帧，避免前八个暂态吞掉后来的根因。
                            # 仅原始部署观测；不增加正式训练计数，不读取真值。
                            publish_runtime_json(
                                root / f"geometry-failure-{failed_sample_count:02d}.json",
                                record, replace_existing=False, maximum_bytes=262144,
                            )
                            failed_sample_count += 1
                            failed_sample_types[str(error)] = failed_sample_types.get(str(error), 0)+1
                        raise
                    finally:
                        entry["wall_ms"] = (time.monotonic() - started) * 1000
                        if len(report["measure_timing"]) < 2000:
                            report["measure_timing"].append(entry)
                        if sample_count < 8:
                            # 独立诊断保留原始输入供离线核对退化方向；不刷新源钟或用于训练。
                            publish_runtime_json(
                                root / f"geometry-input-{sample_count:02d}.json",
                                record,
                                replace_existing=False,
                                maximum_bytes=262144,
                            )
                            sample_count += 1

            source = await asyncio.to_thread(
                TimedSource,
                index=index,
                map_sha256=map_hash,
                binding=identity["map_frame_binding"],
                binding_sha256=identity["map_frame_binding_sha256"],
                clock_domain="px4-gz-sitl:" + run_name,
                noise=noise,
            )
            receiver = LatestPacketReceiver(
                run / "runtime-state/localization-source.json",
                contract=LOCALIZATION_SOURCE_CONTRACT,
            )
            await self.verify_disarmed_before_preflight(timeout_seconds=3.0)
            parameters = [
                ("EKF2_EV_NOISE_MD", "int", 0),
                ("EKF2_EVP_NOISE", "float", 0.03),
                ("EKF2_EVA_NOISE", "float", 0.01),
                ("EKF2_GPS_CTRL", "int", 4),
                # 该输入是固定起点原点的局部 NED 高度，不是海拔高度。
                # 已初始化全球原点后切到视觉高度参考会触发绝对高度重置，
                # 使局部 z 与固定地图绑定分离。保留气压高度基准，视觉只作位置辅助，
                # 由固件偏置估计器衔接；仍须实际回读/独立结果验证，不能因此获飞行资格。
                ("EKF2_BARO_CTRL", "int", 1),
                ("EKF2_HGT_REF", "int", 0),
                ("EKF2_EV_CTRL", "int", 3),
            ]
            report["parameters"] = []
            system = self._require_system()
            for name, kind, value in parameters:
                before = await asyncio.wait_for(
                    getattr(system.param, "get_param_" + kind)(name), 3.0
                )
                await asyncio.wait_for(getattr(system.param, "set_param_" + kind)(name, value), 3.0)
                applied = await asyncio.wait_for(
                    getattr(system.param, "get_param_" + kind)(name), 3.0
                )
                report["parameters"].append(dict(name=name, before=before, applied=applied))
                if abs(applied - value) > 1e-6:
                    raise RuntimeError("MOVING_FUSION_PARAMETER_READBACK_FAILED")

            from .px4_visual_readback import Px4VisualReadback
            visual_readback = Px4VisualReadback()

            # 功能：视觉回读直连同一固件守护进程，其余一次性链路设置仍用原生 CLI。
            # 输入：固定命令字段；输出：有界真实回执，仍由上层逐帧核对时间、位姿和重置计数。
            async def command(binary, *args):
                if binary == 'listener' and args == ('vehicle_visual_odometry', '-n', '1'):
                    result = await visual_readback.read()
                    report['readback_transport'] = dict(kind='px4-posix-daemon-direct-readonly',
                        peer_pid=visual_readback.peer_pid, queries=visual_readback.verified_queries,
                        command_processes_spawned=0, last_response=result[:2048])
                    return result
                process = await asyncio.create_subprocess_exec(
                    str(Path(os.environ.get("DRONEDREAM_MAP_FUSION_PX4_ROOT", "/opt/PX4-Autopilot"))
                        / "build/px4_sitl_default/bin" / ("px4-" + binary)),
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    data, _ = await asyncio.wait_for(process.communicate(), 4.0)
                except BaseException:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
                    raise
                if process.returncode:
                    raise RuntimeError("MOVING_FUSION_COMMAND_FAILED")
                return data.decode(errors="replace")[:32768]

            # 功能：读取实时认证来源，不读历史采样；输入：无；输出：真实新帧或 None。
            def latest():
                payload = receiver.read_latest()
                return decode_localization_source(payload) if payload is not None else None

            with (root / "fusion-measurements.jsonl").open("x", encoding="utf-8") as stream:
                # 功能：保存每次真实发送及回读，已完成八次才允许进入后续预检。
                # 输入：真实测量与回读；输出：独立记录，不读仿真真值。
                def verified(measurement, readback):
                    stream.write(
                        json.dumps(
                            dict(
                                measurement=measurement,
                                readback=readback,
                                received_monotonic=time.monotonic(),
                            ),
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    stream.flush()
                    if report["transport"]["readbacks"] >= 8:
                        self._fusion_ready.set()

                await run_sitl_map_vision_transport(
                    source=source,
                    read_latest=latest,
                    command=command,
                    clock_domain=source.clock_domain,
                    stop=self._fusion_stop,
                    evidence=report["transport"],
                    on_verified=verified,
                    optional_after_initialization=(
                        os.environ.get("DRONEDREAM_INDEPENDENT_ROUTE_CONTROL") == "1"),
                )
        except asyncio.CancelledError:
            report["stopped_by_owner"] = True
            raise
        except Exception as error:
            report["issue"] = type(error).__name__ + ":" + str(error)[:300]
            report["traceback"] = traceback.format_exc()
            # 先保存故障现场；所有者收到停止请求后可能立刻取消本任务。
            publish_runtime_json(output, report, maximum_bytes=16 * 1024 * 1024)
            abort = run / "live_abort.request.json"
            # 保留操作员或其他安全监督者先提交的终止原因。
            with suppress(FileExistsError):
                publish_runtime_json(
                    abort,
                    {"reason": report["issue"], "world_paused": False},
                    replace_existing=False,
                )
            raise
        finally:
            self._fusion_ready.set()
            try:
                if receiver is not None:
                    receiver.close()
                if source is not None:
                    from dronedream_agent_core.live_map_measurement_stream import _drainable_compute

                    report["geometry_process"] = await _drainable_compute(
                        asyncio.get_running_loop().run_in_executor(None, source.close)
                    )
            finally:
                publish_runtime_json(output, report, maximum_bytes=16 * 1024 * 1024)

    # 功能：由执行器完成降落/退出后回收测量线程和连接；输入：无；输出：无后台残留。
    async def close(self):
        stop = getattr(self, "_fusion_stop", None)
        task = getattr(self, "_fusion_task", None)
        if stop is not None:
            stop.set()
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await super().close()
