"""Collect bounded native observations for offline risk probes, not learned flight evidence."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.plugin_files import read_plugin_file
from dronedream_agent_core.training.evidence_publication import write_evidence_object
from dronedream_agent_core.training.flight_environment import ControlPreparationExpired, PilotAction
from dronedream_agent_core.training.px4_environment import (
    Px4GazeboTrainingEnvironment,
    Px4TrainingConfig,
)
from dronedream_plugin_sdk.protocol import decode_json


# 功能：
#   明确限定短时仿真观测采集，不允许配置文件将采集转换成无限时任务或其他控制模式。
# 输入：
#   config：固定资产身份、围栏和传感器运行库的训练端口配置。
#   duration_seconds：起飞就绪后的采集秒数。
# 输出：
#   duration：已验证的有限持续秒数。
def validate_campaign(config, duration_seconds):
    if (type(duration_seconds) not in (int, float) or not math.isfinite(duration_seconds)
            or not 3 <= duration_seconds <= 30):
        raise ValueError("RISK_CONTEXT_DURATION_INVALID")
    if (config.initial_collection_mode != "stream-imitation"
            or config.episode_steps > 240
            or config.expert_role not in ("local-navigation-policy", "precision-maneuver-policy")):
        raise ValueError("RISK_CONTEXT_COLLECTION_MODE_INVALID")
    if (type(config.speed_limit_mps) not in (int, float)
            or not 0.1 <= config.speed_limit_mps <= 1.2):
        raise ValueError("RISK_CONTEXT_RUNNER_SPEED_LIMIT_INVALID")
    duration = float(duration_seconds)
    return duration


# 功能：
#   生成小幅转向与有界低速高度调节；危险方向探针只在落地后离线计算。
# 输入：
#   elapsed_seconds：从采集开始计量的有限非负秒数。
#   altitude_m、vertical_limit_mps：已接纳机载观测的高度及当前四轴垂直物理尺度。
#   target_altitude_m：明确课程路线的最后高度，不修改原始起飞绑定。
#   yaw_profile：小幅往复或限速单向扫描，不改变水平零速度约束。
# 输出：
#   action：水平轴为零，垂直速度不超过0.2米每秒的小幅连续控制提案。
def collection_action(elapsed_seconds, altitude_m, vertical_limit_mps, target_altitude_m, yaw_profile="alternating"):
    if (type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0):
        raise ValueError("RISK_CONTEXT_ELAPSED_INVALID")
    if (any(type(v) not in (int, float) or not math.isfinite(v)
            for v in (altitude_m, vertical_limit_mps, target_altitude_m))
            or not 0 < vertical_limit_mps <= 20 or not 0 < target_altitude_m <= 100):
        raise ValueError("RISK_CONTEXT_ALTITUDE_INPUT_INVALID")
    if yaw_profile not in ("alternating", "sweep"):
        raise ValueError("RISK_CONTEXT_YAW_PROFILE_INVALID")
    # 扫描课程增加真实机体朝向覆盖，不旋转图片伪造机载观测，也不执行危险平移。
    yaw = 0.5 if yaw_profile == "sweep" else (0.2 if int(elapsed_seconds / 3) % 2 == 0 else -0.2)
    speed_cap = min(0.2, vertical_limit_mps * 0.5)
    up = max(-speed_cap, min(speed_cap, 0.8 * (target_altitude_m - altitude_m)))
    action = PilotAction(mode="pilot-control", axes=[0.0, 0.0, up / vertical_limit_mps, yaw])
    return action


# 功能：
#   1. 通过现有安全端口采集真实传感器及执行回执，不改动租约、避障和起飞门槛。
#   2. 无论采集成功与否均请求安全关闭；确认落地后才关联执行记录。
# 输入：
#   config：显式仿真资产和新输出目录的配置。
#   duration_seconds：最多三十秒的空中采集时间。
#   yaw_profile：明确绑定到采集身份的转向课程。
# 输出：
#   report：原始回合路径、实际接受数量及错误，始终不授予模型飞行资格。
def collect(config, duration_seconds, yaw_profile="alternating"):
    duration = validate_campaign(config, duration_seconds)
    route_bytes = read_plugin_file(config.route, limit=4 * 1024**2)
    if hashlib.sha256(route_bytes).hexdigest() != config.asset_sha256["route"]:
        raise ValueError("RISK_CONTEXT_ROUTE_CHANGED")
    route = decode_json(route_bytes, limit=4 * 1024**2)
    target_altitude = route["positions_m"][-1]["z"]
    # 目标取自同一条随后由端口重新核对摘要和几何的路线，不能在起飞后另造课程点。
    collection_action(0, target_altitude, config.speed_limit_mps, target_altitude, yaw_profile)
    identity = sha256_json({"purpose": "scripted-native-risk-context",
                           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "config": config.model_dump(mode="json"), "seconds": duration,
                           "yaw_profile": yaw_profile})
    env = Px4GazeboTrainingEnvironment(config)
    env.bind_policy_identity(identity)
    report = {"purpose": "scripted-native-risk-context", "qualified_for_flight": False,
              "behavior_cloning_dataset": False, "simulation_only": True,
              "policy_sha256": identity, "proposals": 0, "expired": 0,
              "safely_closed": False, "error": None, "yaw_profile": yaw_profile,
              "observation_wait_timeouts": 0}
    try:
        observation = env.reset(seed=2907)
        started = time.monotonic()
        deadline = started + duration
        while report["proposals"] < config.episode_steps and time.monotonic() < deadline:
            try:
                position = env.current_position_m()
                limits = observation.sample.pilot_control_limits
                env.submit_stream_action(collection_action(
                    time.monotonic() - started, position.z,
                    limits.vertical_speed_mps, target_altitude, yaw_profile))
                report["proposals"] += 1
            except ControlPreparationExpired:
                report["expired"] += 1
            if time.monotonic() >= deadline:
                break
            observation = await_fresh_observation(env, deadline=deadline, report=report)
            if observation is None:
                break
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            env.close()
            report["safely_closed"] = True
            if report["proposals"]:
                visits = env.finalize_stream_captures()
                report["accepted"] = len(visits)
                report["safety_interventions"] = sum(v.safety_intervened for v in visits)
            else:
                report["accepted"] = 0
        except Exception as error:
            report["close_error"] = f"{type(error).__name__}: {error}"
        report["episode_path"] = str(env.episode_path) if env.episode_path else None
        write_evidence_object(config.output_root / "collection-report.json", report)
    return report


# 功能：在原采集总截止内等待新观测，短暂无可用帧时只等待，不重放动作或延长控制租约。
# 输入：env：原安全端口；deadline：原三十秒以内的截止；report：保留超时次数的回执。
# 输出：新观测或总时限耗尽时的 None；其他错误原样上抛，最多十六次有界等待。
def await_fresh_observation(env, *, deadline, report):
    for _ in range(16):
        now = time.monotonic()
        if now >= deadline:
            return None
        try:
            return env.next_stream_observation(deadline=min(deadline, now + 1.9))
        except TimeoutError as error:
            if str(error) != "PX4_TRAINING_NEXT_OBSERVATION_TIMEOUT":
                raise
            report["observation_wait_timeouts"] += 1
    raise TimeoutError("RISK_CONTEXT_WAIT_RETRY_LIMIT")


# 功能：
#   读取有界严格配置并启动采集，失败回执不以成功进程退出码隐藏。
# 输入：
#   args：配置路径与有界采集时间。
# 输出：
#   status：采集及安全关闭成功为零，否则为一。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=15)
    parser.add_argument("--yaw-profile", choices=("alternating", "sweep"), default="alternating")
    args = parser.parse_args()
    config = Px4TrainingConfig.model_validate(decode_json(
        read_plugin_file(args.config, limit=4 * 1024**2), limit=4 * 1024**2))
    validate_campaign(config, args.duration_seconds)
    config.output_root.mkdir(parents=True, exist_ok=False)
    report = collect(config, args.duration_seconds, args.yaw_profile)
    print(json.dumps(report, allow_nan=False), flush=True)
    status = int(bool(report["error"] or report.get("close_error")
                      or not report["safely_closed"] or not report.get("accepted")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
