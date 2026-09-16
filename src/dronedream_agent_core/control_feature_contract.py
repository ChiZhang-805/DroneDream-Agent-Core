"""Content identities for learned control inputs, independent of tensor width.

Change the semantic description when units, ordering, masks or history change.
Legacy recordings deliberately have no identity; never infer one from shape.
"""

import re

from .depth_projection import DEPTH_REDUCTION_SEMANTICS
from .hashing import sha256_json

FLIGHT_STATE_HISTORY_LENGTH = 16
FLIGHT_STATE_MAXIMUM_GAP_MS = 250
FLIGHT_STATE_MAXIMUM_COMPONENT_SKEW_MS = 50
GRAVITY_MPS2 = 9.80665
SPATIAL_CELL_COUNT = 26  # horizontal, upper, lower rings of eight, then down/up poles
GEOMETRY_FEATURE_COUNT = SPATIAL_CELL_COUNT * 5
DYNAMIC_TARGET_FEATURE_COUNT = SPATIAL_CELL_COUNT * 11
FLIGHT_STATE_FEATURE_COUNT = 20
SENSOR_FEATURE_COUNT = (
    GEOMETRY_FEATURE_COUNT + DYNAMIC_TARGET_FEATURE_COUNT + FLIGHT_STATE_FEATURE_COUNT
)
ENCODER_ROLES = (
    "metric-geometry-encoder", "dynamic-target-encoder", "flight-state-encoder",
)


# 功能：
#   1. 对编码器的单位、坐标约定、排序、缺测语义和历史时序生成内容身份。
#   2. 非法角色或时序配置不生成身份；同宽度但语义不同的旧模型不能冒充兼容。
# 输入：
#   role：几何、动态目标或飞行状态编码器的标准角色名。
#   history_length：飞行状态历史长度，必须为四到二百五十六之间的整数。
#   maximum_gap_ms：飞行状态允许的最大来源毫秒间隔。
# 输出：
#   digest：当前角色完整特征语义的 SHA-256 字符串。
def encoder_contract_sha256(role: str, *, history_length: int = FLIGHT_STATE_HISTORY_LENGTH,
                            maximum_gap_ms: int = FLIGHT_STATE_MAXIMUM_GAP_MS) -> str:
    if not isinstance(role, str) or role not in ENCODER_ROLES:
        raise ValueError("unknown realtime encoder role")
    if (type(history_length) is not int or not 4 <= history_length <= 256
            or type(maximum_gap_ms) is not int or not 10 <= maximum_gap_ms <= 5_000):
        raise ValueError("invalid flight-state encoder history contract")
    descriptions = {
        "metric-geometry-encoder": {
            "width": GEOMETRY_FEATURE_COUNT,
            "sectors": "26 FRU cells: level,upper,lower rings then down,up; 15/60 degree edges",
            "per_sector": ["nearest-body-hit/max-range", "minimum-ray-reach/max-range", "hit-ratio",
                           "mean-confidence", "angular-coverage"],
            "calibration": "full rigid mount; declared horizontal and vertical FOV",
            "depth_reduction": DEPTH_REDUCTION_SEMANTICS,
            "map_integration": "same-source scan; once per voxel; hit wins; fresh hit revokes free",
            "source_quality": "mean ray quality times measured valid-pixel fraction when available",
            "pose_alignment": (
                "native bounded receive-time history; at most 50ms component skew; "
                "translation and angular footprint inflate variance; no truth-fitted pose"
            ),
            "missing": "zero value and zero mask; not observed free space",
        },
        "dynamic-target-encoder": {
            "width": DYNAMIC_TARGET_FEATURE_COUNT,
            "sectors": "same 26 FRU cells; earliest swept-cylinder contact then distance then id",
            "per_sector": ["distance/max-range", "signed-closing-speed/10", "swept-ttc/30",
                           "confidence", "age-at-encoding/maximum-age", "relative-vx/10",
                           "relative-vy/10", "relative-vz/10", "radius/max-range",
                           "height/max-range", "predicted-center-clearance/max-range"],
            "timing": "predict track by its age; preserve oldest track deadline",
            "depth_tracks": (
                "distinct credible points; source clock bound; confidence capped by measurement; "
                "cylinder encloses observed bounds around tracking centroid; scan-atomic history"
            ),
            "stale": "expired tracks invalidate the control snapshot",
            "clearance": ("two fresh metric scans cover "
                          "acceleration-and-covariance-inflated reachable box"),
        },
        "flight-state-encoder": {
            "width": 20, "history_length": history_length, "maximum_gap_ms": maximum_gap_ms,
            "fields": ["world-ENU-from-body-FLU quaternion wxyz", "velocity-body-FRU/20",
                       "position-covariance-m2/0.25", "linear-acceleration-body-FRU/g",
                       "angular-velocity-body-FLU/5", "per-FRU-axis velocity mean/20,std/5"],
            "acceleration": "native IMU specific force plus body-projected world gravity",
            "gravity_mps2": GRAVITY_MPS2,
            "missing_covariance": "zero placeholder, zero mask, uncertainty one, control blocked",
            "readiness": "measured covariance <=0.25m2 and complete acceleration/gyro required",
            "measured_covariance": (
                "native odometry positional PSD row-sum bound; measured diagonal trace "
                "bounds unreported correlations without assuming independence; "
                "direction-independent bound accepts defined metric SDK odometry frames, "
                "never odometry-frame position or orientation"
            ),
            "timing": "oldest source receive time; never refresh a cached sample",
            "maximum_pose_imu_receive_skew_ms": FLIGHT_STATE_MAXIMUM_COMPONENT_SKEW_MS,
            "source_identity": (
                "physical source contents and original times, excluding polling metadata"
            ),
            "native_clock": "duplicate IMU packets keep deadline; boot-clock reset clears history",
            "pose": ("native NED position and NED-FRD attitude; "
                     "fixed deployment ENU map binding; no truth correction"),
        },
    }
    digest = sha256_json(descriptions[role])
    return digest


# 功能：
#   绑定三个编码器与控制参考、坐标方向和输出单位；缺少任一有效身份时拒绝组合。
# 输入：
#   encoder_hashes：恰好包含三个标准角色及各自小写 SHA-256 的字典。
# 输出：
#   digest：组合后的策略输入语义 SHA-256，输入不合规时为 None。
def policy_feature_contract_sha256(encoder_hashes: dict[str, str]) -> str | None:
    if (not isinstance(encoder_hashes, dict) or set(encoder_hashes) != set(ENCODER_ROLES)
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in encoder_hashes.values())):
        return None
    digest = sha256_json({
        "encoders": encoder_hashes,
        "sensor_width": SENSOR_FEATURE_COUNT,
        "control_reference": ["body-FRU velocity/active-control-speed clipped +/-4",
                              "body-FRU goal/local-radius clipped +/-2",
                              "body-FRU unit-goal", "atan2(right,forward)/pi"],
        "goal_authority": "approved semantic stage goal; never teacher's moving route lookahead",
        "policy_state_velocity": "world-ENU velocity/active-control-speed",
        "pilot_output": "normalized forward,right,up velocity and clockwise NED yaw-rate",
        "action_critic": "proposed post-Harness physical FRU velocity /20 and clockwise yaw /180",
    })
    return digest


CURRENT_POLICY_FEATURE_CONTRACT_SHA256 = policy_feature_contract_sha256({
    role: encoder_contract_sha256(role) for role in ENCODER_ROLES
})
