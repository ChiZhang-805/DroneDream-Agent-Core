"""Record only explicit graphics requests, never infer the actual selected GPU.

Driver/compiler choices can change camera latency and numerical output. Their
requested values belong beside sensor and timing evidence. This module neither
changes the environment nor grants image-equivalence or flight qualification.
"""

from collections.abc import Mapping

from .hashing import sha256_json

GRAPHICS_ENVIRONMENT_KEYS = (
    "GALLIUM_DRIVER",
    "LIBGL_ALWAYS_SOFTWARE",
    "LP_NUM_THREADS",
    "LP_NATIVE_VECTOR_WIDTH",
    "GALLIVM_PERF",
    "MESA_D3D12_DEFAULT_ADAPTER_NAME",
    "MESA_SHADER_CACHE_DIR",
    "MESA_SHADER_CACHE_MAX_SIZE",
    "MESA_SHADER_CACHE_DISABLE",
    "EGL_PLATFORM",
)


# 功能：
#   仅提取图形白名单环境并绑定摘要，拒绝不可执行的值；不记录其他变量中的凭证。
#   记录的是请求配置，不推断实际显卡、图像等价或飞行资格。
# 输入：
#   environment：当前渲染进程请求使用的环境映射。
# 输出：
#   evidence：独立的配置白名单、摘要及未授予资格标志。
def graphics_request_evidence(environment: Mapping[str, str]) -> dict:
    if not isinstance(environment, Mapping):
        raise ValueError("SIMULATION_GRAPHICS_REQUEST_INVALID")
    requested, missing = {}, object()
    for key in GRAPHICS_ENVIRONMENT_KEYS:
        # 缺失与显式非法空值不同；仅缺失变量用 None 标记，空字符串仍保留其真实含义。
        value = environment.get(key, missing)
        if value is not missing and (not isinstance(value, str) or len(value) > 2048
                                     or "\x00" in value):
            raise ValueError("SIMULATION_GRAPHICS_REQUEST_INVALID")
        requested[key] = None if value is missing else value
    evidence = {
        "requested_environment": requested,
        "requested_environment_sha256": sha256_json(requested),
        "actual_driver_verified": False,
        "image_equivalence_claimed": False,
        "flight_qualification_granted": False,
        "purpose": "requested-graphics-settings-only; verify actual renderer separately",
    }
    return evidence
