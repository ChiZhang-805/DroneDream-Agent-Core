import pytest
from test_runtime_local_safety import _vehicle

from dronedream_agent_core.contracts import RuntimeLocalSafetyObservation, Vector3
from dronedream_agent_core.control_timing import LOCAL_DISPATCH_RESERVE_MS
from dronedream_agent_core.runtime_local_safety import (
    evaluate_runtime_local_safety,
    prepare_safety_publication,
)


# 功能：
#   建立使用真实合同校验的短寿命保护命令，供发布边界测试。
# 输入：
#   无。
# 输出：
#   command：原始时间为一万毫秒的安全命令。
def _command():
    observation = RuntimeLocalSafetyObservation(
        sequence=1, observed_at_unix_ms=10_000, source="onboard",
        stream_healthy=False, stream_age_seconds=0., localization_covariance_m2=.01,
        current_position_m=Vector3(x=0., y=0., z=2.),
        current_velocity_mps=Vector3(x=0., y=0., z=0.),
        target_position_m=Vector3(x=1., y=0., z=2.), dynamic_obstacles=[],
    )
    command = evaluate_runtime_local_safety(
        observation=observation, vehicle=_vehicle(), static_primitives=[],
        required_clearance_m=.35, generated_at_unix_ms=10_000,
    )
    return command


# 功能：
#   过期或剩余传输预算不足时丢弃命令，不因负寿命异常中断整个感知周期。
# 输入：
#   remaining：发布时距离截止的毫秒数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("remaining", [-1000, -1, 0, LOCAL_DISPATCH_RESERVE_MS])
def test_expired_evaluation_is_not_rebased_or_serialized(remaining):
    evaluated = _command()
    before = evaluated.model_dump()
    assert prepare_safety_publication(evaluated, evaluated.valid_until_unix_ms - remaining,
                                      Vector3(x=0., y=0., z=0.)) is None
    assert evaluated.model_dump() == before


# 功能：
#   有效发布仅消耗原租期，时钟回退和异常定位偏移仍被明确拒绝。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_publication_preserves_deadline_and_rejects_invalid_alignment_or_clock():
    evaluated = _command()
    result = prepare_safety_publication(evaluated, 10_001, Vector3(x=0., y=0., z=0.))
    assert result.generated_at_unix_ms == 10_001
    assert result.valid_until_unix_ms == evaluated.valid_until_unix_ms
    with pytest.raises(ValueError, match="CLOCK_REGRESSED"):
        prepare_safety_publication(evaluated, 9999, Vector3(x=0., y=0., z=0.))
    with pytest.raises(ValueError, match="exceeds 1 metre"):
        prepare_safety_publication(evaluated, 20_000, Vector3(x=2., y=0., z=0.))
