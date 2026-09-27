"""Read-only, identity-bound position access without full observation copying."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dronedream_agent_core.training.px4_environment import Px4GazeboTrainingEnvironment


# 功能：
#   检查位置查询保留生命周期和观测身份校验，不触发无关回执查询或共享可变状态。
# 输入：
#   damage：当前观测或环境生命周期的故障类型。
# 输出：
#   None：所有状态与隔离断言通过，否则测试失败。
@pytest.mark.parametrize('damage', [
    'none', 'closed', 'quiesced', 'pending', 'observation', 'identity', 'nan'])
def test_position_access_is_owned_and_bound(damage):
    env = object.__new__(Px4GazeboTrainingEnvironment)
    env._pending = {'snapshot': {'snapshot_sha256': 'a' * 64,
                                'current_position_m': {'x': 1., 'y': 2., 'z': 3.}}}
    env._observation = SimpleNamespace(sample=SimpleNamespace(source_snapshot_sha256='a' * 64))
    env._closed, env._quiesced = damage == 'closed', damage == 'quiesced'
    env._receipt_monitor = Mock()
    if damage == 'pending':
        env._pending = None
    elif damage == 'observation':
        env._observation = None
    elif damage == 'identity':
        env._observation.sample.source_snapshot_sha256 = 'b' * 64
    elif damage == 'nan':
        env._pending['snapshot']['current_position_m']['z'] = float('nan')
    if damage == 'none':
        position = env.current_position_m()
        assert (position.x, position.y, position.z) == (1., 2., 3.)
        position.z = 8.
        assert env._pending['snapshot']['current_position_m']['z'] == 3.
    else:
        with pytest.raises((ValueError, RuntimeError)):
            env.current_position_m()
    assert not env._receipt_monitor.mock_calls
