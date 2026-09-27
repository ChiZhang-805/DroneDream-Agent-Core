"""Frozen task contracts travel through dedicated adapter parameters, not overrides."""

from pathlib import Path

import pytest

from dronedream_agent_core.gazebo_adapter import (
    _executor_contract_arguments,
    _validated_executor_options,
)


# 功能：
#   核对两个物理合同与改令资产各传递一次，等待时长保留明确单位。
# 输入：
#   tmp_path：隔离目录。
# 输出：
#   None：合同路径或时长被遗漏、替换或重复则测试失败。
def test_frozen_contract_paths_reach_executor(tmp_path):
    checkpoint, action = tmp_path / 'runtime-checkpoints.json', tmp_path / 'runtime-actions.json'
    semantic, vehicle = tmp_path / 'semantic.json', tmp_path / 'vehicle.json'
    control = tmp_path / 'control'
    arguments = _executor_contract_arguments(checkpoint, action, semantic, vehicle, control,
                                              (180., 12., 180., 60.))
    options = dict(zip(arguments[::2], arguments[1::2], strict=True))
    assert len(options) * 2 == len(arguments)
    assert options == {'--checkpoint-contract': str(checkpoint),
        '--runtime-action-contract': str(action), '--runtime-control-dir': str(control),
        '--semantic': str(semantic), '--vehicle-metadata': str(vehicle),
        '--checkpoint-timeout-seconds': '180.0', '--runtime-hold-timeout-seconds': '12.0',
        '--runtime-decision-timeout-seconds': '180.0', '--runtime-replan-hold-seconds': '60.0'}
    assert _executor_contract_arguments(None, None, semantic, vehicle, None,
                                         (180., 12., 180., 60.)) == []


# 功能：
#   禁止单独动作、改令或缺失资产的入口绕开检查点合同。
# 输入：
#   values：不完整的合同路径组合。
# 输出：
#   None：启动参数编译立即拒绝，否则测试失败。
@pytest.mark.parametrize('values', [
    (None, Path('actions'), None, None, None),
    (None, None, Path('semantic'), Path('vehicle'), Path('control')),
    (Path('checkpoint'), None, None, Path('vehicle'), Path('control')),
    (Path('checkpoint'), None, Path('semantic'), None, Path('control'))])
def test_executor_contracts_require_bound_dependencies(values):
    with pytest.raises(ValueError):
        _executor_contract_arguments(*values, (180., 12., 180., 60.))


# 功能：
#   检查等待时长为有界有限数值，拒绝布尔值和不合法范围。
# 输入：
#   value：待测试时长。
# 输出：
#   None：无效时长不能到达执行器。
@pytest.mark.parametrize('value', [True, '180', float('nan'), float('inf'), 0., -1., 3601.])
def test_executor_contracts_reject_invalid_timeout(value):
    with pytest.raises(ValueError):
        _executor_contract_arguments(None, None, None, None, None, (value, 12., 180., 60.))


# 功能：
#   确认新增专用传输入口没有重新开放扩展参数覆盖合同、资产或控制权限。
# 输入：
#   name：敏感参数名称。
# 输出：
#   None：敏感参数仍不能经额外参数注入。
@pytest.mark.parametrize('name', ['--checkpoint-contract', '--runtime-action-contract',
    '--runtime-control-dir', '--semantic', '--vehicle-metadata', '--teacher-payload-checkpoints'])
def test_contracts_are_not_generic_executor_overrides(name):
    with pytest.raises(ValueError):
        _validated_executor_options([name, 'other'])
