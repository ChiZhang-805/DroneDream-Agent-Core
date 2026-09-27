"""Payload measurement authority cannot become mixed or production authority."""

import pytest

from dronedream_agent_core.payload_collection_contract import payload_collection_mode


# 功能：
#   构造只在离线单元测试使用的完整教师测量权限。
# 输入：
#   无。
# 输出：
#   arguments：无模型权限、启用双路原始记录的参数。
def teacher_arguments():
    arguments = dict(provider=None, teacher=True, recording=True, model_authority=False,
                     model_packages=False, admission=False, qualification=False,
                     multimodal=True, fallback=False)
    return arguments


# 功能：
#   检查教师不需要旧模型包，同时拒绝混合模型权限、缺失记录和正式资格。
# 输入：
#   change：相对合法教师配置的单项损坏。
# 输出：
#   None：合法模式返回名称，损坏配置明确失败。
@pytest.mark.parametrize('change', [None, ('provider', 'local-policy'), ('recording', False),
    ('model_authority', True), ('model_packages', True), ('admission', True),
    ('qualification', True), ('multimodal', False), ('fallback', True), ('teacher', 1)])
def test_teacher_collection_is_exclusive(change):
    arguments = teacher_arguments()
    if change is None:
        assert payload_collection_mode(**arguments) == 'recorded-simulation-teacher'
    else:
        arguments[change[0]] = change[1]
        with pytest.raises(ValueError):
            payload_collection_mode(**arguments)


# 功能：
#   检查研究模型测量必须绑定真实模型包及仿真准入，不能混入教师或其他回退控制。
# 输入：
#   change：相对完整模型测量权限的单项损坏。
# 输出：
#   None：权限完整时通过，否则明确失败。
@pytest.mark.parametrize('change', [None, ('provider', 'cloud'), ('teacher', True),
    ('model_authority', False), ('model_packages', False), ('admission', False),
    ('qualification', True), ('multimodal', False), ('fallback', True)])
def test_model_collection_retains_admission(change):
    arguments = teacher_arguments()
    arguments.update(provider='local-policy', teacher=False, model_authority=True,
                     model_packages=True, admission=True)
    if change is None:
        assert payload_collection_mode(**arguments) == 'simulation-admitted-model'
    else:
        arguments[change[0]] = change[1]
        with pytest.raises(ValueError):
            payload_collection_mode(**arguments)
