import pytest

from dronedream_agent_core.contracts import Px4RuntimeAbortRequest
from dronedream_agent_core.gazebo_adapter import _read_external_abort_evidence


# 功能：
#   验证坏停止请求不会进入严格证据对象，也不能被当成没有发生停止；原文件保持原样。
# 输入：
#   tmp_path：测试证据目录。
#   content：未知字段、非法布尔、非对象或损坏 JSON。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("content", [
    '{"reason":"operator-stop","qualification_granted":false}',
    '{"reason":"operator-stop","world_paused":"false"}', '[1,2]', '{',
])
def test_invalid_abort_preserves_failure_instead_of_crashing_evidence(tmp_path, content):
    path = tmp_path / "abort.json"
    path.write_text(content)
    request = _read_external_abort_evidence(path)
    assert request["reason"] == "EXTERNAL_ABORT_REQUEST_INVALID"
    Px4RuntimeAbortRequest.model_validate(request, strict=True)
    assert path.read_text() == content


# 功能：
#   合法停止请求完整保留，没有请求文件时才返回 None。
# 输入：
#   tmp_path：测试证据目录。
# 输出：
#   None：不返回业务数据。
def test_missing_and_valid_abort_have_distinct_meanings(tmp_path):
    path = tmp_path / "abort.json"
    assert _read_external_abort_evidence(path) is None
    path.write_text('{"reason":"operator-stop","world_paused":false}')
    assert _read_external_abort_evidence(path) == {"reason": "operator-stop", "world_paused": False}
