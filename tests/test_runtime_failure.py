"""Failure messages are bounded diagnostics, not execution authority."""
import json

import pytest

from dronedream_agent_app.runtime_failure import executor_failure_detail


# 功能：
#   验证真实预检错误可被中英文界面解释，而任意异常、成功记录或损坏回执不会泄露正文。
# 输入：
#   tmp_path：隔离目录；english：对话语言。
# 输出：
#   无：断言失败会由测试框架报告。
@pytest.mark.parametrize("english", [True, False])
def test_known_preflight_failure_is_explained_without_raw_exception(tmp_path, english):
    path = tmp_path / "offboard_timing.json"
    assert executor_failure_detail(tmp_path, english=english) is None


    receipt = {"status": "failed", "failure":
               "RuntimeError: NATIVE_PERCEPTION_NOT_READY_BEFORE_ARM:"
               "NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED:ValueError"}
    path.write_text(json.dumps(receipt), encoding="utf-8")
    code, message = executor_failure_detail(tmp_path, english=english)
    assert code == "NATIVE_ROUTE_UNCERTAINTY_BUDGET_UNFUNDED"
    assert ("localization" if english else "定位误差") in message
    for value in ([], {**receipt, "status": "completed"},
                  {**receipt, "failure": receipt["failure"] + ":SECRET"}):
        path.write_text(json.dumps(value), encoding="utf-8")
        assert executor_failure_detail(tmp_path, english=english) is None
    path.write_text('{"status":NaN}', encoding="utf-8")
    assert executor_failure_detail(tmp_path, english=english) is None


# 功能：只翻译已确认地图融合失败，不误报成功或泄露追加的原始异常。
# 输入：测试夹具、参数化值或函数签名中的显式参数。
# 输出：无业务返回值；以断言或预期异常表达结果。
@pytest.mark.parametrize(
    'reason',
    ['LIVE_MAP_STARTUP_GEOMETRY_INCOMPLETE', 'LIVE_MAP_STREAM_OBSERVATIONS_LOST'],
)
@pytest.mark.parametrize('english', [True, False])
def test_map_failure_is_bounded_and_localized(tmp_path, reason, english):
    path = tmp_path / 'offboard_timing.json'
    receipt = dict(status='failed', failure='RuntimeError: ' + reason)
    path.write_text(json.dumps(receipt), encoding='utf-8')
    code, message = executor_failure_detail(tmp_path, english=english)
    assert code == reason and ('定位' in message) is not english
    for invalid in (
        {**receipt, 'status': 'completed'},
        {**receipt, 'failure': receipt['failure'] + ':SECRET'},
    ):
        path.write_text(json.dumps(invalid), encoding='utf-8')
        assert executor_failure_detail(tmp_path, english=english) is None


# 功能：验证桌面端能从长时飞行证据中提取白名单失败，不被普通消息节点上限误伤。
# 输入：包含大量有界追踪事件的已知失败回执。
# 输出：仍返回本地化的已知错误码。
def test_long_flight_failure_evidence_uses_explicit_node_budget(tmp_path):
    path = tmp_path / "offboard_timing.json"
    receipt = {
        "status": "failed",
        "failure": "RuntimeError: LIVE_MAP_STREAM_OBSERVATIONS_LOST",
        "tracking_recoveries": [
            {"index": index, "samples": [0, 1, 2, 3]} for index in range(20_000)
        ],
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")

    code, message = executor_failure_detail(tmp_path, english=False)
    assert code == "LIVE_MAP_STREAM_OBSERVATIONS_LOST"
    assert "定位" in message
