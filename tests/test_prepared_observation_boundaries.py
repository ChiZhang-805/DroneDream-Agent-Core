"""Sensor-to-learner identity boundaries with controlled clocks and no flight."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_training_observation_boundary import current_request
from test_training_policy_exchange import proposal

from dronedream_agent_core.contracts import TextNavigationDecision
from dronedream_agent_core.hashing import sha256_json
from dronedream_agent_core.training import policy_port
from dronedream_agent_core.training.observations import (
    PreparedTrainingInput,
    compile_training_observation,
    load_policy_observations,
)


# 功能：
#   验证准备好的样本或融合特征被调用方修改后不可继续接纳，冻结外壳不代表嵌套容器不可变。
# 输入：
#   field：需要改写的准备结果部分。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("field", ["sample", "features"])
def test_prepared_nested_state_cannot_be_modified(field):
    request = current_request()
    prepared = PreparedTrainingInput.from_request(request)
    if field == "sample":
        prepared.sample.state_features[0] += 1
    else:
        prepared.features.fused_features[0] += 1
    with pytest.raises(ValueError, match="PREPARED.*CHANGED"):
        prepared.admit(request, now_unix_ms=1100)


# 功能：
#   验证 NaN 等非法时钟不能利用比较结果绕过准备观测的时效检查。
# 输入：
#   now：非法的当前 Unix 毫秒值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("now", [float("nan"), 1100.0, True, "1100"])
def test_prepared_input_rejects_invalid_clock(now):
    request = current_request()
    prepared = PreparedTrainingInput.from_request(request)
    with pytest.raises(ValueError, match="CLOCK"):
        prepared.admit(request, now_unix_ms=now)


# 功能：
#   验证原始观测编译也检查时钟，不能只依赖之后的准备对象接纳入口。
# 输入：
#   无：使用一份有效快照与 NaN 时钟。
# 输出：
#   None：不返回业务数据。
def test_compilation_rejects_nan_clock():
    with pytest.raises(ValueError, match="CLOCK"):
        compile_training_observation(current_request()["snapshot"], now_unix_ms=float("nan"))


# 功能：
#   验证快照中的策略上下文必须是对象，错误容器不能成为未经处理的属性访问异常。
# 输入：
#   task：错误的任务上下文结构。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("task", [None, [], "wrong"])
def test_compilation_rejects_malformed_task_context(task):
    snapshot = current_request()["snapshot"]
    snapshot.pop("snapshot_sha256")
    snapshot["strategic_context"] = {"task": task}
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    with pytest.raises(ValueError, match="CONTEXT"):
        compile_training_observation(snapshot, now_unix_ms=1000)


# 功能：
#   验证离线观测历史拒绝重复字段和空白记录，不能静默替换值或丢弃样本。
# 输入：
#   tmp_path：测试独立目录。
#   invalid：需要注入的 JSONL 错误。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("invalid", ["duplicate", "blank"])
def test_observation_history_requires_strict_records(tmp_path, invalid):
    sample = PreparedTrainingInput.from_request(current_request()).sample
    encoded = sample.model_dump_json()
    if invalid == "duplicate":
        encoded = '{"source_snapshot_sha256":"wrong",' + encoded[1:]
    else:
        encoded += "\n\n"
    path = tmp_path / "history.jsonl"
    path.write_text(encoded, encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy_observations(path)


# 功能：
#   验证学习器交换期间外部修改原输入，不会改变已开始调用的输入摘要和快照身份。
# 输入：
#   monkeypatch：提供受控时钟与本地客户端替身的夹具。
# 输出：
#   None：不返回业务数据。
def test_training_port_owns_input_for_entire_exchange(monkeypatch):
    request = current_request()
    artifact = {"text_navigation_snapshot": request["snapshot"]}
    expected = sha256_json(artifact)
    port = object.__new__(policy_port.SimulationTrainingPolicyPort)

    # 功能：
    #   模拟交换等待期间调用方改写原始快照，并返回绑定所收请求的合法合成提案。
    # 输入：
    #   value：策略入口提交的请求快照。
    # 输出：
    #   result：绑定该请求的合成操纵提案。
    def mutate_during_exchange(value):
        artifact["text_navigation_snapshot"]["goal_position_m"]["x"] += 1
        result = proposal(value)
        return result

    port.client = SimpleNamespace(propose=Mock(side_effect=mutate_during_exchange))
    monkeypatch.setattr(policy_port, "time",
                        SimpleNamespace(time=lambda: 1., monotonic=lambda: 10.))
    result = port.call(role="local_navigation_advisor", output_type=TextNavigationDecision,
                       instructions="", input_artifact=artifact)
    assert result.record.input_sha256 == expected
    assert sha256_json(artifact) != expected


# 功能：
#   验证训练入口拒绝错误或超预算图像，不能先执行巨大 Base64 分配或联系学习器。
# 输入：
#   monkeypatch：提供受控时钟和解码监视器的夹具。
#   media：非法媒体容器或超出单帧预算的像素记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("media", [[None], [{}, {}], {"wrong": "container"},
                                  [{"model_rgb_bytes": bytes(2 * 1024 * 1024)}]])
def test_training_port_rejects_media_before_conversion(monkeypatch, media):
    port = object.__new__(policy_port.SimulationTrainingPolicyPort)
    port.client = Mock()
    encoder = Mock(side_effect=AssertionError("invalid pixels reached Base64 encoder"))
    monkeypatch.setattr(policy_port.base64, "b64encode", encoder)
    monkeypatch.setattr(policy_port, "time",
                        SimpleNamespace(time=lambda: 1., monotonic=lambda: 10.))
    with pytest.raises(ValueError, match="MEDIA"):
        port.call(role="local_navigation_advisor", output_type=TextNavigationDecision,
                  instructions="", input_artifact={"text_navigation_snapshot":
                  current_request()["snapshot"]}, multimodal=media)
    encoder.assert_not_called()
    port.client.propose.assert_not_called()
