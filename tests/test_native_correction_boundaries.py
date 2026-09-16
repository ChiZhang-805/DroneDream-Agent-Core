"""Offline correction evidence and cache boundary tests using synthetic receipts."""

import json

import pytest
from test_native_corrections import fixture


# 功能：
#   验证返回的教师状态可被调用方修改，但不能污染下一次读取的缓存状态。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_returned_teacher_context_does_not_alias_cache(tmp_path):
    oracle, observation, _, _ = fixture(tmp_path)
    first = oracle._context(observation)
    original_x = first.position.x
    first.position.x += 3
    second = oracle._context(observation)
    assert second.position.x == original_x
    second.velocity.x += 4
    assert oracle._context(observation).velocity.x != second.velocity.x


# 功能：
#   验证证据文件中的重复键在缓存命中前也会被拒绝，不由最后一个字段静默覆盖。
# 输入：
#   tmp_path：合成回合根目录。
#   target：注入歧义字段的文件类别。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("target", ["transition", "terminal", "reset"])
def test_native_correction_rejects_ambiguous_json(tmp_path, target):
    oracle, observation, path, terminal = fixture(tmp_path)
    oracle._context(observation)
    if target == "terminal":
        path = terminal
    elif target == "reset":
        path = path.parent / "reset.json"
    original = path.read_text(encoding="utf-8")
    key, value = next(iter(json.loads(original).items()))
    path.write_text("{" + json.dumps(key) + ":" + json.dumps(value) + "," + original[1:],
                    encoding="utf-8")
    with pytest.raises(ValueError):
        oracle._context(observation)


# 功能：
#   验证超过转移字节预算的文件在离线教师读取前被拒绝，即使多余字节只是空白。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_native_correction_bounds_transition_bytes(tmp_path):
    oracle, observation, path, _ = fixture(tmp_path)
    with path.open("ab") as stream:
        stream.write(b" " * (4 * 1024 * 1024))
    with pytest.raises(ValueError, match="LARGE|SIZE"):
        oracle._context(observation)


# 功能：
#   验证读取相同内容期间发生的文件身份替换仍被发现，不能靠内容相同掩盖替换。
# 输入：
#   tmp_path：合成回合根目录。
#   monkeypatch：受控替换 Path 的读取方法。
# 输出：
#   None：不返回业务数据。
def test_native_correction_detects_same_content_file_replacement(tmp_path, monkeypatch):
    oracle, observation, path, _ = fixture(tmp_path)
    original_open = type(path).open
    original = path.read_bytes()

    # 功能：
    #   在目标流关闭后用新文件替换同名路径，避免依赖 Windows 不允许的打开文件重命名。
    # 输入：
    #   selected：准备打开的路径。
    #   args、kwargs：传给真实打开方法的参数。
    # 输出：
    #   stream：具有相同读取能力及受控关闭行为的测试流。
    def replacing_open(selected, *args, **kwargs):
        stream = original_open(selected, *args, **kwargs)
        if selected == path and args and args[0] == "rb":
            original_close = stream.close

            # 功能：
            #   先关闭目标流，再用新文件身份替换同内容文件；只在测试独占目录内执行。
            # 输入：
            #   无。
            # 输出：
            #   None：不返回业务数据。
            def close_and_replace():
                original_close()
                if not getattr(stream, "already_replaced", False):
                    stream.already_replaced = True
                    replacement = path.with_suffix(".replacement")
                    with original_open(replacement, "wb") as target:
                        target.write(original)
                    replacement.replace(path)

            stream.close = close_and_replace
        return stream

    monkeypatch.setattr(type(path), "open", replacing_open)
    with pytest.raises(ValueError, match="CHANGED"):
        oracle._context(observation)


# 功能：
#   验证故障调用不能用未经验证的观测字段访问已建立的上下文缓存。
# 输入：
#   tmp_path：合成回合根目录。
# 输出：
#   None：不返回业务数据。
def test_native_correction_revalidates_observation_before_cache(tmp_path):
    oracle, observation, _, _ = fixture(tmp_path)
    oracle._context(observation)
    observation.sample.state_features[0] = float("nan")
    with pytest.raises(ValueError):
        oracle._context(observation)
