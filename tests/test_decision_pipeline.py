"""Training archive test discovery must not silently omit new verifiers."""

from scripts.decision_pipeline import CODE, bundled_tests


# 功能：所有冻结测试都要运行，但辅助夹具不当成测试入口；输入：完整白名单；输出：精确集合。
def test_bundle_runs_every_allowlisted_test():
    names = bundled_tests(CODE)
    assert len(names) == len(set(names))
    assert set(names) == {n for n in CODE if n.startswith("tests/test_")}
    assert "tests/test_decision_stage_wait.py" in names
    assert "tests/test_decision_input_evidence.py" in names
    assert "tests/decision_goal_fixture.py" not in names


# 功能：新测试无需维护第二张运行名单；输入：新增文件及非测试源；输出：新增测试确实入选。
def test_new_test_is_not_only_packaged():
    names = bundled_tests([*CODE, "tests/test_new_verifier.py", "tests/helper.py"])
    assert "tests/test_new_verifier.py" in names
    assert "tests/helper.py" not in names
