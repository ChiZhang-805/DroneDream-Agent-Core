"""资产诊断的显示边界；不把提示信息当作可飞行资格。"""

from copy import deepcopy

import pytest

from dronedream_agent_app import asset_issue_catalog as catalog


# 功能：
#   验证修改返回的提示卡不会污染下一次请求使用的共享说明模板。
# 输入：
#   monkeypatch：隔离共享模板的测试夹具。
# 输出：
#   None：不返回业务数据。
def test_required_input_actions_are_independent(monkeypatch):
    monkeypatch.setattr(catalog, "_INPUT_GUIDANCE", deepcopy(catalog._INPUT_GUIDANCE))
    expected = deepcopy(catalog.describe_required_input("asset_kind"))
    first = catalog.describe_required_input("asset_kind")
    first["actions"][0]["en-US"] = "changed by caller"
    first["actions"].clear()
    first["title"]["en-US"] = "also changed"
    assert catalog.describe_required_input("asset_kind") == expected


# 功能：
#   核对错误分类的关键优先级、双语说明和操作标签保持一致。
# 输入：
#   code：来源错误码。
#   severity：期望严重等级。
#   location：期望检查位置。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("code", "severity", "location"),
    [
        ("DDPKG_MEMBER_PATH_INVALID", "critical", "source.package"),
        ("ASSET_REMOTE_PRIVATE_NETWORK_FORBIDDEN", "critical", "source.location"),
        ("ASSET_REMOTE_GIT_SIZE_EXCEEDED", "error", "source.git"),
        ("ASSET_REMOTE_DOWNLOAD_FAILED", "error", "source.remote"),
        ("ASSET_SOURCE_SIZE_EXCEEDED", "error", "source.package"),
        ("ASSET_SOURCE_ADAPTER_REQUIRED", "error", "source.adapter"),
        ("MAP_MANIFEST_INVALID", "error", "source.format"),
        ("VEHICLE_IDENTITY_MISMATCH", "error", "identity"),
        ("MAP_CLEARANCE_INVALID", "error", "map.qualification"),
        ("VEHICLE_MASS_INVALID", "error", "vehicle.qualification"),
        ("ASSET_QUALIFICATION_FAILED", "error", "qualification.evidence"),
        ("FUTURE_UNKNOWN_CODE", "error", "job"),
    ],
)
def test_issue_classification(code, severity, location):
    issue = catalog.describe_asset_issue(code)
    assert issue["code"] == code
    assert issue["severity"] == severity
    assert issue["location"] == location
    for field in ("title", "detail"):
        assert set(issue[field]) == {"zh-CN", "en-US"}
        assert all(issue[field].values())
    assert all(set(action) == {"id", "zh-CN", "en-US"} for action in issue["actions"])


# 功能：
#   确认空记录保留兼容默认值，正常列表按首次出现顺序去重且不修改来源。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_job_report_defaults_and_stable_deduplication():
    assert catalog.asset_job_issue_report({}) == {
        "schema_version": "dronedream.asset-issue-report.v1",
        "job_id": "",
        "state": "unknown",
        "progress_percent": 0,
        "issues": [],
    }
    job = {
        "job_id": "asset-job-example",
        "state": "needs_input",
        "progress_percent": 30,
        "issue_codes": ["MAP_CLEARANCE_INVALID", "", "MAP_CLEARANCE_INVALID", "NEW_CODE"],
        "required_inputs": ["asset_kind", "asset_kind", "future_field"],
    }
    original = deepcopy(job)
    report = catalog.asset_job_issue_report(job)
    assert [issue["code"] for issue in report["issues"]] == [
        "MAP_CLEARANCE_INVALID",
        "NEW_CODE",
        "REQUIRED_INPUT_ASSET_KIND",
        "REQUIRED_INPUT_FUTURE_FIELD",
    ]
    assert report["issues"][-1]["location"] == "required_inputs.future_field"
    assert report["progress_percent"] == 30
    assert job == original


# 功能：
#   拒绝把字符串拆成多个错误码、把对象强转文本或把非整数进度显示为正常值。
# 输入：
#   field：被破坏的记录字段。
#   value：不符合已存任务契约的值。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("issue_codes", "MAP_INVALID"),
        ("issue_codes", {"MAP_INVALID": True}),
        ("issue_codes", None),
        ("issue_codes", [1]),
        ("required_inputs", "asset_kind"),
        ("required_inputs", [None]),
        ("job_id", 12),
        ("state", False),
        ("progress_percent", True),
        ("progress_percent", 30.5),
        ("progress_percent", "30"),
        ("progress_percent", -1),
        ("progress_percent", 101),
    ],
)
def test_malformed_job_fields_are_not_coerced(field, value):
    with pytest.raises(ValueError):
        catalog.asset_job_issue_report({field: value})


# 功能：
#   使用任务契约的原始列表上限检查输入，重复值不能规避条目预算。
# 输入：
#   field：受限列表字段。
#   limit：导入任务允许的最大条目数。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize(("field", "limit"), [("issue_codes", 1000), ("required_inputs", 256)])
def test_issue_list_budget_precedes_deduplication(field, limit):
    assert len(catalog.asset_job_issue_report({field: ["example"] * limit})["issues"]) == 1
    with pytest.raises(ValueError):
        catalog.asset_job_issue_report({field: ["example"] * (limit + 1)})


# 功能：
#   确认直接诊断入口对非法标识返回明确输入错误，而不是任意对象的字符串表示。
# 输入：
#   value：非文本或空标识。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("value", [None, True, 12, [], {}, ""])
def test_diagnostic_identifiers_require_nonempty_text(value):
    with pytest.raises(ValueError):
        catalog.describe_asset_issue(value)
    with pytest.raises(ValueError):
        catalog.describe_required_input(value)


# 功能：
#   检查错误卡和缺失输入卡仅是解释数据，不包含可执行指令或资格放行字段。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_diagnostics_do_not_grant_flight_readiness():
    for issue in (
        catalog.describe_required_input("qualification_evidence"),
        catalog.describe_asset_issue("ASSET_QUALIFICATION_FAILED"),
    ):
        assert set(issue) == {
            "schema_version",
            "code",
            "severity",
            "stage",
            "location",
            "title",
            "detail",
            "actions",
        }
        assert "evidence" in issue["detail"]["en-US"]
