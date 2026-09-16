"""Actionable, bilingual diagnostics for asset import and qualification jobs."""

from __future__ import annotations

from collections.abc import Mapping


# 功能：
#   创建独立的双语操作标签；标签仅供界面识别，不执行插件或资产操作。
# 输入：
#   action_id：界面识别的操作标识。
#   zh：简体中文标签。
#   en：英文标签。
# 输出：
#   action：包含操作标识和两种语言标签的字典。
def _action(action_id: str, zh: str, en: str) -> dict[str, str]:
    action = {"id": action_id, "zh-CN": zh, "en-US": en}
    return action


_INPUT_GUIDANCE: dict[str, tuple[str, str, str, str, list[dict[str, str]]]] = {
    "asset_kind": (
        "source.asset_kind",
        "请选择这是地图、世界还是无人机。",
        "Choose whether this source is a map, world, or aircraft.",
        "来源本身没有足够信息让系统安全判断资产类型。",
        [_action("select-kind", "选择资产类型", "Choose asset kind")],
    ),
    "local_qualification_run": (
        "qualification.runtime",
        "请在已安装的 Runtime 中运行 Gazebo、ROS 2 与 PX4 资格流水线。",
        "Run the Gazebo, ROS 2, and PX4 qualification pipeline in the installed Runtime.",
        "可飞行性必须由真实运行证据证明，不能仅凭文件结构推断。",
        [_action("start-qualification", "开始真实仿真认证", "Start real simulation qualification")],
    ),
    "qualification_evidence": (
        "qualification.evidence",
        "请完成资格运行并生成绑定当前资产哈希的证据包。",
        "Complete qualification and produce evidence bound to the current asset hash.",
        "当前包没有足以支持可飞行或 Qualified 状态的证据。",
        [_action("view-qualification", "查看资格认证", "Open qualification")],
    ),
    "qualification_environment_versions": (
        "qualification.environment_versions",
        "请在当前 Gazebo、ROS 2、PX4 与策略版本下重新认证。",
        "Requalify with the current Gazebo, ROS 2, PX4, and policy versions.",
        "既有证据与当前运行环境不一致，资格已失效。",
        [_action("requalify", "重新认证", "Requalify")],
    ),
}


# 功能：
#   按错误码选择诊断说明，优先解释安全隔离和具体来源，再匹配资产类型。
#   未识别错误保持失败提示；本函数既不修改任务状态，也不签发飞行资格。
# 输入：
#   code：已验证为非空字符串的错误码。
# 输出：
#   profile：严重等级、阶段、位置、中英文标题和详情、操作标签组成的元组。
def _profile(code: str) -> tuple[str, str, str, str, str, str, str, list[dict[str, str]]]:
    # 精确安全错误先于普通来源匹配，避免隔离失败被描述成可直接重试的下载错误。
    if code in {
        "ASSET_SOURCE_EXECUTABLE_MEMBER_FORBIDDEN",
        "ASSET_SOURCE_SYMLINK_FORBIDDEN",
        "ASSET_SOURCE_MEMBER_PATH_INVALID",
        "ASSET_SOURCE_ENCRYPTED_MEMBER_FORBIDDEN",
        "DDPKG_EXECUTABLE_MEMBER_FORBIDDEN",
        "DDPKG_SYMLINK_FORBIDDEN",
        "DDPKG_MEMBER_PATH_INVALID",
        "ASSET_BUNDLE_PATH_TRAVERSAL",
    }:
        profile = (
            "critical",
            "quarantine",
            "source.package",
            "外部资产被安全隔离",
            "External asset blocked in quarantine",
            "资产包含可执行内容、越界路径、符号链接或其他不允许进入运行时的成员。请在来源软件中导出纯声明式资产后重试。",
            "The package contains executable content, escaping paths, symbolic links, "
            "or another member that cannot enter the Runtime. Export a declarative-only "
            "asset and try again.",
            [_action("replace-source", "重新选择安全的来源", "Choose a safe source")],
        )
        return profile
    if code in {
        "ASSET_REMOTE_HTTPS_REQUIRED",
        "ASSET_REMOTE_PRIVATE_NETWORK_FORBIDDEN",
        "ASSET_REMOTE_HOST_INVALID",
    }:
        profile = (
            "critical",
            "acquiring",
            "source.location",
            "远程来源被安全策略阻止",
            "Remote source blocked by security policy",
            "仅允许不含账号密码的公网 HTTPS 来源。本机、局域网、链接本地地址和 "
            "file/ssh 等协议必须通过受信任连接器访问。",
            "Only public HTTPS sources without embedded credentials are allowed. Local, "
            "private, link-local, file, and SSH sources require a trusted connector.",
            [
                _action("use-connector", "使用受信任连接器", "Use a trusted connector"),
                _action("replace-url", "更换来源地址", "Change source URL"),
            ],
        )
        return profile
    if code.startswith("ASSET_REMOTE_GIT_"):
        profile = (
            "error",
            "acquiring",
            "source.git",
            "Git 来源无法安全导入",
            "Git source could not be imported safely",
            "检查仓库地址、分支或子目录。仓库必须公开可读，并且不能依赖子模块、交互式凭据、符号链接或超过安全预算的文件。私有仓库请使用凭据隔离的连接器。",
            "Check the repository URL, ref, and subpath. The repository must be publicly "
            "readable and cannot depend on submodules, interactive credentials, symbolic "
            "links, or files beyond the safety budget. Use a credential-isolated connector "
            "for private repositories.",
            [
                _action("edit-git-source", "修改 Git 来源", "Edit Git source"),
                _action("use-connector", "使用私有仓库连接器", "Use private-repository connector"),
            ],
        )
        return profile
    if code.startswith("ASSET_REMOTE_"):
        profile = (
            "error",
            "acquiring",
            "source.remote",
            "远程资产下载失败",
            "Remote asset download failed",
            "检查公网 HTTPS 地址、文件大小和可选 SHA-256。下载结果必须完整进入隔离区"
            "并通过哈希检查后才能解析。",
            "Check the public HTTPS URL, file size, and optional SHA-256. The download must "
            "enter quarantine completely and pass hash verification before parsing.",
            [
                _action("retry-download", "重试下载", "Retry download"),
                _action("import-file", "改为本地导入", "Import a local file"),
            ],
        )
        return profile
    if "COMPRESSION" in code or "SIZE_EXCEEDED" in code or "MEMBER_LIMIT" in code:
        profile = (
            "error",
            "quarantine",
            "source.package",
            "资产包超出安全预算",
            "Asset package exceeds the safety budget",
            "请降低网格、纹理或归档复杂度，并确保压缩包不是递归归档或压缩炸弹。",
            "Reduce mesh, texture, or archive complexity and ensure the package is not "
            "recursive or a compression bomb.",
            [_action("optimize-source", "优化来源资产", "Optimize source asset")],
        )
        return profile
    if (
        code in {"ASSET_SOURCE_ADAPTER_REQUIRED", "ASSET_SOURCE_ADAPTER_UNAVAILABLE"}
        or "ADAPTER" in code
    ):
        profile = (
            "error",
            "parsing",
            "source.adapter",
            "需要匹配的来源适配器",
            "A matching source adapter is required",
            "启用对应连接器，或在外部建模软件中导出经过该连接器签名的 `.ddpkg`。"
            "未知插件不会被自动执行。",
            "Enable the matching connector, or export a connector-signed `.ddpkg` from the "
            "external modeling tool. Unknown plugins are never executed automatically.",
            [
                _action("open-plugins", "打开插件", "Open plugins"),
                _action("submit-result", "提交转换结果", "Submit converted package"),
            ],
        )
        return profile
    if "FORMAT" in code or "ENTRYPOINT" in code or "XML" in code or "MANIFEST" in code:
        profile = (
            "error",
            "parsing",
            "source.format",
            "无法解析资产结构",
            "Asset structure could not be parsed",
            "请检查入口 SDF、URDF、Xacro 或 manifest，并确认所有引用使用包内相对路径。",
            "Check the entry SDF, URDF, Xacro, or manifest and ensure every reference uses "
            "a package-relative path.",
            [
                _action("view-source", "检查来源文件", "Inspect source files"),
                _action("replace-source", "重新选择来源", "Choose another source"),
            ],
        )
        return profile
    if "KIND" in code or "IDENTITY" in code:
        profile = (
            "error",
            "normalizing",
            "identity",
            "资产身份或类型不一致",
            "Asset identity or kind is inconsistent",
            "来源、转换结果和目标仓库必须声明相同的稳定资产 ID 与类型。",
            "The source, converted package, and target library must declare the same stable "
            "asset ID and kind.",
            [_action("review-identity", "检查资产身份", "Review asset identity")],
        )
        return profile
    if code.startswith("MAP_"):
        profile = (
            "error",
            "validating",
            "map.qualification",
            "地图资格检查未通过",
            "Map qualification check failed",
            "检查几何连接、碰撞、道路/楼梯连续性、净空、语义图、任务可达性及对应数值证据。",
            "Review geometric seams, collision, road/stair continuity, clearance, semantic "
            "graph, task reachability, and the associated numeric evidence.",
            [
                _action("view-evidence", "查看地图证据", "View map evidence"),
                _action("repair-source", "返回来源软件修复", "Repair in source tool"),
            ],
        )
        return profile
    if code.startswith("VEHICLE_"):
        profile = (
            "error",
            "validating",
            "vehicle.qualification",
            "无人机资格检查未通过",
            "Aircraft qualification check failed",
            "检查质量、惯量、碰撞体、执行器映射、推重比、传感器、载荷接口和 PX4 参数来源。",
            "Review mass, inertia, collision geometry, actuator mapping, thrust-to-weight "
            "ratio, sensors, payload interface, and PX4 parameter provenance.",
            [
                _action("view-evidence", "查看飞行证据", "View flight evidence"),
                _action("repair-source", "返回来源软件修复", "Repair in source tool"),
            ],
        )
        return profile
    if code.startswith("ASSET_QUALIFICATION_") or code.startswith("QUALIFICATION_"):
        profile = (
            "error",
            "qualification",
            "qualification.evidence",
            "真实仿真认证未完成",
            "Real-simulation qualification is incomplete",
            "请查看失败的 Gazebo、ROS 2、PX4 门禁和证据；修复后使用相同的资产版本重新运行。",
            "Inspect the failed Gazebo, ROS 2, PX4 gates and evidence, then rerun the same "
            "asset version after repair.",
            [
                _action("view-evidence", "查看认证证据", "View qualification evidence"),
                _action("retry", "重新运行", "Run again"),
            ],
        )
        return profile
    profile = (
        "error",
        "unknown",
        "job",
        "资产处理需要关注",
        "Asset processing needs attention",
        "查看错误码和该阶段生成的证据；问题未解决前，资产不会进入正式任务。",
        "Review the issue code and evidence produced by this stage. The asset cannot enter "
        "a formal mission until the issue is resolved.",
        [_action("view-details", "查看详情", "View details")],
    )
    return profile


# 功能：
#   将非空错误码整理成双语诊断卡，保留未知错误码供进一步追踪。
# 输入：
#   code：导入或资格检查产生的错误码。
# 输出：
#   issue：含错误身份、说明位置和操作标签的独立诊断字典。
def describe_asset_issue(code: str) -> dict[str, object]:
    if not isinstance(code, str) or not code:
        raise ValueError("ASSET_ISSUE_CODE_INVALID")
    severity, stage, location, title_zh, title_en, detail_zh, detail_en, actions = _profile(code)
    issue = {
        "schema_version": "dronedream.asset-issue.v1",
        "code": code,
        "severity": severity,
        "stage": stage,
        "location": location,
        "title": {"zh-CN": title_zh, "en-US": title_en},
        "detail": {"zh-CN": detail_zh, "en-US": detail_en},
        "actions": actions,
    }
    return issue


# 功能：
#   解释需要用户补充的结构化输入；未知字段显示通用提示，不猜测工程参数。
#   每次返回独立操作标签，防止界面修改诊断卡后污染后续请求的共享说明。
# 输入：
#   value：任务缺失的非空字段标识。
# 输出：
#   issue：包含双语说明、字段位置和补充输入操作的警告卡。
def describe_required_input(value: str) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        raise ValueError("ASSET_REQUIRED_INPUT_INVALID")
    location, title_zh, title_en, detail_zh, actions = _INPUT_GUIDANCE.get(
        value,
        (
            f"required_inputs.{value}",
            "需要补充结构化输入。",
            "Additional structured input is required.",
            "请按字段要求补全信息；系统不会猜测缺失的工程参数。",
            [_action("provide-input", "补充信息", "Provide input")],
        ),
    )
    detail_en = {
        "asset_kind": (
            "The source does not contain enough information to determine the asset kind safely."
        ),
        "local_qualification_run": (
            "Flight readiness must be proven by runtime evidence rather than inferred from files."
        ),
        "qualification_evidence": (
            "The package does not contain evidence sufficient for flight-ready or Qualified status."
        ),
        "qualification_environment_versions": (
            "Existing evidence does not match the current Runtime, so qualification is stale."
        ),
    }.get(
        value,
        "Complete the requested field. The system will not guess missing engineering parameters.",
    )
    issue = {
        "schema_version": "dronedream.asset-issue.v1",
        "code": f"REQUIRED_INPUT_{value.upper()}",
        "severity": "warning",
        "stage": "needs_input",
        "location": location,
        "title": {"zh-CN": title_zh, "en-US": title_en},
        "detail": {"zh-CN": detail_zh, "en-US": detail_en},
        # 模板的叶子值全是字符串，只需逐项复制字典，不能直接暴露模板列表。
        "actions": [dict(action) for action in actions],
    }
    return issue


# 功能：
#   在去重前检查诊断列表的类型与数量，拒绝字符串按字符拆分或对象隐式转文本。
#   保留旧记录中空字符串被忽略的行为，不修改调用者的原始列表。
# 输入：
#   value：任务记录中的列表字段。
#   limit：现有任务契约允许的原始条目数上限。
# 输出：
#   labels：按首次出现顺序去重的非空文本列表。
def _issue_labels(value: object, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) > limit:
        raise ValueError("ASSET_ISSUE_LIST_INVALID")
    if any(not isinstance(item, str) for item in value):
        raise ValueError("ASSET_ISSUE_LIST_INVALID")
    labels = list(dict.fromkeys(item for item in value if item))
    return labels


# 功能：
#   将任务快照转换成按来源顺序展示的错误卡和缺失输入卡，不更改任务或资格状态。
#   缺省字段保留兼容值；显式非法类型和越界进度报错，不能伪装成正常记录。
# 输入：
#   job：资产导入或资产组合资格任务的映射快照。
# 输出：
#   report：任务身份、状态、整数进度和诊断卡列表组成的显示报告。
def asset_job_issue_report(job: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(job, Mapping):
        raise ValueError("ASSET_ISSUE_JOB_INVALID")
    job_id = job.get("job_id", "")
    state = job.get("state", "unknown")
    progress_percent = job.get("progress_percent", 0)
    if not isinstance(job_id, str) or not isinstance(state, str):
        raise ValueError("ASSET_ISSUE_JOB_INVALID")
    if type(progress_percent) is not int or not 0 <= progress_percent <= 100:
        raise ValueError("ASSET_ISSUE_PROGRESS_INVALID")
    # 导入任务最多 1000 个错误和 256 个缺失项；资格任务也在此较宽预算内。
    issue_codes = _issue_labels(job.get("issue_codes", []), 1_000)
    required_inputs = _issue_labels(job.get("required_inputs", []), 256)
    issues = [describe_asset_issue(code) for code in issue_codes]
    issues.extend(describe_required_input(value) for value in required_inputs)
    report = {
        "schema_version": "dronedream.asset-issue-report.v1",
        "job_id": job_id,
        "state": state,
        "progress_percent": progress_percent,
        "issues": issues,
    }
    return report
