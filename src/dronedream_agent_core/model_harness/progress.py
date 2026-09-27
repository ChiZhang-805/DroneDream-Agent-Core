"""Public execution summaries only; never raw prompts or private model reasoning."""
from contextvars import ContextVar

progress_sink: ContextVar = ContextVar("public_preparation_progress", default=None)


# 功能：
#   向当前准备请求报告可公开的事实摘要，显示故障不能改变业务结果。
# 输入：
#   stage：实际阶段；zh：中文摘要；en：英文摘要。
# 输出：
#   无返回值。
def report_progress(stage: str, zh: str, en: str) -> None:
    sink = progress_sink.get()
    _emit(sink, stage, zh, en)


# 功能：
#   隔离非业务观察器异常，进度展示失败不得使已完成的模型调用失败或重试扣费。
# 输入：
#   sink：观察器；stage、zh、en：公开阶段事件。
# 输出：
#   无返回值。
def _emit(sink, stage: str, zh: str, en: str) -> None:
    if sink is not None:
        try:
            sink(stage, zh, en)
        except Exception:
            # 进度是旁路；业务的错误处理和验收结果由调用链本身保存。
            pass


ROLE_STAGES = {
    "intent_parser": ("intent", "理解任务", "Understanding the task"),
    "intent_critic": ("intent_review", "复核任务要求", "Reviewing task requirements"),
    "plugin_router": ("tools", "选择任务工具", "Selecting task tools"),
    "task_decomposer": ("decomposition", "分解任务步骤", "Decomposing the task"),
    "global_planner": ("route", "规划路线", "Planning the route"),
    "plan_critic": ("verification", "复核计划", "Reviewing the plan"),
    "context_summarizer": ("interpretation", "整理地图与任务知识", "Organizing asset and task knowledge"),
    "completion_verifier": ("verification", "验证完成条件", "Verifying completion conditions"),
}

# 公开工作范围，不是模型内部推理；开始请求时只描述待核对内容。
ROLE_DETAILS = {
    "intent_parser": ("将自然语言整理为任务目标、起点、目的地、返回位置和附加约束。会区分用户明确提出的条件与仍需确认的信息，例如停留时长、是否返回、是否要求等待确认。地点名称必须与地图中的实体对应，不能用猜出的坐标补齐；这里尚未得出可以起飞的结论。", "Converting the request into goal, start, destination, return location and constraints. Explicit requirements are separated from missing facts, including dwell time and confirmation. Locations must bind to map entities, not invented coordinates; no takeoff decision has been made."),
    "intent_critic": ("复核提取后的目标是否与用户原话一致，检查地点歧义、遗漏条件和互相冲突的要求。如果缺少决定任务可行性的关键信息，应返回追问，而不是悄悄替用户决定。本次复核只判断任务表达，实际空间、路线和机型能力还需要后续工具验证。", "Reviewing the extracted goal for ambiguity, omissions and conflicting requirements. Critical unknowns require clarification rather than silent assumptions. This review concerns task meaning; geometry, routes and vehicle capability need later tool checks."),
    "plugin_router": ("根据任务需要识别可能用到的工具和载荷操作，并核对哪些能力由当前安装的插件提供。工具名称、输入和输出都必须符合接口约定；模型提出使用某个工具不代表工具已成功运行，也不能凭文字声明无人机具备未安装的取物能力。", "Identifying required tools and payload operations against installed plugin capabilities. Names, inputs and outputs must match their contracts. Proposing a tool is not executing it, and text cannot establish missing pickup hardware."),
    "task_decomposer": ("把目标拆成有先后依赖的任务步骤，分别描述每一步的目标和完成证据。涉及取物时，还需表达接近、停留、载荷确认与返回之间的关系；具体采用哪些步骤由当前任务与能力约束决定，不套用固定四步模板。只有依赖满足，后续步骤才具备执行条件。", "Decomposing the goal into dependent steps with targets and completion evidence. Pickup may involve approach, dwell, custody and return dependencies according to the current task and capabilities, not a fixed four-step template. Later steps require their prerequisites."),
    "global_planner": ("结合已解析的地点功能、地图连接关系和任务步骤，提出有顺序的候选目标。候选路线仍需转换成真实米制空间中的路径，检查室内外过渡、楼梯和通行间隙；模型不能仅凭地点名称宣布路线安全。可飞行偏好区域用于规划参考，不替代障碍物几何检查。", "Proposing ordered targets using interpreted places, map connectivity and task steps. Candidates still require metric routing and checks of transitions, stairs and clearance. Place names and preferred corridors do not establish collision safety."),
    "plan_critic": ("结合任务条件和已生成的路线结果复核计划，检查目标是否遗漏、步骤是否矛盾，以及存在的问题是否需要重新规划。模型复核通过不等于执行许可；界面还必须验证本次计划的资产绑定、内容摘要与执行前条件，随后等待用户确认。", "Reviewing the candidate for missing goals, contradictions and reasons to replan. Model approval is not execution authorization: asset bindings, content digests and pre-execution conditions must still be checked before user confirmation."),
}


# 功能：
#   从已返回的公开结构化产物提取任务事实摘要，不展示推理文本或完整模型输入。
# 输入：
#   sink：当前请求观察器；artifact：经过结构验证的业务产物。
# 输出：
#   无返回值。
def report_artifact_progress(sink, artifact) -> None:
    from dronedream_agent_core.contracts import IntentArtifact, SemanticPlan, TaskGraphArtifact, PlanCritique
    if isinstance(artifact, IntentArtifact):
        goal = artifact.goal[:240]
        _emit(sink, "intent", f"模型提取的任务目标：{goal}。起点：{artifact.start_entity}；目的地：{artifact.target_entity}；返回点：{artifact.return_entity}。仍需核对地图实体与约束。", f"Extracted goal: {goal}. Start: {artifact.start_entity}; destination: {artifact.target_entity}; return: {artifact.return_entity}. Map references and constraints still need checking.")
        _emit(sink, "intent", f"已提取 {len(artifact.constraints)} 条任务约束，{len(artifact.missing_critical_fields)} 项关键信息待补充；缺失信息不会被自动当作已确认。", f"Extracted {len(artifact.constraints)} constraints; {len(artifact.missing_critical_fields)} critical fields remain missing. Missing facts are not treated as confirmed.")
    elif isinstance(artifact, SemanticPlan):
        targets = " → ".join(artifact.ordered_targets[:12])
        _emit(sink, "route", f"候选路线顺序：{targets}。共 {len(artifact.ordered_targets)} 个目标，采用 {artifact.route_policy} 策略；这只是候选，尚不是飞行许可。", f"Proposed order: {targets}. {len(artifact.ordered_targets)} targets; {artifact.route_policy} policy. This remains a candidate, not flight authorization.")
    elif isinstance(artifact, TaskGraphArtifact):
        _emit(sink, "decomposition", f"任务已拆成 {len(artifact.graph.nodes)} 个步骤，接下来核对步骤依赖、所需工具和完成条件。", f"Created {len(artifact.graph.nodes)} task steps; dependencies, required tools and completion conditions are being checked next.")
    elif isinstance(artifact, PlanCritique):
        _emit(sink, "verification", f"计划复核模型已返回 {len(artifact.issue_codes)} 项问题记录；模型意见还需与程序的确定性校验共同判定。", f"Plan review returned {len(artifact.issue_codes)} issue records; model feedback must be combined with deterministic checks.")


# 功能：
#   报告真实模型调用的角色、尝试次数和完成状态，不输出模型内部推理或凭据。
# 输入：
#   sink：请求绑定的观察器；role：模型角色；attempt：尝试序号；state：调用状态；elapsed_ms：实际耗时。
# 输出：
#   无返回值。
def report_model_progress(sink, role: str, attempt: int, state: str, elapsed_ms: int | None = None) -> None:
    if sink is None:
        return
    stage, zh, en = ROLE_STAGES.get(role, ("model", "处理结构化任务信息", "Processing structured task information"))
    if state == "started":
        _emit(sink, stage, f"正在{zh}：第 {attempt} 次模型请求已开始，等待模型返回结构化结果。", f"{en}: model attempt {attempt} has started; waiting for structured output.")
        if role in ROLE_DETAILS:
            detail_zh, detail_en = ROLE_DETAILS[role]
            _emit(sink, stage, detail_zh, detail_en)
    elif state == "completed":
        _emit(sink, stage, f"{zh}的第 {attempt} 次请求已返回，输出已通过结构校验；后续仍需结合几何和任务约束检查。", f"{en}: attempt {attempt} returned schema-validated output. Geometry and mission constraints still require validation.")
        if elapsed_ms is not None:
            _emit(sink, stage, f"本次模型请求与结构处理实际用时 {elapsed_ms / 1000:.1f} 秒，正在把已验证的结构化结果交给后续阶段。", f"This model request and schema processing took {elapsed_ms / 1000:.1f} s; handing the validated structured result to the next stage.")
    else:
        _emit(sink, stage, f"{zh}的第 {attempt} 次请求未成功，正在按调用预算处理失败。", f"{en}: attempt {attempt} failed; the bounded retry policy is handling the failure.")
