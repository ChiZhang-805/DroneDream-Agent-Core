"""Bounded processing-age evidence, separate from all motion authorization."""

import copy


class PerceptionSourceDiagnostics:
    # 功能：
    #   初始化常量容量的阶段年龄和编码拒绝统计，不保存图片或改变控制判定。
    # 输入：
    #   无。
    # 输出：
    #   无。
    def __init__(self):
        self._ages = {}
        self._issues = {}

    # 功能：
    #   记录原始来源到实际阶段时刻的年龄；未来时钟单列，不裁成零或改写来源。
    # 输入：
    #   stage：选帧或健康发布阶段；now_ms：实际消费时刻。
    #   depth_ms、state_ms：当前真实深度、状态来源时刻。
    # 输出：
    #   无。
    def record(self, stage: str, *, now_ms: int, depth_ms: int, state_ms: int):
        if stage not in ("selected", "health") or any(
                type(value) is not int or not 0 <= value < 2**63
                for value in (now_ms, depth_ms, state_ms)):
            raise ValueError("PERCEPTION_DIAGNOSTIC_CLOCK_INVALID")
        for source, stamp in (("depth", depth_ms), ("state", state_ms)):
            age = now_ms - stamp
            key = stage + ":" + source
            row = self._ages.setdefault(key, {"count": 0, "total_ms": 0, "maximum_ms": 0,
                                               "future_count": 0, "over_250_ms_count": 0})
            if age < 0:
                row["future_count"] += 1
                continue
            row["count"] += 1
            row["total_ms"] += age
            row["maximum_ms"] = max(row["maximum_ms"], age)
            row["over_250_ms_count"] += int(age > 250)

    # 功能：
    #   累计每轮融合的有限编码阻断原因，不因任务时长或未知诊断增长内存。
    # 输入：
    #   issues：内部融合快照的诊断列表。
    # 输出：
    #   无。
    def record_feature_issues(self, issues: list[str]):
        if type(issues) is not list:
            raise ValueError("PERCEPTION_DIAGNOSTIC_ISSUES_INVALID")
        bounded = issues[:8]
        if any(type(code) is not str or len(code) > 192 for code in bounded):
            raise ValueError("PERCEPTION_DIAGNOSTIC_ISSUES_INVALID")
        for code in dict.fromkeys(bounded):
            key = code if code in self._issues or len(self._issues) < 32 else "OTHER"
            self._issues[key] = self._issues.get(key, 0) + 1

    # 功能：
    #   冻结诊断副本供退出时持久化，不参与来源续期、训练接纳或飞行授权。
    # 输入：
    #   无。
    # 输出：
    #   summary：来源年龄与编码阻断计数。
    def snapshot(self):
        summary = {"schema_version": "dronedream.perception-source-diagnostics.v1",
                   "source_ages": copy.deepcopy(self._ages),
                   "feature_issue_cycle_counts": dict(self._issues),
                   "qualification_granted": False}
        return summary
