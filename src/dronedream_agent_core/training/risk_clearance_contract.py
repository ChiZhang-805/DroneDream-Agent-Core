"""Explicit offline risk-label semantics; never an online safety override."""

import math

LEGACY_CLEARANCE_LABELS = "configured-clearance-v1"
OBSERVED_CLEARANCE_LABELS = "observation-clearance-v2"


# 功能：从训练与部署共用的第十三列恢复请求净空，拒绝缺失、饱和或非数值条件。
# 输入：state_features：未改写的四十六维状态特征。
# 输出：clearance：请求保留的机体外净空，单位米，不包含传感器真值。
def observed_required_clearance(state_features):
    if type(state_features) not in (list, tuple) or len(state_features) != 46:
        raise ValueError("ACTION_RISK_CLEARANCE_FEATURES_INVALID")
    value = state_features[13]
    if type(value) not in (int, float) or not 0 < value < 1:
        raise ValueError("ACTION_RISK_CLEARANCE_CONDITION_UNAVAILABLE")
    return float(value * 20.)


# 功能：把已扣除不确定度的扫掠净空转换成与运行阈值一致的离线监督分数。
# 输入：clearance_lower_bound_m：独立几何教师得到的保守净空；required_clearance_m：原请求净空。
# 输出：score：零净空为一，所需净空处为零点五，两倍所需净空及以上为零；不是碰撞概率。
def clearance_risk_score(clearance_lower_bound_m, required_clearance_m):
    if (type(clearance_lower_bound_m) not in (int, float)
            or not math.isfinite(clearance_lower_bound_m)
            or type(required_clearance_m) not in (int, float)
            or not 0 < required_clearance_m < 20):
        raise ValueError("ACTION_RISK_CLEARANCE_SCORE_INPUT_INVALID")
    return max(0., min(1., 1. - clearance_lower_bound_m / (2. * required_clearance_m)))


# 功能：生成可重算的标签定义，绑定真实请求净空、分类边界和原几何下界。
# 输入：state_features：原状态；clearance_lower_bound_m：未变更的离线预测结果。
# 输出：score、contract：新分数及必须随教师回执保存的标签契约。
def observation_clearance_label(state_features, clearance_lower_bound_m):
    required = observed_required_clearance(state_features)
    score = clearance_risk_score(clearance_lower_bound_m, required)
    contract = dict(kind=OBSERVED_CLEARANCE_LABELS,
                    clearance_source="state_features[13]*20m",
                    required_clearance_m=required,
                    unsafe_score_threshold=.5,
                    boundary_policy="clearance-at-or-below-required-is-unsafe",
                    score_definition="clip(1-clearance_lower_bound/(2*required_clearance),0,1)")
    return score, contract
