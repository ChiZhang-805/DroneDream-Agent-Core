"""Synthetic contract fixtures, never flight qualification or training evidence."""

from dronedream_agent_core.control_feature_contract import (
    DYNAMIC_TARGET_FEATURE_COUNT,
    GEOMETRY_FEATURE_COUNT,
    encoder_contract_sha256,
)
from dronedream_agent_core.local_policy_quality import (
    LocalPolicyTrainingMetrics,
    summarize_pilot_axes,
)
from dronedream_agent_core.realtime_feature_encoders import (
    RealtimeFeatureEncoding,
    RealtimeFeatureSnapshot,
    fuse_realtime_features,
)


def complete_feature_snapshot(timestamp: int = 1_000) -> RealtimeFeatureSnapshot:
    encodings = []
    for role, width in (
        ("metric-geometry-encoder", GEOMETRY_FEATURE_COUNT),
        ("dynamic-target-encoder", DYNAMIC_TARGET_FEATURE_COUNT),
        ("flight-state-encoder", 20),
    ):
        features = [0.0] * width
        if role == "flight-state-encoder":
            features[0] = 1.0
        encodings.append(
            RealtimeFeatureEncoding(
                encoder_role=role,
                feature_contract_sha256=encoder_contract_sha256(role),
                source_ids=["contract-fixture"],
                source_sha256="a" * 64,
                observed_at_unix_ms=timestamp,
                encoded_at_unix_ms=timestamp,
                maximum_age_milliseconds=250,
                quality=1.0,
                coverage=1.0,
                uncertainty=0.0,
                features=features,
                valid_mask=[1.0] * width,
                inference_latency_ms=0.1,
            )
        )
    return fuse_realtime_features(encodings, captured_at_unix_ms=timestamp)


def qualified_pilot_metrics() -> LocalPolicyTrainingMetrics:
    targets = [[0.5] * 4, [-0.5] * 4] * 20
    return LocalPolicyTrainingMetrics(
        sample_count=100,
        risky_sample_count=40,
        safe_sample_count=60,
        motion_sample_count=40,
        non_motion_sample_count=60,
        authorized_motion_recall=1.0,
        non_motion_recall=1.0,
        action_accuracy=1.0,
        motion_authorization_accuracy=1.0,
        candidate_selection_accuracy=1.0,
        risk_hold_recall=1.0,
        safe_motion_recall=1.0,
        risk_mean_absolute_error=0.0,
        mean_cross_entropy=0.0,
        pilot_control_mean_absolute_error=0.0,
        pilot_axis_evidence=summarize_pilot_axes(targets, targets),
    )
