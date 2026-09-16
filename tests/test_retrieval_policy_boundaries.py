"""Exercise the actual preparation policy guard without model calls or persistent sessions."""

import pytest

from dronedream_agent_core.orchestrator import (
    MissionPreparationBlocked,
    PreparationConfig,
    _retrieval_event_limit,
)


@pytest.mark.parametrize("value", [True, False, "24", 1.5, 0, -1, 201, None, float("nan")])
def test_invalid_retrieval_limit_is_not_coerced(value):
    """Invalid policy cannot silently reduce or expand what the model reads."""
    with pytest.raises(MissionPreparationBlocked, match="CONTEXT_RETRIEVAL_POLICY_INVALID"):
        _retrieval_event_limit({"maximum_recent_events": value})


@pytest.mark.parametrize("value", [1, 24, 200])
def test_valid_retrieval_limit_is_preserved(value):
    """Preserve the declared bounded scope, including both supported endpoints."""
    assert _retrieval_event_limit({"maximum_recent_events": value}) == value
    assert _retrieval_event_limit({}) == 24


@pytest.mark.parametrize(
    "field",
    [
        "max_provider_attempts",
        "max_intent_rounds",
        "max_planning_rounds",
        "plugin_router_rounds",
        "maximum_plugin_calls",
        "intent_reviews_per_round",
        "plan_reviews_per_round",
        "maximum_model_calls",
        "maximum_optional_tool_calls",
    ],
)
@pytest.mark.parametrize("value", [True, 1.5, "2", float("nan")])
def test_preparation_counts_are_exact_integers(field, value):
    """Budgets cannot pass comparison checks then fail later in range/provider calls."""
    with pytest.raises(ValueError, match=field):
        PreparationConfig(provider="kimi", critic_provider="kimi", **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "model_timeout_seconds",
        "vehicle_diameter_m",
        "vehicle_height_m",
        "waypoint_hold_seconds",
    ],
)
@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan")])
def test_preparation_physical_values_are_finite(field, value):
    """Invalid dimensions/timeouts must not enter clearance math or timer setup."""
    with pytest.raises(ValueError, match=field):
        PreparationConfig(provider="kimi", critic_provider="kimi", **{field: value})


def test_persisted_context_flag_is_not_truthiness_consent():
    """String false is not permission to persist task context; zero hold remains supported."""
    with pytest.raises(ValueError, match="persisted_task_context"):
        PreparationConfig(provider="kimi", critic_provider="kimi", persisted_task_context="false")
    assert PreparationConfig(provider="kimi", critic_provider="kimi", waypoint_hold_seconds=0)
