from __future__ import annotations

from argparse import Namespace

import pytest

from dronedream_agent_core.cli import (
    CliProfileError,
    _execute_prepared_mission,
    _export_px4_track,
    _intent_critic,
    _model_probe,
    _prepare_mission,
    _require_development_profile,
    _require_runtime_manager_profile,
    _reverify_prepared_run,
    _run_px4_track,
    _submit_runtime_message,
)


@pytest.mark.parametrize(
    "handler",
    (
        _model_probe,
        _intent_critic,
        _export_px4_track,
        _run_px4_track,
        _prepare_mission,
        _submit_runtime_message,
        _reverify_prepared_run,
    ),
)
def test_direct_model_orchestrator_and_runtime_handlers_are_locked_by_default(handler) -> None:
    with pytest.raises(CliProfileError, match="DEVELOPMENT_OR_TEST_PROFILE_REQUIRED"):
        handler(Namespace(profile="locked"))


def test_canonical_execution_handler_requires_runtime_manager_profile() -> None:
    with pytest.raises(CliProfileError, match="RUNTIME_MANAGER_PROFILE_REQUIRED"):
        _execute_prepared_mission(Namespace(profile="locked"))


def test_explicit_profiles_are_narrowly_accepted() -> None:
    _require_development_profile(Namespace(profile="development"))
    _require_development_profile(Namespace(profile="test"))
    _require_runtime_manager_profile(Namespace(profile="runtime-manager"))
    _require_runtime_manager_profile(Namespace(profile="test"))
    with pytest.raises(CliProfileError, match="DEVELOPMENT_OR_TEST_PROFILE_REQUIRED"):
        _require_development_profile(Namespace(profile="runtime-manager"))
