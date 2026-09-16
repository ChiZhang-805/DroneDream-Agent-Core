"""Offline transport/retry/vision-boundary regressions; no paid provider calls or real API keys."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from dronedream_agent_core.contracts import IntentArtifact, TaskGraphArtifact
from dronedream_agent_core.model_harness.model_port import (
    FailoverStructuredModelPort,
    ModelConfigurationError,
    ModelInvocationError,
    ProviderSettings,
    StructuredModelPort,
    _safe_attempt_diagnostic,
)


def _port() -> StructuredModelPort:
    """Use an invalid-domain fixture provider; tests replace invocation before transport I/O."""
    return StructuredModelPort(
        "test-provider",
        max_attempts=3,
        settings=ProviderSettings(
            name="test-provider",
            model="test-model",
            api_key_env="TEST_PROVIDER_API_KEY",
            base_url="https://example.invalid/v1",
            api_style="responses",
        ),
        api_key="not-a-real-secret",
    )


def test_model_invocation_reason_code_is_machine_safe() -> None:
    """Persist a bounded machine code, not arbitrary provider text, as failure classification."""
    error = ModelInvocationError(
        "safe operator-facing diagnostic",
        attempts_used=1,
        reason_code="LOCAL_POLICY_QUALIFIED_LATENCY_EXCEEDED",
    )

    assert error.reason_code == "LOCAL_POLICY_QUALIFIED_LATENCY_EXCEEDED"
    with pytest.raises(ValueError, match="uppercase machine code"):
        ModelInvocationError(
            "unsafe external value",
            attempts_used=1,
            reason_code="1_non-machine-code",
        )


def test_openai_empty_optional_environment_values_use_safe_defaults(monkeypatch) -> None:
    """Blank optional settings retain the supported default rather than an empty model ID."""
    monkeypatch.setenv("OPENAI_API_STYLE", "")
    monkeypatch.setenv("OPENAI_MODEL", "")

    settings = ProviderSettings.from_env("openai")

    assert settings.api_style == "responses"
    assert settings.model == "gpt-5.4"
    assert settings.supports_image_input is True


def test_unknown_openai_environment_model_does_not_inherit_image_capability(
    monkeypatch,
) -> None:
    """Unknown compatible models must explicitly declare multimodal capability."""
    monkeypatch.setenv("OPENAI_MODEL", "unknown-compatible-model")

    assert ProviderSettings.from_env("openai").supports_image_input is False


def test_text_only_port_rejects_image_before_provider_invocation() -> None:
    """Fail at the capability boundary before reading media or invoking a provider."""
    port = _port()

    with pytest.raises(ModelConfigurationError, match="does not declare image input support"):
        port.call(
            role="intent_parser",
            output_type=IntentArtifact,
            instructions="Return structured intent.",
            input_artifact={"message": "inspect the image"},
            multimodal=[{"kind": "image-file", "path": "never-read.png"}],
        )


def test_failover_port_routes_timeout_recovery_to_text_only_peer() -> None:
    """Recovery uses a bounded peer streak and removes images unsupported by that peer."""
    class FakePort:
        def __init__(self, name: str, timeout_seconds: float) -> None:
            self.settings = SimpleNamespace(name=name)
            self.invocation_timeout_seconds = timeout_seconds
            self.calls: list[dict[str, object]] = []
            self.reset_count = 0

        def call(self, **kwargs):
            self.calls.append(kwargs)
            return self.settings.name

        def reset_transport(self) -> None:
            self.reset_count += 1

    primary = FakePort("openai", 10.0)
    fallback = FakePort("deepseek", 35.0)
    port = FailoverStructuredModelPort(primary, fallback)

    assert port.invocation_timeout_seconds == 10.0
    assert port.call(multimodal=[{"kind": "image-file"}]) == "openai"
    assert primary.calls[-1]["multimodal"] == [{"kind": "image-file"}]
    port.reset_transport()
    assert primary.reset_count == 1
    assert port.invocation_timeout_seconds == 35.0
    assert port.call(multimodal=[{"kind": "image-file"}]) == "deepseek"
    assert fallback.calls[-1]["multimodal"] == []
    assert port.invocation_timeout_seconds == 35.0
    assert port.call(multimodal=[{"kind": "image-file"}]) == "deepseek"
    assert port.invocation_timeout_seconds == 35.0
    assert port.call(multimodal=[{"kind": "image-file"}]) == "deepseek"
    assert port.invocation_timeout_seconds == 10.0
    assert port.call(multimodal=[{"kind": "image-file"}]) == "openai"


def test_failed_fallback_reenters_primary_instead_of_sticking_to_peer() -> None:
    """A failed recovery port must not trap subsequent calls on the same failed peer."""
    class FakePort:
        def __init__(self, name: str, timeout_seconds: float) -> None:
            self.settings = SimpleNamespace(name=name)
            self.invocation_timeout_seconds = timeout_seconds
            self.reset_count = 0

        def call(self, **_kwargs):
            return self.settings.name

        def reset_transport(self) -> None:
            self.reset_count += 1

    primary = FakePort("openai", 10.0)
    fallback = FakePort("deepseek", 35.0)
    port = FailoverStructuredModelPort(primary, fallback)

    port.reset_transport()
    assert port.invocation_timeout_seconds == 35.0
    port.reset_transport()

    assert primary.reset_count == 1
    assert fallback.reset_count == 1
    assert port.invocation_timeout_seconds == 10.0
    assert port.call() == "openai"


def test_returned_primary_failure_can_retry_real_time_primary_without_failover() -> None:
    """After a recovery streak, primary failure restores the intended retry state."""
    class FakePort:
        def __init__(self, name: str, timeout_seconds: float) -> None:
            self.settings = SimpleNamespace(name=name)
            self.invocation_timeout_seconds = timeout_seconds
            self.reset_count = 0

        def call(self, **_kwargs):
            return self.settings.name

        def reset_transport(self) -> None:
            self.reset_count += 1

    primary = FakePort("local-policy", 0.25)
    fallback = FakePort("kimi", 8.0)
    port = FailoverStructuredModelPort(
        primary,
        fallback,
        primary_probe_after_fallback_successes=1,
        retry_primary_after_returned_failure=True,
    )

    port.advance_after_failure()

    assert primary.reset_count == 1
    assert fallback.reset_count == 0
    assert port.invocation_timeout_seconds == 0.25
    assert port.call() == "local-policy"


def test_failover_forwards_visual_prefetch_only_to_multimodal_active_port() -> None:
    """Prefetch follows current capability/ownership and does not leak media to text-only peers."""
    class FakePort:
        def __init__(self, name: str) -> None:
            self.settings = SimpleNamespace(name=name)
            self.invocation_timeout_seconds = 1.0
            self.primed: list[list[dict[str, object]]] = []
            self.closed = False

        def call(self, **_kwargs):
            return self.settings.name

        def reset_transport(self) -> None:
            return None

        def prime_multimodal(self, media: list[dict[str, object]]) -> None:
            self.primed.append(media)

        def close(self) -> None:
            self.closed = True

    primary = FakePort("local-policy")
    fallback = FakePort("kimi")
    port = FailoverStructuredModelPort(
        primary,
        fallback,
        fallback_accepts_multimodal=False,
    )
    media = [{"kind": "image-file", "path": "frame.png"}]

    port.prime_multimodal(media)
    media[0]["path"] = "mutated.png"
    port.reset_transport()
    port.prime_multimodal(media)
    port.close()

    assert primary.primed == [[{"kind": "image-file", "path": "frame.png"}]]
    assert fallback.primed == []
    assert primary.closed is True
    assert fallback.closed is True


def test_transport_reset_does_not_block_safety_thread_on_slow_close(monkeypatch) -> None:
    """Dispose the old client asynchronously so cleanup cannot stall a safety decision."""
    from dronedream_agent_core.model_harness import model_port

    # Measure the reset/close ownership contract, not host certificate loading
    # or SDK construction. The old test also timed real client initialization
    # before its fake slow close, conflating unrelated work under suite load.
    replacements = []

    def new_client(**_kwargs):
        client = SimpleNamespace(close=lambda: None)
        replacements.append(client)
        return client

    monkeypatch.setattr(model_port, "OpenAI", new_client)
    port = _port()
    close_started = threading.Event()
    release_close = threading.Event()
    closed = threading.Event()

    class SlowCloseClient:
        def close(self) -> None:
            close_started.set()
            if release_close.wait(timeout=2.0):
                closed.set()

    port._client = SlowCloseClient()
    started = time.monotonic()
    port.reset_transport()

    try:
        assert time.monotonic() - started < 0.1
        assert close_started.wait(timeout=1.0)
        assert port._transport_ready.is_set()
        assert port._client is replacements[-1]
        assert len(replacements) == 2
        assert not closed.is_set()  # Reset returned while close is still blocked.
    finally:
        release_close.set()
    assert closed.wait(timeout=1.0)


def test_success_record_reports_every_physical_provider_attempt(monkeypatch) -> None:
    """A successful logical response still reports all retries for budgets and usage accounting."""
    port = _port()
    attempts = 0

    def fake_call(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary provider failure")
        return (
            IntentArtifact(
                goal="inspect gate",
                start_entity="office",
                target_entity="gate",
                return_entity="office",
                payload_action="none",
            ),
            "response-1",
            10,
            5,
        )

    monkeypatch.setattr(port, "_responses_call", fake_call)
    result = port.call(
        role="intent_parser",
        output_type=IntentArtifact,
        instructions="Return the schema.",
        input_artifact={"message": "inspect gate"},
        maximum_physical_attempts=3,
    )

    assert attempts == 3
    assert result.record.attempt == 3
    assert result.attempt_failures == ("RuntimeError", "RuntimeError")


def test_retry_diagnostics_exclude_failed_model_payload(monkeypatch) -> None:
    """Keep schema paths/reason codes without copying rejected sensitive payloads into logs."""
    port = _port()
    attempts = 0

    def fake_call(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return IntentArtifact.model_validate(
                {
                    "goal": "x",
                    "start_entity": "office",
                    "target_entity": "gate",
                    "return_entity": "office",
                    "payload_action": "none",
                    "secret_prompt_echo": "must-not-be-retained",
                }
            )
        return (
            IntentArtifact(
                goal="inspect gate",
                start_entity="office",
                target_entity="gate",
                return_entity="office",
                payload_action="none",
            ),
            "response-2",
            10,
            5,
        )

    monkeypatch.setattr(port, "_responses_call", fake_call)
    result = port.call(
        role="intent_parser",
        output_type=IntentArtifact,
        instructions="Return the schema.",
        input_artifact={"message": "inspect gate"},
    )

    assert len(result.attempt_failures) == 1
    assert result.attempt_failures[0].startswith("ValidationError:")
    assert "must-not-be-retained" not in result.attempt_failures[0]


def test_custom_validator_message_cannot_echo_secret_into_retry_diagnostics() -> None:
    """User-defined validator text is untrusted even when wrapped by Pydantic."""
    class Output(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def reject(cls, value):
            raise ValueError(f"rejected private value: {value}")

    with pytest.raises(ValidationError) as failure:
        Output(value="must-not-be-retained")
    diagnostic = _safe_attempt_diagnostic(failure.value)
    assert "must-not-be-retained" not in diagnostic
    assert "value_error" in diagnostic


@pytest.mark.parametrize("timeout", [True, 0.0, -1.0, float("nan"), float("inf"), 10**400])
def test_invalid_timeout_rejected_before_transport_creation(monkeypatch, timeout) -> None:
    """Invalid deadlines cannot create a network client before validation fails."""
    def forbidden_client(**kwargs):
        raise AssertionError("Invalid configuration must not construct a transport")

    monkeypatch.setattr("dronedream_agent_core.model_harness.model_port.OpenAI", forbidden_client)
    with pytest.raises(ValueError, match="timeout_seconds"):
        StructuredModelPort(
            "test-provider",
            timeout_seconds=timeout,
            api_key="unit-test-key",
            settings=ProviderSettings(
                "test-provider", "test-model", "UNUSED", "https://example.invalid/v1", "responses"
            ),
        )


def test_subsecond_constructor_deadline_cannot_be_raised_by_mission_configuration():
    """A configured upper bound may tighten but never extend an existing shorter deadline."""
    port = _port()
    port._timeout_ceiling_seconds = 0.05
    try:
        port.configure_execution_policy(maximum_attempts=1, timeout_seconds=10)
        assert port.invocation_timeout_seconds == 0.05
    finally:
        port.close()


def test_image_growth_after_stat_is_still_bounded(tmp_path, monkeypatch):
    """Bound the actual read even if the file grows after metadata inspection."""
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"small")
    real_open = Path.open
    reads = []

    class GrowingFrame(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    def open_frame(path, *args, **kwargs):
        if path == image_path:
            return GrowingFrame(b"x" * (12 * 1024 * 1024 + 1))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_frame)
    with pytest.raises(ValueError, match="bounded size"):
        StructuredModelPort._media_data_url(
            {
                "kind": "image-file",
                "path": str(image_path),
                "sha256": hashlib.sha256(b"small").hexdigest(),
            }
        )
    assert reads == [12 * 1024 * 1024 + 1]


def test_remaining_hard_budget_caps_provider_retry_loop(monkeypatch) -> None:
    """Remaining physical-call budget overrides a larger per-port retry allowance."""
    port = _port()
    attempts = 0

    def always_fail(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("temporary provider failure")

    monkeypatch.setattr(port, "_responses_call", always_fail)
    with pytest.raises(ModelInvocationError) as failure:
        port.call(
            role="intent_parser",
            output_type=IntentArtifact,
            instructions="Return the schema.",
            input_artifact={"message": "inspect gate"},
            maximum_physical_attempts=2,
        )

    assert attempts == 2
    assert failure.value.attempts_used == 2


def test_strict_schema_detection_rejects_free_form_argument_map() -> None:
    """Dynamic tool arguments require JSON mode rather than an unsupported strict schema."""
    assert StructuredModelPort._supports_strict_structured_output(
        IntentArtifact.model_json_schema()
    )
    assert not StructuredModelPort._supports_strict_structured_output(
        TaskGraphArtifact.model_json_schema()
    )


def test_responses_json_mode_preserves_dynamic_task_arguments() -> None:
    """The Responses adapter preserves task-specific maps and validates the resulting artifact."""
    port = _port()
    artifact_json = json.dumps(
        {
            "schema_version": "dronedream.task-graph-artifact.v1",
            "graph": {
                "schema_version": "dronedream.task-graph.v2",
                "revision": 1,
                "nodes": [
                    {
                        "task_id": "fly-office-gate",
                        "action": "navigate",
                        "target_node": "gate",
                        "arguments": {"altitude_m": 8.0, "camera": "front"},
                        "depends_on": [],
                        "success_evidence": ["arrival"],
                        "max_retries": 1,
                        "fallback": "abort",
                    }
                ],
            },
        }
    )

    class FakeResponses:
        def __init__(self) -> None:
            self.create_request = None

        def create(self, **request):
            self.create_request = request
            return SimpleNamespace(
                output_text=artifact_json,
                id="response-json-mode",
                usage=SimpleNamespace(input_tokens=12, output_tokens=8),
            )

        def parse(self, **_request):  # pragma: no cover - branch must not be used
            raise AssertionError("strict parse must not receive free-form object schemas")

    responses = FakeResponses()

    class FakeClient:
        def __init__(self) -> None:
            self.responses = responses

        def with_options(self, **_options):
            return self

    port._client = FakeClient()
    result = port._responses_call(
        output_type=TaskGraphArtifact,
        instructions="Build the task graph.",
        input_json="{}",
        previous_response_id=None,
        repair_errors=[],
        multimodal=[],
    )

    artifact = result[0]
    assert artifact.graph.nodes[0].arguments == {"altitude_m": 8.0, "camera": "front"}
    assert responses.create_request["text"] == {"format": {"type": "json_object"}}


def test_kimi_k2_6_disables_default_thinking_for_bounded_json_calls() -> None:
    """Bounded structured calls explicitly set the provider's supported thinking policy."""
    port = StructuredModelPort(
        "kimi",
        settings=ProviderSettings(
            name="kimi",
            model="kimi-k2.6",
            api_key_env="KIMI_API_KEY",
            base_url="https://api.moonshot.ai/v1",
            api_style="chat-completions",
        ),
        api_key="not-a-real-secret",
    )
    artifact_json = IntentArtifact(
        goal="inspect gate",
        start_entity="office",
        target_entity="gate",
        return_entity="office",
        payload_action="none",
    ).model_dump_json()

    class FakeCompletions:
        def __init__(self) -> None:
            self.request = None

        def create(self, **request):
            self.request = request
            return SimpleNamespace(
                id="kimi-response",
                choices=[SimpleNamespace(message=SimpleNamespace(content=artifact_json))],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
            )

    completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=completions)

        def with_options(self, **_options):
            return self

    port._client = FakeClient()
    artifact, *_ = port._chat_call(
        output_type=IntentArtifact,
        instructions="Return the bounded intent.",
        input_json="{}",
        repair_errors=[],
        multimodal=[],
    )

    assert artifact.goal == "inspect gate"
    assert completions.request["extra_body"] == {"thinking": {"type": "disabled"}}
