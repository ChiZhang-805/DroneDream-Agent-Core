from __future__ import annotations

from types import SimpleNamespace

import pytest

from dronedream_agent_core.contracts import IntentArtifact
from dronedream_agent_core.orchestrator import MissionOrchestrator, MissionPreparationBlocked


def test_multimodal_intent_fails_closed_when_router_has_only_text_models() -> None:
    orchestrator = MissionOrchestrator.__new__(MissionOrchestrator)
    text_port = SimpleNamespace(supports_image_input=False)
    orchestrator.primary = text_port
    orchestrator.critic = SimpleNamespace(supports_image_input=False)
    orchestrator.model_ports = {"primary": text_port}
    orchestrator._model_media = [{"kind": "image-file", "path": "camera.png"}]

    def select_extension(slot_id: str, _hook: str, **_kwargs: object) -> dict[str, object]:
        if slot_id == "models.role-policy":
            return {"port": "primary"}
        if slot_id == "models.runtime-router":
            return {"candidates": ["primary"]}
        raise AssertionError(f"unexpected extension slot {slot_id}")

    orchestrator._invoke_single_extension = select_extension

    with pytest.raises(MissionPreparationBlocked, match="MODEL_ROUTER_NO_MULTIMODAL_PORT"):
        orchestrator._call(
            port=text_port,
            role="intent_parser",
            output_type=IntentArtifact,
            instructions="Parse the request and image.",
            input_artifact={"message": "inspect this scene"},
            conversation_id="thread-1",
            evidence=object(),
        )
