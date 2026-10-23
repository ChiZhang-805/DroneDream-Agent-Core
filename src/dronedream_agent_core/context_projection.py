"""Bounded model-facing projections for durable mission conversation events."""

from __future__ import annotations

from collections.abc import Iterable

from .hashing import canonical_json

MAX_INTENT_REQUEST_CONTEXT_BYTES = 24 * 1024


def _bounded_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized[:maximum] if normalized else None


def _bounded_strings(value: object, *, maximum_items: int, maximum_length: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value[:maximum_items]:
        text = _bounded_text(item, maximum=maximum_length)
        if text is not None:
            result.append(text)
    return result


def _selected_scalars(
    value: object,
    names: Iterable[str],
    *,
    maximum_string_length: int = 512,
) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for name in names:
        item = value.get(name)
        if isinstance(item, str):
            text = _bounded_text(item, maximum=maximum_string_length)
            if text is not None:
                result[name] = text
        elif type(item) in {bool, int, float} or item is None:
            result[name] = item
    return result


def _asset_versions(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {
        kind: _selected_scalars(binding, ("asset_id", "content_sha256", "version_id"))
        for kind in ("map", "vehicle")
        if (binding := value.get(kind)) is not None
    }


def _asset_understanding(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for kind in ("map", "vehicle"):
        raw = value.get(kind)
        if not isinstance(raw, dict):
            continue
        projected = _selected_scalars(
            raw,
            ("asset_id", "content_sha256", "authority"),
        )
        understanding = raw.get("understanding")
        if isinstance(understanding, dict):
            projected["understanding"] = {
                # Intent grounding only needs the asset's short identity summary.
                # Detailed limitations remain in immutable evidence and are fed to
                # later deterministic qualification and planning stages.
                "summary": _bounded_text(understanding.get("summary"), maximum=320) or "",
                # 逐项解释会与语义地图目录重复；只保留数量用于说明摘要完整度。
                "item_count": min(
                    len(understanding.get("items", []))
                    if isinstance(understanding.get("items"), list)
                    else 0,
                    100_000,
                ),
            }
        result[kind] = projected
    return result


def compact_mission_request_event(payload: object) -> dict[str, object]:
    """Keep conversational facts while excluding runtime/control-plane bulk.

    The complete request remains in the immutable evidence chain. This view is
    only what may be replayed into a later language-model prompt.
    """

    if not isinstance(payload, dict):
        return {}
    projected = _selected_scalars(
        payload,
        (
            "schema_version",
            "conversation_id",
            "message",
            "start_entity",
            "locale",
            "input_channel",
        ),
        maximum_string_length=4_000,
    )
    attachments = payload.get("attachments")
    if isinstance(attachments, list):
        projected["attachments"] = [
            _selected_scalars(
                item,
                (
                    "attachment_id",
                    "display_name",
                    "content_type",
                    "size_bytes",
                    "source_sha256",
                    "decoded_kind",
                ),
            )
            for item in attachments[:32]
            if isinstance(item, dict)
        ]
    metadata = payload.get("input_metadata")
    if isinstance(metadata, dict):
        compact_metadata = _selected_scalars(
            metadata,
            (
                "public_harness_context_sha256",
                "public_edition",
                "public_conversation_id",
                "request_purpose",
                "transcript_source",
                "attachment_count",
            ),
        )
        envelope = metadata.get("model_harness_input")
        if isinstance(envelope, dict):
            compact_metadata["selected_assets"] = _selected_scalars(
                envelope.get("current_request"),
                (
                    "map_asset_id",
                    "map_content_sha256",
                    "vehicle_asset_id",
                    "vehicle_content_sha256",
                    "model_selection_id",
                ),
            )
        compact_metadata["asset_versions"] = _asset_versions(metadata.get("asset_versions"))
        projected["input_metadata"] = compact_metadata
    return projected


def intent_model_request_projection(context: dict[str, object]) -> dict[str, object]:
    """Project a validated Harness envelope to the facts needed by intent roles."""

    projected = _selected_scalars(
        context,
        (
            "schema_version",
            "request_id",
            "task_id",
            "thread_id",
            "owner_binding_sha256",
            "tenant_binding_sha256",
            "source_edition",
            "domain",
            "control_plane_selection_sha256",
        ),
    )
    projected["current_request"] = _selected_scalars(
        context.get("current_request"),
        (
            "message",
            "locale",
            "input_channel",
            "start_entity",
            "map_asset_id",
            "map_content_sha256",
            "vehicle_asset_id",
            "vehicle_content_sha256",
            "model_selection_id",
        ),
        maximum_string_length=4_000,
    )
    current = context.get("current_request")
    if isinstance(current, dict):
        projected["current_request"]["attachment_ids"] = _bounded_strings(
            current.get("attachment_ids"), maximum_items=32, maximum_length=96
        )

    session = context.get("session_context")
    session_projection: dict[str, object] = {}
    if isinstance(session, dict):
        session_projection["memory_policy"] = _selected_scalars(
            session.get("memory_policy"),
            (
                "namespace",
                "owner_bound",
                "persisted_task_context",
                "remember_asset_choices",
                "safety_boundaries_are_non_relaxable",
            ),
        )
        session_projection["asset_versions"] = _asset_versions(session.get("asset_versions"))
        session_projection["asset_pair_qualification"] = _selected_scalars(
            session.get("asset_pair_qualification"),
            (
                "schema_version",
                "qualification_id",
                "receipt_sha256",
                "runtime_evidence_sha256",
                "map_asset_id",
                "map_qualified_content_sha256",
                "vehicle_asset_id",
                "vehicle_qualified_content_sha256",
                "qualified_at",
            ),
        )
        airspace = session.get("preferred_airspace")
        airspace_projection = _selected_scalars(
            airspace,
            ("airspace_sha256", "binding", "authority", "node_samples_truncated"),
        )
        if isinstance(airspace, dict):
            samples = airspace.get("named_node_height_samples")
            airspace_projection["named_node_height_sample_count"] = (
                len(samples) if isinstance(samples, list) else 0
            )
        session_projection["preferred_airspace"] = airspace_projection
        session_projection["harness_revision"] = _selected_scalars(
            session.get("harness_revision"),
            ("revision", "semantic_sha256", "layout_sha256", "profile_id", "frozen_at"),
        )
        session_projection["asset_understanding"] = _asset_understanding(
            session.get("asset_understanding")
        )
    projected["session_context"] = session_projection
    projected["memory_record_ids"] = _bounded_strings(
        context.get("memory_record_ids"), maximum_items=32, maximum_length=160
    )
    projected["projection"] = "intent-bounded-v1"
    if len(canonical_json(projected).encode("utf-8")) > MAX_INTENT_REQUEST_CONTEXT_BYTES:
        raise ValueError("INTENT_MODEL_REQUEST_CONTEXT_TOO_LARGE")
    return projected
