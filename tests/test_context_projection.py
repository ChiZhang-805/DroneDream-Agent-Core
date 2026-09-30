"""Regression tests for bounded mission context sent to model roles."""

from __future__ import annotations

from dronedream_agent_core.context_projection import (
    MAX_INTENT_REQUEST_CONTEXT_BYTES,
    compact_mission_request_event,
    intent_model_request_projection,
)
from dronedream_agent_core.hashing import canonical_json


def test_intent_projection_removes_dense_airspace_and_asset_item_lists() -> None:
    context = {
        "schema_version": "dronedream.model-harness-input.v1",
        "request_id": "request-12345678",
        "task_id": "thread-12345678",
        "thread_id": "thread-12345678",
        "owner_binding_sha256": "1" * 64,
        "tenant_binding_sha256": "2" * 64,
        "source_edition": "autonomy",
        "domain": "autonomy.mission",
        "control_plane_selection_sha256": "3" * 64,
        "current_request": {
            "message": "去补给站取餐",
            "locale": "zh-CN",
            "input_channel": "text",
            "start_entity": "office",
            "attachment_ids": [],
            "map_asset_id": "office",
            "vehicle_asset_id": "my-drone",
        },
        "session_context": {
            "preferred_airspace": {
                "airspace_sha256": "4" * 64,
                "authority": "advisory-only",
                "named_node_height_samples": [
                    {"node_id": f"node-{index}", "geometry": "x" * 500}
                    for index in range(100)
                ],
                "node_samples_truncated": False,
            },
            "asset_understanding": {
                "map": {
                    "asset_id": "office",
                    "authority": "advisory-only",
                    "understanding": {
                        "summary": "办公室三维地图",
                        "items": [
                            {"source_id": f"node-{index}", "explanation": "x" * 500}
                            for index in range(100)
                        ],
                        "limitations": ["需要运行时避障"],
                    },
                }
            },
        },
        "memory_record_ids": [],
    }

    projected = intent_model_request_projection(context)
    encoded_size = len(canonical_json(projected).encode("utf-8"))

    airspace = projected["session_context"]["preferred_airspace"]
    understanding = projected["session_context"]["asset_understanding"]["map"][
        "understanding"
    ]
    assert airspace["named_node_height_sample_count"] == 100
    assert "named_node_height_samples" not in airspace
    assert understanding["summary"] == "办公室三维地图"
    assert understanding["item_count"] == 100
    assert "items" not in understanding
    assert encoded_size < MAX_INTENT_REQUEST_CONTEXT_BYTES


def test_persisted_request_projection_keeps_message_and_asset_identity_only() -> None:
    payload = {
        "conversation_id": "thread-12345678",
        "message": "supplies（补给站）",
        "input_metadata": {
            "request_purpose": "initial_plan",
            "model_harness_control_plane": {"bulk": "x" * 50_000},
            "model_harness_input": {
                "current_request": {
                    "map_asset_id": "office",
                    "map_content_sha256": "a" * 64,
                    "vehicle_asset_id": "my-drone",
                    "vehicle_content_sha256": "b" * 64,
                }
            },
        },
    }

    projected = compact_mission_request_event(payload)

    assert projected["message"] == "supplies（补给站）"
    assert projected["input_metadata"]["selected_assets"]["vehicle_asset_id"] == "my-drone"
    assert "model_harness_control_plane" not in projected["input_metadata"]
    assert len(canonical_json(projected).encode("utf-8")) < 2_000
