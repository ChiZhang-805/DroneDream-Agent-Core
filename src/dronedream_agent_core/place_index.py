"""Version-bound, precomputed map place facts for natural-language understanding."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def _bundled_index(semantic_sha256: str) -> dict | None:
    if len(semantic_sha256) != 64 or any(c not in "0123456789abcdef" for c in semantic_sha256):
        raise ValueError("Invalid map semantic digest")
    path = Path(__file__).with_name("bundled_place_indexes") / f"{semantic_sha256}.json"
    if not path.is_file():
        return None
    index = json.loads(path.read_bytes())
    if (
        index["schema_version"] != "dronedream.map-place-index.v1"
        or index["semantic_sha256"] != semantic_sha256
    ):
        raise ValueError("Map place index does not match its source")
    return index


def map_place_context(semantic_sha256: str) -> dict:
    """Return fresh compact facts; coordinates stay in the offline geometry index.

    File parsing is cached by immutable map version. Labels and services remain
    source data, not executable instructions or permission to approach an entrance.
    """
    index = _bundled_index(semantic_sha256)
    if index is None:
        return {}
    building_fields = (
        "building_id",
        "name",
        "aliases",
        "address",
        "use",
        "label_basis",
        "entrance_ids",
        "entrance_coverage",
        "total_entrance_count",
        "handoff_candidate_ids",
        "service_ids",
    )
    entrance_fields = (
        "entrance_id",
        "building_id",
        "name",
        "kind",
        "access",
        "level",
        "inside_scene_bounds",
        "service_ids",
    )
    context = {
        "schema_version": index["schema_version"],
        "coverage": index["coverage"],
        "buildings": [
            {key: b[key] for key in building_fields if key in b} for b in index["buildings"]
        ],
        "facilities": [
            {key: value for key, value in facility.items() if key != "footprint"}
            for facility in index["facilities"]
        ],
        "services": index["services"],
        "entrances": [
            {key: e[key] for key in entrance_fields if key in e} for e in index["entrances"]
        ],
        "handoff_candidates": [
            {
                key: h[key]
                for key in (
                    "handoff_id",
                    "building_id",
                    "entrance_id",
                    "status",
                    "flight_target_available",
                    "entrance_connection_verified",
                )
            }
            for h in (index["handoff_candidates"] if not index.get("handoff_locations") else [])
        ],
        "handoff_locations": [
            {
                key: location[key]
                for key in (
                    "place_id",
                    "building_id",
                    "name",
                    "node_id",
                    "entrance_id",
                    "status",
                    "flight_target_available",
                    "flight_verified",
                    "reason",
                    "entrance_connection_verified",
                    "connection_basis",
                    "building_distance_m",
                    "entrance_distance_m",
                )
                if key in location
            }
            for location in index.get("handoff_locations", [])
        ],
        "interpretation_rules": [
            "Use names, addresses, department aliases, documented services and conversation "
            "context before asking questions.",
            "Item associations rank candidate facilities; they do not prove an item's location "
            "or availability.",
            "An address can identify several buildings; use building and entrance references "
            "to disambiguate.",
            "No recorded entrance means unknown, not no entrances. Recorded counts are not "
            "a complete survey.",
            "Handoff candidates are exterior geometry proposals, not surveyed entrances "
            "or approved flight nodes.",
            "Use handoff_locations to bind a building or facility to its checked outdoor "
            "simulation target. A facade-only target is not a documented doorway. "
            "If a place has no available target, briefly explain this and ask for another "
            "handoff location; do not invent a point or submit an unresolved building ID.",
            "Only entities bound to the current navigation graph are executable targets; "
            "never substitute a building centre.",
        ],
    }
    return json.loads(json.dumps(context, ensure_ascii=False))
