"""Bind map observation pixels to the exact selected simulator asset."""

from __future__ import annotations

import hashlib

from dronedream_agent_core.assets import read_map_semantic_object
from dronedream_agent_core.map_gap_refinement import MapImageSource
from dronedream_agent_core.plugin_files import read_plugin_file

from .asset_runtime_resolver import resolve_versioned_map
from .live_map import resolve_live_map


def map_observation_source(record: dict[str, object]) -> MapImageSource:
    selected = resolve_versioned_map(record)
    live = resolve_live_map(record)
    a, b, c, d, e, f = live.metadata["world_to_image"]
    determinant = a * e - b * d
    if abs(determinant) < 1e-12:
        raise ValueError("MAP_OBSERVATION_TRANSFORM_SINGULAR")
    semantic = read_map_semantic_object(selected.semantic)
    layer = semantic.get("traversability", {})
    heights = {float(item["ground_z_m"]) for key in ("areas", "corridors")
               for item in layer.get(key, [])}
    registration = semantic.get("live_map", {})
    if "ground_z_m" in registration:
        ground = float(registration["ground_z_m"])
    elif len(heights) == 1:
        ground = heights.pop()
    else:
        raise ValueError("MAP_OBSERVATION_GROUND_REGISTRATION_REQUIRED")
    return MapImageSource(
        map_id=selected.asset_id,
        semantic_sha256=hashlib.sha256(read_plugin_file(selected.semantic,
            limit=64 * 1024 * 1024)).hexdigest(),
        image_bytes=live.image, image_sha256=hashlib.sha256(live.image).hexdigest(),
        pixel_to_map=(e / determinant, -b / determinant, (b * f - e * c) / determinant,
                      -d / determinant, a / determinant, (d * c - a * f) / determinant),
        source_id=f"selected-map:{selected.asset_id}:{selected.content_sha256}",
        ground_z_m=ground,
    )
