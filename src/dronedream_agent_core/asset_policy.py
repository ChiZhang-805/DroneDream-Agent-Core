"""Machine-readable import and real-simulation qualification policy."""

from __future__ import annotations

from .asset_packages import (
    MAX_ASSET_IR_BYTES,
    MAX_COMPRESSION_RATIO,
    MAX_MANIFEST_BYTES,
    MAX_MEMBER_UNCOMPRESSED_BYTES,
    MAX_PACKAGE_MEMBERS,
    MAX_PACKAGE_UNCOMPRESSED_BYTES,
    MAX_XML_INSPECTION_BYTES,
)
from .asset_pair_qualification import REQUIRED_RUNTIME_GATES


def asset_qualification_policy() -> dict[str, object]:
    """Describe current admission gates for clients, not grant qualification.

    Limits come from the enforcing modules to avoid a second drifting policy.
    Each call returns detached data; advertising a maturity level never promotes
    an asset without the separately verified pair-specific runtime evidence.
    """
    return {
        "schema_version": "dronedream.asset-qualification-policy.v1",
        "package_format": "ddpkg",
        "maturity_levels": [
            {
                "id": "visual_only",
                "runtime_evidence_required": False,
                "mission_binding_allowed": False,
            },
            {
                "id": "physics_ready",
                "runtime_evidence_required": False,
                "mission_binding_allowed": False,
            },
            {
                "id": "simulation_ready",
                "runtime_evidence_required": True,
                "mission_binding_allowed": False,
            },
            {
                "id": "flight_ready",
                "runtime_evidence_required": True,
                "mission_binding_allowed": False,
            },
            {
                "id": "qualified",
                "runtime_evidence_required": True,
                "mission_binding_allowed": True,
            },
        ],
        "admission_limits": {
            "maximum_members": MAX_PACKAGE_MEMBERS,
            "maximum_package_uncompressed_bytes": MAX_PACKAGE_UNCOMPRESSED_BYTES,
            "maximum_member_uncompressed_bytes": MAX_MEMBER_UNCOMPRESSED_BYTES,
            "maximum_compression_ratio": MAX_COMPRESSION_RATIO,
            "maximum_manifest_bytes": MAX_MANIFEST_BYTES,
            "maximum_asset_ir_bytes": MAX_ASSET_IR_BYTES,
            "maximum_xml_inspection_bytes": MAX_XML_INSPECTION_BYTES,
        },
        "remote_sources": {
            "allowed": ["public_https_file", "public_https_git"],
            "credentialed_sources_require_connector": True,
            "private_network_requires_connector": True,
            "source_hash_supported": True,
        },
        "qualification_pair": {
            "simulator": "gazebo-harmonic",
            "autopilot": "px4",
            "middleware": "ros2",
            "minimum_translation_m": 1.0,
            "maximum_translation_m": 30.0,
            "required_runtime_gates": list(REQUIRED_RUNTIME_GATES),
        },
        "promotion": {
            "exact_map_hash_required": True,
            "exact_vehicle_hash_required": True,
            "environment_versions_required": True,
            "evidence_hash_required": True,
            "unknown_engineering_values_may_be_inferred": False,
            "stale_qualification_is_rejected": True,
        },
    }
