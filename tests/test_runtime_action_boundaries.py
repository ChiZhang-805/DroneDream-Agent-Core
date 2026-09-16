"""Offline adapter-boundary regressions; these tests never issue actuator commands."""

from pathlib import Path

import pytest

from dronedream_agent_core import runtime_actions


def test_unknown_payload_executor_is_not_silently_converted_to_detach(monkeypatch):
    monkeypatch.setattr(runtime_actions, "_detachable_joint_bindings", lambda *_: {
        "attach_topic": "/attach", "detach_topic": "/detach", "output_topic": "/state",
    })
    monkeypatch.setattr(runtime_actions, "_payload_mount_binding", lambda *_: {})
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="UNSUPPORTED"):
        runtime_actions._resolved_parameters(
            driver="gazebo-payload", runtime_executor="native.payload.typo",
            defaults={}, arguments={}, vehicle_sdf=Path("unused.sdf"),
        )


def test_resolved_camera_parameters_own_their_nested_input_values():
    defaults, arguments = {"calibration": {"bias": [1]}}, {"metadata": {"labels": ["target"]}}
    parameters = runtime_actions._resolved_parameters(
        driver="mavsdk-camera", runtime_executor="native.camera.capture",
        defaults=defaults, arguments=arguments, vehicle_sdf=Path("unused.sdf"),
    )
    parameters["calibration"]["bias"].append(2)
    parameters["arguments"]["metadata"]["labels"].clear()
    assert defaults == {"calibration": {"bias": [1]}}
    assert arguments == {"metadata": {"labels": ["target"]}}


@pytest.mark.parametrize("component", [True, 1.2, "100", -1, 256])
def test_camera_component_requires_a_real_mavlink_component_identifier(component):
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="COMPONENT_ID"):
        runtime_actions._resolved_parameters(
            driver="mavsdk-camera", runtime_executor="native.camera.capture",
            defaults={}, arguments={"component_id": component}, vehicle_sdf=Path("unused.sdf"),
        )


def test_invalid_adapter_schema_is_rejected_during_catalog_assembly():
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="SCHEMA_INVALID"):
        runtime_actions.merge_runtime_action_adapters([{"adapters": [{
            "adapter_id": "camera", "runtime_executors": ["native.camera.capture"],
            "driver": "mavsdk-camera", "authority": "control",
            "parameter_schema": {"type": "not-a-json-type"},
        }]}])


def test_ros_arguments_cannot_serialize_non_finite_json():
    with pytest.raises(runtime_actions.RuntimeActionContractError, match="ARGUMENTS_INVALID"):
        runtime_actions._resolved_parameters(
            driver="ros2-service", runtime_executor="native.payload.verify",
            defaults={}, arguments={"amount": float("nan")}, vehicle_sdf=Path("unused.sdf"),
        )
