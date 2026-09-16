"""Command construction must bind the current held execution, not just valid JSON."""

import pytest
from test_runtime_commands import _ack, _decision, _message

from dronedream_agent_core.runtime_commands import RuntimeCommandError, build_runtime_command


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_id", "execution-" + "f" * 32),
        ("message_id", "runtime-msg-" + "f" * 32),
        ("message_sha256", "f" * 64),
    ],
)
def test_hold_from_another_message_or_execution_cannot_authorize_current_command(field, value):
    message = _message()
    acknowledgement = _ack(message).model_copy(update={field: value})
    decision = _decision(
        message,
        acknowledgement,
        action="set_avoidance",
        parameters={"enabled": True},
    )
    with pytest.raises(RuntimeCommandError, match="GATE_FAILED"):
        build_runtime_command(message=message, acknowledgement=acknowledgement, decision=decision)


@pytest.mark.parametrize("mutation", ["authorization", "amendment"])
def test_decision_cannot_hide_failed_gate_or_different_amendment(mutation):
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(
        message,
        acknowledgement,
        action="set_avoidance",
        parameters={"enabled": True},
    )
    if mutation == "authorization":
        decision.authorization_gates["stable_hold"] = False
    else:
        decision.amendment_directive.action = "payload_control"
    with pytest.raises(RuntimeCommandError, match="GATE_FAILED"):
        build_runtime_command(message=message, acknowledgement=acknowledgement, decision=decision)


@pytest.mark.parametrize(
    "action,parameters",
    [
        ("camera_control", {"command": "take_photo", "component_id": True}),
        (
            "payload_control",
            {
                "protocol": "mavsdk-actuator",
                "actuator_index": True,
                "actuator_value": 0.5,
            },
        ),
        (
            "payload_control",
            {
                "protocol": "mavsdk-actuator",
                "actuator_index": 1,
                "actuator_value": True,
            },
        ),
    ],
)
def test_boolean_is_not_an_actuator_index_or_amplitude(action, parameters):
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(message, acknowledgement, action=action, parameters=parameters)
    with pytest.raises(RuntimeCommandError, match="command_parameters_valid"):
        build_runtime_command(message=message, acknowledgement=acknowledgement, decision=decision)


def test_valid_topic_syntax_does_not_authorize_unbound_payload_transport():
    """Even replacing the policy plugin cannot turn arbitrary addresses into device authority."""
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(
        message,
        acknowledgement,
        action="payload_control",
        parameters={
            "operation": "detach",
            "protocol": "gazebo-transport",
            "topic": "/model/another_vehicle/detach",
            "output_topic": "/model/another_vehicle/state",
        },
    )
    with pytest.raises(RuntimeCommandError, match="payload_transport_bound"):
        build_runtime_command(message=message, acknowledgement=acknowledgement, decision=decision)


def test_core_does_not_discard_plugin_directive_failure():
    message = _message()
    acknowledgement = _ack(message)
    decision = _decision(
        message, acknowledgement, action="set_avoidance", parameters={"enabled": True}
    )
    decision.amendment_directive.issue_codes.append("INJECTED_POLICY_REJECTION")
    with pytest.raises(RuntimeCommandError, match="directive_verdict_passed"):
        build_runtime_command(message=message, acknowledgement=acknowledgement, decision=decision)
