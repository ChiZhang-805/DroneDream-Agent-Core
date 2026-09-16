from argparse import Namespace
from types import SimpleNamespace

from test_native_flight_state import _identity
from test_runtime_commands import _load_executor

from dronedream_agent_core.contracts import Px4CoordinateContract, Vector3
from dronedream_agent_core.native_pose import native_map_pose


def test_spawn_rebase_does_not_discard_source_timestamp():
    import asyncio

    module = _load_executor()

    class Client:
        async def sample_position_velocity_ned(self, timeout_seconds):
            return SimpleNamespace(north_m=10., east_m=20., down_m=-3.,
                                   north_m_s=0., east_m_s=0., down_m_s=0.,
                                   received_at_unix_ms=1000)

    wrapped = module.SpawnRelativeOffboardClient(
        Client(), SimpleNamespace(north_m=5., east_m=10., down_m=-1.),
    )
    observed = asyncio.run(wrapped.sample_position_velocity_ned(.25))
    assert observed.received_at_unix_ms == 1000
    assert (observed.north_m, observed.east_m, observed.down_m) == (5., 10., -2.)


def test_identity_publication_keeps_original_receive_time_and_fixed_map_binding(
    tmp_path, monkeypatch
):
    import json

    module = _load_executor()
    monkeypatch.setattr(module.time, "time", lambda: 1.05)
    measured = SimpleNamespace(
        north_m=2.0,
        east_m=3.0,
        down_m=-1.0,
        north_m_s=0.0,
        east_m_s=1.0,
        down_m_s=0.0,
        received_at_unix_ms=1000,
    )
    coordinate = Px4CoordinateContract(
        model_root_world_enu_m=[10, 20, 0], collision_center_offset_model_m=[0, 0, 0.2]
    )
    published = []
    def accept_before_disk(payload):
        assert not (tmp_path / "runtime-state" / "px4-identity-telemetry.json").exists()
        published.append(payload)
    channel = SimpleNamespace(send=accept_before_disk)
    module._publish_px4_identity_telemetry(
        args=Namespace(run_dir=tmp_path, _native_state_channel=channel),
        coordinate_contract=coordinate,
        observed=measured,
        dynamics_telemetry=_identity()["dynamics"],
    )
    payload = json.loads((tmp_path / "runtime-state" / "px4-identity-telemetry.json").read_text())
    assert published == [payload]
    assert payload["position_received_at_unix_ms"] == 1000
    assert payload["updated_at_unix_ms"] == 1050
    assert native_map_pose(payload, now_unix_ms=1050).position_world_enu_m == Vector3(
        x=13, y=22, z=1.2
    )
