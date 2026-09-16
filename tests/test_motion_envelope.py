"""Conservative predicted intervals, not simulator or physical-flight qualification."""

import math
import random

import pytest
from test_dynamic_safety import _request
from test_runtime_local_safety import _vehicle

from dronedream_agent_core import dynamic_safety
from dronedream_agent_core.contracts import (
    DynamicObstacleObservation,
    RuntimeLocalSafetyObservation,
    Vector3,
)
from dronedream_agent_core.motion_envelope import (
    interval_clearance_lower_bound,
    prediction_query_radius_m,
)
from dronedream_agent_core.runtime_local_safety import runtime_safety_query_radius_m
from dronedream_agent_core.static_geometry_index import StaticGeometryIndex
from scripts.runtime_depth_safety_worker import _primitive_bounds


def test_clearance_cones_preserve_linear_escape_and_bound_hidden_minimum():
    assert interval_clearance_lower_bound(0.1, 0.3, 0.2) == pytest.approx(0.1)
    assert interval_clearance_lower_bound(0.3, 0.1, 0.2) == pytest.approx(0.1)
    assert interval_clearance_lower_bound(0.1, 0.1, 0.4) == pytest.approx(-0.1)
    assert interval_clearance_lower_bound(0.1, 0.1, 0.0) == 0.1


@pytest.mark.parametrize(
    "values", [(math.nan, 1.0, 1.0), (1.0, math.inf, 1.0), (1.0, 1.0, math.inf), (1.0, 1.0, -0.1)]
)
def test_invalid_interval_cannot_authorize_motion(values):
    with pytest.raises(ValueError, match="INTERVAL_INVALID"):
        interval_clearance_lower_bound(*values)


def test_interval_bound_never_exceeds_densely_sampled_obstacle_clearance():
    rng = random.Random(805)
    for _ in range(120):
        start, end, center = [tuple(rng.uniform(-2.0, 2.0) for _ in range(3)) for _ in range(3)]
        radius = rng.uniform(0.01, 0.5)
        bound = interval_clearance_lower_bound(
            math.dist(start, center) - radius,
            math.dist(end, center) - radius,
            math.dist(start, end),
        )
        sampled = min(
            math.dist(tuple(a + (b - a) * i / 300 for a, b in zip(start, end, strict=True)), center)
            - radius
            for i in range(301)
        )
        assert bound <= sampled + 1e-12


@pytest.mark.parametrize("center_x", [0.1, 1.0, 2.25])
def test_thin_wall_between_samples_including_origin_and_last_short_interval(center_x):
    request = _request().model_copy(
        update={
            "prediction_horizon_seconds": 0.25,
            "vehicle_radius_m": 0.01,
            "vehicle_height_m": 0.02,
            "max_speed_mps": 10.0,
        }
    )
    wall = dict(
        name="thin",
        center_x=center_x,
        center_y=0.0,
        center_z=1.0,
        size_x=0.01,
        size_y=2.0,
        size_z=2.0,
    )
    velocity = (10.0, 0.0, 0.0)
    endpoints = [
        dynamic_safety._point_clearance(
            (x, 0.0, 1.0), elapsed=0.0, request=request, static_primitives=[wall]
        )[0]
        for x in (0.0, 2.0, 2.5)
    ]
    assert min(endpoints) > 0  # The old point-only check could miss this wall.
    scalar = dynamic_safety._predict(request, velocity, [wall])
    batch = dynamic_safety._predict_candidates(request, [velocity], [wall])[0]
    assert scalar == batch
    assert scalar[1] < 0
    assert scalar[3] == "static:thin"
    assert scalar[0] == [(2.0, 0.0, 1.0), (2.5, 0.0, 1.0)]


def test_fast_dynamic_crossing_cannot_tunnel_past_stationary_vehicle():
    obstacle = DynamicObstacleObservation(
        obstacle_id="crossing",
        position_m=Vector3(x=0.0, y=-1.0, z=1.0),
        velocity_mps=Vector3(x=0.0, y=10.0, z=0.0),
        radius_m=0.02,
        height_m=0.04,
        confidence=1.0,
        age_seconds=0.0,
    )
    request = _request(obstacles=[obstacle]).model_copy(
        update={
            "prediction_horizon_seconds": 0.2,
            "vehicle_radius_m": 0.01,
            "vehicle_height_m": 0.02,
        }
    )
    assert dynamic_safety._dynamic_clearance((0.0, 0.0, 1.0), elapsed=0.0, request=request)[0] > 0.9
    assert dynamic_safety._dynamic_clearance((0.0, 0.0, 1.0), elapsed=0.2, request=request)[0] > 0.9
    result = dynamic_safety._predict(request, (0.0, 0.0, 0.0), [])
    assert result[1] < 0 and result[3] == "crossing"


def test_relative_motion_does_not_invent_a_fast_collision_for_matching_velocities():
    obstacle = DynamicObstacleObservation(
        obstacle_id="parallel",
        position_m=Vector3(x=0.0, y=1.0, z=1.0),
        velocity_mps=Vector3(x=10.0, y=0.0, z=0.0),
        radius_m=0.02,
        height_m=0.04,
        confidence=1.0,
        age_seconds=0.0,
    )
    request = _request(obstacles=[obstacle]).model_copy(
        update={
            "vehicle_radius_m": 0.01,
            "vehicle_height_m": 0.02,
        }
    )
    result = dynamic_safety._predict(request, (10.0, 0.0, 0.0), [])
    assert result[1] == pytest.approx(0.97)


def observation(speed=0.0):
    return RuntimeLocalSafetyObservation(
        sequence=1,
        observed_at_unix_ms=1000,
        source="onboard",
        stream_healthy=True,
        stream_age_seconds=0.01,
        localization_covariance_m2=0.01,
        current_position_m=Vector3(x=0.0, y=0.0, z=1.0),
        current_velocity_mps=Vector3(x=speed, y=0.0, z=0.0),
        target_position_m=Vector3(x=10.0, y=0.0, z=1.0),
    )


def test_runtime_query_includes_horizon_hazards_beyond_six_meters_and_actual_overspeed():
    wall = dict(
        name="far-wall",
        center_x=7.0,
        center_y=0.0,
        center_z=1.0,
        size_x=0.1,
        size_y=1.0,
        size_z=2.0,
    )
    index = StaticGeometryIndex([wall], bounds=_primitive_bounds)
    assert not index.nearby((0.0, 0.0, 1.0))
    radius = runtime_safety_query_radius_m(
        observation=observation(),
        vehicle=_vehicle(),
        required_clearance_m=0.3,
        maximum_speed_mps=2.0,
    )
    assert index.nearby((0.0, 0.0, 1.0), radius_m=radius) == [wall]
    slow = runtime_safety_query_radius_m(
        observation=observation(),
        vehicle=_vehicle(),
        required_clearance_m=0.3,
        maximum_speed_mps=0.4,
    )
    overspeed = runtime_safety_query_radius_m(
        observation=observation(4.0),
        vehicle=_vehicle(),
        required_clearance_m=0.3,
        maximum_speed_mps=0.4,
    )
    assert slow == 6.0 and overspeed > 14.0


@pytest.mark.parametrize("speed", [0.0, -1.0, math.nan, math.inf])
def test_runtime_query_rejects_invalid_cap_before_minimum_can_conceal_it(speed):
    with pytest.raises(ValueError, match="finite and positive"):
        runtime_safety_query_radius_m(
            observation=observation(),
            vehicle=_vehicle(),
            required_clearance_m=0.3,
            maximum_speed_mps=speed,
        )


def test_query_never_truncates_when_larger_reachable_volume_exceeds_capacity():
    walls = [
        dict(
            name=str(i),
            center_x=7.0,
            center_y=0.0,
            center_z=1.0,
            size_x=0.1,
            size_y=1.0,
            size_z=2.0,
        )
        for i in range(257)
    ]
    index = StaticGeometryIndex(walls, bounds=_primitive_bounds)
    radius = prediction_query_radius_m(
        current_velocity_mps=(0.0, 0.0, 0.0),
        maximum_speed_mps=3.0,
        horizon_seconds=3.0,
        vehicle_radius_m=0.3,
        vehicle_height_m=0.4,
        required_clearance_m=0.3,
    )
    with pytest.raises(ValueError, match="CAPACITY_EXCEEDED"):
        index.nearby((0.0, 0.0, 1.0), radius_m=radius)


def test_runtime_geometry_query_includes_retained_acceleration_during_braking():
    measured = observation(4.).model_copy(update={
        "current_acceleration_world_enu_mps2": Vector3(x=3., y=0., z=0.),
    })
    vehicle = _vehicle().model_copy(update={"max_acceleration_mps2": 3.})
    base = runtime_safety_query_radius_m(observation=observation(4.), vehicle=vehicle,
        required_clearance_m=.3, maximum_speed_mps=.4)
    actual = runtime_safety_query_radius_m(observation=measured, vehicle=vehicle,
        required_clearance_m=.3, maximum_speed_mps=.4)
    assert actual == pytest.approx(base + 3. * .25 * 3.)
    wall = dict(name="retained-inertia", center_x=base + 1., center_y=0., center_z=1.,
                size_x=.1, size_y=1., size_z=2.)
    index = StaticGeometryIndex([wall], bounds=_primitive_bounds)
    assert not index.nearby((0., 0., 1.), radius_m=base)
    assert index.nearby((0., 0., 1.), radius_m=actual) == [wall]


@pytest.mark.parametrize("margin", [-.1, math.nan, math.inf])
def test_query_rejects_invalid_inertial_speed_margin(margin):
    with pytest.raises(ValueError, match="QUERY_ENVELOPE_INVALID"):
        prediction_query_radius_m(current_velocity_mps=(0., 0., 0.), maximum_speed_mps=1.,
            horizon_seconds=3., vehicle_radius_m=.2, vehicle_height_m=.3,
            required_clearance_m=.3, acceleration_speed_margin_mps=margin)


def test_inertial_radius_encloses_random_coupled_jerk_limited_predictions():
    rng = random.Random(91)
    for _ in range(250):
        current, acceleration, desired = [
            tuple(rng.uniform(-4., 4.) for _ in range(3)) for _ in range(3)]
        request = _request().model_copy(update={
            "current_velocity_mps": Vector3(**dict(zip("xyz", current, strict=True))),
            "current_acceleration_mps2": Vector3(**dict(zip("xyz", acceleration, strict=True))),
            "max_acceleration_mps2": rng.uniform(.1, 4.),
            "max_jerk_mps3": rng.uniform(.1, 20.),
        })
        forecast = dynamic_safety._reachable_velocity(request, desired)
        acceleration_bound = min(math.hypot(*acceleration), request.max_acceleration_mps2)
        radius = prediction_query_radius_m(current_velocity_mps=current,
            maximum_speed_mps=request.max_speed_mps,
            horizon_seconds=request.prediction_horizon_seconds,
            vehicle_radius_m=request.vehicle_radius_m, vehicle_height_m=request.vehicle_height_m,
            required_clearance_m=request.required_clearance_m,
            acceleration_speed_margin_mps=acceleration_bound * request.prediction_step_seconds)
        assert math.hypot(*forecast) * request.prediction_horizon_seconds < radius
        reached_acceleration = tuple((a - b) / request.prediction_step_seconds
                                     for a, b in zip(forecast, current, strict=True))
        clamped_acceleration = dynamic_safety._limit_magnitude(acceleration,
                                                              request.max_acceleration_mps2)
        assert math.hypot(*reached_acceleration) <= request.max_acceleration_mps2 + 1e-12
        assert math.dist(reached_acceleration, clamped_acceleration) <= (
            request.max_jerk_mps3 * request.prediction_step_seconds + 1e-12)
