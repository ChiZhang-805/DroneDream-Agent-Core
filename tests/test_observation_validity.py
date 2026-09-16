"""Deadlines are necessary evidence, not a replacement for collision checks."""
import math
from dataclasses import replace

import pytest

from dronedream_agent_core.observation_validity import (
    SourceDeadlineLatch,
    image_control_deadline,
    observation_validity,
    publication_deadline,
)


def assess(**changes):
    arguments = dict(source_observed_at_unix_ms=10_000, now_unix_ms=10_010,
        inherited_deadline_unix_ms=10_250, clearance_margin_m=1.,
        ego_speed_bound_mps=1., obstacle_speed_bound_mps=0.,
        acceleration_bound_mps2=1., uncertainty_margin_m=.05,
        downstream_reserve_ms=70)
    arguments.update(changes)
    return observation_validity(**arguments)


def test_open_space_keeps_hard_ceiling_and_cannot_renew_at_publication():
    receipt = assess()
    assert receipt.control_deadline_unix_ms == 10_250
    assert receipt.usable_at(10_179, reserve_ms=70)
    assert not receipt.usable_at(10_180, reserve_ms=70)
    assert publication_deadline(evaluated_deadline_unix_ms=10_250,
        published_at_unix_ms=10_150, reserve_ms=70) == 10_250
    assert publication_deadline(evaluated_deadline_unix_ms=10_250,
        published_at_unix_ms=10_180, reserve_ms=70) is None


def test_motion_near_obstacles_shortens_age_not_obstacle_retention():
    receipt = assess(clearance_margin_m=.2, ego_speed_bound_mps=1.5)
    assert 10_090 <= receipt.control_deadline_unix_ms < 10_100
    assert receipt.disposition == "control-eligible"
    assert not receipt.usable_at(10_100)
    expired = assess(clearance_margin_m=.2, ego_speed_bound_mps=1.5, now_unix_ms=10_030)
    assert expired.disposition == "context-only"
    assert not expired.usable_at(10_030)  # cannot resurrect with a smaller reserve


@pytest.mark.parametrize("changes", [
    {"source_observed_at_unix_ms": 10_011}, {"now_unix_ms": True},
    {"downstream_reserve_ms": -1}, {"clearance_margin_m": math.nan},
    {"ego_speed_bound_mps": math.inf}, {"uncertainty_margin_m": -.1},
    {"angular_error_budget_rad": 0}, {"acceleration_bound_mps2": True},
    {"margin_kind": "guess"}, {"minimum_braking_deceleration_mps2": 1.},
])
def test_invalid_bounds_never_admit(changes):
    assert assess(**changes).disposition == "reject"


def test_unknown_braking_cannot_use_vehicle_maximum_acceleration():
    receipt = assess(margin_kind="swept-free-distance", acceleration_bound_mps2=30.)
    assert receipt.reason == "qualified-braking-bound-unavailable"
    assert receipt.disposition == "context-only"


def test_braking_root_includes_growing_speed_and_obstacle_motion_until_stop():
    v, u, a, b, distance = 1., .2, 1., 2., .55
    receipt = assess(clearance_margin_m=distance, uncertainty_margin_m=0.,
        ego_speed_bound_mps=v, obstacle_speed_bound_mps=u, acceleration_bound_mps2=a,
        minimum_braking_deceleration_mps2=b, margin_kind="swept-free-distance",
        downstream_reserve_ms=0)
    t = (receipt.control_deadline_unix_ms - 10_000) / 1000

    def travelled(t):
        return (v+u)*t + a*t*t/2 + (v+a*t)**2/(2*b) + u*(v+a*t)/b

    assert travelled(t) <= distance
    assert travelled(t+.001) > distance
    assert receipt.disposition == "control-eligible"
    assert assess(clearance_margin_m=distance, uncertainty_margin_m=0.,
        ego_speed_bound_mps=v, obstacle_speed_bound_mps=u, acceleration_bound_mps2=a,
        minimum_braking_deceleration_mps2=b, margin_kind="swept-free-distance",
        actuator_response_ms=150).disposition == "context-only"


def test_load_worse_braking_cannot_extend_the_deadline():
    strong = assess(clearance_margin_m=.7, minimum_braking_deceleration_mps2=3.,
                    margin_kind="swept-free-distance")
    weak = assess(clearance_margin_m=.7, minimum_braking_deceleration_mps2=1.,
                  margin_kind="swept-free-distance")
    assert weak.control_deadline_unix_ms < strong.control_deadline_unix_ms


def test_rotation_can_expire_a_stationary_camera():
    receipt = assess(ego_speed_bound_mps=0., acceleration_bound_mps2=0.,
                     angular_speed_bound_rad_s=math.radians(180))
    assert receipt.control_deadline_unix_ms == 10_055
    assert receipt.disposition == "context-only"


def test_empty_margin_and_inherited_revocation_do_not_recover():
    assert assess(clearance_margin_m=.01).disposition == "context-only"
    assert assess(inherited_deadline_unix_ms=0).disposition == "context-only"
    assert not replace(assess(), disposition="reject").usable_at(10_020)


def test_monotonicity_grid_and_numeric_edge_cases():
    for speed in (0., .01, .2, 1., 5., 20.):
        previous = 10_250
        for acceleration in (0., .001, .2, 1., 10., 30.):
            deadline = assess(clearance_margin_m=.3, ego_speed_bound_mps=speed,
                              acceleration_bound_mps2=acceleration).control_deadline_unix_ms
            assert deadline <= previous
            previous = deadline
        base = assess(ego_speed_bound_mps=speed, clearance_margin_m=.4)
        for change in ({"obstacle_speed_bound_mps": 3.}, {"uncertainty_margin_m": .3},
                       {"clearance_margin_m": .15}, {"ego_speed_bound_mps": speed+1}):
            arguments = dict(ego_speed_bound_mps=speed, clearance_margin_m=.4)
            arguments.update(change)
            assert assess(**arguments).control_deadline_unix_ms <= base.control_deadline_unix_ms
    assert assess(acceleration_bound_mps2=1e-250).control_deadline_unix_ms == 10_250
    assert assess(clearance_margin_m=1e308).disposition == "control-eligible"
    assert assess(ego_speed_bound_mps=1e308,
                  acceleration_bound_mps2=1e308).disposition != "control-eligible"


def test_source_exposure_not_receive_time_controls_rgb_deadline():
    image = dict(timestamp_basis="simulation-scene-capture",
        scene_source_unix_ns=10_000_500_000, host_received_unix_ns=10_200_000_000)
    assert image_control_deadline(image, now_unix_ms=10_201) == 10_250
    for altered in ({"scene_source_unix_ns": None}, {"scene_source_unix_ns": True},
                    {"scene_source_unix_ns": 10_300_000_000},
                    {"timestamp_basis": "host-receive-not-hardware-exposure"}):
        assert image_control_deadline({**image, **altered}, now_unix_ms=10_201) == 0
    assert image_control_deadline(image, now_unix_ms=10_251) == 0


def test_latch_blocks_same_frame_rejuvenation_and_out_of_order_replay():
    latch = SourceDeadlineLatch()
    assert latch.restrict(source_unix_ms=1000, deadline_unix_ms=1250) == 1250
    assert latch.restrict(source_unix_ms=1000, deadline_unix_ms=1100) == 1100
    assert latch.restrict(source_unix_ms=1000, deadline_unix_ms=1250) == 1100
    assert latch.restrict(source_unix_ms=1000, deadline_unix_ms=0) == 0
    assert latch.restrict(source_unix_ms=1000, deadline_unix_ms=1250) == 0
    assert latch.restrict(source_unix_ms=1100, deadline_unix_ms=1350) == 1350
    assert latch.restrict(source_unix_ms=1000, deadline_unix_ms=1400) == 0


@pytest.mark.parametrize("numeric", [float, int])
def test_extreme_finite_braking_bound_fails_closed_without_arithmetic_exception(numeric):
    # JSON may spell physical values as integer literals. Their type must not
    # turn an unrepresentable braking calculation into a crashed control tick.
    receipt = assess(
        ego_speed_bound_mps=numeric(1e300),
        acceleration_bound_mps2=numeric(1e300),
        minimum_braking_deceleration_mps2=1.,
        margin_kind="swept-free-distance",
    )
    assert receipt.disposition == "reject"
    assert receipt.reason == "motion-bound-overflow"
