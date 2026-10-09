"""A transient bad depth frame must not pin the live selector indefinitely."""

import math

import pytest

from scripts.runtime_depth_safety_worker import _advance_rejected_depth_cursor


def test_rejected_new_frame_is_consumed_once_so_newer_input_can_recover():
    assert _advance_rejected_depth_cursor(10.0, 10.25, new_depth_frame=True) == 10.25


def test_maintenance_failure_does_not_renew_or_advance_source_cursor():
    assert _advance_rejected_depth_cursor(10.0, 10.25, new_depth_frame=False) == 10.0


def test_old_callback_cannot_regress_consumed_cursor():
    assert _advance_rejected_depth_cursor(10.25, 10.0, new_depth_frame=True) == 10.25


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invalid_cursor_values_fail_closed(value):
    with pytest.raises(ValueError):
        _advance_rejected_depth_cursor(10.0, value, new_depth_frame=True)
