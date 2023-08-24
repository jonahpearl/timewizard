import pytest
import numpy as np
import timewizard.perievent as twp
# import timewizard.util as twu


def test_time_to_event():
    times = np.arange(7)
    events = [1.2, 4]

    ans = twp.time_to_event(times, events, resolve_equality='left', side="last")
    expected = np.array([np.nan, np.nan, 0.8, 1.8, 2.8, 1., 2.])
    assert np.allclose(ans, expected, equal_nan=True)

    ans = twp.time_to_event(times, events, resolve_equality='right', side="last")
    expected = np.array([np.nan, np.nan, 0.8, 1.8, 0, 1., 2.])
    assert np.allclose(ans, expected, equal_nan=True)

    ans = twp.time_to_event(times, events, resolve_equality='left', side="next")
    expected = np.array([1.2, 0.2, 2, 1, 0, np.nan, np.nan])
    assert np.allclose(ans, expected, equal_nan=True)

    ans = twp.time_to_event(times, events, resolve_equality='right', side="next")
    expected = np.array([1.2, 0.2, 2, 1, np.nan, np.nan, np.nan])
    assert np.allclose(ans, expected, equal_nan=True)


def test_index_of_nearest_value():
    # Simple test
    t = np.arange(10)
    evts = [2]
    assert twp.index_of_nearest_value(t, evts) == 2

    # Harder test
    timestamps = np.array([0, 1, 2, 3, 4, 5])
    event_timestamps = np.array([-10, 3.4, 3.5, 3.6, 300])

    # warn
    ans = twp.index_of_nearest_value(timestamps, event_timestamps, oob_behavior="warn")
    expected = np.array([-1, 3, 4, 4, -1])
    assert np.allclose(ans, expected)

    # error
    with pytest.raises(ValueError):
        ans = twp.index_of_nearest_value(timestamps, event_timestamps, oob_behavior="error")

    # remove
    ans = twp.index_of_nearest_value(timestamps, event_timestamps, oob_behavior="remove")
    expected = np.array([3, 4, 4])
    assert np.allclose(ans, expected)

    # warn and force left
    ans = twp.index_of_nearest_value(timestamps, event_timestamps, oob_behavior="warn", force_side="left")
    expected = np.array([-1, 3, 3, 3, -1])
    assert np.allclose(ans, expected)

    # warn and force right
    ans = twp.index_of_nearest_value(timestamps, event_timestamps, oob_behavior="warn", force_side="right")
    expected = np.array([-1, 4, 4, 4, -1])
    assert np.allclose(ans, expected)

    # remove and force left
    ans = twp.index_of_nearest_value(timestamps, event_timestamps, oob_behavior="remove", force_side="left")
    expected = np.array([3, 3, 3])
    assert np.allclose(ans, expected)

    # remove and force right
    ans = twp.index_of_nearest_value(timestamps, event_timestamps, oob_behavior="remove", force_side="right")
    expected = np.array([4, 4, 4])
    assert np.allclose(ans, expected)
