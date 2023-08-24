# import pytest
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
