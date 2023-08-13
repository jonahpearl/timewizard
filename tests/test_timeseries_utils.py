# import pytest
import numpy as np
import timewizard.perievent as twp
import timewizard.util as twu


def test_issorted():
    assert twu.issorted(np.arange(10))
    assert not twu.issorted(np.array([0, 1, 2, 1]))


def test_perievent_traces():
    timestamps = np.linspace(0, 100, 1001)
    data = np.sin(timestamps)
    event_timestamps = np.array([20, 30, 50])
    time_window = [-1, 1]
    start_idx = np.where(timestamps == 19)[0][0]
    end_idx = np.where(timestamps == 21)[0][0]

    # Test non-fs version
    _, traces = twp.perievent_traces(
        timestamps, data, event_timestamps, time_window, fs=None
    )
    np.testing.assert_allclose(traces[0], data[start_idx:end_idx])
    assert len(traces) == 3
    assert len(traces[0]) == 20

    # Test fs version
    _, traces = twp.perievent_traces(
        timestamps, data, event_timestamps, time_window, fs=10
    )
    np.testing.assert_allclose(traces[0, :], data[start_idx:end_idx])
    assert traces.shape == (3, 20)


def test_get_perievent_traces_multidim():
    timestamps = np.arange(4)
    data = np.arange(4**4).reshape((4, 4, 4, 4))
    event_timestamps = [2]
    window = [-1, 1]
    idx, traces = twp.perievent_traces(timestamps, data, event_timestamps, window, fs=1)
    np.testing.assert_allclose(traces[0, :], data[1:3, :])


def test_index_of_nearest_value():
    # Simple test
    timestamps = np.arange(10)
    event_timestamps = [2]
    assert twp.index_of_nearest_value(timestamps, event_timestamps) == 2

    # Harder test
    timestamps = np.array([0, 1, 2, 3, 4, 5])
    event_timestamps = np.array([-10, 3.4, 3.5, 3.6, 300])
    answer = np.array([-1, 3, 4, 4, -1])
    assert np.all(answer == twp.index_of_nearest_value(timestamps, event_timestamps))


def test_perievent_events():
    # Simple test
    discrete_timestamps = np.arange(0, 100, 5)
    event_timestamps = [20, 50]
    time_window = (-5, 5)
    times = twp.perievent_events(
        discrete_timestamps,
        event_timestamps,
        time_window,
        zeroed=False
    )
    assert np.all(times[0] == [15, 20, 25])

    times = twp.perievent_events(
        discrete_timestamps,
        event_timestamps,
        time_window,
        zeroed=True
    )
    assert np.all(times[0] == [-5, 0, 5])


def test_find_perievent_times():
    o = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    t = np.arange(o.shape[0])

    i_on, t_on = twp.find_perievent_times(
        o, t, mode="raw", kind="onsets"
    )
    i_off, t_off = twp.find_perievent_times(
        o, t, mode="raw", kind="offsets"
    )
    assert np.all(i_on == np.array([3, 6, 15]))
    assert np.all(i_off == np.array([5, 8, 16]))
    assert np.all(t_on == np.array([3, 6, 15]))
    assert np.all(t_off == np.array([5, 8, 16]))

    i_on, t_on = twp.find_perievent_times(
        o, t, mode="initial_onset", block_min_spacing=3, kind="onsets"
    )
    i_off, t_off = twp.find_perievent_times(
        o, t, mode="initial_onset", block_min_spacing=3, kind="offsets"
    )
    assert np.all(i_on == np.array([3, 15]))
    assert np.all(i_off == np.array([8, 16]))
    assert np.all(t_on == np.array([3, 15]))
    assert np.all(t_off == np.array([8, 16]))


def test_find_perievent_times_boundaries():
    o = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1])
    t = np.arange(o.shape[0])

    i_on, t_on = twp.find_perievent_times(
        o, t, mode="raw", kind="onsets"
    )

    i_off, t_off = twp.find_perievent_times(
        o, t, mode="raw", kind="offsets"
    )

    assert np.all(i_on == np.array([0, 5, 9, 12]))
    assert np.all(i_off == np.array([2, 7, 10, 13]))
    assert np.all(t_on == np.array([0, 5, 9, 12]))
    assert np.all(t_off == np.array([2, 7, 10, 13]))


def test_mutually_nearest_pts():
    t = np.arange(10)
    arr = np.array([0, 1.83, 1.86, 2.01, 2.2, 2.8, 2.99, 3.001, 3.05, 6.9, 7.3, 8.05, 9, 12])
    dts_1, i1, i2 = twu.mutually_nearest_pts(t, arr)
    dts_2, i1, i2 = twu.mutually_nearest_pts(arr, t)

    assert np.allclose(arr[i1], np.array([0, 2.01, 3.001, 6.9, 8.05, 9]))
    assert np.allclose(t[i2], np.array([0, 2, 3, 7, 8, 9]))
    assert np.allclose(dts_1, np.array([0., -0.01, -0.001, 0.1, -0.05, 0.]))
    assert np.allclose(dts_1, -1 * dts_2)
