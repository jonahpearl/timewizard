import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings

from . import util as twu
from .allenbrainobs.obs_utils import index_of_nearest_value, time_to_event, generate_perievent_slices


def perievent_traces(
    data_timestamps,
    data_vals,
    event_timestamps,
    time_window,
    fs=None,
    ts_err_behav='error',
):
    """Get peri-event traces of data.
    TODO: allow interpolation.

    Parameters
    ----------
        data_timestamps : array of shape (N,...)
            Timestamps of the data. Must be sorted along the time (0th) axis!

        data_vals : array of shape (N,...)
            The data corresponding to the timestamps.

        event_timestamps : array of shape (M,)
            Times to which to align the data.
            The output will be sorted in the same order as event_timestamps.

        time_window : tuple (start_offset, end_offset), in the same time base as the data.
            For example, (-2,5) with times in seconds will use a window from 2 seconds before to 5 seconds after each event timestamp.

        fs : int, default=None
            The sampling rate of the data, in Hz.
            If None, data are not assumed to be regularly sampled, and things are handled more carefully (eg traces will be a nested list instead of a matrix).

        ts_err_behav: str, default="error"
            Throw an error if fs but timestamps diffs not all equal.
            Else if "warn", report a warning but keep going.
            # TODO: replace this and other similar arguments with global settings that dictate how much double-checking tw should do of the user's inputs
            # This global setting can also obv be overridden with kwargs to specific functions.

    Returns
    --------
        times: array of shape (fs * (time_window[1] - time_window[0]),)
            The timestamps for the peri-event traces.
            If fs is None, this is a nested list of timestamps for each event.

        traces : array of shape (M, fs * (time_window[1] - time_window[0]), ...)
            The aligned peri-event data.
            If fs is None, this is a list of arrays with data for each event

    Raises
    ------
    ValueError:
        If size of data_timestamps and data_vals don't match.
    """

    data_timestamps, data_vals, event_timestamps = twu.castnp(
        data_timestamps, data_vals, event_timestamps
    )

    # TODO: incorporate global double-checking settings on these two things
    if time_window[0] > 0:
        warnings.warn(
            f"The first element ({time_window[0]}) of your time window is positive and will start after the stim -- did you mean ({-1*time_window[0]}?"
        )
    assert twu.issorted(data_timestamps)

    if len(data_timestamps) != data_vals.shape[0]:
        raise ValueError('data_timestamps and data_vals must have the same length!')

    if fs:
        if fs < 1:
            warnings.warn(
                "Provided sampling rate is less than 1 Hz -- did you accidentally provide the sampling period? (ie 1/rate?)"
            )
        start_ind_offset = int(time_window[0] * fs)
        end_ind_offset = int(time_window[1] * fs)
        traces = np.zeros((len(event_timestamps), (end_ind_offset - start_ind_offset), *data_vals.shape[1:]))
        times = np.arange(time_window[0], time_window[1], 1 / fs)
        assert times.shape[0] == traces.shape[1]
        g = generate_perievent_slices(
            data_timestamps, event_timestamps, time_window, sampling_rate=fs, behavior_on_non_identical_timestamp_diffs=ts_err_behav
        )
        for iSlice, s in enumerate(g):
            traces[iSlice, :] = _get_padded_slice(data_vals, s)
    else:
        if data_vals.ndim > 1:
            raise NotImplementedError("Multi dim for data_values and no fs not implemented yet...use nested lists!")
        traces = []
        times = []
        g = generate_perievent_slices(
            data_timestamps, event_timestamps, time_window, sampling_rate=None
        )
        for iSlice, s in enumerate(g):
            traces.append(_get_padded_slice(data_vals, s))
            times.append(_get_padded_slice(data_timestamps, s))

    return times, traces


def _get_padded_slice(full_trace, s, pad_val=np.nan):
    """Get slices of data, and preserve the size of the slice even if the starts / stops are out of bounds of the data.

    Parameters
    ----------
    full_trace : array of shape (N,...)
        The data from which to slice the trace. If multi-dimensional, the first axis is the sliced axis.

    s : slice of (start, stop)
        A slice object. Negative values imply an intention to get data before the start of the trace,
        rather than typical negative indexing in python.

    pad_val : float, default=np.nan
        Value for parts of the trace that don't exist in the full_trace.

    Returns
    -------
    trace : np.array
        The sliced trace, respecting boundaries with padding.

    Raises
    ------
    ValueError:
        If slice stop is less than slice start.

    """
    if s.stop < s.start:
        raise ValueError('Slice stop cannot be less than slice start!')
    trace_len = s.stop - s.start
    if (s.start > full_trace.shape[0]):
        new_size = (trace_len, *full_trace.shape[1:])
        trace = np.full(new_size, pad_val)
    elif s.start < 0:
        n_left_pad = -1 * s.start
        pad_size = (n_left_pad, *full_trace.shape[1:])
        trace = np.hstack([np.full(pad_size, pad_val), full_trace[0:s.stop]])
    elif s.stop > full_trace.shape[0]:
        n_right_pad = s.stop - full_trace.shape[0]
        pad_size = (n_right_pad, *full_trace.shape[1:])
        trace = np.hstack([full_trace[s], np.full(pad_size, pad_val)])
    else:
        trace = full_trace[s]
    return trace


def perievent_events(
        discrete_timestamps,
        event_timestamps,
        time_window,
        zeroed=True,
        also_return_indices=False
):
    """Get nested list of discrete times that fall within each peri-event window.

        For example, if you have a list of lick times (discrete_timestamps)
        and a list of trial start times (event_timestamps), you could use
        this function to get a list of lick times around the start of each trial.

    Parameters
    ----------
    discrete_timestamps : array of shape (N,)
        Discrete timestamps (i.e. events like licks, spikes, rears, etc.) to be windowed. Must be sorted!

    event_timestamps : array of shape (M,)
        Times to which to align the data.
        The output will be sorted in the same order as event_timestamps.

    time_window : tuple (start_offset, end_offset), in the same time base as the data.
            For example, (-2,5) with times in seconds will use a window from 2 seconds before to 5 seconds after each event timestamp.

    zeroed : bool, default=True
        If True, returns peri-stimulus timestamps, i.e. t=0 at event onset. 
        If False, returns the un-aligned timestamp.

    also_return_indices : bool, default=False
        If True, return both the times and their indices.

    Returns
    -------
    times : list of np.arrays
        List of arrays with times for each peri-stimulus window.

    idxs : list of lists, optional
        List of indices corresponding to the times, returned only if `also_return_indices` is True.
    """
    discrete_timestamps, event_timestamps, time_window = twu.castnp(
        discrete_timestamps, event_timestamps, time_window
    )

    # TODO: incorporate global double-checking settings on this
    assert twu.issorted(discrete_timestamps)

    aligned_timestamps = []
    idxs = []
    start_indices = np.searchsorted(
        discrete_timestamps, event_timestamps + time_window[0], side="left"
    )
    end_indices = np.searchsorted(
        discrete_timestamps, event_timestamps + time_window[1], side="right"
    )

    # cast to iterables in case of only one
    start_indices, end_indices = twu.castnp(
        start_indices, end_indices
    )
    for iEvent, (start, end) in enumerate(zip(start_indices, end_indices)):
        if zeroed:
            these_times = discrete_timestamps[start:end] - event_timestamps[iEvent]
        else:
            these_times = discrete_timestamps[start:end]
        aligned_timestamps.append(these_times)
        idxs.append(list(range(start, end)))

    if also_return_indices:
        return aligned_timestamps, idxs
    else:
        return aligned_timestamps


def event_times_from_train(
    data_train,
    data_timestamps,
    mode="raw",
    kind="onsets",
    block_min_spacing=None,
):
    """
    Return onset (or offset) times for when some boolean timeseries becomes (or stops being) True.

    Parameters
    ----------
    data_boolean : array of size (N,) and type bool or signed integers.
        Array of 0s/1s indicating stimulus state (0: OFF, 1: ON).
        Also acceptable: array with runs of any integer values, interspersed with zeros: `[0,0,1,1,1,0,0,0,3,3,3,0,0,0,...]`
        Not acceptable: mixing values within a run: `~[0,0,0,1,2,3,0,0,0,...]~`

    data_timestamps : array-like of size (N,)
        Array of timestamps for the data.

    mode : str, default="raw"
        Either 'raw' for every onset or 'initial_onset' for onsets with a gap defined by block_min_spacing.

    kind : str, default="onsets"
        Either 'onsets' or 'offsets'.

    block_min_spacing : int, default=None
        Time unit gap required between events if mode is 'initial_onset'.

    Returns
    -------
    tuple of np.array
        Indices and timestamps for the onsets.
    """

    # Check kwargs
    if kind not in ["onsets", "offsets"]:
        raise ValueError("kind must be 'onsets' or 'offsets'")
    if mode not in ["raw", "initial_onset"]:
        raise ValueError("mode must be 'raw' or 'initial_onset'")
    if mode == 'initial_onset' and block_min_spacing is None:
        raise ValueError("block_min_spacing must be provided if mode is 'initial_onset'")

    # Check for compatible input data types
    if data_train.dtype == 'bool':
        data_train = data_train.astype('int')
    if np.issubdtype(data_train.dtype, np.unsignedinteger):
        raise TypeError(f'stim_bool is type {data_train.dtype} but must be signed in order for diff to work properly!')
    if np.any(data_train < 0):
        raise ValueError('stim_bool must be all positive or 0')

    # Some sanity checks
    data_train, data_timestamps = twu.castnp(data_train, data_timestamps)
    assert data_train.shape[0] == data_timestamps.shape[0]
    if data_train.sum() == 0:
        warnings.warn('No stims detected')
        return None, None

    # Get onsets (or offsets)
    if kind == "onsets":
        # Can't just use diff == 1, since we allow arbitrary non-zero integers
        event_func = lambda stim_bool: np.where(np.diff(stim_bool) > 0)[0] + 1
    elif kind == "offsets":
        event_func = lambda stim_bool: np.where(np.diff(stim_bool) < 0)[0] + 1
    event_idx = event_func(data_train)
    event_times = data_timestamps[event_idx]

    # Catch boundary cases
    # (This is weird in the case that the very final point is an onset -- then it's also an offset?? Ditto if only the very first point is a 1.)
    if data_train[0] == 1 and kind == "onsets":
        event_idx = np.hstack([0, event_idx])
        event_times = np.hstack([data_timestamps[0], event_times])
    elif data_train[-1] == 1 and kind == "offsets":
        print(data_train.shape)
        event_idx = np.hstack([event_idx, data_train.shape[0] - 1])
        event_times = np.hstack([event_times, data_timestamps[-1]])

    # Chunk detected events if requested
    if mode == "raw":
        pass
    elif mode == "initial_onset":
        event_times, event_idx = chunk_events(event_times, block_min_spacing, *(event_idx,), kind=kind)
    else:
        raise ValueError("mode not recognized")

    event_times, event_idx = twu.castnp(event_times, event_idx)

    return event_idx, event_times


def chunk_events(
        event_times,
        block_min_spacing,
        *args,
        kind='onsets'
):
    """Return only the event times (and corresponding vals) that are more than block_min_spacing apart.

    Parameters
    ----------
    event_times : np.array of shape (M,)
        Vector of event times.

    block_min_spacing : int
        Minimum time difference between two events.

    *args: np.arrays of shape (M,)
        Other arrays that will be chunked in the same way as onset_times.

    kind : str, default="onsets"
        Specifies if we're working with "onsets" or "offsets".
    Returns
    -------
    np.array
        Filtered onset times that are more than block_min_spacing apart.
    """

    if kind == "onsets":
        time_diff_func = lambda times: np.diff(times)
    elif kind == "offsets":
        time_diff_func = lambda times: -1 * np.diff(times[::-1])[::-1]
    else:
        raise ValueError("kind not recognized (either 'onsets'or 'offsets'")

    # Find differences in onset times
    time_diffs = time_diff_func(event_times)

    # Concat on a big num on the end to get first / last one
    if kind == "onsets":
        time_diffs = np.concatenate([(block_min_spacing * 2,), time_diffs])
    elif kind == "offsets":
        time_diffs = np.concatenate([time_diffs, (block_min_spacing * 2,)])

    # Only take times with spacing larger than requested
    initial_idx = time_diffs > block_min_spacing

    return (event_times[initial_idx], *(arg[initial_idx] for arg in args))


def event_epochs(
    data_train,
    data_timestamps,
    mode="raw",
    block_min_spacing=None,
):
    """Quantifies characteristics (onset, offset, duration, value) of event epochs in a boolean timeseries.

    Parameters:
    -----------
    data_train : np.array of shape (N,)
        Boolean or integer time series representing the presence (1/True) or absence (0/False) of events.
    data_timestamps : np.array of shape (N,)
        Timestamps corresponding to the data_train.
    mode : str, optional (default = "raw")
        Specifies the event quantification mode:
        - "raw": Extracts all onsets and offsets as they occur.
        - "initial_onset": Consolidates events that are spaced closer than `block_min_spacing`.
    block_min_spacing : float, optional
        Minimum spacing between event epochs. Only relevant if mode is "initial_onset".

    Returns:
    --------
    df : pd.DataFrame of len N
        DataFrame containing the following columns:
        - 'onset_index': Index of event onset in data_train.
        - 'onset_time': Timestamp of event onset.
        - 'offset_index': Index of event offset in data_train.
        - 'offset_time': Timestamp of event offset.
        - 'duration': Duration of the event.
        - 'value': Value of the event at the onset (usually 1 for boolean series).

    Raises:
    -------
    ValueError, TypeError, AssertionError
        Various input checks to ensure data integrity.
    """

    # Check kwargs
    if mode not in ["raw", "initial_onset"]:
        raise ValueError("mode must be 'raw' or 'initial_onset'")
    if mode == 'initial_onset' and block_min_spacing is None:
        raise ValueError("block_min_spacing must be provided if mode is 'initial_onset'")

    # Check for compatible input data types
    if data_train.dtype == 'bool':
        data_train = data_train.astype('int')
    if np.issubdtype(data_train.dtype, np.unsignedinteger):
        raise TypeError(f'stim_bool is type {data_train.dtype} but must be signed in order for diff to work properly!')
    if np.any(data_train < 0):
        raise ValueError('stim_bool must be all positive or 0')

    # Some sanity checks
    data_train, data_timestamps = twu.castnp(data_train, data_timestamps)
    assert data_train.shape[0] == data_timestamps.shape[0]
    if data_train.sum() == 0:
        warnings.warn('No stims detected')
        return None, None

    # Get onsets (or offsets)
    onset_idx, onset_times = event_times_from_train(
        data_train,
        data_timestamps,
        kind="onsets",
    )

    # Get offsets (or onsets)
    offset_idx, offset_times = event_times_from_train(
        data_train,
        data_timestamps,
        kind="offsets",
    )

    # In order to chunk events correctly, need to jointly consider onsets and offsets
    # If the spacing between a given offset and the next onset is less than block_min_spacing, then we need to remove that offset
    if mode == 'initial_onset':

        # Get spacing between offsets and onsets
        offset_to_next_onset_spacing = onset_times[1:] - offset_times[:-1]

        # Find offsets that are too close to the next onset
        tmp_mask = offset_to_next_onset_spacing < block_min_spacing
        offsets_mask = np.concatenate([~tmp_mask, [True]])  # always keep last offset
        onsets_mask = np.concatenate([[True], ~tmp_mask])  # always keep first onset

        # Remove those offsets and their corresponding subsequent onsets
        offset_idx = offset_idx[offsets_mask]
        offset_times = offset_times[offsets_mask]
        onset_idx = onset_idx[onsets_mask]
        onset_times = onset_times[onsets_mask]

    # Get durations
    durations = offset_times - onset_times

    # Get values
    values = data_train[onset_idx]

    df = pd.DataFrame({
        'onset_index': onset_idx,
        'onset_time': onset_times,
        'offset_index': offset_idx,
        'offset_time': offset_times,
        'duration': durations,
        'value': values,
    })
    return df


def map_values(data_timestamps, data_vals, event_timestamps, interpolate=False):
    """Map values of data to event times. If interpolating, thin wrapper around scipy's interp1d.

    Parameters
    ----------
    data_timestamps : array of shape (N,...)
        Timestamps of the data. Must be sorted along the time (0th) axis!

    data_vals : array of shape (N,...)
        The data corresponding to the timestamps.

    event_timestamps : array of shape (M,)
        Times to which to align the data.
        The output will be sorted in the same order as event_timestamps.

    interpolate : bool, default=False
        If False, use the value at the nearest timestamp.
        If True, use interp1d to interpolate between timestamps.

    Returns
    --------
    vals : array of shape (M, ...)
        The signal value at each event time.

    Raises
    ------
    ValueError:
        If size of data_timestamps and data_vals don't match.
    """

    data_timestamps, data_vals, event_timestamps = twu.castnp(
        data_timestamps, data_vals, event_timestamps
    )

    if data_timestamps.shape[0] != data_vals.shape[0]:
        raise ValueError('data_timestamps and data_vals must have the same number of timepoints!')

    if interpolate:
        f = interp1d(data_timestamps, data_vals, axis=0, bounds_error=False, fill_value=np.nan)
        vals = f(event_timestamps)
    else:
        nearest_idx = index_of_nearest_value(data_timestamps, event_timestamps, oob_behavior='warn')
        vals = data_vals[nearest_idx].astype('float')
        vals[nearest_idx == -1] = np.nan

    return vals
