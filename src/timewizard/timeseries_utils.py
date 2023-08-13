
import numpy as np
from scipy.interpolate import interp1d
import warnings

from . import np_utils
from .allenbrainobs.obs_utils import index_of_nearest_value, generate_perievent_slices

# PERI-EVENTS UTILS #


def perievent_traces(
    data_timestamps,
    data_vals,
    event_timestamps,
    time_window,
    fs=None,
    ts_err_behav='error',
):
    """Get peri-event traces of data.

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

        # TODO: replace this and other similar arguments with global settings that dictate how much double-checking tw should do of the user's inputs
        # This global setting can also obv be overridden with kwargs to specific functions.
        ts_err_behav: str, default="error"
            Throw an error if fs but timestamps diffs not all equal.
            Else if "warn", report a warning but keep going.

    Returns:
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

    data_timestamps, data_vals, event_timestamps = np_utils.castnp(
        data_timestamps, data_vals, event_timestamps
    )

    # TODO: incorporate global double-checking settings on these two things
    if time_window[0] > 0:
        warnings.warn(
            f"The first element ({time_window[0]}) of your time window is positive and will start after the stim -- did you mean ({-1*time_window[0]}?"
        )
    assert np_utils.issorted(data_timestamps)

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
        trace = np.empty(new_size).fill(pad_val)
    elif s.start < 0:
        n_left_pad = -1 * s.start
        pad_size = (n_left_pad, *full_trace.shape[1:])
        trace = np.hstack([np.empty(pad_size).fill(pad_val), full_trace[0:s.stop]])
    elif s.stop > full_trace.shape[0]:
        n_right_pad = s.stop - full_trace.shape[0]
        pad_size = (n_right_pad, *full_trace.shape[1:])
        trace = np.hstack([full_trace[s], np.empty(pad_size).fill(pad_val)])
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
    discrete_timestamps, event_timestamps, time_window = np_utils.castnp(
        discrete_timestamps, event_timestamps, time_window
    )

    # TODO: incorporate global double-checking settings on this
    assert np_utils.issorted(discrete_timestamps)

    aligned_timestamps = []
    idxs = []
    start_indices = np.searchsorted(
        discrete_timestamps, event_timestamps + time_window[0], side="left"
    )
    end_indices = np.searchsorted(
        discrete_timestamps, event_timestamps + time_window[1], side="right"
    )

    # cast to iterables in case of only one
    start_indices, end_indices = np_utils.castnp(
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


# INTERPOLATION UTILS #


def interp_continuous(old_times, old_data, fs=200, interp_kind="linear", **kwargs):
    """
    Interpolate data to a steady sampling rate.

    Parameters
    ----------
    old_times : np.array of shape (N,)
        Timestamps for the data, in seconds.

    old_data : np.array of shape (N,...)
        The data corresponding to the timestamps.

    fs : int, default=200
        New sampling rate in Hz.

    interp_kind : str, default="linear"
        The type of interpolation to use.
        Passed to `interp1d` argument "kind".
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    **kwargs : dict
        Additional keyword arguments passed to `interp1d`.

    Returns
    -------
    new_times : np.array of shape (M,)
        Timestamps for the new data, where M depends on the new sampling rate.

    new_data : np.array of shape (M,...)
        Resampled data.
    """
    old_times, old_data = np_utils.castnp(old_times, old_data)

    f = interp1d(old_times, old_data, kind=interp_kind, **kwargs)
    new_times = np.arange(np.min(old_times), np.max(old_times), 1 / fs)
    if (
        new_times[-1] > old_times[-1]
    ):  # occasionally happens due to rounding; will throw error with defaults; remove for simplicity.
        new_times = np.delete(new_times, -1)
    new_data = f(new_times)
    return new_times, new_data


# SMALL UTILS #


def rle(stateseq):
    """
    Run length encoding. Shamelessly taken from Thomas Browne on stack overflow: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
    He siad: "Partial credit to R's rle function. Multi datatype arrays catered for including non-Numpy."

    Parameters
    ----------
    stateseq : array of shape (N,)
        Input array with various potential runs of values.

    Returns
    -------
    runlengths : np.array of shape (M,)
        Length of sequences of identical values.

    startpositions : np.array of shape (M,)
        Starting position of sequences in the input array.

    values : np.array of shape (M,)
        The values that are repeated.
    """
    ia = np.asarray(stateseq)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def moving_average(x, w, convolve_mode="valid"):
    """
    Calculate a moving average along x, using width w.

    Parameters
    ----------
    x : array-like
        The data.

    w : int
        Window size.

    convolve_mode : str, default="valid"
        Argument passed to np.convolve mode.

    Returns
    -------
    np.array of shape (depends on input and mode)
        Normalized moving average.
    """
    return np.convolve(x, np.ones(w), mode=convolve_mode) / w


def issorted(a):
    """Check if an array is sorted

    Parameters
    ----------
    a : array-like
        The data to check.

    Returns
    -------
    bool
        Whether the array is sorted in ascending order.
    """
    if type(a) is not np.ndarray:
        a = np.array(a)
    return np.all(a[:-1] <= a[1:])


def discrete_deriv(arr, fill_value=np.nan, fill_side='right'):
    """
    Take the first discrete derivative of an array, keeping the overall length the same.

    Parameters
    ----------
    arr : array-like
        The data.

    fill_value : number, default=np.nan
        Value to fill the first element with.

    Returns
    -------
    np.array of shape (matches input shape)
        The derivative, via np.diff.
    """
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    darr = np.diff(arr)
    if fill_side == 'right':
        darr = np.insert(darr, len(darr), fill_value)
    elif fill_side == 'left':
        darr = np.insert(darr, 0, fill_value)
    return darr


def find_perievent_times(
    data_boolean,
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
        Also acceptable: array with runs of any integer values, interspersed with zeros.
            OK: [0,0,1,1,1,0,0,0,3,3,3,0,0,0,...]
        Not acceptable: mixing values within a run.
            NOT OK: [0,0,0,1,2,3,0,0,0,...]

    data_timestamps : array-like of size (N,)
        Array of timestamps for the data.

    mode : str, default="raw"
        Either 'raw' for every onset or 'initial_onset' for onsets with a gap defined by block_min_spacing.

    kind : str, default="onsets"
        Either 'onsets' or 'offsets'.

    block_min_spacing : int, default=None
        Time unit gap required between events if mode is 'initial_onset'.

    custom_onset_times : array-like or None, default=None
        Custom onset times to filter with block_min_spacing if provided.

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
    if data_boolean.dtype == 'bool':
        data_boolean = data_boolean.astype('int')
    if np.issubdtype(data_boolean.dtype, np.unsignedinteger):
        raise TypeError(f'stim_bool is type {data_boolean.dtype} but must be signed in order for diff to work properly!')
    if np.any(data_boolean < 0):
        raise ValueError('stim_bool must be all positive or 0')

    # Some sanity checks
    data_boolean, data_timestamps = np_utils.castnp(data_boolean, data_timestamps)
    assert data_boolean.shape[0] == data_timestamps.shape[0]
    if data_boolean.sum() == 0:
        warnings.warn('No stims detected')
        return None, None

    # Get onsets (or offsets)
    if kind == "onsets":
        # Can't just use diff == 1, since we allow arbitrary non-zero integers
        event_func = lambda stim_bool: np.where(np.diff(stim_bool) > 0)[0] + 1
    elif kind == "offsets":
        event_func = lambda stim_bool: np.where(np.diff(stim_bool) < 0)[0] + 1
    event_idx = event_func(data_boolean)
    event_times = data_timestamps[event_idx]

    # Catch boundary cases
    # (This is weird in the case that the very final point is an onset -- then it's also an offset?? Ditto if only the very first point is a 1.)
    if data_boolean[0] == 1 and kind == "onsets":
        event_idx = np.hstack([0, event_idx])
        event_times = np.hstack([data_timestamps[0], event_times])
    elif data_boolean[-1] == 1 and kind == "offsets":
        print(data_boolean.shape)
        event_idx = np.hstack([event_idx, data_boolean.shape[0] - 1])
        event_times = np.hstack([event_times, data_timestamps[-1]])

    # Chunk detected events if requested
    if mode == "raw":
        pass
    elif mode == "initial_onset":
        event_times, event_idx = chunk_events(event_times, block_min_spacing, *(event_idx,), kind=kind)
    else:
        raise ValueError("mode not recognized")

    event_times, event_idx = np_utils.castnp(event_times, event_idx)

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


# def describe_pulse_trains(
#         stim_boolean,
#         stim_timestamps,
#         block_min_spacing,
# ):
#     """
#     Describe a pulse train by its frequency, duty cycle, and number of pulses.

#     For example, a


#     Parameters
#     ----------
#     stim_boolean : np.array of shape (N,)
#         Boolean array of stimulus state (0: OFF, 1: ON).

#     stim_timestamps : np.array of shape (N,)
#         Timestamps for the stimulus state.

#     block_min_spacing : int
#         Minimum time difference between two stim blocks.

        
#     """

def mutually_nearest_pts(t1, t2):
    """
    Identify pairs of mutually nearest times from two vectors of times.

    Given two vectors of times, this function finds the pairs where each time from the
    first vector is closest to a time in the second vector and vice-versa. The function
    is symmetric, meaning swapping the input vectors (t1 with t2) would yield the inverse
    results.

    Parameters:
    -----------
    t1, t2 : array-like
        Input vectors of times.

    Returns:
    --------
    diffs : np.array
        Time differences between the mutually nearest times. It is computed as (t1 minus t2),
        meaning a positive value indicates t2 leads t1.

    mutually_nearest_1_bool : np.array (boolean mask)
        Boolean mask indicating which times in t1 are part of the mutually nearest pairs.

    mutually_nearest_2_bool : np.array (boolean mask)
        Boolean mask indicating which times in t2 are part of the mutually nearest pairs.

    Notes:
    ------
    Only times within the range of both vectors are considered.

    For visualization: If the times are plotted on a raster as:
        t1: |||  |     |
        t2: |     |||
    The mutually closest pairs are (t1's first time, t2's first time),
    (t1's fourth time, t2's second time), etc.


    Examples:
    ---------
    >>> t = np.arange(10)
    >>> arr = np.array([0, 1.83, 1.86, 2.01, 2.2, 2.8, 2.99, 3.001, 3.05, 7.02, 7.03, 8.05, 9, 12])
    >>> dts_1, i1, i2 = dt_from_paired_nearest_cps(t, arr)
    >>> dts_2, i1, i2 = dt_from_paired_nearest_cps(arr, t)
    >>> np.arange(10)[i2]
    array([0, 2, 3, 7, 8, 9])
    >>> dts_1
    array([ 0.   , -0.01 , -0.001, -0.02 , -0.05 ,  0.   ])
    >>> np.all(dts_1 == dts_2*-1)
    True

    To visualize with plot:
    >>> t1 = np.array([0,1,2,3,4.1,4.2,4.3,5,7,8])
    >>> t2 = np.array([0, 1.6, 3.7, 4.23, 4.8, 6,7.2])
    >>> dts, t1_bool, t2_bool = dt_from_paired_nearest_cps(t1, t2)
    >>> ls = [['solid' if b else (0,(1,3)) for b in vec] for vec in [t1_bool, t2_bool]]
    >>> plt.figure()
    >>> plt.eventplot([t1, t2], lineoffsets=[1,2], colors=['C0', 'C1'])
    >>> plt.figure()
    >>> plt.eventplot([t1, t2], lineoffsets=[1,2], colors=['C0', 'C1'], linestyles=ls)
    """

    # Get initial pairings of groups of times
    # Eg, which times in 1 are nearest each time of 2.
    idxs_of_1_nearest_to_each_2 = index_of_nearest_value(t1, t2)  # in eg above, (0, 3, 3, 3, ...)
    idxs_of_2_nearest_to_each_1 = index_of_nearest_value(t2, t1)  # in eg above, (0, 0, 0, 1, ...)

    # Only take pairs of times which are each other's mutually closest time.
    # You can do this in one line via idx_1_2[idx_2_1] == np.arange(idx_2_1.shape[0]).
    # Eg, consider i=4 for t1. Say t1_4 was closest to t2_3, and t2_3 was in turn closest to t1_4.
    # Then idxs_of_1_nearest_to_each_2[3] == 4 and idxs_of_2_nearest_to_each_1[4] == 3.
    # So idxs_of_1_nearest_to_each_2[idxs_of_2_nearest_to_each_1[i]] == 4 == np.arange(idxs_of_2_nearest_to_each_1.shape[0])[i].
    # We also exclude edge issues by discarding where index_of_nearest_value returned -1 (ie invalid).

    mutually_nearest_1_bool = (idxs_of_1_nearest_to_each_2[idxs_of_2_nearest_to_each_1] == np.arange(idxs_of_2_nearest_to_each_1.shape[0])) & (idxs_of_2_nearest_to_each_1 != -1)
    mutually_nearest_2_bool = (idxs_of_2_nearest_to_each_1[idxs_of_1_nearest_to_each_2] == np.arange(idxs_of_1_nearest_to_each_2.shape[0])) & (idxs_of_1_nearest_to_each_2 != -1)
    assert mutually_nearest_1_bool.sum() == mutually_nearest_2_bool.sum()
    diffs = t1[mutually_nearest_1_bool] - t2[mutually_nearest_2_bool]
    return diffs, mutually_nearest_1_bool, mutually_nearest_2_bool
