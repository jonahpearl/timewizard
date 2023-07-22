
import bisect
import numpy as np
import os
from scipy.interpolate import interp1d
import warnings

from . import np_utils

### PERI-EVENTS UTILS ###

def get_aligned_traces(
    data_timestamps, data_values, event_timestamps, time_window, fs=None, ts_err_behav='error'
):
    """Get traces of data, aligned to event times

    Arguments:
        data_timestamps {np.array} -- timestamps for the data.
        data_values {np.array} -- the data itself. Timestamps must align to the first dim.
        event_timestamps {np.array} -- times to align to
        time_window {iterable} -- [start_offset, end_offset] in same time base as data.
            Eg, [-2, 5] for a window from 2 sec before to 5 sec after.

    Keyword Arguments:
        fs {int} -- sampling rate in Hz. If none, data are not assumed to be regularly sampled,
             and things are handled more carefully
             (eg traces will be a nested list instead of a matrix,
             because there might be different numbers of datapoints for each window)
            (default: {None}).
        ts_err_behav {str} -- if "error", throw an error if fs but timestamps diffs not all equal.
            Else if "warn", report a warning but keep going.
    Returns:
        times, traces {tuple of np.arrays or lists}
            if fs: a single timestamps array (zeroed at event onset), and an events x time x (extra dims) matrix
            else: lists with times (absolute) and data for each event
    """

    data_timestamps, data_values, event_timestamps = np_utils.castnp(
        data_timestamps, data_values, event_timestamps
    )
    if time_window[0] > 0:
        warnings.warn(
            f"The first element ({time_window[0]}) of your time window is positive and will start after the stim -- did you mean ({-1*time_window[0]}?"
        )

    assert np_utils.issorted(data_timestamps)
    assert len(data_timestamps) == data_values.shape[0]

    if fs:
        if fs < 1:
            warnings.warn(
                "Provided sampling rate is less than 1 Hz -- did you accidentally provide the sampling period? (1/rate)"
            )
        start_ind_offset = int(time_window[0] * fs)
        end_ind_offset = int(time_window[1] * fs)
        traces = np.zeros((len(event_timestamps), (end_ind_offset - start_ind_offset), *data_values.shape[1:]))
        times = np.arange(time_window[0], time_window[1], 1 / fs)
        assert times.shape[0] == traces.shape[1]
        g = generate_perievent_slices(
            data_timestamps, event_timestamps, time_window, sampling_rate=fs, behavior_on_non_identical_timestamp_diffs=ts_err_behav
        )
        for iSlice, s in enumerate(g):
            traces[iSlice, :] = get_padded_slice(data_values, s)
    else:
        if data_values.ndim > 1:
            raise NotImplementedError("Multi dim for data_values and no fs not implemented yet...use nested lists!")
        traces = []
        times = []
        g = generate_perievent_slices(
            data_timestamps, event_timestamps, time_window, sampling_rate=None
        )
        for iSlice, s in enumerate(g):
            traces.append(get_padded_slice(data_values, s))
            times.append(get_padded_slice(data_timestamps, s))

    return times, traces


def get_padded_slice(full_trace, s, pad_val=np.nan):
    """Get slices from data while respecting boundaries

    Arguments:
        full_trace {np.array} -- the data from which to slice the trace. If multi-dimemsional, the first axis is the sliced axis.
        s {slice} -- a slice object. Negative values 

    Keyword Arguments:
        pad_val {float} -- value for parts of the trace that dont exist in the full_trace (default: {np.nan})
    """
    if s.stop < s.start:
        raise ValueError('Slice stop cannot be less than slice start!')
    trace_len = s.stop - s.start
    if (s.start > full_trace.shape[0]):
        new_size = (trace_len, *full_trace.shape[1:])
        trace = np.zeros(new_size)*np.nan
    elif s.start < 0:
        n_left_pad = -1 * s.start
        pad_size = (n_left_pad, *full_trace.shape[1:])
        trace = np.hstack([np.zeros(pad_size) * np.nan, full_trace[0 : s.stop]])
    elif s.stop > full_trace.shape[0]:
        n_right_pad = s.stop - full_trace.shape[0]
        pad_size = (n_right_pad, *full_trace.shape[1:])
        trace = np.hstack([full_trace[s], np.zeros(pad_size) * np.nan,])
    else:
        trace = full_trace[s]
    return trace

def get_times_in_perievent_windows(times, event_timestamps, time_window, zeroed=True, also_return_indices=False):
    """Get nested list of times that fall within each peri-event window.
        Eg, you have a list of lick times (times) and a list of trial start times (event_timestamps), and you want to find the list of lick times within each trial.
    Arguments:
        times {np.array} -- potentially discrete timestamps to be windowed (must be sorted!)
        event_timestamps {np.array} -- times to align to
        time_window {iterable} -- [start_offset, end_offset] in same time base as data.
            Eg, [-2, 5] for a window from 2 sec before to 5 sec after.
            Assumed to be symmetric for each event.
            TODO: allow passing a list of time windows for each event.
        zereod {boolean} -- if True, returns peri-stimulus time, i.e. t=0 at event onset. If false, returns the un-aligned timestamp.

    Returns:
        list -- list of np arrays with times for each peri-stimulus window
    """
    data_timestamps, event_timestamps, time_window = np_utils.castnp(
        times, event_timestamps, time_window
    )
    assert np_utils.issorted(data_timestamps)

    times = []
    idxs = []
    start_indices = np.searchsorted(
        data_timestamps, event_timestamps + time_window[0], side="left"
    )
    end_indices = np.searchsorted(
        data_timestamps, event_timestamps + time_window[1], side="right"
    )
    start_indices, end_indices = np_utils.castnp(
        start_indices, end_indices
    )  # cast to iterables in case of only one
    for iEvent, (start, end) in enumerate(zip(start_indices, end_indices)):
        if zeroed:
            these_times = data_timestamps[start:end] - event_timestamps[iEvent]
        else:
            these_times = data_timestamps[start:end]
        times.append(these_times)
        idxs.append(list(range(start,end)))

    if also_return_indices:
        return times, idxs
    else:
        return times


### INTERPOLATION UTILS
def interp_continuous(old_times, old_data, fs=200, interp_kind="linear", **kwargs):
    """Interpolate data to a steady sampling rate

    Arguments:
        old_times {np.array} -- timestamps for the data
        old_data {np.array} -- the data

    Keyword Arguments:
        fs {int} -- new sampling rate (default: {200})
        interp_kind {str} -- passed to interp1d argument "kind" (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
        **kwargs {dict} -- pased as extra kwargs to interp1d

    Returns:
        new_times {np.array} -- timestamps for the new data
        new_data {np.array} -- resampled data
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


### SMALL UTILS


def rle(inarray):
    """https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
    run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
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
    """Calculate a moving average along x, using width w

    Arguments:
        x {array-like} -- the data
        w {int} -- window size

    Keyword Arguments:
        convolve_mode {str} -- passed to np.convolve mode

    Returns:
        np.array -- normalized moving average
    """
    return np.convolve(x, np.ones(w), mode=convolve_mode) / w


def issorted(a):
    """Check if an array is sorted

    Arguments:
        a {array-like} -- the data

    Returns:
        {bool} -- whether array is sorted in ascending order
    """
    if type(a) is not np.ndarray:
        a = np.array(a)
    return np.all(a[:-1] <= a[1:])


def get_arr_deriv(arr, fill_value=np.nan):
    """Take the first discrete derivative of an array, keeping the overall length the same.

    Arguments:
        arr {array-like} -- the data

    Keyword Arguments:
        fill_value {number} -- value to fill the first element with (default: {np.nan})

    Returns:
        [np.array] -- the derivative, via np.diff
    """
    if type(arr) is not np.ndarray:
        arr = np.array(arr)
    darr = np.diff(arr)
    darr = np.insert(darr, 0, fill_value)
    return darr


def get_peristim_times(
    stim_bool, 
    times, 
    mode="raw", 
    block_min_spacing=500, 
    onsets_or_offsets="onsets",
    custom_onset_times=None,
):
    """Take a list of times and aligned bools for a stimulus, and return onset times for the stimulus.
    Alternatively, pass a list of custom_onset_times and use block_min_spacing to filter it.

    Arguments:
        stim_bool {array-like} -- array of 0s / 1s indicating when stimulus was on. 0: OFF, 1: ON, length T.
            Alternatively, may be array of 0s / integers greater than 0 (eg pwm).
            However, do not mix integers within a stim.
                Acceptable: [0,0,1,1,1,0,0,0,3,3,3,0,0,0,...]
                Not acceptable: [0,0,0,1,2,3,0,0,0,...]
        times {array-like} -- array of timestamps, length T.

    Keyword Arguments:
        mode {str} -- {'raw', 'initial_onset'}
            'raw': get every onset (places where diff > 0). Default behavior.
            'initial_onset': take only onsets that have no onsets at least block_min_spacing units of time before them.

        block_min_spacing {int} -- if mode is initial_onset, how long should bl(default: {500}). Units should match "times".

    Returns:
        [tuple(np.array, np.array)] -- indices (into "times") and timestamps (from "times") for the onsets
    """
    
    if np.issubdtype(stim_bool.dtype, np.unsignedinteger):
        raise TypeError(f'stim_bool is type {stim_bool.dtype} but must be signed in order for diff to work properly!')

    if custom_onset_times is None:
        stim_bool, times = np_utils.castnp(stim_bool, times)
        if stim_bool.dtype == 'bool':
            stim_bool = stim_bool.astype('int')  # np.diff doesn't case to int automatically, so do it here.
        assert stim_bool.shape[0] == times.shape[0]
        if stim_bool.sum() == 0:
            warnings.warn('No stims detected')
            return None, None

        # I use "onset" in var names for simplicity but it could equally be offsets
        if onsets_or_offsets == "onsets":
            # sometimes I report PWM by spitting out the pwm val in optoLed, so can't just look for diff == 1 (ie 00001111000).
            onset_func = lambda stim_bool: np.where(np.diff(stim_bool) > 0)[0] + 1
            time_diff_func = lambda onset_times: np.diff(onset_times)
        elif onsets_or_offsets == "offsets":
            onset_func = lambda stim_bool: np.where(np.diff(stim_bool) < 0)[0] + 1
            time_diff_func = lambda onset_times: -1 * np.diff(onset_times[::-1])[::-1]
        else:
            raise ValueError(
                "onset/offset kwarg not recognized (either 'onsets'or 'offsets'"
            )

        onset_idx = onset_func(stim_bool)
        onset_times = times[onset_idx]
    else:
        onset_times = np_utils.castnp(custom_onset_times)

    # Catch boundary cases
    # (This is weird in the case that the very final point is an onset -- then it's also an offset?? Ditto if only the very first point is a 1.)
    if stim_bool[0] == 1 and onsets_or_offsets=="onsets":
        onset_idx = np.hstack([0, onset_idx])
        onset_times = np.hstack([times[0], onset_times])
    elif stim_bool[-1] == 1 and onsets_or_offsets=="offsets":
        onset_idx = np.hstack([onset_idx, stim_bool.shape[0]-1])
        onset_times = np.hstack([onset_times, times[-1]])

    # Filter detected onsets if requested
    if mode == "raw":
        pass  # no filtering

    elif mode == "initial_onset":

        # Ignore onsets that closely follow others

        # Find differences in onset times 
        time_diffs = time_diff_func(onset_times)

        # Concat on a big num on the end to get first / last one
        if onsets_or_offsets == "onsets":
            initial_idx = (
                np.concatenate([(block_min_spacing * 2,), time_diffs])
                > block_min_spacing
            )
        elif onsets_or_offsets == "offsets":
            initial_idx = (
                np.concatenate([time_diffs, (block_min_spacing * 2,)])
                > block_min_spacing
            )

        # Only take times with spacing larger than requested
        onset_times = onset_times[initial_idx]
        onset_idx = onset_idx[initial_idx]
    else:
        raise ValueError("mode not recognized")

    onset_idx, onset_times = np_utils.castnp(onset_idx, onset_times)

    return onset_idx, onset_times


def get_stim_durations(stim_bool, stim_times, onset_idx):
    """Given a vector of 0's and 1's, timestamps, and list of onset idx, get durations of periods of 1's"""
    stim_bool, stim_times, onset_idx = np_utils.castnp(stim_bool, stim_times, onset_idx)
    if len(stim_bool) != len(stim_times):
        raise ValueError("Stim bool and stim times lengths must match!")

    all_off_idx = np.where(stim_bool == 0)[0]
    durations = []
    for idx in onset_idx:

        # If this isn't in a stim, skip it
        if stim_bool[idx] == 0:
            durations.append(-1)
            continue

        # Catch indices accidentally in the middle of a stim
        if stim_bool[idx - 1] == 1:
            warnings.warn(
                "Some stim idx do not appear to be the starts of stimulations"
            )

        this_off_idx = np.argmax(
            all_off_idx > idx
        )  # first off after this on; using argmax() is a nice little hack
        durations.append(stim_times[all_off_idx[this_off_idx]] - stim_times[idx])

    return np.array(durations)


def mutually_nearest_pts(t1, t2):
    """ Given two vectors of times, find the set of time pairs which are mutually closest to each other.
    
    Arguments:
        t1, t2: vectors of times

    Returns: a tuple of
        diffs: vector of time differences between the mutually nearest times (t1 minus t2 --> positive means t2 leads t1).
        mutually_nearest_1_bool: boolean mask into t1 for the mutually nearest times
        mutually_nearest_2_bool: boolean mask into t2 for the mutually nearest times
    
    This function is symmetric to swapping t1 <--> t2. 
    Only includes times within both vector's min/max times.
    
    Visually shown on a raster, if the time vectors were:
    |||  |     |
    |     |||
    then the mutually closest pairs are (row1 #0 , row2 #0), (row1 #3, row2 #1), etc
    
    Code example:
    t = np.arange(10)
    arr = np.array([0, 1.83, 1.86, 2.01, 2.2, 2.8, 2.99, 3.001, 3.05, 7.02, 7.03, 8.05, 9, 12])
    dts_1, i1, i2 = dt_from_paired_nearest_cps(t, arr)
    dts_2, i1, i2 = dt_from_paired_nearest_cps(arr, t)
    >>> np.arange(10)[i2]
            array([0, 2, 3, 7, 8, 9])
    >>> dts_1
            array([ 0.   , -0.01 , -0.001, -0.02 , -0.05 ,  0.   ])
    np.all(dts_1 == dts_2*-1)  # True
    
    Example with plot:
    t1 = np.array([0,1,2,3,4.1,4.2,4.3,5,7,8])
    t2 = np.array([0, 1.6, 3.7, 4.23, 4.8, 6,7.2])
    dts, t1_bool, t2_bool = dt_from_paired_nearest_cps(t1, t2)
    ls = [['solid' if b else (0,(1,3)) for b in vec] for vec in [t1_bool, t2_bool]]
    plt.figure()
    plt.eventplot([t1, t2], lineoffsets=[1,2], colors=['C0', 'C1'])
    plt.figure()
    plt.eventplot([t1, t2], lineoffsets=[1,2], colors=['C0', 'C1'], linestyles=ls)
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
    