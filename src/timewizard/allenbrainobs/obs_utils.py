import numpy as np
import os
from scipy.interpolate import interp1d
import warnings

from .. import np_utils



# This module contains functions that are modified from the Allen Institute's Mindscope Utilities (now Brain Observatory Utils).
# The Allen's code is licensed under the Allen Institute Software License, which allows modification and redistribution of the code
# so long as the license is included. The license is included in this folder, to make explicit which code in timewizard is derived from the Allen's code.

def time_from_last(timestamps, event_times, side="right", ):
    """
    https://github.com/AllenInstitute/mindscope_utilities/blob/e5aa1e6aebf3f62570aaff0e2e9dba835c999a23/mindscope_utilities/visual_behavior_ophys/data_formatting.py#L885

    For each timestamp, returns the time from the most recent event time (in event_times).
    NB: contrary to `tsu.index_of_nearest_value`, this function only looks one direction in time to find "nearest" events (hence "time from last" and not "time to nearest")

    Args:
        timestamps (np.array): array of timestamps for which the 'time from last event' will be returned
        event_times (np.array): event timestamps
        side: if there is a pair of ts / event time that are identical, should it measure wrt the previous event time (left) or the current one (right)?
            eg: tsu.time_from_last(np.arange(7), [1.2, 4], side='left')  --> array([nan, nan, 0.8, 1.8, 2.8, 1. , 2. ]) 
                tsu.time_from_last(np.arange(7), [1.2, 4], side='right') --> array([nan, nan, 0.8, 1.8, 0. , 1. , 2. ])
        
    Returns
        time_from_last_event (np.array): the time from the last event for each timestamp
    """

    timestamps, event_times = np_utils.castnp(timestamps, event_times)
    
    last_event_index = np.searchsorted(a=event_times, v=timestamps, side=side) - 1
    time_from_last_event = timestamps - event_times[last_event_index]
    
    # flashes that happened before the other thing happened should return nan
    time_from_last_event = time_from_last_event.astype('float')  # cast to float to accomodate nans
    time_from_last_event[last_event_index == -1] = np.nan

    return time_from_last_event


def index_of_nearest_value(data_timestamps, event_timestamps, boundary_tol=None):
    """
    https://github.com/AllenInstitute/mindscope_utilities/blob/e5aa1e6aebf3f62570aaff0e2e9dba835c999a23/mindscope_utilities/general_utilities.py#L149

    The index of the nearest sample time for each event time.
    Parameters:
    -----------
    sample_timestamps : np.ndarray of floats
        sorted 1-d vector of data sample timestamps.
    event_timestamps : np.ndarray of floats
        1-d vector of event timestamps.
    boundary_tol: float (default: None)
        If None, excludes event timestamps that are outside of the data timestamps range
        If a float, values within edge_tol of the first/last times are allowed.

    Returns:
    --------
    event_aligned_ind : np.ndarray of int
        An array of nearest sample time index for each event times.
        Event times outside of the bounds are given an index of -1.
        (NB, of course -1 is still a valid index in Python! But there is no integer nan, so -1 is the best way we have to say "invalid intger")
    """
    data_timestamps, event_timestamps = np_utils.castnp(data_timestamps, event_timestamps)

    insertion_ind = np.searchsorted(data_timestamps, event_timestamps)

    # Sanitize event timestamps outside of the data timestamps range
    outside_range_bool = np.logical_or(
        event_timestamps < data_timestamps[0], event_timestamps > data_timestamps[-1]
    )
    if outside_range_bool.sum() > 0:
        warnings.warn(
            "Some event timestamps are outside the range of data timestamps. Such indices will be denoted -1 in the returned vector."
        )
    if boundary_tol is None:
        insertion_ind = insertion_ind[~outside_range_bool]
    else:
        raise NotImplementedError()

    # Is the value closer to data at insertion_ind or insertion_ind-1?
    ind_diff = data_timestamps[insertion_ind] - event_timestamps[~outside_range_bool]
    ind_minus_one_diff = np.abs(
        data_timestamps[np.clip(insertion_ind - 1, 0, np.inf).astype(int)]
        - event_timestamps[~outside_range_bool]
    )
    event_indices = insertion_ind - (ind_diff > ind_minus_one_diff).astype(int)

    # Pad with nans in case some values were removed due to boundaries
    event_indices_full = -1 * np.ones(event_timestamps.shape, dtype="int")
    event_indices_full[~outside_range_bool] = event_indices

    return event_indices_full


def generate_perievent_slices(
    data_timestamps, event_timestamps, time_window=None, event_end_timestamps=None, sampling_rate=None, behavior_on_non_identical_timestamp_diffs='error'
):  # NOQA E501
    """
    Modified from https://github.com/AllenInstitute/mindscope_utilities/blob/e5aa1e6aebf3f62570aaff0e2e9dba835c999a23/mindscope_utilities/general_utilities.py#L112

    Given a relatively continuous time series, generate slices around events in that time series (eg time_window = (-1,5) for from 1 second before to 5 seconds after).
    See also: get_aligned_traces for a nice wrapper around this.
    
    Alternatively, pass event_end_timestamps instead of time_window if each event is a different length.

    NB, to make this play nicely with get_aligned_traces, we make each slice have the same length out, even if 
    it runs over the edge of the data. Eg if your event is 1 with a window of (-2,2), the slice will be slice(-1,3).
    Obviously what you really want here is np.hstack[np.nan, data[slice(0,3)] -- get_aligned_traces performs this internally.

    If you have a discrete list of times, and want to know which of those times fall within a window,
    e.g. stimulus presentation times within a trial, use get_times_within_event_windows() instead.

    Note: this generates end-exclusive slices! So if your time window is (-5,5), your data are (0,5,10,15,...), and your event time is 20, you will get out 15, 20, and NOT 25!
        This is to prevent unexpected weirdness with non-evenly-sampled data.
        Given high enough sampling rates, you won't even notice this.

    Parameters:
    -----------
    data_timestamps : np.array
        Timestamps of the datatrace.
    
    event_timestamps : np.array
        Timestamps of events around which to slice windows,  in same time base as data.
        If event_end_timestamps is passed, window begins at each event_timestamp.

    time_window : tuple (must pass either this or event_end_timestamps)
        [start_offset, end_offset] in same time base as data, if passed.
        Eg, [-2, 5] for a window from 2 sec before to 5 sec after.
    
    event_end_timestamps: np.array (must pass either this or time_window)
        End time for each event, if passed.

    sampling_rate : float, optional, default=None
        Sampling rate of the datatrace.
        If left as None, gets indices manually for all window starts/stops.
        If provided, just gets event indices and infers starts/stops by adding the correct number of samples.

    Yields:
    --------
    gen:
        A generator object with python slices for each peri-event window
        

    """

    data_timestamps, event_timestamps, time_window, event_end_timestamps = np_utils.castnp(
        data_timestamps, event_timestamps, time_window, event_end_timestamps
    )

    if (time_window is not None) and (event_end_timestamps is not None):
        raise ValueError('Cannot pass both time window and event end timestamps!')
    elif (sampling_rate is not None) and (event_end_timestamps is not None):
        raise ValueError('Cannot pass both sampling rate and event end timestamps!')

    if sampling_rate:
        # This way takes 1/2 the time because it only calls the indexing function once.
            
        # Check that data is in fact regularly sampled
        ts_diffs = np.diff(data_timestamps)
        if not np.all(np.isclose(ts_diffs, ts_diffs[0], atol=0, rtol=0.01)):
            if behavior_on_non_identical_timestamp_diffs == 'error':
                raise ValueError(
                    "Passing sampling_rate implies continuous timestamps with identical inter-sample intervals, but diffs are not all equal!"
                )
            elif behavior_on_non_identical_timestamp_diffs == 'warn':
                warnings.warn("Passing sampling_rate implies continuous timestamps with identical inter-sample intervals, but diffs are not all equal!")    
                warnings.warn("Proceed at your own risk, or dont pass sampling_rate!")    
        event_indices = index_of_nearest_value(data_timestamps, event_timestamps)
        start_ind_offset = int(time_window[0] * sampling_rate)
        end_ind_offset = int(time_window[1] * sampling_rate)
        for event in event_indices:
            yield slice(int(event + start_ind_offset), int(event + end_ind_offset))
    else:
        # ...but this way will work with data of varying sampling rate
        
        if time_window is not None:
            start_indices = index_of_nearest_value(
                data_timestamps, event_timestamps + time_window[0]
            )
            end_indices = index_of_nearest_value(
                data_timestamps, event_timestamps + time_window[1]
            )

        elif event_end_timestamps is not None:
            start_indices = index_of_nearest_value(
                data_timestamps, event_timestamps
            )
            end_indices = index_of_nearest_value(
                data_timestamps, event_end_timestamps
            )

        # Deal with any out of bounds issues in the ending timestamps (to avoid s.stop == -1)
        end_indices[end_indices==-1] = len(data_timestamps)

        for start, end in zip(start_indices, end_indices):
            yield slice(start, end)
