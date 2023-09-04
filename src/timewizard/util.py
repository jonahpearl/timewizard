
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from .allenbrainobs.obs_utils import index_of_nearest_value


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
    old_times, old_data = castnp(old_times, old_data)

    f = interp1d(old_times, old_data, kind=interp_kind, **kwargs)
    new_times = np.arange(np.min(old_times), np.max(old_times), 1 / fs)
    if (
        new_times[-1] > old_times[-1]
    ):  # occasionally happens due to rounding; will throw error with defaults; remove for simplicity.
        new_times = np.delete(new_times, -1)
    new_data = f(new_times)
    return new_times, new_data


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
    idxs_of_1_nearest_to_each_2 = index_of_nearest_value(t1, t2, oob_behavior='warn')  # in eg above, (0, 3, 3, 3, ...)
    idxs_of_2_nearest_to_each_1 = index_of_nearest_value(t2, t1, oob_behavior='warn')  # in eg above, (0, 0, 0, 1, ...)

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


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def round_to_multiple(number, multiple, decimals=2):
    return multiple * np.round(number / multiple, decimals)


def castnp(*args):
    """Cast any number of args into numpy arrays

    Returns:
        tuple -- tuple of input args, each cast to np.array(arg, ndmin=1).
        Special case: if only 1 arg, returns the array directly (not in a tuple).
    """

    out = []
    for arg in args:

        # Catch None args
        if arg is None:
            out.append(None)
            continue

        # Convert to np if needed
        type_ = type(arg)
        if (type_ is not np.ndarray) or (type_ is np.ndarray and arg.ndim == 0):
            out.append(np.array(arg, ndmin=1))
        else:
            out.append(arg)

    # Transform for output
    if len(out) == 1:  # special case, 1 item only: return array directly, without nested tuple.
        out = out[0]
    elif len(out) == 0:
        out = ()
    else:
        out = tuple(out)

    return out


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


def get_dict_map_np(my_dict):
    """Vectorize getting from a dictionary, for convenience

    Arguments:
        my_dict {dict} -- the dictionary to vectorize

    Returns:
        {np.vectorize} -- a vectorized dictionary getter

    Example:
    x = np.arange(100).reshape((10,10))
    x2 = x**2
    d = {i:i2 for i,i2 in zip(x.ravel(), x2.ravel())}
    d_vec = get_dict_map_np(d)
    d_vec(x[(x%2)==0]) # equivalent: np.array([d[i] for i in x.ravel() if i%2==0])
    
    seq = np.random.choice(np.arange(100), size=(1000,), replace=True)
    squared = np.array([d[x] for x in seq])
    assert np.allclose(sq, d_vec(seq))
    """
    return np.vectorize(my_dict.get)


def xticks_from_timestamps(timestamps, window=None, ax=None, calculate_only=False, interval=1):
    """Generate xticks for a heatmap, from the corresponding timestamps (ie cols)

    Arguments:
        timestamps {np.array} -- timestamps for the columns (x-axis) of the heatmap

    Keyword Arguments:
        window {tuple} -- optional window to use for timestamps (default: {None})
        ax {plt axis} -- option ax object, otherwise uses plt.xticks (default: {None})
        calculate_only {bool} -- if True, returns values without setting (default: {False})
        interval {int} -- desired interval in units of timestamps for tick spacing (default: {1})

    Returns:
        [type] -- [description]

    Example:
        data = np.random.normal(0, 1, size=(10,41))
        ts = np.arange(-2,2.01,0.1)
        plt.imshow(data)
        moseq_fo.viz.xticks_from_timestamps(ts, interval=0.5)

    """
    if window:
        timestamps = timestamps[window[0] : window[1]]
    range_ = timestamps[-1] - timestamps[0]  # num seconds
    by = np.floor(timestamps.shape[0] / range_ * interval).astype("int")  # num (interval)-ths of seconds
    offset = int(index_of_nearest_value(timestamps, 0) % by)
    locs = np.arange(offset, timestamps.shape[0], by)
    labs = round_to_multiple(timestamps[offset::by], interval)

    if calculate_only:
        return locs, labs

    if ax:
        ax.set_xticks(locs)
        ax.set_xticklabels(labs)
    else:
        plt.xticks(ticks=locs, labels=labs)

    return locs, labs
