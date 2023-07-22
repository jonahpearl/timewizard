import numpy as np
from fastcluster import linkage
from scipy.spatial.distance import squareform


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


def seriation(Z, N, cur_index):
    """
    input:
        - Z is a hierarchical tree (dendrogram)
        - N is the number of points given to the clustering process
        - cur_index is the position in the tree for the recursive traversal
    output:
        - order implied by the hierarchical tree Z

    seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, method="ward"):
    """https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html

    input:
        - dist_mat is a distance matrix
        - method = ["ward","single","average","complete"]
    output:
        - seriated_dist is the input dist_mat,
          but with re-ordered rows and columns
          according to the seriation, i.e. the
          order implied by the hierarchical tree
        - res_order is the order implied by
          the hierarhical tree
        - res_linkage is the hierarhical tree (dendrogram)

    compute_serial_matrix transforms a distance matrix into
    a sorted distance matrix according to the order implied
    by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage
