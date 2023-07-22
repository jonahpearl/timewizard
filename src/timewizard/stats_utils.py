import numpy as np

def true_val_and_bootstrap_std(values, func, nboot=100):
    """Bootstrap the standard deviation of a function applied to a set of values

    Arguments:
        values {array} -- the data to bootstrap
        func {function object} -- the function to apply to the data

    Keyword Arguments:
        nboot {int} -- number of bootstrap iterations (default: {100})

    Returns:
        tuple -- true_value, bootstrap_std
    """
    true_val = func(values)
    boot_vals = np.zeros((nboot,))
    for i in range(nboot):
        boot_samples = np.random.choice(values, size=values.shape[0], replace=True)
        boot_vals[i] = func(boot_samples)
    return true_val, np.std(boot_vals)

