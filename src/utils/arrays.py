import numpy as np

def nd_cartesian(*arrays, repeat=1):
    """Generate a Cartesian product of ndarrays.

    The order of the returned values is the same as `itertools.product`.
    """
    return np.stack(np.meshgrid(*arrays * repeat, indexing='ij'), -1).reshape(-1, len(arrays))
