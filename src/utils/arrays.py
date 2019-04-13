import numpy as np


def nd_cartesian_grid(*arrays, repeat=1):
    """Generate a Cartesian product of ndarrays, arranged in a grid.

    The element at index `(i0, i1, ..., iN)` in the return value is an array of
    `[a0[i0], a1[i1], ... aN[iN]]`, where aN is the Nth ndarray of `arrays`.
    """
    return np.stack(np.meshgrid(*arrays * repeat, indexing='ij'), -1)


def nd_cartesian(*arrays, repeat=1):
    """Generate a Cartesian product of ndarrays.

    The order of the returned values is the same as `itertools.product`.
    """
    return nd_cartesian_grid(*arrays, repeat=repeat).reshape(-1, len(arrays))


def convert_to_coords(coords, d=None):
    """Convert `coords` to a 1D integer ndarray of length `d`, or raise a
    ValueError if it cannot be converted.
    """
    try:
        coords = np.array(coords, dtype=np.int64)
        if d is not None:
            assert coords.shape == (d,)
        return coords
    except Exception:
        pass
    msg = f"Argument {coords} is not convertible to coordinate ndarray"
    if d is not None:
        msg += f" of shape ({d},)"
    raise ValueError(msg)
