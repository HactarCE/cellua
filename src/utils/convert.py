import numpy as np


def to_dimen(dimensions):
    """Convert `dimensions` to an integer >= 1, or raise a ValueError if it
    cannot be converted.
    """
    try:
        d = int(dimensions)
        assert d >= 1
        return d
    except Exception:
        pass
    raise ValueError(f"Argument {dimensions} is not convertible to a dimension count.")


def to_coords(coords, dimensions=None):
    """Convert `coords` to a 1D integer ndarray of length `dimensions`, or raise
    a ValueError if it cannot be converted.

    If `dimensions` is None (the default), then any number of dimensions is
    allowed.
    """
    try:
        coords = np.array(coords, dtype=np.int64)
        assert len(coords.shape) == 1
        if dimensions is None:
            assert coords.size >= 1
        else:
            assert coords.size == dimensions
        return coords
    except Exception:
        pass
    msg = f"Argument {coords} is not convertible to coordinate ndarray"
    if dimensions is not None:
        msg += f" of shape ({dimensions},)"
    raise ValueError(msg)
