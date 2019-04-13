from hypothesis import assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np

from automaton.grid import Grid
from automaton.region import Region


byte_strategy = lambda: st.integers(-128, 127)

def np_int64_arrays(shape, *args, **kwargs):
    """Return a strategy for integer ndarrays with a given shape.

    All extra arguments are given to st.integers().
    """
    return np_st.arrays(np.int64, shape, st.integers(*args, **kwargs))

def dimensions_strategy(min_dim=1, max_dim=10):
    """Return a strategy for reasonable cellular automaton dimension
    numbers.

    This is really just st.integers() with different defaults.

    Optional arguments:
    - min_dim (default 1)
    - max_dimi (default 10)
    """
    return st.integers(min_dim, max_dim)

def cell_coords_strategy(dimen, max_val=50):
    """Return a strategy for cell coordinates of a given dimensionality.

    By default, the coordinate values are limited to -50..+50.

    Arguments:
    - dimen -- number of dimensions

    Optional arguments:
    - max_val (default 50)
    """
    return np_int64_arrays(dimen, -max_val, max_val)

def cell_offset_strategy(dimen, max_val=4):
    """Return a strategy for cell offsets of a given dimensionality.

    This is just `cell_coords_strategy` with a default limit of -4..+4.

    Arguments:
    - dimen -- number of dimensions

    Optional arguments:
    - max_val (default 4)
    """
    return np_int64_arrays(dimen, -max_val, max_val)

def grid_strategy(dimen):
    """Return a strategy for Grids of a given dimensionality."""
    return st.builds(Grid, st.just(dimen))

def region_strategy(dimen, max_extent=50, *, max_cell_count=1000, random_mask=True, **kwargs):
    """Return a strategy for Regions of a given dimensionality.

    By default, the Region will extend at most 50 in every direction and will
    contain at most one thousand cells.

    Arguments:
    - dimen -- number of dimensions

    Optional arguments:
    - max_extent (default 50)

    Optional keyword arguments:
    - max_cell_count (default 1000)
    - random_mask (default True) -- whether to use a not-all-True mask

    Any other keyword arguments are treated as strategies passed to Region().
    """
    max_extent = min(max_extent, int(max_cell_count ** (1 / dimen) / 2))
    bounds_strategy = np_int64_arrays((2, dimen), -max_extent, max_extent)
    def add_mask(region):
        if random_mask:
            def assume_some(mask):
                assume(mask.any())
                return mask
            # This `tuple(map(int, ...))` nonsense is stupid and I don't know
            # why it's necessary. (maybe bug in Hypothesis?)
            region_shape = tuple(map(int, region.shape))
            mask_strat = np_st.arrays(bool, region_shape).map(assume_some)
            return st.builds(region.remask, mask_strat)
        else:
            return st.just(region)
    return st.builds(Region, bounds_strategy).flatmap(add_mask)
