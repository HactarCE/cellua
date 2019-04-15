from hypothesis import assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np

from automaton.grid import Grid
from automaton.region import Region, EmptyRegion


def byte_strategy():
    return st.integers(-128, 127)


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
    - max_dim (default 10)
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


def region_mask_strategy(region, *, allow_empty):
    """Return a strategy for masks for a given Region.

    Arguments:
    - region

    Keyword arguments:
    - allow_empty
    """
    # This `tuple(map(int, ...))` nonsense is stupid and I don't know
    # why it's necessary. (maybe bug in Hypothesis?)
    region_shape = tuple(map(int, region.shape))
    def assume_nonempty(mask):
        if not allow_empty:
            assume(mask.any())
        return mask
    return np_st.arrays(bool, region_shape).map(assume_nonempty)


def region_strategy(dimen,
                    *,
                    allow_empty=True,
                    allow_nonrectangular=True,
                    max_cell_count=1000,
                    max_extent=50,
                    **kwargs):
    """Return a strategy for Regions of a given dimensionality.

    By default, the Region will extend at most 50 in every direction and will
    contain at most one thousand cells.

    Arguments:
    - dimen -- number of dimensions

    Optional keyword arguments:
    - allow_empty (default True)
    - allow_nonrectangular (default True)
    - max_cell_count (default 1000)
    - max_extent (default 50)

    Any other keyword arguments are treated as strategies passed to Region.span().
    """

    max_extent = min(max_extent, int(max_cell_count ** (1 / dimen) / 2))
    bounds_strategy = np_int64_arrays((2, dimen), -max_extent, max_extent)

    def masker(region):
        if allow_nonrectangular:
            mask_strat = region_mask_strategy(region, allow_empty=allow_empty)
            return st.builds(lambda mask: Region.span(region, mask), mask_strat)
        elif allow_empty:
            return st.just(Region.empty(dimen)) | st.just(region)
        else:
            return st.just(region)

    return st.builds(Region.span, bounds_strategy).flatmap(masker)


def neighborhood_strategy(dimen,
                          *,
                          allow_empty=False,
                          max_extent=3,
                          **kwargs):
    """Return a strategy for reasonable cellular automaton neighborhoods.

    This is really just region_strategy() with different defaults.
    """
    return region_strategy(dimen, allow_empty=allow_empty, max_extent=max_extent, **kwargs)
