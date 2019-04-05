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
    return np_st.arrays(
        np.int64,
        st.just(shape),
        st.integers(*args, **kwargs),
    )

def dimensions_strategy(min_dim=1, max_dim=10):
    """Return a strategy for reasonable cellular automaton dimension
    numbers.

    This is really just st.integers() with different defaults.
    """
    return st.integers(min_dim, max_dim)

def cell_coords_strategy(dimen, max_val=50):
    """Return a strategy for cell coordinates of a given dimensionality.

    By default, the coordinate values are limited to -50..+50.
    """
    return np_int64_arrays(dimen, -max_val, max_val)

def cell_offset_strategy(dimen, max_val=4):
    """Return a strategy for cell offsets of a given dimensionality.

    This is just `cell_coords_strategy` with a default limit of -4..+4.
    """
    return np_int64_arrays(dimen, -max_val, max_val)

def grid_strategy(dimen):
    """Return a strategy for Grids of a given dimensionality."""
    return st.builds(Grid, st.just(dimen))

def neighborhood_strategy(dimen, max_cell_count=10000):
    """Return a strategy for neighborhoods of a given dimensionality.

    By default, the maximum radius of the neighborhood is set so that no
    Region will contain more than ten thousand cells.
    """
    max_radius = int(max_cell_count ** (1 / dimen) / 2)
    bounds_strategy = np_int64_arrays((2, dimen), -max_radius, max_radius)
    return st.builds(Region, bounds_strategy)
