import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np


byte_strategy = lambda: st.integers(-128, 127)

def dimensions_strategy(min_dim=1, max_dim=10):
    """A testing strategy for Hypothesis that generates reasonable cellular
    automaton dimension numbers.

    This is really just st.integers() with different defaults.
    """
    return st.integers(min_dim, max_dim)

def np_int64_arrays(shape, *args, **kwargs):
    """Returns a strategy for generating integer ndarrays with a given shape.

    All extra arguments are given to st.integers().
    """
    return np_st.arrays(
        np.int64,
        st.just(shape),
        st.integers(*args, **kwargs),
    )

# -50 to 50 is plenty of range for cell coordinates
def cell_coords_strategy(dimen, max_val=50):
    return np_int64_arrays(dimen, -max_val, max_val)

def cell_offset_strategy(dimen, max_val=4):
    return np_int64_arrays(dimen, -max_val, max_val)

def neighborhood_size_strategy(dimen, max_cell_count=1000000):
    # 1 million cells = ~10 Mb.
    max_radius = int(max_cell_count ** (1 / dimen) / 2)
    # `.map(np.sort)` ensures that each pair has the lower number first.
    return np_int64_arrays((dimen, 2), -max_radius, max_radius).map(np.sort)
