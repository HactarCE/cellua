from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import math
import numpy as np

from utils.testing import dimensions_strategy

from ..grid import Grid


byte_strategy = lambda: st.integers(-128, 127)

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

grid_strategy = lambda dimen: st.builds(Grid, st.just(dimen))


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        grid_strategy(d),
        cell_coords_strategy(d),
    )),
    value=byte_strategy(),
)
def test_grid_set_get(dimensioned_args, value):
    grid, coords = dimensioned_args
    grid.set_cell(coords, value)
    assert grid.get_cell(coords) == value


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        grid_strategy(d),
        cell_coords_strategy(d),
        cell_offset_strategy(d),
    )),
    value1=byte_strategy(),
    value2=byte_strategy(),
)
def test_grid_set_get_nearby(dimensioned_args, value1, value2):
    grid, coords1, offset = dimensioned_args
    coords2 = coords1 + offset
    grid.set_cell(coords1, value1)
    grid.set_cell(coords2, value2)
    # If any(offset), then another cell was set
    # If not any(offset), then the same cell was set again
    assert grid.get_cell(coords1) == (value1 if offset.any() else value2)
    assert grid.get_cell(coords2) == value2


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        grid_strategy(d),
        cell_coords_strategy(d),
    )),
    value=byte_strategy(),
)
def test_grid_del_chunk_if_empty(dimensioned_args, value):
    grid, coords = dimensioned_args[:2]
    grid.set_cell(coords, value)
    chunk_coords, _ = grid.get_coords_pair(coords)
    assert(grid.has_chunk(chunk_coords))
    grid.del_chunk_if_empty(chunk_coords)
    assert(grid.has_chunk(chunk_coords) == (value != 0))
    grid.set_cell(coords, 0)
    grid.del_chunk_if_empty(chunk_coords)
    assert(not grid.has_chunk(chunk_coords))


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        grid_strategy(d),
        cell_coords_strategy(d),
        neighborhood_size_strategy(d),
        st.lists(st.tuples(cell_offset_strategy(d), byte_strategy())),
    )),
)
def test_napkin(dimensioned_args):
    grid, center_coords, neighborhood, neighbor_cells = dimensioned_args
    radius = np.absolute(neighborhood).max()
    square_napkin = np.zeros(shape=(radius * 2 + 1,) * grid.dimensions, dtype=np.byte)
    for offset, value in neighbor_cells:
        grid.set_cell(center_coords + offset, value)
        if all(abs(offset) <= radius):
            square_napkin[tuple(offset + radius)] = value
    napkin_slice = tuple(slice(lower, upper + 1) for lower, upper in neighborhood + radius)
    assert grid.get_cell_napkin(center_coords, neighborhood).tolist() == square_napkin[napkin_slice].tolist()
