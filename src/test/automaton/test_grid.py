from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from .custom_strategies import (
    byte_strategy,
    cell_coords_strategy,
    cell_offset_strategy,
    dimensions_strategy,
    grid_strategy,
    neighborhood_strategy,
)


def assert_grid_iter(grid):
    for chunk_coords, chunk in grid:
        assert chunk is grid.get_chunk(chunk_coords)


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
    assert_grid_iter(grid)


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        grid_strategy(d),
        cell_coords_strategy(d),
        cell_offset_strategy(d),
    )),
    value1=byte_strategy(),
    value2=byte_strategy(),
)
def test_grid_multi_set(dimensioned_args, value1, value2):
    grid, coords1, offset = dimensioned_args
    coords2 = coords1 + offset
    grid.set_cell(coords1, value1)
    grid.set_cell(coords2, value2)
    # If any(offset), then another cell was set
    # If not any(offset), then the same cell was set again
    assert grid.get_cell(coords1) == (value1 if offset.any() else value2)
    assert grid.get_cell(coords2) == value2
    assert_grid_iter(grid)


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
    value_is_zero = not value
    value_is_nonzero = not value_is_zero
    assert grid.has_chunk(chunk_coords)
    assert grid.is_chunk_empty(chunk_coords) == value_is_zero
    assert grid.is_empty()                   == value_is_zero
    grid.del_chunk_if_empty(chunk_coords)
    assert grid.has_chunk(chunk_coords)      == value_is_nonzero
    assert grid.is_chunk_empty(chunk_coords) == value_is_zero
    assert grid.is_empty()                   == value_is_zero
    grid.set_cell(coords, 0)
    grid.del_chunk_if_empty(chunk_coords)
    assert not grid.has_chunk(chunk_coords)


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        grid_strategy(d),
        neighborhood_strategy(d),
    ))
)
def test_inverse_chunk_neighborhood(dimensioned_args):
    grid, neighborhood = dimensioned_args
    chunk_neighborhood_of_inverse = grid.get_chunk_neighborhood(neighborhood.get_inverse())
    inverse_of_chunk_neighborhood = grid.get_chunk_neighborhood(neighborhood).get_inverse()
    assert chunk_neighborhood_of_inverse == inverse_of_chunk_neighborhood


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        grid_strategy(d),
        cell_coords_strategy(d),
        neighborhood_strategy(d),
        st.lists(st.tuples(cell_offset_strategy(d), byte_strategy())),
    )),
)
def test_napkin(dimensioned_args):
    grid, center_coords, neighborhood, neighbor_cells = dimensioned_args
    radius = neighborhood.max_radius
    square_napkin = np.zeros(shape=(radius * 2 + 1,) * grid.dimensions, dtype=np.byte)
    for offset, value in neighbor_cells:
        grid.set_cell(center_coords + offset, value)
        if all(abs(offset) <= radius):
            square_napkin[tuple(offset + radius)] = value
    napkin_slice = tuple(slice(lower, upper + 1) for lower, upper in neighborhood.extents + radius)
    assert grid.get_cell_napkin(center_coords, neighborhood).tolist() == square_napkin[napkin_slice].tolist()
