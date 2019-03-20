from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from .custom_strategies import (
    byte_strategy,
    cell_coords_strategy,
    cell_offset_strategy,
    dimensions_strategy,
    neighborhood_size_strategy,
)

from automaton.grid import Grid

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
