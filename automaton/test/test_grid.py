from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np

from utils.testing import dimensions_strategy

from ..grid import Grid

byte_strategy = lambda: st.integers(-128, 127)
# -50 to 50 is plenty of range for cell coordinates
cell_coords_strategy = lambda dimen, max_val=50: np_st.arrays(np.byte, st.just(dimen), st.integers(-max_val, max_val))
cell_offset_strategy = lambda dimen, max_val=4: cell_coords_strategy(dimen, max_val)

@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        st.builds(Grid, st.just(d)),
        cell_coords_strategy(d),
    )),
    value=byte_strategy(),
)
def test_grid_set_get(dimensioned_args, value):
    grid, coord = dimensioned_args
    grid.set_cell(coord, value)
    assert grid.get_cell(coord) == value

@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        st.builds(Grid, st.just(d)),
        cell_coords_strategy(d),
        cell_offset_strategy(d),
    )),
    value1=byte_strategy(),
    value2=byte_strategy(),
)
def test_grid_set_get_nearby(dimensioned_args, value1, value2):
    grid, coord1, offset = dimensioned_args
    coord2 = coord1 + offset
    grid.set_cell(coord1, value1)
    grid.set_cell(coord2, value2)
    # If any(offset), then another cell was set
    # If not any(offset), then the same cell was set again
    assert grid.get_cell(coord1) == (value1 if offset.any() else value2)
    assert grid.get_cell(coord2) == value2
