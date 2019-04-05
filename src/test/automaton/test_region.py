from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np

from .custom_strategies import (
    dimensions_strategy,
    neighborhood_strategy,
)


@given(
    region=dimensions_strategy().flatmap(neighborhood_strategy)
)
def test_region(region):
    offsets = list(region)
    # Test that len(region) is accurate.
    assert len(region) == len(offsets)
    # Test that there are no duplicate coordinates.
    assert len(region) == len(set(map(tuple, offsets)))
    # Test region.get_neighbor_offset_grid() against iter(region).
    offset_grid = region.get_offset_grid()
    flatter_offsets = offset_grid.reshape((-1, region.dimensions))
    assert (flatter_offsets == list(region)).all()
    # Test that all coordinates are within the region.
    for offset in region:
        assert offset in region
    # Test that not everything is within the region.
    offsets[0][0] -= 1  # Test below lower bound on first axis.
    assert offsets[0] not in region
    if len(offsets) > 1:
        offsets[-1][-1] += 1  # Test above upper bound on last axis.
        assert offsets[-1] not in region

@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        neighborhood_strategy(d),
        np_st.arrays(np.bool, st.just(d), st.booleans()).map(np.nonzero),
    ))
)
def test_region_invert(dimensioned_args):
    region, axes = dimensioned_args
    # Test that the inverse of the inverse is the original
    assert region.invert().invert() == region
    assert region.invert(axes).invert(axes) == region
    if not axes:
        assert region.invert(axes) == region.invert()
