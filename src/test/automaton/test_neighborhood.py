from hypothesis import given
import hypothesis.strategies as st
import numpy as np

from .custom_strategies import (
    dimensions_strategy,
    neighborhood_strategy,
)


@given(
    neighborhood=dimensions_strategy().flatmap(neighborhood_strategy)
)
def test_neighborhood(neighborhood):
    offsets = list(neighborhood)
    # Test that len(neighborhood) is accurate.
    assert len(neighborhood) == len(offsets)
    # Test that there are no duplicate coordinates.
    assert len(neighborhood) == len(set(map(tuple, offsets)))
    # Test neighborhood.get_neighbor_offset_grid() against iter(neighborhood).
    offset_grid = neighborhood.get_offset_grid()
    flatter_offsets = offset_grid.reshape((-1, neighborhood.dimensions))
    assert (flatter_offsets == list(neighborhood)).all()
    # Test that all coordinates are within the neighborhood.
    for offset in neighborhood:
        assert offset in neighborhood
    # Test that not everything is within the neighborhood.
    offsets[0][0] -= 1  # Test below lower bound on first axis.
    assert offsets[0] not in neighborhood
    if len(offsets) > 1:
        offsets[-1][-1] += 1  # Test above upper bound on last axis.
        assert offsets[-1] not in neighborhood
