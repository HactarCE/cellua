from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np

from .custom_strategies import (
    dimensions_strategy,
    region_strategy,
)


@given(
    region=dimensions_strategy().flatmap(region_strategy)
)
def test_region(region):
    positions = list(region)
    # Test that len(region) is accurate.
    assert len(region) == len(positions)
    # Test that there are no duplicate coordinates.
    assert len(region) == len(set(map(tuple, positions)))
    # Test region.get_coordinates_grid() against iter(region.remask()).
    coordinates_grid = region.get_coordinates_grid()
    flatter_positions = coordinates_grid.reshape((-1, region.dimensions))
    assert flatter_positions.tolist() == np.array(list(region.remask())).tolist()
    # Test that all coordinates are within the region.
    for position in region:
        assert position in region
    # Test that not everything is within the region.
    positions[0][0] -= 1  # Test below lower bound on first axis.
    assert positions[0] not in region
    if len(positions) > 1:  # TODO try remove the conditional
        positions[-1][-1] += 1  # Test above upper bound on last axis.
        assert positions[-1] not in region

@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        region_strategy(d),
        (np_st.arrays(np.bool, st.just(d), st.booleans())
            # Yay functional programming!
            .map(np.nonzero)
            .map(lambda a: a[0])
            .map(tuple)),
    ))
)
def test_region_invert(dimensioned_args):
    region, axes = dimensioned_args
    # Test that the inverse of the inverse is the original
    assert region.invert().invert() == region
    assert region.invert(axes).invert(axes) == region
    if not axes:
        assert region.invert(axes) == region.invert()


@given(
    unminified_region=dimensions_strategy().flatmap(lambda d:
        region_strategy(d, minify=st.just(False)),
    )
)
def test_region_minify(unminified_region):
    minified_region = unminified_region.minify()
    # Make sure that the minified region is within the unminified region.
    assert minified_region in unminified_region
    for axis in range(minified_region.dimensions):
        rolled_minified = np.moveaxis(minified_region.get_mask(), axis, 0)
        # Make sure that there is something on every edge.
        assert rolled_minified[0].any()
        assert rolled_minified[-1].any()
    # And make sure that nothing got cut off.
    slices = minified_region.intersecting_slices(unminified_region)
    mask_copy = minified_region.get_mask().copy()
    # Cut out the part that's in the minified Region ...
    mask_copy[slices] = False
    # ... and check that the rest is false.
    assert not mask_copy.any()
