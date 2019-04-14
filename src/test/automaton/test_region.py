from hypothesis import given, note
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np

from .custom_strategies import (
    dimensions_strategy,
    cell_coords_strategy,
    region_strategy,
)
from automaton.region import Region


@given(
    region=dimensions_strategy().flatmap(region_strategy)
)
def test_region(region):
    positions = list(region)
    # Test that len(region) is accurate.
    assert len(region) == len(positions)
    # Test that there are no duplicate coordinates.
    assert len(region) == len(set(map(tuple, positions)))
    # Test region.get_coordinates_list() against iter(region).
    coordinates_list = region.get_coordinates_list()
    assert coordinates_list.tolist() == np.array(list(region)).tolist()
    # Test region.get_coordinates_grid() against iter(region.remask()).
    coordinates_grid = region.get_coordinates_grid()
    flatter_grid = coordinates_grid.reshape((-1, region.dimensions))
    assert flatter_grid.tolist() == np.array(list(region.remask())).tolist()
    # Test position-in-region containment.
    for position in region:
        assert position in region
    # Test that not everything is within the region.
    if region.empty:
        assert (0,) * region.dimensions not in region
    else:
        # - Test below lower bound on first axis.
        positions[0][0] -= 1
        assert positions[0] not in region
        positions[0][0] += 1
        # - Test above upper bound on last axis.
        positions[-1][-1] += 1
        assert positions[-1] not in region
        positions[-1][-1] -= 1
    # Test equality and minifying.
    assert region == region.minify()
    assert (region != region.remask()) == region.has_mask
    # Test stringifying.
    assert str(region)
    assert repr(region)


@given(
    regions=dimensions_strategy(max_dim=4).flatmap(lambda d: st.tuples(
        region_strategy(d),
        region_strategy(d),
    ))
)
def test_region_operators(regions):
    r1, r2 = regions
    note('r1 = ' + str(r1))
    note('r2 = ' + str(r2))
    should_intersect = any(pos in r2 for pos in r1)
    assert should_intersect == r1.intersects(r2)
    intersection = r1 & r2
    difference = r1 ^ r2
    union = r1 | r2
    subtraction = r1 ^ r1 & r2
    note('and = ' + str(intersection))
    note('xor = ' + str(difference))
    note('or  = ' + str(union))
    note('sub = ' + str(subtraction))
    for pos in r1:
        assert pos in union
        if pos in r2:
            # Test cells in r1 and r2.
            assert pos in intersection
            assert pos not in difference
            assert pos not in subtraction
        else:
            # Test cells in r1 but not r2.
            assert pos not in intersection
            assert pos in difference
            assert pos in subtraction
    for pos in r2:
        if pos not in r1:
            # Test cells in r2 but not r1.
            assert pos not in intersection
            assert pos in difference
            assert pos in union
            assert pos not in subtraction
    for pos in intersection:
        assert pos in r1 and pos in r2
    for pos in union:
        assert pos in r1 or pos in r2
    for pos in difference:
        assert (pos in r1) != (pos in r2)
    for pos in subtraction:
        assert pos in r1 and pos not in r2
    # Test region-in-region containment.
    assert intersection in r1 and intersection in r2
    assert r1 in union and r2 in union
    assert subtraction in r1
    # Test other miscellaneous properties.
    assert (r1 == intersection) == (r1 in r2)
    assert (r2 == intersection) == (r2 in r1)
    assert (r1 == difference) == r2.empty
    assert (r2 == difference) == r1.empty
    assert (r1 == union) == (r2 in r1)
    assert (r2 == union) == (r1 in r2)
    assert (r1 == subtraction) == intersection.empty


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        region_strategy(d),
        cell_coords_strategy(d),
    ))
)
def test_region_offset(dimensioned_args):
    region, offset = dimensioned_args
    added = region + offset
    subtracted = region - offset
    # Test invariant properties, such as number of cells and emptiness.
    assert len(added) == len(subtracted) == len(region)
    assert added.empty == subtracted.empty == region.empty
    if offset.any() and not region.empty:
        assert added != subtracted != region
    else:
        assert added == subtracted == region
    for pos in region:
        assert pos + offset in added
        assert pos - offset in subtracted


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
    # Test "negation."
    assert region.invert() == -region
    # Test that the inverse of the inverse is the original.
    assert region.invert().invert() == region
    assert region.invert(axes).invert(axes) == region
    if not axes:
        assert region.invert(axes) == region.invert()


@given(
    unminified_region=dimensions_strategy().flatmap(lambda d:
        region_strategy(d, allow_empty=False, minify=st.just(False)),
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
    slices = minified_region.slices(unminified_region)
    mask_copy = minified_region.get_mask().copy()
    # Cut out the part that's in the minified Region ...
    mask_copy[slices] = False
    # ... and check that the rest is false.
    assert not mask_copy.any()


@given(
    original=dimensions_strategy().flatmap(region_strategy)
)
def test_empty_region(original):
    # Test minifying to an empty region.
    new_mask = original.get_mask().copy()
    new_mask.fill(False)
    region = original.remask(new_mask)
    assert region.empty
    # Test position-in-region containment.
    for pos in original:
        assert pos not in region
    # Test region-in-region containment.
    assert region in original
    assert (original in region) == original.empty
    # Test iteration.
    assert len(list(region)) == 0
    # Test length.
    assert len(region) == 0
    # Test stringifying.
    assert str(region)
    assert repr(region)
    # Test dimension count and construction from dimension count.
    region2 = region.empty_copy()
    region3 = Region(region.dimensions)
    region4 = region3.empty_copy()
    # Test equality.
    assert region == region2 == region3 == region4
    assert (region == original) == original.empty
