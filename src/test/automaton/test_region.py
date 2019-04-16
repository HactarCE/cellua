from hypothesis import assume, given, note
import hypothesis.strategies as st
import hypothesis.extra.numpy as np_st
import numpy as np

from .custom_strategies import (
    dimensions_strategy,
    cell_coords_strategy,
    region_mask_strategy,
    region_strategy,
)
from automaton.region import Region, RectRegion, MaskedRegion


@given(
    region=dimensions_strategy().flatmap(region_strategy)
)
def test_region(region):
    note('r = ' + str(region))
    positions = list(region)
    # Test that len(region) is accurate.
    assert len(region) == len(positions)
    # Test that there are no duplicate coordinates.
    assert len(region) == len(set(map(tuple, positions)))
    # Test region.positions against iter(region).
    assert region.positions.tolist() == np.array(list(region)).tolist()
    # Test region.position_grid against iter(region.box).
    flatter_grid = region.position_grid.reshape((-1, region.dimensions))
    assert flatter_grid.tolist() == np.array(list(region.box)).tolist()
    # Test position-in-region containment.
    for position in region:
        assert position in region
    # Test that not everything is within the region.
    if region.is_empty:
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
    # Test equality.
    assert (region != region.box) == isinstance(region, MaskedRegion)
    # Test stringifying.
    assert str(region)
    assert repr(region)


@given(
    regions=dimensions_strategy(max_dim=4).flatmap(lambda d: st.tuples(
        region_strategy(d, max_extent=5),
        region_strategy(d, max_extent=5),
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
    assert (r1 == difference) == r2.is_empty
    assert (r2 == difference) == r1.is_empty
    assert (r1 == union) == (r2 in r1)
    assert (r2 == union) == (r1 in r2)
    assert (r1 == subtraction) == intersection.is_empty


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
    note('r = ' + str(region))
    note('r+ = ' + str(added))
    note('r- = ' + str(subtracted))
    # Test invariant properties, such as number of cells and emptiness.
    assert len(added) == len(subtracted) == len(region)
    assert added.is_empty == subtracted.is_empty == region.is_empty
    if offset.any() and not region.is_empty:
        assert added != subtracted != region
    else:
        assert added == subtracted == region
    for pos in region:
        assert pos + offset in added
        assert pos - offset in subtracted


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        cell_coords_strategy(d),
        region_strategy(d).flatmap(region_mask_strategy),
    ))
)
def test_region_from_cell(dimensioned_args):
    pos, mask = dimensioned_args
    note('pos = ' + str(pos))
    note('mask = ' + str(mask))
    r1 = Region.span(pos)
    note('r1 = ' + str(r1))
    assert len(r1) == 1
    assert pos in r1
    r2 = Region.span(pos, mask)
    note('r2 = ' + str(r2))
    assert len(r2) == np.count_nonzero(mask)
    if isinstance(r1, RectRegion):
        assert (r1.bounds == pos).all()
    if isinstance(r2, MaskedRegion):
        assert np.all(r2.lower_bounds >= pos)
        assert Region.span(r2.lower_bounds, r2.mask) == r2


@given(
    dimensioned_args=dimensions_strategy().flatmap(lambda d: st.tuples(
        region_strategy(d),
        (st.sets(st.integers(min_value=0, max_value=d - 1))
            .map(tuple)
            .map(sorted)),
    ))
)
def test_region_invert(dimensioned_args):
    region, axes = dimensioned_args
    note('axes = ' + str(axes))
    note('original = ' + str(region))
    inverted = region.invert(axes)
    note('inverted = ' + str(inverted))
    # Test length.
    assert len(region) == len(inverted)
    # Test negated cell positions.
    for pos in region:
        new_pos = pos.copy()
        for axis in axes or range(region.dimensions):
            new_pos[axis] *= -1
        assert new_pos in inverted
    # Test that the inverse of the inverse is the original.
    assert inverted.invert(axes) == region
    if not axes:
        assert region.invert() == inverted


@given(
    region_args=dimensions_strategy().flatmap(lambda d:
        region_strategy(
            d,
            allow_empty=False,
            allow_nonrectangular=False,
            minify=st.just(False)
        ).flatmap(lambda r: st.tuples(
            st.just(r),
            region_mask_strategy(r, allow_empty=True),
        )),
    )
)
def test_region_minify(region_args):
    original, mask = region_args
    masked = Region.span(original, mask)
    # Make sure that the minified region is within the unminified region.
    assert masked in original
    if not masked.is_empty:
        for axis in range(masked.dimensions):
            rolled_minified = np.moveaxis(masked.mask, axis, 0)
            # Make sure that there is something on every edge.
            assert rolled_minified[0].any()
            assert rolled_minified[-1].any()
    # And make sure that nothing got cut off.
    assert len(masked) == np.count_nonzero(mask)


@given(
    original=dimensions_strategy().flatmap(lambda d: region_strategy(d, allow_empty=False))
)
def test_empty_region(original):
    # Test minifying to an empty region.
    new_mask = original.mask.copy()
    new_mask.fill(False)
    region = Region.span(original, new_mask)
    assert region.is_empty
    # Test position-in-region containment.
    for pos in original:
        assert pos not in region
    # Test region-in-region containment.
    assert region in original
    assert (original in region) == original.is_empty
    # Test iteration.
    assert len(list(region)) == 0
    # Test length.
    assert len(region) == 0
    # Test stringifying.
    assert str(region)
    assert repr(region)
    # Test dimension count and construction from dimension count.
    region2 = Region.empty(region)
    region3 = Region.empty(region.dimensions)
    region4 = Region.empty(region2)
    # Test equality.
    assert region == region2 == region3 == region4
    assert region != original
