import numpy as np

import utils


class Region:
    """An immutable object describing an integer hyperrectangle.

    Public read-only properties:
    - empty -- bool; whether the region is empty; if this is True, then no other
      public properties besides `dimensions`, `size`, `shape`, `has_mask`, and
      `mask` are defined.
    - dimensions -- integer number of dimensions
    - bounds -- integer ndarray of shape (2, d); each row is a set of
      coordinate offsets from the center cell; for example [[-3, 0, -1], [2, 0,
      1] describes a 6x1x3 3D region
    - lower_bounds -- 1D integer ndarray of size d representing the lower
      bounds of the region along each axis; equivalent to `bounds[0]`
    - upper_bounds -- 1D integer ndarray of size d representing the upper
      bounds of the region along each axis; equivalent to `bounds[1]`
    - max_radius -- integer absolute maximum value of `bounds`
    - mask -- None or bool ndarray with same shape as region
    - has_mask -- bool; whether `mask is not None`
    - size -- integer; number of cells in region
    - shape -- integer tuple; Numpy-style shape of region

    A Region object can be used as an iterator to get the relative coordinates
    (offsets) of each cell:

    ```py
    >>> for offset in Region([[-2, 0], [0, 1]]):
    ...     print(offset)
    [-2 0]
    [-2 1]
    [-1 0]
    [-1 1]
    [0 0]
    [0 1]
    ```

    A region can be used with the `in` keyword to test whether it contains some
    position or fully contains some other region.

    `len()` returns the number of cells in a region (same as `region.size`).

    Boolean operators can be used to find the intersection (`r1 & r2`), union
    (`r1 | r2`), and difference (`r1 ^ r2`) between two regions. To "subtract"
    one region from another, use `r1 & r2 ^ r1`. To check for intersection,
    prefer `Region.intersects()` over `&`.

    Addition, subtraction, and negation all operate on each coordinate
    separately. A region can be added or subtracted with a coordinate tuple;
    e.g. `region + (2, -1)` will offset a region by +2 along X and -1 along Y.
    """

    def __init__(self, bounds, mask=None, *, minify=True):
        """Create a Region object from bounds and an optional mask.

        Optional arguments:
        - bounds -- ndarray of shape (2, d); a 2D array [corner1, corner2]
          (inclusive) defining the bounds of the region.
        - mask -- None or bool ndarray with same shape as region (defaults to
          None)

        Alternatively, `bounds` may be an integer number of dimensions, in which
        case an empty region will be created.

        Optional keyword arguments:
        - minify -- bool (defaults to True); if True and a mask is present, the
          region will be made as small as possible while still including all the
          positions included in the mask.

        If `mask` is all True, then it will be replaced with None.

        If there are smaller bounds containing every True value of `mask`, then
        the bounds will be shrunk accordingly.
        """
        bounds = np.array(bounds)  # Copy `bounds` and ensure it's an ndarray.
        if bounds.shape and (bounds.ndim != 2 or bounds.shape[0] != 2):
            return ValueError(f"Invalid region array shape: {bounds}")
        if not bounds.shape or (mask is not None and not mask.any()):
            if bounds.shape:
                self.dimensions = bounds.shape[1]
            else:
                self.dimensions = int(bounds)
            self.empty = True
            self.shape = (0,) * self.dimensions
            self.size = 0
            self.has_mask = False
            self.mask = None
            return
        bounds.sort(0)  # Ensure that `lower_bounds < upper_bounds`.
        self.dimensions = bounds.shape[1]
        if isinstance(mask, np.ndarray):
            tmp_shape = tuple(bounds[1] - bounds[0] + 1)
            if mask.shape != tmp_shape:
                raise ValueError(f"Mask shape {mask.shape} does not match region shape {tmp_shape}")
            if minify:
                # For each axis, try to make the region as small as possible
                # while still including all the True values in the mask.
                for axis in range(self.dimensions):
                    lower, upper = 0, -1
                    # Move the relevant axis to the zeroth position for
                    # convenience.
                    mask = np.moveaxis(mask, axis, 0)
                    # Increase the lower bound
                    while not mask[lower].any():
                        lower += 1
                    # Decrease the upper bound
                    while not mask[upper].any():
                        upper -= 1
                    bounds[:, axis] += lower, upper + 1
                    mask = mask[lower:] if upper == -1 else mask[lower:upper + 1]
                    # Move the axis back to where it belongs.
                    mask = np.moveaxis(mask, 0, axis)
                # If the mask is smaller, make a copy to save memory and make it
                # contiguous.
                if mask.shape != tmp_shape:
                    mask = mask.copy()
            # If the mask is all True, don't bother storing it.
            if mask.all():
                mask = None
        elif mask is not None:
            raise TypeError(f"Mask must be an ndarray or None; {type(mask)} found instead")
        self.empty = False
        self.bounds = bounds
        self.lower_bounds, self.upper_bounds = self.bounds
        shape_ndarray = self.upper_bounds - self.lower_bounds + 1
        self.mask = mask
        self.has_mask = mask is not None
        self.shape = tuple(shape_ndarray)
        self.size = np.count_nonzero(mask) if self.has_mask else shape_ndarray.prod()
        self.max_radius = np.absolute(self.bounds).max()

    def __contains__(self, other):
        """Return whether some offset or region is within `self`."""
        if isinstance(other, Region):
            if other.empty:
                return True
            elif self.empty:
                return False
            # Check whether the bounding rectangles fit.
            if not ((other.upper_bounds <= self.upper_bounds).all() and
                    (self.lower_bounds <= other.lower_bounds).all()):
                return False
            # If the bounding rectangles fit and this region has no mask, then
            # the other region will definitely fit inside.
            if not self.has_mask:
                return True
            # Subtract `self` from `other` and check whether any cells remain.
            return (self & other ^ other).empty
        else:
            if self.empty:
                return False
            coords = utils.arrays.convert_to_coords(other, self.dimensions)
            above_lower = (self.lower_bounds <= coords).all()
            below_upper = (coords <= self.upper_bounds).all()
            if not (above_lower and below_upper):
                return False
            return (not self.has_mask) or self.mask[tuple(coords - self.lower_bounds)]

    def __iter__(self):
        """Iterate over cell coordinate offsets.

        Each element of the iterator is a 1D ndarray of coordinates.
        """
        if self.empty:
            return iter(())
        positions = np.ndindex(*self.shape)
        positions = (np.array(i) + self.lower_bounds for i in positions)
        if self.has_mask:
            positions = filter(self.__contains__, positions)
        return positions

    def __len__(self):
        """Return the number of cells in the region."""
        return self.size

    def __repr__(self):
        s = f'{self.__class__.__name__}('
        if self.empty:
            s += f'{self.dimensions!r}'
        else:
            s += f'{self.bounds!r}'
            if self.has_mask:
                s += f', {self.mask!r}'
        s += ')'
        return s

    def __str__(self):
        if self.empty:
            s = f'empty {self.dimensions}D region'
        else:
            s = 'x'.join(map(str, self.shape))
            if self.dimensions == 1:
                s += '-length'
            s += f' region from {tuple(self.lower_bounds)} to {tuple(self.upper_bounds)}'
            if self.has_mask:
                s += f' with {np.count_nonzero(self.mask)}-element mask'
        return f'[{s}]'

    def __eq__(self, other):
        if not isinstance(other, Region):
            return False
        if self.empty or other.empty:
            return self.empty == other.empty
        return (np.all(self.bounds == other.bounds)
                and np.all(self.mask == other.mask))

    def __and__(self, other):
        """Return the intersection of two regions."""
        return self._boolean_op('&', other)

    def __xor__(self, other):
        """Return the difference between two regions."""
        return self._boolean_op('^', other)

    def __or__(self, other):
        """Return the union of two regions."""
        return self._boolean_op('|', other)

    def __add__(self, other):
        """Offset a region by a coordinate tuple."""
        return self._offset(other, +1)

    def __sub__(self, other):
        """Offset a region by a negated coordinate tuple."""
        return self._offset(other, -1)

    def __neg__(self):
        """Invert a region along all axes. (See `Region.invert()`.)"""
        return self.invert()

    def _boolean_op(self, op, other):
        """Internal function used for boolean operator magic methods."""
        if isinstance(other, Region):
            return self.op(op, other)
        else:
            return NotImplemented

    def _offset(self, offset, multiplier):
        """Internal function used for addition and subtraction magic methods."""
        try:
            offset = np.array(offset, dtype=np.int64)
        except ValueError:
            return NotImplemented
        offset *= multiplier
        if not offset.shape == (self.dimensions,):
            if len(offset.shape) > 1:
                return ValueError(f"Offset {offset} has too many dimensions")
            else:
                return ValueError(f"Dimension mismatch between {self} and {offset}")
        if self.empty:
            return self
        return Region(self.bounds + offset, self.mask)

    def op(self, op, other):
        """Apply a boolean operator between the cells sets of two regions.

        Arguments:
        - self -- Region
        - op -- string; one of ('&', '|', '^', '&~')
        - other -- Region
        """
        if self.dimensions != other.dimensions:
            raise ValueError(f"Dimension mismatch between {self} and {other}")
        if not (self.empty or other.empty):
            if op == '&':
                # Optimization: When computing intersection beytween two regions
                # with non-intersecting bounding boxes, the result is empty.
                # (Remove masks because `Region.intersects()` depends on this
                # function for handling masked regions.)
                if not self.remask().intersects(other.remask()):
                    return self.empty_copy()
                # Optimization: When computing intersection, only include the
                # intersection between the regions (obviously).
                lower_bounds = np.maximum(self.lower_bounds, other.lower_bounds)
                upper_bounds = np.minimum(self.upper_bounds, other.upper_bounds)
                # Optimization: When computing intersection between two regions
                # without masks, the final region does not need a mask.
                if not (self.has_mask or other.has_mask):
                    return Region([lower_bounds, upper_bounds])
            else:
                # For anything besides intersectioni, get the lower and upper
                # bounds of the union of both regions. (Compute the mask later.)
                lower_bounds = np.minimum(self.lower_bounds, other.lower_bounds)
                upper_bounds = np.maximum(self.upper_bounds, other.upper_bounds)
            r = Region([lower_bounds, upper_bounds])
            new_mask = np.zeros(r.shape, dtype=bool)
            new_mask[r.slices(self)] = self.mask[self.slices(r)] if self.has_mask else True
            sliced_other_mask = other.mask[other.slices(r)] if other.has_mask else True
        if op == '&':
            if self.empty:
                return self
            if other.empty:
                return other
            new_mask[r.slices(other)] &= sliced_other_mask
        elif op == '|':
            if self.empty:
                return other
            if other.empty:
                return self
            new_mask[r.slices(other)] |= sliced_other_mask
        elif op == '^':
            if self.empty:
                return other
            if other.empty:
                return self
            new_mask[r.slices(other)] ^= sliced_other_mask
        elif op == '&~':
            if self.empty or other.empty:
                return self
            new_mask[r.slices(other)] &= ~sliced_other_mask
        return r.remask(new_mask)

    def intersects(self, other):
        """Return whether `self` and `other` intersect at all."""
        if self.empty or other.empty:
            return False
        # Check whether the bounding rectangles intersect.
        if not ((self.lower_bounds <= other.upper_bounds) &
                (other.lower_bounds <= self.upper_bounds)).all():
            return False
        # If the bounding rectangles overlap and neither region has a mask, then
        # they must intersect.
        if not (self.has_mask or other.has_mask):
            return True
        return not (self & other).empty

    def slices(self, other):
        """Return an iterator of the slices that can be used to get the part of
        `self`'s mask which is inside of the bounding rectangle of `other`.

        Raises ValueError if `self` and `other` do not intersect.
        """
        if self.empty or other.empty:
            return False
        if not self.remask().intersects(other.remask()):
            raise ValueError(f"Regions {self} and {other} do not intersect; cannot compute intersecting slices")
        shared_lower, shared_upper = np.clip(self.bounds, *other.bounds) - self.lower_bounds
        return tuple(map(slice, shared_lower, shared_upper + 1))

    def empty_copy(self):
        """Return an empty region with the same dimension count as this one."""
        return self.__class__(self.dimensions)

    def minify(self):
        """Return a minified copy of this region; reduce the bounding box to the
        minimum possible according to the mask.

        This has no effect on unmasked regions or regions created without
        minify=False.
        """
        if self.empty:
            return self
        return self.__class__(self.bounds, self.mask)

    def remask(self, new_mask=None):
        """Return a copy of this region with a different mask."""
        if self.empty:
            return self
        return self.__class__(self.bounds, new_mask)

    def invert(self, axes=None):
        """Invert (mirror) this region along each axis in the tuple `axes`.

        If `axes` is `None` or omitted, invert all axes. This can be used to
        figure out which cells have a region containing a given cell.
        """
        if self.empty:
            return self
        axes = axes or None  # Turn empty tuple into None.
        if self.has_mask:
            new_mask = np.flip(self.mask, axes)
        else:
            new_mask = None
        if axes:
            new_bounds = self.bounds.copy()
            for axis in axes:
                new_bounds[:, axis] *= -1
        else:
            new_bounds = -self.bounds
        return Region(new_bounds, new_mask)

    def get_mask(self):
        """If `self.has_mask`, return `self.mask`; otherwise return an all-True
        mask with the same size/shape as this region.
        """
        if (not self.empty) and self.has_mask:
            return self.mask
        else:
            return np.ones(self.shape, dtype=bool)

    def get_coordinates_list(self):
        """Return a 2D ndarray listing of cell coordinates.

        The shape of the resulting ndarray is `(region.size, d)`. This result
        has the same contents as the iterator.
        """
        offsets = np.transpose(np.nonzero(self.get_mask()))
        if self.empty:
            return offsets
        return offsets + self.lower_bounds

    def get_coordinates_grid(self):
        """Return a (d+1)-dimensionial ndarray of cell coordinates.

        The shape of the resulting ndarray is `region.shape + (d,)`. If the
        region has no mask then this result, reshaped to (-1, d), is the same as
        `get_coordinates_list()`.
        """
        # For each axis, get the range along that axis.
        if self.empty:
            axis_ranges = (np.arange(0),) * self.dimensions
        else:
            axis_ranges = map(np.arange, self.lower_bounds, self.upper_bounds + 1)
        # Take a Cartesian product of those ranges to get the offsets.
        return utils.arrays.nd_cartesian_grid(*axis_ranges)
