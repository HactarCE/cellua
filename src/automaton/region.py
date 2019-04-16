from abc import ABC, abstractmethod
import numpy as np

import utils.arrays
import utils.convert


class Region(ABC):
    """An immutable base class for any finite set of positions on a grid.

    Public read-only properties:
    - count -- integer; number of members
    - dimensions -- integer; number of dimensions
    - is_empty -- bool; whether count == 0
    - shape -- integer tuple; Numpy-like shape of bounding box

    A region can be used as an iterator to get the coordinates of each cell:

    ```py
    >>> for offset in Region.span([[-2, 0], [0, 1]]):
    ...     print(offset)
    [-2 0]
    [-2 1]
    [-1 0]
    [-1 1]
    [0 0]
    [0 1]
    ```

    A region can be used with the `in` keyword to test whether it contains some
    position or fully contains some other region:

    ```py
    >>> some_coords in some_region
    >>> inner_region in outer_region
    ```

    `len()` returns the number of cells in a region (same as `region.count`).

    Boolean operators can be used to find the intersection (`r1 & r2`), union
    (`r1 | r2`), and difference (`r1 ^ r2`) between two regions. To "subtract"
    one region from another, use `r1 & r2 ^ r1`. To check for intersection,
    prefer `Region.intersects()` over `&`.

    Addition and subtraction all operate on each coordinate separately. A
    coordinate tuple can be added to or subtracted from a region; e.g. `region +
    (2, -1)` will offset a 2D region by +2 along X and -1 along Y.

    Regions are equal iff they contain the same set of cells. Region equality
    can be tested using `==`.
    """

    def span(bounds, mask=None):
        """Instantiate a Region subclass from bounds and an optional mask.

        Arguments:
        - bounds -- one of the following:
            - integer ndarray of shape (2, d); two opposite corners
            - integer ndarray of shape (d,); single cell
            - Region; bounds will be reused

        Optional arguments:
        - mask (default None) -- None or boolean ndarray with shape the same as
          region

        If a single cell is supplied with no mask, a 1^d region will be
        returned. If a single cell is supplied with a non-null mask, the single
        cell will be used as the minimum bound and the region size will be
        inferred from the mask.
        """
        if isinstance(bounds, EmptyRegion):
            region = bounds
            if mask is None:
                return region
            else:
                raise TypeError(f"Cannot mask {region} of type {region.__class__.__name__}")
        elif isinstance(bounds, RectRegion):
            region = bounds
            bounds = region.bounds
        else:
            dimensions = None if mask is None else mask.ndim
            try:
                lower_bounds = utils.convert.to_coords(bounds, dimensions)
                if mask is None:
                    upper_bounds = lower_bounds
                else:
                    upper_bounds = lower_bounds + mask.shape - 1
                bounds = np.stack((lower_bounds, upper_bounds))
            except ValueError:
                bounds = utils.convert.to_bounds(bounds)
        dimensions = bounds.shape[1]
        if mask is not None:
            if mask.ndim != dimensions:
                raise ValueError(f"Cannot mask {dimensions}D bounds with {mask.ndim}D mask")
            region_shape = tuple(bounds[1] - bounds[0] + 1)
            if mask.shape != region_shape:
                raise ValueError(f"Mask shape {mask.shape} does not match region shape {region_shape}")
            if not mask.any():
                return Region.empty(dimensions)
            # For each axis, try to make the region as small as possible
            # while still including all the True values in the mask.
            for axis in range(dimensions):
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
            # If the mask is all True, don't bother storing it.
            if not mask.all():
                # Copy the mask to guarantee immutability and potentially save
                # memory by making the array contiguous.
                mask = mask.copy()
                return MaskedRegion(bounds=bounds, mask=mask)
        return RectRegion(bounds=bounds)

    def empty(arg):
        """Instantiate an empty region.

        Arguments:
        - arg -- one of the following:
            - integer; number of dimensions
            - Region; number of dimensions will be inferred from Region
        """
        if isinstance(arg, Region):
            return EmptyRegion(d=arg.dimensions)
        else:
            return EmptyRegion(d=utils.convert.to_dimen(arg))

    @property
    @abstractmethod
    def dimensions(self):
        ...

    @property
    def is_empty(self):
        return not self.count

    @property
    @abstractmethod
    def shape(self):
        ...

    @property
    @abstractmethod
    def count(self):
        ...

    def __len__(self):
        return self.count

    def __contains__(self, other):
        """Return whether some offset or region is within this region."""
        if isinstance(other, Region):
            return self._contains_region(other)
        else:
            return self._contains_position(utils.convert.to_coords(other, self.dimensions))

    @abstractmethod
    def _contains_region(self, other):
        """Return whether some region is within this region."""

    @abstractmethod
    def _contains_position(self, other):
        """Return whether some position is within this region."""

    @abstractmethod
    def __iter__(self):
        """Iterate over cell positions.

        Each element of the iterator is a 1D ndarray of coordinates.
        """

    @abstractmethod
    def __eq__(self, other):
        ...

    @abstractmethod
    def _op(self, op, other):
        """Apply a set operation between two regions.

        Arguments:
        - self -- Region
        - op -- string; one of ('&', '|', '^')
        - other -- Region
        """

    def _boolean_op(self, op, other):
        """Internal function used for boolean operator magic methods."""
        if isinstance(other, Region):
            if self.dimensions == other.dimensions:
                result = self._op(op, other)
                if result is NotImplemented:
                    result = other._op(op, self)
                return result
            else:
                raise ValueError(f"Dimension mismatch between {self} and {other}")
        else:
            return NotImplemented

    def __and__(self, other):
        """Return the intersection of two regions."""
        return self._boolean_op('&', other)

    def __xor__(self, other):
        """Return the symmetric difference between two regions."""
        return self._boolean_op('^', other)

    def __or__(self, other):
        """Return the union of two regions."""
        return self._boolean_op('|', other)

    @abstractmethod
    def _offset(self, offset):
        """Offset a region by a coordinate tuple."""

    def __add__(self, other):
        """Offset a region by a coordinate tuple."""
        return self._offset(utils.convert.to_coords(other, self.dimensions))

    def __sub__(self, other):
        """Offset a region by a negated coordinate tuple."""
        return self._offset(-utils.convert.to_coords(other, self.dimensions))

    @abstractmethod
    def intersects(self, other):
        """Return whether `self` and `other` intersect at all."""

    @abstractmethod
    def invert(self, axes=None):
        """Invert (mirror) this region along each axis in the tuple `axes`.

        If `axes` is `None` or omitted, invert all axes. This can be used to
        figure out which cells have a region containing a given cell.
        """

    @property
    @abstractmethod
    def box(self):
        """A region representing the bounding box of this region."""

    @property
    @abstractmethod
    def positions(self):
        """A 2D ndarray listing of cell coordinates.

        The shape of the resulting ndarray is `(region.count, d)`. This result
        has the same contents as the iterator.
        """

    @property
    @abstractmethod
    def position_grid(self):
        """Return a (d+1)-dimensionial ndarray of cell coordinates in the
        bounding box of this region.

        The shape of the resulting ndarray is `region.shape + (d,)`. If the
        region is hyperrectangular then this result, reshaped to (-1, d), is the
        same as `region.positions`.
        """


class EmptyRegion(Region):
    """An immutable empty set of positions on a grid.

    Do not instantiate this class directly; use Region.empty() instead.
    """

    def __init__(self, *args, d):
        self._dimensions = d

    @property
    def dimensions(self):
        """Implements Region.dimensions."""
        return self._dimensions

    is_empty = True  # Overrides Region.is_empty.

    @property
    def shape(self):
        """Implements Region.shape."""
        return (0,) * self.dimensions

    count = 0  # Overrides Region.count.

    def _contains_region(self, other):
        """Implements Region._contains_region()."""
        return other.is_empty

    def _contains_position(self, other):
        """Implements Region._contains_position()."""
        return False

    def __iter__(self):
        """Implements Region.__iter__()."""
        return iter(())

    def __repr__(self):
        return f'Region.empty({self.dimensions})'

    def __str__(self):
        return f'[empty {self.dimensions}D region]'

    def __eq__(self, other):
        """Implements Region.__eq__()."""
        return isinstance(other, Region) and other.is_empty

    def _op(self, op, other):
        """Implements Region._op()."""
        if op == '&':
            return self
        if op in ('|', '^'):
            return other

    def _offset(self, offset):
        """Implements Region._offset()."""
        return self

    def intersects(self, other):
        """Implements Region.intersects()."""
        return False

    def invert(self, axes=None):
        """Implements Region.invert()."""
        return self

    @property
    def box(self):
        """Implements Region.box"""
        return self

    @property
    def positions(self):
        """Implements Region.positions."""
        return np.empty(shape=(0, self.dimensions), dtype=np.int64)

    @property
    def position_grid(self):
        """Implements Region.position_grid."""
        return np.empty(shape=self.shape + (self.dimensions,), dtype=np.int64)


class RectRegion(Region):
    """An immutable non-empty hyperrectangle of positions on a grid.

    Do not instantiate this class directly; use Region.span() instead.

    Public read-only properties:
    - bounds -- integer ndarray of shape (2, d); each row is a set of coordinate
      offsets from the center cell; for example [[-3, 0, -1], [2, 0, 1]
      describes a 6x1x3 3D region
    - lower_bounds -- 1D integer ndarray of size d representing the lower bounds
      of the region along each axis; equivalent to `bounds[0]`
    - upper_bounds -- 1D integer ndarray of size d representing the upper bounds
      of the region along each axis; equivalent to `bounds[1]`
    - max_radius -- integer absolute maximum value of `bounds`
    - has_mask -- bool; whether the regions is masked (only True for subclasses)
    ... and all of Region's read-only properties.
    """

    def __init__(self, *args, bounds):
        self.lower_bounds, self.upper_bounds = self.bounds = bounds

    @property
    def dimensions(self):
        """Implements Region.dimensions."""
        return self.lower_bounds.size

    is_empty = False  # Overrides Region.is_empty.

    has_mask = False

    @property
    def mask(self):
        return np.ones(self.shape, dtype=np.bool)

    @property
    def _shape_tuple(self):
        return self.upper_bounds - self.lower_bounds + 1

    @property
    def shape(self):
        """Implements Region.shape."""
        return tuple(self._shape_tuple)

    @property
    def count(self):
        """Implements Region.count."""
        return np.prod(self._shape_tuple)

    def _contains_region(self, other):
        """Implements Region._contains_region()."""
        if other.is_empty:
            return True
        if not isinstance(other, RectRegion):
            return NotImplemented
        # Check whether the bounding boxes fit.
        if not ((other.upper_bounds <= self.upper_bounds).all() and
                (self.lower_bounds <= other.lower_bounds).all()):
            return False
        # If the bounding boxes fit and this region has no mask, then the other
        # region will definitely fit inside.
        if not (self.has_mask or other.has_mask):
            return True
        # Subtract `self` from `other` and check whether any cells remain.
        return (self & other ^ other).is_empty

    def _contains_position(self, other):
        """Implements Region._contains_position()."""
        coords = utils.convert.to_coords(other, self.dimensions)
        above_lower = (self.lower_bounds <= coords).all()
        below_upper = (coords <= self.upper_bounds).all()
        return above_lower and below_upper

    def __iter__(self):
        """Implements Region.__iter__()."""
        offsets = np.ndindex(*self.shape)
        return (np.array(i) + self.lower_bounds for i in offsets)

    def __repr__(self):
        if len(self) == 1:
            b = self.lower_bounds
        else:
            b = self.bounds
        return f'Region.span({b!r})'

    def __str__(self):
        if len(self) == 1:
            s = f'single-cell region at {tuple(self.lower_bounds)}'
        else:
            s = 'x'.join(map(str, self.shape))
            if self.dimensions == 1:
                s = f'length-{s}'
            s += f' region from {tuple(self.lower_bounds)} to {tuple(self.upper_bounds)}'
        return f'[{s}]'

    def __eq__(self, other):
        """Overrides Region.__eq__()."""
        if not isinstance(other, RectRegion):
            return False
        return ((self.bounds == other.bounds).all()
                and not (self.has_mask or other.has_mask))

    def _op(self, op, other):
        """Overrides Region._op()."""
        if self.dimensions != other.dimensions:
            raise ValueError(f"Dimension mismatch between {self} and {other}")
        if other.is_empty:
            return other._op(op, self)
        if op == '&':
            # Optimization: When computing intersection beytween two regions
            # with non-intersecting bounding boxes, the result is empty.
            # (Remove masks because `Region.intersects()` depends on this
            # function for handling masked regions.)
            if not Region.span(self).intersects(Region.span(other)):
                return Region.empty(self)
            # Optimization: When computing intersection, only include the
            # intersection between the regions (obviously).
            lower_bounds = np.maximum(self.lower_bounds, other.lower_bounds)
            upper_bounds = np.minimum(self.upper_bounds, other.upper_bounds)
            # Optimization: When computing intersection between two regions
            # without masks, the final region does not need a mask.
            if not (self.has_mask or other.has_mask):
                return Region.span([lower_bounds, upper_bounds])
        else:
            # For anything besides intersection, get the lower and upper
            # bounds of the union of both regions. (Compute the mask later.)
            lower_bounds = np.minimum(self.lower_bounds, other.lower_bounds)
            upper_bounds = np.maximum(self.upper_bounds, other.upper_bounds)
        r = Region.span([lower_bounds, upper_bounds])
        new_mask = np.zeros(r.shape, dtype=bool)
        new_mask[r.slices(self)] = self.mask[self.slices(r)] if self.has_mask else True
        sliced_other_mask = other.mask[other.slices(r)] if other.has_mask else True
        if op == '&':
            new_mask[r.slices(other)] &= sliced_other_mask
        elif op == '|':
            new_mask[r.slices(other)] |= sliced_other_mask
        elif op == '^':
            new_mask[r.slices(other)] ^= sliced_other_mask
        return Region.span(r, new_mask)

    def _offset(self, offset):
        """Implements Region._offset()."""
        return Region.span(self.bounds + offset)

    def intersects(self, other):
        """Implements Region.intersects()."""
        if other.is_empty:
            return False
        # Check whether the bounding boxes intersect.
        if not ((self.lower_bounds <= other.upper_bounds) &
                (other.lower_bounds <= self.upper_bounds)).all():
            return False
        # If the bounding boxes overlap and neither region has a mask, then they
        # must intersect.
        if not (self.has_mask or other.has_mask):
            return True
        return not (self & other).is_empty

    def invert(self, axes=None):
        """Implements Region.invert()."""
        axes = axes or None  # Turn empty tuple into None.
        if axes:
            new_bounds = self.bounds.copy()
            for axis in axes:
                new_bounds[:, axis] *= -1
        else:
            new_bounds = -self.bounds
        return Region.span(new_bounds)

    @property
    def box(self):
        """Implements Region.box"""
        return self

    @property
    def positions(self):
        """Implements Region.positions."""
        return self.position_grid.reshape(-1, self.dimensions)

    @property
    def position_grid(self):
        """Implements Region.position_grid."""
        # For each axis, get the range along that axis.
        axis_ranges = map(np.arange, self.lower_bounds, self.upper_bounds + 1)
        # Take a Cartesian product of those ranges to get the offsets.
        return utils.arrays.nd_cartesian_grid(*axis_ranges)

    def slices(self, other):
        """Return a slice tuple that selects the intersection of `self` and
        `other`'s bounding boxes relative to `self.lower_bound`.

        Raises ValueError if `self` and `other` do not intersect.
        """
        if other.is_empty:
            return False
        if not self.box.intersects(other.box):
            raise ValueError(f"Regions {self} and {other} do not intersect; cannot compute intersecting slices")
        shared_lower, shared_upper = np.clip(self.bounds, *other.bounds) - self.lower_bounds
        return tuple(map(slice, shared_lower, shared_upper + 1))


class MaskedRegion(RectRegion):
    """An immutable non-empty masked hyperrectangle of positions on a grid.

    Do not instantiate this class directly; use Region.span() instead.

    Public read-only properties:
    - mask -- bool ndarray with same shape as region
    ... and all of RectRegion's read-only properties.
    """

    def __init__(self, *args, bounds, mask):
        super().__init__(*args, bounds=bounds)
        error_extra = " Region.__new__() should prevent this exception."
        not_allowed = f" is not allowed for {self.__class__.__name__}."
        if mask is None:
            raise ValueError("Null mask" + not_allowed + error_extra)
        if mask.all():
            raise ValueError("Full mask" + not_allowed + error_extra)
        if not mask.any():
            raise ValueError("Empty mask" + not_allowed + error_extra)
        if mask.shape != self.shape:
            raise ValueError("Mask shape does not match region shape." + error_extra)
        self._count = np.count_nonzero(mask)
        self._mask = mask

    has_mask = True  # Overrides RectRegion.has_mask.

    @property
    def mask(self):
        """Overrides RectRegion.mask."""
        return self._mask

    @property
    def count(self):
        return self._count

    def _mask_contains(self, coords):
        """Return whether a given position is included in the mask.

        `coords` is assumed to be within the bounding box of the region.
        """
        return self.mask[tuple(coords - self.lower_bounds)]

    def _contains_position(self, other):
        """Overrides RectRegion._contains_position()."""
        coords = utils.convert.to_coords(other, self.dimensions)
        if not super()._contains_position(coords):
            return False
        return self._mask_contains(coords)

    def __iter__(self):
        """Overrides RectRegion.__iter__()."""
        return filter(self._mask_contains, super().__iter__())

    def __repr__(self):
        """Overrides RectRegion.__repr__()."""
        s = super().__repr__()[:-1]
        s += f', {self.mask!r}'
        s += ')'
        return s

    def __str__(self):
        """Overrides RectRegion.__str__()."""
        s = super().__str__()[:-1]
        s += f' with {len(self)}-element mask'
        s += ']'
        return s

    def __eq__(self, other):
        """Overrides RectRegion.__eq__()"""
        return (isinstance(other, MaskedRegion)
                and (self.bounds == other.bounds).all()
                and (self.mask == other.mask).all())

    def _offset(self, offset):
        """Overrides RectRegion._offset()."""
        return Region.span(self.bounds + offset, self.mask)

    def invert(self, axes=None):
        """Overrides RectRegion.invert()."""
        axes = axes or None  # Turn empty tuple into None.
        new_mask = np.flip(self.mask, axes)
        return Region.span(super().invert(axes), new_mask)

    @property
    def box(self):
        """Overrides RectRegion.box"""
        return Region.span(self.bounds)

    @property
    def positions(self):
        """Overrides RectRegion.positions."""
        offsets = np.transpose(np.nonzero(self.mask))
        return offsets + self.lower_bounds
