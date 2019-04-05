import numpy as np
import itertools

import utils


class Region:
    """An immutable object describing an integer hyperrectangle.

    Public read-only properties:
    - `dimensions` -- integer number of dimensions
    - `bounds` -- integer ndarray of shape (2, d); each row is a set of
      coordinate offsets from the center cell; for example [[-3, 0, -1], [2, 0,
      1] describes a 6x1x3 3D region
    - `lower_bounds` -- 1D integer ndarray of size d representing the lower
      bounds of the region along each axis; equivalent to `bounds[0]`
    - `upper_bounds` -- 1D integer ndarray of size d representing the upper
      bounds of the region along each axis; equivalent to `bounds[1]`
    - `max_radius` -- integer absolute maximum value of `bounds`

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

    A Region object can be used with the `in` keyword to test whether it
    contains some given offset:

    ```py
    >>> region = Region([[-2, 0], [0, 1]])
    >>> np.array([-1, 0]) in region
    True
    >>> np.array([0, -1]) in region:
    False
    ```

    `len()` can be used to get the number of cells in a Region object:
    ```py
    >>> len(Region([[-2, 0], [0, 1]]))
    6
    """

    def __init__(self, bounds):
        bounds = np.array(bounds)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            return ValueError(f"Invalid region array shape: {bounds}")
        bounds.sort(0)
        self.dimensions = bounds.shape[1]
        self.bounds = bounds
        self.lower_bounds, self.upper_bounds = self.bounds
        shape_ndarray = self.upper_bounds - self.lower_bounds + 1
        self.shape = tuple(shape_ndarray)
        self.size = shape_ndarray.prod()
        self.max_radius = np.absolute(bounds).max()

    def __contains__(self, offset):
        """Determine whether some offset is within the region."""
        above_lower = (self.lower_bounds <= np.array(offset)).all()
        below_upper = (np.array(offset) <= self.upper_bounds).all()
        return above_lower and below_upper

    def __iter__(self):
        """Iterate over cell coordinate offsets.

        Each element of the iterator is a 1-dimensional ndarray representing an
        offset from the "center" cell of the region.
        """
        # For each axis, get the range along that axis.
        axis_range_iters = map(range, self.lower_bounds, self.upper_bounds + 1)
        # Take a Cartesian product of those ranges to get the offsets.
        return map(np.array, itertools.product(*axis_range_iters))

    def __len__(self):
        return self.size

    def __repr__(self):
        return f'{self.__class__.__name__}({self.bounds!r})'

    def __str__(self):
        return 'x'.join(map(str, self.shape)) + ' region'

    def __eq__(self, other):
        return isinstance(other, Region) and (self.bounds == other.bounds).all()

    def invert(self, axes=None):
        """Return a Region that is inverted (mirrored) along each axis in
        `axes`.

        If `axes` is `None` or omitted, invert all axes. This can be used to
        figure out which cells have a region containing a given cell.
        """
        if axes:
            new_bounds = self.bounds.copy()
            for axis in axes:
                new_bounds[:, axis] *= -1
            return Region(new_bounds)
        else:
            return Region(-self.bounds)

    def get_offset_grid(self):
        """Return a d-dimensionial ndarray of cell coordinate offsets.

        The shape of the resulting array is `region.shape + (d,)`. This
        result, reshaped to (-1, d), has the same contents as the iterator.
        """
        # See Region.__iter__() for an explanation.
        axis_ranges = map(np.arange, self.lower_bounds, self.upper_bounds + 1)
        return utils.arrays.nd_cartesian_grid(*axis_ranges)
