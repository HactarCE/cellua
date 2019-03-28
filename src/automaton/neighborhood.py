import numpy as np
import itertools

import utils


class Neighborhood:
    """An immutable object representing a cellular automaton neighborhood.

    A neighborhood describes the set of relative cell positions that can
    influence a given cell's next state. A neighborhood object can exist on the
    cell level or on the chunk level; this class makes no distinction, since
    operations on them are the same.

    This program only allows rectangular (or hyperrectangular) neighborhoods,
    but certain cells can obviously be ignored by the transition function. The
    only real restriction is that the neigborhood must be finite.

    Public read-only properties:
    - `dimensions` -- integer number of dimensions
    - `bounds` -- integer ndarray of shape (2, d); each row is a set of
      coordinate offsets from the center cell; for example [[-3, 0, -1], [2, 0,
      1] describes a 6x1x3 3D neighborhood
    - `lower_bounds` -- 1D integer ndarray of size d representing the lower
      bounds of the neighborhood along each axis; equivalent to `bounds[0]`
    - `upper_bounds` -- 1D integer ndarray of size d representing the upper
      bounds of the neighborhood along each axis; equivalent to `bounds[1]`
    - `max_radius` -- integer absolute maximum value of `bounds`; i.e. the
      Moore radius of the neighborhood

    A Neighborhood object can be used as an iterator to get the relative
    coordinates (offsets) of each neighbor:
    ```py
    >>> for offset in Neighborhood([[-2, 0], [0, 1]]):
    ...     print(offset)
    [-2 0]
    [-2 1]
    [-1 0]
    [-1 1]
    [0 0]
    [0 1]
    ```

    A Neigborhood object can be used with the `in` keyword to test whether it
    contains some given offset:
    ```py
    >>> neighborhood = Neighborhood([[-2, 0], [0, 1]])
    >>> np.array([-1, 0]) in neighborhood
    True
    >>> np.array([0, -1]) in neighborhood:
    False
    ```

    `len()` can be used to get the number of neighbors in a Neighborhood object:
    ```py
    >>> len(Neighborhood([[-2, 0], [0, 1]]))
    6
    """

    def __init__(self, bounds):
        bounds = np.array(bounds)
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            return ValueError(f"Invalid neighborhood array shape: {bounds}")
        bounds.sort(0)
        self.dimensions = bounds.shape[1]
        self.bounds = bounds
        self.lower_bounds, self.upper_bounds = self.bounds
        shape_ndarray = self.upper_bounds - self.lower_bounds + 1
        self.shape = tuple(shape_ndarray)
        self.size = shape_ndarray.prod()
        self.max_radius = np.absolute(bounds).max()

    def __contains__(self, offset):
        """Determine whether some offset is within the neighborhood."""
        above_lower = (self.lower_bounds <= np.array(offset)).all()
        below_upper = (np.array(offset) <= self.upper_bounds).all()
        return above_lower and below_upper

    def __iter__(self):
        """Iterate over neighbor coordinate offsets.

        Each element of the iterator is a 1-dimensional ndarray representing an
        offset from the "center" cell of the neighborhood.
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
        return 'x'.join(map(str, self.shape)) + ' neighborhood'

    def __eq__(self, other):
        return isinstance(other, Neighborhood) and (self.bounds == other.bounds).all()

    def copy(self):
        return Neighborhood(self.bounds.copy())

    def get_inverse(self):
        """Return a Neighborhood that is inverted (mirrored) along each axis.

        This can be used to figure out which cells have a neighborhood
        containing a given cell.
        """
        return Neighborhood(-self.bounds)

    def get_offset_grid(self):
        """Return a d-dimensionial ndarray of neighbor coordinate offsets.

        The shape of the resulting array is `neighborhood.shape + (d,)`. This
        result, reshaped to (-1, d), has the same contents as the iterator.
        """
        # See Neighborhood.__iter__() for an explanation.
        axis_ranges = map(np.arange, self.lower_bounds, self.upper_bounds + 1)
        return utils.nd_cartesian_grid(*axis_ranges)
