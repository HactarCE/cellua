import itertools
import numpy as np

import utils


def get_recommended_chunk_size(dimensions):
    """Return the "recommended" chunk size for a given dimension count.

    This is based on trying to keep the chunk size big, but still reasonable
    (such that a full chunk is at most 4k) and always a power of 2.

    There isn't actually any data to back this up right now; just that Numpy is
    "good with really big arrays," powers of 2 are fun, and 4kB seems like a
    reasonable amount of RAM to use per chunk.
    """
    max_power = 12  # 2¹² = 4069
    if not 1 <= dimensions <= max_power:
        return ValueError(f'dimension count outside of range: {dimensions}')
    return 2 ** (max_power // dimensions or 1)


class Grid:
    """A data structure tracking the cells and boundary conditions of an
    automaton.

    For now, all grids are infinite.

    Conventions:
    - `global_coords` -- integer ndarray of dimension 1 and size d; identifies a
      cell's position in the whole grid (on the cell scale)
    - `local_coords` -- integer ndarray of dimension 1 and size d; identifies a
      cell's position within a chunk
    - `chunk_coords` -- integer ndarray of dimension 1 and size d; identifies a
      chunk's position in the whole grid (on the chunk scale)
    - `chunk` -- ndarray of dimension d and size `chunk_size`; contains the
      cells of a chunk
    - `neighborhood` -- integer ndarray of shape (d, 2); each number is an
      offset (usually in [negative, positive] pairs) from the center cell; for
      example [[-3, 3], [0, 0], [-1, 1]] describes a 7x1x3 3D neighborhood
    - `chunk_neighborhood` -- same as `neighborhood`, but on a chunk scale

    Public properties
    - `dimensions` -- integer number of dimensions (read-only)
    - `chunk_size` -- integer edge length of each chunk (read-only)
    - `cell_dtype` -- Numpy data type to use for each cell (read-only)
    - `chunks` -- dictionary mapping a tuple of chunk_coords to a chunk (readonly)
    """

    def __init__(self, dimensions, cell_dtype=np.byte):
        self.dimensions = dimensions
        self.cell_dtype = cell_dtype
        self.chunk_size = get_recommended_chunk_size(dimensions)
        self._empty_chunk_prototype = np.zeros([self.chunk_size] * self.dimensions, self.cell_dtype)
        self.chunks = {}

    def empty_copy(self):
        """Return an empty copy of the current grid with all the same settings."""
        return Grid(
            dimensions=self.dimensions,
            cell_dtype=self.cell_dtype
        )

    def copy(self):
        """Return a copy of the current grid with all the same settings and
        cells.
        """
        new_grid = self.empty_copy()
        for chunk_coords, chunk in self.chunks:
            new_grid.set_chunk(chunk_coords, chunk.copy())
        return new_grid

    def get_coords_pair(self, global_coords):
        """Return a tuple `(chunk_coords, local_coords)` for a given global
        location.

        `coords_within_chunk` is modulo `chunk_size`.
        """
        return global_coords // self.chunk_size, global_coords % self.chunk_size

    def has_chunk(self, chunk_coords=None):
        """Check whether a chunk exists.

        This function returns whether the chunk is stored as an array in memory;
        it may still return True even if the chunk is empty.
        """
        return chunk_coords is not None and tuple(chunk_coords) in self.chunks

    def get_chunk(self, chunk_coords=None):
        """Get a chunk from the grid.

        If `chunk_coords` is specified, return the specified chunk. If the
        specified chunk does not exist or `chunk_coords` is None, return a new
        blank chunk. Either way, returns a d-dimensional cell array of size
        `chunk_size`.
        """
        chunk_key = chunk_coords is not None and tuple(chunk_coords)
        return self.chunks.get(chunk_key, self._empty_chunk_prototype.copy())

    def set_chunk(self, chunk_coords, new_chunk):
        """Set the chunk at the specified chunk coordinates.

        This does not have any special handling for empty chunks.
        """
        self.chunks[tuple(chunk_coords)] = new_chunk

    def del_chunk(self, chunk_coords):
        """Delete the chunk at the specified coordinates.

        If the chunk does not exist, this has no effect.
        """
        if self.has_chunk(chunk_coords):
            del self.chunks[tuple(chunk_coords)]

    def del_chunk_if_empty(self, chunk_coords):
        """Delete the chunk at the specified coordinates iff it is empty.

        If the chunk is not empty or does not exist, this has no effect.
        """
        if self.has_chunk(chunk_coords) and not self.get_chunk(chunk_coords).any():
            del self.chunks[tuple(chunk_coords)]

    def set_cell(self, global_coords, new_state):
        """Set the state of the cell at the specified global coordinates.

        If the chunk containing this cell does not exist yet, automatically
        create it.
        """
        chunk_coords, coords_within_chunk = self.get_coords_pair(global_coords)
        chunk = self.get_chunk(chunk_coords)
        if not self.has_chunk(chunk_coords):
            self.set_chunk(chunk_coords, chunk)
        chunk[tuple(coords_within_chunk)] = new_state

    def get_cell(self, global_coords):
        """Get the state of the cell at the specified global coordinates."""
        chunk_coords, coords_within_chunk = self.get_coords_pair(global_coords)
        return self.get_chunk(chunk_coords)[tuple(coords_within_chunk)]

    def _get_chunk_neighborhood(self, neighborhood):
        """Get the chunk neighborhood given a cell neighborhood.

        This returns the neighborhood, on the chunk scale, that is guaranteed to
        contain the neighborhood of every cell in the origin chunk. This
        determines the size/shape of the array returned by get_chunk_napkin().
        """
        # Lower bound examples (chunk_size=16):
        # -32..-17 --> -2
        # -16..-1  --> -1
        #   0..15  --> 0
        # Upper bound (chunk_size=16):
        # -15..0   --> 0
        #   1..16  --> 1
        #  17..32  --> 2
        return np.stack((
            neighborhood[:,0] // self.chunk_size,  # Calculate lower bound.
            (neighborhood[:,1] - 1) // self.chunk_size + 1,  # Calculate upper bound.
        ), axis=-1)

    def get_chunk_napkin(self, chunk_coords, neighborhood):
        """Get the d-dimensional napkin of a chunk.

        `neighborhood` is on the cell scale; see `get_cell_napkin()`.

        The size of the return value can be predicted from
        `_get_chunk_neighborhood()`, but for external callers that shouldn't be
        necessary. Just pass the result of this function to `get_cell_napkin()`.
        """
        d = self.dimensions
        chunk_neighborhood = self._get_chunk_neighborhood(neighborhood)
        # Calculate the eventual shape for the final array. In future comments,
        # we'll call this M.
        chunk_neighborhood_shape = chunk_neighborhood[:,1] - chunk_neighborhood[:,0] + 1
        axis_ranges = (np.arange(a[0], a[1] + 1) for a in chunk_neighborhood)
        # Get an M.size*d array of offsets.
        chunk_offsets = utils.nd_cartesian(*axis_ranges)
        # Reshape that to M.shape*d.
        chunk_offsets = chunk_offsets.reshape(tuple(chunk_neighborhood_shape) + (d,))
        # Using the last axis (the d-sized one), get the chunk for each
        # coordinate bunch.
        chunks = np.apply_along_axis(self.get_chunk, -1, chunk_coords + chunk_offsets)
        # Now the array is M.shape*chunk_size^d. The outermost d dimensions correspond to
        # chunk layout, while the innermost d dimensions correspond to cell
        # layout within each chunk. We want to merge each nth dimension with the
        # (n+d)th, since they're really the same spatially. We could call
        # np.concatenate() repeatedly, but instead we'll swap dimensions such
        # that each nth and (n+d)th dimension are "adjacent," and then just
        # reshape the array.
        chunks = chunks.transpose(*(axis for n in range(d) for axis in [n, n + d]))
        # Now that we've transposed the axes, we can reshape it, merging pairs
        # of adjacent dimensions.
        chunks = chunks.reshape(tuple(chunk_neighborhood_shape * self.chunk_size))
        return chunks

    def get_cell_napkin(self, global_coords, neighborhood, chunk_napkin=None):
        """Given a cell's global coordinaites, and optionially a chunk napkin,
        extract the cell's napkin.

        `chunk_napkin` must be the return value from `get_chunk_napkin()` when
        passed the same `global_coords` and `neighborhood`. If left blank, it
        will be inferred.
        """
        chunk_coords, local_coords = self.get_coords_pair(global_coords)
        chunk_neighborhood = self._get_chunk_neighborhood(neighborhood)
        if chunk_napkin is None:
            chunk_napkin = self.get_chunk_napkin(chunk_coords, neighborhood)
        # Find the coordinates of the given cell within `chunk_napkin`.
        coords_within_cn = -chunk_neighborhood[:,0] * self.chunk_size + local_coords
        start = coords_within_cn + neighborhood[:,0]
        end = coords_within_cn + neighborhood[:,1] + 1
        return chunk_napkin[tuple(map(slice, start, end))]
