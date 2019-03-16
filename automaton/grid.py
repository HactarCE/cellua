import itertools
import numpy as np

import utils


def get_recommended_chunk_size(dimensions):
    """Return the "recommended" chunk size for a given dimension count.

    This is based on trying to keep the chunk size big, but still reasonable
    (such that a full chunk is at most 4k) and always a power of 2.

    There isn't actually any data to back this up right now; just that Numpy is
    "good with really big arrays" and that 4kB seems like a reasonable amount of
    RAM to use per chunk.
    """
    max_power = 12  # 2¹² = 4069
    if not 1 <= dimensions <= 12:
        return ValueError(f'dimension count outside of range: {dimensions}')
    return 2 ** (12 // dimensions or 1)


class Grid:
    """A data structure tracking the cells and boundary conditions of an
    automaton.

    For now, all grids are infinite.

    Conventions:
    - `global_coords` -- integer ndarray of dimension 1 and size d; identifies a
      cell's position in the whole pattern
    - `local_coords` -- integer ndarray of dimension 1 and size d; identifies a
      cell's position within a chunk
    - `chunk_coords` -- integer ndarray of dimension 1 and size d; identifies a
      chunk's position in the whole pattern

    Public properties
    - `dimensions` -- integer number of dimensions (read-only)
    - `chunk_size` -- integer edge length of each chunk (read-only)
    - `cell_dtype` -- Numpy data type (read-only)
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
        cells."""
        new_grid = self.empty_copy()
        for chunk_coords, chunk in self.chunks:
            new_grid.set_chunk(chunk_coords, chunk.copy())
        return new_grid

    def get_coords_pair(self, global_coords):
        """Return a tuple `(chunk_coords, coords_within_chunk)` for a given
        global location.

        `coords_within_chunk` is modulo `chunk_size`.
        """
        return global_coords // self.chunk_size, global_coords % self.chunk_size

    def has_chunk(self, chunk_coords=None):
        """Check whether a chunk exists.

        TODO test
        """
        return chunk_coords is not None and tuple(chunk_coords) in self.chunks

    def get_chunk(self, chunk_coords=None):
        """Get a chunk from the grid.

        If `chunk_coords` is specified, return the specified chunk. If the
        specified chunk does not exist or `chunk_coords` is None, return a new
        blank chunk. Either way, returns a d-dimensional cell array of size
        `chunk_size`.

        TODO test
        """
        chunk_key = chunk_coords is not None and tuple(chunk_coords)
        return self.chunks.get(chunk_key, self._empty_chunk_prototype.copy())

    def set_chunk(self, chunk_coords, new_chunk):
        """Set the chunk at the specified chunk coordinates.

        This does not have any special handling for empty chunks

        TODO test
        """
        self.chunks[utils.trytuple(chunk_coords)] = new_chunk

    def del_chunk(self, chunk_coords):
        """Delete the chunk at the specified coordinates.

        If the chunk does not exist, this has no effect.

        TODO test
        """
        if self.has_chunk(chunk_coords):
            del self.chunks[tuple(chunk_coords)]

    def del_chunk_if_empty(self, chunk_coords):
        """Delete the chunk at the specified coordinates iff it is empty.

        If the chunk is not empty or does not exist, this has no effect.

        TODO test
        """
        if self.has_chunk(chunk_coords) and not self.get_chunk(chunk_coords).any():
            del self.chunks[chunk_coords]

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

    def get_chunk_napkin(self, chunk_coords):
        """Get the 3^d napkin of a chunk.

        Returns a d-dimensional cell array of size `3*chunk_size`.

        TODO test
        """
        chunk_offsets = utils.nd_cartesian((-1, 0, 1), repeat=self.dimensions)
        absolute_coords_for_chunks = (utils.global_coords.add(chunk_coords, offset) for offset in chunk_offsets)
        chunks = map(self.get_chunk, absolute_coords_for_chunks)
        return np.block(np.reshape(chunks, np.repeat(3, self.dimensions)))
        return self.get_chunk(chunk_coords)

    def get_cell_napkin(self, chunk_napkin, global_coords, neighborhood_size):
        """Given a chunk napkin, extract a cell's napkin.

        `global_coords` is assumed to refer to the center chunk of `chunk_napkin`; it is
        modulus `chunk_size`.

        `neighborhood` is a d-dimensional boolean array of size 2k+1, where k <=
        chunk_size. The mask is centered on the cell at `global_coords`.

        TODO test
        """
        center_cell = global_coords - self.chunk_size
        start = center_cell - neighborhood_size
        end = center_cell + neighborhood_size
        return chunk_napkin[start:end]
