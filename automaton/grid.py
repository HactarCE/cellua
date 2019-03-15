import itertools
import numpy as np

import utils

CHUNK_SIZE = 16
CELL_DTYPE = np.byte

def get_chunk_coords(cell_coords):
    """Get the chunk key (i.e. chunk coordinates) for a given cell location.

    `cell_coords` is an ndarray of dimension 1 and size d.
    """
    return cell_coords // CHUNK_SIZE

def get_coords_within_chunk(cell_coords):
    """Get the coordinates with a chunk (i.e. mod `CHUNK_SIZE`) for a given
    cell location.

    `cell_coords` is an ndarray of dimension 1 and size d.
    """
    return cell_coords % CHUNK_SIZE

def get_cell_coords_pair(cell_coords):
    """Return a tuple `(chunk_coords, coords_within_chunk)` for a given cell
    location.

    `cell_coords` is an ndarray of dimension 1 and size d.
    """
    return get_chunk_coords(cell_coords), get_coords_within_chunk(cell_coords)

class Grid:
    """A data structure tracking the cells and boundary conditions of an
    automaton.

    For now, all grids are infinite.
    """

    def __init__(self, dimensions, cell_dtype=CELL_DTYPE):
        self.dimensions = dimensions
        self.cell_dtype = cell_dtype
        self._empty_chunk_prototype = np.zeros([CHUNK_SIZE] * self.dimensions, self.cell_dtype)
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
        `CHUNK_SIZE`.
        """
        chunk_key = chunk_coords is not None and tuple(chunk_coords)
        return self.chunks.get(chunk_key, self._empty_chunk_prototype.copy())

    def set_chunk(self, chunk_coords, new_chunk):
        """TODO document and test"""
        self.chunks[utils.trytuple(chunk_coords)] = new_chunk
        # self.remove_chunk_if_empty(chunk_coords)

    def remove_chunk_if_empty(self, chunk_coords):
        """TODO document and test"""
        if not np.any(self.chunks[chunk_coords]):
            del self.chunks[chunk_coords]

    def set_cell(self, cell_coords, new_cell):
        """TODO document"""
        chunk_coords, coords_within_chunk = get_cell_coords_pair(cell_coords)
        chunk_coords = get_chunk_coords(cell_coords)
        coords_within_chunk = get_coords_within_chunk(cell_coords)
        chunk = self.get_chunk(chunk_coords)
        if not self.has_chunk(chunk_coords):
            self.set_chunk(chunk_coords, chunk)
        chunk[tuple(coords_within_chunk)] = new_cell

    def get_cell(self, cell_coords):
        """TODO document"""
        chunk_coords, coords_within_chunk = get_cell_coords_pair(cell_coords)
        return self.get_chunk(chunk_coords)[tuple(coords_within_chunk)]
