class Pattern:
    """A pattern of cells in a cellular automaton."""

    def __init__(self, cell_array, center_coords, mask=None):
        self.cell_array = cell_array
        self.center_coords = center_coords
        self.dimensions = center_coords.size

    def __iter__(self):
        """Iterate over all cells."""
        return self.cell_array.flat

    def get_cell(self, *coords):
        """Get the cell at given coordinates, relative to the center cell.

        `coords` is a sequence of coordinates, which must be equal in length to
        the number of dimensions. If `coords` is blank, then the center cell is
        returned.
        """
        # TODO: Handle IndexError.
        if coords:
            return self.cell_array[tuple(center_coords + coords)]
        return self.cell_array[tuple(self.center_coords)]
