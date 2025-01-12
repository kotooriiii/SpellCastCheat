class Table:
    def __init__(self, tiles, rows=5, cols=5):
        if len(tiles) != rows * cols:
            raise ValueError("Number of tiles must match grid size.")
        self.__rows = rows
        self.__cols = cols
        self.__grid = self.__initialize_table(tiles)

    def __initialize_table(self, tiles):
        """Private method to populate the grid with tiles."""
        grid = []
        tile_index = 0
        for i in range(self.__rows):
            row = []
            for j in range(self.__cols):
                row.append(tiles[tile_index])
                tile_index += 1
            grid.append(row)
        return grid

    def get_tile(self, row, col):
        """Retrieve a tile at a specific position."""
        if 0 <= row < self.__rows and 0 <= col < self.__cols:
            return self.__grid[row][col]
        raise IndexError("Invalid row or column index.")

    def get_neighbors(self, row, col):
        """Get neighbors of a tile at (row, col)."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
        neighbors = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.__rows and 0 <= c < self.__cols:
                neighbors.append(self.__grid[r][c])
        return neighbors

    # Getter for name
    def get_rows(self):
        return self.__rows

    def get_cols(self):
        return self.__cols

    def visualize(self):
        """Visualize the grid."""
        for row in self.__grid:
            print(" | ".join(map(str, row)))
        print()
