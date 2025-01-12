class WordFinder:
    def __init__(self, table, dictionary):
        self.table = table
        self.dictionary = set(dictionary)  # Use a set for fast lookups.
        self.prefix_set = self.__build_prefix_set(dictionary)  # For early pruning.

    def __build_prefix_set(self, dictionary):
        """Build a set of all prefixes for words in the dictionary."""
        prefix_set = set()
        for word in dictionary:
            for i in range(1, len(word) + 1):
                prefix_set.add(word[:i])
        return prefix_set

    def find_words(self):
        """Find all valid words in the grid."""
        rows, cols = self.table.get_rows(), self.table.get_cols()
        found_words = set()

        def dfs(row, col, path, visited, current_word):
            """Perform DFS to find words."""
            # Check if the current word is valid.
            if current_word in self.dictionary:
                found_words.add(current_word)

            # Early pruning if the current word is not a prefix.
            if current_word not in self.prefix_set:
                return

            # Explore neighbors.
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited:
                    tile = self.table.get_tile(r, c)
                    dfs(r, c, path + [(r, c)], visited | {(r, c)}, current_word + tile.get_letter())

        # Start DFS from every tile.
        for row in range(rows):
            for col in range(cols):
                start_tile = self.table.get_tile(row, col)
                dfs(row, col, [(row, col)], {(row, col)}, start_tile.get_letter())

        return sorted(found_words, key=len, reverse=True) #found_words
