import heapq

from logic.word_state import WordState


class WordFinder:
    def __init__(self, table, dictionary):
        """Initialize the word finder."""
        self.table = table
        self.dictionary = set(dictionary)
        self.prefix_set = {word[:i] for word in dictionary for i in range(1, len(word) + 1)}

    def find_words(self, top_n=5):
        """Find the top N highest scoring words in the grid."""
        rows, cols = self.table.get_rows(), self.table.get_cols()
        found_words = []  # Min-heap to store the top N words as tuples (-cost, word)

        def dfs(row, col, state, visited):
            """Perform DFS to find words."""
            current_word = state.get_current_word()

            # Check if the current word is valid.
            if current_word in self.dictionary:
                total_cost = state.calculate_total_cost()
                # Push to the heap and ensure only the top N words remain
                heapq.heappush(found_words, (total_cost, current_word))
                if len(found_words) > top_n:
                    heapq.heappop(found_words)  # Remove the lowest scoring word

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
                    new_state = WordState(tile)
                    new_state.__dict__ = state.__dict__.copy()  # Copy current state
                    new_state.add_tile(tile)
                    dfs(r, c, new_state, visited | {(r, c)})

        # Start DFS from every tile.
        for row in range(rows):
            for col in range(cols):
                start_tile = self.table.get_tile(row, col)
                initial_state = WordState(start_tile)
                dfs(row, col, initial_state, {(row, col)})

        # Return the top N words sorted by total cost (highest first).
        return sorted(found_words, key=lambda x: (-x[0], x[1]))  # Sort by cost descending