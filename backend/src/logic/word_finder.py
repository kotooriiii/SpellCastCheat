import heapq
import os
from typing import List

import pandas as pd
import copy

from logic.word_state import WordState


class WordFinder:
    dictionary = set()  # Shared across all instances
    prefix_set = set()  # Shared across all instances

    @classmethod
    def initialize_shared_data(cls):
        """Initialize class-level dictionary and prefix set."""
        cls.dictionary = set(cls.__read_dictionary())
        cls.prefix_set = {word[:i] for word in cls.dictionary for i in range(1, len(word) + 1)}

    @classmethod
    def __read_dictionary(cls):
        df = pd.read_csv("../../resources/dictionary.csv", usecols=[0])  # Read only the first column
        # Convert the column values to strings and then to uppercase
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.upper()
        # Convert the column to a list
        return df.iloc[:, 0].tolist()

    def __init__(self, table):
        """Initialize the word finder."""
        self.table = table

    def find_words(self, top_n=5) -> List[WordState]:
        """Find the top N highest scoring words in the grid."""
        rows, cols = self.table.get_rows(), self.table.get_cols()
        found_words = []  # Min-heap to store the top N words as tuples (-cost, word)

        def dfs(row, col, state, visited):
            """Perform DFS to find words."""
            current_word = state.get_current_word()

            # Check if the current word is valid.
            if current_word in self.__class__.dictionary:
                # Push to the heap and ensure only the top N words remain
                heapq.heappush(found_words, state)
                if len(found_words) > top_n:
                    heapq.heappop(found_words)  # Remove the lowest scoring word

            # Early pruning if the current word is not a prefix.
            if current_word not in self.__class__.prefix_set:
                return

            # Explore neighbors.
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited:
                    tile = self.table.get_tile(r, c)
                    new_state = copy.deepcopy(state)  # Copy current state
                    new_state.add_tile(tile)
                    dfs(r, c, new_state, visited | {(r, c)})

        # Start DFS from every tile.
        for row in range(rows):
            for col in range(cols):
                start_tile = self.table.get_tile(row, col)
                initial_state = WordState(start_tile)
                dfs(row, col, initial_state, {(row, col)})

        # Sort by total_cost descending, then current_word ascending
        return sorted(found_words, key=lambda x: (-x.get_total_cost(), x.get_current_word()))
