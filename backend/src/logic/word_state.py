class WordState:
    def __init__(self, initial_tile):
        """Initialize the word state with the starting tile."""
        self.__current_word = initial_tile.get_letter()
        self.__path = [(initial_tile.get_row(), initial_tile.get_col())]
        self.__total_cost = initial_tile.get_number()
        self.__double_point_active = initial_tile.is_double_point()

    def add_tile(self, tile):
        """Add a tile to the state."""
        self.__current_word += tile.get_letter()
        self.__path.append((tile.get_row(), tile.get_col()))
        self.__total_cost += tile.get_number()
        self.__double_point_active = self.__double_point_active or tile.is_double_point()

    def calculate_total_cost(self):
        """Calculate the total cost with double point multiplier."""
        return self.__total_cost * 2 if self.__double_point_active else self.__total_cost

    def get_current_word(self):
        return self.__current_word

    def get_path(self):
        return self.__path