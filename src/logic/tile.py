class Tile:
    def __init__(self, letter: str, number: int, powerup: bool,
                 double_letter: bool, triple_letter: bool, double_point: bool):
        # Private variables
        self.__letter = letter
        self.__number = number
        self.__powerup = powerup
        self.__double_letter = double_letter
        self.__triple_letter = triple_letter
        self.__double_point = double_point

    # Getter and setter for letter
    def get_letter(self):
        return self.__letter

    def set_letter(self, letter: str):
        if len(letter) == 1 and letter.isalpha():
            self.__letter = letter
        else:
            raise ValueError("Letter must be a single alphabetic character.")

    # Getter and setter for number
    def get_number(self):
        return self.__number

    def set_number(self, number: int):
        if isinstance(number, int):
            self.__number = number
        else:
            raise ValueError("Number must be an integer.")

    # Getter and setter for powerup
    def get_powerup(self):
        return self.__powerup

    def set_powerup(self, powerup: bool):
        if isinstance(powerup, bool):
            self.__powerup = powerup
        else:
            raise ValueError("Powerup must be a boolean.")

    # Getter and setter for double_letter
    def get_double_letter(self):
        return self.__double_letter

    def set_double_letter(self, double_letter: bool):
        if isinstance(double_letter, bool):
            self.__double_letter = double_letter
        else:
            raise ValueError("Double letter must be a boolean.")

    # Getter and setter for triple_letter
    def get_triple_letter(self):
        return self.__triple_letter

    def set_triple_letter(self, triple_letter: bool):
        if isinstance(triple_letter, bool):
            self.__triple_letter = triple_letter
        else:
            raise ValueError("Triple letter must be a boolean.")

    # Getter and setter for double_point
    def get_double_point(self):
        return self.__double_point

    def set_double_point(self, double_point: bool):
        if isinstance(double_point, bool):
            self.__double_point = double_point
        else:
            raise ValueError("Double point must be a boolean.")

    def __str__(self):
        """Return a human-readable string representation of the Tile."""
        return f"{self.__letter}-{self.__number}"