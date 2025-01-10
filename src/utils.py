class NodePredictionUtils:
    # Define possible classes
    letter_classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    number_classes = range(1, 26)  # Numbers 1-25
    powerup_classes = ["CRYSTAL", "NONE"]
    double_letter_classes = ["DL", "NONE"]
    triple_letter_classes = ["TL", "NONE"]
    double_point_classes = ["2X", "NONE"]

    # Mapping classes to indices
    letter_to_index = {letter: idx for idx, letter in enumerate(letter_classes)}
    number_to_index = {str(i): i-1 for i in range(1, 26)}  # Mapping for numbers (1-25)
    powerup_to_index = {shape: idx for idx, shape in enumerate(powerup_classes)}
    double_letter_to_index = {double_letter: idx for idx, double_letter in enumerate(double_letter_classes)}
    triple_letter_to_index = {triple_letter: idx for idx, triple_letter in enumerate(triple_letter_classes)}
    double_point_to_index = {double_point: idx for idx, double_point in enumerate(double_point_classes)}

    # Reverse mappings (index to class)
    index_to_letter = {idx: letter for letter, idx in letter_to_index.items()}
    index_to_number = {idx: str(i+1) for i, idx in enumerate(number_to_index.values())}
    index_to_powerup = {idx: shape for shape, idx in powerup_to_index.items()}
    index_to_double_letter = {idx: double_letter for double_letter, idx in double_letter_to_index.items()}
    index_to_triple_letter = {idx: triple_letter for triple_letter, idx in triple_letter_to_index.items()}
    index_to_double_point = {idx: double_point for double_point, idx in double_point_to_index.items()}

    @staticmethod
    def class_to_index(class_type, class_name):
        """Get the index of a class from its name."""
        class_map = {
            "letter": NodePredictionUtils.letter_to_index,
            "number": NodePredictionUtils.number_to_index,
            "powerup": NodePredictionUtils.powerup_to_index,
            "double_letter": NodePredictionUtils.double_letter_to_index,
            "triple_letter": NodePredictionUtils.triple_letter_to_index,
            "double_point": NodePredictionUtils.double_point_to_index,
        }

        return class_map[class_type].get(class_name, -1)  # Return -1 if class not found

    @staticmethod
    def index_to_class(class_type, index):
        """Get the class name from an index."""
        class_map = {
            "letter": NodePredictionUtils.index_to_letter,
            "number": NodePredictionUtils.index_to_number,
            "powerup": NodePredictionUtils.index_to_powerup,
            "double_letter": NodePredictionUtils.index_to_double_letter,
            "triple_letter": NodePredictionUtils.index_to_triple_letter,
            "double_point": NodePredictionUtils.index_to_double_point,
        }

        return class_map[class_type].get(index, "NONE")  # Return None if index not found