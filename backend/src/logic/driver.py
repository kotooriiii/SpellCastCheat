import numpy as np

from ai import extract_grid_nodes, utils
from logic.table import Table
from logic.tile import Tile
from logic.word_finder import WordFinder

import pandas as pd

from tensorflow.keras.models import load_model


def main():
    # Step 1: Path to the input image
    image_path = 'C:\\Users\\Kotori\\Desktop\\del\\Screenshot_1.png'

    # Step 2: Extract grid nodes
    nodes = extract_grid_nodes.extract_grid_nodes(image_path)

    # Step 3: Preprocess nodes
    processed_nodes = extract_grid_nodes.preprocess_nodes(nodes)  # Normalize, resize, etc.

    # Step 4: Load your trained model
    model = load_model("../../../models/spellcast_cheat_model.keras")  # Replace with your model path

    # Step 5: Predict outputs
    predictions = model.predict(processed_nodes)

    # Step 6: Decode predictions into tiles
    tiles = decode_predictions_to_tiles(predictions)

    # Step 7: Create a table with the predicted tiles
    table = Table(tiles)

    # Step 8: Visualize the table
    print("Table:")
    table.visualize()

    # Step 9: Word finding logic
    word_list = read_dictionary()
    finder = WordFinder(table, word_list)
    found_words = finder.find_words()

    # Output the found words
    print("Words Found:")
    print(found_words)


def decode_predictions_to_tiles(predictions):
    """Convert model predictions to a list of Tile objects."""
    tiles = []

    # Unpack predictions
    letter_predictions = predictions[0]  # Shape: (25, 26)
    number_predictions = predictions[1]  # Shape: (25, 25)
    powerup_predictions = predictions[2]  # Shape: (25, 2)
    double_letter_predictions = predictions[3]  # Shape: (25, 2)
    triple_letter_predictions = predictions[4]  # Shape: (25, 2)
    double_point_predictions = predictions[5]  # Shape: (25, 2)

    # Iterate over the 25 nodes
    for i in range(len(letter_predictions)):
        # Decode predictions
        letter = utils.NodePredictionUtils.index_to_tile_value('letter', np.argmax(letter_predictions[i]))
        number = utils.NodePredictionUtils.index_to_tile_value('number', np.argmax(number_predictions[i]))
        powerup = utils.NodePredictionUtils.index_to_tile_value('powerup', np.argmax(powerup_predictions[i]))
        double_letter = utils.NodePredictionUtils.index_to_tile_value('double_letter', np.argmax(double_letter_predictions[i]))
        triple_letter = utils.NodePredictionUtils.index_to_tile_value('triple_letter', np.argmax(triple_letter_predictions[i]))
        double_point = utils.NodePredictionUtils.index_to_tile_value('double_point', np.argmax(double_point_predictions[i]))

        # Create a Tile instance
        tile = Tile(letter, int(number), powerup, double_letter, triple_letter, double_point)
        tiles.append(tile)

    return tiles


def read_dictionary():
    df = pd.read_csv("../../../resources/dictionary.csv", usecols=[0])  # Read only the first column
    # Convert the column values to strings and then to uppercase
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.upper()
    # Convert the column to a list
    return df.iloc[:, 0].tolist()


if __name__ == "__main__":
    main()
