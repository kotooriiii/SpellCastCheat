import numpy as np

from ai import extract_grid_nodes, utils
from business_model.table import Table
from logic.tile import Tile

from dto.sequence_model import SequenceModel
from dto.spellcast_result_model import SpellCastResultModel
from dto.tile_model import TileModel
from logic.word_finder import WordFinder
import cv2

from tensorflow.keras.models import load_model


def process_image(image_data, is_debugging=False) -> SpellCastResultModel:
    # Step 1: Extract grid nodes
    nodes = extract_grid_nodes.extract_grid_nodes(image_data)

    # Step 2: Preprocess nodes
    processed_nodes = extract_grid_nodes.preprocess_nodes(nodes)  # Normalize, resize, etc.

    # Step 3: Load your trained model
    model = load_model("../../models/spellcast_cheat_model.keras")  # Replace with your model path

    # Step 4: Predict outputs
    predictions = model.predict(processed_nodes)

    # Step 5: Decode predictions into tiles
    tiles = __decode_predictions_to_tiles(predictions)

    # Step 6: Create a table with the predicted tiles
    table = Table(tiles)

    # Step 7: Visualize the table
    if is_debugging:
        # Print the number of extracted nodes
        print(f'Extracted {len(nodes)} nodes.')
        print("Table:")
        table.visualize()

    # Step 8: Word finding logic
    finder = WordFinder(table)
    found_words = finder.find_words()

    # Output the found words
    if is_debugging:
        print("Words Found:")
        print(found_words)

    # Step 9: Transform from business logic objects to DTO for client

    # Transform found words into sequences
    sequences = [
        SequenceModel(cost=word.get_total_cost(), word=word.get_current_word(), path=word.get_path())
        for word in found_words
    ]

    # Convert the table to a 2D grid of tiles
    grid = [
        [TileModel(letter=tile.get_letter(), number=tile.get_number(), powerup=tile.is_powerup(),
                   double_letter=tile.is_double_letter(), triple_letter=tile.is_triple_letter(),
                   double_point=tile.is_double_point()) for tile in row]
        for row in table.get_grid()
    ]

    return SpellCastResultModel(sequences=sequences, grid=grid)


def __decode_predictions_to_tiles(predictions):
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
        double_letter = utils.NodePredictionUtils.index_to_tile_value('double_letter',
                                                                      np.argmax(double_letter_predictions[i]))
        triple_letter = utils.NodePredictionUtils.index_to_tile_value('triple_letter',
                                                                      np.argmax(triple_letter_predictions[i]))
        double_point = utils.NodePredictionUtils.index_to_tile_value('double_point',
                                                                     np.argmax(double_point_predictions[i]))

        # Create a Tile instance
        tile = Tile(letter, int(number), powerup, double_letter, triple_letter, double_point)
        tiles.append(tile)

    return tiles


if __name__ == "__main__":
    WordFinder.initialize_shared_data()
    process_image(cv2.imread('C:\\Users\\Kotori\\Desktop\\del\\Screenshot_1.png'), is_debugging=True)
