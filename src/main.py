import extract_grid_nodes
import numpy as np
from tensorflow.keras.models import load_model

import utils


def main():
    # Path to the input image
    image_path = 'C:\\Users\\Kotori\\Desktop\\del\\Screenshot_2.png'

    # Step 1: Extract grid nodes
    nodes = extract_grid_nodes.extract_grid_nodes(image_path)

    # Step 2: Preprocess nodes
    processed_nodes = extract_grid_nodes.preprocess_nodes(nodes)  # Normalize, resize, etc.

    # Step 3: Load your trained model
    model = load_model("../models/spellcast_cheat_model.keras")  # Replace with your model path

    # Step 4: Predict outputs
    predictions = model.predict(processed_nodes)

    # Step 5: Process predictions
    decode_prediction(predictions)


def decode_prediction(predictions):
    """Decode model predictions into human-readable format."""

    # Each output in predictions is a 2D array of shape (25, N) where N is the number of classes
    letter_predictions = predictions[0]  # Shape: (25, 26)
    number_predictions = predictions[1]  # Shape: (25, 25)
    powerup_predictions = predictions[2]  # Shape: (25, 2)
    double_letter_predictions = predictions[3]  # Shape: (25, 2)
    triple_letter_predictions = predictions[4]  # Shape: (25, 2)
    double_point_predictions = predictions[5]  # Shape: (25, 2)

    # Iterate over each of the 25 nodes (rows in the arrays)
    for i in range(len(letter_predictions)):  # There are 25 nodes
        # Get the predicted class for each output by finding the index of the max value in each row
        letter_pred = np.argmax(letter_predictions[i])  # Index of max value (letter class)
        number_pred = np.argmax(number_predictions[i])  # Index of max value (number class)
        powerup_pred = np.argmax(powerup_predictions[i])  # Index of max value (powerup class)
        double_letter_pred = np.argmax(double_letter_predictions[i])  # Index of max value (double letter class)
        triple_letter_pred = np.argmax(triple_letter_predictions[i])  # Index of max value (triple letter class)
        double_point_pred = np.argmax(double_point_predictions[i])  # Index of max value (double point class)

        # Print the predictions for each component for the current node
        print(f"Node {i + 1}:")
        print(f"  Letter: {utils.NodePredictionUtils.index_to_class('letter', letter_pred)}")
        print(f"  Number: {utils.NodePredictionUtils.index_to_class('number', number_pred)}")
        print(f"  Power-up: {utils.NodePredictionUtils.index_to_class('powerup', powerup_pred)}")
        print(f"  Double Letter: {utils.NodePredictionUtils.index_to_class('double_letter', double_letter_pred)}")
        print(f"  Triple Letter: {utils.NodePredictionUtils.index_to_class('triple_letter', triple_letter_pred)}")
        print(f"  Double Point: {utils.NodePredictionUtils.index_to_class('double_point', double_point_pred)}")
        print()


if __name__ == "__main__":
    main()
