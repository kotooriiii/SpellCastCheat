import os

import cv2
import numpy as np
import tensorflow as tf
from keras.src.applications.resnet import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from PIL import Image
import utils

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 25
EPOCHS = 10

# Dataset path
dataset_path = "../dataset/"

# Load and preprocess data
image_data = []
letter_labels = []
number_labels = []
powerup_labels = []
double_letter_labels = []
triple_letter_labels = []
double_point_labels = []

# Loop through subdirectories
for subdir in os.listdir(dataset_path):
    subdir_path = os.path.join(dataset_path, subdir)

    if os.path.isdir(subdir_path):
        # Each subdirectory name follows the format: A_5_circle
        parts = subdir.split("_")
        letter = parts[0]  # First component (e.g., 'A')
        number = int(parts[1])  # Second component (e.g., '5')
        powerup = parts[2]  # Third component (e.g., 'crystal')
        double_letter = parts[3]  # Fourth component e.g, 'double letter')
        triple_letter = parts[4]  # Fifth component e.g, 'triple letter')
        double_point = parts[5]  # Sixth component e.g, '2X')

        # Load all images in this subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith(".png"):
                # Map components to indices
                letter_labels.append(utils.NodePredictionUtils.class_to_index("letter", letter))
                number_labels.append(utils.NodePredictionUtils.class_to_index("number",
                                                                              number - 1))  # 1 - based

                # Default to none if not found.
                powerup_labels.append(utils.NodePredictionUtils.class_to_index("powerup", powerup))
                double_letter_labels.append(utils.NodePredictionUtils.class_to_index("double_letter", double_letter))
                triple_letter_labels.append(utils.NodePredictionUtils.class_to_index("triple_letter", triple_letter))
                double_point_labels.append(utils.NodePredictionUtils.class_to_index("double_point", double_point))

                # Read the image using OpenCV
                image = cv2.imread(os.path.join(subdir_path, filename))

                # Resize the image using OpenCV
                image_resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_CUBIC)

                image_data.append(np.array(image_resized))

# Convert to NumPy arrays
image_data = np.array(image_data) / 255.0  # Normalize pixel values
letter_labels = to_categorical(letter_labels, num_classes=len(utils.NodePredictionUtils.letter_classes))
number_labels = to_categorical(number_labels, num_classes=len(utils.NodePredictionUtils.number_classes))
powerup_labels = to_categorical(powerup_labels, num_classes=len(utils.NodePredictionUtils.powerup_classes))
double_letter_labels = to_categorical(double_letter_labels,
                                      num_classes=len(utils.NodePredictionUtils.double_letter_classes))
triple_letter_labels = to_categorical(triple_letter_labels,
                                      num_classes=len(utils.NodePredictionUtils.triple_letter_classes))
double_point_labels = to_categorical(double_point_labels,
                                     num_classes=len(utils.NodePredictionUtils.double_point_classes))

# Split data
train_size = int(0.8 * len(image_data))
X_train, X_test = image_data[:train_size], image_data[train_size:]
y_train_letter, y_test_letter = letter_labels[:train_size], letter_labels[train_size:]
y_train_number, y_test_number = number_labels[:train_size], number_labels[train_size:]
y_train_powerup, y_test_powerup = powerup_labels[:train_size], powerup_labels[train_size:]
y_train_double_letter, y_test_double_letter = double_letter_labels[:train_size], double_letter_labels[train_size:]
y_train_triple_letter, y_test_triple_letter = triple_letter_labels[:train_size], triple_letter_labels[train_size:]
y_train_double_point, y_test_double_point = double_point_labels[:train_size], double_point_labels[train_size:]


def build_multitask_model(input_shape, num_letters, num_numbers, num_powerups, num_double_letters, num_triple_letters,
                          num_double_points):
    # # Load pretrained model as feature extractor
    # base_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    # base_model.trainable = False  # Freeze the base model
    # # Shared convolutional backbone
    # inputs = Input(shape=input_shape)
    # x = base_model(inputs, training=False)
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)


    # Shared input and convolutional layers
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)


    # Letter branch
    letter_output = Dense(256, activation="relu")(x)
    letter_output = Dense(num_letters, activation="softmax", name="letter_output")(letter_output)

    # Number branch
    number_output = Dense(256, activation="relu")(x)
    number_output = Dense(num_numbers, activation="softmax", name="number_output")(number_output)

    # Power up branch
    powerup_output = Dense(256, activation="relu")(x)
    powerup_output = Dense(num_powerups, activation="softmax", name="powerup_output")(powerup_output)

    # Double letter branch
    double_letter_output = Dense(256, activation="relu")(x)
    double_letter_output = Dense(num_double_letters, activation="softmax", name="double_letter_output")(
        double_letter_output)

    # Triple letter branch
    triple_letter_output = Dense(256, activation="relu")(x)
    triple_letter_output = Dense(num_triple_letters, activation="softmax", name="triple_letter_output")(
        triple_letter_output)

    # Double point branch
    double_point_output = Dense(256, activation="relu")(x)
    double_point_output = Dense(num_double_points, activation="softmax", name="double_point_output")(
        double_point_output)

    # Create the model
    model = Model(inputs=inputs,
                  outputs=[letter_output, number_output, powerup_output, double_letter_output, triple_letter_output,
                           double_point_output])
    return model


# Build and compile the model
model = build_multitask_model(
    input_shape=(128, 128, 3),
    num_letters=len(utils.NodePredictionUtils.letter_classes),
    num_numbers=len(utils.NodePredictionUtils.number_classes),
    num_powerups=len(utils.NodePredictionUtils.powerup_classes),
    num_double_letters=len(utils.NodePredictionUtils.double_letter_classes),
    num_triple_letters=len(utils.NodePredictionUtils.triple_letter_classes),
    num_double_points=len(utils.NodePredictionUtils.double_point_classes),
)

# Compile the model
model.compile(
    optimizer="adam",
    loss={
        "letter_output": "categorical_crossentropy",
        "number_output": "categorical_crossentropy",
        "powerup_output": "categorical_crossentropy",
        "double_letter_output": "categorical_crossentropy",
        "triple_letter_output": "categorical_crossentropy",
        "double_point_output": "categorical_crossentropy"
    },
    metrics=["accuracy"]*6
)


#     """
#     Fine-tune the pretrained base model within the multitask model.
#
#     Parameters:
#     - model: The multitask model.
#     - base_model_name: The name of the pretrained base model (e.g., 'resnet50').
#     - unfreeze_from_layer: Layer name in the base model to start unfreezing (optional).
#
#     Returns:
#     - model: The fine-tuned model.
#     """
#     # Locate the pretrained base model within the multitask model
#     base_model = model.get_layer(base_model_name)
#
#     # Unfreeze all layers if no specific layer is given
#     if unfreeze_from_layer is None:
#         for layer in base_model.layers:
#             layer.trainable = True
#     else:
#         # Unfreeze layers starting from a specific layer
#         unfreeze = False
#         for layer in base_model.layers:
#             if layer.name == unfreeze_from_layer:
#                 unfreeze = True
#             layer.trainable = unfreeze
#
#     # Recompile the model with a reduced learning rate
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Lower learning rate for fine-tuning
#         loss={
#             "letter_output": "categorical_crossentropy",
#             "number_output": "categorical_crossentropy",
#             "powerup_output": "categorical_crossentropy",
#             "double_letter_output": "categorical_crossentropy",
#             "triple_letter_output": "categorical_crossentropy",
#             "double_point_output": "categorical_crossentropy",
#         },
#         metrics=["accuracy"]*6
#     )
#     return model
#
#
# # Fine-tune the model
# model = fine_tune_model(model)  # Adjust layer name if needed

# Train the model
history = model.fit(
    X_train,
    {"letter_output": y_train_letter, "number_output": y_train_number, "powerup_output": y_train_powerup,
     "double_letter_output": y_train_double_letter, "triple_letter_output": y_train_triple_letter,
     "double_point_output": y_train_double_point},
    validation_data=(
        X_test,
        {"letter_output": y_test_letter, "number_output": y_test_number, "powerup_output": y_test_powerup,
         "double_letter_output": y_test_double_letter, "triple_letter_output": y_test_triple_letter,
         "double_point_output": y_test_double_point}
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)  # Early stopping to avoid overfitting
    ]
)

# Save the model
model.save("../models/spellcast_cheat_model.keras")

# Evaluate the model
loss, letter_loss, number_loss, powerup_loss, double_letter_loss, triple_letter_loss, double_point_loss, letter_acc, \
    number_acc, powerup_acc, double_letter_acc, triple_letter_acc, double_point_acc = \
    model.evaluate(
        X_test,
        {"letter_output": y_test_letter, "number_output": y_test_number, "powerup_output": y_test_powerup,
         "double_letter_output": y_test_double_letter, "triple_letter_output": y_test_triple_letter,
         "double_point_output": y_test_double_point}
    )

print(f"Letter Accuracy: {letter_acc:.2f}")
print(f"Number Accuracy: {number_acc:.2f}")
print(f"Powerup Accuracy: {powerup_acc:.2f}")
print(f"DL Accuracy: {double_letter_acc:.2f}")
print(f"TL Accuracy: {triple_letter_acc:.2f}")
print(f"2X Accuracy: {double_point_acc:.2f}")
