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

import ai.utils as utils

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 25
EPOCHS = 10

# Dataset path
dataset_path = "../../dataset/train_and_test"


def train_test_tune_model():
    # Initialize lists for training and test data
    train_images, test_images = [], []
    train_letter_labels, test_letter_labels = [], []
    train_number_labels, test_number_labels = [], []
    train_powerup_labels, test_powerup_labels = [], []
    train_double_letter_labels, test_double_letter_labels = [], []
    train_triple_letter_labels, test_triple_letter_labels = [], []
    train_double_point_labels, test_double_point_labels = [], []

    # Iterate through subdirectories
    for subdir in os.listdir(dataset_path):
        subdir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(subdir_path):
            subdir_images, subdir_letter_labels, subdir_number_labels = [], [], []
            subdir_powerup_labels, subdir_double_letter_labels = [], []
            subdir_triple_letter_labels, subdir_double_point_labels = [], []

            for filename in os.listdir(subdir_path):
                if filename.endswith(".png"):
                    # Load and preprocess the image
                    image = cv2.imread(os.path.join(subdir_path, filename))
                    image_resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
                    subdir_images.append(np.array(image_resized))  # divide by 255 to normalize or no ?

                    # Extract labels
                    letter = subdir.split("_")[0]  # Example: 'A' from 'A_5_circle'
                    number = subdir.split("_")[1]
                    powerup = subdir.split("_")[2]
                    double_letter = subdir.split("_")[3]
                    triple_letter = subdir.split("_")[4]
                    double_point = subdir.split("_")[5]

                    subdir_letter_labels.append(utils.NodePredictionUtils.class_to_index("letter", letter))
                    subdir_number_labels.append(utils.NodePredictionUtils.class_to_index("number", number))
                    subdir_powerup_labels.append(utils.NodePredictionUtils.class_to_index("powerup", powerup))
                    subdir_double_letter_labels.append(
                        utils.NodePredictionUtils.class_to_index("double_letter", double_letter))
                    subdir_triple_letter_labels.append(
                        utils.NodePredictionUtils.class_to_index("triple_letter", triple_letter))
                    subdir_double_point_labels.append(
                        utils.NodePredictionUtils.class_to_index("double_point", double_point))

            # Convert lists to arrays
            subdir_images = np.array(subdir_images)
            subdir_letter_labels = to_categorical(subdir_letter_labels,
                                                  num_classes=len(utils.NodePredictionUtils.letter_classes))
            subdir_number_labels = to_categorical(subdir_number_labels,
                                                  num_classes=len(utils.NodePredictionUtils.number_classes))
            subdir_powerup_labels = to_categorical(subdir_powerup_labels,
                                                   num_classes=len(utils.NodePredictionUtils.powerup_classes))
            subdir_double_letter_labels = to_categorical(subdir_double_letter_labels,
                                                         num_classes=len(
                                                             utils.NodePredictionUtils.double_letter_classes))
            subdir_triple_letter_labels = to_categorical(subdir_triple_letter_labels,
                                                         num_classes=len(
                                                             utils.NodePredictionUtils.triple_letter_classes))
            subdir_double_point_labels = to_categorical(subdir_double_point_labels,
                                                        num_classes=len(utils.NodePredictionUtils.double_point_classes))

            # Split subdirectory data
            train_size = int(0.8 * len(subdir_images))
            X_train_subdir, X_test_subdir = subdir_images[:train_size], subdir_images[train_size:]
            y_train_letter_subdir, y_test_letter_subdir = subdir_letter_labels[:train_size], subdir_letter_labels[
                                                                                             train_size:]
            y_train_number_subdir, y_test_number_subdir = subdir_number_labels[:train_size], subdir_number_labels[
                                                                                             train_size:]
            y_train_powerup_subdir, y_test_powerup_subdir = subdir_powerup_labels[:train_size], subdir_powerup_labels[
                                                                                                train_size:]
            y_train_double_letter_subdir, y_test_double_letter_subdir = subdir_double_letter_labels[
                                                                        :train_size], subdir_double_letter_labels[
                                                                                      train_size:]
            y_train_triple_letter_subdir, y_test_triple_letter_subdir = subdir_triple_letter_labels[
                                                                        :train_size], subdir_triple_letter_labels[
                                                                                      train_size:]
            y_train_double_point_subdir, y_test_double_point_subdir = subdir_double_point_labels[
                                                                      :train_size], subdir_double_point_labels[
                                                                                    train_size:]

            # Append to the main training and test datasets
            train_images.extend(X_train_subdir)
            test_images.extend(X_test_subdir)
            train_letter_labels.extend(y_train_letter_subdir)
            test_letter_labels.extend(y_test_letter_subdir)
            train_number_labels.extend(y_train_number_subdir)
            test_number_labels.extend(y_test_number_subdir)
            train_powerup_labels.extend(y_train_powerup_subdir)
            test_powerup_labels.extend(y_test_powerup_subdir)
            train_double_letter_labels.extend(y_train_double_letter_subdir)
            test_double_letter_labels.extend(y_test_double_letter_subdir)
            train_triple_letter_labels.extend(y_train_triple_letter_subdir)
            test_triple_letter_labels.extend(y_test_triple_letter_subdir)
            train_double_point_labels.extend(y_train_double_point_subdir)
            test_double_point_labels.extend(y_test_double_point_subdir)

    # Convert final datasets to NumPy arrays
    X_train = np.array(train_images)
    X_test = np.array(test_images)
    y_train_letter = np.array(train_letter_labels)
    y_test_letter = np.array(test_letter_labels)
    y_train_number = np.array(train_number_labels)
    y_test_number = np.array(test_number_labels)
    y_train_powerup = np.array(train_powerup_labels)
    y_test_powerup = np.array(test_powerup_labels)
    y_train_double_letter = np.array(train_double_letter_labels)
    y_test_double_letter = np.array(test_double_letter_labels)
    y_train_triple_letter = np.array(train_triple_letter_labels)
    y_test_triple_letter = np.array(test_triple_letter_labels)
    y_train_double_point = np.array(train_double_point_labels)
    y_test_double_point = np.array(test_double_point_labels)

    def build_multitask_model(input_shape, num_letters, num_numbers, num_powerups, num_double_letters,
                              num_triple_letters,
                              num_double_points):
        # Load pretrained model as feature extractor
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        base_model.trainable = False  # Freeze the base model
        # Shared convolutional backbone
        inputs = Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        # # Shared input and convolutional layers
        # inputs = Input(shape=input_shape)
        # x = Conv2D(32, (3, 3), activation="relu")(inputs)
        # x = MaxPooling2D((2, 2))(x)
        # x = Conv2D(64, (3, 3), activation="relu")(x)
        # x = MaxPooling2D((2, 2))(x)
        # x = Flatten()(x)
        # x = Dropout(0.5)(x)

        # Letter branch
        letter_output = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        letter_output = Dense(num_letters, activation="softmax", name="letter_output")(letter_output)

        # Number branch
        number_output = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        number_output = Dense(num_numbers, activation="softmax", name="number_output")(number_output)

        # Power up branch
        powerup_output = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        powerup_output = Dense(num_powerups, activation="softmax", name="powerup_output")(powerup_output)

        # Double letter branch
        double_letter_output = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        double_letter_output = Dense(num_double_letters, activation="softmax", name="double_letter_output")(
            double_letter_output)

        # Triple letter branch
        triple_letter_output = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        triple_letter_output = Dense(num_triple_letters, activation="softmax", name="triple_letter_output")(
            triple_letter_output)

        # Double point branch
        double_point_output = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
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
        metrics=["accuracy"] * 6
    )

    def fine_tune_model(model, base_model_name="resnet50", unfreeze_from_layer=None):
        """
        Fine-tune the pretrained base model within the multitask model.

        Parameters:
        - model: The multitask model.
        - base_model_name: The name of the pretrained base model (e.g., 'resnet50').
        - unfreeze_from_layer: Layer name in the base model to start unfreezing (optional).

        Returns:
        - model: The fine-tuned model.
        """
        # Locate the pretrained base model within the multitask model
        base_model = model.get_layer(base_model_name)

        # Unfreeze all layers if no specific layer is given
        if unfreeze_from_layer is None:
            for layer in base_model.layers:
                layer.trainable = True
        else:
            # Unfreeze layers starting from a specific layer
            unfreeze = False
            for layer in base_model.layers:
                if layer.name == unfreeze_from_layer:
                    unfreeze = True
                layer.trainable = unfreeze

        # Recompile the model with a reduced learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
            loss={
                "letter_output": "categorical_crossentropy",
                "number_output": "categorical_crossentropy",
                "powerup_output": "categorical_crossentropy",
                "double_letter_output": "categorical_crossentropy",
                "triple_letter_output": "categorical_crossentropy",
                "double_point_output": "categorical_crossentropy",
            },
            metrics=["accuracy"] * 6
        )
        return model

    # Fine-tune the model
    model = fine_tune_model(model)  # Adjust layer name if needed

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
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
        ]
    )

    # Save the model
    model.save("../../models/spellcast_cheat_model.keras")

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
