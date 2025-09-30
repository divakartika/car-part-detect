import tensorflow as tf
from tensorflow.keras import layers, models

def build_car_part_model(input_shape=(128, 128, 3), num_classes=5):
    """
    Builds a Convolutional Neural Network (CNN) for multi-label car part detection.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes to predict.
                           (5 parts, each can be open or closed).

    Returns:
        A TensorFlow Keras model.
    """
    model = models.Sequential([
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the feature map to feed into dense layers
        layers.Flatten(),

        # Dense (fully connected) layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        # Use sigmoid activation for multi-label classification
        # This allows each output neuron to be independent (e.g., hood can be open AND a door can be open)
        layers.Dense(num_classes, activation='sigmoid')
    ])

    return model

if __name__ == '__main__':
    # This part is for testing the model architecture
    # Create an instance of the model
    car_part_classifier = build_car_part_model()

    # Print a summary of the model architecture
    car_part_classifier.summary()

