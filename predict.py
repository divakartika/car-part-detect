import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- Configuration ---
IMAGE_SIZE = (128, 128)
PART_LABELS = ['hood', 'front_left_door', 'front_right_door', 'rear_left_door', 'rear_right_door']
MODEL_PATH = 'car_part_multilabel_classifier_real_v2.h5'

def predict_image(img_path, model_path=MODEL_PATH, class_labels=PART_LABELS):
    """
    Loads an image, preprocesses it, and makes a multi-label prediction.

    Args:
        model: The trained Keras model.
        img_path (str): The path to the image file.
        class_labels (list): A list of part names.

    Returns:
        A dictionary with predicted states for each part.
    """

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Rescale to [0, 1]

    # Make prediction
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(img_array)[0] # Get the first (and only) prediction
    
    # Interpret results using a 50% threshold
    threshold = 0.5
    predicted_states = {label: ('open' if prob >= threshold else 'closed') for label, prob in zip(class_labels, predictions)}

    return predicted_states

