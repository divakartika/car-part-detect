import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- Configuration ---
IMAGE_SIZE = (128, 128)
PART_LABELS = ['hood', 'front_left_door', 'front_right_door', 'rear_left_door', 'rear_right_door']
model_path = 'car_part_multilabel_classifier_real.h5'

def predict_image(img_path, model_path=model_path, class_labels=PART_LABELS):
    """
    Loads an image, preprocesses it, and makes a multi-label prediction.

    Args:
        model: The trained Keras model.
        img_path (str): The path to the image file.
        class_labels (list): A list of part names.

    Returns:
        A dictionary with predicted states for each part.
    """
    # if not os.path.exists(img_path):
    #     print(f"Error: Image file not found at '{img_path}'")
    #     return None

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

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Predict car part states from a single image.")
    # parser.add_argument("image_file", type=str, help="Path to the image file for prediction.")
    # parser.add_argument("--model_path", type=str, default="car_part_multilabel_classifier_real.h5", 
    #                     help="Path to the trained .h5 model file.")
    # args = parser.parse_args()

    # model_path = 'car_part_multilabel_classifier_real.h5'
    # image_file = car_app.image

    # Load the trained model
    # try:
    #     model = tf.keras.models.load_model(args.model_path)
    # except (IOError, ImportError) as e:
    #     print(f"Error loading model from '{args.model_path}': {e}")
    #     print("Please make sure the model file exists and you have trained it first.")
    #     exit()

    # # --- Predict on the user-provided image ---
    # predicted_states = predict_image(model, args.image_file, PART_LABELS)

    # if predicted_states:
    #     print(f"\n--- Prediction Results for: {os.path.basename(args.image_file)} ---")
    #     for part, state in predicted_states.items():
    #         print(f"  - {part}: {state.upper()}")

