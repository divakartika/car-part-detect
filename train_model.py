import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from car_part_model import build_car_part_model
import pandas as pd
import os

# --- Configuration ---
DATASET_DIR = 'car_dataset_real'
LABELS_FILE = f'{DATASET_DIR}/labels.csv'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 5 # 5 parts: hood, front_left_door, etc.
NUM_SAMPLES = 1000

def train():
    """
    Loads the synthetic dataset and trains the multi-label car part detection model.
    """
    # --- Pre-training Checks ---
    if not os.path.exists(LABELS_FILE):
        print(f"Error: Labels file not found at '{LABELS_FILE}'.")
        print("Please run a data generation script (e.g., generate_synthetic_data.py or prepare_video_data.py) first.")
        return

    # Load the labels from the CSV file
    df = pd.read_csv(LABELS_FILE)
    if df.empty:
        print(f"Error: The labels file '{LABELS_FILE}' is empty. There is no data to train on.")
        return
    df['filename'] = df['filename'].str.replace('\\','/')
    print(df.head())

    # --- Optional: Randomly sample the data ---
    num_samples = NUM_SAMPLES
    if num_samples > len(df):
        print(f"Warning: Requested {num_samples} samples, but only {len(df)} are available. Using all {len(df)} samples.")
    else:
        print(f"Randomly sampling {num_samples} images from the total {len(df)} available images.")
        df = df.sample(n=num_samples, random_state=42) # Use a random_state for reproducibility

    # Get the column names for the labels
    columns = list(df.columns)[1:]

    # --- Data Preprocessing and Augmentation ---
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=DATASET_DIR,
        x_col='filename',
        y_col=columns,
        subset="training",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode="raw", # Use "raw" for multi-label regression/classification
        target_size=IMAGE_SIZE
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=DATASET_DIR,
        x_col='filename',
        y_col=columns,
        subset="validation",
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=IMAGE_SIZE
    )

    # --- Verify that generators found images ---
    if train_generator.samples == 0:
        print("Error: No training images were found. The training generator is empty.")
        print("Please check that the image paths in 'labels.csv' are correct and exist within the 'images' subfolder.")
        return

    # --- Model Building and Compilation ---
    model = build_car_part_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=NUM_CLASSES)

    # For multi-label classification, use binary_crossentropy loss
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'] # Accuracy here measures the fraction of labels that are correct
    )

    print(f"Found {train_generator.samples} training images and {validation_generator.samples} validation images.")
    print("Starting multi-label model training...")

    # --- Model Training ---
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS
    )
    # --- Save the Trained Model ---
    model.save('car_part_multilabel_classifier_real.h5')
    print("\nModel training complete and saved as car_part_multilabel_classifier_real.h5")

    return history

if __name__ == '__main__':
    train()