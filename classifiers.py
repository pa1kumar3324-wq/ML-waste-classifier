import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

# --- Configuration Constants ---
MODEL_PATH = "models/waste-classifier-mini-project.h5"
LABELS_PATHS = ["Labels.txt", "labels.txt"]
INPUT_SIZE = (224, 224)          # Size for model inference (e.g., MobileNet)
SMOOTH_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.50
DISPLAY_TEXT_POS = (20, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2

# Display size for the output window
DISPLAY_SIZE = (1280, 720)

# Colors (BGR format)
COLOR_SUCCESS = (0, 255, 0)  # Green
COLOR_DETECTING = (0, 0, 255)  # Red


def find_label_file():
    """Return the first existing label file path from LABELS_PATHS."""
    for path in LABELS_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No label file found. Checked: {', '.join(LABELS_PATHS)}"
    )


def load_model_and_labels():
    """Loads the Keras model and class labels."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Loading model from: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    labels_path = find_label_file()
    print(f"Loading labels from: {labels_path}...")
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    if not labels:
        raise ValueError("Label file is empty. Add one class per line.")

    print("Loaded classes:", labels)
    return model, labels


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resizes and normalizes the input frame for model inference."""
    img = cv2.resize(frame, INPUT_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def smooth_prediction(pred_buffer: deque, labels: list) -> tuple:
    """Performs majority voting on the prediction buffer."""
    if not pred_buffer:
        return "Detecting...", -1

    stable_class = max(set(pred_buffer), key=pred_buffer.count)
    stable_name = labels[stable_class]
    return stable_name, stable_class


def run_classifier():
    """Main function to run the real-time waste classification."""
    try:
        model, labels = load_model_and_labels()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    pred_buffer = deque(maxlen=SMOOTH_WINDOW)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (Index 0).")
        return

    cv2.namedWindow("Waste Classifier", cv2.WINDOW_NORMAL)
    print("\nCamera started... Press 'Q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame, verbose=0)[0]
            class_id = int(np.argmax(prediction))

            pred_buffer.append(class_id)
            stable_name, stable_class = smooth_prediction(pred_buffer, labels)

            stable_conf = 0.0
            if 0 <= stable_class < len(prediction):
                stable_conf = float(prediction[stable_class])

            if stable_conf >= CONFIDENCE_THRESHOLD:
                text = f"{stable_name} ({stable_conf*100:.1f}%)"
                color = COLOR_SUCCESS
            else:
                text = "Detecting..."
                color = COLOR_DETECTING

            cv2.putText(frame, text, DISPLAY_TEXT_POS, FONT, FONT_SCALE, color, FONT_THICKNESS)
            display_frame = cv2.resize(frame, DISPLAY_SIZE)
            cv2.imshow("Waste Classifier", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\nReleasing camera and closing windows...")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_classifier()
