import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

# --- Configuration Constants ---
MODEL_PATH = "models/waste-classifier-mini-project.h5"
LABELS_PATH = "labels.txt"
INPUT_SIZE = (224, 224)          # Size for model inference (e.g., MobileNet)
SMOOTH_WINDOW = 7
CONFIDENCE_THRESHOLD = 0.50
DISPLAY_TEXT_POS = (20, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2

# --- NEW CONSTANT FOR DISPLAY RESOLUTION ---
# Common HD size. Choose a size that fits your screen well.
DISPLAY_SIZE = (1280, 720) 

# Colors (BGR format)
COLOR_SUCCESS = (0, 255, 0)  # Green
COLOR_DETECTING = (0, 0, 255)  # Red


# ==================================
# 1. CORE UTILITY FUNCTIONS
# ==================================

def load_model_and_labels():
    """Loads the Keras model and class labels."""
    print(f"Loading model from: {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise SystemExit(1)
        
    print(f"Loading labels from: {LABELS_PATH}...")
    try:
        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print("Loaded classes:", labels)
        return model, labels
    except FileNotFoundError:
        print(f"Error: Label file not found at {LABELS_PATH}")
        raise SystemExit(1)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resizes and normalizes the input frame for model inference."""
    img = cv2.resize(frame, INPUT_SIZE)
    img = img.astype("float32") / 255.0
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    return img

def smooth_prediction(pred_buffer: deque, labels: list) -> tuple:
    """
    Performs majority voting on the prediction buffer.

    Returns:
        tuple: (stable_name: str, stable_class: int)
    """
    if not pred_buffer:
        return "Detecting...", -1

    # Majority vote
    stable_class = max(set(pred_buffer), key=pred_buffer.count)
    stable_name = labels[stable_class]
    
    return stable_name, stable_class


# ==================================
# 2. MAIN EXECUTION FLOW
# ==================================

def run_classifier():
    """Main function to run the real-time waste classification."""
    
    # --- Initialization ---
    model, labels = load_model_and_labels()
    pred_buffer = deque(maxlen=SMOOTH_WINDOW)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera (Index 0).")
        return

    # Optional: Set camera's initial capture resolution (if supported)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
    
    # Optional: Create a resizable window
    cv2.namedWindow("Waste Classifier", cv2.WINDOW_NORMAL)
    
    print("\nCamera started... Press 'Q' to quit.")

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 1. Preprocess and Predict
        processed_frame = preprocess_frame(frame)
        prediction = model.predict(processed_frame, verbose=0)[0]
        
        # 2. Get instantaneous prediction
        class_id = np.argmax(prediction)
        
        # 3. Smoothing
        pred_buffer.append(class_id)
        stable_name, stable_class = smooth_prediction(pred_buffer, labels)
        
        # Get the confidence of the stable class *from the current frame's prediction*
        stable_conf = prediction[stable_class]
        
        # 4. Display Logic
        if stable_conf >= CONFIDENCE_THRESHOLD:
            text = f"{stable_name} ({stable_conf*100:.1f}%)"
            color = COLOR_SUCCESS
        else:
            text = "Detecting..."
            color = COLOR_DETECTING

        # Add text overlay
        cv2.putText(frame, text, DISPLAY_TEXT_POS, FONT, FONT_SCALE, color, FONT_THICKNESS)
        
        # 5. Display Size Fix: Resize the frame before displaying
        # This makes the displayed image larger than the default capture size.
        display_frame = cv2.resize(frame, DISPLAY_SIZE)
        
        # 6. Display and Exit Check
        cv2.imshow("Waste Classifier", display_frame) # Display the resized frame!

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("\nReleasing camera and closing windows...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_classifier()