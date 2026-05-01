# ML Waste Classifier

## Overview

`ML-waste-classifier` is a real-time waste classification project that uses a pre-trained TensorFlow image classification model to identify common waste categories from a webcam feed. The project demonstrates how OpenCV can be used for live video capture, frame preprocessing, and result display with overlay text.

## What it does

- Loads a Keras model from `models/waste-classifier-mini-project.h5`
- Loads class labels from `Labels.txt`
- Captures live video from the default webcam
- Preprocesses each frame to the model input size
- Runs inference and smooths predictions using a moving vote buffer
- Displays the predicted waste type and confidence on the video feed
- Supports clean shutdown via `Q` or keyboard interrupt

## Files

- `classifiers.py`: main application script that runs the live classifier
- `Labels.txt`: class labels, one label per line
- `models/waste-classifier-mini-project.h5`: trained waste classification model
- `requirements.txt`: Python dependencies for the project

## Installation

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the classifier with:

```bash
python3 classifiers.py
```

Then point your webcam at the object to classify. Press `Q` to quit.

## Notes

- The project expects the model file at `models/waste-classifier-mini-project.h5`.
- The label file name is automatically detected from `Labels.txt` or `labels.txt`.
- The script resizes the camera feed to `1280x720` for display.

## Troubleshooting

- `ModuleNotFoundError: No module named 'cv2'` — install dependencies from `requirements.txt`
- `Model file not found` — ensure `models/waste-classifier-mini-project.h5` is present
- `No label file found` — ensure `Labels.txt` exists in the repository root
- `Error: Could not open camera (Index 0)` — verify your webcam is available or change the camera index

## License

This repository is provided as-is for educational use and demonstration.
