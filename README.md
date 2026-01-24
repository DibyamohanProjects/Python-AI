# Emotion Detection Using MobileNetV2

## Overview

This project is a **facial emotion recognition system** that detects emotions in real-time using a webcam or static images. It leverages **MobileNetV2** transfer learning for high accuracy and **OpenCV** for face detection.

- Uses **MobileNetV2** pretrained on ImageNet for feature extraction
- Fine-tuned on a **Kaggle facial emotion dataset** with 7 classes:
  - Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
- Real-time webcam prediction with **live emotion probabilities**
- Compatible with **macOS (Apple M1) GPU acceleration via Metal)**

---

## Project Structure
```bash
emotion-detection/
├── data/ # Kaggle dataset: train/test folders
├── models/ # Saved models & Haar Cascade
│ ├── emotion_mobilenet.h5
│ └── haarcascade_frontalface_default.xml
├── src/ # Python scripts
│ ├── train_mobilenet.py # Final training script (MobileNetV2)
│ ├── webcam_emotion_live.py # Real-time webcam demo with probabilities
│ ├── confusion_matrix.py # Evaluation & confusion matrix
│ └── predict.py # Single image prediction (optional)
├── README.md # This file
└── requirements.txt # Python dependencies
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repo_url>
cd emotion-detection
```
### 2. Create Python 3.10 virtual environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Prepare dataset
Ensure the dataset has the structure:
```bash
data/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    └── ...
```
### Usage
### 1. Train MobileNetV2
```bash
python src/train_mobilenet.py
```
- Trains in two phases:
  1. Freeze base model, train classification head
  2. Fine-tune top layers
- Saves model to ```bash models/emotion_mobilenet.h5 ```

### 2. Evaluate Model
```bash
python src/confusion_matrix.py
```
- Outputs classification report
- Plots confusion matrix

### 3. Real-Time Webcam Emotion Detection
```bash
python src/webcam_emotion_live.py
```
- Detects faces
- Shows emotion label + probabilities
- Press ```q``` to quit

### Dependencies

- Python 3.10
- TensorFlow 2.16 (tensorflow-macos, tensorflow-metal)
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

See ```requirements.txt``` for exact versions.

### Notes
- MobileNetV2 expects 96×96 RGB input
- Webcam demo works best under good lighting
- GPU acceleration automatically enabled on Apple M1/M2 via Metal
- Fine-tuning improves accuracy to ~82–88%

### Author
Dibyamohan Panda 
- Backend developer and ML hobbyist
- Built a real-time facial emotion recognition system using MobileNetV2