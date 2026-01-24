import cv2
import numpy as np
import tensorflow as tf

# ===============================
# Configuration
# ===============================
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
MODEL_PATH = "../models/emotion_mobilenet.h5"
FACE_CASCADE_PATH = "../models/haarcascade_frontalface_default.xml"
IMG_SIZE = 96

# ===============================
# Load model & face detector
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
        roi_rgb = roi_rgb / 255.0
        roi_rgb = roi_rgb.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        # Predict probabilities
        pred_probs = model.predict(roi_rgb, verbose=0)[0]
        top_idx = np.argmax(pred_probs)
        emotion = EMOTIONS[top_idx]

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display main emotion
        cv2.putText(frame, f"{emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display probabilities below face
        for i, (label, prob) in enumerate(zip(EMOTIONS, pred_probs)):
            text = f"{label}: {prob*100:.1f}%"
            cv2.putText(frame, text, (x, y+h + 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Emotion Detection - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
