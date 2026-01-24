import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ===============================
# Configuration
# ===============================
IMG_SIZE = 96
BATCH_SIZE = 32
TEST_DIR = "../data/test"

# Emotion labels (must match your training labels)
EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# ===============================
# Load Trained Model
# ===============================
model = tf.keras.models.load_model("../models/emotion_mobilenet.h5")
print("✅ MobileNetV2 model loaded")

# ===============================
# Test Data Generator
# ===============================
test_gen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ===============================
# Make Predictions
# ===============================
pred_probs = model.predict(test_data, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_data.classes

# ===============================
# Classification Report
# ===============================
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=EMOTIONS))

# ===============================
# Confusion Matrix
# ===============================
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - MobileNetV2 Emotion Model")
plt.show()
