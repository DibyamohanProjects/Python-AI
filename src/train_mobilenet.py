import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# Configuration
# ===============================
IMG_SIZE = 96
BATCH_SIZE = 32
PHASE_1_EPOCHS = 15
PHASE_2_EPOCHS = 10

train_dir = "../data/train"
test_dir = "../data/test"

# ===============================
# Data Generators (RGB)
# ===============================
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ===============================
# Load MobileNetV2 Base Model
# ===============================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# ===============================
# Custom Classification Head
# ===============================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ===============================
# PHASE 1: Feature Extraction
# ===============================
base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("🔵 Phase 1: Training classification head")
model.fit(
    train_data,
    validation_data=test_data,
    epochs=PHASE_1_EPOCHS
)

# ===============================
# PHASE 2: Fine-Tuning
# ===============================
print("🔴 Phase 2: Fine-tuning MobileNetV2")

base_model.trainable = True

# Freeze bottom layers, unfreeze top layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=test_data,
    epochs=PHASE_2_EPOCHS
)

# ===============================
# Save Final Model
# ===============================
model.save("../models/emotion_mobilenet.h5")
print("✅ MobileNetV2 emotion model saved successfully")
