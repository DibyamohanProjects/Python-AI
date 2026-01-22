from training.extract_java import get_java_files
from training.labeler import detect_smells
from utils.ml_tokenizer import tokenize
from training.vocab import Vocabulary
from training.padding import pad_sequence
from training.tensors import to_tensor

# --------------------
# Configuration
# --------------------
JAVA_SRC_ROOT = "../data/commons-lang/src/main/java"
MAX_LEN = 1024

LABEL_MAP = {
    "LONG_METHOD": 0,
    "GOD_CLASS": 1,
    "HIGH_COMPLEXITY": 2
}

# --------------------
# Step 1: Build raw dataset
# --------------------
dataset = []

java_files = get_java_files(JAVA_SRC_ROOT)
print("Java files found:", len(java_files))

for file in java_files:
    code = file.read_text(errors="ignore")

    tokens = list(tokenize(code))     # IMPORTANT: list, not generator
    print("Number of tokens:", len(tokens))
    labels = detect_smells(code)

    dataset.append({
        "tokens": tokens,
        "labels": labels
    })

print("Dataset size:", len(dataset))

# --------------------
# Step 2: Build vocabulary
# --------------------
vocab = Vocabulary(min_freq=2)
vocab.build(dataset)

print("\nVocabulary size:", len(vocab))

# --------------------
# Step 3: Encode + Pad
# --------------------
X = []
y = []

for sample in dataset:
    if not sample["labels"]:
        continue  # skip unlabeled samples

    encoded = vocab.encode(sample["tokens"])
    padded = pad_sequence(encoded, MAX_LEN)

    # Single-label classification for now
    label = LABEL_MAP[sample["labels"][0]]

    X.append(padded)
    y.append(label)

# --------------------
# Step 4: Convert to tensors
# --------------------
X_tensor, y_tensor = to_tensor(X, y)

print("\nFinal tensor shapes:")
print("X:", X_tensor.shape)
print("y:", y_tensor.shape)
