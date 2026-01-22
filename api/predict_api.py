from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from training.build_dataset import vocab, MAX_LEN
from training.cnn_model import CodeSmellCNN
from training.padding import pad_sequence

# ------------------------
# Model & Config
# ------------------------
NUM_CLASSES = 3
EMBED_DIM = 64
LABEL_MAP = {0: "LONG_METHOD", 1: "GOD_CLASS", 2: "HIGH_COMPLEXITY"}

vocab_size = len(vocab)
model = CodeSmellCNN(vocab_size, EMBED_DIM, NUM_CLASSES)
# Load your trained weights if saved
# model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# ------------------------
# FastAPI setup
# ------------------------
app = FastAPI(title="Java Code Smell Detector")


class PredictionRequest(BaseModel):
    code: str


class PredictionResponse(BaseModel):
    predicted_smells: List[str]
    confidences: List[float]


# ------------------------
# Endpoint
# ------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    tokens = list(request.code)
    encoded = vocab.encode(tokens)
    padded = pad_sequence(encoded, MAX_LEN)
    input_tensor = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top2_vals, top2_indices = torch.topk(probs, k=2)
        predicted_smells = [LABEL_MAP[i.item()] for i in top2_indices]
        confidences = [float(v.item()) for v in top2_vals]  # force Python float

    print("Predicted:", predicted_smells)
    print("Confidences:", confidences)

    return PredictionResponse(
        predicted_smells=predicted_smells,
        confidences=confidences
    )

