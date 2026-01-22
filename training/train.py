import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from build_dataset import X_tensor, y_tensor
from cnn_model import CodeSmellCNN
from metrics import confusion_matrix
from split import train_val_split

# --------------------
# Config
# --------------------
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
EMBED_DIM = 64
NUM_CLASSES = 3

train_ds, val_ds = train_val_split(X_tensor, y_tensor)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# --------------------
# Model
# --------------------
vocab_size = int(X_tensor.max().item()) + 1
model = CodeSmellCNN(
    vocab_size=vocab_size,
    embed_dim=64,
    num_classes=NUM_CLASSES
)


class_counts = torch.bincount(y_tensor)
weights = 1.0 / class_counts.float()

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    model.train()
    return correct / total


# --------------------
# Training loop
# --------------------
for epoch in range(EPOCHS):
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc = evaluate(model, val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"Val Acc: {val_acc:.2%}"
    )

cm = confusion_matrix(model, val_loader, NUM_CLASSES)
print("\nConfusion Matrix:")
print(cm)
print("Training complete.")
