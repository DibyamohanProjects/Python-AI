import torch
from collections import defaultdict

def confusion_matrix(model, loader, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = torch.argmax(model(X_batch), dim=1)
            for t, p in zip(y_batch, preds):
                matrix[t, p] += 1

    return matrix
