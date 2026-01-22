import torch
from torch.utils.data import random_split

def train_val_split(X, y, val_ratio=0.2):
    dataset = torch.utils.data.TensorDataset(X, y)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    return random_split(dataset, [train_size, val_size])
