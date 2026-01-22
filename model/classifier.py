import torch.nn as nn

class CodeSmellClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        return self.fc(embedded)