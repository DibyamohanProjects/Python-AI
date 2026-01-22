import torch.nn as nn

class CodeSmellClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        embedded = self.embedding(x)              # (B, L, D)
        pooled = embedded.mean(dim=1)             # (B, D)
        logits = self.classifier(pooled)           # (B, C)
        return logits
