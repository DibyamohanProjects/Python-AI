import torch
import torch.nn as nn
import torch.nn.functional as F

class CodeSmellCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2),
            nn.Conv1d(embed_dim, 128, kernel_size=7, padding=3),
        ])

        self.fc = nn.Linear(128 * 3, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, L)
        x = self.embedding(x)          # (B, L, D)
        x = x.permute(0, 2, 1)         # (B, D, L)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(x))        # (B, C, L)
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)
            conv_outs.append(p)

        features = torch.cat(conv_outs, dim=1)
        features = self.dropout(features)

        return self.fc(features)
