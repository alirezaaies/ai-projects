import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re
import string
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# ────────────────────────────────────────────────
#   Model Definitions
# ────────────────────────────────────────────────

# 1. Basic RNN
class BasicRNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers=1, 
                 dropout=0.3,
                 bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, indices, lengths=None):
        embedded = self.dropout(self.embedding(indices))
        _, hidden = self.rnn(embedded)
        return self.fc(self.dropout(hidden[-1])).squeeze(1)
    
    
class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        num_layers=1,
        bidirectional=False,
        dropout=0.2
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # RNN layer
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """

        # (B, T) -> (B, T, E)
        embedded = self.embedding(x)

        # output: (B, T, H*num_directions)
        # hidden: (num_layers*num_directions, B, H)
        output, hidden = self.rnn(embedded)

        if self.bidirectional:
            # last layer forward & backward hidden
            h_forward = hidden[-2]   # (B, H)
            h_backward = hidden[-1]  # (B, H)
            h_final = torch.cat((h_forward, h_backward), dim=1)
        else:
            h_final = hidden[-1]     # (B, H)

        h_final = self.dropout(h_final)

        logits = self.fc(h_final)    # (B, num_classes)

        return logits