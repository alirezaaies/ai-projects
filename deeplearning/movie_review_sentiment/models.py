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

# class RNNClassifier(nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         embed_dim,
#         hidden_dim,
#         num_classes,
#         num_layers=1,
#         bidirectional=False,
#         dropout=0.2
#     ):
#         super().__init__()

#         self.bidirectional = bidirectional
#         self.hidden_dim = hidden_dim
#         self.num_directions = 2 if bidirectional else 1

#         # Embedding layer
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

#         # RNN layer
#         self.rnn = nn.RNN(
#             input_size=embed_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional
#         )

#         # Fully connected output layer
#         self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """
#         x: (batch_size, seq_len)
#         """

#         # (B, T) -> (B, T, E)
#         embedded = self.embedding(x)

#         # output: (B, T, H*num_directions)
#         # hidden: (num_layers*num_directions, B, H)
#         output, hidden = self.rnn(embedded)

#         if self.bidirectional:
#             # last layer forward & backward hidden
#             h_forward = hidden[-2]   # (B, H)
#             h_backward = hidden[-1]  # (B, H)
#             h_final = torch.cat((h_forward, h_backward), dim=1)
#         else:
#             h_final = hidden[-1]     # (B, H)

#         h_final = self.dropout(h_final)

#         logits = self.fc(h_final)    # (B, num_classes)

#         return logits

# -----------------------------
# 1️⃣ RNN Classifier (your original)
# -----------------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, bidirectional=False, dropout=0.2):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        if self.bidirectional:
            h_final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            h_final = hidden[-1]
        h_final = self.dropout(h_final)
        logits = self.fc(h_final)
        return logits


# -----------------------------
# 2️⃣ LSTM Classifier
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=1, bidirectional=False, dropout=0.2):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        if self.bidirectional:
            h_final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            h_final = hidden[-1]
        h_final = self.dropout(h_final)
        logits = self.fc(h_final)
        return logits


# -----------------------------
# 3️⃣ 1D CNN Classifier for text
# -----------------------------
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 kernel_sizes=[3,4,5], num_filters=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (B, T)
        embedded = self.embedding(x)   # (B, T, E)
        embedded = embedded.permute(0,2,1)  # (B, E, T)
        conv_outputs = [torch.relu(conv(embedded)) for conv in self.convs]  # list of (B, F, L_out)
        pooled = [torch.max(o, dim=2)[0] for o in conv_outputs]  # global max pooling (B, F)
        cat = torch.cat(pooled, dim=1)  # (B, F * len(kernel_sizes))
        cat = self.dropout(cat)
        logits = self.fc(cat)
        return logits


# -----------------------------
# 4️⃣ Transformer Encoder Classifier
# -----------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_heads=4, hidden_dim=256, num_layers=2, dropout=0.2, max_len=500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        embedded = self.embedding(x) + self.pos_embedding(positions)
        transformed = self.transformer(embedded)
        pooled = transformed.mean(dim=1)  # simple mean pooling
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits